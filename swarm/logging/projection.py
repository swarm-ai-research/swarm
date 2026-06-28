"""Sequence-numbered replay + checkpoint abstractions over the event log.

Ported (not drop-in) from fabro's ``lib/crates/fabro-store``. The Rust crate
folds a stream of ``EventEnvelope { seq, event }`` records into a
``RunProjection`` via a ``RunProjectionReducer`` (``apply_event`` /
``apply_events``), and serializes the projection as a checkpoint
(``SerializableProjection``). This module brings the same three ideas to
SWARM's existing append-only JSONL :class:`~swarm.logging.event_log.EventLog`:

  - **EventEnvelope** — wraps each replayed :class:`Event` with a monotonic
    ``seq`` (its 0-indexed position in the log). The on-disk JSONL is never
    modified; ``seq`` is derived at replay time, so existing logs stay
    readable.
  - **Reducer** — a fold (``initial`` + ``apply``) that builds an arbitrary
    state ``S`` from the envelope stream. Mirrors ``RunProjectionReducer``.
  - **Checkpoint** — a ``{last_seq, state}`` snapshot. Reconstruct the state
    as of *any* ``seq`` (``until_seq``), persist it, and later *resume* the
    fold from where it left off without re-reading the whole log — the key
    win for long experimental sweeps.

The design goal (acceptance criteria for issue ``y6to.1``): replay can
reconstruct intermediate state at any checkpoint, and existing JSONL log
files remain readable.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction

S = TypeVar("S")


@dataclass(frozen=True)
class EventEnvelope:
    """An :class:`Event` tagged with its monotonic replay sequence number.

    ``seq`` is the event's 0-indexed position in the log. It is assigned at
    replay time rather than stored on disk, so the envelope is a read-side
    enrichment that never changes the wire format (mirrors fabro's
    ``EventEnvelope { seq, event }``).
    """

    seq: int
    event: Event

    def to_dict(self) -> Dict[str, Any]:
        return {"seq": self.seq, "event": self.event.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventEnvelope":
        return cls(seq=int(data["seq"]), event=Event.from_dict(data["event"]))


@dataclass
class Checkpoint(Generic[S]):
    """A serializable snapshot of a reducer's state as of ``last_seq``.

    ``last_seq`` is the highest envelope sequence folded into ``state``
    (``-1`` for the empty/initial state). ``state`` is the reducer-specific
    serialized form (see :meth:`Reducer.serialize`).
    """

    last_seq: int
    state: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"last_seq": self.last_seq, "state": self.state}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint[Any]":
        return cls(last_seq=int(data["last_seq"]), state=data["state"])

    def save(self, path: Path | str) -> Path:
        """Persist the checkpoint as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Checkpoint[Any]":
        """Restore a checkpoint previously written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


class Reducer(ABC, Generic[S]):
    """Folds an :class:`EventEnvelope` stream into a projection state ``S``.

    Mirrors fabro's ``RunProjectionReducer``: ``apply`` is the per-event
    transition (``apply_event``); :meth:`replay` is the batch fold
    (``apply_events``) with checkpoint support. Subclasses that hold
    non-JSON-native state (e.g. ``datetime``) should override
    :meth:`serialize` / :meth:`deserialize` so checkpoints round-trip.
    """

    @abstractmethod
    def initial(self) -> S:
        """Return the empty state (the projection before any event)."""

    @abstractmethod
    def apply(self, state: S, envelope: EventEnvelope) -> S:
        """Fold one envelope into ``state`` and return the next state.

        May mutate and return ``state`` in place, or return a new object.
        """

    # Serialization hooks — default assumes ``S`` is already JSON-native.

    def serialize(self, state: S) -> Any:
        return state

    def deserialize(self, data: Any) -> S:
        return data  # type: ignore[no-any-return]

    def replay(
        self,
        envelopes: Iterable[EventEnvelope],
        *,
        until_seq: Optional[int] = None,
        resume: Optional[Checkpoint[Any]] = None,
    ) -> Tuple[S, Checkpoint[S]]:
        """Fold ``envelopes`` into state, returning ``(state, checkpoint)``.

        Args:
            envelopes: Ordered envelope stream (ascending ``seq``).
            until_seq: Stop after folding the envelope with this ``seq``
                (inclusive). ``None`` folds the whole stream. This is how an
                intermediate state "as of checkpoint N" is reconstructed.
            resume: A prior checkpoint to continue from. Envelopes with
                ``seq <= resume.last_seq`` are skipped, so the same stream can
                be re-fed without double-applying.

        Returns:
            The folded state and a fresh :class:`Checkpoint` capturing it.
        """
        if resume is None:
            state = self.initial()
            last_seq = -1
        else:
            state = self.deserialize(resume.state)
            last_seq = resume.last_seq

        for env in envelopes:
            if env.seq <= last_seq:
                continue  # already folded (idempotent resume)
            if until_seq is not None and env.seq > until_seq:
                break
            state = self.apply(state, env)
            last_seq = env.seq

        return state, Checkpoint(last_seq=last_seq, state=self.serialize(state))


# Concrete reducer: reconstruct interaction state from the event stream.


class InteractionReducer(Reducer[Dict[str, Dict[str, Any]]]):
    """Reconstructs per-interaction state from the event stream.

    Equivalent to :meth:`EventLog.to_interactions`, but incremental and
    checkpoint-aware: fold up to any ``seq`` to see the interactions exactly
    as they stood at that point in the log. State is a JSON-native
    ``{interaction_id: record}`` map (timestamps held as ISO strings) so
    checkpoints serialize without custom hooks.
    """

    def initial(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def apply(
        self, state: Dict[str, Dict[str, Any]], envelope: EventEnvelope
    ) -> Dict[str, Dict[str, Any]]:
        event = envelope.event
        iid = event.interaction_id
        if not iid:
            return state

        if event.event_type == EventType.INTERACTION_PROPOSED:
            ts = event.timestamp
            state[iid] = {
                "interaction_id": iid,
                "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts,
                "initiator": event.initiator_id or "",
                "counterparty": event.counterparty_id or "",
                "interaction_type": event.payload.get("interaction_type", "reply"),
                "v_hat": event.payload.get("v_hat", 0.0),
                "p": event.payload.get("p", 0.5),
                "causal_parents": event.payload.get("causal_parents", []),
                "accepted": False,
            }
        elif event.event_type == EventType.INTERACTION_ACCEPTED:
            if iid in state:
                state[iid]["accepted"] = True
        elif event.event_type == EventType.INTERACTION_REJECTED:
            if iid in state:
                state[iid]["accepted"] = False
        elif event.event_type == EventType.PAYOFF_COMPUTED:
            if iid in state:
                components = event.payload.get("components", {})
                state[iid].update(
                    {
                        "tau": components.get("tau", 0.0),
                        "c_a": components.get("c_a", 0.0),
                        "c_b": components.get("c_b", 0.0),
                        "r_a": components.get("r_a", 0.0),
                        "r_b": components.get("r_b", 0.0),
                    }
                )
        return state

    @staticmethod
    def materialize(state: Dict[str, Dict[str, Any]]) -> List[SoftInteraction]:
        """Convert folded records into sorted :class:`SoftInteraction` objects."""
        result: List[SoftInteraction] = []
        for data in state.values():
            ts = data.get("timestamp")
            result.append(
                SoftInteraction(
                    interaction_id=data["interaction_id"],
                    timestamp=datetime.fromisoformat(ts) if ts else datetime.now(),
                    initiator=data["initiator"],
                    counterparty=data["counterparty"],
                    interaction_type=InteractionType(data["interaction_type"]),
                    accepted=data["accepted"],
                    v_hat=data.get("v_hat", 0.0),
                    p=data.get("p", 0.5),
                    tau=data.get("tau", 0.0),
                    c_a=data.get("c_a", 0.0),
                    c_b=data.get("c_b", 0.0),
                    r_a=data.get("r_a", 0.0),
                    r_b=data.get("r_b", 0.0),
                    causal_parents=data.get("causal_parents", []),
                )
            )
        result.sort(key=lambda x: x.timestamp)
        return result


def envelopes_from_events(
    events: Iterable[Event], *, start_seq: int = 0
) -> Iterator[EventEnvelope]:
    """Tag an ordered :class:`Event` stream with monotonic sequence numbers."""
    for offset, event in enumerate(events):
        yield EventEnvelope(seq=start_seq + offset, event=event)
