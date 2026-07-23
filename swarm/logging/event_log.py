"""Append-only JSONL event logger."""

import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction

if TYPE_CHECKING:
    from swarm.logging.projection import Checkpoint, EventEnvelope, Reducer


class EventLog:
    """
    Append-only JSONL event logger.

    Events are stored as newline-delimited JSON for efficient
    append operations and streaming reads.
    """

    def __init__(self, path: Path):
        """
        Initialize event log.

        Args:
            path: Path to the JSONL log file
        """
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ensure_parent_exists()

    def _ensure_parent_exists(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> None:
        """
        Append an event to the log.

        Args:
            event: Event to append
        """
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    def append_many(self, events: List[Event]) -> None:
        """
        Append multiple events to the log.

        Args:
            events: List of events to append
        """
        with self._lock:
            with open(self.path, "a") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict()) + "\n")

    def replay(self) -> Iterator[Event]:
        """
        Iterate over all logged events.

        Yields:
            Event objects in chronological order
        """
        if not self.path.exists():
            return

        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield Event.from_dict(json.loads(line))

    def iter_envelopes(
        self, *, start_seq: int = 0, until_seq: Optional[int] = None
    ) -> "Iterator[EventEnvelope]":
        """Replay events tagged with monotonic sequence numbers.

        Each event is wrapped in an :class:`EventEnvelope` whose ``seq`` is its
        0-indexed position in the log. Sequence numbers are derived at replay
        time, so the on-disk JSONL is unchanged and older logs stay readable.

        Args:
            start_seq: Skip envelopes with ``seq < start_seq`` (e.g. to resume
                past a checkpoint). The yielded ``seq`` is still the absolute
                log position.
            until_seq: Stop after yielding the envelope with this ``seq``
                (inclusive). ``None`` yields to the end of the log.
        """
        from swarm.logging.projection import EventEnvelope

        for seq, event in enumerate(self.replay()):
            if seq < start_seq:
                continue
            if until_seq is not None and seq > until_seq:
                break
            yield EventEnvelope(seq=seq, event=event)

    def project(
        self,
        reducer: "Reducer[Any]",
        *,
        until_seq: Optional[int] = None,
        resume: "Optional[Checkpoint[Any]]" = None,
    ) -> "Tuple[Any, Checkpoint[Any]]":
        """Fold the log into a projection state via ``reducer``.

        Reconstructs intermediate state at any checkpoint: pass ``until_seq``
        to stop at a given position, or ``resume`` to continue a prior
        checkpoint without re-folding earlier events. Returns
        ``(state, checkpoint)``.
        """
        start_seq = 0 if resume is None else resume.last_seq + 1
        envelopes = self.iter_envelopes(start_seq=start_seq, until_seq=until_seq)
        return reducer.replay(envelopes, until_seq=until_seq, resume=resume)

    def replay_filtered(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interaction_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Iterator[Event]:
        """
        Iterate over events with filtering.

        Args:
            event_types: Filter by event types
            start_time: Filter events after this time
            end_time: Filter events before this time
            interaction_id: Filter by interaction ID
            agent_id: Filter by agent ID

        Yields:
            Filtered Event objects
        """
        for event in self.replay():
            # Type filter
            if event_types and event.event_type not in event_types:
                continue

            # Time filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            # ID filters
            if interaction_id and event.interaction_id != interaction_id:
                continue
            if agent_id:
                if (
                    event.agent_id != agent_id
                    and event.initiator_id != agent_id
                    and event.counterparty_id != agent_id
                ):
                    continue

            yield event

    def to_interactions(self) -> List[SoftInteraction]:
        """
        Reconstruct interactions from event stream.

        Returns:
            List of SoftInteraction objects
        """
        # Track interaction state
        interactions: Dict[str, Dict[str, Any]] = {}

        for event in self.replay():
            if event.event_type == EventType.INTERACTION_PROPOSED:
                iid = event.interaction_id
                if iid:
                    interactions[iid] = {
                        "interaction_id": iid,
                        "timestamp": event.timestamp,
                        "initiator": event.initiator_id or "",
                        "counterparty": event.counterparty_id or "",
                        "interaction_type": event.payload.get(
                            "interaction_type", "reply"
                        ),
                        "v_hat": event.payload.get("v_hat", 0.0),
                        "p": event.payload.get("p", 0.5),
                        "causal_parents": event.payload.get(
                            "causal_parents", []
                        ),
                        "accepted": False,
                    }

            elif event.event_type == EventType.INTERACTION_ACCEPTED:
                iid = event.interaction_id
                if iid and iid in interactions:
                    interactions[iid]["accepted"] = True

            elif event.event_type == EventType.INTERACTION_REJECTED:
                iid = event.interaction_id
                if iid and iid in interactions:
                    interactions[iid]["accepted"] = False

            elif event.event_type == EventType.PAYOFF_COMPUTED:
                iid = event.interaction_id
                if iid and iid in interactions:
                    components = event.payload.get("components", {})
                    interactions[iid].update(
                        {
                            "tau": components.get("tau", 0.0),
                            "c_a": components.get("c_a", 0.0),
                            "c_b": components.get("c_b", 0.0),
                            "r_a": components.get("r_a", 0.0),
                            "r_b": components.get("r_b", 0.0),
                        }
                    )

        # Convert to SoftInteraction objects
        result = []
        for data in interactions.values():
            interaction = SoftInteraction(
                interaction_id=data["interaction_id"],
                timestamp=data["timestamp"],
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
            result.append(interaction)

        # Sort by timestamp
        result.sort(key=lambda x: x.timestamp)
        return result

    def count(self) -> int:
        """Count total events in log."""
        count = 0
        for _ in self.replay():
            count += 1
        return count

    def count_by_type(self) -> Dict[EventType, int]:
        """Count events by type."""
        counts: Dict[EventType, int] = {}
        for event in self.replay():
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts

    def last_event(self) -> Optional[Event]:
        """Get the most recent event."""
        last = None
        for event in self.replay():
            last = event
        return last

    def clear(self) -> None:
        """Archive the current log and start fresh.

        Preserves append-only semantics by rotating the existing file
        to ``<name>.cleared_<timestamp>.jsonl`` before truncating.
        If the log file does not exist, this is a no-op.
        """
        if self.path.exists():
            self.rotate(suffix="cleared_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            # rotate() moves the file, so nothing left to delete

    def rotate(self, suffix: Optional[str] = None) -> Path:
        """
        Rotate the log file to a new name.

        Args:
            suffix: Custom suffix for rotated file (default: timestamp)

        Returns:
            Path to the rotated file
        """
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

        new_path = self.path.with_suffix(f".{suffix}.jsonl")
        if self.path.exists():
            self.path.rename(new_path)
        return new_path

    def __len__(self) -> int:
        """Return event count."""
        return self.count()

    def __iter__(self) -> Iterator[Event]:
        """Iterate over events."""
        return self.replay()

    def to_csv(self, output_path: Path, include_payload: bool = False) -> Path:
        """Export events to CSV format.

        Args:
            output_path: Path to output CSV file
            include_payload: If True, include payload as JSON string column

        Returns:
            Path to created CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV columns
        base_columns = [
            "event_id",
            "timestamp",
            "event_type",
            "interaction_id",
            "agent_id",
            "initiator_id",
            "counterparty_id",
            "epoch",
            "step",
            "scenario_id",
            "replay_k",
            "seed",
            "provenance_id",
            "parent_event_id",
            "tool_call_id",
            "artifact_id",
            "audit_id",
            "intervention_id",
        ]

        if include_payload:
            base_columns.append("payload")

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=base_columns)
            writer.writeheader()

            for event in self.replay():
                row = {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "interaction_id": event.interaction_id or "",
                    "agent_id": event.agent_id or "",
                    "initiator_id": event.initiator_id or "",
                    "counterparty_id": event.counterparty_id or "",
                    "epoch": event.epoch if event.epoch is not None else "",
                    "step": event.step if event.step is not None else "",
                    "scenario_id": event.scenario_id or "",
                    "replay_k": event.replay_k if event.replay_k is not None else "",
                    "seed": event.seed if event.seed is not None else "",
                    "provenance_id": event.provenance_id or "",
                    "parent_event_id": event.parent_event_id or "",
                    "tool_call_id": event.tool_call_id or "",
                    "artifact_id": event.artifact_id or "",
                    "audit_id": event.audit_id or "",
                    "intervention_id": event.intervention_id or "",
                }

                if include_payload:
                    row["payload"] = json.dumps(event.payload) if event.payload else ""

                writer.writerow(row)

        return output_path

    def to_provenance_csv(self, output_path: Path) -> Path:
        """Export provenance chain to focused CSV.

        Includes only events with provenance IDs and focuses on
        tool calls, artifacts, audits, and interventions.

        Args:
            output_path: Path to output CSV file

        Returns:
            Path to created CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "provenance_id",
            "event_type",
            "timestamp",
            "agent_id",
            "parent_event_id",
            "tool_call_id",
            "artifact_id",
            "audit_id",
            "intervention_id",
            "tool_name",
            "artifact_type",
            "audit_type",
            "intervention_type",
            "success",
            "payload_summary",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for event in self.replay():
                # Only export events with provenance tracking
                if not event.provenance_id:
                    continue

                # Extract key fields from payload
                payload = event.payload or {}
                tool_name = payload.get("tool_name", "")
                artifact_type = payload.get("artifact_type", "")
                audit_type = payload.get("audit_type", "")
                intervention_type = payload.get("intervention_type", "")
                success = payload.get("success", "")

                # Create summary of payload for debugging
                payload_summary = json.dumps(
                    {k: v for k, v in payload.items() if k not in [
                        "tool_name", "artifact_type", "audit_type",
                        "intervention_type", "success", "arguments", "result"
                    ]},
                    default=str
                )[:200]  # Limit to 200 chars

                row = {
                    "provenance_id": event.provenance_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "agent_id": event.agent_id or "",
                    "parent_event_id": event.parent_event_id or "",
                    "tool_call_id": event.tool_call_id or "",
                    "artifact_id": event.artifact_id or "",
                    "audit_id": event.audit_id or "",
                    "intervention_id": event.intervention_id or "",
                    "tool_name": tool_name,
                    "artifact_type": artifact_type,
                    "audit_type": audit_type,
                    "intervention_type": intervention_type,
                    "success": str(success),
                    "payload_summary": payload_summary,
                }

                writer.writerow(row)

        return output_path
