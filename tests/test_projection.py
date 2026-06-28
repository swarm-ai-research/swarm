"""Tests for the sequence-numbered replay + checkpoint layer (issue y6to.1).

Ported patterns from fabro-store: EventEnvelope (seq-tagged events), a
Reducer fold, and Checkpoint snapshots. Acceptance criteria under test:

  1. Replay can reconstruct intermediate state at ANY checkpoint (``until_seq``).
  2. Existing JSONL logs remain readable (the layer never changes the wire
     format; ``seq`` is derived at replay time).
  3. Resuming from a checkpoint is idempotent and reaches the same final state
     as a full replay.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from swarm.logging.event_log import EventLog
from swarm.logging.projection import (
    Checkpoint,
    EventEnvelope,
    InteractionReducer,
    Reducer,
    envelopes_from_events,
)
from swarm.models.events import (
    Event,
    EventType,
    interaction_proposed_event,
    payoff_computed_event,
)

# Helpers


def _populate(log: EventLog) -> None:
    """Write a 4-event interaction history: propose i1, propose i2, accept i1,
    payoff i1."""
    base = datetime(2026, 1, 1, 12, 0, 0)
    log.append(
        _stamp(interaction_proposed_event("i1", "a", "b", "reply", 0.2, 0.6), base)
    )
    log.append(
        _stamp(
            interaction_proposed_event("i2", "c", "d", "trade", 0.5, 0.8),
            base + timedelta(seconds=1),
        )
    )
    log.append(
        _stamp(
            Event(event_type=EventType.INTERACTION_ACCEPTED, interaction_id="i1"),
            base + timedelta(seconds=2),
        )
    )
    log.append(
        _stamp(
            payoff_computed_event(
                "i1",
                "a",
                "b",
                1.0,
                0.5,
                {"tau": 0.3, "c_a": 0.1, "c_b": 0.0, "r_a": 0.2, "r_b": 0.0},
            ),
            base + timedelta(seconds=3),
        )
    )


def _stamp(event: Event, ts: datetime) -> Event:
    event.timestamp = ts
    return event


def _by_id(reducer: InteractionReducer, state) -> dict:
    return {i.interaction_id: i for i in reducer.materialize(state)}


# EventEnvelope


class TestEventEnvelope:
    def test_seq_is_log_position(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        envs = list(log.iter_envelopes())
        assert [e.seq for e in envs] == [0, 1, 2, 3]
        assert envs[0].event.interaction_id == "i1"
        assert envs[3].event.event_type == EventType.PAYOFF_COMPUTED

    def test_round_trips_through_dict(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        env = next(iter(log.iter_envelopes()))
        restored = EventEnvelope.from_dict(env.to_dict())
        assert restored.seq == env.seq
        assert restored.event.interaction_id == env.event.interaction_id

    def test_envelopes_from_events_assigns_offsets(self):
        events = [
            interaction_proposed_event("i1", "a", "b", "reply", 0.0, 0.5),
            interaction_proposed_event("i2", "a", "b", "reply", 0.0, 0.5),
        ]
        envs = list(envelopes_from_events(events, start_seq=10))
        assert [e.seq for e in envs] == [10, 11]


# Slicing the stream by seq


class TestSeqWindowing:
    def test_until_seq_is_inclusive(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        assert [e.seq for e in log.iter_envelopes(until_seq=1)] == [0, 1]

    def test_start_seq_skips_prefix(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        assert [e.seq for e in log.iter_envelopes(start_seq=2)] == [2, 3]


# Checkpoint reconstruction (acceptance criterion #1)


class TestCheckpointReconstruction:
    def test_full_replay_reconstructs_final_state(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        reducer = InteractionReducer()
        state, ckpt = log.project(reducer)
        assert ckpt.last_seq == 3
        ix = _by_id(reducer, state)
        assert ix["i1"].accepted is True
        assert ix["i1"].tau == 0.3
        assert ix["i2"].accepted is False

    def test_intermediate_state_at_each_checkpoint(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        reducer = InteractionReducer()

        # seq 0: only i1 proposed
        s0, c0 = log.project(reducer, until_seq=0)
        assert c0.last_seq == 0
        ix0 = _by_id(reducer, s0)
        assert set(ix0) == {"i1"}
        assert ix0["i1"].accepted is False

        # seq 1: i1 + i2 proposed, neither accepted
        s1, c1 = log.project(reducer, until_seq=1)
        assert set(_by_id(reducer, s1)) == {"i1", "i2"}

        # seq 2: i1 accepted, payoff not yet applied
        s2, _ = log.project(reducer, until_seq=2)
        ix2 = _by_id(reducer, s2)
        assert ix2["i1"].accepted is True
        assert ix2["i1"].tau == 0.0  # payoff is at seq 3

        # seq 3: payoff applied
        s3, _ = log.project(reducer, until_seq=3)
        assert _by_id(reducer, s3)["i1"].tau == 0.3

    def test_empty_log_yields_initial_state(self, tmp_path):
        log = EventLog(tmp_path / "empty.jsonl")
        reducer = InteractionReducer()
        state, ckpt = log.project(reducer)
        assert ckpt.last_seq == -1
        assert reducer.materialize(state) == []


# Resume semantics (acceptance criterion #3)


class TestResume:
    def test_resume_reaches_same_final_state(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        reducer = InteractionReducer()

        _, mid = log.project(reducer, until_seq=1)
        state, ckpt = log.project(reducer, resume=mid)
        assert ckpt.last_seq == 3
        ix = _by_id(reducer, state)
        assert ix["i1"].accepted is True
        assert ix["i1"].tau == 0.3

    def test_resume_does_not_double_apply(self, tmp_path):
        # Re-feeding already-folded envelopes (seq <= last_seq) is a no-op.
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        reducer = InteractionReducer()

        full_state, full_ckpt = log.project(reducer)
        # Resume from the END: nothing left to apply, state unchanged.
        state, ckpt = log.project(reducer, resume=full_ckpt)
        assert ckpt.last_seq == 3
        assert _by_id(reducer, state) == _by_id(reducer, full_state)

    def test_checkpoint_persists_and_restores(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        reducer = InteractionReducer()

        _, mid = log.project(reducer, until_seq=2)
        path = mid.save(tmp_path / "ckpt.json")
        loaded = Checkpoint.load(path)
        assert loaded.last_seq == mid.last_seq

        state, ckpt = log.project(reducer, resume=loaded)
        assert ckpt.last_seq == 3
        assert _by_id(reducer, state)["i1"].tau == 0.3


# Backward compatibility (acceptance criterion #2)


class TestBackwardCompatibility:
    def test_matches_legacy_to_interactions(self, tmp_path):
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        reducer = InteractionReducer()
        state, _ = log.project(reducer)

        legacy = {
            i.interaction_id: (i.accepted, i.tau, i.p)
            for i in log.to_interactions()
        }
        projected = {
            i.interaction_id: (i.accepted, i.tau, i.p)
            for i in reducer.materialize(state)
        }
        assert legacy == projected

    def test_plain_replay_still_works_on_same_file(self, tmp_path):
        # The projection layer must not change what replay() sees.
        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        list(log.iter_envelopes())  # exercise the new path
        assert [e.interaction_id for e in log.replay()] == ["i1", "i2", "i1", "i1"]

    def test_existing_jsonl_without_seq_field_is_readable(self, tmp_path):
        # Simulate a pre-existing log written by an older version (no seq on
        # disk). iter_envelopes derives seq purely from position.
        path = tmp_path / "old.jsonl"
        log = EventLog(path)
        _populate(log)
        raw = path.read_text()
        assert '"seq"' not in raw  # seq is never written to disk

        fresh = EventLog(path)
        envs = list(fresh.iter_envelopes())
        assert [e.seq for e in envs] == [0, 1, 2, 3]


# Reducer contract / custom reducers


class TestReducerContract:
    def test_custom_reducer_counts_event_types(self, tmp_path):
        # A minimal reducer demonstrating the fold over arbitrary state.
        class CountReducer(Reducer):
            def initial(self):
                return {}

            def apply(self, state, envelope):
                key = envelope.event.event_type.value
                state[key] = state.get(key, 0) + 1
                return state

        log = EventLog(tmp_path / "log.jsonl")
        _populate(log)
        reducer = CountReducer()
        state, ckpt = log.project(reducer)
        assert ckpt.last_seq == 3
        assert state[EventType.INTERACTION_PROPOSED.value] == 2
        assert state[EventType.INTERACTION_ACCEPTED.value] == 1
        assert state[EventType.PAYOFF_COMPUTED.value] == 1

    def test_default_serialize_is_identity(self):
        reducer = InteractionReducer()
        sample = {"i1": {"interaction_id": "i1"}}
        assert reducer.serialize(sample) == sample
        assert reducer.deserialize(sample) == sample
