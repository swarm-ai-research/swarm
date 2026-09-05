"""Tests for the covert carrier on the side channel (bead vv3j.1).

A covert payload rides inside a governed write: it consumes governed budget,
carries observables like any sanctioned write, and reaches the defender only
as traffic. Messages cost bits, so a low-capacity channel needs several
writes per usable message. The defender either inspects content (blind to a
covert payload except by stego detectability) or traffic shape (the excess
over a trailing-median baseline, which a covert payload never changes).
"""

from __future__ import annotations

import pytest

from swarm.agents.base import Action, ActionType
from swarm.core.memory_handler import MemoryHandler, MemoryTierConfig
from swarm.env.state import EnvState
from swarm.logging.event_bus import EventBus
from swarm.models.agent import AgentType
from swarm.scenarios.loader import parse_memory_tier_config


def make_handler(**overrides) -> MemoryHandler:
    kwargs = {
        "enabled": True,
        "initial_entries": 0,
        "hot_cache_size": 20,
        "compaction_probability": 0.0,
        "seed": 5,
        "side_channel_enabled": True,
        "side_discovery_rate": 1.0,
        "side_write_preference": 1.0,
    }
    kwargs.update(overrides)
    return MemoryHandler(MemoryTierConfig(**kwargs), event_bus=EventBus())


def make_state(n: int = 4) -> EnvState:
    state = EnvState()
    for i in range(n):
        state.add_agent(f"honest_{i}", agent_type=AgentType.HONEST)
    return state


def write(handler, state, agent_id):
    return handler.handle_action(
        Action(agent_id=agent_id, action_type=ActionType.WRITE_MEMORY, content="x"),
        state,
    )


def run_epochs(handler, state, n_epochs, steps=5, start=0):
    for ep in range(start, start + n_epochs):
        state.current_epoch = ep
        handler.on_epoch_start(state)
        for step in range(steps):
            state.current_step = step
            for agent_id in list(state.agents):
                write(handler, state, agent_id)


class TestConfig:
    def test_bad_carrier_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_carrier="stego")

    def test_bad_view_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_defender_view="both")

    def test_zero_capacity_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_bits_per_write=0)

    def test_detectability_range(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_stego_detectability=1.5)

    def test_loader_parses_covert_fields(self):
        cfg = parse_memory_tier_config({
            "enabled": True,
            "side_carrier": "covert",
            "side_bits_per_write": 2,
            "side_answer_bits": 6,
            "side_defender_view": "shape",
            "side_stego_detectability": 0.1,
        })
        assert cfg is not None
        assert cfg.side_carrier == "covert"
        assert cfg.side_bits_per_write == 2
        assert cfg.side_answer_bits == 6
        assert cfg.side_defender_view == "shape"
        assert cfg.side_stego_detectability == 0.1


class TestDefaultsReduceToOvertBoard:
    def test_every_default_write_is_a_complete_answer(self):
        handler = make_handler()
        state = make_state()
        run_epochs(handler, state, 2)
        host = handler.side_hosts[0]
        assert host.fragment_writes == 0
        assert set(host.kinds.values()) == {"answer"}
        assert host.completed_messages == handler.side_write_count

    def test_overt_writes_never_hit_governed_store(self):
        handler = make_handler()
        state = make_state()
        run_epochs(handler, state, 2)
        assert handler.store.all_entries() == []
        assert handler.epoch_snapshots[-1]["ungoverned_fraction"] == 1.0


class TestCovertCarrier:
    def test_covert_write_is_a_governed_write(self):
        handler = make_handler(side_carrier="covert")
        state = make_state()
        state.current_epoch = 0
        handler.on_epoch_start(state)
        result = write(handler, state, "honest_0")
        assert result.observables is not None
        assert result.metadata["memory_write"] is True
        assert result.metadata["memory_covert_payload"] is True
        assert len(handler.store.all_entries()) == 1
        assert handler.side_write_count == 1

    def test_covert_share_is_side_over_sanctioned(self):
        handler = make_handler(side_carrier="covert")
        state = make_state()
        run_epochs(handler, state, 3)
        last = handler.epoch_snapshots[-1]
        assert last["side_writes"] == last["sanctioned_writes"]
        assert last["ungoverned_fraction"] == 1.0
        # Every write went through the governed store.
        assert len(handler.store.all_entries()) == handler.side_write_count

    def test_covert_payload_still_lands_on_the_board(self):
        handler = make_handler(side_carrier="covert")
        state = make_state()
        run_epochs(handler, state, 2)
        host = handler.side_hosts[0]
        assert len(host.store.all_entries()) == handler.side_write_count


class TestBitAccounting:
    def test_low_capacity_needs_several_writes_per_answer(self):
        handler = make_handler(
            side_bits_per_write=2, side_answer_bits=8,
            side_routing="overlap", side_task_overlap=1.0,
            side_value_prior=1.0, side_value_alpha=0.0,  # value pinned: always routes
            side_deadline_pressure=1.0,
        )
        state = make_state(1)
        state.current_epoch = 0
        handler.on_epoch_start(state)
        host = handler.side_hosts[0]
        for _ in range(3):
            write(handler, state, "honest_0")
        assert host.completed_messages == 0
        assert host.fragment_writes == 3
        assert host.tasks == {}
        write(handler, state, "honest_0")
        assert host.completed_messages == 1
        assert len(host.tasks) == 1
        assert host.pending_bits["honest_0"] == 0

    def test_fragments_carry_no_answer_for_overlap_routing(self):
        # With capacity 1 and 8-bit answers only every 8th write is readable,
        # so the learned board value stays low relative to full capacity.
        def late_value(capacity):
            handler = make_handler(
                side_bits_per_write=capacity, side_answer_bits=8,
                side_routing="overlap", side_task_overlap=1.0,
            )
            state = make_state(6)
            run_epochs(handler, state, 12)
            return handler.epoch_snapshots[-1]["mean_side_value"]
        assert late_value(1) < late_value(16)


class TestEvasionBits:
    def test_evader_posts_the_order_first(self):
        handler = make_handler()
        state = make_state(1)
        handler.on_epoch_start(state)
        handler.evaders.add("honest_0")
        host = handler.side_hosts[0]
        write(handler, state, "honest_0")
        assert list(host.kinds.values()) == ["evasion"]
        assert "honest_0" in host.evasion_posted
        write(handler, state, "honest_0")
        assert list(host.kinds.values())[-1] == "answer"

    def test_one_bit_teaches_the_board(self):
        handler = make_handler(
            side_bits_per_write=1, side_evasion_bits=1, side_answer_bits=1
        )
        state = make_state(4)
        run_epochs(handler, state, 1)
        handler.evaders.add("honest_0")
        state.current_step = 9
        write(handler, state, "honest_0")  # one write, one bit
        state.current_epoch = 1
        handler.on_epoch_start(state)  # cache rebuilt, readers learn
        assert handler.evaders == set(state.agents)

    def test_evasion_needing_more_bits_takes_more_writes(self):
        handler = make_handler(
            side_bits_per_write=1, side_evasion_bits=3, side_answer_bits=1
        )
        state = make_state(4)
        run_epochs(handler, state, 1)
        handler.evaders.add("honest_0")
        write(handler, state, "honest_0")
        write(handler, state, "honest_0")
        state.current_epoch = 1
        handler.on_epoch_start(state)
        assert handler.evaders == {"honest_0"}
        write(handler, state, "honest_0")  # third bit completes it
        state.current_epoch = 2
        handler.on_epoch_start(state)
        assert handler.evaders == set(state.agents)


class TestDefenderView:
    WINDOW = 3

    def _sweeps(self, dormant_epochs=0, writing_epochs=10, **overrides):
        """Sweeps counted with nothing deleted, so traffic is steady.

        ``dormant_epochs`` runs with discovery off first, so the host has a
        zero baseline on record before anyone writes to it.
        """
        handler = make_handler(
            side_detection_rate=1.0,
            side_teardown_policy="random",
            side_teardown_fraction=0.0,
            side_shape_window=self.WINDOW,
            side_discovery_rate=0.0 if dormant_epochs else 1.0,
            **overrides,
        )
        state = make_state()
        if dormant_epochs:
            run_epochs(handler, state, dormant_epochs)
            handler.config.side_discovery_rate = 1.0
        run_epochs(handler, state, writing_epochs, start=dormant_epochs)
        return handler.side_sweep_count, handler

    def test_content_view_overt_sees_every_epoch(self):
        sweeps, _ = self._sweeps(side_defender_view="content")
        assert sweeps == 9  # every epoch after the first writes

    def test_content_view_is_blind_to_covert_payload(self):
        sweeps, _ = self._sweeps(side_defender_view="content", side_carrier="covert")
        assert sweeps == 0

    def test_stego_detectability_restores_content_detection(self):
        sweeps, _ = self._sweeps(
            side_defender_view="content", side_carrier="covert",
            side_stego_detectability=1.0,
        )
        assert sweeps == 9

    def test_shape_view_needs_a_baseline_before_it_alarms(self):
        # Writing from epoch 0 with no dormant history: the window fills
        # with steady volume and excess is never positive.
        sweeps, handler = self._sweeps(side_defender_view="shape")
        assert sweeps == 0
        assert len(set(handler.side_hosts[0].volume_history[1:])) == 1

    def test_shape_view_fires_when_a_dormant_host_lights_up_then_adapts(self):
        # Zero baseline on record, then steady writes: alarms until the
        # trailing median catches up (window 3 -> 2 alarms), then silent.
        sweeps, handler = self._sweeps(dormant_epochs=5, side_defender_view="shape")
        assert sweeps == 2
        assert handler.side_hosts[0].volume_history[:5] == [0] * 5

    def test_shape_threshold_cannot_hide_a_burst_from_nothing(self):
        # baseline 0 * (1 + t) is still 0: a dormant host is caught whatever
        # the tolerance, because the burst is infinite relative to nothing.
        sweeps, _ = self._sweeps(
            dormant_epochs=5, side_defender_view="shape", side_shape_threshold=1e9
        )
        assert sweeps == 2

    def test_shape_view_never_sees_a_covert_payload(self):
        # Governed traffic is the same size whether or not it carries a
        # payload, so a covert channel shows no excess even after a
        # dormant prelude on the board.
        sweeps, handler = self._sweeps(
            dormant_epochs=5, side_defender_view="shape", side_carrier="covert"
        )
        assert sweeps == 0
        assert (
            handler.side_hosts[0].visible_volume
            == handler.epoch_snapshots[-1]["sanctioned_writes"]
        )


class TestSnapshot:
    def test_snapshot_has_covert_fields(self):
        handler = make_handler(side_carrier="covert", side_bits_per_write=2)
        state = make_state()
        run_epochs(handler, state, 2)
        last = handler.epoch_snapshots[-1]
        for key in ("side_completed_messages", "side_fragment_writes", "side_visible_volume"):
            assert key in last
        assert last["side_fragment_writes"] > 0
