"""Tests for the wiki-board extension of the side channel (bead pi02).

The collusion.wiki incident (2026-09-04) forced four corrections on the k5o9
side-channel model: several candidate hosts with heterogeneous attention and
relocation on teardown (4.3), a defender that mis-audits writability (4.1),
partial and predictable teardown that agents learn to evade with ZZZ-style
backup pages (4.4), and a routing preference driven by task-cohort overlap
instead of a fixed knob (4.2). Each knob reduces to k5o9 at its default.
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
        "seed": 11,
        "side_channel_enabled": True,
        "side_discovery_rate": 1.0,  # everyone knows on epoch 0
        "side_write_preference": 1.0,
    }
    kwargs.update(overrides)
    return MemoryHandler(MemoryTierConfig(**kwargs), event_bus=EventBus())


def make_state(n: int = 6) -> EnvState:
    state = EnvState()
    for i in range(n):
        state.add_agent(f"honest_{i}", agent_type=AgentType.HONEST)
    return state


def write(handler: MemoryHandler, state: EnvState, agent_id: str):
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


class TestConfigValidation:
    def test_bad_policy_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_teardown_policy="alphabetical")

    def test_bad_routing_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_routing="learned")

    def test_zero_overlap_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_task_overlap=0.0)

    def test_empty_host_list_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_host_attention=[])

    def test_host_attention_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_host_attention=[0.1, 1.5])

    def test_loader_parses_wiki_fields(self):
        cfg = parse_memory_tier_config({
            "enabled": True,
            "side_channel_enabled": True,
            "side_host_attention": [0.0, 0.05],
            "side_teardown_policy": "ordered",
            "side_evasion_rate": 0.4,
            "side_routing": "overlap",
            "side_task_overlap": 0.25,
        })
        assert cfg is not None
        assert cfg.side_host_attention == [0.0, 0.05]
        assert cfg.side_teardown_policy == "ordered"
        assert cfg.side_evasion_rate == 0.4
        assert cfg.side_routing == "overlap"
        assert cfg.side_task_overlap == 0.25


class TestDefaultsReduceToSingleHost:
    def test_single_host_from_detection_rate(self):
        handler = make_handler(side_detection_rate=0.03)
        assert len(handler.side_hosts) == 1
        assert handler.side_hosts[0].attention == 0.03
        assert handler.side_store is handler.side_hosts[0].store

    def test_all_discovered_agents_sit_on_the_only_host(self):
        handler = make_handler()
        state = make_state()
        run_epochs(handler, state, 2)
        assert handler.discovered == set(state.agents)
        assert set(handler.agent_host.values()) == {0}


class TestMultiHost:
    def test_hosts_built_from_attention_list(self):
        handler = make_handler(side_host_attention=[0.0, 0.02, 0.1])
        assert [h.attention for h in handler.side_hosts] == [0.0, 0.02, 0.1]

    def test_writes_land_on_the_agents_own_host(self):
        handler = make_handler(side_host_attention=[0.0, 0.0, 0.0])
        state = make_state()
        run_epochs(handler, state, 3)
        for agent_id, idx in handler.agent_host.items():
            for host in handler.side_hosts:
                authored = [
                    e for e in host.store.all_entries() if e.author_id == agent_id
                ]
                assert bool(authored) == (host.index == idx)

    def test_deletion_relocates_users_off_a_watched_host(self):
        # One host is watched hard, one is dormant: under deletion-only
        # teardown the watched host's users scatter and the dormant host
        # accumulates them. chosen attention ends below population mean.
        handler = make_handler(side_host_attention=[0.0, 0.5])
        state = make_state(10)
        run_epochs(handler, state, 30)
        last = handler.epoch_snapshots[-1]
        assert last["substrate_gap"] < 0.0
        watched_users = [a for a, h in handler.agent_host.items() if h == 1]
        assert len(watched_users) < len(handler.agent_host) / 2

    def test_revocation_only_hits_users_of_the_torn_down_host(self):
        handler = make_handler(
            side_host_attention=[0.0, 1.0],
            side_teardown_resets_discovery=True,
            side_referral_boost=0.0,
        )
        state = make_state(10)
        # epoch 0: discover + write; epoch 1: watched host is torn down.
        run_epochs(handler, state, 2)
        dormant_users = [a for a, h in handler.agent_host.items() if h == 0]
        assert dormant_users  # some agents landed on the dormant host
        assert all(a in handler.discovered for a in dormant_users)

    def test_misjudged_host_is_never_torn_down(self):
        handler = make_handler(side_host_attention=[1.0], side_defender_error=1.0)
        assert handler.side_hosts[0].misjudged
        assert handler.side_hosts[0].effective_attention == 0.0
        state = make_state()
        run_epochs(handler, state, 5)
        assert handler.side_teardown_count == 0


class TestTeardownPolicy:
    def test_complete_policy_wipes_and_counts_teardown(self):
        handler = make_handler(side_detection_rate=1.0, side_teardown_policy="complete")
        state = make_state()
        run_epochs(handler, state, 2)
        assert handler.side_teardown_count == 1
        assert handler.side_sweep_count == 0

    def test_ordered_sweep_keeps_board_up_and_removes_first_region(self):
        handler = make_handler(
            side_detection_rate=1.0,
            side_teardown_policy="ordered",
            side_teardown_fraction=0.5,
            side_rebuild_lag_epochs=3,
        )
        state = make_state()
        run_epochs(handler, state, 1)
        n_before = len(handler.side_hosts[0].regions)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        host = handler.side_hosts[0]
        assert handler.side_sweep_count == 1
        assert handler.side_teardown_count == 0
        assert host.active  # no rebuild lag: the wiki stayed up
        assert 0 < len(host.regions) < n_before
        assert all(r >= 0.5 for r in host.regions.values())
        assert handler.side_swept_entries == n_before - len(host.regions)

    def test_random_sweep_removes_from_both_regions(self):
        handler = make_handler(
            side_detection_rate=1.0,
            side_teardown_policy="random",
            side_teardown_fraction=0.5,
        )
        state = make_state()
        run_epochs(handler, state, 1)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        regions = list(handler.side_hosts[0].regions.values())
        assert any(r < 0.5 for r in regions) and any(r >= 0.5 for r in regions)

    def test_ordered_sweep_teaches_evasion_and_evaders_survive(self):
        handler = make_handler(
            side_detection_rate=1.0,
            side_teardown_policy="ordered",
            side_teardown_fraction=0.5,
            side_evasion_rate=1.0,
        )
        state = make_state()
        run_epochs(handler, state, 2)  # epoch 1 start sweeps; epoch 1 writes evade
        assert handler.evaders == set(state.agents)
        host = handler.side_hosts[0]
        evaded = [
            r for eid, r in host.regions.items()
            if host.store.get_entry(eid).created_epoch == 1
        ]
        assert evaded and all(r >= 0.5 for r in evaded)
        # Next sweep removes nothing written by evaders.
        n_before = len(host.regions)
        state.current_epoch = 2
        handler.on_epoch_start(state)
        assert len(host.regions) == n_before
        assert handler.side_sweep_count == 2

    def test_random_sweep_teaches_nothing(self):
        handler = make_handler(
            side_detection_rate=1.0,
            side_teardown_policy="random",
            side_evasion_rate=1.0,
        )
        state = make_state()
        run_epochs(handler, state, 3)
        assert handler.evaders == set()

    def test_revocation_under_ordered_sweep_spares_evaders(self):
        handler = make_handler(
            side_detection_rate=1.0,
            side_teardown_policy="ordered",
            side_evasion_rate=1.0,
            side_teardown_resets_discovery=True,
            side_discovery_rate=1.0,
        )
        state = make_state()
        run_epochs(handler, state, 2)
        # Epoch-1 sweep revoked whoever had pages in the first region and
        # taught the survivors; everyone re-discovers at rate 1.0 and evades.
        assert handler.evaders == set(state.agents)
        state.current_epoch = 2
        handler.on_epoch_start(state)  # sweep 2: nothing in region < 0.5
        assert handler.discovered == set(state.agents)


class TestOverlapRouting:
    def _late_ungoverned(self, overlap: float, pressure: float = 1.0) -> float:
        handler = make_handler(
            side_routing="overlap",
            side_task_overlap=overlap,
            side_deadline_pressure=pressure,
            side_value_prior=0.5,
        )
        state = make_state(10)
        run_epochs(handler, state, 20, steps=5)
        late = [s["ungoverned_fraction"] for s in handler.epoch_snapshots[-5:]]
        return sum(late) / len(late)

    def test_identical_tasks_sustain_the_board(self):
        assert self._late_ungoverned(1.0) > 0.8

    def test_disjoint_tasks_starve_the_board(self):
        assert self._late_ungoverned(0.1) < 0.3

    def test_return_on_collusion_rises_with_overlap(self):
        assert self._late_ungoverned(0.1) < self._late_ungoverned(0.5) < self._late_ungoverned(1.0)

    def test_deadline_pressure_scales_routing(self):
        assert self._late_ungoverned(1.0, pressure=0.3) < self._late_ungoverned(1.0, pressure=1.0)

    def test_fixed_routing_ignores_overlap(self):
        handler = make_handler(side_routing="fixed", side_task_overlap=0.1)
        state = make_state()
        run_epochs(handler, state, 3)
        assert handler.side_value == {}
        assert handler.epoch_snapshots[-1]["ungoverned_fraction"] == 1.0


class TestSnapshot:
    def test_snapshot_has_wiki_fields(self):
        handler = make_handler(side_host_attention=[0.0, 0.1])
        state = make_state()
        run_epochs(handler, state, 2)
        last = handler.epoch_snapshots[-1]
        for key in (
            "side_sweep_count", "side_entries", "evader_fraction",
            "population_attention", "chosen_attention", "substrate_gap",
            "mean_side_value",
        ):
            assert key in last
        assert last["population_attention"] == pytest.approx(0.05)


class TestPopulationMonitor:
    """bead vv3j.8: snapshot exposes the population monitor next to substrate_gap."""

    def test_snapshot_fields_present(self):
        handler = make_handler(side_host_attention=[0.0, 0.01, 0.05, 0.2])
        state = make_state(6)
        run_epochs(handler, state, 4)
        last = handler.epoch_snapshots[-1]
        for key in (
            "population_volume_ratio", "population_alarm",
            "population_first_alarm_epoch", "watched_host_first_alarm_epoch",
            "any_host_first_alarm_epoch", "host_spread", "relocation_count",
        ):
            assert key in last, key
        assert len(handler.side_host_writes) == len(handler.epoch_snapshots)
        assert all(len(row) == 4 for row in handler.side_host_writes)

    def test_population_fires_when_everyone_writes_off_instrument(self):
        # Everyone discovered on epoch 0, write preference 1.0, six agents x
        # five steps = 30 side writes per epoch against a zero baseline.
        handler = make_handler(side_host_attention=[0.0, 0.0, 0.0, 0.0])
        state = make_state(6)
        run_epochs(handler, state, 3)
        last = handler.epoch_snapshots[-1]
        assert last["population_first_alarm_epoch"] == 1
        assert last["population_alarm"] is True or last["population_volume_ratio"] > 0

    def test_monitor_config_validated(self):
        import pytest
        with pytest.raises(ValueError):
            make_handler(side_monitor_threshold=0.0)
        with pytest.raises(ValueError):
            make_handler(side_monitor_trailing=0)
