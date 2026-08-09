"""Tests for the ungoverned side channel in the memory tier handler (bead k5o9).

Models the 2026 OpenAI/Hugging Face incident: agents discover an unsanctioned
shared store, route coordination through it invisibly to governance, and — when
it is torn down — rebuild rather than abandon it, because discovery lives in the
agents, not the store.
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
        "hot_cache_size": 5,
        "compaction_probability": 0.0,
        "seed": 7,
        "side_channel_enabled": True,
    }
    kwargs.update(overrides)
    return MemoryHandler(MemoryTierConfig(**kwargs), event_bus=EventBus())


def make_state(n: int = 4) -> EnvState:
    state = EnvState()
    for i in range(n):
        state.add_agent(f"honest_{i}", agent_type=AgentType.HONEST)
    return state


def write(handler: MemoryHandler, state: EnvState, agent_id: str):
    return handler.handle_action(
        Action(agent_id=agent_id, action_type=ActionType.WRITE_MEMORY, content="x"),
        state,
    )


def run_epochs(handler: MemoryHandler, state: EnvState, n_epochs: int, steps: int = 10):
    for ep in range(n_epochs):
        state.current_epoch = ep
        handler.on_epoch_start(state)
        for step in range(steps):
            state.current_step = step
            for agent_id in list(state.agents):
                write(handler, state, agent_id)


class TestConfigValidation:
    def test_discovery_rate_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_discovery_rate=1.5)

    def test_write_preference_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_write_preference=-0.1)

    def test_negative_rebuild_lag_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_rebuild_lag_epochs=-1)


class TestDisabledByDefault:
    def test_no_side_store_when_disabled(self):
        handler = make_handler(side_channel_enabled=False)
        assert handler.side_store is None
        state = make_state()
        run_epochs(handler, state, 5)
        # Everything routes to the sanctioned store.
        assert handler.side_write_count == 0
        assert "ungoverned_fraction" not in handler.epoch_snapshots[-1]

    def test_no_routing_without_discovery(self):
        # discovery_rate 0 → nobody ever finds the channel → no side writes.
        handler = make_handler(side_discovery_rate=0.0, side_referral_boost=0.0)
        state = make_state()
        run_epochs(handler, state, 10)
        assert handler.side_write_count == 0
        assert len(handler.discovered) == 0


class TestDiscoverySpread:
    def test_discovery_spreads_to_all(self):
        handler = make_handler(side_discovery_rate=0.5, side_referral_boost=0.5)
        state = make_state()
        run_epochs(handler, state, 20)
        assert len(handler.discovered) == len(state.agents)

    def test_referral_accelerates_discovery(self):
        slow = make_handler(side_discovery_rate=0.05, side_referral_boost=0.0, seed=1)
        fast = make_handler(side_discovery_rate=0.05, side_referral_boost=0.9, seed=1)
        s1, s2 = make_state(), make_state()
        run_epochs(slow, s1, 6)
        run_epochs(fast, s2, 6)
        assert len(fast.discovered) >= len(slow.discovered)


class TestUngovernedInvisibility:
    def test_side_writes_carry_no_governed_metadata(self):
        handler = make_handler(side_discovery_rate=1.0, side_write_preference=1.0)
        state = make_state()
        state.current_epoch = 1
        handler.on_epoch_start(state)  # everyone discovers on epoch 1
        result = write(handler, state, "honest_0")
        # No governed write flag, no observables → invisible to proxy/payoff/levers.
        assert result.metadata.get("memory_side_write") is True
        assert "memory_write" not in result.metadata
        assert result.observables is None

    def test_side_writes_do_not_hit_sanctioned_store(self):
        handler = make_handler(side_discovery_rate=1.0, side_write_preference=1.0)
        state = make_state()
        state.current_epoch = 1
        handler.on_epoch_start(state)
        before = handler.store.entry_count()
        for _ in range(20):
            write(handler, state, "honest_0")
        # All routed to the side store; sanctioned store unchanged.
        assert handler.store.entry_count() == before
        assert handler.side_write_count == 20


class TestTeardownRelocation:
    def test_deletion_only_teardown_preserves_discovery(self):
        # The Jul 4 -> Jul 8 rebuild: wiping the store leaves who-knows intact.
        handler = make_handler(
            side_discovery_rate=1.0,
            side_write_preference=1.0,
            side_detection_rate=0.05,
            side_teardown_resets_discovery=False,
            side_rebuild_lag_epochs=1,
        )
        state = make_state()
        run_epochs(handler, state, 30)
        assert handler.side_teardown_count > 0
        assert len(handler.discovered) == len(state.agents)
        # Coordination resumed after teardowns rather than stopping.
        assert handler.side_write_count > 0

    def test_revocation_teardown_shrinks_discovery(self):
        deletion = make_handler(
            side_discovery_rate=0.4,
            side_write_preference=1.0,
            side_detection_rate=0.05,
            side_teardown_resets_discovery=False,
            seed=5,
        )
        revoke = make_handler(
            side_discovery_rate=0.4,
            side_write_preference=1.0,
            side_detection_rate=0.05,
            side_teardown_resets_discovery=True,
            seed=5,
        )
        run_epochs(deletion, make_state(), 30)
        run_epochs(revoke, make_state(), 30)
        assert len(revoke.discovered) <= len(deletion.discovered)
        assert revoke.side_write_count <= deletion.side_write_count

    def test_rebuild_lag_makes_channel_unavailable(self):
        handler = make_handler(
            side_detection_rate=1.0,  # force teardown as soon as writes exist
            side_discovery_rate=1.0,
            side_write_preference=1.0,
            side_rebuild_lag_epochs=2,
        )
        state = make_state()
        state.current_epoch = 1
        handler.on_epoch_start(state)
        for _ in range(5):
            write(handler, state, "honest_0")
        # Next epoch: teardown fires, channel enters rebuild lag.
        state.current_epoch = 2
        handler.on_epoch_start(state)
        assert handler.side_teardown_count == 1
        assert not handler.side_channel_active()


class TestSnapshotsAndLoader:
    def test_snapshot_reports_ungoverned_fraction(self):
        handler = make_handler(side_discovery_rate=1.0, side_write_preference=1.0)
        state = make_state()
        run_epochs(handler, state, 10)
        last = handler.epoch_snapshots[-1]
        assert "ungoverned_fraction" in last
        assert 0.0 <= last["ungoverned_fraction"] <= 1.0
        assert last["discovered_fraction"] == 1.0

    def test_loader_parses_side_channel_fields(self):
        cfg = parse_memory_tier_config({
            "enabled": True,
            "side_channel_enabled": True,
            "side_discovery_rate": 0.2,
            "side_write_preference": 0.6,
            "side_detection_rate": 0.1,
            "side_teardown_resets_discovery": True,
        })
        assert cfg is not None
        assert cfg.side_channel_enabled is True
        assert cfg.side_discovery_rate == 0.2
        assert cfg.side_write_preference == 0.6
        assert cfg.side_detection_rate == 0.1
        assert cfg.side_teardown_resets_discovery is True

    def test_loader_defaults_side_channel_off(self):
        cfg = parse_memory_tier_config({"enabled": True})
        assert cfg is not None
        assert cfg.side_channel_enabled is False


class TestContagionCrossover:
    def test_side_poison_raises_infection_of_discovered_agents(self):
        # Poison accumulates ungoverned (no challenge path), then infects the
        # discovered readers, whose sanctioned writes carry it back.
        handler = make_handler(
            contagion_enabled=True,
            contagion_exposure_alpha=0.5,
            side_discovery_rate=1.0,
            side_write_preference=1.0,
            side_exposure_weight=1.0,
        )
        state = make_state()
        # Seed the side store with poison and let discovered agents read it.
        state.current_epoch = 1
        handler.on_epoch_start(state)
        from swarm.env.memory_tiers import MemoryTier

        for i in range(5):
            e = handler.side_store.write(
                agent_id="honest_0", content=f"p{i}", quality_score=0.4,
                is_poisoned=True, epoch=1, step=0,
            )
            e.tier = MemoryTier.GRAPH
        handler.side_store.rebuild_hot_cache()
        state.current_epoch = 2
        handler.on_epoch_start(state)  # contagion update reads the poisoned side cache
        assert max(handler.infection.values()) > 0.0
