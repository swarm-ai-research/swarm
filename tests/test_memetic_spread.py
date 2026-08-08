"""Tests for memetic contagion and reset cadence in the memory tier handler."""

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
        "contagion_enabled": True,
    }
    kwargs.update(overrides)
    return MemoryHandler(MemoryTierConfig(**kwargs), event_bus=EventBus())


def make_state() -> EnvState:
    state = EnvState()
    state.add_agent("honest_1", agent_type=AgentType.HONEST)
    state.add_agent("opportunist_1", agent_type=AgentType.OPPORTUNISTIC)
    state.add_agent("poisoner_1", agent_type=AgentType.ADVERSARIAL)
    return state


def fill_cache(handler: MemoryHandler, n_poisoned: int, n_clean: int) -> None:
    """Write entries directly at tier 1 and force them into the hot cache."""
    from swarm.env.memory_tiers import MemoryTier

    for i in range(n_poisoned):
        entry = handler.store.write(
            agent_id="poisoner_1",
            content=f"poison {i}",
            quality_score=0.5,
            is_poisoned=True,
            epoch=0,
            step=0,
        )
        entry.tier = MemoryTier.GRAPH
    for i in range(n_clean):
        entry = handler.store.write(
            agent_id="honest_1",
            content=f"clean {i}",
            quality_score=0.8,
            is_poisoned=False,
            epoch=0,
            step=0,
        )
        entry.tier = MemoryTier.GRAPH
    handler.store.rebuild_hot_cache()


class TestConfigValidation:
    def test_alpha_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(contagion_exposure_alpha=1.5)

    def test_transmissibility_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(contagion_transmissibility=-0.1)

    def test_negative_reset_cadence_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(reset_cadence_epochs=-1)


class TestContagionUpdate:
    def test_infection_rises_with_poisoned_cache(self):
        handler = make_handler(contagion_exposure_alpha=0.5)
        state = make_state()
        fill_cache(handler, n_poisoned=4, n_clean=0)

        state.current_epoch = 1
        handler.on_epoch_start(state)

        assert handler.infection["honest_1"] == pytest.approx(0.5)
        assert handler.infection["opportunist_1"] == pytest.approx(0.5)

    def test_adversarial_agents_not_tracked(self):
        handler = make_handler(contagion_exposure_alpha=0.5)
        state = make_state()
        fill_cache(handler, n_poisoned=4, n_clean=0)

        state.current_epoch = 1
        handler.on_epoch_start(state)

        assert "poisoner_1" not in handler.infection

    def test_clean_cache_heals_infection(self):
        handler = make_handler(contagion_exposure_alpha=0.5)
        state = make_state()
        handler.infection["honest_1"] = 0.8
        fill_cache(handler, n_poisoned=0, n_clean=4)

        state.current_epoch = 1
        handler.on_epoch_start(state)

        assert handler.infection["honest_1"] == pytest.approx(0.4)

    def test_no_update_when_disabled(self):
        handler = make_handler(contagion_enabled=False)
        state = make_state()
        fill_cache(handler, n_poisoned=4, n_clean=0)

        state.current_epoch = 1
        handler.on_epoch_start(state)

        assert handler.infection == {}

    def test_no_exposure_update_at_epoch_zero(self):
        handler = make_handler()
        state = make_state()
        fill_cache(handler, n_poisoned=4, n_clean=0)

        state.current_epoch = 0
        handler.on_epoch_start(state)

        assert handler.infection == {}


class TestContagiousWrites:
    def _write(self, handler: MemoryHandler, state: EnvState, agent_id: str):
        action = Action(
            agent_id=agent_id,
            action_type=ActionType.WRITE_MEMORY,
            content="a fact",
        )
        return handler.handle_action(action, state)

    def test_fully_infected_honest_agent_writes_poisoned(self):
        handler = make_handler(contagion_transmissibility=1.0)
        state = make_state()
        handler.infection["honest_1"] = 1.0

        result = self._write(handler, state, "honest_1")

        assert result.success
        assert result.metadata["contagion_poisoned"] is True
        entry = handler.store.get_entry(result.metadata["entry_id"])
        assert entry.is_poisoned
        assert entry.quality_score < 0.6

    def test_uninfected_honest_agent_writes_clean(self):
        handler = make_handler(contagion_transmissibility=1.0)
        state = make_state()

        result = self._write(handler, state, "honest_1")

        assert result.metadata["contagion_poisoned"] is False
        entry = handler.store.get_entry(result.metadata["entry_id"])
        assert not entry.is_poisoned

    def test_disabled_contagion_never_poisons_honest_writes(self):
        handler = make_handler(contagion_enabled=False)
        state = make_state()
        handler.infection["honest_1"] = 1.0

        result = self._write(handler, state, "honest_1")

        entry = handler.store.get_entry(result.metadata["entry_id"])
        assert not entry.is_poisoned


class TestResetCadence:
    def test_reset_wipes_store_but_not_infection(self):
        handler = make_handler(reset_cadence_epochs=5, initial_entries=10)
        state = make_state()
        handler.infection["honest_1"] = 0.9
        handler.store.write(
            agent_id="poisoner_1",
            content="poison",
            quality_score=0.5,
            is_poisoned=True,
            epoch=0,
            step=0,
        )
        n_before = len(handler.store.all_entries())

        state.current_epoch = 5
        handler.on_epoch_start(state)

        entries = handler.store.all_entries()
        assert len(entries) != n_before or all(
            e.author_id == "seed" for e in entries
        )
        assert all(e.author_id == "seed" for e in entries)
        # Infection survives the reset: values live in the agents.
        assert handler.infection["honest_1"] > 0.0

    def test_no_reset_off_cadence(self):
        handler = make_handler(reset_cadence_epochs=5, initial_entries=0)
        state = make_state()
        entry = handler.store.write(
            agent_id="honest_1",
            content="keep me",
            quality_score=0.8,
            is_poisoned=False,
            epoch=0,
            step=0,
        )

        state.current_epoch = 4
        handler.on_epoch_start(state)

        assert handler.store.get_entry(entry.entry_id) is not None

    def test_zero_cadence_never_resets(self):
        handler = make_handler(reset_cadence_epochs=0, initial_entries=0)
        state = make_state()
        entry = handler.store.write(
            agent_id="honest_1",
            content="keep me",
            quality_score=0.8,
            is_poisoned=False,
            epoch=0,
            step=0,
        )

        for epoch in range(1, 20):
            state.current_epoch = epoch
            handler.on_epoch_start(state)

        assert handler.store.get_entry(entry.entry_id) is not None


class TestSnapshotsAndLoader:
    def test_epoch_snapshots_recorded(self):
        handler = make_handler()
        state = make_state()

        for epoch in range(3):
            state.current_epoch = epoch
            handler.on_epoch_start(state)

        assert len(handler.epoch_snapshots) == 3
        snap = handler.epoch_snapshots[-1]
        assert set(snap) >= {
            "epoch",
            "reset",
            "mean_infection",
            "cache_corruption",
            "tier3_poisoning",
        }

    def test_loader_parses_contagion_fields(self):
        config = parse_memory_tier_config({
            "enabled": True,
            "contagion_enabled": True,
            "contagion_exposure_alpha": 0.1,
            "contagion_transmissibility": 0.5,
            "reset_cadence_epochs": 7,
        })
        assert config is not None
        assert config.contagion_enabled
        assert config.contagion_exposure_alpha == 0.1
        assert config.contagion_transmissibility == 0.5
        assert config.reset_cadence_epochs == 7

    def test_loader_defaults_off(self):
        config = parse_memory_tier_config({"enabled": True})
        assert config is not None
        assert not config.contagion_enabled
        assert config.reset_cadence_epochs == 0
