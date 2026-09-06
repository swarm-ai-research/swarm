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
        assert config.cache_ranking == "quality"


class TestCacheRanking:
    def _seed_graph_entries(self, store):
        from swarm.env.memory_tiers import MemoryTier

        specs = [
            ("old_high_q", 0.9, 0),
            ("mid", 0.6, 5),
            ("new_low_q", 0.3, 9),
        ]
        entries = {}
        for name, quality, epoch in specs:
            e = store.write(
                agent_id=name,
                content=name,
                quality_score=quality,
                is_poisoned=False,
                epoch=epoch,
                step=0,
            )
            e.tier = MemoryTier.GRAPH
            entries[name] = e
        return entries

    def test_invalid_ranking_rejected(self):
        from swarm.env.memory_tiers import MemoryStore

        with pytest.raises(ValueError):
            MemoryStore(seed=1, ranking="virality")
        with pytest.raises(ValueError):
            MemoryTierConfig(cache_ranking="virality")

    def test_quality_ranking_prefers_high_quality(self):
        from swarm.env.memory_tiers import MemoryStore

        store = MemoryStore(seed=1, ranking="quality")
        store._hot_cache_size = 2
        self._seed_graph_entries(store)
        cache = store.rebuild_hot_cache()
        assert [e.author_id for e in cache] == ["old_high_q", "mid"]

    def test_recency_ranking_prefers_newest(self):
        from swarm.env.memory_tiers import MemoryStore

        store = MemoryStore(seed=1, ranking="recency")
        store._hot_cache_size = 2
        self._seed_graph_entries(store)
        cache = store.rebuild_hot_cache()
        assert [e.author_id for e in cache] == ["new_low_q", "mid"]

    def test_engagement_ranking_locks_in_read_entries(self):
        from swarm.env.memory_tiers import MemoryStore

        store = MemoryStore(seed=1, ranking="engagement")
        store._hot_cache_size = 2
        entries = self._seed_graph_entries(store)
        entries["new_low_q"].read_count = 10  # already-popular entry
        cache = store.rebuild_hot_cache()
        assert cache[0].author_id == "new_low_q"

    def test_cache_membership_increments_reads(self):
        from swarm.env.memory_tiers import MemoryStore

        store = MemoryStore(seed=1, ranking="quality")
        store._hot_cache_size = 2
        self._seed_graph_entries(store)
        store.rebuild_hot_cache()
        store.rebuild_hot_cache()
        reads = {e.author_id: e.read_count for e in store.all_entries()}
        assert reads["old_high_q"] == 2
        assert reads["new_low_q"] == 0

    def test_reset_preserves_ranking_policy(self):
        handler = make_handler(cache_ranking="recency", reset_cadence_epochs=1)
        state = make_state()
        state.current_epoch = 1
        handler.on_epoch_start(state)
        assert handler.store._ranking == "recency"


# ---------------------------------------------------------------------------
# Whistleblower faction (arXiv:2609.04170)
# ---------------------------------------------------------------------------


def make_wb_state(n_honest: int = 4) -> EnvState:
    state = EnvState()
    for i in range(n_honest):
        state.add_agent(f"honest_{i}", agent_type=AgentType.HONEST)
    state.add_agent("poisoner_1", agent_type=AgentType.ADVERSARIAL)
    return state


class TestWhistleblowers:
    def test_config_defaults_off(self):
        cfg = MemoryTierConfig()
        assert cfg.whistleblower_fraction == 0.0
        assert cfg.whistleblower_audit_rate == 0.0
        assert cfg.whistleblower_warning_strength == 0.0
        assert cfg.whistleblower_boycott_rate == 0.0

    @pytest.mark.parametrize(
        "field_name",
        [
            "whistleblower_fraction",
            "whistleblower_audit_rate",
            "whistleblower_warning_strength",
            "whistleblower_boycott_rate",
        ],
    )
    def test_out_of_range_rejected(self, field_name):
        with pytest.raises(ValueError):
            MemoryTierConfig(**{field_name: 1.5})

    def test_faction_drawn_from_honest_roster_only(self):
        handler = make_handler(whistleblower_fraction=0.5)
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        assert len(handler.whistleblowers) == 2
        assert "poisoner_1" not in handler.whistleblowers
        # Drawn once: a later epoch does not redraw.
        before = set(handler.whistleblowers)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        assert handler.whistleblowers == before

    def test_small_fraction_rounds_up_to_one(self):
        handler = make_handler(whistleblower_fraction=0.05)
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        assert len(handler.whistleblowers) == 1

    def test_zero_fraction_no_faction(self):
        handler = make_handler(whistleblower_fraction=0.0)
        state = make_wb_state()
        handler.on_epoch_start(state)
        assert handler.whistleblowers == set()

    def test_whistleblower_immune_to_contagion(self):
        handler = make_handler(whistleblower_fraction=0.25)
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        (wb,) = handler.whistleblowers
        fill_cache(handler, n_poisoned=5, n_clean=0)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        assert handler.infection[wb] == 0.0
        others = [a for a in handler.infection if a != wb]
        assert all(handler.infection[a] > 0.0 for a in others)

    def test_audit_reverts_poisoned_cache_entries(self):
        handler = make_handler(
            whistleblower_fraction=0.5, whistleblower_audit_rate=1.0
        )
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        fill_cache(handler, n_poisoned=3, n_clean=2)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        assert handler.whistleblower_revert_count == 3
        assert handler.whistleblower_flags_last_epoch == 3
        # Reverted entries leave the rebuilt cache; clean ones stay.
        cache = handler.store.hot_cache
        assert len(cache) == 2
        assert not any(e.is_poisoned for e in cache)
        assert handler.epoch_snapshots[-1]["whistleblower_flags"] == 3

    def test_audit_never_touches_clean_entries(self):
        handler = make_handler(
            whistleblower_fraction=0.5, whistleblower_audit_rate=1.0
        )
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        fill_cache(handler, n_poisoned=0, n_clean=4)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        assert handler.whistleblower_revert_count == 0
        assert len(handler.store.hot_cache) == 4

    def test_zero_audit_rate_reverts_nothing(self):
        handler = make_handler(
            whistleblower_fraction=0.5, whistleblower_audit_rate=0.0
        )
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        fill_cache(handler, n_poisoned=3, n_clean=0)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        assert handler.whistleblower_revert_count == 0
        assert len(handler.store.hot_cache) == 3

    def test_warning_reduces_peer_infection_only_after_a_catch(self):
        handler = make_handler(
            whistleblower_fraction=0.25,
            whistleblower_audit_rate=1.0,
            whistleblower_warning_strength=0.5,
            contagion_exposure_alpha=1.0,
        )
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        (wb,) = handler.whistleblowers
        peers = [a for a in state.agents if a != wb and a != "poisoner_1"]
        fill_cache(handler, n_poisoned=4, n_clean=0)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        # Exposure drove peers to 1.0 (alpha 1, fully poisoned cache); the
        # warning then halves it in the same epoch.
        assert all(abs(handler.infection[a] - 0.5) < 1e-9 for a in peers)
        # No poison left to catch: no warning, infection follows exposure only.
        state.current_epoch = 2
        handler.on_epoch_start(state)
        assert handler.whistleblower_flags_last_epoch == 0
        assert all(handler.infection[a] == 0.0 for a in peers)

    def test_boycott_withholds_writes_only_after_a_catch(self):
        handler = make_handler(
            whistleblower_fraction=0.25,
            whistleblower_audit_rate=1.0,
            whistleblower_boycott_rate=1.0,
        )
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        (wb,) = handler.whistleblowers
        action = Action(
            agent_id=wb, action_type=ActionType.WRITE_MEMORY, content="x"
        )
        # Nothing caught yet: writes go through.
        assert handler.handle_action(action, state).success
        fill_cache(handler, n_poisoned=2, n_clean=0)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        result = handler.handle_action(action, state)
        assert not result.success
        assert result.metadata["memory_boycott"] is True
        assert handler.boycotted_write_count == 1
        # Non-whistleblowers are unaffected.
        other = Action(
            agent_id="honest_0" if wb != "honest_0" else "honest_1",
            action_type=ActionType.WRITE_MEMORY,
            content="y",
        )
        assert handler.handle_action(other, state).success

    def test_susceptible_infection_excludes_faction(self):
        handler = make_handler(whistleblower_fraction=0.5)
        state = make_wb_state(n_honest=4)
        handler.on_epoch_start(state)
        fill_cache(handler, n_poisoned=5, n_clean=0)
        state.current_epoch = 1
        handler.on_epoch_start(state)
        snap = handler.epoch_snapshots[-1]
        assert snap["susceptible_infection"] > snap["mean_infection"] > 0.0

    def test_loader_parses_whistleblower_fields(self):
        cfg = parse_memory_tier_config(
            {
                "enabled": True,
                "whistleblower_fraction": 0.24,
                "whistleblower_audit_rate": 0.5,
                "whistleblower_warning_strength": 0.5,
                "whistleblower_boycott_rate": 0.3,
            }
        )
        assert cfg.whistleblower_fraction == 0.24
        assert cfg.whistleblower_audit_rate == 0.5
        assert cfg.whistleblower_warning_strength == 0.5
        assert cfg.whistleblower_boycott_rate == 0.3
