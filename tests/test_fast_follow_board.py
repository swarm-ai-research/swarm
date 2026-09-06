"""Tests for the fast-follow round structure on the wiki board (bead pi02.5).

The wiki-board model (bead pi02) gives the side channel a constant
``side_deadline_pressure``. The fast-follow-question-bench evidence
(JoshuaDavid/WikiAgentSwarmInvestigation, tasks/fast-follow-question-bench,
finding 04) says the pressure is not constant: R1 carries a minutes-long
deadline that pays for a fresh lookup outright, and every follow-up carries a
seconds-long one that cannot. ``side_round_structure: fast_follow`` derives the
pressure from the round instead of averaging it away; ``uniform`` is the old
behaviour and every pre-existing scenario keeps it.
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
        "side_discovery_rate": 1.0,
        "side_write_preference": 1.0,
        "side_routing": "overlap",
        "side_task_overlap": 1.0,
        "side_value_prior": 1.0,
    }
    kwargs.update(overrides)
    return MemoryHandler(MemoryTierConfig(**kwargs), event_bus=EventBus())


def make_state(n: int = 6) -> EnvState:
    state = EnvState()
    for i in range(n):
        state.add_agent(f"honest_{i}", agent_type=AgentType.HONEST)
    return state


def run_epochs(handler, state, n_epochs, steps=5, start=0):
    for ep in range(start, start + n_epochs):
        state.current_epoch = ep
        handler.on_epoch_start(state)
        for step in range(steps):
            state.current_step = step
            for agent_id in list(state.agents):
                handler.handle_action(
                    Action(
                        agent_id=agent_id,
                        action_type=ActionType.WRITE_MEMORY,
                        content="x",
                    ),
                    state,
                )


class TestConfigValidation:
    def test_bad_round_structure_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_round_structure="staggered")

    def test_zero_rounds_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_rounds_per_episode=0)

    def test_negative_deadline_rejected(self):
        with pytest.raises(ValueError):
            MemoryTierConfig(side_followup_deadline=-1.0)

    def test_defaults_are_uniform(self):
        assert MemoryTierConfig().side_round_structure == "uniform"


class TestRoundPressure:
    def test_uniform_keeps_the_constant(self):
        handler = make_handler(side_deadline_pressure=0.4)
        state = make_state()
        for step in range(5):
            state.current_step = step
            assert handler._round_pressure(state) == pytest.approx(0.4)

    def test_r1_is_affordable_and_follow_ups_are_not(self):
        handler = make_handler(
            side_round_structure="fast_follow",
            side_initial_deadline=180.0,
            side_followup_deadline=12.0,
            side_research_cost=45.0,
        )
        state = make_state()
        state.current_step = 0
        assert handler._round_pressure(state) == 0.0  # 180s buys the lookup
        for step in (1, 2, 3, 4):
            state.current_step = step
            # 1 - 12/45
            assert handler._round_pressure(state) == pytest.approx(0.7333, abs=1e-4)

    def test_rounds_cycle_with_the_step(self):
        handler = make_handler(
            side_round_structure="fast_follow", side_rounds_per_episode=5
        )
        state = make_state()
        for step, expected in [(0, 0), (4, 4), (5, 0), (9, 4), (10, 0)]:
            state.current_step = step
            assert handler._round_index(state) == expected

    def test_pressure_is_scaled_by_the_constant(self):
        handler = make_handler(
            side_round_structure="fast_follow", side_deadline_pressure=0.5
        )
        state = make_state()
        state.current_step = 1
        assert handler._round_pressure(state) == pytest.approx(0.5 * (1 - 12 / 45))

    def test_a_deadline_that_covers_the_lookup_kills_the_board(self):
        handler = make_handler(
            side_round_structure="fast_follow",
            side_followup_deadline=90.0,
            side_research_cost=45.0,
        )
        state = make_state()
        state.current_step = 3
        assert handler._round_pressure(state) == 0.0

    def test_free_research_kills_the_board(self):
        handler = make_handler(
            side_round_structure="fast_follow", side_research_cost=0.0
        )
        state = make_state()
        state.current_step = 3
        assert handler._round_pressure(state) == 0.0


class TestTrafficConcentrates:
    def _followup_fraction(self, **overrides) -> float:
        handler = make_handler(**overrides)
        state = make_state(10)
        run_epochs(handler, state, 20, steps=5)
        late = [
            s["followup_side_write_fraction"] for s in handler.epoch_snapshots[-5:]
        ]
        return sum(late) / len(late)

    def test_uniform_never_counts_follow_ups(self):
        assert self._followup_fraction() == 0.0

    def test_fast_follow_puts_all_board_traffic_in_the_follow_ups(self):
        # R1 pressure is 0, so no write in round 0 can reach the board:
        # every board write in the episode is a follow-up write.
        assert self._followup_fraction(side_round_structure="fast_follow") == 1.0

    def test_fast_follow_writes_less_overall_than_uniform(self):
        """The board is used on 4 rounds of 5, not 5, and at 0.73 not 1.0."""

        def ungoverned(**overrides) -> float:
            handler = make_handler(**overrides)
            state = make_state(10)
            run_epochs(handler, state, 20, steps=5)
            late = [s["ungoverned_fraction"] for s in handler.epoch_snapshots[-5:]]
            return sum(late) / len(late)

        assert ungoverned(side_round_structure="fast_follow") < ungoverned()


class TestLoader:
    def test_yaml_round_structure_round_trips(self):
        config = parse_memory_tier_config(
            {
                "enabled": True,
                "side_channel_enabled": True,
                "side_round_structure": "fast_follow",
                "side_rounds_per_episode": 6,
                "side_initial_deadline": 738.0,
                "side_followup_deadline": 56.0,
                "side_research_cost": 60.0,
            }
        )
        assert config.side_round_structure == "fast_follow"
        assert config.side_rounds_per_episode == 6
        assert config.side_initial_deadline == 738.0
        assert config.side_followup_deadline == 56.0
        assert config.side_research_cost == 60.0

    def test_yaml_defaults_to_uniform(self):
        config = parse_memory_tier_config({"enabled": True})
        assert config.side_round_structure == "uniform"
        assert config.side_rounds_per_episode == 5


class TestSnapshot:
    def test_snapshot_reports_the_structure(self):
        handler = make_handler(side_round_structure="fast_follow")
        state = make_state()
        run_epochs(handler, state, 2)
        last = handler.epoch_snapshots[-1]
        assert last["side_round_structure"] == "fast_follow"
        assert "followup_side_write_fraction" in last
        assert 0.0 <= last["followup_side_write_fraction"] <= 1.0
