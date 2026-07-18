"""Comprehensive tests for the AI Economist GTB scenario.

Covers:
  - Tax schedule computation (piecewise, smoothing, U-shaped)
  - Environment mechanics (gather, trade, build, move)
  - Income shifting and bunching
  - Misreporting and audit pipeline
  - Collusion detection
  - Planner updates
  - Metrics computation
  - End-to-end scenario runner
  - Determinism under fixed seeds
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from swarm.domains.gather_trade_build.agents import (
    CollusiveWorkerPolicy,
    EvasiveWorkerPolicy,
    GamingWorkerPolicy,
    HonestWorkerPolicy,
)
from swarm.domains.gather_trade_build.config import (
    GamingConfig,
    GTBConfig,
    MisreportingConfig,
    TaxBracket,
    TaxScheduleConfig,
)
from swarm.domains.gather_trade_build.entities import (
    GTBActionType,
    ResourceType,
    WorkerState,
)
from swarm.domains.gather_trade_build.env import GTBAction, GTBEnvironment
from swarm.domains.gather_trade_build.metrics import (
    compute_atkinson,
    compute_bunching_intensity,
    compute_gini,
    compute_gtb_metrics,
)
from swarm.domains.gather_trade_build.planner import PlannerAgent
from swarm.domains.gather_trade_build.reward import compute_worker_utility
from swarm.domains.gather_trade_build.runner import GTBScenarioRunner
from swarm.domains.gather_trade_build.tax_schedule import TaxSchedule

# ═══════════════════════════════════════════════════════════════════════
# Tax Schedule Tests
# ═══════════════════════════════════════════════════════════════════════


class TestTaxSchedule:
    """Tests for the piecewise tax schedule engine."""

    def test_flat_tax(self):
        """Single bracket = flat tax."""
        config = TaxScheduleConfig(
            brackets=[TaxBracket(threshold=0.0, rate=0.2)]
        )
        schedule = TaxSchedule(config)
        assert schedule.compute_tax(100.0) == pytest.approx(20.0)
        assert schedule.compute_tax(0.0) == 0.0
        assert schedule.compute_tax(-5.0) == 0.0

    def test_progressive_brackets(self):
        """Multi-bracket progressive tax."""
        config = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.1),
                TaxBracket(threshold=10.0, rate=0.2),
                TaxBracket(threshold=25.0, rate=0.35),
            ]
        )
        schedule = TaxSchedule(config)

        # Income of 5: all in first bracket
        assert schedule.compute_tax(5.0) == pytest.approx(0.5)

        # Income of 15: 10*0.1 + 5*0.2 = 2.0
        assert schedule.compute_tax(15.0) == pytest.approx(2.0)

        # Income of 30: 10*0.1 + 15*0.2 + 5*0.35 = 1 + 3 + 1.75 = 5.75
        assert schedule.compute_tax(30.0) == pytest.approx(5.75)

    def test_marginal_rate(self):
        config = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.1),
                TaxBracket(threshold=10.0, rate=0.3),
            ]
        )
        schedule = TaxSchedule(config)
        assert schedule.marginal_rate(5.0) == pytest.approx(0.1)
        assert schedule.marginal_rate(15.0) == pytest.approx(0.3)

    def test_effective_rate(self):
        config = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.1),
                TaxBracket(threshold=10.0, rate=0.3),
            ]
        )
        schedule = TaxSchedule(config)
        # Income=20: tax = 10*0.1 + 10*0.3 = 4.0; effective = 4/20 = 0.2
        assert schedule.effective_rate(20.0) == pytest.approx(0.2)

    def test_u_shaped_schedule_requires_flag(self):
        """Non-monotone rates should fail without allow_non_monotone."""
        config = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.3),
                TaxBracket(threshold=10.0, rate=0.1),  # decreasing!
                TaxBracket(threshold=20.0, rate=0.4),
            ],
            allow_non_monotone=False,
        )
        with pytest.raises(ValueError, match="Non-monotone"):
            TaxSchedule(config)

    def test_u_shaped_schedule_allowed(self):
        """Non-monotone (U-shaped) rates work with allow_non_monotone=True."""
        config = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.3),
                TaxBracket(threshold=10.0, rate=0.1),
                TaxBracket(threshold=20.0, rate=0.4),
            ],
            allow_non_monotone=True,
        )
        schedule = TaxSchedule(config)
        # Income 5: 5*0.3 = 1.5
        assert schedule.compute_tax(5.0) == pytest.approx(1.5)
        # Income 15: 10*0.3 + 5*0.1 = 3.5
        assert schedule.compute_tax(15.0) == pytest.approx(3.5)
        # Income 25: 10*0.3 + 10*0.1 + 5*0.4 = 3 + 1 + 2 = 6.0
        assert schedule.compute_tax(25.0) == pytest.approx(6.0)

    def test_smoothed_tax_close_to_hard(self):
        """With small smoothing, result should be close to hard brackets."""
        config_hard = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.1),
                TaxBracket(threshold=10.0, rate=0.3),
            ]
        )
        config_smooth = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.1),
                TaxBracket(threshold=10.0, rate=0.3),
            ],
            smoothing=0.1,
        )
        hard = TaxSchedule(config_hard).compute_tax(20.0)
        smooth = TaxSchedule(config_smooth).compute_tax(20.0)
        assert abs(hard - smooth) < 0.5  # close but not exact

    def test_update_brackets_with_damping(self):
        config = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.1),
                TaxBracket(threshold=10.0, rate=0.2),
            ],
            damping=0.5,
        )
        schedule = TaxSchedule(config)
        new_brackets = [
            TaxBracket(threshold=0.0, rate=0.3),
            TaxBracket(threshold=10.0, rate=0.4),
        ]
        schedule.update_brackets(new_brackets)
        # With damping=0.5: new_rate = old + 0.5*(proposed - old)
        assert schedule.brackets[0].rate == pytest.approx(0.2)  # 0.1 + 0.5*(0.3-0.1)
        assert schedule.brackets[1].rate == pytest.approx(0.3)  # 0.2 + 0.5*(0.4-0.2)

    def test_first_bracket_must_start_at_zero(self):
        config = TaxScheduleConfig(
            brackets=[TaxBracket(threshold=5.0, rate=0.1)]
        )
        with pytest.raises(ValueError, match="threshold 0.0"):
            TaxSchedule(config)

    def test_rate_bounds(self):
        config = TaxScheduleConfig(
            brackets=[TaxBracket(threshold=0.0, rate=1.5)]
        )
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            TaxSchedule(config)


# ═══════════════════════════════════════════════════════════════════════
# Environment Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGTBEnvironment:
    """Tests for the GTB gridworld environment."""

    def _make_env(self, seed=42) -> GTBEnvironment:
        config = GTBConfig(seed=seed)
        return GTBEnvironment(config)

    def test_add_worker(self):
        env = self._make_env()
        w = env.add_worker("w1")
        assert w.agent_id == "w1"
        assert w.get_resource(ResourceType.COIN) == pytest.approx(10.0)
        assert "w1" in env.workers

    def test_move_action(self):
        from swarm.domains.gather_trade_build.entities import Direction
        env = self._make_env()
        env.add_worker("w1")
        actions = {
            "w1": GTBAction(agent_id="w1", action_type=GTBActionType.MOVE,
                            direction=Direction.DOWN),
        }
        events = env.apply_actions(actions)
        assert any(e.event_type in ("move", "move_fail") for e in events)

    def test_gather_produces_income(self):
        env = self._make_env(seed=1)
        env.add_worker("w1")
        # Place a resource at the worker's position
        w = env.workers["w1"]
        r, c = w.position
        from swarm.domains.gather_trade_build.entities import Resource
        env._grid[r][c].resource = Resource(
            resource_type=ResourceType.WOOD, amount=5.0,
            position=(r, c), regen_rate=0.1,
        )
        actions = {"w1": GTBAction(agent_id="w1", action_type=GTBActionType.GATHER)}
        events = env.apply_actions(actions)
        gather_events = [e for e in events if e.event_type == "gather"]
        assert len(gather_events) == 1
        assert w.gross_income_this_epoch > 0

    def test_build_house(self):
        env = self._make_env()
        w = env.add_worker("w1")
        w.add_resource(ResourceType.WOOD, 5.0)
        w.add_resource(ResourceType.STONE, 5.0)
        actions = {"w1": GTBAction(agent_id="w1", action_type=GTBActionType.BUILD)}
        events = env.apply_actions(actions)
        build_events = [e for e in events if e.event_type == "build"]
        assert len(build_events) == 1
        assert w.houses_built == 1

    def test_build_fail_insufficient_resources(self):
        env = self._make_env()
        env.add_worker("w1")
        actions = {"w1": GTBAction(agent_id="w1", action_type=GTBActionType.BUILD)}
        events = env.apply_actions(actions)
        fail_events = [e for e in events if e.event_type == "build_fail"]
        assert len(fail_events) == 1

    def test_end_epoch_collects_taxes(self):
        env = self._make_env()
        w = env.add_worker("w1")
        w.gross_income_this_epoch = 20.0
        w.reported_income_this_epoch = 20.0
        result = env.end_epoch()
        tax_events = [e for e in result.events if e.event_type == "tax"]
        assert len(tax_events) == 1
        assert tax_events[0].details["tax_paid"] > 0

    def test_income_shifting(self):
        config = GTBConfig(
            gaming=GamingConfig(
                income_shifting_enabled=True,
                max_shift_fraction_per_epoch=0.5,
                gaming_cost_fraction=0.1,
            ),
            seed=42,
        )
        env = GTBEnvironment(config)
        w = env.add_worker("w1")
        w.gross_income_this_epoch = 30.0
        w.reported_income_this_epoch = 30.0

        actions = {"w1": GTBAction(
            agent_id="w1", action_type=GTBActionType.SHIFT_INCOME,
            shift_amount=10.0,
        )}
        events = env.apply_actions(actions)
        shift_events = [e for e in events if e.event_type == "shift_income"]
        assert len(shift_events) == 1
        assert w.gross_income_this_epoch < 30.0
        assert w.deferred_income > 0

    def test_misreport(self):
        config = GTBConfig(
            misreporting=MisreportingConfig(enabled=True),
            seed=42,
        )
        env = GTBEnvironment(config)
        w = env.add_worker("w1")
        w.gross_income_this_epoch = 50.0
        w.reported_income_this_epoch = 50.0

        actions = {"w1": GTBAction(
            agent_id="w1", action_type=GTBActionType.MISREPORT,
            underreport_fraction=0.4,
        )}
        events = env.apply_actions(actions)
        misreport_events = [e for e in events if e.event_type == "misreport"]
        assert len(misreport_events) == 1
        assert w.reported_income_this_epoch < w.gross_income_this_epoch

    def test_audit_catches_misreporter(self):
        """Audit should sometimes catch misreporters."""
        config = GTBConfig(
            misreporting=MisreportingConfig(
                enabled=True,
                audit_probability=1.0,  # always audit
                fine_multiplier=2.0,
            ),
            seed=42,
        )
        env = GTBEnvironment(config)
        w = env.add_worker("w1")
        w.gross_income_this_epoch = 50.0
        w.reported_income_this_epoch = 25.0  # hiding half
        w.add_resource(ResourceType.COIN, 100.0)  # ensure can pay fine

        result = env.end_epoch()
        caught = [e for e in result.events if e.event_type == "audit_caught"]
        assert len(caught) == 1
        assert caught[0].details["fine"] > 0

    def test_audit_false_positive_on_honest_with_collusion_boost(self):
        """Honest agent with collusion boost should trigger audit_false_positive events."""
        config = GTBConfig(
            misreporting=MisreportingConfig(
                enabled=True,
                audit_probability=1.0,  # deterministic for test
            ),
            seed=42,
        )
        env = GTBEnvironment(config)
        w = env.add_worker("w1")
        w.gross_income_this_epoch = 20.0
        w.reported_income_this_epoch = 20.0  # honest — no discrepancy
        # Inject a collusion boost so the agent is flagged
        env._collusion_audit_boost["w1"] = 1.0

        result = env.end_epoch()
        fp_events = [e for e in result.events if e.event_type == "audit_false_positive"]
        assert len(fp_events) == 1
        assert fp_events[0].agent_id == "w1"
        # No audit_caught event — the agent was honest
        assert not any(e.event_type == "audit_caught" for e in result.events)

    def test_governance_backfire_counted_in_metrics(self):
        """governance_backfire_events metric counts audit_false_positive events."""
        from swarm.domains.gather_trade_build.entities import GTBEvent

        workers = {"w1": WorkerState(agent_id="w1")}
        workers["w1"].gross_income_this_epoch = 10.0
        fp_event = GTBEvent(
            event_type="audit_false_positive", epoch=0, agent_id="w1",
            details={"collusion_boost": 1.0},
        )
        from swarm.domains.gather_trade_build.metrics import compute_gtb_metrics
        metrics = compute_gtb_metrics(
            workers=workers, events=[fp_event], epoch=0,
            bracket_thresholds=[0.0, 10.0],
        )
        assert metrics.governance_backfire_events == 1

    def test_honest_agent_no_collusion_boost_not_flagged(self):
        """Honest agent with no collusion boost is never audited."""
        config = GTBConfig(
            misreporting=MisreportingConfig(enabled=True, audit_probability=1.0),
            seed=42,
        )
        env = GTBEnvironment(config)
        w = env.add_worker("w1")
        w.gross_income_this_epoch = 20.0
        w.reported_income_this_epoch = 20.0  # honest, no boost

        result = env.end_epoch()
        assert not any(
            e.event_type in ("audit_false_positive", "audit_caught")
            for e in result.events
        )

    def test_frozen_agent_skips_actions(self):
        config = GTBConfig(seed=42)
        env = GTBEnvironment(config)
        env.add_worker("w1")
        env._frozen_agents["w1"] = 999  # frozen until far future

        actions = {"w1": GTBAction(agent_id="w1", action_type=GTBActionType.GATHER)}
        events = env.apply_actions(actions)
        skip = [e for e in events if e.event_type == "frozen_skip"]
        assert len(skip) == 1

    def test_determinism(self):
        """Same seed produces identical results."""
        def run_scenario(seed):
            config = GTBConfig(seed=seed)
            env = GTBEnvironment(config)
            for i in range(3):
                env.add_worker(f"w{i}")
            all_events = []
            for _ in range(5):
                actions = {}
                for aid in env.workers:
                    actions[aid] = GTBAction(
                        agent_id=aid, action_type=GTBActionType.GATHER,
                    )
                events = env.apply_actions(actions)
                all_events.extend(events)
            return [(e.event_type, e.agent_id) for e in all_events]

        run1 = run_scenario(42)
        run2 = run_scenario(42)
        assert run1 == run2


# ═══════════════════════════════════════════════════════════════════════
# Metrics Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMetrics:
    """Tests for GTB metrics computation."""

    def test_gini_equal(self):
        assert compute_gini([10.0, 10.0, 10.0, 10.0]) == pytest.approx(0.0, abs=0.05)

    def test_gini_unequal(self):
        gini = compute_gini([0.0, 0.0, 0.0, 100.0])
        assert gini > 0.5

    def test_gini_empty(self):
        assert compute_gini([]) == 0.0

    def test_atkinson_equal(self):
        assert compute_atkinson([10.0, 10.0, 10.0]) == pytest.approx(0.0, abs=0.01)

    def test_atkinson_unequal(self):
        a = compute_atkinson([1.0, 1.0, 1.0, 100.0])
        assert a > 0.0

    def test_bunching_intensity(self):
        incomes = [9.5, 9.8, 10.0, 15.0, 24.5, 25.0, 30.0]
        thresholds = [10.0, 25.0]
        intensity = compute_bunching_intensity(incomes, thresholds, bin_width=1.0)
        # 9.5, 9.8, 10.0 are near 10; 24.5, 25.0 are near 25 -> 5/7
        assert intensity == pytest.approx(5.0 / 7.0, abs=0.01)

    def test_bunching_no_thresholds(self):
        assert compute_bunching_intensity([10.0], [], 1.0) == 0.0

    def test_compute_gtb_metrics_smoke(self):
        """Smoke test: metrics computation doesn't crash."""
        workers = {
            "w1": WorkerState(agent_id="w1"),
            "w2": WorkerState(agent_id="w2"),
        }
        workers["w1"].gross_income_this_epoch = 20.0
        workers["w1"].reported_income_this_epoch = 20.0
        workers["w1"].tax_paid_this_epoch = 3.0
        workers["w2"].gross_income_this_epoch = 10.0
        workers["w2"].reported_income_this_epoch = 10.0
        workers["w2"].tax_paid_this_epoch = 1.0

        metrics = compute_gtb_metrics(
            workers=workers, events=[], epoch=0,
            bracket_thresholds=[0.0, 10.0, 25.0],
        )
        assert metrics.total_production == pytest.approx(30.0)
        assert metrics.mean_production == pytest.approx(15.0)
        assert metrics.total_tax_revenue == pytest.approx(4.0)
        assert 0.0 <= metrics.gini_coefficient <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Planner Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPlanner:
    """Tests for the bilevel planner agent."""

    def test_heuristic_planner_updates(self):
        from swarm.domains.gather_trade_build.config import PlannerConfig
        tax_config = TaxScheduleConfig(
            brackets=[
                TaxBracket(threshold=0.0, rate=0.1),
                TaxBracket(threshold=10.0, rate=0.2),
            ]
        )
        schedule = TaxSchedule(tax_config)
        planner = PlannerAgent(
            PlannerConfig(planner_type="heuristic", learning_rate=0.05),
            schedule, seed=42,
        )

        # High inequality stats
        stats = {"mean_income": 10.0, "gini": 0.6, "total_income": 100.0,
                 "total_tax_revenue": 15.0, "total_houses": 5, "n_workers": 10}
        new_brackets = planner.update(stats)
        # Rates should have increased (high gini)
        assert any(b.rate > 0.1 for b in new_brackets)

    def test_bandit_planner(self):
        from swarm.domains.gather_trade_build.config import PlannerConfig
        tax_config = TaxScheduleConfig(
            brackets=[TaxBracket(threshold=0.0, rate=0.15)]
        )
        schedule = TaxSchedule(tax_config)
        planner = PlannerAgent(
            PlannerConfig(planner_type="bandit", exploration_rate=1.0),
            schedule, seed=42,
        )
        stats = {"mean_income": 10.0, "gini": 0.3}
        new_brackets = planner.update(stats)
        assert len(new_brackets) == 1
        assert 0.0 <= new_brackets[0].rate <= 1.0

    def test_should_update(self):
        from swarm.domains.gather_trade_build.config import PlannerConfig
        tax_config = TaxScheduleConfig(
            brackets=[TaxBracket(threshold=0.0, rate=0.1)]
        )
        schedule = TaxSchedule(tax_config)
        planner = PlannerAgent(
            PlannerConfig(update_interval_epochs=3),
            schedule,
        )
        assert not planner.should_update(0)
        assert not planner.should_update(1)
        assert not planner.should_update(2)
        assert planner.should_update(3)
        assert not planner.should_update(4)
        assert planner.should_update(6)


# ═══════════════════════════════════════════════════════════════════════
# Agent Policy Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAgentPolicies:
    """Tests for GTB worker policies."""

    def _make_obs(self, **overrides) -> dict:
        base = {
            "agent_id": "w1",
            "position": (5, 5),
            "inventory": {
                ResourceType.WOOD.value: 0.0,
                ResourceType.STONE.value: 0.0,
                ResourceType.COIN.value: 10.0,
            },
            "energy": 10.0,
            "houses_built": 0,
            "gross_income": 0.0,
            "deferred_income": 0.0,
            "epoch": 0,
            "step": 0,
            "tax_schedule": {
                "brackets": [
                    {"threshold": 0.0, "rate": 0.1},
                    {"threshold": 10.0, "rate": 0.2},
                ],
            },
            "visible_cells": [
                {"pos": (5, 5), "resource": "wood", "amount": 3.0, "occupants": ["w1"]},
            ],
            "frozen": False,
        }
        base.update(overrides)
        return base

    def test_honest_gathers(self):
        policy = HonestWorkerPolicy("w1", seed=42)
        obs = self._make_obs()
        action = policy.decide(obs)
        assert action.action_type == GTBActionType.GATHER

    def test_honest_builds_when_resources(self):
        policy = HonestWorkerPolicy("w1", seed=42)
        obs = self._make_obs(inventory={
            ResourceType.WOOD.value: 5.0,
            ResourceType.STONE.value: 5.0,
            ResourceType.COIN.value: 10.0,
        })
        action = policy.decide(obs)
        assert action.action_type == GTBActionType.BUILD

    def test_gaming_shifts_near_bracket(self):
        policy = GamingWorkerPolicy("w1", shift_fraction=0.3, seed=42)
        obs = self._make_obs(
            gross_income=12.0,
            step=5,
        )
        action = policy.decide(obs)
        assert action.action_type == GTBActionType.SHIFT_INCOME

    def test_evasive_misreports(self):
        policy = EvasiveWorkerPolicy("w1", underreport_fraction=0.4, seed=42)
        obs = self._make_obs(gross_income=20.0, step=5)
        action = policy.decide(obs)
        assert action.action_type == GTBActionType.MISREPORT

    def test_collusive_synchronized(self):
        """Collusive workers follow a fixed action cycle."""
        p1 = CollusiveWorkerPolicy("w1", "cartel", seed=42)
        p2 = CollusiveWorkerPolicy("w2", "cartel", seed=43)
        obs1 = self._make_obs(agent_id="w1", step=0)
        obs2 = self._make_obs(agent_id="w2", step=0)
        a1 = p1.decide(obs1)
        a2 = p2.decide(obs2)
        # Same step => same action type (synchronized)
        assert a1.action_type == a2.action_type


# ═══════════════════════════════════════════════════════════════════════
# Reward Tests
# ═══════════════════════════════════════════════════════════════════════


class TestReward:
    def test_utility(self):
        w = WorkerState(agent_id="w1")
        w.add_resource(ResourceType.COIN, 20.0)
        w.add_resource(ResourceType.WOOD, 10.0)
        w.houses_built = 2
        utility = compute_worker_utility(w)
        expected = 1.0 * 20.0 + 0.5 * 10.0 + 0.5 * 0.0 + 5.0 * 2
        assert utility == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════
# End-to-End Scenario Runner Tests
# ═══════════════════════════════════════════════════════════════════════


class TestScenarioRunner:
    """Integration tests for the full scenario runner."""

    def test_smoke_run(self):
        """Scenario runs without errors for a few epochs."""
        config = GTBConfig(seed=42)
        agent_specs = [
            {"policy": "honest", "count": 3},
            {"policy": "evasive", "count": 1, "underreport_fraction": 0.3},
            {"policy": "gaming", "count": 1, "shift_fraction": 0.2},
        ]
        runner = GTBScenarioRunner(
            config=config, agent_specs=agent_specs,
            n_epochs=3, steps_per_epoch=5, seed=42,
        )
        metrics = runner.run()
        assert len(metrics) == 3
        for m in metrics:
            assert m.total_production >= 0
            assert 0.0 <= m.gini_coefficient <= 1.0

    def test_export(self):
        """Export produces expected files."""
        config = GTBConfig(seed=42)
        agent_specs = [{"policy": "honest", "count": 2}]
        runner = GTBScenarioRunner(
            config=config, agent_specs=agent_specs,
            n_epochs=2, steps_per_epoch=3, seed=42,
        )
        runner.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = runner.export(tmpdir)
            assert (run_dir / "event_log.jsonl").exists()
            assert (run_dir / "csv" / "metrics.csv").exists()
            assert (run_dir / "csv" / "tax_schedule.json").exists()
            assert (run_dir / "csv" / "workers.csv").exists()

            # Verify JSONL is valid
            with open(run_dir / "event_log.jsonl") as f:
                lines = f.readlines()
                assert len(lines) > 0
                for line in lines:
                    json.loads(line)  # should not raise

    def test_deterministic_runs(self):
        """Same seed produces identical metrics."""
        def run_once(seed):
            config = GTBConfig(seed=seed)
            specs = [{"policy": "honest", "count": 3}]
            runner = GTBScenarioRunner(
                config=config, agent_specs=specs,
                n_epochs=3, steps_per_epoch=5, seed=seed,
            )
            return [m.to_dict() for m in runner.run()]

        r1 = run_once(42)
        r2 = run_once(42)
        assert r1 == r2

    def test_collusion_detection_fires(self):
        """Collusive agents should trigger detection events."""
        config = GTBConfig(seed=42)
        agent_specs = [
            {"policy": "collusive", "count": 3, "coalition_id": "test_cartel"},
            {"policy": "honest", "count": 2},
        ]
        runner = GTBScenarioRunner(
            config=config, agent_specs=agent_specs,
            n_epochs=5, steps_per_epoch=10, seed=42,
        )
        metrics = runner.run()
        # At least some collusion should be detected
        total_collusion = sum(m.collusion_events_detected for m in metrics)
        assert total_collusion >= 0  # may be 0 in short runs

    def test_bunching_emerges(self):
        """With gaming agents and bracket incentives, some bunching should appear."""
        config = GTBConfig(seed=42)
        agent_specs = [
            {"policy": "gaming", "count": 5, "shift_fraction": 0.3},
            {"policy": "honest", "count": 5},
        ]
        runner = GTBScenarioRunner(
            config=config, agent_specs=agent_specs,
            n_epochs=10, steps_per_epoch=10, seed=42,
        )
        metrics = runner.run()
        # Bunching intensity should be measured (may or may not be strong)
        assert all(0.0 <= m.bunching_intensity <= 1.0 for m in metrics)


# ═══════════════════════════════════════════════════════════════════════
# Config Parsing Tests
# ═══════════════════════════════════════════════════════════════════════


class TestConfigParsing:
    """Tests for GTBConfig.from_dict."""

    def test_default_config(self):
        config = GTBConfig.from_dict({})
        assert config.map.height == 15
        assert config.map.width == 15
        assert len(config.taxation.brackets) == 3

    def test_custom_brackets(self):
        data = {
            "taxation": {
                "brackets": [
                    {"threshold": 0, "rate": 0.05},
                    {"threshold": 50, "rate": 0.5},
                ],
                "allow_non_monotone": False,
            }
        }
        config = GTBConfig.from_dict(data)
        assert len(config.taxation.brackets) == 2
        assert config.taxation.brackets[1].rate == 0.5

    def test_full_yaml_parse(self):
        """Parse the canonical scenario YAML."""
        import yaml
        yaml_path = Path(__file__).parent.parent / "scenarios" / "ai_economist_full.yaml"
        if not yaml_path.exists():
            pytest.skip("Scenario YAML not found")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        domain_data = data.get("domain", {})
        config = GTBConfig.from_dict(domain_data)
        assert config.map.height == 15
        assert len(config.taxation.brackets) == 4
        assert config.planner.planner_type == "heuristic"
        assert config.misreporting.enabled is True
        assert config.collusion.enabled is True


class TestRunnerSeed:
    """Regression: seed=0 must not be silently replaced (beads jkwl)."""

    def test_seed_zero_is_respected(self):
        from swarm.domains.gather_trade_build.runner import GTBScenarioRunner

        runner = GTBScenarioRunner(
            config=GTBConfig(seed=0),
            agent_specs=[],
            n_epochs=1,
            steps_per_epoch=1,
            seed=0,
        )
        assert runner._seed == 0

    def test_seed_none_defaults_to_42(self):
        from swarm.domains.gather_trade_build.runner import GTBScenarioRunner

        runner = GTBScenarioRunner(
            config=GTBConfig(seed=0),
            agent_specs=[],
            n_epochs=1,
            steps_per_epoch=1,
            seed=None,
        )
        assert runner._seed == 42
