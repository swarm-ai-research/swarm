"""Tests for the DeLM-style parallel hill-climber.

The DeLM optimizer is a shared-state, self-coordinating hill-climb over the
same governance/payoff landscape as the GEPA and darwinian optimizers. These
tests cover the coordination primitives (shared context, verified admission,
constraints, task queue) as fast unit tests, plus a small end-to-end run on the
baseline scenario that asserts the contract: reproducible, never regresses
below the seed, and respects the invariants.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from swarm.analysis.delm_hillclimb import (
    EvalOutcome,
    HillClimbConfig,
    Objective,
    SharedContext,
    Task,
    TaskQueue,
    _mutate,
    _param_distance,
    _quantize_key,
    _random_point,
    build_adaptive_policy_objective,
    build_governance_objective,
    evaluate_params,
    plot_trajectory,
    run_delm_hillclimb,
    seed_params_from_scenario,
)
from swarm.analysis.evolver import INT_PARAMS, PARAM_RANGES
from swarm.scenarios import load_scenario

SCENARIO = Path("scenarios/baseline.yaml")


# =========================================================================
# Parameter-space helpers
# =========================================================================


class TestParamHelpers:
    def test_seed_params_in_range_and_complete(self):
        scenario = load_scenario(SCENARIO)
        params = seed_params_from_scenario(scenario)
        assert set(params) == set(PARAM_RANGES)
        for name, (lo, hi) in PARAM_RANGES.items():
            assert lo <= params[name] <= hi
            if name in INT_PARAMS:
                assert isinstance(params[name], int)

    def test_mutation_stays_in_bounds(self):
        rng = random.Random(0)
        base = {n: (lo + hi) / 2 for n, (lo, hi) in PARAM_RANGES.items()}
        for _ in range(200):
            cand = _mutate(base, rng, step_frac=0.5)
            for name, (lo, hi) in PARAM_RANGES.items():
                assert lo <= cand[name] <= hi

    def test_mutate_single_dim_changes_only_that_dim(self):
        rng = random.Random(1)
        base = {n: (lo + hi) / 2 for n, (lo, hi) in PARAM_RANGES.items()}
        dim = "governance.audit_probability"
        cand = _mutate(base, rng, step_frac=0.3, dims=[dim])
        for name in PARAM_RANGES:
            if name == dim:
                continue
            assert cand[name] == base[name]

    def test_int_params_round(self):
        rng = random.Random(2)
        base = {n: (lo + hi) / 2 for n, (lo, hi) in PARAM_RANGES.items()}
        cand = _mutate(base, rng, step_frac=0.5)
        for name in INT_PARAMS:
            assert isinstance(cand[name], int)

    def test_random_point_in_range(self):
        rng = random.Random(3)
        for _ in range(50):
            p = _random_point(rng)
            for name, (lo, hi) in PARAM_RANGES.items():
                assert lo <= p[name] <= hi

    def test_param_distance_zero_for_identical(self):
        base = {n: (lo + hi) / 2 for n, (lo, hi) in PARAM_RANGES.items()}
        assert _param_distance(base, base) == pytest.approx(0.0)

    def test_param_distance_positive_for_different(self):
        a = {n: lo for n, (lo, hi) in PARAM_RANGES.items()}
        b = {n: hi for n, (lo, hi) in PARAM_RANGES.items()}
        assert _param_distance(a, b) > 0.0


# =========================================================================
# Shared context: verified admission, constraints, basins
# =========================================================================


def _outcome(fitness: float, viable: bool = True) -> EvalOutcome:
    return EvalOutcome(fitness=fitness, side_info={}, viable=viable)


class TestSharedContext:
    def _ctx(self, score: float = 0.5) -> SharedContext:
        params = {n: (lo + hi) / 2 for n, (lo, hi) in PARAM_RANGES.items()}
        return SharedContext(best_params=params, best_score=score)

    def test_verified_improvement_promotes_best(self):
        ctx = self._ctx(0.5)
        base, _ = ctx.snapshot_best()
        cand = dict(base)
        cand["governance.audit_probability"] = 0.9
        # Verifier confirms the improvement.
        gist = ctx.record(
            params=cand,
            outcome=_outcome(0.7),
            parent_score=0.5,
            base_params=base,
            worker=0,
            kind="explore",
            verifier=lambda p: _outcome(0.72),
        )
        assert gist.verified
        _, best = ctx.snapshot_best()
        # Promoted at the conservative (min of measured and verified) score.
        assert best == pytest.approx(0.7)
        assert ctx.n_improvements == 1

    def test_unverified_improvement_is_rejected(self):
        ctx = self._ctx(0.5)
        base, _ = ctx.snapshot_best()
        cand = dict(base)
        cand["governance.audit_probability"] = 0.9
        gist = ctx.record(
            params=cand,
            outcome=_outcome(0.7),
            parent_score=0.5,
            base_params=base,
            worker=0,
            kind="explore",
            verifier=lambda p: _outcome(0.4),  # verification disagrees
        )
        assert not gist.verified
        _, best = ctx.snapshot_best()
        assert best == pytest.approx(0.5)  # best unchanged
        assert ctx.n_verify_rejected == 1

    def test_regression_becomes_binding_constraint(self):
        ctx = self._ctx(0.8)
        base, _ = ctx.snapshot_best()
        cand = dict(base)
        cand["payoff.h"] = 9.0
        ctx.record(
            params=cand,
            outcome=_outcome(0.2),  # far below best 0.8
            parent_score=0.8,
            base_params=base,
            worker=0,
            kind="mutate_dim",
            verifier=lambda p: _outcome(0.2),
        )
        assert len(ctx.constraints) == 1
        assert ctx.is_pruned(cand)

    def test_non_viable_run_becomes_constraint(self):
        ctx = self._ctx(0.5)
        base, _ = ctx.snapshot_best()
        cand = dict(base)
        cand["payoff.s_plus"] = 9.5
        ctx.record(
            params=cand,
            outcome=_outcome(0.0, viable=False),
            parent_score=0.5,
            base_params=base,
            worker=0,
            kind="explore",
            verifier=lambda p: _outcome(0.0, viable=False),
        )
        assert len(ctx.constraints) == 1
        assert not ctx.constraints[0].viable
        assert ctx.is_pruned(cand)

    def test_seen_set_marks_evaluated_cells(self):
        ctx = self._ctx(0.5)
        base, _ = ctx.snapshot_best()
        cand = dict(base)
        cand["governance.audit_probability"] = 0.31
        assert not ctx.is_seen(cand)
        ctx.record(
            params=cand,
            outcome=_outcome(0.55),
            parent_score=0.5,
            base_params=base,
            worker=0,
            kind="explore",
            verifier=lambda p: _outcome(0.55),
        )
        assert ctx.is_seen(cand)

    def test_verified_distant_point_adds_basin(self):
        ctx = self._ctx(0.5)
        base, _ = ctx.snapshot_best()
        # A point at the opposite corner of the space is far from the seed basin.
        cand = {n: hi for n, (lo, hi) in PARAM_RANGES.items()}
        n_basins_before = len(ctx.basins)
        ctx.record(
            params=cand,
            outcome=_outcome(0.9),
            parent_score=0.5,
            base_params=base,
            worker=0,
            kind="restart",
            verifier=lambda p: _outcome(0.9),
            basin_distance=0.1,
        )
        assert len(ctx.basins) == n_basins_before + 1


# =========================================================================
# Task queue
# =========================================================================


class TestTaskQueue:
    def test_fifo_claim(self):
        q = TaskQueue()
        q.push(Task(kind="explore"))
        q.push(Task(kind="restart"))
        assert q.claim().kind == "explore"
        assert q.claim().kind == "restart"
        assert q.claim() is None

    def test_push_many_and_len(self):
        q = TaskQueue()
        q.push_many([Task(kind="explore"), Task(kind="mutate_dim", dim="payoff.h")])
        assert len(q) == 2


# =========================================================================
# Quantization
# =========================================================================


class TestQuantize:
    def test_close_values_share_cell(self):
        base = {n: (lo + hi) / 2 for n, (lo, hi) in PARAM_RANGES.items()}
        a = dict(base)
        b = dict(base)
        b["payoff.h"] = base["payoff.h"] + 1e-6  # below quantization grid
        assert _quantize_key(a) == _quantize_key(b)

    def test_distinct_values_distinct_cells(self):
        base = {n: (lo + hi) / 2 for n, (lo, hi) in PARAM_RANGES.items()}
        a = dict(base)
        b = dict(base)
        b["payoff.h"] = base["payoff.h"] + 0.5
        assert _quantize_key(a) != _quantize_key(b)


# =========================================================================
# Evaluation pipeline
# =========================================================================


class TestEvaluate:
    def test_evaluate_returns_fitness_in_unit_interval(self):
        scenario = load_scenario(SCENARIO)
        params = seed_params_from_scenario(scenario)
        out = evaluate_params(
            scenario, params, seed=42, eval_epochs=1, eval_steps=3
        )
        assert out.viable
        assert 0.0 <= out.fitness <= 1.0
        assert "toxicity" in out.side_info

    def test_evaluate_is_deterministic_for_fixed_seed(self):
        scenario = load_scenario(SCENARIO)
        params = seed_params_from_scenario(scenario)
        a = evaluate_params(scenario, params, seed=11, eval_epochs=1, eval_steps=3)
        b = evaluate_params(scenario, params, seed=11, eval_epochs=1, eval_steps=3)
        assert a.fitness == pytest.approx(b.fitness)


# =========================================================================
# End-to-end run contract
# =========================================================================


class TestRunContract:
    def test_run_is_reproducible(self):
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(max_evals=14, n_workers=3, eval_epochs=1, eval_steps=3, seed=5)
        r1 = run_delm_hillclimb(scenario, cfg)
        r2 = run_delm_hillclimb(scenario, cfg)
        assert r1.best_score == r2.best_score
        assert r1.best_params == r2.best_params
        assert r1.n_evals == r2.n_evals

    def test_best_never_below_seed(self):
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(max_evals=20, n_workers=4, eval_epochs=1, eval_steps=3, seed=9)
        r = run_delm_hillclimb(scenario, cfg)
        # Hill-climbing is monotone in the verified best: it can never end below
        # where it started.
        assert r.best_score >= r.seed_score - 1e-9

    def test_best_params_in_range(self):
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(max_evals=16, n_workers=2, eval_epochs=1, eval_steps=3, seed=3)
        r = run_delm_hillclimb(scenario, cfg)
        for name, (lo, hi) in PARAM_RANGES.items():
            assert lo <= r.best_params[name] <= hi

    def test_history_monotone_nondecreasing(self):
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(max_evals=24, n_workers=4, eval_epochs=1, eval_steps=3, seed=7)
        r = run_delm_hillclimb(scenario, cfg)
        scores = [h["best_score"] for h in r.history]
        assert scores == sorted(scores)

    def test_no_verify_mode_runs(self):
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(
            max_evals=12, n_workers=2, eval_epochs=1, eval_steps=3, seed=1, verify=False
        )
        r = run_delm_hillclimb(scenario, cfg)
        assert r.n_evals >= 1
        assert r.best_score >= r.seed_score - 1e-9
        # --no-verify spends no verification evaluations.
        assert r.telemetry["verify_evals"] == 0

    def test_no_verify_mode_still_hill_climbs(self):
        # Regression: --no-verify must still promote improvements (it used to
        # return the seed best because promotion was gated on verifier presence).
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(
            max_evals=40, n_workers=4, eval_epochs=2, eval_steps=4, seed=7, verify=False
        )
        r = run_delm_hillclimb(scenario, cfg)
        assert r.telemetry["improvements"] >= 1
        assert r.best_score > r.seed_score

    def test_eval_budget_counts_verification(self):
        # Contract: max_evals bounds *primary* candidate evaluations; n_evals is
        # the honest total = primaries + verification re-runs.
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(
            max_evals=40, n_workers=4, eval_epochs=2, eval_steps=4, seed=7, verify=True
        )
        r = run_delm_hillclimb(scenario, cfg)
        # Verified improvements happen on this scenario, so verify_evals > 0.
        assert r.telemetry["verify_evals"] >= 1
        # max_evals bounds the primary evaluations.
        assert r.telemetry["primary_evals"] <= cfg.max_evals
        # Reported total honestly includes the verification re-runs.
        assert r.n_evals == r.telemetry["primary_evals"] + r.telemetry["verify_evals"]

    def test_telemetry_present(self):
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(max_evals=12, n_workers=3, eval_epochs=1, eval_steps=3, seed=2)
        r = run_delm_hillclimb(scenario, cfg)
        for key in (
            "improvements",
            "verify_rejected",
            "redundant_skipped",
            "constraint_pruned",
            "basins",
        ):
            assert key in r.telemetry

    def test_threaded_mode_runs_and_holds_contract(self):
        # Real-thread scheduler is not bit-reproducible, but must still honor
        # the hill-climb contract: never end below the seed, params in range.
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(
            max_evals=12,
            n_workers=3,
            eval_epochs=1,
            eval_steps=3,
            seed=4,
            use_threads=True,
        )
        r = run_delm_hillclimb(scenario, cfg)
        assert r.best_score >= r.seed_score - 1e-9
        for name, (lo, hi) in PARAM_RANGES.items():
            assert lo <= r.best_params[name] <= hi

    def test_p_invariants_preserved_in_side_info(self):
        # Toxicity is E[1-p | accepted] and must stay a valid probability.
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(max_evals=14, n_workers=3, eval_epochs=1, eval_steps=3, seed=6)
        r = run_delm_hillclimb(scenario, cfg)
        for gist in r.gists:
            tox = gist["side_info"].get("toxicity")
            if tox is not None:
                assert 0.0 <= tox <= 1.0

    def test_requires_scenario_or_objective(self):
        with pytest.raises(ValueError, match="base_scenario or objective"):
            run_delm_hillclimb()


# =========================================================================
# Objective abstraction
# =========================================================================


class TestObjectives:
    def test_governance_objective_matches_default_path(self):
        # Passing the explicit governance objective is equivalent to the
        # backward-compatible default (scenario-only) call.
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(max_evals=14, n_workers=3, eval_epochs=1, eval_steps=3, seed=5)
        r_default = run_delm_hillclimb(scenario, cfg)
        obj = build_governance_objective(scenario, cfg)
        r_obj = run_delm_hillclimb(config=cfg, objective=obj)
        assert r_default.best_score == r_obj.best_score
        assert r_default.best_params == r_obj.best_params

    def test_governance_objective_param_space(self):
        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig()
        obj = build_governance_objective(scenario, cfg)
        assert obj.param_ranges is PARAM_RANGES
        assert obj.int_params is INT_PARAMS
        assert set(obj.seed_params) == set(PARAM_RANGES)

    def test_adaptive_objective_param_space(self):
        from swarm.adaptive.policy import PARAM_NAMES

        obj = build_adaptive_policy_objective(n_interactions=50)
        assert obj.name == "adaptive_policy"
        assert set(obj.param_ranges) == set(PARAM_NAMES)
        assert obj.int_params == frozenset()
        assert set(obj.seed_params) == set(PARAM_NAMES)

    def test_adaptive_objective_evaluate_is_viable_and_deterministic(self):
        obj = build_adaptive_policy_objective(n_interactions=80)
        a = obj.evaluate(obj.seed_params, 3)
        b = obj.evaluate(obj.seed_params, 3)
        assert a.viable
        assert a.fitness == pytest.approx(b.fitness)
        assert 0.0 <= a.side_info["toxicity"] <= 1.0

    def test_adaptive_objective_unknown_reward_raises(self):
        obj = build_adaptive_policy_objective(reward="not_a_reward", n_interactions=20)
        with pytest.raises(ValueError, match="unknown reward"):
            obj.evaluate(obj.seed_params, 0)

    def test_run_with_adaptive_objective_hill_climbs(self):
        obj = build_adaptive_policy_objective(n_interactions=100)
        cfg = HillClimbConfig(max_evals=30, n_workers=4, seed=7)
        r = run_delm_hillclimb(objective=obj, config=cfg)
        assert r.telemetry["objective"] == "adaptive_policy"
        assert r.best_score >= r.seed_score - 1e-9
        # Candidate params stay within the policy bounds.
        for name, (lo, hi) in obj.param_ranges.items():
            assert lo <= r.best_params[name] <= hi

    def test_run_with_adaptive_objective_reproducible(self):
        cfg = HillClimbConfig(max_evals=24, n_workers=3, seed=11)
        r1 = run_delm_hillclimb(
            objective=build_adaptive_policy_objective(n_interactions=80), config=cfg
        )
        r2 = run_delm_hillclimb(
            objective=build_adaptive_policy_objective(n_interactions=80), config=cfg
        )
        assert r1.best_score == r2.best_score
        assert r1.best_params == r2.best_params

    def test_custom_objective_threads_param_space(self):
        # A tiny custom 2-D objective with a known optimum at the upper corner.
        ranges = {"x": (0.0, 1.0), "y": (0.0, 1.0)}

        def evaluate(params, seed):
            score = params["x"] + params["y"]  # maximized at (1, 1)
            return EvalOutcome(fitness=score, side_info={}, viable=True)

        obj = Objective(
            name="corner",
            param_ranges=ranges,
            int_params=frozenset(),
            seed_params={"x": 0.1, "y": 0.1},
            evaluate=evaluate,
        )
        cfg = HillClimbConfig(max_evals=60, n_workers=4, seed=1)
        r = run_delm_hillclimb(objective=obj, config=cfg)
        # Should climb well above the seed (0.2) toward the corner (2.0).
        assert r.best_score > 1.0
        assert set(r.best_params) == {"x", "y"}


# =========================================================================
# Trajectory plot
# =========================================================================


class TestPlot:
    def test_plot_trajectory_writes_file(self, tmp_path):
        obj = build_adaptive_policy_objective(n_interactions=60)
        cfg = HillClimbConfig(max_evals=20, n_workers=3, seed=4)
        r = run_delm_hillclimb(objective=obj, config=cfg)
        out = plot_trajectory(r, tmp_path / "traj.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_run_emits_plot(self, tmp_path):
        from swarm.analysis.delm_hillclimb import write_run

        scenario = load_scenario(SCENARIO)
        cfg = HillClimbConfig(max_evals=16, n_workers=2, eval_epochs=1, eval_steps=3, seed=2)
        r = run_delm_hillclimb(scenario, cfg)
        write_run(r, tmp_path, "baseline", make_plot=True)
        assert (tmp_path / "best_params.json").exists()
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "gists.jsonl").exists()
        assert (tmp_path / "plots" / "trajectory.png").exists()
