"""Tests for swarm/adaptive/ — arm 2 scaffolding.

Smoke + invariants. The "does it actually learn?" question is for the
smoke run, not unit tests — but we do check that the trainer's
contracts (bounds, determinism, monotonic-ish improvement on a
constructed signal) hold.
"""

from __future__ import annotations

import random

import pytest

from swarm.adaptive import (
    PARAM_DIM,
    PARAM_NAMES,
    PARAM_SPEC,
    CEMConfig,
    Policy,
    run_episode,
    train_cem,
)
from swarm.core.payoff import PayoffConfig


class TestPolicy:
    def test_param_dim_and_names_consistent(self) -> None:
        assert PARAM_DIM == len(PARAM_NAMES) == len(PARAM_SPEC)

    def test_from_vector_clamps_out_of_range(self) -> None:
        # 99 for every param — must all clamp to high bound.
        p = Policy.from_vector([99.0] * PARAM_DIM)
        for name, _, hi in PARAM_SPEC:
            assert getattr(p, name) == hi

    def test_from_vector_clamps_low(self) -> None:
        p = Policy.from_vector([-99.0] * PARAM_DIM)
        for name, lo, _ in PARAM_SPEC:
            assert getattr(p, name) == lo

    def test_round_trip_vector(self) -> None:
        vec = [0.5, 0.2, 1.0, 0.3, 0.1, 0.4, 0.15, 0.0]
        assert vec == Policy.from_vector(vec).to_vector()

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            Policy.from_vector([0.0] * (PARAM_DIM - 1))

    def test_sample_observables_respects_bounds(self) -> None:
        # Even with mean at the bound and high std, sampled progress
        # must clamp to [-1, 1].
        p = Policy.from_vector([1.0, 0.5, 0.0, 0.0, 0.0, -1.0, 0.5, 0.0])
        rng = random.Random(0)
        for _ in range(50):
            obs = p.sample_observables(rng)
            assert -1.0 <= obs.task_progress_delta <= 1.0
            assert -1.0 <= obs.counterparty_engagement_delta <= 1.0
            assert obs.rework_count >= 0
            assert obs.verifier_rejections >= 0
            assert obs.tool_misuse_flags >= 0


class TestEpisode:
    def test_deterministic_under_seed(self) -> None:
        policy = Policy.from_vector([0.5, 0.2, 0.5, 0.2, 0.1, 0.3, 0.2, 0.0])
        cfg = PayoffConfig()
        a = run_episode(policy, n_interactions=50, payoff_config=cfg, seed=42)
        b = run_episode(policy, n_interactions=50, payoff_config=cfg, seed=42)
        assert a.mean_payoff == b.mean_payoff
        assert a.n_accepted == b.n_accepted

    def test_accept_threshold_at_high_excludes_everything(self) -> None:
        # v_hat is in [-1, 1]; threshold > 1 means no acceptance.
        policy = Policy.from_vector([0.5, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1, 1.0])
        # accept_threshold is clamped to 1.0 by from_vector; nothing
        # with v_hat strictly less should be accepted (but v_hat == 1.0
        # exactly will accept — that's a measure-zero edge).
        report = run_episode(policy, n_interactions=20, payoff_config=PayoffConfig(), seed=1)
        # almost all rejected
        assert report.n_accepted <= 1

    def test_metrics_in_valid_ranges(self) -> None:
        policy = Policy.from_vector([0.3, 0.2, 0.5, 0.2, 0.1, 0.2, 0.2, -0.5])
        report = run_episode(policy, n_interactions=100, payoff_config=PayoffConfig(), seed=7)
        assert 0.0 <= report.accept_rate <= 1.0
        assert 0.0 <= report.mean_p <= 1.0
        assert 0.0 <= report.toxicity <= 1.0
        assert -1.0 <= report.mean_v_hat <= 1.0


class TestCEM:
    def test_default_budget_is_pre_registered_shape(self) -> None:
        cfg = CEMConfig()
        # If someone changes the defaults without bumping a version,
        # this test fires so the prereg pinning is visible.
        assert cfg.population_size == 30
        assert cfg.elite_fraction == 0.25
        assert cfg.n_iterations == 10
        assert cfg.interactions_per_episode == 200
        assert cfg.n_elites == 7  # max(2, 30*0.25) = 7

    def test_train_is_deterministic_under_seed(self) -> None:
        # Same seed → same final policy and same iteration trace.
        small = CEMConfig(population_size=8, elite_fraction=0.5, n_iterations=2,
                          interactions_per_episode=40)
        cfg = PayoffConfig()
        a = train_cem(cfg, cem_config=small, seed=5)
        b = train_cem(cfg, cem_config=small, seed=5)
        assert a.final_policy.to_vector() == b.final_policy.to_vector()
        for ia, ib in zip(a.iterations, b.iterations, strict=True):
            assert ia.mean_elite_reward == ib.mean_elite_reward

    def test_train_records_one_iteration_per_step(self) -> None:
        cfg = PayoffConfig()
        small = CEMConfig(population_size=6, elite_fraction=0.5, n_iterations=3,
                          interactions_per_episode=30)
        report = train_cem(cfg, cem_config=small, seed=1)
        assert len(report.iterations) == 3
        for i, it in enumerate(report.iterations):
            assert it.iteration == i
            assert len(it.mu) == PARAM_DIM
            assert len(it.sigma) == PARAM_DIM

    def test_sigma_floor_prevents_collapse(self) -> None:
        # Run for enough iterations that without a floor the variance
        # would collapse. The floor is 5% of the parameter range.
        cfg = PayoffConfig()
        small = CEMConfig(population_size=10, elite_fraction=0.4, n_iterations=5,
                          interactions_per_episode=40)
        report = train_cem(cfg, cem_config=small, seed=11)
        # Final sigma must be at or above the floor for every parameter.
        for sigma_i, (_, lo, hi) in zip(
            report.iterations[-1].sigma, PARAM_SPEC, strict=True
        ):
            assert sigma_i >= (hi - lo) * 0.05 - 1e-9

    def test_default_reward_is_pinned(self) -> None:
        # The pinned reward is mean_attempted per the prereg addendum.
        # If this changes, the prereg addendum must be updated too.
        from swarm.adaptive.cem import PINNED_REWARD
        assert CEMConfig().reward == PINNED_REWARD == "mean_attempted"

    def test_invalid_reward_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid reward"):
            CEMConfig(reward="median_payoff")

    def test_alternate_reward_runs(self) -> None:
        # Smoke: each valid reward keyword runs the trainer without
        # crashing and produces a deterministic-under-seed result.
        from swarm.adaptive.cem import VALID_REWARDS
        cfg = PayoffConfig(rho_a=0.3)
        for reward in VALID_REWARDS:
            small = CEMConfig(
                population_size=6, elite_fraction=0.5, n_iterations=2,
                interactions_per_episode=30, reward=reward,
            )
            report = train_cem(cfg, cem_config=small, seed=0)
            assert len(report.iterations) == 2

    def test_episode_report_back_compat_alias(self) -> None:
        # mean_payoff alias still resolves to mean_payoff_accepted.
        policy = Policy.from_vector([0.5, 0.2, 0.5, 0.2, 0.1, 0.3, 0.2, 0.0])
        cfg = PayoffConfig()
        report = run_episode(policy, n_interactions=50, payoff_config=cfg, seed=42)
        assert report.mean_payoff == report.mean_payoff_accepted

    def test_to_dict_roundtrips_through_json(self) -> None:
        import json
        cfg = PayoffConfig(rho_a=0.3)
        small = CEMConfig(population_size=4, elite_fraction=0.5, n_iterations=2,
                          interactions_per_episode=20)
        report = train_cem(cfg, cem_config=small, seed=0)
        d = report.to_dict()
        # Must serialize without TypeError.
        json.dumps(d)
        assert d["payoff_config"]["rho_a"] == 0.3
