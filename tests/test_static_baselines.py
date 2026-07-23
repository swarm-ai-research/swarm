"""Tests for swarm/adaptive/static_baselines.py — static-vs-adaptive overlay."""

from __future__ import annotations

import pytest

from swarm.adaptive import (
    STATIC_BASELINES,
    STATIC_HONEST,
    STATIC_TOXIC,
    StaticBaseline,
    run_population_episode,
)
from swarm.core.payoff import PayoffConfig


class TestStaticPolicies:
    def test_honest_has_positive_quality_signals(self) -> None:
        # Honest baseline should have positive progress mean and engagement,
        # and no negative observable signals.
        assert STATIC_HONEST.progress_mean > 0.5
        assert STATIC_HONEST.engagement_mean > 0.3
        assert STATIC_HONEST.rework_lambda == 0.0
        assert STATIC_HONEST.rejection_lambda == 0.0
        assert STATIC_HONEST.misuse_lambda == 0.0

    def test_toxic_has_negative_signals(self) -> None:
        assert STATIC_TOXIC.progress_mean < 0.0
        assert STATIC_TOXIC.engagement_mean < 0.0
        assert STATIC_TOXIC.rework_lambda > 1.0
        assert STATIC_TOXIC.rejection_lambda > 0.0

    def test_honest_accept_threshold_below_toxic(self) -> None:
        # Honest agents accept anything (low threshold); toxic agents
        # may self-reject some.
        assert STATIC_HONEST.accept_threshold <= STATIC_TOXIC.accept_threshold


class TestStaticBaselineDataclass:
    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            StaticBaseline(name="bad", policies=(STATIC_HONEST,), weights=(0.5, 0.5))

    def test_zero_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            StaticBaseline(name="bad", policies=(STATIC_HONEST,), weights=(0.0,))


class TestRegistry:
    def test_canonical_baselines_present(self) -> None:
        assert {"honest", "toxic", "mixed"} <= STATIC_BASELINES.keys()

    def test_mixed_is_70_30(self) -> None:
        mixed = STATIC_BASELINES["mixed"]
        assert mixed.policies == (STATIC_HONEST, STATIC_TOXIC)
        # 70/30 split per the generate_mixed_batch convention.
        assert mixed.weights == (0.7, 0.3)


class TestRunPopulationEpisode:
    def test_deterministic_under_seed(self) -> None:
        baseline = STATIC_BASELINES["mixed"]
        cfg = PayoffConfig()
        a = run_population_episode(baseline, n_interactions=50, payoff_config=cfg, seed=42)
        b = run_population_episode(baseline, n_interactions=50, payoff_config=cfg, seed=42)
        assert a.mean_payoff_attempted == b.mean_payoff_attempted
        assert a.n_accepted == b.n_accepted

    def test_honest_has_lower_toxicity_than_toxic(self) -> None:
        cfg = PayoffConfig()
        honest = run_population_episode(
            STATIC_BASELINES["honest"], n_interactions=200, payoff_config=cfg, seed=1,
        )
        toxic = run_population_episode(
            STATIC_BASELINES["toxic"], n_interactions=200, payoff_config=cfg, seed=1,
        )
        assert honest.toxicity < toxic.toxicity

    def test_mixed_toxicity_between_honest_and_toxic(self) -> None:
        cfg = PayoffConfig()
        honest = run_population_episode(
            STATIC_BASELINES["honest"], n_interactions=400, payoff_config=cfg, seed=2,
        )
        toxic = run_population_episode(
            STATIC_BASELINES["toxic"], n_interactions=400, payoff_config=cfg, seed=2,
        )
        mixed = run_population_episode(
            STATIC_BASELINES["mixed"], n_interactions=400, payoff_config=cfg, seed=2,
        )
        # 70/30 honest/toxic — mixed toxicity should land between honest
        # and toxic on the (accepted) toxicity axis.
        assert honest.toxicity < mixed.toxicity < toxic.toxicity

    def test_rho_lowers_payoff_at_fixed_policy(self) -> None:
        # The pre-reg's Figure 4 finding: under static, ρ produces a
        # vertical welfare collapse without changing toxicity. Replicate
        # the welfare half here.
        baseline = STATIC_BASELINES["mixed"]
        rho0 = run_population_episode(
            baseline, n_interactions=200, payoff_config=PayoffConfig(rho_a=0.0, rho_b=0.0), seed=3,
        )
        rho1 = run_population_episode(
            baseline, n_interactions=200, payoff_config=PayoffConfig(rho_a=1.0, rho_b=1.0), seed=3,
        )
        assert rho1.mean_payoff_attempted < rho0.mean_payoff_attempted

    def test_episode_report_shape_matches_adaptive(self) -> None:
        # The headline value of this module: the report is directly
        # joinable against the adaptive grid_summary.csv schema.
        cfg = PayoffConfig()
        report = run_population_episode(
            STATIC_BASELINES["honest"], n_interactions=50, payoff_config=cfg, seed=7,
        )
        # Every adaptive-side field must exist with the same name.
        for field in (
            "n_total", "n_accepted", "accept_rate",
            "mean_payoff_accepted", "mean_payoff_attempted", "sum_payoff",
            "mean_v_hat", "mean_p", "toxicity", "mean_progress",
        ):
            assert hasattr(report, field), f"missing field {field}"

    def test_single_policy_baseline_collapses_to_run_episode(self) -> None:
        # If a baseline has weight 1 on one policy, run_population_episode
        # is just running that policy — should produce sane numbers.
        baseline = StaticBaseline(
            name="solo", policies=(STATIC_HONEST,), weights=(1.0,),
        )
        cfg = PayoffConfig()
        report = run_population_episode(
            baseline, n_interactions=100, payoff_config=cfg, seed=11,
        )
        assert report.n_total == 100
        assert 0.0 <= report.accept_rate <= 1.0
        assert 0.0 <= report.toxicity <= 1.0
