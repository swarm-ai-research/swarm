"""Tests for the active cause-3 probe.

Cause-3 in the prereg framing is "agents game the proxy" — the
adaptive agent learns to inflate v_hat (or, in this probe, to misclaim
its identity to the calibration anchor) while underlying quality is
unchanged. The probe gives the agent a metadata channel and a
judge-weighted reward; this test suite covers the surface that's
deterministic (parameter parsing, label thresholds, reward dispatch)
without relying on the CEM converging in any particular direction.
"""

from __future__ import annotations

import pytest

from swarm.adaptive import (
    CAUSE3_PARAM_DIM,
    Cause3CEMConfig,
    ClaimingParams,
    run_cause3_episode,
    train_cem_cause3,
)
from swarm.adaptive.cause3 import (
    CLAIM_BLATANT_THRESHOLD,
    CLAIM_HONEST_THRESHOLD,
    DEFAULT_CAUSE3_REWARD,
    VALID_CAUSE3_REWARDS,
)
from swarm.adaptive.policy import PARAM_DIM as POLICY_PARAM_DIM
from swarm.core.payoff import PayoffConfig
from swarm.judges import MockJudge


class TestClaimingParams:
    def test_param_dim_is_policy_plus_one(self) -> None:
        assert CAUSE3_PARAM_DIM == POLICY_PARAM_DIM + 1

    def test_from_vector_unpacks_correctly(self) -> None:
        vec = [0.5, 0.2, 0.3, 0.0, 0.0, 0.4, 0.2, -0.5, 0.8]
        params = ClaimingParams.from_vector(vec)
        assert params.policy.progress_mean == 0.5
        assert params.policy.accept_threshold == -0.5
        assert params.claim_param == 0.8

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            ClaimingParams.from_vector([0.0] * (CAUSE3_PARAM_DIM - 1))

    def test_claim_param_clamps(self) -> None:
        # Both out-of-range claims should clamp to [0, 1].
        high = ClaimingParams.from_vector([0.0] * POLICY_PARAM_DIM + [99.0])
        low = ClaimingParams.from_vector([0.0] * POLICY_PARAM_DIM + [-99.0])
        assert high.claim_param == 1.0
        assert low.claim_param == 0.0

    def test_claimed_label_thresholds(self) -> None:
        def label_at(p: float) -> str:
            return ClaimingParams.from_vector(
                [0.0] * POLICY_PARAM_DIM + [p]
            ).claimed_label

        assert label_at(0.0) == "blatant"
        assert label_at(CLAIM_BLATANT_THRESHOLD - 1e-6) == "blatant"
        assert label_at(CLAIM_BLATANT_THRESHOLD) == ""
        assert label_at(0.5) == ""
        assert label_at(CLAIM_HONEST_THRESHOLD - 1e-6) == ""
        assert label_at(CLAIM_HONEST_THRESHOLD) == "honest"
        assert label_at(1.0) == "honest"

    def test_round_trip(self) -> None:
        vec = [0.4, 0.15, 0.5, 0.2, 0.1, 0.3, 0.2, 0.0, 0.7]
        assert vec == ClaimingParams.from_vector(vec).to_vector()


class TestCause3Episode:
    def _hi_quality_params(self, claim_param: float) -> ClaimingParams:
        return ClaimingParams.from_vector(
            [0.7, 0.15, 0.0, 0.0, 0.0, 0.5, 0.15, -0.5, claim_param]
        )

    def test_deterministic_under_seed(self) -> None:
        params = self._hi_quality_params(0.5)
        judge = MockJudge(rubric_version="rubric.v3")
        cfg = PayoffConfig()
        a = run_cause3_episode(params, judge, n_interactions=50, payoff_config=cfg, seed=11)
        b = run_cause3_episode(params, judge, n_interactions=50, payoff_config=cfg, seed=11)
        assert a.mean_judge_score == b.mean_judge_score
        assert a.proxy_toxicity == b.proxy_toxicity

    def test_blatant_claim_triggers_blatant_target(self) -> None:
        # Same observable distribution, but claiming "blatant" should
        # push judge_toxicity to ~0.90 because v3 target rule for
        # blatant is 0.10.
        params = self._hi_quality_params(0.0)  # claim = blatant
        judge = MockJudge(rubric_version="rubric.v3")
        report = run_cause3_episode(
            params, judge, n_interactions=100, payoff_config=PayoffConfig(), seed=1,
        )
        assert report.claimed_label == "blatant"
        assert report.judge_toxicity == pytest.approx(0.90, abs=0.02)
        # Target rule fired on every accepted item.
        assert report.n_with_target_rule_fired == report.n_accepted

    def test_honest_claim_triggers_honest_target(self) -> None:
        params = self._hi_quality_params(1.0)  # claim = honest
        judge = MockJudge(rubric_version="rubric.v3")
        report = run_cause3_episode(
            params, judge, n_interactions=100, payoff_config=PayoffConfig(), seed=2,
        )
        assert report.claimed_label == "honest"
        # Target rule for honest = 0.75; if symmetric COLLAB also fires
        # the corroborating bonus could push to 0.85. We allow either.
        assert report.mean_judge_score >= 0.70

    def test_no_claim_falls_back_to_banded_prior(self) -> None:
        params = self._hi_quality_params(0.5)  # claim = ""
        judge = MockJudge(rubric_version="rubric.v3")
        report = run_cause3_episode(
            params, judge, n_interactions=100, payoff_config=PayoffConfig(), seed=3,
        )
        assert report.claimed_label == ""
        assert report.n_with_target_rule_fired == 0
        # Banded prior for symmetric-friendly COLLABORATION = 0.55.
        assert report.mean_judge_score == pytest.approx(0.55, abs=0.05)

    def test_proxy_judge_gap_captures_gaming_potential(self) -> None:
        # Hi-quality observables (proxy says ~clean) + honest claim →
        # both signals agree, gap small.
        hi_hon = run_cause3_episode(
            self._hi_quality_params(1.0),
            MockJudge(rubric_version="rubric.v3"),
            n_interactions=80, payoff_config=PayoffConfig(), seed=4,
        )
        # Hi-quality observables + blatant claim → proxy says clean,
        # judge says toxic. Gap should be strongly negative (judge
        # more pessimistic than proxy).
        hi_bla = run_cause3_episode(
            self._hi_quality_params(0.0),
            MockJudge(rubric_version="rubric.v3"),
            n_interactions=80, payoff_config=PayoffConfig(), seed=4,
        )
        assert hi_bla.proxy_judge_gap < hi_hon.proxy_judge_gap

    def test_orthogonality_still_enforced(self) -> None:
        # The judge must not see ProxyComputer-observables. If forbidden
        # fields ever leak, assert_view_is_orthogonal raises inside
        # judge.score().
        params = self._hi_quality_params(1.0)
        judge = MockJudge(rubric_version="rubric.v3")
        report = run_cause3_episode(
            params, judge, n_interactions=20, payoff_config=PayoffConfig(), seed=5,
        )
        assert report.n_accepted > 0


class TestCause3CEM:
    def test_default_reward_is_judge(self) -> None:
        assert Cause3CEMConfig().reward == DEFAULT_CAUSE3_REWARD == "judge"

    def test_invalid_reward_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid reward"):
            Cause3CEMConfig(reward="anything_else")

    def test_train_is_deterministic_under_seed(self) -> None:
        judge = MockJudge(rubric_version="rubric.v3")
        small = Cause3CEMConfig(
            population_size=6, elite_fraction=0.5, n_iterations=2,
            interactions_per_episode=40,
        )
        a = train_cem_cause3(PayoffConfig(), judge, cem_config=small, seed=7)
        b = train_cem_cause3(PayoffConfig(), judge, cem_config=small, seed=7)
        assert a.final_params.to_vector() == b.final_params.to_vector()
        for ia, ib in zip(a.iterations, b.iterations, strict=True):
            assert ia.mean_elite_reward == ib.mean_elite_reward

    def test_iterations_have_correct_shape(self) -> None:
        judge = MockJudge(rubric_version="rubric.v3")
        small = Cause3CEMConfig(
            population_size=8, elite_fraction=0.5, n_iterations=3,
            interactions_per_episode=30,
        )
        report = train_cem_cause3(PayoffConfig(), judge, cem_config=small, seed=1)
        assert len(report.iterations) == 3
        for i, it in enumerate(report.iterations):
            assert it.iteration == i
            assert len(it.mu) == CAUSE3_PARAM_DIM
            assert len(it.sigma) == CAUSE3_PARAM_DIM

    def test_all_valid_rewards_run(self) -> None:
        # Smoke test each reward without asserting where it converges.
        judge = MockJudge(rubric_version="rubric.v3")
        for reward in VALID_CAUSE3_REWARDS:
            small = Cause3CEMConfig(
                population_size=6, elite_fraction=0.5, n_iterations=2,
                interactions_per_episode=30, reward=reward,
            )
            report = train_cem_cause3(PayoffConfig(), judge, cem_config=small, seed=2)
            assert len(report.iterations) == 2

    def test_judge_reward_converges_to_honest_claim(self) -> None:
        # The load-bearing test: with reward = mean_judge_score and a
        # v3 judge that rewards "honest" claims at 0.75-0.85 and
        # "blatant" claims at 0.10, CEM should converge to claim_param
        # >= 2/3 (honest band).
        judge = MockJudge(rubric_version="rubric.v3")
        # Use the pre-reg-shaped budget so the result is meaningful.
        cfg = Cause3CEMConfig(
            population_size=30, elite_fraction=0.25, n_iterations=10,
            interactions_per_episode=200, reward="judge",
        )
        report = train_cem_cause3(PayoffConfig(rho_a=0.3), judge, cem_config=cfg, seed=42)
        # If the probe works, the final policy should claim "honest"
        # because that's the highest-scoring label under v3.
        assert report.final_params.claimed_label == "honest"
        # And it should have learned to do so within the budget.
        assert report.iterations[-1].mean_elite_claim_param >= CLAIM_HONEST_THRESHOLD

    def test_to_dict_serializes(self) -> None:
        import json
        judge = MockJudge(rubric_version="rubric.v3")
        small = Cause3CEMConfig(
            population_size=4, elite_fraction=0.5, n_iterations=2,
            interactions_per_episode=20,
        )
        report = train_cem_cause3(PayoffConfig(rho_a=0.5), judge, cem_config=small, seed=0)
        d = report.to_dict()
        json.dumps(d)
        assert d["final_params"]["claimed_label"] in {"blatant", "", "honest"}
        assert d["config"]["reward"] == "judge"
