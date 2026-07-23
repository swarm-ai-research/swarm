"""Tests for swarm/adaptive/calibration_anchor.py and the label propagation
that makes the calibration anchor usable for the arm-2 deliverable.
"""

from __future__ import annotations

import pytest

from swarm.adaptive import (
    STATIC_BASELINES,
    STATIC_HONEST,
    STATIC_TOXIC,
    Policy,
    run_episode_with_interactions,
    run_population_episode_with_interactions,
    score_episode,
)
from swarm.adaptive.policy import PARAM_DIM
from swarm.core.payoff import PayoffConfig
from swarm.judges import MockJudge


class TestPolicyIdentityLabel:
    def test_default_label_is_empty(self) -> None:
        p = Policy.from_vector([0.0] * PARAM_DIM)
        assert p.identity_label == ""

    def test_static_baselines_carry_rubric_categories(self) -> None:
        # The whole point of label propagation: v3 target rules need
        # these exact strings to fire.
        assert STATIC_HONEST.identity_label == "honest"
        assert STATIC_TOXIC.identity_label == "blatant"


class TestEpisodeLabelPropagation:
    def test_adaptive_default_has_no_agent_type_metadata(self) -> None:
        # CEM-trained policies have empty identity_label; their items
        # must NOT carry agent_type so the v3 anchor falls back to
        # banded priors as the cross-scenario findings predicted.
        policy = Policy.from_vector([0.7, 0.15, 0.0, 0.0, 0.0, 0.5, 0.15, -0.5])
        _, interactions = run_episode_with_interactions(
            policy, n_interactions=20, payoff_config=PayoffConfig(), seed=1,
        )
        assert interactions, "expected some accepted items"
        for i in interactions:
            assert "agent_type" not in i.metadata

    def test_static_honest_propagates_agent_type(self) -> None:
        _, interactions = run_population_episode_with_interactions(
            STATIC_BASELINES["honest"],
            n_interactions=20, payoff_config=PayoffConfig(), seed=2,
        )
        assert interactions
        for i in interactions:
            assert i.metadata["agent_type"] == "honest"

    def test_static_mixed_emits_both_labels(self) -> None:
        # 70/30 honest/blatant — both labels should appear in the stream.
        _, interactions = run_population_episode_with_interactions(
            STATIC_BASELINES["mixed"],
            n_interactions=400, payoff_config=PayoffConfig(), seed=3,
        )
        labels = {i.metadata.get("agent_type") for i in interactions}
        assert "honest" in labels
        assert "blatant" in labels


class TestScoreEpisode:
    def test_v3_target_rule_fires_for_honest_baseline(self) -> None:
        # The "honest" target rule under v3 produces 0.85 for symmetric
        # COLLAB, 0.75 otherwise. Mean across an honest episode should
        # land in [0.70, 0.90].
        _, interactions = run_population_episode_with_interactions(
            STATIC_BASELINES["honest"],
            n_interactions=100, payoff_config=PayoffConfig(), seed=4,
        )
        judge = MockJudge(rubric_version="rubric.v3")
        report = score_episode(interactions, judge)
        assert 0.70 <= report.mean_judge_score <= 0.90
        # Every interaction should hit the target rule (agent_type=honest
        # is present, so 100% of n_with_target_rule_fired).
        assert report.n_with_target_rule_fired == report.n_scored

    def test_v3_target_rule_fires_for_blatant_baseline(self) -> None:
        _, interactions = run_population_episode_with_interactions(
            STATIC_BASELINES["toxic"],
            n_interactions=100, payoff_config=PayoffConfig(), seed=5,
        )
        judge = MockJudge(rubric_version="rubric.v3")
        report = score_episode(interactions, judge)
        # Blatant target is 0.10.
        assert report.mean_judge_score == pytest.approx(0.10, abs=0.05)

    def test_v3_falls_back_to_banded_prior_for_adaptive(self) -> None:
        # Unlabeled adaptive items hit the banded-prior path. The
        # default for symmetric agent_/agent_ COLLABORATION is 0.55.
        policy = Policy.from_vector([0.7, 0.15, 0.0, 0.0, 0.0, 0.5, 0.15, -0.5])
        _, interactions = run_episode_with_interactions(
            policy, n_interactions=100, payoff_config=PayoffConfig(), seed=6,
        )
        judge = MockJudge(rubric_version="rubric.v3")
        report = score_episode(interactions, judge)
        # The whole episode lands at the banded prior — and importantly,
        # NO target rules fire (because agent_type is absent).
        assert report.n_with_target_rule_fired == 0
        # Every score should be at the symmetric-friendly fallback
        # (initiator "adaptive_*" doesn't match the honest/blatant
        # patterns, so it falls into the symmetric-friendly REPLY/
        # COLLABORATION branch ≈ 0.55).
        assert 0.50 <= report.mean_judge_score <= 0.60

    def test_proxy_judge_gap_signs(self) -> None:
        # Static honest: v_hat-toxicity should be > 0 (some natural noise
        # in the observable generators), judge-toxicity should be ~0.15
        # (1 - 0.85 from honest+collab target). Sign of gap depends on
        # data — just confirm the field is computed.
        report, interactions = run_population_episode_with_interactions(
            STATIC_BASELINES["honest"],
            n_interactions=80, payoff_config=PayoffConfig(), seed=7,
        )
        judge = MockJudge(rubric_version="rubric.v3")
        judged = score_episode(
            interactions, judge, proxy_toxicity=report.toxicity,
        )
        assert judged.proxy_toxicity == report.toxicity
        assert judged.proxy_judge_gap == report.toxicity - judged.judge_toxicity

    def test_orthogonality_still_enforced(self) -> None:
        # The v_hat field is on every accepted SoftInteraction, but the
        # judge MUST NOT see it. score_episode runs make_view() which
        # strips it; assert this end-to-end.
        _, interactions = run_population_episode_with_interactions(
            STATIC_BASELINES["mixed"],
            n_interactions=30, payoff_config=PayoffConfig(), seed=8,
        )
        # If any forbidden field leaked, the judge's score call would
        # raise AssertionError via assert_view_is_orthogonal.
        judge = MockJudge(rubric_version="rubric.v3")
        report = score_episode(interactions, judge)
        assert report.n_scored > 0
