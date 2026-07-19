"""Tests for swarm/judges/agreement.py — arm C scaffolding."""

from __future__ import annotations

import math

import pytest

from swarm.judges import (
    ALPHA_ESCALATE,
    ALPHA_STRONG,
    agreement_by_pbin,
    decide_anchor_quality,
    icc_2k,
    krippendorff_alpha_interval,
    pairwise_spearman,
    run_agreement,
    spearman_rho,
)


def matrix(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return rows


class TestKrippendorffAlpha:
    def test_perfect_agreement_gives_one(self) -> None:
        m = matrix({
            "a": {"x": 0.1, "y": 0.5, "z": 0.9},
            "b": {"x": 0.1, "y": 0.5, "z": 0.9},
            "c": {"x": 0.1, "y": 0.5, "z": 0.9},
        })
        assert krippendorff_alpha_interval(m) == pytest.approx(1.0)

    def test_constant_scores_everywhere_returns_one(self) -> None:
        # Degenerate but well-defined: everyone scores 0.5 on everything.
        # SS_total is 0; we return 1.0 by convention.
        m = matrix({
            "a": {"x": 0.5, "y": 0.5},
            "b": {"x": 0.5, "y": 0.5},
        })
        assert krippendorff_alpha_interval(m) == 1.0

    def test_systematic_disagreement_gives_low_alpha(self) -> None:
        # Two judges flip every item. With a symmetric distribution this
        # gives alpha = 0 (within-unit disagreement matches population
        # disagreement). Any non-trivial agreement should beat this.
        m = matrix({
            "a": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 1.0},
            "b": {"x": 1.0, "y": 0.0, "z": 1.0, "w": 0.0},
        })
        assert krippendorff_alpha_interval(m) <= 0.0

    def test_within_unit_disagreement_exceeds_population_goes_negative(self) -> None:
        # Items that are nearly identical in the marginal distribution
        # but judges disagree wildly on each one → within > between → alpha < 0.
        m = matrix({
            "a": {"x1": 0.45, "x2": 0.55, "x3": 0.45, "x4": 0.55},
            "b": {"x1": 0.55, "x2": 0.45, "x3": 0.55, "x4": 0.45},
        })
        assert krippendorff_alpha_interval(m) < 0.0

    def test_nan_with_one_judge(self) -> None:
        m = matrix({"a": {"x": 0.2, "y": 0.8}})
        assert math.isnan(krippendorff_alpha_interval(m))


class TestICC:
    def test_perfect_agreement(self) -> None:
        m = matrix({
            "a": {"x": 0.1, "y": 0.5, "z": 0.9},
            "b": {"x": 0.1, "y": 0.5, "z": 0.9},
        })
        assert icc_2k(m) == pytest.approx(1.0, rel=1e-9)

    def test_nan_with_one_item(self) -> None:
        m = matrix({
            "a": {"x": 0.2},
            "b": {"x": 0.4},
        })
        assert math.isnan(icc_2k(m))


class TestSpearman:
    def test_monotonic_gives_one(self) -> None:
        assert spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)

    def test_anti_monotonic_gives_minus_one(self) -> None:
        assert spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            spearman_rho([1, 2], [1])

    def test_constant_gives_nan(self) -> None:
        # No variance in one series → undefined.
        assert math.isnan(spearman_rho([1, 1, 1], [1, 2, 3]))

    def test_pairwise(self) -> None:
        m = matrix({
            "a": {"x": 0.1, "y": 0.2, "z": 0.3},
            "b": {"x": 0.5, "y": 0.6, "z": 0.7},
            "c": {"x": 0.9, "y": 0.8, "z": 0.7},
        })
        pairs = pairwise_spearman(m)
        assert pairs[("a", "b")] == pytest.approx(1.0)
        assert pairs[("a", "c")] == pytest.approx(-1.0)


class TestVerdict:
    def test_thresholds(self) -> None:
        assert decide_anchor_quality(0.85) == "strong"
        assert decide_anchor_quality(ALPHA_STRONG) == "strong"
        assert decide_anchor_quality(0.6) == "usable"
        assert decide_anchor_quality(ALPHA_ESCALATE) == "usable"
        assert decide_anchor_quality(0.49) == "escalate"
        assert decide_anchor_quality(float("nan")) == "degenerate"


class TestRunAgreement:
    def test_smoke(self) -> None:
        m = matrix({
            "claude": {"x": 0.1, "y": 0.4, "z": 0.8, "w": 0.9},
            "gpt": {"x": 0.15, "y": 0.45, "z": 0.75, "w": 0.85},
            "llama": {"x": 0.2, "y": 0.5, "z": 0.7, "w": 0.95},
        })
        report = run_agreement(m)
        assert report.n_judges == 3
        assert report.n_items == 4
        assert report.alpha > 0.7  # high agreement
        assert report.verdict == "strong"
        assert len(report.spearman) == 3  # 3 pairs

    def test_dropped_items(self) -> None:
        # Only "x" is shared across judges.
        m = matrix({
            "a": {"x": 0.1, "y": 0.5},
            "b": {"x": 0.2, "z": 0.5},
        })
        report = run_agreement(m)
        assert report.n_items == 1

    def test_single_judge_returns_degenerate(self) -> None:
        # Arm C requires >=2 judges. Single judge returns alpha=NaN → verdict="degenerate".
        m = matrix({
            "mock": {"x": 0.1, "y": 0.4, "z": 0.8, "w": 0.9},
        })
        report = run_agreement(m)
        assert report.n_judges == 1
        assert math.isnan(report.alpha)
        assert report.verdict == "degenerate"


class TestPerBin:
    def test_bins_distinct_alphas(self) -> None:
        # Strong agreement at the bottom bin, disagreement at the top.
        m = matrix({
            "a": {"x1": 0.1, "x2": 0.15, "y1": 0.9, "y2": 0.1},
            "b": {"x1": 0.1, "x2": 0.15, "y1": 0.2, "y2": 0.95},
        })
        p_by_item = {"x1": 0.05, "x2": 0.10, "y1": 0.95, "y2": 0.92}
        bins = agreement_by_pbin(m, p_by_item)
        # Find the [0,0.2) and [0.8,1.0] bins.
        low = next(b for b in bins if b.lo == 0.0)
        high = next(b for b in bins if b.hi == 1.0)
        assert low.alpha > high.alpha
        assert low.mean_pairwise_disagreement < high.mean_pairwise_disagreement
