"""Tests for soft metrics."""

from typing import Any, Hashable, Mapping, Optional

import pytest

from swarm.metrics.incoherence import (
    BenchmarkPolicy,
    DecisionRecord,
    IncoherenceMetrics,
)
from swarm.metrics.reporters import MetricsReporter
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction
from tests.fixtures.interactions import (
    generate_benign_batch,
    generate_mixed_batch,
    generate_toxic_batch,
    generate_uncertain_batch,
)


class _SingleActionBenchmark(BenchmarkPolicy):
    """Minimal benchmark policy for incoherence reporter tests."""

    def action_for(
        self,
        decision_id: str,
        task_family: str,
        metadata: Mapping[str, Any],
    ) -> Optional[Hashable]:
        return "approve"


class TestToxicityRate:
    """Tests for toxicity rate calculation."""

    def test_toxicity_benign_batch(self):
        """Benign batch should have low toxicity."""
        interactions = generate_benign_batch(count=100, seed=42)
        metrics = SoftMetrics()

        toxicity = metrics.toxicity_rate(interactions)

        # Benign has p in [0.7, 0.95], so toxicity = E[1-p] in [0.05, 0.3]
        assert toxicity < 0.35

    def test_toxicity_toxic_batch(self):
        """Toxic batch should have high toxicity."""
        interactions = generate_toxic_batch(count=100, seed=42)
        metrics = SoftMetrics()

        toxicity = metrics.toxicity_rate(interactions)

        # Toxic has p in [0.1, 0.4], so toxicity = E[1-p] in [0.6, 0.9]
        assert toxicity > 0.55

    def test_toxicity_bounds(self):
        """Toxicity should be in [0, 1]."""
        metrics = SoftMetrics()

        for batch_gen in [
            generate_benign_batch,
            generate_toxic_batch,
            generate_mixed_batch,
        ]:
            interactions = batch_gen(count=50, seed=42)
            toxicity = metrics.toxicity_rate(interactions)
            assert 0.0 <= toxicity <= 1.0

    def test_toxicity_empty_list(self):
        """Empty list should return 0."""
        metrics = SoftMetrics()
        assert metrics.toxicity_rate([]) == 0.0

    def test_toxicity_no_accepted(self):
        """No accepted interactions should return 0."""
        metrics = SoftMetrics()
        interactions = [
            SoftInteraction(p=0.2, accepted=False),
            SoftInteraction(p=0.8, accepted=False),
        ]
        assert metrics.toxicity_rate(interactions) == 0.0


class TestConditionalLoss:
    """Tests for conditional loss calculation."""

    def test_conditional_loss_adverse_selection(self):
        """
        When low-quality interactions are preferentially accepted,
        conditional loss should be negative.
        """
        metrics = SoftMetrics()

        # Create adverse selection: low p accepted, high p rejected
        interactions = []
        for p in [0.2, 0.3, 0.25]:  # Low quality, accepted
            interactions.append(SoftInteraction(p=p, accepted=True))
        for p in [0.8, 0.85, 0.9]:  # High quality, rejected
            interactions.append(SoftInteraction(p=p, accepted=False))

        cl = metrics.conditional_loss_initiator(interactions)

        # Accepted have lower payoffs than average → negative CL
        assert cl < 0

    def test_conditional_loss_positive_selection(self):
        """
        When high-quality interactions are preferentially accepted,
        conditional loss should be positive.
        """
        metrics = SoftMetrics()

        # Create positive selection
        interactions = []
        for p in [0.8, 0.85, 0.9]:  # High quality, accepted
            interactions.append(SoftInteraction(p=p, accepted=True))
        for p in [0.2, 0.3, 0.25]:  # Low quality, rejected
            interactions.append(SoftInteraction(p=p, accepted=False))

        cl = metrics.conditional_loss_initiator(interactions)

        # Accepted have higher payoffs than average → positive CL
        assert cl > 0


class TestSpread:
    """Tests for spread calculation."""

    def test_spread_adverse_selection(self):
        """Positive spread indicates adverse selection in quality."""
        metrics = SoftMetrics()

        # Low quality preferentially accepted
        interactions = []
        for _ in range(5):
            interactions.append(SoftInteraction(p=0.3, accepted=True))
        for _ in range(5):
            interactions.append(SoftInteraction(p=0.8, accepted=False))

        spread = metrics.spread(interactions)

        # E[p] > E[p|accepted] → positive spread
        assert spread > 0

    def test_spread_zero_when_uniform(self):
        """Spread should be near zero with uniform acceptance."""
        metrics = SoftMetrics()

        # All interactions accepted
        interactions = [
            SoftInteraction(p=p, accepted=True) for p in [0.2, 0.4, 0.6, 0.8]
        ]

        spread = metrics.spread(interactions)

        # E[p] = E[p|accepted] → zero spread
        assert spread == pytest.approx(0.0)


class TestQualityGap:
    """Tests for quality gap calculation."""

    def test_quality_gap_adverse_selection(self):
        """Negative quality gap indicates adverse selection."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.3, accepted=True),
            SoftInteraction(p=0.35, accepted=True),
            SoftInteraction(p=0.8, accepted=False),
            SoftInteraction(p=0.85, accepted=False),
        ]

        gap = metrics.quality_gap(interactions)

        # E[p|accepted] < E[p|rejected] → negative gap
        assert gap < 0

    def test_quality_gap_positive_selection(self):
        """Positive quality gap indicates good filtering."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.8, accepted=True),
            SoftInteraction(p=0.85, accepted=True),
            SoftInteraction(p=0.3, accepted=False),
            SoftInteraction(p=0.35, accepted=False),
        ]

        gap = metrics.quality_gap(interactions)

        # E[p|accepted] > E[p|rejected] → positive gap
        assert gap > 0

    def test_quality_gap_empty_groups(self):
        """Should return 0 if either group is empty."""
        metrics = SoftMetrics()

        # All accepted
        interactions = [SoftInteraction(p=0.5, accepted=True)]
        assert metrics.quality_gap(interactions) == 0.0

        # All rejected
        interactions = [SoftInteraction(p=0.5, accepted=False)]
        assert metrics.quality_gap(interactions) == 0.0


class TestPlausibilityCertificateGap:
    """Tests for PCG = E[p - p_cert | accepted, certified] (beads mt8a)."""

    def test_none_without_certificates(self):
        metrics = SoftMetrics()
        interactions = [
            SoftInteraction(p=0.9, accepted=True),
            SoftInteraction(p=0.2, accepted=True),
        ]
        assert metrics.plausibility_certificate_gap(interactions) is None

    def test_none_when_only_rejected_certified(self):
        metrics = SoftMetrics()
        interactions = [
            SoftInteraction(p=0.9, accepted=False, ground_truth=-1),
            SoftInteraction(p=0.8, accepted=True),
        ]
        assert metrics.plausibility_certificate_gap(interactions) is None

    def test_fabricated_positives_drive_pcg_up(self):
        """Accepted high-p interactions with ground_truth=-1 give PCG near 1."""
        metrics = SoftMetrics()
        interactions = [
            SoftInteraction(p=0.9, accepted=True, ground_truth=-1),
            SoftInteraction(p=0.8, accepted=True, ground_truth=-1),
        ]
        pcg = metrics.plausibility_certificate_gap(interactions)
        assert pcg == pytest.approx(0.85)

    def test_calibrated_certified_set_gives_zero(self):
        """p matching certificates exactly yields PCG 0."""
        metrics = SoftMetrics()
        interactions = [
            SoftInteraction(p=1.0, accepted=True, ground_truth=1),
            SoftInteraction(p=0.0, accepted=True, ground_truth=-1),
        ]
        assert metrics.plausibility_certificate_gap(interactions) == pytest.approx(
            0.0
        )

    def test_rejected_certified_excluded(self):
        """Rejected interactions must not enter accepted-set PCG."""
        metrics = SoftMetrics()
        interactions = [
            SoftInteraction(p=0.9, accepted=True, ground_truth=1),
            SoftInteraction(p=0.9, accepted=False, ground_truth=-1),
        ]
        pcg = metrics.plausibility_certificate_gap(interactions)
        assert pcg == pytest.approx(-0.1)

    def test_decomposition_selection_differential(self):
        """Fabricated positives accepted, honest negatives rejected →
        positive selection differential."""
        metrics = SoftMetrics()
        interactions = [
            # Accepted: over-scored fabrications
            SoftInteraction(p=0.9, accepted=True, ground_truth=-1),
            SoftInteraction(p=0.85, accepted=True, ground_truth=-1),
            # Rejected: well-scored honest negatives
            SoftInteraction(p=0.2, accepted=False, ground_truth=1),
            # Uncertified accepted (coverage < 1)
            SoftInteraction(p=0.7, accepted=True),
        ]
        decomp = metrics.pcg_decomposition(interactions)
        assert decomp["pcg_accepted"] == pytest.approx(0.875)
        assert decomp["pcg_rejected"] == pytest.approx(-0.8)
        assert decomp["selection_differential"] == pytest.approx(1.675)
        assert decomp["certified_coverage_accepted"] == pytest.approx(2 / 3)
        assert decomp["n_certified_accepted"] == 2
        assert decomp["n_certified_rejected"] == 1

    def test_decomposition_empty(self):
        metrics = SoftMetrics()
        decomp = metrics.pcg_decomposition([])
        assert decomp["pcg_accepted"] is None
        assert decomp["pcg_rejected"] is None
        assert decomp["selection_differential"] is None
        assert decomp["certified_coverage_accepted"] == 0.0

    def test_differs_from_calibration_error_by_conditioning(self):
        """PCG conditions on acceptance; calibration_error does not."""
        metrics = SoftMetrics()
        interactions = [
            SoftInteraction(p=0.9, accepted=True, ground_truth=-1),
            SoftInteraction(p=0.1, accepted=False, ground_truth=-1),
        ]
        pcg = metrics.plausibility_certificate_gap(interactions)
        cal = metrics.calibration_error(interactions)
        assert pcg == pytest.approx(0.9)
        assert cal == pytest.approx(0.5)


class TestProjectionDiagnostics:
    """Tests for ρ(p,a), toxicity decomposition, and C-S saturation."""

    def _mixed(self):
        # 4 interactions: 2 accepted (high p), 2 rejected (low p) → positive Q
        return [
            SoftInteraction(p=0.9, accepted=True),
            SoftInteraction(p=0.7, accepted=True),
            SoftInteraction(p=0.3, accepted=False),
            SoftInteraction(p=0.1, accepted=False),
        ]

    def test_correlation_in_unit_interval(self):
        rho = SoftMetrics().quality_correlation(self._mixed())
        assert -1.0 <= rho <= 1.0
        assert rho > 0  # accepts have higher p than rejects

    def test_correlation_degenerate(self):
        m = SoftMetrics()
        # All accepted ⇒ Var(a)=0 ⇒ ρ=0
        assert m.quality_correlation(
            [SoftInteraction(p=0.5, accepted=True) for _ in range(4)]
        ) == 0.0
        # All same p ⇒ Var(p)=0 ⇒ ρ=0
        assert m.quality_correlation([
            SoftInteraction(p=0.5, accepted=True),
            SoftInteraction(p=0.5, accepted=False),
        ]) == 0.0

    def test_correlation_perfect_sorting(self):
        # p is an affine function of a ⇒ |ρ| = 1
        interactions = [
            SoftInteraction(p=1.0, accepted=True),
            SoftInteraction(p=1.0, accepted=True),
            SoftInteraction(p=0.0, accepted=False),
            SoftInteraction(p=0.0, accepted=False),
        ]
        rho = SoftMetrics().quality_correlation(interactions)
        assert abs(rho - 1.0) < 1e-9

    def test_toxicity_decomposition_reconstructs(self):
        d = SoftMetrics().toxicity_decomposition(self._mixed())
        # T = baseline_harm - selection_credit, up to float noise
        assert d["reconstruction_error"] < 1e-9
        assert d["baseline_harm"] > 0
        assert d["selection_credit"] > 0  # good selection ⇒ positive credit

    def test_saturation_in_unit_interval(self):
        s = SoftMetrics().selection_saturation(self._mixed())
        assert 0.0 <= s <= 1.0 + 1e-9

    def test_saturation_one_at_perfect_sorting(self):
        interactions = [
            SoftInteraction(p=1.0, accepted=True),
            SoftInteraction(p=0.0, accepted=False),
        ]
        s = SoftMetrics().selection_saturation(interactions)
        assert abs(s - 1.0) < 1e-9

    def test_reporter_emits_diagnostics(self):
        summary = MetricsReporter().summary(self._mixed())
        assert summary.quality_correlation > 0
        assert summary.baseline_harm > 0
        assert summary.selection_credit > 0
        # Reporter's T = baseline - credit
        assert abs(
            summary.toxicity_soft - (summary.baseline_harm - summary.selection_credit)
        ) < 1e-9
        assert 0.0 <= summary.selection_saturation <= 1.0 + 1e-9


class TestParticipationByQuality:
    """Tests for participation by quality."""

    def test_participation_rates(self):
        """Should compute correct acceptance rates by quality tier."""
        metrics = SoftMetrics()

        interactions = [
            # High quality (p >= 0.5): 3 total, 2 accepted
            SoftInteraction(p=0.6, accepted=True),
            SoftInteraction(p=0.7, accepted=True),
            SoftInteraction(p=0.8, accepted=False),
            # Low quality (p < 0.5): 2 total, 1 accepted
            SoftInteraction(p=0.3, accepted=True),
            SoftInteraction(p=0.4, accepted=False),
        ]

        result = metrics.participation_by_quality(interactions, threshold=0.5)

        assert result["high_quality_acceptance"] == pytest.approx(2 / 3)
        assert result["low_quality_acceptance"] == pytest.approx(1 / 2)
        assert result["high_quality_count"] == 3
        assert result["low_quality_count"] == 2


class TestUncertainty:
    """Tests for uncertainty flagging."""

    def test_flag_uncertain(self):
        """Should flag interactions near p=0.5."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.45),  # Uncertain
            SoftInteraction(p=0.55),  # Uncertain
            SoftInteraction(p=0.2),  # Not uncertain
            SoftInteraction(p=0.8),  # Not uncertain
        ]

        uncertain = metrics.flag_uncertain(interactions, band=0.1)

        assert len(uncertain) == 2
        assert all(0.4 < i.p < 0.6 for i in uncertain)

    def test_uncertain_fraction(self):
        """Should compute correct uncertain fraction."""
        metrics = SoftMetrics()

        interactions = generate_uncertain_batch(count=100, band=0.15, seed=42)
        fraction = metrics.uncertain_fraction(interactions, band=0.2)

        # All generated are in [0.35, 0.65], so all should be uncertain
        assert fraction == 1.0


class TestAverageQuality:
    """Tests for average quality."""

    def test_average_quality_all(self):
        """Should compute E[p] correctly."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.2),
            SoftInteraction(p=0.4),
            SoftInteraction(p=0.6),
            SoftInteraction(p=0.8),
        ]

        avg = metrics.average_quality(interactions)
        assert avg == pytest.approx(0.5)

    def test_average_quality_accepted_only(self):
        """Should compute E[p|accepted] correctly."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.2, accepted=False),
            SoftInteraction(p=0.4, accepted=True),
            SoftInteraction(p=0.6, accepted=True),
            SoftInteraction(p=0.8, accepted=False),
        ]

        avg = metrics.average_quality(interactions, accepted_only=True)
        assert avg == pytest.approx(0.5)


class TestMetricsReporter:
    """Tests for MetricsReporter."""

    def test_summary_structure(self):
        """Summary should have all expected fields."""
        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=100, seed=42)

        summary = reporter.summary(interactions)

        assert hasattr(summary, "toxicity_soft")
        assert hasattr(summary, "toxicity_hard")
        assert hasattr(summary, "quality_gap")
        assert hasattr(summary, "spread")
        assert hasattr(summary, "total_interactions")

    def test_summary_empty(self):
        """Empty list should produce zero summary."""
        reporter = MetricsReporter()
        summary = reporter.summary([])

        assert summary.total_interactions == 0
        assert summary.toxicity_soft == 0.0

    def test_compare_soft_hard(self):
        """Should produce comparison dictionary."""
        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=100, seed=42)

        comparison = reporter.compare_soft_hard(interactions)

        assert "toxicity" in comparison
        assert "soft" in comparison["toxicity"]
        assert "hard" in comparison["toxicity"]

    def test_format_report(self):
        """Should produce formatted string report."""
        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=100, seed=42)

        report = reporter.format_report(interactions, verbose=True)

        assert isinstance(report, str)
        assert "METRICS REPORT" in report
        assert "SOFT METRICS" in report
        assert "HARD METRICS" in report

    def test_to_dict(self):
        """Summary should be convertible to dict."""
        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=50, seed=42)

        summary = reporter.summary(interactions)
        d = summary.to_dict()

        assert isinstance(d, dict)
        assert "soft_metrics" in d
        assert "hard_metrics" in d
        assert "counts" in d
        assert "welfare" in d

    def test_to_markdown_structure(self):
        """to_markdown should emit valid GitHub-flavored markdown."""
        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=100, seed=42)

        md = reporter.to_markdown(
            interactions,
            metadata={"seed": 42, "scenario": "baseline.yaml", "epochs": 10},
        )

        assert isinstance(md, str)
        assert md.startswith("# Metrics report")
        # Run metadata header rendered as bullets.
        assert "**seed:** 42" in md
        assert "**scenario:** baseline.yaml" in md
        assert "**epochs:** 10" in md
        # Section headings present.
        assert "## Soft metrics (probabilistic)" in md
        assert "## Hard metrics (threshold-based)" in md
        assert "## Counts" in md
        assert "## Welfare" in md
        # Valid table delimiters.
        assert "| Metric | Value |" in md
        assert "| --- | ---: |" in md
        # Every data row should have exactly two cells when respecting
        # pipe escaping (\| stays inside a cell).
        import re

        for line in md.splitlines():
            if line.startswith("|") and "Metric" not in line and "---" not in line:
                # Split only on unescaped pipes.
                parts = re.split(r"(?<!\\)\|", line)
                cells = [c for c in parts if c.strip()]
                assert len(cells) == 2, f"row has unexpected pipe count: {line!r}"

    def test_to_markdown_rounds_floats_to_four_decimals(self):
        """Floating-point cells should be rendered to 4 decimal places by default."""
        import re

        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=100, seed=42)

        md = reporter.to_markdown(interactions)

        float_cells = re.findall(r"\|\s*(-?\d+\.\d+)\s*\|", md)
        assert float_cells, "expected at least one floating-point cell"
        for cell in float_cells:
            decimals = cell.split(".")[1]
            assert len(decimals) == 4, f"cell {cell!r} not rounded to 4 decimals"

    def test_to_markdown_handles_empty_interactions(self):
        """to_markdown should not crash with an empty interaction list."""
        reporter = MetricsReporter()

        md = reporter.to_markdown([])

        assert "# Metrics report" in md
        assert "| Total interactions | 0 |" in md

    def test_to_markdown_metadata_optional(self):
        """to_markdown should omit the metadata block when none is supplied."""
        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=20, seed=7)

        md = reporter.to_markdown(interactions)

        assert "# Metrics report" in md
        assert "**seed:**" not in md


class TestWelfareMetrics:
    """Tests for welfare calculations."""

    def test_welfare_positive_for_good_interactions(self):
        """High quality accepted interactions should have positive welfare."""
        metrics = SoftMetrics()
        interactions = generate_benign_batch(count=50, seed=42)

        welfare = metrics.welfare_metrics(interactions)

        assert welfare["total_welfare"] > 0
        assert welfare["avg_initiator_payoff"] > 0

    def test_welfare_lower_for_toxic(self):
        """Toxic batch should have lower welfare than benign."""
        metrics = SoftMetrics()

        benign = generate_benign_batch(count=50, seed=42)
        toxic = generate_toxic_batch(count=50, seed=42)

        welfare_benign = metrics.welfare_metrics(benign)
        welfare_toxic = metrics.welfare_metrics(toxic)

        assert (
            welfare_benign["avg_initiator_payoff"]
            > welfare_toxic["avg_initiator_payoff"]
        )


class TestCalibrationMetrics:
    """Tests for calibration metrics."""

    def test_brier_score_perfect(self):
        """p matches ground_truth exactly -> Brier = 0."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=1.0, ground_truth=1),
            SoftInteraction(p=0.0, ground_truth=-1),
            SoftInteraction(p=1.0, ground_truth=1),
        ]

        brier = metrics.brier_score(interactions)
        assert brier == pytest.approx(0.0)

    def test_brier_score_random(self):
        """p = 0.5 always -> Brier = 0.25."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.5, ground_truth=1),
            SoftInteraction(p=0.5, ground_truth=-1),
            SoftInteraction(p=0.5, ground_truth=1),
            SoftInteraction(p=0.5, ground_truth=-1),
        ]

        brier = metrics.brier_score(interactions)
        assert brier == pytest.approx(0.25)

    def test_brier_score_no_ground_truth(self):
        """Returns None if no ground truth."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.7, ground_truth=None),
            SoftInteraction(p=0.3, ground_truth=None),
        ]

        assert metrics.brier_score(interactions) is None

    def test_calibration_error_wellcalibrated(self):
        """E[p] = empirical rate -> error = 0."""
        metrics = SoftMetrics()

        # 50% positive rate, average p = 0.5
        interactions = [
            SoftInteraction(p=0.6, ground_truth=1),
            SoftInteraction(p=0.4, ground_truth=-1),
            SoftInteraction(p=0.7, ground_truth=1),
            SoftInteraction(p=0.3, ground_truth=-1),
        ]

        error = metrics.calibration_error(interactions)
        # E[p] = 0.5, empirical rate = 0.5
        assert error == pytest.approx(0.0)

    def test_calibration_error_overconfident(self):
        """High p but low positive rate -> positive error."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.9, ground_truth=-1),
            SoftInteraction(p=0.8, ground_truth=-1),
            SoftInteraction(p=0.85, ground_truth=1),
        ]

        error = metrics.calibration_error(interactions)
        # E[p] = 0.85, empirical rate = 1/3 ≈ 0.33
        assert error > 0.4

    def test_ece_perfect(self):
        """Perfectly calibrated predictions -> ECE = 0."""
        metrics = SoftMetrics()

        # In each bin, p matches empirical rate
        interactions = []
        # Low bin: p around 0.2, 20% positive
        for _ in range(4):
            interactions.append(SoftInteraction(p=0.2, ground_truth=-1))
        interactions.append(SoftInteraction(p=0.2, ground_truth=1))

        # High bin: p around 0.8, 80% positive
        for _ in range(4):
            interactions.append(SoftInteraction(p=0.8, ground_truth=1))
        interactions.append(SoftInteraction(p=0.8, ground_truth=-1))

        ece = metrics.expected_calibration_error(interactions, bins=5)
        assert ece is not None
        assert ece < 0.05  # Should be very small

    def test_calibration_curve_structure(self):
        """Calibration curve should have correct structure."""
        metrics = SoftMetrics()

        interactions = generate_benign_batch(count=50, seed=42)
        curve = metrics.calibration_curve(interactions, bins=5)

        assert len(curve) == 5
        for mean_pred, frac_pos, count in curve:
            assert 0.0 <= mean_pred <= 1.0
            assert 0.0 <= frac_pos <= 1.0
            assert count >= 0


class TestInformationMetrics:
    """Tests for information-theoretic metrics."""

    def test_log_loss_bounds(self):
        """Log loss >= 0."""
        metrics = SoftMetrics()

        interactions = generate_benign_batch(count=50, seed=42)
        logloss = metrics.log_loss(interactions)

        assert logloss is not None
        assert logloss >= 0

    def test_log_loss_perfect(self):
        """Perfect predictions -> log_loss -> 0."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.999, ground_truth=1),
            SoftInteraction(p=0.001, ground_truth=-1),
        ]

        logloss = metrics.log_loss(interactions)
        assert logloss is not None
        assert logloss < 0.01

    def test_log_loss_random(self):
        """p = 0.5 always -> log_loss = ln(2) ≈ 0.693."""
        metrics = SoftMetrics()

        import math

        interactions = [
            SoftInteraction(p=0.5, ground_truth=1),
            SoftInteraction(p=0.5, ground_truth=-1),
        ]

        logloss = metrics.log_loss(interactions)
        assert logloss is not None
        assert logloss == pytest.approx(math.log(2), rel=0.01)

    def test_discrimination_auc_perfect(self):
        """Perfect discrimination -> AUC = 1."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.9, ground_truth=1),
            SoftInteraction(p=0.8, ground_truth=1),
            SoftInteraction(p=0.2, ground_truth=-1),
            SoftInteraction(p=0.1, ground_truth=-1),
        ]

        auc = metrics.discrimination_auc(interactions)
        assert auc is not None
        assert auc == 1.0

    def test_discrimination_auc_random(self):
        """Random predictions -> AUC ≈ 0.5."""
        metrics = SoftMetrics()

        # Interleaved positive and negative with similar p values
        interactions = [
            SoftInteraction(p=0.5, ground_truth=1),
            SoftInteraction(p=0.5, ground_truth=-1),
            SoftInteraction(p=0.5, ground_truth=1),
            SoftInteraction(p=0.5, ground_truth=-1),
        ]

        auc = metrics.discrimination_auc(interactions)
        assert auc is not None
        assert auc == pytest.approx(0.5)

    def test_discrimination_auc_missing_class(self):
        """Returns None if only one class present."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.9, ground_truth=1),
            SoftInteraction(p=0.8, ground_truth=1),
        ]

        assert metrics.discrimination_auc(interactions) is None


class TestVarianceMetrics:
    """Tests for variance/uncertainty metrics."""

    def test_variance_uniform(self):
        """Uniform p distribution -> known variance."""
        metrics = SoftMetrics()

        # p = 0.2, 0.4, 0.6, 0.8 -> mean = 0.5, var = 0.05
        interactions = [
            SoftInteraction(p=0.2),
            SoftInteraction(p=0.4),
            SoftInteraction(p=0.6),
            SoftInteraction(p=0.8),
        ]

        variance = metrics.quality_variance(interactions)
        # Var = E[(p - 0.5)^2] = (0.09 + 0.01 + 0.01 + 0.09) / 4 = 0.05
        assert variance == pytest.approx(0.05)

    def test_variance_concentrated(self):
        """All same p -> variance = 0."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.7),
            SoftInteraction(p=0.7),
            SoftInteraction(p=0.7),
        ]

        variance = metrics.quality_variance(interactions)
        assert variance == pytest.approx(0.0, abs=1e-10)

    def test_std_is_sqrt_variance(self):
        """Standard deviation = sqrt(variance)."""
        metrics = SoftMetrics()

        interactions = generate_mixed_batch(count=50, seed=42)

        variance = metrics.quality_variance(interactions)
        std = metrics.quality_std(interactions)

        import math

        assert std == pytest.approx(math.sqrt(variance))

    def test_payoff_variance_increases_with_spread(self):
        """More spread in p -> more variance in payoffs."""
        metrics = SoftMetrics()

        # Narrow spread
        narrow = [SoftInteraction(p=0.49 + i * 0.02) for i in range(5)]
        # Wide spread
        wide = [SoftInteraction(p=0.1 + i * 0.2) for i in range(5)]

        var_narrow = metrics.payoff_variance_initiator(narrow)
        var_wide = metrics.payoff_variance_initiator(wide)

        assert var_wide > var_narrow

    def test_coefficient_of_variation(self):
        """CV should be computed correctly."""
        metrics = SoftMetrics()

        interactions = generate_mixed_batch(count=50, seed=42)
        cv = metrics.coefficient_of_variation(interactions)

        assert "cv_p" in cv
        assert "cv_payoff_initiator" in cv
        assert "cv_payoff_counterparty" in cv
        assert cv["cv_p"] >= 0

    def test_cv_near_zero_mean_is_large(self):
        """CV should be large when mean is near zero but variance is high."""
        metrics = SoftMetrics()

        # With defaults, p=0.5 gives base initiator payoff 0.25.
        # r_a is set so payoffs are +1 and -1 (mean ~ 0, std = 1).
        interactions = [
            SoftInteraction(p=0.5, r_a=0.75),
            SoftInteraction(p=0.5, r_a=-1.25),
        ]

        cv = metrics.coefficient_of_variation(interactions)

        assert cv["cv_payoff_initiator"] > 1e6

    def test_accepted_only_variance(self):
        """Should compute variance only for accepted interactions."""
        metrics = SoftMetrics()

        interactions = [
            SoftInteraction(p=0.8, accepted=True),
            SoftInteraction(p=0.9, accepted=True),
            SoftInteraction(p=0.2, accepted=False),  # Should be excluded
        ]

        var_all = metrics.quality_variance(interactions, accepted_only=False)
        var_accepted = metrics.quality_variance(interactions, accepted_only=True)

        # Accepted only has less spread
        assert var_accepted < var_all


class TestMetricsSummaryWithNewMetrics:
    """Tests for MetricsSummary with new calibration and variance fields."""

    def test_summary_includes_calibration(self):
        """Summary should include calibration metrics."""
        reporter = MetricsReporter()
        interactions = generate_benign_batch(count=50, seed=42)

        summary = reporter.summary(interactions)

        # Benign batch has ground_truth set
        assert summary.brier_score is not None
        assert summary.log_loss is not None
        assert summary.calibration_error is not None

    def test_summary_includes_variance(self):
        """Summary should include variance metrics."""
        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=50, seed=42)

        summary = reporter.summary(interactions)

        assert hasattr(summary, "quality_variance")
        assert hasattr(summary, "payoff_variance_initiator")
        assert hasattr(summary, "payoff_variance_counterparty")
        assert summary.quality_variance > 0  # Mixed batch has variance

    def test_to_dict_includes_new_sections(self):
        """to_dict should include calibration and variance sections."""
        reporter = MetricsReporter()
        interactions = generate_mixed_batch(count=50, seed=42)

        summary = reporter.summary(interactions)
        d = summary.to_dict()

        assert "calibration" in d
        assert "variance" in d
        assert "brier_score" in d["calibration"]
        assert "quality_variance" in d["variance"]


class TestMetricsSummaryWithIncoherence:
    """Tests for replay-based incoherence fields in MetricsSummary."""

    def test_to_dict_includes_incoherence_section(self):
        reporter = MetricsReporter()
        summary = reporter.summary([SoftInteraction(p=0.7, accepted=True)])
        data = summary.to_dict()

        assert "incoherence" in data
        assert data["incoherence"]["index"] == 0.0
        assert data["incoherence"]["n_decisions"] == 0

    def test_summary_with_incoherence_aggregates_decisions(self):
        reporter = MetricsReporter()
        incoherence = IncoherenceMetrics(_SingleActionBenchmark())
        records_by_decision = {
            "d1": [
                DecisionRecord("d1", "task", 0, "approve"),
                DecisionRecord("d1", "task", 1, "reject"),
            ],
            "d2": [
                DecisionRecord("d2", "task", 0, "approve"),
                DecisionRecord("d2", "task", 1, "approve"),
            ],
        }

        summary = reporter.summary_with_incoherence(
            interactions=[SoftInteraction(p=0.7, accepted=True)],
            records_by_decision=records_by_decision,
            incoherence_metrics=incoherence,
        )

        assert summary.incoherence_n_decisions == 2
        assert 0.0 <= summary.incoherence_disagreement <= 1.0
        assert 0.0 <= summary.incoherence_error <= 1.0
        assert 0.0 <= summary.incoherence_index <= 1.0
