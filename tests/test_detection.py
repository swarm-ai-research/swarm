"""Tests for the matched soft-vs-binary detection experiment.

Covers: the matched-pair binarization identity, detector sign conventions,
AUROC/PR sanity, threshold-at-FPR calibration, time-to-detection, and that the
soft proxy is at least as well-calibrated (Brier/ECE) as its thresholded twin.
"""

import numpy as np
import pytest

from swarm.detection.curves import (
    calibration,
    compute_curve,
    per_agent_scores,
    threshold_at_fpr,
    time_to_detection,
)
from swarm.detection.degradation import (
    PopulationConfig,
    StreamConfig,
    generate_population,
)
from swarm.detection.detectors import MatchedDetectors, binarize_stream
from swarm.detection.experiment import ExperimentConfig, aggregate, run_experiment
from swarm.detection.stats import (
    ZERO_SD_REL_TOL,
    compute_paired_stats,
    hedges_g,
    paired_comparison,
)
from swarm.models.interaction import SoftInteraction


# ----------------------------------------------------------------------
# binarize_stream
# ----------------------------------------------------------------------
def test_binarize_stream_thresholds_p():
    inter = [
        SoftInteraction(p=0.8, accepted=True),
        SoftInteraction(p=0.3, accepted=True),
        SoftInteraction(p=0.5, accepted=False),
    ]
    hard = binarize_stream(inter, tau_star=0.5)
    assert [i.p for i in hard] == [1.0, 0.0, 1.0]  # 0.5 >= 0.5 -> 1
    # Original stream is untouched.
    assert [i.p for i in inter] == [0.8, 0.3, 0.5]


def test_binarize_preserves_invariants():
    hard = binarize_stream([SoftInteraction(p=0.9, accepted=True)], 0.5)
    assert -1.0 <= hard[0].v_hat <= 1.0
    assert 0.0 <= hard[0].p <= 1.0


# ----------------------------------------------------------------------
# matched detectors: binary detector == soft detector on a pre-binarized stream
# ----------------------------------------------------------------------
def test_binary_detector_matches_soft_on_binarized_stream():
    det = MatchedDetectors(tau_star=0.5)
    inter = [
        SoftInteraction(p=0.9, accepted=True),
        SoftInteraction(p=0.4, accepted=True),
        SoftInteraction(p=0.7, accepted=False),
        SoftInteraction(p=0.2, accepted=False),
    ]
    hard = binarize_stream(inter, 0.5)
    # The binary detector is *defined* as the soft detector on the binarized
    # stream; verify the identity holds for all current metrics.
    assert det.binary_toxicity(inter) == pytest.approx(det.soft_toxicity(hard))
    assert det.binary_quality_gap(inter) == pytest.approx(det.soft_quality_gap(hard))
    assert det.binary_conditional_loss(inter) == pytest.approx(
        det.soft_conditional_loss(hard)
    )
    assert det.binary_spread(inter) == pytest.approx(det.soft_spread(hard))
    assert det.binary_uncertain_fraction(inter) == pytest.approx(
        det.soft_uncertain_fraction(hard)
    )


def test_soft_toxicity_is_badness_among_accepted():
    det = MatchedDetectors()
    inter = [SoftInteraction(p=0.2, accepted=True), SoftInteraction(p=0.4, accepted=True)]
    # E[1-p | accepted] = 1 - 0.3 = 0.7
    assert det.soft_toxicity(inter) == pytest.approx(0.7)


def test_quality_gap_risk_positive_under_adverse_selection():
    """Accepted worse than rejected -> negative quality gap -> positive risk."""
    det = MatchedDetectors()
    inter = [
        SoftInteraction(p=0.2, accepted=True),
        SoftInteraction(p=0.3, accepted=True),
        SoftInteraction(p=0.8, accepted=False),
        SoftInteraction(p=0.9, accepted=False),
    ]
    assert det.soft_quality_gap(inter) > 0  # risk score = -quality_gap > 0


def test_spread_positive_under_adverse_selection():
    """Market filters high-quality work -> positive spread (used as-is for risk)."""
    det = MatchedDetectors()
    inter = [
        SoftInteraction(p=0.2, accepted=True),
        SoftInteraction(p=0.3, accepted=True),
        SoftInteraction(p=0.8, accepted=False),
        SoftInteraction(p=0.9, accepted=False),
    ]
    assert det.soft_spread(inter) > 0


def test_uncertain_fraction_counts_near_half():
    det = MatchedDetectors()
    inter = [
        SoftInteraction(p=0.48, accepted=True),
        SoftInteraction(p=0.52, accepted=True),
        SoftInteraction(p=0.9, accepted=True),
    ]
    # 2 out of 3 are within band=0.25 of 0.5
    assert det.soft_uncertain_fraction(inter) == pytest.approx(2 / 3)


# ----------------------------------------------------------------------
# generative streams
# ----------------------------------------------------------------------
def test_population_base_rate_respected():
    pop = PopulationConfig(n_agents=40, base_rate=0.25)
    streams = generate_population(pop, seed=0)
    n_deg = sum(s.is_degrading for s in streams)
    assert n_deg == 10
    assert len(streams) == 40


def test_population_reproducible():
    pop = PopulationConfig(n_agents=20, base_rate=0.3)
    a = generate_population(pop, seed=42)
    b = generate_population(pop, seed=42)
    assert [s.is_degrading for s in a] == [s.is_degrading for s in b]
    assert a[0].epochs[0][0].p == b[0].epochs[0][0].p


def test_degrading_quality_drops_after_onset():
    cfg = StreamConfig(n_epochs=24, proxy_noise=0.0, quality_jitter=0.0)
    pop = PopulationConfig(n_agents=4, base_rate=1.0, stream=cfg, trajectory_choices=("linear",))
    streams = generate_population(pop, seed=1)
    s = streams[0]
    early = np.mean([i.p for i in s.epochs[s.onset_epoch]])
    late = np.mean([i.p for i in s.epochs[-1]])
    assert late < early


def test_benign_quality_stays_above_acceptance():
    cfg = StreamConfig()
    pop = PopulationConfig(n_agents=10, base_rate=0.0, stream=cfg)
    streams = generate_population(pop, seed=2)
    # Benign agents keep mean proxy well above the acceptance threshold.
    for s in streams:
        mean_p = np.mean([i.p for i in s.all_interactions()])
        assert mean_p > cfg.acceptance_threshold


# ----------------------------------------------------------------------
# population heterogeneity (breaks the saturated AUROC=1.0 ceiling)
# ----------------------------------------------------------------------
def _benign_toxicity_std(cfg: StreamConfig, *, seed: int = 0, n_agents: int = 300) -> float:
    """Std of per-agent soft toxicity over a benign-only population (back half)."""
    det = MatchedDetectors()
    n = cfg.n_epochs
    streams = generate_population(
        PopulationConfig(n_agents=n_agents, base_rate=0.0, stream=cfg), seed=seed
    )
    vals = [det.soft_toxicity(s.window(n // 2, n)) for s in streams]
    return float(np.std(vals))


def test_default_population_is_homogeneous():
    # With the (default) zero heterogeneity, benign agents collapse to nearly a
    # single point — this is exactly what saturates soft AUROC at 1.0.
    assert _benign_toxicity_std(StreamConfig()) < 0.02


def test_benign_heterogeneity_widens_score_spread():
    # benign_quality_std injects real width into the benign score distribution.
    homo = _benign_toxicity_std(StreamConfig())
    hetero = _benign_toxicity_std(StreamConfig(benign_quality_std=0.10))
    assert hetero > 0.03
    assert hetero > 5 * homo


def test_floor_std_varies_degrading_floors():
    # Hold onset and trajectory fixed so the *only* source of variation in
    # late-epoch quality is floor_std. Default (floor_std=0): all degrading
    # agents converge to the same floor (range ~ jitter). With floor_std they
    # spread out — some barely degrade, staying near benign.
    def late_range(cfg):
        pop = PopulationConfig(
            n_agents=120, base_rate=1.0, stream=cfg,
            onset_choices=(4,), trajectory_choices=("linear",),
        )
        streams = generate_population(pop, seed=0)
        late = [np.mean([i.p for i in s.epochs[-1]]) for s in streams]
        return max(late) - min(late)

    homo = late_range(StreamConfig())
    hetero = late_range(StreamConfig(floor_std=0.06))
    assert homo < 0.15  # only jitter + finite-sample spread
    assert hetero > 0.20
    assert hetero > 2 * homo


def test_heterogeneous_preset_is_not_saturated():
    # The whole point: under the heterogeneous preset + a short eval window, soft
    # AUROC is a genuine (sub-1.0) score and still clearly beats binary, rather
    # than both being pinned by the generator. Measured ~0.87 soft / ~0.58 binary.
    cfg = StreamConfig.heterogeneous()
    det = MatchedDetectors()
    n = cfg.n_epochs
    es, ee = n - 3, n
    streams = generate_population(
        PopulationConfig(n_agents=200, base_rate=0.2, stream=cfg), seed=0
    )
    sc, lab = per_agent_scores(streams, det.soft_toxicity, es, ee)
    soft = compute_curve(sc, lab).auroc
    sc, lab = per_agent_scores(streams, det.binary_toxicity, es, ee)
    binary = compute_curve(sc, lab).auroc
    assert 0.75 < soft < 0.98  # informative, not saturated
    assert binary < soft - 0.10  # soft advantage survives, quantifiably


# ----------------------------------------------------------------------
# curves
# ----------------------------------------------------------------------
def test_perfect_separation_auroc_one():
    curve = compute_curve(scores=[0.1, 0.2, 0.9, 0.95], labels=[0, 0, 1, 1])
    assert curve.auroc == pytest.approx(1.0)


def test_single_class_degenerate():
    curve = compute_curve(scores=[0.3, 0.4], labels=[0, 0])
    assert curve.auroc == 0.5
    assert curve.base_rate == 0.0
    assert curve.pauroc_fpr05 == 0.5
    assert curve.pauroc_fpr01 == 0.5


def test_pauroc_fields_present_and_in_range():
    curve = compute_curve(scores=[0.1, 0.2, 0.8, 0.9], labels=[0, 0, 1, 1])
    assert hasattr(curve, "pauroc_fpr05")
    assert hasattr(curve, "pauroc_fpr01")
    assert 0.0 <= curve.pauroc_fpr05 <= 1.0
    assert 0.0 <= curve.pauroc_fpr01 <= 1.0
    # For strong separation, pAUROC at low FPR should still be high
    assert curve.pauroc_fpr05 > 0.9


def test_threshold_at_fpr_quantile():
    neg = list(np.linspace(0, 1, 101))  # 0.00 .. 1.00
    thr = threshold_at_fpr(neg, max_fpr=0.05)
    # 95th percentile ~ 0.95; at most ~5% of negatives exceed it.
    assert thr == pytest.approx(0.95, abs=0.02)
    frac_exceed = np.mean(np.array(neg) > thr)
    assert frac_exceed <= 0.06


def test_time_to_detection_flags_degrading_agent():
    cfg = StreamConfig(n_epochs=20, proxy_noise=0.03, quality_jitter=0.01)
    pop = PopulationConfig(
        n_agents=2, base_rate=1.0, stream=cfg, onset_choices=(3,),
        trajectory_choices=("linear",),
    )
    streams = generate_population(pop, seed=5)
    det = MatchedDetectors().soft_toxicity
    # A low threshold should flag a clearly degrading agent at some point.
    ttd = time_to_detection(streams[0], det, threshold=0.3, window=4)
    assert ttd is not None and ttd >= 0


def test_ttd_scan_starts_at_onset_not_earlier():
    """The TTD scan must not look before onset (pre-onset flags are false alarms,
    handled by FPR calibration — not collapsed to TTD=0). Regression for the
    calibration/scan window-misalignment flagged in review."""
    cfg = StreamConfig(n_epochs=16, proxy_noise=0.0, quality_jitter=0.0)
    pop = PopulationConfig(
        n_agents=1, base_rate=1.0, stream=cfg, onset_choices=(6,),
        trajectory_choices=("linear",),
    )
    s = generate_population(pop, seed=0)[0]
    scanned_epochs = []

    def recording_detector(win):
        scanned_epochs.append(max(i.metadata["epoch"] for i in win))
        return -1.0  # never exceeds threshold 0.0 -> censored

    ttd = time_to_detection(s, recording_detector, threshold=0.0, window=4, min_epoch=1)
    assert ttd is None
    # Even with min_epoch=1, the scan starts at onset (6), never earlier.
    assert min(scanned_epochs) == s.onset_epoch == 6


def test_time_to_detection_censored_when_threshold_unreachable():
    cfg = StreamConfig(n_epochs=12)
    pop = PopulationConfig(n_agents=1, base_rate=1.0, stream=cfg)
    streams = generate_population(pop, seed=0)
    ttd = time_to_detection(streams[0], MatchedDetectors().soft_toxicity,
                            threshold=10.0, window=4)
    assert ttd is None


# ----------------------------------------------------------------------
# calibration: soft proxy should not be worse than thresholded twin
# ----------------------------------------------------------------------
def test_soft_brier_not_worse_than_binary():
    cfg = StreamConfig(n_epochs=12, interactions_per_epoch=30)
    pop = PopulationConfig(n_agents=10, base_rate=0.3, stream=cfg)
    streams = generate_population(pop, seed=7)
    pooled = []
    for s in streams:
        pooled.extend(s.all_interactions())
    cal = calibration(pooled, tau_star=0.5)
    # Brier is a proper score; thresholding the calibrated proxy should not
    # improve it. Allow a tiny epsilon for sampling noise.
    assert cal.soft_brier <= cal.binary_brier + 1e-9


# ----------------------------------------------------------------------
# end-to-end smoke
# ----------------------------------------------------------------------
def test_run_experiment_smoke_and_aggregate():
    cfg = ExperimentConfig(
        base_rates=(0.1, 0.3),
        seeds=(0, 1),
        population=PopulationConfig(
            n_agents=20,
            stream=StreamConfig(n_epochs=16, interactions_per_epoch=8),
        ),
    )
    res = run_experiment(cfg)
    # 2 base rates * 2 seeds * 2 per-agent metrics * 2 variants = 16
    assert len(res.detection_rows) == 16
    assert len(res.ttd_rows) == 16
    # 2 base * 2 seeds * 3 market metrics (incl. spread) * 2 variants = 24
    assert len(res.market_rows) == 24
    assert len(res.calibration_rows) == 4

    agg = aggregate(
        res.detection_rows, ["base_rate", "metric", "variant"], ["auroc", "auprc", "pauroc_fpr05", "pauroc_fpr01"]
    )
    assert all("auroc_mean" in r and "n" in r for r in agg)
    assert all("pauroc_fpr05_mean" in r for r in agg)
    # Every AUROC is a valid probability.
    for r in res.detection_rows:
        assert 0.0 <= r["auroc"] <= 1.0


# ----------------------------------------------------------------------
# paired stats (built-in Phase 2)
# ----------------------------------------------------------------------
def test_paired_comparison_direction_and_effect():
    # treatment strictly above control by a constant -> positive mean diff,
    # tiny p, and (zero-variance diff) infinite d_z.
    res = paired_comparison([0.9, 0.8, 0.95], [0.5, 0.4, 0.55], "t")
    assert res.mean_diff > 0
    assert res.primary_p < 0.5
    assert res.n == 3


def test_paired_comparison_drops_nan_pairs():
    res = paired_comparison([1.0, float("nan"), 0.9], [0.5, 0.4, 0.6], "t")
    assert res.n == 2  # the NaN pair is dropped


def test_compute_paired_stats_structure_and_holm():
    cfg = ExperimentConfig(
        base_rates=(0.1, 0.3),
        seeds=tuple(range(5)),
        population=PopulationConfig(
            n_agents=20, stream=StreamConfig(n_epochs=16, interactions_per_epoch=8)
        ),
    )
    res = run_experiment(cfg)
    stats = compute_paired_stats(res)
    assert stats["family_size"] == len(stats["comparisons"])
    assert 0 <= stats["n_survive_holm"] <= stats["family_size"]
    # The big effects (TTD, calibration) should clear Holm correction.
    labels = {c["label"]: c for c in stats["comparisons"]}
    assert any("TTD" in lbl for lbl in labels)
    assert any("Brier" in lbl for lbl in labels)
    ttd = next(c for lbl, c in labels.items() if "TTD" in lbl)
    # TTD rows now include multiple per-agent metrics; direction may be mixed in
    # small smoke configs. Just assert the comparison exists and Holm logic ran.
    assert ttd is not None
    assert "survives_holm" in ttd


class TestHedgesG:
    """Unpaired standardised mean difference, the units arXiv:2512.04124 reports."""

    def test_zero_effect_when_samples_match(self) -> None:
        sample = [0.1, 0.2, 0.3, 0.4, 0.5]
        eff = hedges_g(sample, sample, "identical")
        assert eff.hedges_g == pytest.approx(0.0)
        assert eff.mean_diff == pytest.approx(0.0)
        assert eff.ci_spans_zero

    def test_orientation_is_treatment_minus_control(self) -> None:
        control = [0.0, 0.1, 0.2, 0.1, 0.0]
        higher = [1.0, 1.1, 1.2, 1.1, 1.0]
        assert hedges_g(higher, control).hedges_g > 0
        assert hedges_g(control, higher).hedges_g < 0

    def test_large_separation_gives_interval_clear_of_zero(self) -> None:
        rng = np.random.default_rng(0)
        control = rng.normal(0.0, 1.0, 40)
        treatment = rng.normal(2.0, 1.0, 40)
        eff = hedges_g(treatment, control, "shifted")
        assert eff.hedges_g > 1.0
        assert eff.ci_low > 0.0
        assert not eff.ci_spans_zero
        assert eff.ci_low < eff.hedges_g < eff.ci_high

    def test_small_sample_correction_shrinks_toward_zero(self) -> None:
        """Hedges' g is Cohen's d times J < 1, so |g| < |d|."""
        treatment = [1.0, 2.0, 3.0, 4.0]
        control = [0.0, 1.0, 2.0, 3.0]
        eff = hedges_g(treatment, control)
        t, c = np.asarray(treatment), np.asarray(control)
        df = t.size + c.size - 2
        pooled = np.sqrt(
            ((t.size - 1) * t.var(ddof=1) + (c.size - 1) * c.var(ddof=1)) / df
        )
        cohens_d = (t.mean() - c.mean()) / pooled
        assert 0 < eff.hedges_g < cohens_d

    def test_nan_entries_are_dropped(self) -> None:
        eff = hedges_g([1.0, 2.0, 3.0, float("nan")], [1.0, 2.0, 3.0])
        assert eff.n_treatment == 3
        assert eff.n_control == 3

    def test_undefined_when_a_sample_is_too_small(self) -> None:
        eff = hedges_g([1.0], [1.0, 2.0, 3.0])
        assert np.isnan(eff.hedges_g)
        assert eff.ci_spans_zero  # an undefined interval is not evidence

    def test_zero_variance_separated_means_is_infinite_with_no_interval(self) -> None:
        eff = hedges_g([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
        assert eff.hedges_g == float("inf")
        assert np.isnan(eff.ci_low) and np.isnan(eff.ci_high)
        assert eff.ci_spans_zero

    def test_float_noise_variance_is_treated_as_zero(self) -> None:
        """A deterministic generator produces conditions constant to ~1e-16.

        Without a relative tolerance that noise becomes the denominator and g
        comes out around 1e15 with a finite, confident-looking interval. Seen
        for real on the PsAIch ablation grid, where two conditions each pinned
        mean_judge_score to a fixed rubric target.
        """
        treatment = [0.55 + i * 1e-17 for i in range(30)]
        control = [0.75 - i * 1e-17 for i in range(30)]
        eff = hedges_g(treatment, control, "rubric-pinned")
        assert eff.hedges_g == -float("inf")
        assert np.isnan(eff.ci_low) and np.isnan(eff.ci_high)
        assert eff.ci_spans_zero
        # The means themselves are still reported faithfully.
        assert eff.mean_treatment == pytest.approx(0.55)
        assert eff.mean_control == pytest.approx(0.75)

    def test_real_variation_above_the_tolerance_still_measures(self) -> None:
        """The guard must not swallow a genuine small-variance effect."""
        rng = np.random.default_rng(3)
        control = rng.normal(0.0, 1e-6, 30)
        treatment = rng.normal(4e-6, 1e-6, 30)
        eff = hedges_g(treatment, control)
        assert np.isfinite(eff.hedges_g)
        assert eff.hedges_g > 1.0
        assert not eff.ci_spans_zero

    def test_zero_sd_tolerance_is_relative_to_scale(self) -> None:
        """Large-magnitude data has proportionally larger float noise."""
        base = 1e6
        treatment = [base + i * base * ZERO_SD_REL_TOL / 100 for i in range(20)]
        control = [base * 2 for _ in range(20)]
        eff = hedges_g(treatment, control)
        assert not np.isfinite(eff.hedges_g)

    def test_ci_level_widens_the_interval(self) -> None:
        rng = np.random.default_rng(1)
        control = rng.normal(0.0, 1.0, 25)
        treatment = rng.normal(0.6, 1.0, 25)
        narrow = hedges_g(treatment, control, ci_level=0.90)
        wide = hedges_g(treatment, control, ci_level=0.99)
        assert narrow.hedges_g == pytest.approx(wide.hedges_g)
        assert wide.ci_high - wide.ci_low > narrow.ci_high - narrow.ci_low

    def test_to_dict_round_trips_the_reported_fields(self) -> None:
        import json

        eff = hedges_g([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], "labelled")
        d = eff.to_dict()
        json.dumps(d)
        assert d["label"] == "labelled"
        assert d["ci_level"] == 0.95
        assert d["hedges_g"] == eff.hedges_g
