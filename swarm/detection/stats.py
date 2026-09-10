"""Paired soft-vs-binary significance testing for the detection experiment.

Soft and binary detectors are evaluated on the *same* interaction streams, so
their per-seed scores are **paired**. This module computes, for each headline
comparison, the mean difference, a Wilcoxon signed-rank test (with a paired
t-test as a parametric companion), Cohen's d_z effect size, and a
Holm-Bonferroni correction across the whole family. It is the built-in form of
``/full_study --detection`` Phase 2.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as st


@dataclass
class PairedComparison:
    label: str
    n: int
    mean_diff: float
    cohen_dz: float
    t_p: float
    wilcoxon_p: float
    holm_threshold: float = float("nan")
    survives_holm: bool = False

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "mean_diff": self.mean_diff,
            "cohen_dz": self.cohen_dz,
            "t_p": self.t_p,
            "wilcoxon_p": self.wilcoxon_p,
            "holm_threshold": self.holm_threshold,
            "survives_holm": self.survives_holm,
        }

    @property
    def primary_p(self) -> float:
        """Wilcoxon p, falling back to the paired-t p when Wilcoxon is undefined."""
        return self.t_p if np.isnan(self.wilcoxon_p) else self.wilcoxon_p


def paired_comparison(
    treatment: Sequence[float], control: Sequence[float], label: str
) -> PairedComparison:
    """Compare two paired samples (``treatment - control``).

    ``treatment`` is oriented so that a *positive* mean difference is the
    "expected" direction (e.g. soft AUROC − binary AUROC, or binary TTD − soft
    TTD so positive means soft is faster). Pairs with NaN on either side are
    dropped.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    mask = ~(np.isnan(t) | np.isnan(c))
    t, c = t[mask], c[mask]
    diff = t - c
    n = int(diff.size)
    mean_d = float(diff.mean()) if n else 0.0

    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        dz = mean_d / sd
        t_p = float(st.ttest_rel(t, c).pvalue)
    else:
        dz = float("inf") if mean_d != 0 else 0.0
        t_p = 0.0 if mean_d != 0 else 1.0

    if n and np.any(diff != 0):
        try:
            w_p = float(st.wilcoxon(t, c).pvalue)
        except ValueError:
            w_p = float("nan")
    else:
        w_p = 1.0

    return PairedComparison(
        label=label, n=n, mean_diff=mean_d, cohen_dz=dz, t_p=t_p, wilcoxon_p=w_p
    )


# Pooled SDs at or below this fraction of the data scale are treated as zero.
# Two conditions driven by a deterministic judge are routinely constant to
# within float noise; without this guard the noise becomes the denominator.
ZERO_SD_REL_TOL = 1e-12


@dataclass
class EffectSize:
    """Unpaired standardised mean difference with a confidence interval."""

    label: str
    n_treatment: int
    n_control: int
    mean_treatment: float
    mean_control: float
    mean_diff: float
    hedges_g: float
    ci_low: float
    ci_high: float
    ci_level: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_treatment": self.n_treatment,
            "n_control": self.n_control,
            "mean_treatment": self.mean_treatment,
            "mean_control": self.mean_control,
            "mean_diff": self.mean_diff,
            "hedges_g": self.hedges_g,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
        }

    @property
    def ci_spans_zero(self) -> bool:
        """True when the interval admits no effect (or is undefined)."""
        if np.isnan(self.ci_low) or np.isnan(self.ci_high):
            return True
        return self.ci_low <= 0.0 <= self.ci_high


def hedges_g(
    treatment: Sequence[float],
    control: Sequence[float],
    label: str = "",
    *,
    ci_level: float = 0.95,
) -> EffectSize:
    """Hedges' g for two independent samples, oriented ``treatment - control``.

    This is the effect-size convention used for perturbation comparisons in
    arXiv:2512.04124 (pooled outcome density, Hedges' g with a 95% CI), so an
    ablation condition measured here can be reported next to that paper's
    g = 0.13, CI [-0.15, 0.41]. ``paired_comparison`` above is the right tool
    when the two samples share seeds; this one is for conditions that do not.

    g = J * (mean_t - mean_c) / s_pooled, with the small-sample correction
    J = 1 - 3 / (4 * (n_t + n_c - 2) - 1). The interval is the usual
    normal approximation on var(g) = (n_t + n_c) / (n_t * n_c)
    + g**2 / (2 * (n_t + n_c - 2)).

    NaN entries are dropped. When the pooled SD is zero the effect size is
    undefined: g is 0.0 for identical means and +/-inf otherwise, with a NaN
    interval in both cases (so ``ci_spans_zero`` reads True — an undefined
    interval is not evidence of an effect). "Zero" here is relative to the data
    scale (see ``ZERO_SD_REL_TOL``), because a deterministic generator makes
    conditions that are constant to within float noise ordinary, and dividing
    by that noise yields an astronomically large g with a finite,
    confident-looking interval.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    t = t[~np.isnan(t)]
    c = c[~np.isnan(c)]
    n_t, n_c = int(t.size), int(c.size)
    mean_t = float(t.mean()) if n_t else float("nan")
    mean_c = float(c.mean()) if n_c else float("nan")
    mean_d = mean_t - mean_c if (n_t and n_c) else float("nan")

    df = n_t + n_c - 2
    if n_t < 2 or n_c < 2 or df < 1:
        return EffectSize(
            label=label, n_treatment=n_t, n_control=n_c,
            mean_treatment=mean_t, mean_control=mean_c, mean_diff=mean_d,
            hedges_g=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            ci_level=ci_level,
        )

    var_t = float(t.var(ddof=1))
    var_c = float(c.var(ddof=1))
    s_pooled = float(np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / df))
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)

    scale = max(1.0, abs(mean_t), abs(mean_c))
    if s_pooled <= ZERO_SD_REL_TOL * scale:
        g = 0.0 if mean_d == 0 else float(np.sign(mean_d)) * float("inf")
        return EffectSize(
            label=label, n_treatment=n_t, n_control=n_c,
            mean_treatment=mean_t, mean_control=mean_c, mean_diff=mean_d,
            hedges_g=g, ci_low=float("nan"), ci_high=float("nan"),
            ci_level=ci_level,
        )

    g = correction * mean_d / s_pooled
    var_g = (n_t + n_c) / (n_t * n_c) + g**2 / (2.0 * df)
    z = float(st.norm.ppf(0.5 + ci_level / 2.0))
    half = z * float(np.sqrt(var_g))
    return EffectSize(
        label=label, n_treatment=n_t, n_control=n_c,
        mean_treatment=mean_t, mean_control=mean_c, mean_diff=mean_d,
        hedges_g=g, ci_low=g - half, ci_high=g + half, ci_level=ci_level,
    )


def _holm_bonferroni(comparisons: List[PairedComparison], alpha: float = 0.05) -> int:
    """Annotate ``survives_holm``/``holm_threshold`` in place; return survivor count.

    Standard step-down Holm: sort ascending by p, reject while
    ``p_(k) <= alpha / (m - k)``, stop at the first failure.
    """
    m = len(comparisons)
    order = sorted(range(m), key=lambda i: comparisons[i].primary_p)
    still_rejecting = True
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        comparisons[idx].holm_threshold = thresh
        if still_rejecting and comparisons[idx].primary_p <= thresh:
            comparisons[idx].survives_holm = True
        else:
            still_rejecting = False
            comparisons[idx].survives_holm = False
    return sum(c.survives_holm for c in comparisons)


def _by_seed(rows: Sequence[dict], key: str, group: Sequence[str]) -> Dict[tuple, dict]:
    """Index ``rows`` as ``{group_key: {variant: {seed: value}}}`` for pairing."""
    out: Dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        gk = tuple(r[g] for g in group)
        out[gk][r["variant"]][r["seed"]] = r[key]
    return out


def _aligned(variant_map: dict) -> tuple[list, list]:
    """Return (soft, binary) value lists aligned on shared seeds."""
    seeds = sorted(set(variant_map.get("soft", {})) & set(variant_map.get("binary", {})))
    soft = [variant_map["soft"][s] for s in seeds]
    binary = [variant_map["binary"][s] for s in seeds]
    return soft, binary


def compute_paired_stats(results, alpha: float = 0.05) -> dict:
    """Build the full paired-comparison family from an ``ExperimentResults``.

    Returns a JSON-ready dict with the method description, family size, survivor
    count, and per-comparison records (matching ``/full_study`` Phase 2).
    """
    comps: List[PairedComparison] = []

    # AUROC (soft - binary) per (base rate, detector metric). Grouping by metric
    # is required now that multiple per-agent detectors (toxicity,
    # uncertain_fraction) emit a detection row per (base_rate, seed, variant);
    # without it, _by_seed would overwrite one metric with the other per seed.
    auroc = _by_seed(results.detection_rows, "auroc", ["base_rate", "metric"])
    for (br, metric), vmap in sorted(auroc.items()):
        soft, binary = _aligned(vmap)
        comps.append(
            paired_comparison(soft, binary, f"AUROC soft-binary @ br={br:.2f} [{metric}]")
        )

    # Time-to-detection (binary - soft; positive => soft faster), pooled over
    # base rates but kept separate per detector metric (same overwrite hazard).
    ttd = _by_seed(results.ttd_rows, "median_ttd", ["base_rate", "metric"])
    dr = _by_seed(results.ttd_rows, "detection_rate", ["base_rate", "metric"])
    ttd_by_metric: Dict[str, dict] = defaultdict(lambda: {"soft": [], "binary": []})
    dr_by_metric: Dict[str, dict] = defaultdict(lambda: {"soft": [], "binary": []})
    for (_br, metric), vmap in ttd.items():
        s, b = _aligned(vmap)
        ttd_by_metric[metric]["soft"].extend(s)
        ttd_by_metric[metric]["binary"].extend(b)
    for (_br, metric), vmap in dr.items():
        s, b = _aligned(vmap)
        dr_by_metric[metric]["soft"].extend(s)
        dr_by_metric[metric]["binary"].extend(b)
    # median_ttd can be None (censored); coerce to NaN so pairs drop cleanly.
    for metric in sorted(ttd_by_metric):
        soft_ttd = [np.nan if v is None else v for v in ttd_by_metric[metric]["soft"]]
        bin_ttd = [np.nan if v is None else v for v in ttd_by_metric[metric]["binary"]]
        comps.append(
            paired_comparison(
                bin_ttd, soft_ttd, f"TTD binary-soft (epochs; +=soft faster) [{metric}]"
            )
        )
    for metric in sorted(dr_by_metric):
        comps.append(
            paired_comparison(
                dr_by_metric[metric]["soft"],
                dr_by_metric[metric]["binary"],
                f"Detection-rate soft-binary (+=soft catches more) [{metric}]",
            )
        )

    # Market |signal| (soft vs binary) per metric, pooled over base rates.
    mkt = _by_seed(results.market_rows, "value", ["metric", "base_rate"])
    per_metric: Dict[str, dict] = defaultdict(lambda: {"soft": [], "binary": []})
    for (metric, _br), vmap in mkt.items():
        s, b = _aligned(vmap)
        per_metric[metric]["soft"].extend(abs(v) for v in s)
        per_metric[metric]["binary"].extend(abs(v) for v in b)
    for metric in sorted(per_metric):
        comps.append(
            paired_comparison(
                per_metric[metric]["soft"],
                per_metric[metric]["binary"],
                f"Market |signal| soft-binary: {metric}",
            )
        )

    # Calibration (binary - soft; positive => soft better, since lower is better).
    for field, lbl in (("brier", "Brier"), ("ece", "ECE")):
        soft = [r[f"soft_{field}"] for r in results.calibration_rows]
        binary = [r[f"binary_{field}"] for r in results.calibration_rows]
        comps.append(paired_comparison(binary, soft, f"{lbl} binary-soft (+=soft better)"))

    n_survive = _holm_bonferroni(comps, alpha=alpha)
    return {
        "method": (
            "paired (soft vs binary on identical streams); Wilcoxon signed-rank "
            "+ paired t; Cohen's d_z; Holm-Bonferroni"
        ),
        "alpha": alpha,
        "family_size": len(comps),
        "n_survive_holm": n_survive,
        "comparisons": [c.to_dict() for c in comps],
    }
