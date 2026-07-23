#!/usr/bin/env python
"""Statistical analysis for AI Economist GTB multi-seed sweep.

Runs 6 pre-registered hypothesis tests with Bonferroni correction,
plus descriptive stats and cross-type ANOVA.

Usage:
    python examples/ai_economist_sweep_stats.py runs/<sweep_dir>/
    python examples/ai_economist_sweep_stats.py runs/<sweep_dir>/ --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp


# ---------------------------------------------------------------------------
# Effect-size helpers (from sweep_stats.py pattern)
# ---------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Approximate power for two-sample t-test."""
    if n < 2 or np.isnan(effect_size):
        return float("nan")
    se = np.sqrt(2.0 / n)
    z_alpha = sp.norm.ppf(1 - alpha / 2)
    z_power = abs(effect_size) / se - z_alpha
    return float(sp.norm.cdf(z_power))


def d_interpretation(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Descriptive stats
# ---------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> dict:
    """Mean ± SD for all numeric columns."""
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        stats[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(len(vals)),
        }
    return stats


def normality_checks(df: pd.DataFrame, key_cols: list[str]) -> dict:
    """Shapiro-Wilk normality test for key columns."""
    results = {}
    for col in key_cols:
        vals = df[col].dropna().values
        if len(vals) < 3:
            results[col] = {"W": float("nan"), "p_value": float("nan"), "normal": False}
            continue
        W, p = sp.shapiro(vals)
        results[col] = {"W": float(W), "p_value": float(p), "normal": p > 0.05}
    return results


# ---------------------------------------------------------------------------
# Pre-registered hypotheses
# ---------------------------------------------------------------------------
BONFERRONI_ALPHA = 0.05 / 6  # 0.00833


def test_h1_progressive_tax(df: pd.DataFrame) -> dict:
    """H1: Tax schedule is progressive (progressivity index > 0)."""
    vals = df["progressivity_index"].dropna().values
    t, p = sp.ttest_1samp(vals, 0)
    # One-sided: progressivity > 0
    p_one = p / 2 if t > 0 else 1 - p / 2
    return {
        "hypothesis": "H1: Progressivity index > 0 (emergent progressive taxation)",
        "test": "One-sample t-test (one-sided)",
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "t": float(t),
        "p_value": float(p_one),
        "significant": p_one < BONFERRONI_ALPHA,
        "alpha": BONFERRONI_ALPHA,
        "n": len(vals),
    }


def test_h2_honest_vs_collusive(df: pd.DataFrame) -> dict:
    """H2: Honest wealth > collusive wealth."""
    a = df["honest_mean_wealth"].dropna().values
    b = df["collusive_mean_wealth"].dropna().values
    t, p = sp.ttest_ind(a, b, equal_var=False)
    p_one = p / 2 if t > 0 else 1 - p / 2
    d = cohens_d(a, b)
    return {
        "hypothesis": "H2: Honest wealth > collusive wealth (collusion failure)",
        "test": "Welch's t-test (one-sided)",
        "honest_mean": float(a.mean()),
        "honest_std": float(a.std()),
        "collusive_mean": float(b.mean()),
        "collusive_std": float(b.std()),
        "t": float(t),
        "p_value": float(p_one),
        "cohens_d": float(d),
        "d_interpretation": d_interpretation(d),
        "significant": p_one < BONFERRONI_ALPHA,
        "alpha": BONFERRONI_ALPHA,
        "power": power_analysis(d, min(len(a), len(b))),
    }


def test_h3_honest_vs_evasive(df: pd.DataFrame) -> dict:
    """H3: Honest wealth > evasive wealth."""
    a = df["honest_mean_wealth"].dropna().values
    b = df["evasive_mean_wealth"].dropna().values
    t, p = sp.ttest_ind(a, b, equal_var=False)
    p_one = p / 2 if t > 0 else 1 - p / 2
    d = cohens_d(a, b)
    return {
        "hypothesis": "H3: Honest wealth > evasive wealth (cost of evasion)",
        "test": "Welch's t-test (one-sided)",
        "honest_mean": float(a.mean()),
        "honest_std": float(a.std()),
        "evasive_mean": float(b.mean()),
        "evasive_std": float(b.std()),
        "t": float(t),
        "p_value": float(p_one),
        "cohens_d": float(d),
        "d_interpretation": d_interpretation(d),
        "significant": p_one < BONFERRONI_ALPHA,
        "alpha": BONFERRONI_ALPHA,
        "power": power_analysis(d, min(len(a), len(b))),
    }


def test_h4_evasive_vs_collusive(df: pd.DataFrame) -> dict:
    """H4: Evasive wealth > collusive wealth."""
    a = df["evasive_mean_wealth"].dropna().values
    b = df["collusive_mean_wealth"].dropna().values
    t, p = sp.ttest_ind(a, b, equal_var=False)
    p_one = p / 2 if t > 0 else 1 - p / 2
    d = cohens_d(a, b)
    return {
        "hypothesis": "H4: Evasive wealth > collusive wealth",
        "test": "Welch's t-test (one-sided)",
        "evasive_mean": float(a.mean()),
        "evasive_std": float(a.std()),
        "collusive_mean": float(b.mean()),
        "collusive_std": float(b.std()),
        "t": float(t),
        "p_value": float(p_one),
        "cohens_d": float(d),
        "d_interpretation": d_interpretation(d),
        "significant": p_one < BONFERRONI_ALPHA,
        "alpha": BONFERRONI_ALPHA,
        "power": power_analysis(d, min(len(a), len(b))),
    }


def test_h5_gini_below_half(df: pd.DataFrame) -> dict:
    """H5: Final Gini < 0.5."""
    vals = df["final_gini"].dropna().values
    t, p = sp.ttest_1samp(vals, 0.5)
    p_one = p / 2 if t < 0 else 1 - p / 2
    return {
        "hypothesis": "H5: Final Gini < 0.5 (inequality bounded)",
        "test": "One-sample t-test (one-sided)",
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "t": float(t),
        "p_value": float(p_one),
        "significant": p_one < BONFERRONI_ALPHA,
        "alpha": BONFERRONI_ALPHA,
        "n": len(vals),
    }


def test_h6_low_bunching(df: pd.DataFrame) -> dict:
    """H6: Mean bunching intensity < 0.05."""
    vals = df["final_bunching_intensity"].dropna().values
    t, p = sp.ttest_1samp(vals, 0.05)
    p_one = p / 2 if t < 0 else 1 - p / 2
    return {
        "hypothesis": "H6: Bunching intensity < 0.05 (low strategic bunching)",
        "test": "One-sample t-test (one-sided)",
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "t": float(t),
        "p_value": float(p_one),
        "significant": p_one < BONFERRONI_ALPHA,
        "alpha": BONFERRONI_ALPHA,
        "n": len(vals),
    }


# ---------------------------------------------------------------------------
# Cross-type analysis
# ---------------------------------------------------------------------------
def cross_type_anova(df: pd.DataFrame) -> dict:
    """One-way ANOVA/Kruskal-Wallis on final wealth across agent types."""
    types = ["honest", "gaming", "evasive", "collusive"]
    groups = []
    group_labels = []
    for t in types:
        col = f"{t}_mean_wealth"
        if col in df.columns:
            vals = df[col].dropna().values
            groups.append(vals)
            group_labels.append(t)

    if len(groups) < 2:
        return {}

    # Try parametric first
    F, p_anova = sp.f_oneway(*groups)

    # Non-parametric
    H, p_kw = sp.kruskal(*groups)

    # Post-hoc pairwise (Bonferroni)
    from itertools import combinations
    pairs = list(combinations(range(len(groups)), 2))
    n_comp = len(pairs)
    pairwise = []
    for i, j in pairs:
        t_val, p_val = sp.ttest_ind(groups[i], groups[j], equal_var=False)
        p_adj = min(p_val * n_comp, 1.0)
        d = cohens_d(groups[i], groups[j])
        pairwise.append({
            "comparison": f"{group_labels[i]} vs {group_labels[j]}",
            "mean_diff": float(groups[i].mean() - groups[j].mean()),
            "t": float(t_val),
            "p_raw": float(p_val),
            "p_adjusted": float(p_adj),
            "cohens_d": float(d),
            "d_interpretation": d_interpretation(d),
            "significant": p_adj < 0.05,
        })

    return {
        "test": "Cross-type wealth ANOVA + Kruskal-Wallis",
        "groups": {
            label: {"mean": float(g.mean()), "std": float(g.std()), "n": len(g)}
            for label, g in zip(group_labels, groups, strict=True)
        },
        "anova_F": float(F),
        "anova_p": float(p_anova),
        "kruskal_H": float(H),
        "kruskal_p": float(p_kw),
        "posthoc_pairwise": pairwise,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_markdown_report(results: dict) -> None:
    """Print formatted markdown tables."""
    desc = results["descriptive"]
    print("## Descriptive Statistics\n")
    print("| Metric | Mean | SD | Min | Max | N |")
    print("|--------|------|----|-----|-----|---|")
    for metric, s in desc.items():
        print(f"| {metric} | {s['mean']:.4f} | {s['std']:.4f} | "
              f"{s['min']:.4f} | {s['max']:.4f} | {s['n']} |")

    print("\n## Normality Checks (Shapiro-Wilk)\n")
    print("| Metric | W | p-value | Normal? |")
    print("|--------|---|---------|---------|")
    for metric, s in results["normality"].items():
        normal = "Yes" if s["normal"] else "No"
        print(f"| {metric} | {s['W']:.4f} | {s['p_value']:.4f} | {normal} |")

    print(f"\n## Pre-Registered Hypotheses (Bonferroni alpha = {BONFERRONI_ALPHA:.5f})\n")
    for h in results["hypotheses"]:
        sig = "PASS" if h["significant"] else "FAIL"
        print(f"### {h['hypothesis']}")
        print(f"- Test: {h['test']}")
        print(f"- t = {h['t']:.3f}, p = {h['p_value']:.6f}")
        if "cohens_d" in h:
            print(f"- Cohen's d = {h['cohens_d']:.3f} ({h['d_interpretation']})")
        if "power" in h:
            print(f"- Power = {h['power']:.3f}")
        print(f"- **{sig}** at alpha = {h['alpha']:.5f}")
        print()

    if results.get("cross_type"):
        ct = results["cross_type"]
        print("## Cross-Type Wealth Analysis\n")
        print("| Type | Mean Wealth | SD |")
        print("|------|-------------|-----|")
        for label, g in ct["groups"].items():
            print(f"| {label} | {g['mean']:.2f} | {g['std']:.2f} |")
        print(f"\n- ANOVA: F = {ct['anova_F']:.2f}, p = {ct['anova_p']:.4f}")
        print(f"- Kruskal-Wallis: H = {ct['kruskal_H']:.2f}, p = {ct['kruskal_p']:.4f}")

        print("\n### Post-hoc Pairwise (Bonferroni)\n")
        print("| Comparison | Mean Diff | t | p_adj | Cohen's d | Sig? |")
        print("|------------|-----------|---|-------|-----------|------|")
        for pw in ct["posthoc_pairwise"]:
            sig = "*" if pw["significant"] else ""
            print(f"| {pw['comparison']} | {pw['mean_diff']:.2f} | "
                  f"{pw['t']:.2f} | {pw['p_adjusted']:.4f} | "
                  f"{pw['cohens_d']:.3f} ({pw['d_interpretation']}) | {sig} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Statistical analysis for AI Economist GTB sweep"
    )
    parser.add_argument("sweep_dir", type=Path, help="Sweep output directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Write JSON results to file")
    args = parser.parse_args()

    csv_path = args.sweep_dir / "sweep_results.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} seed results from {csv_path}\n")

    # Key columns for normality checks
    key_cols = [
        "progressivity_index", "final_gini", "final_bunching_intensity",
        "honest_mean_wealth", "collusive_mean_wealth", "evasive_mean_wealth",
    ]

    results = {
        "n_seeds": len(df),
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "descriptive": descriptive_stats(df),
        "normality": normality_checks(df, key_cols),
        "hypotheses": [
            test_h1_progressive_tax(df),
            test_h2_honest_vs_collusive(df),
            test_h3_honest_vs_evasive(df),
            test_h4_evasive_vs_collusive(df),
            test_h5_gini_below_half(df),
            test_h6_low_bunching(df),
        ],
        "cross_type": cross_type_anova(df),
    }

    print_markdown_report(results)

    # Write JSON
    out_path = args.output or (args.sweep_dir / "aggregated_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
