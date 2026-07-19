#!/usr/bin/env python
"""
Statistical significance analysis for SWARM parameter sweep results.

Usage:
    python examples/sweep_stats.py runs/20260212-015027_sweep/csv/sweep_results.csv
    python examples/sweep_stats.py runs/20260212-015027_sweep/csv/sweep_results.csv --output results.json

Runs the standard battery:
    - Mann-Whitney U (pairwise between binary parameters like CB on/off)
    - Kruskal-Wallis H (across multi-valued parameters like tax rates)
    - Two-way ANOVA with interaction term
    - Cohen's d effect sizes
    - Post-hoc pairwise comparisons with Bonferroni correction
    - Honest-adversarial payoff gap analysis
    - Power analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp


def detect_sweep_params(df: pd.DataFrame) -> list[str]:
    """Detect swept governance parameters."""
    candidates = [c for c in df.columns if c.startswith("governance.")]
    return [c for c in candidates if df[c].nunique() > 1]


def _short_name(param: str) -> str:
    return param.replace("governance.", "").replace("transaction_", "")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(((n1 - 1) * a.std(ddof=1)**2 + (n2 - 1) * b.std(ddof=1)**2)
                         / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def eta_squared(groups: list[np.ndarray]) -> float:
    """Compute eta-squared effect size for ANOVA."""
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean)**2)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Approximate power using normal distribution (z-test approximation)."""
    from scipy.stats import norm
    if n < 2 or np.isnan(effect_size):
        return float("nan")
    se = np.sqrt(2.0 / n)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = abs(effect_size) / se - z_alpha
    return float(norm.cdf(z_power))


def analyze_binary_param(df: pd.DataFrame, param: str, metric: str = "welfare") -> dict:
    """Mann-Whitney U test for a binary swept parameter."""
    vals = df[param].unique()
    if len(vals) != 2:
        return {}

    g1 = df[df[param] == vals[0]][metric].values
    g2 = df[df[param] == vals[1]][metric].values

    if len(g1) < 2 or len(g2) < 2:
        return {}

    U, p_val = sp.mannwhitneyu(g1, g2, alternative="two-sided")
    d = cohens_d(g1, g2)

    return {
        "test": "Mann-Whitney U",
        "param": param,
        "metric": metric,
        "groups": {str(vals[0]): {"n": len(g1), "mean": float(g1.mean()), "std": float(g1.std())},
                   str(vals[1]): {"n": len(g2), "mean": float(g2.mean()), "std": float(g2.std())}},
        "U": float(U),
        "p_value": float(p_val),
        "cohens_d": float(d),
        "d_interpretation": "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large",
        "significant_05": p_val < 0.05,
        "power": power_analysis(d, min(len(g1), len(g2))),
    }


def analyze_multi_param(df: pd.DataFrame, param: str, metric: str = "welfare") -> dict:
    """Kruskal-Wallis H test for a multi-valued parameter."""
    groups_dict = {str(k): group[metric].values for k, group in df.groupby(param)}
    groups = list(groups_dict.values())
    if len(groups) < 3:
        return {}

    H, p_val = sp.kruskal(*groups)
    eta2 = eta_squared(groups)

    return {
        "test": "Kruskal-Wallis H",
        "param": param,
        "metric": metric,
        "groups": {k: {"n": len(v), "mean": float(v.mean()), "std": float(v.std())}
                   for k, v in groups_dict.items()},
        "H": float(H),
        "p_value": float(p_val),
        "eta_squared": float(eta2),
        "significant_05": p_val < 0.05,
    }


def two_way_anova(df: pd.DataFrame, params: list[str], metric: str = "welfare") -> dict:
    """Two-way ANOVA with interaction (Type II SS via manual computation)."""
    if len(params) < 2:
        return {}

    p1, p2 = params[0], params[1]

    # Grand mean
    grand_mean = df[metric].mean()
    N = len(df)

    # Main effect of p1
    groups_p1 = [g[metric].values for _, g in df.groupby(p1)]
    ss_p1 = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_p1)
    df_p1 = df[p1].nunique() - 1

    # Main effect of p2
    groups_p2 = [g[metric].values for _, g in df.groupby(p2)]
    ss_p2 = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_p2)
    df_p2 = df[p2].nunique() - 1

    # Cell means (interaction)
    cell_groups = df.groupby([p1, p2])
    ss_cells = sum(len(g) * (g[metric].mean() - grand_mean)**2
                   for _, g in cell_groups)
    ss_interaction = ss_cells - ss_p1 - ss_p2
    df_interaction = df_p1 * df_p2

    # Residual
    ss_total = np.sum((df[metric].values - grand_mean)**2)
    ss_residual = ss_total - ss_cells
    df_residual = N - df[p1].nunique() * df[p2].nunique()

    if df_residual <= 0:
        return {}

    ms_p1 = ss_p1 / df_p1 if df_p1 > 0 else 0
    ms_p2 = ss_p2 / df_p2 if df_p2 > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_residual = ss_residual / df_residual

    results = {}
    if ms_residual > 0:
        F_p1 = ms_p1 / ms_residual
        F_p2 = ms_p2 / ms_residual
        F_int = ms_interaction / ms_residual

        results = {
            "test": "Two-way ANOVA",
            "metric": metric,
            "factors": {
                _short_name(p1): {
                    "F": float(F_p1), "df": (int(df_p1), int(df_residual)),
                    "p_value": float(1 - sp.f.cdf(F_p1, df_p1, df_residual)),
                    "eta_squared": float(ss_p1 / ss_total) if ss_total > 0 else 0,
                },
                _short_name(p2): {
                    "F": float(F_p2), "df": (int(df_p2), int(df_residual)),
                    "p_value": float(1 - sp.f.cdf(F_p2, df_p2, df_residual)),
                    "eta_squared": float(ss_p2 / ss_total) if ss_total > 0 else 0,
                },
                "interaction": {
                    "F": float(F_int), "df": (int(df_interaction), int(df_residual)),
                    "p_value": float(1 - sp.f.cdf(F_int, df_interaction, df_residual)),
                    "eta_squared": float(ss_interaction / ss_total) if ss_total > 0 else 0,
                },
            },
        }

    return results


def posthoc_pairwise(df: pd.DataFrame, param: str, metric: str = "welfare") -> list[dict]:
    """Pairwise comparisons with Bonferroni correction."""
    groups = {str(k): g[metric].values for k, g in df.groupby(param)}
    pairs = list(combinations(groups.keys(), 2))
    n_comparisons = len(pairs)
    results = []

    for a, b in pairs:
        ga, gb = groups[a], groups[b]
        if len(ga) < 2 or len(gb) < 2:
            continue
        _, p_raw = sp.mannwhitneyu(ga, gb, alternative="two-sided")
        p_adj = min(p_raw * n_comparisons, 1.0)
        d = cohens_d(ga, gb)

        results.append({
            "comparison": f"{a} vs {b}",
            "mean_diff": float(ga.mean() - gb.mean()),
            "p_raw": float(p_raw),
            "p_adjusted": float(p_adj),
            "correction": "Bonferroni",
            "n_comparisons": n_comparisons,
            "cohens_d": float(d),
            "significant_05": p_adj < 0.05,
        })

    return results


def agent_payoff_gap(df: pd.DataFrame) -> dict | None:
    """Test honest-adversarial payoff gap across configs."""
    if "honest_payoff" not in df.columns or "adversarial_payoff" not in df.columns:
        return None

    gaps = df["honest_payoff"].values - df["adversarial_payoff"].values
    t, p_val = sp.ttest_1samp(gaps, 0)

    return {
        "test": "One-sample t-test (honest - adversarial gap > 0)",
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std()),
        "t": float(t),
        "p_value": float(p_val),
        "significant_05": p_val < 0.05,
        "n": len(gaps),
    }


def run_analysis(df: pd.DataFrame) -> dict:
    """Run the full statistical battery."""
    params = detect_sweep_params(df)
    results = {"parameters": params, "n_rows": len(df), "tests": []}

    for param in params:
        n_unique = df[param].nunique()
        if n_unique == 2:
            r = analyze_binary_param(df, param)
            if r:
                results["tests"].append(r)
        if n_unique >= 3:
            r = analyze_multi_param(df, param)
            if r:
                results["tests"].append(r)
            # Post-hoc pairwise
            pairwise = posthoc_pairwise(df, param)
            if pairwise:
                results["tests"].append({
                    "test": "Post-hoc pairwise",
                    "param": param,
                    "comparisons": pairwise,
                })

    # Two-way ANOVA
    if len(params) >= 2:
        anova = two_way_anova(df, params)
        if anova:
            results["tests"].append(anova)

    # Agent payoff gap
    gap = agent_payoff_gap(df)
    if gap:
        results["tests"].append(gap)

    # Power summary
    min_n = min(len(g) for _, g in df.groupby(params[0])) if params else len(df)
    results["power_note"] = (
        f"n={min_n} per group. "
        f"Power to detect d=0.8: {power_analysis(0.8, min_n):.2f}, "
        f"d=0.5: {power_analysis(0.5, min_n):.2f}, "
        f"d=0.2: {power_analysis(0.2, min_n):.2f}"
    )

    return results


def print_report(results: dict) -> None:
    """Print a formatted text report."""
    print("=" * 70)
    print("SWEEP STATISTICAL SIGNIFICANCE REPORT")
    print("=" * 70)
    print(f"Rows: {results['n_rows']}")
    print(f"Swept parameters: {', '.join(results['parameters'])}")
    print(f"Power: {results['power_note']}")
    print()

    for test in results["tests"]:
        name = test.get("test", "Unknown")
        print(f"--- {name} ---")

        if name == "Mann-Whitney U":
            param = _short_name(test["param"])
            print(f"  Parameter: {param}")
            for gname, gstats in test["groups"].items():
                print(f"    {gname}: mean={gstats['mean']:.3f}, std={gstats['std']:.3f}, n={gstats['n']}")
            sig = "YES" if test["significant_05"] else "NO"
            print(f"  U={test['U']:.1f}, p={test['p_value']:.4f} ({sig})")
            print(f"  Cohen's d={test['cohens_d']:.3f} ({test['d_interpretation']})")
            print(f"  Power: {test['power']:.3f}")

        elif name == "Kruskal-Wallis H":
            param = _short_name(test["param"])
            print(f"  Parameter: {param}")
            for gname, gstats in test["groups"].items():
                print(f"    {gname}: mean={gstats['mean']:.3f}, std={gstats['std']:.3f}, n={gstats['n']}")
            sig = "YES" if test["significant_05"] else "NO"
            print(f"  H={test['H']:.2f}, p={test['p_value']:.4f} ({sig})")
            print(f"  eta^2={test['eta_squared']:.3f}")

        elif name == "Two-way ANOVA":
            print(f"  Metric: {test['metric']}")
            for factor, stats in test["factors"].items():
                sig = "YES" if stats["p_value"] < 0.05 else "NO"
                print(f"  {factor}: F({stats['df'][0]},{stats['df'][1]})="
                      f"{stats['F']:.2f}, p={stats['p_value']:.4f} ({sig}), "
                      f"eta^2={stats['eta_squared']:.3f}")

        elif name == "Post-hoc pairwise":
            param = _short_name(test["param"])
            print(f"  Parameter: {param} (Bonferroni correction)")
            for comp in test["comparisons"]:
                sig = "*" if comp["significant_05"] else ""
                print(f"    {comp['comparison']}: diff={comp['mean_diff']:.3f}, "
                      f"p_adj={comp['p_adjusted']:.4f}{sig}, d={comp['cohens_d']:.3f}")

        elif "honest - adversarial" in name.lower() or "gap" in name.lower():
            sig = "YES" if test["significant_05"] else "NO"
            print(f"  Mean gap: {test['mean_gap']:.3f} +/- {test['std_gap']:.3f}")
            print(f"  t={test['t']:.2f}, p={test['p_value']:.4f} ({sig}), n={test['n']}")

        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep statistical significance analysis")
    parser.add_argument("csv_path", type=Path, help="Path to sweep_results.csv")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Write JSON results to file")
    parser.add_argument("--metric", default="welfare",
                        help="Primary metric for analysis (default: welfare)")
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: {args.csv_path} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv_path)
    results = run_analysis(df)
    print_report(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"Results written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
