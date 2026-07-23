#!/usr/bin/env python
"""
Generate standard plots from a SWARM parameter sweep CSV.

Usage:
    python examples/plot_sweep.py runs/20260212-015027_sweep/csv/sweep_results.csv
    python examples/plot_sweep.py runs/20260212-015027_sweep/csv/sweep_results.csv --output-dir runs/20260212-015027_sweep/plots

Detects governance.* columns as swept parameters and generates:
    1. welfare_by_config.png    — grouped bar chart (welfare by config)
    2. toxicity_by_config.png   — grouped bar chart (toxicity by config)
    3. payoff_by_agent_type.png — side-by-side panels per agent type
    4. welfare_boxplot.png      — box plots per config
    5. heatmap_tax_cb.png       — heatmap grid (welfare/toxicity/quality_gap)
    6. cb_welfare_boost_ci.png  — circuit breaker effect with 95% CIs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _setup_matplotlib(output_dir: Path):
    """Configure matplotlib with non-interactive backend."""
    config_dir = output_dir / ".mplconfig"
    cache_dir = output_dir / ".cache"
    config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def detect_sweep_params(df: pd.DataFrame) -> list[str]:
    """Detect swept governance parameters (columns with multiple unique values)."""
    candidates = [c for c in df.columns if c.startswith("governance.")]
    return [c for c in candidates if df[c].nunique() > 1]


def _config_label(row: pd.Series, params: list[str]) -> str:
    """Build a short label from swept parameter values."""
    parts = []
    for p in params:
        short = p.replace("governance.", "").replace("transaction_", "")
        val = row[p]
        if isinstance(val, bool) or str(val).lower() in ("true", "false"):
            parts.append(f"{short}={'Y' if str(val).lower() == 'true' else 'N'}")
        elif isinstance(val, float):
            parts.append(f"{short}={val:.0%}" if val < 1 else f"{short}={val:.1f}")
        else:
            parts.append(f"{short}={val}")
    return "\n".join(parts)


def plot_grouped_bar(plt, df: pd.DataFrame, params: list[str], metric: str,
                     output_dir: Path, ylabel: str | None = None) -> Path:
    """Grouped bar chart of a metric by config, with individual points + error bars."""
    configs = df.groupby(params)
    labels, means, stds, all_vals = [], [], [], []
    for _keys, group in configs:
        row = group.iloc[0]
        labels.append(_config_label(row, params))
        means.append(group[metric].mean())
        stds.append(group[metric].std())
        all_vals.append(group[metric].values)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.7, color="steelblue",
           edgecolor="navy", linewidth=0.5)

    for i, vals in enumerate(all_vals):
        ax.scatter(np.full_like(vals, x[i]) + np.random.uniform(-0.15, 0.15, len(vals)),
                   vals, color="darkred", s=20, alpha=0.6, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(f"{metric} by configuration")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / f"{metric}_by_config.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_payoff_by_agent_type(plt, df: pd.DataFrame, params: list[str],
                              output_dir: Path) -> Path | None:
    """Side-by-side panels showing payoffs for each agent type."""
    agent_cols = [c for c in df.columns if c.endswith("_payoff") and c != "avg_payoff"]
    n_types = len(agent_cols)
    if n_types == 0:
        return None

    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5), sharey=True)
    if n_types == 1:
        axes = [axes]

    configs = df.groupby(params)
    labels = []
    for _keys, group in configs:
        labels.append(_config_label(group.iloc[0], params))

    for ax, col in zip(axes, agent_cols, strict=False):
        agent_name = col.replace("_payoff", "").title()
        config_means, config_stds = [], []
        for _keys, group in configs:
            config_means.append(group[col].mean())
            config_stds.append(group[col].std())

        x = np.arange(len(labels))
        ax.bar(x, config_means, yerr=config_stds, capsize=3, alpha=0.7,
               color="teal", edgecolor="darkslategray", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_title(agent_name)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean payoff")
    fig.suptitle("Agent payoffs by configuration", fontsize=13)
    fig.tight_layout()

    out = output_dir / "payoff_by_agent_type.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_welfare_boxplot(plt, df: pd.DataFrame, params: list[str],
                         output_dir: Path) -> Path:
    """Box plots of welfare per configuration."""
    configs = df.groupby(params)
    labels, data = [], []
    for _keys, group in configs:
        labels.append(_config_label(group.iloc[0], params))
        data.append(group["welfare"].values)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
        patch.set_edgecolor("navy")

    ax.set_ylabel("Welfare")
    ax.set_title("Welfare distribution by configuration")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(fontsize=8)
    fig.tight_layout()

    out = output_dir / "welfare_boxplot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmap(plt, df: pd.DataFrame, params: list[str],
                 output_dir: Path) -> Path | None:
    """Heatmap of welfare/toxicity/quality_gap across 2D parameter grid."""
    if len(params) < 2:
        return None

    p1, p2 = params[0], params[1]
    metrics = ["welfare", "toxicity_rate", "quality_gap"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available, strict=False):
        pivot = df.groupby([p1, p2])[metric].mean().unstack()
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn" if metric == "welfare" else "YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.0%}" if isinstance(v, float) and v < 1 else str(v)
                            for v in pivot.index], fontsize=8)
        ax.set_xlabel(p2.replace("governance.", ""))
        ax.set_ylabel(p1.replace("governance.", ""))
        ax.set_title(metric)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Parameter grid heatmap (mean values)", fontsize=13)
    fig.tight_layout()

    out = output_dir / "heatmap_tax_cb.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cb_welfare_boost(plt, df: pd.DataFrame, params: list[str],
                          output_dir: Path) -> Path | None:
    """Circuit breaker welfare boost with 95% CIs and significance annotations."""
    cb_col = next((p for p in params if "circuit_breaker" in p), None)
    if cb_col is None:
        return None

    other_params = [p for p in params if p != cb_col]
    if not other_params:
        return None

    from scipy import stats as sp_stats

    groups = df.groupby(other_params)
    labels, boosts, cis, pvals = [], [], [], []

    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = "\n".join(
            f"{p.replace('governance.', '')}={k}" for p, k in zip(other_params, keys, strict=False)
        )
        cb_on = group[group[cb_col].astype(str).str.lower() == "true"]["welfare"]
        cb_off = group[group[cb_col].astype(str).str.lower() == "false"]["welfare"]

        if len(cb_on) < 2 or len(cb_off) < 2:
            continue

        diff = cb_on.mean() - cb_off.mean()
        se = np.sqrt(cb_on.var() / len(cb_on) + cb_off.var() / len(cb_off))
        ci = 1.96 * se
        _, p_val = sp_stats.mannwhitneyu(cb_on, cb_off, alternative="two-sided")

        labels.append(label)
        boosts.append(diff)
        cis.append(ci)
        pvals.append(p_val)

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    colors = ["forestgreen" if b > 0 else "firebrick" for b in boosts]

    ax.bar(x, boosts, yerr=cis, capsize=5, color=colors, alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for i, (b, p) in enumerate(zip(boosts, pvals, strict=False)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_pos = b + cis[i] + 0.3 if b >= 0 else b - cis[i] - 0.5
        ax.text(i, y_pos, f"{sig}\np={p:.3f}", ha="center", va="bottom" if b >= 0 else "top",
                fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Welfare boost (CB on - CB off)")
    ax.set_title("Circuit breaker welfare effect with 95% CI")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "cb_welfare_boost_ci.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate plots from a SWARM sweep CSV")
    parser.add_argument("csv_path", type=Path, help="Path to sweep_results.csv")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory for plots (default: sibling plots/ dir)")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    params = detect_sweep_params(df)
    if not params:
        print("Error: No swept governance.* parameters detected", file=sys.stderr)
        return 1

    print(f"Detected swept parameters: {params}")
    print(f"Rows: {len(df)}, Configs: {df.groupby(params).ngroups}")

    output_dir = args.output_dir
    if output_dir is None:
        if csv_path.parent.name == "csv":
            output_dir = csv_path.parent.parent / "plots"
        else:
            output_dir = csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt = _setup_matplotlib(output_dir)
    written = []

    # 1. Welfare by config
    p = plot_grouped_bar(plt, df, params, "welfare", output_dir, ylabel="Welfare")
    written.append(p)

    # 2. Toxicity by config
    if "toxicity_rate" in df.columns:
        p = plot_grouped_bar(plt, df, params, "toxicity_rate", output_dir,
                             ylabel="Toxicity rate")
        written.append(p)

    # 3. Payoff by agent type
    p = plot_payoff_by_agent_type(plt, df, params, output_dir)
    if p:
        written.append(p)

    # 4. Welfare boxplot
    p = plot_welfare_boxplot(plt, df, params, output_dir)
    written.append(p)

    # 5. Heatmap (2+ params only)
    p = plot_heatmap(plt, df, params, output_dir)
    if p:
        written.append(p)

    # 6. CB welfare boost (if circuit_breaker is swept)
    try:
        p = plot_cb_welfare_boost(plt, df, params, output_dir)
        if p:
            written.append(p)
    except ImportError:
        print("  Skipping CB boost plot (scipy not available)")

    print(f"\nGenerated {len(written)} plots in {output_dir}/:")
    for p in written:
        print(f"  {p.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
