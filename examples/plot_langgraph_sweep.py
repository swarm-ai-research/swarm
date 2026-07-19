#!/usr/bin/env python
"""
Generate plots from a LangGraph governed handoff sweep_results.csv.

Usage:
    python examples/plot_langgraph_sweep.py runs/<run_id>

Outputs:
    <run_dir>/plots/completion_by_max_cycles.png
    <run_dir>/plots/denial_rate_heatmap.png
    <run_dir>/plots/handoff_distribution.png
    <run_dir>/plots/governance_sensitivity.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _ensure_plots_dir(run_dir: Path) -> Path:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_completion_by_max_cycles(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Bar chart: completion rate grouped by max_cycles."""
    import matplotlib.pyplot as plt

    grouped = df.groupby("max_cycles")["task_completed"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.plot(kind="bar", ax=ax, color="#4C78A8", edgecolor="black")
    ax.set_xlabel("max_cycles")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Task Completion Rate by max_cycles")
    ax.set_ylim(0, 100)
    for i, v in enumerate(grouped):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_xticklabels(grouped.index, rotation=0)
    fig.tight_layout()
    path = plots_dir / "completion_by_max_cycles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_denial_rate_heatmap(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Heatmap: denial rate by max_cycles x max_handoffs."""
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(
        values="denial_rate",
        index="max_cycles",
        columns="max_handoffs",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("max_handoffs")
    ax.set_ylabel("max_cycles")
    ax.set_title("Mean Denial Rate by Governance Parameters")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="Denial Rate")
    fig.tight_layout()
    path = plots_dir / "denial_rate_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_handoff_distribution(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Histogram of total handoffs per run."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    max_h = int(df["total_handoffs"].max()) + 1
    bins = range(0, max(max_h + 1, 2))
    ax.hist(
        df["total_handoffs"],
        bins=bins,
        color="#E45756",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xlabel("Total Handoffs per Run")
    ax.set_ylabel("Count")
    ax.set_title("Handoff Volume Distribution")
    ax.axvline(
        df["total_handoffs"].mean(),
        color="black",
        linestyle="--",
        label=f'Mean: {df["total_handoffs"].mean():.1f}',
    )
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "handoff_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_governance_sensitivity(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Grouped bar: completion rate by max_cycles, split by trust_boundaries."""
    import matplotlib.pyplot as plt
    import numpy as np

    pivot = df.pivot_table(
        values="task_completed",
        index="max_cycles",
        columns="trust_boundaries",
        aggfunc="mean",
    ) * 100

    x = np.arange(len(pivot.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    cols = sorted(pivot.columns)
    labels = {True: "Trust Boundaries ON", False: "Trust Boundaries OFF"}
    colors = {True: "#4C78A8", False: "#F58518"}

    for i, col in enumerate(cols):
        offset = -width / 2 + i * width
        vals = pivot[col].values
        bars = ax.bar(x + offset, vals, width, label=labels.get(col, str(col)),
                      color=colors.get(col, "#999"), edgecolor="black")
        for bar, v in zip(bars, vals, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                    f"{v:.0f}%", ha="center", fontsize=9)

    ax.set_xlabel("max_cycles")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Governance Sensitivity: Completion by max_cycles & trust_boundaries")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "governance_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cross_seed_stability(df: pd.DataFrame, plots_dir: Path) -> Path | None:
    """Per-seed completion rates to show cross-seed stability."""
    import matplotlib.pyplot as plt

    seeds = sorted(df["seed"].unique())
    if len(seeds) < 2:
        return None

    rates = [df[df["seed"] == s]["task_completed"].mean() * 100 for s in seeds]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"Seed {s}" for s in seeds], rates,
                  color="#54A24B", edgecolor="black")
    for bar, v in zip(bars, rates, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontweight="bold")
    overall = df["task_completed"].mean() * 100
    ax.axhline(overall, color="black", linestyle="--",
               label=f"Overall: {overall:.1f}%")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Cross-Seed Completion Stability")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "cross_seed_stability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot LangGraph sweep results")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    args = parser.parse_args()

    csv_path = args.run_dir / "sweep_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    plots_dir = _ensure_plots_dir(args.run_dir)
    generated = []

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        readme = plots_dir / "README.txt"
        readme.write_text(
            "Plotting requires matplotlib. Install with: pip install matplotlib\n"
        )
        print(f"matplotlib not available. Wrote {readme}")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")

    for plot_fn in [
        plot_completion_by_max_cycles,
        plot_denial_rate_heatmap,
        plot_handoff_distribution,
        plot_governance_sensitivity,
        plot_cross_seed_stability,
    ]:
        try:
            path = plot_fn(df, plots_dir)
            if path:
                generated.append(path)
                print(f"  Generated: {path}")
        except Exception as e:
            print(f"  WARN: {plot_fn.__name__} failed: {e}")

    print(f"\n{len(generated)} plots written to {plots_dir}/")

    # Print summary stats
    n = len(df)
    completed = df["task_completed"].sum()
    total_handoffs = df["total_handoffs"].sum()
    denied = df["denied_handoffs"].sum()
    print(f"\nSummary: {int(completed)}/{n} completed ({completed/n*100:.1f}%)")
    print(f"  Total handoffs: {int(total_handoffs)}, Denied: {int(denied)}")
    seeds = sorted(df["seed"].unique())
    for s in seeds:
        sub = df[df["seed"] == s]
        rate = sub["task_completed"].mean() * 100
        print(f"  Seed {s}: {rate:.1f}%")


if __name__ == "__main__":
    main()
