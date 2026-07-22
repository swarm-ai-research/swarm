#!/usr/bin/env python
"""Generate multi-seed publication figures for the AI Economist GTB paper.

Reads sweep output (all_metrics.csv, sweep_results.csv, per-seed tax schedules)
and produces 5 figures with mean ± SD ribbons and significance brackets.

Usage:
    python examples/plot_ai_economist_multiseed.py runs/<sweep_dir>/

Outputs to <sweep_dir>/plots/ and copies to docs/papers/figures/ai_economist_gtb/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ai_economist_colors import AGENT_TYPE_COLORS as _AGENT_TYPE_COLORS  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from swarm.analysis.theme import COLORS, swarm_theme  # noqa: E402

# Default initial flat 10% schedule
_INITIAL_BRACKETS = [
    {"threshold": 0.0, "rate": 0.10},
    {"threshold": 10.0, "rate": 0.10},
    {"threshold": 25.0, "rate": 0.10},
    {"threshold": 50.0, "rate": 0.10},
]

DPI = 300


def _bracket_steps(brackets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert bracket defs to step-plot arrays."""
    max_income = 120.0
    xs, ys = [], []
    for i, b in enumerate(brackets):
        xs.append(b["threshold"])
        ys.append(b["rate"])
        if i < len(brackets) - 1:
            xs.append(brackets[i + 1]["threshold"])
            ys.append(b["rate"])
    xs.append(max_income)
    ys.append(brackets[-1]["rate"])
    return np.array(xs), np.array(ys)


def _ribbon_plot(
    ax, epochs: np.ndarray, mean: np.ndarray, std: np.ndarray,
    color: str, label: str, **kwargs,
) -> None:
    """Plot mean line with ±1 SD ribbon."""
    ax.fill_between(epochs, mean - std, mean + std,
                    alpha=0.15, color=color)
    ax.plot(epochs, mean, color=color, linewidth=2.5, label=label, **kwargs)


def _significance_bracket(
    ax, x1: float, x2: float, y: float, label: str, color: str = "#E6EDF3",
) -> None:
    """Draw a significance bracket between two bars."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=1.2)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom",
            fontsize=9, color=color)


# ---------------------------------------------------------------------------
# Figure 1: Tax Schedule Convergence
# ---------------------------------------------------------------------------
def fig1_tax_schedule(
    sweep_dir: Path, sweep_df: pd.DataFrame, plots_dir: Path,
) -> Path:
    """Initial flat 10% (dashed) vs mean final schedule with ±1 SD ribbon."""
    # Collect final bracket rates from each seed
    seeds = sweep_df["seed"].values
    all_brackets: list[list[dict]] = []

    for seed in seeds:
        sched_path = sweep_dir / f"seed_{seed}" / "csv" / "tax_schedule.json"
        if sched_path.exists():
            with open(sched_path) as f:
                sched = json.load(f)
            all_brackets.append(sched["brackets"])

    if not all_brackets:
        print("  Warning: no tax schedule files found, skipping fig1")
        return plots_dir / "fig1_tax_schedule_convergence.png"

    # Compute mean and std rate at each bracket
    n_brackets = len(all_brackets[0])
    thresholds = [all_brackets[0][i]["threshold"] for i in range(n_brackets)]

    mean_rates = []
    std_rates = []
    for i in range(n_brackets):
        rates = [b[i]["rate"] for b in all_brackets]
        mean_rates.append(np.mean(rates))
        std_rates.append(np.std(rates))

    with swarm_theme():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Initial flat schedule
        xi, yi = _bracket_steps(_INITIAL_BRACKETS)
        ax.plot(xi, yi, color=COLORS.TEXT_MUTED, linewidth=2,
                linestyle="--", label="Initial (flat 10%)", drawstyle="steps-post")

        # Mean final schedule
        mean_brackets = [{"threshold": t, "rate": r}
                         for t, r in zip(thresholds, mean_rates, strict=True)]
        xf, yf = _bracket_steps(mean_brackets)
        ax.plot(xf, yf, color=COLORS.PLANNER, linewidth=3,
                label=f"Final mean (n={len(seeds)})", drawstyle="steps-post")

        # ±1 SD ribbon (approximate: shade between mean-std and mean+std)
        lo_brackets = [{"threshold": t, "rate": max(0, r - s)}
                       for t, r, s in zip(thresholds, mean_rates, std_rates, strict=True)]
        hi_brackets = [{"threshold": t, "rate": min(1, r + s)}
                       for t, r, s in zip(thresholds, mean_rates, std_rates, strict=True)]
        xl, yl = _bracket_steps(lo_brackets)
        xh, yh = _bracket_steps(hi_brackets)
        ax.fill_between(xf, yl, yh, alpha=0.15, step="post",
                        color=COLORS.PLANNER)

        # Annotate mean rates
        for t, r, s in zip(thresholds, mean_rates, std_rates, strict=True):
            ax.annotate(
                f"{r:.1%}±{s:.1%}",
                xy=(t + 2, r), fontsize=9, color=COLORS.PLANNER, fontweight="bold",
            )

        ax.set_xlabel("Income Bracket Threshold", fontsize=12)
        ax.set_ylabel("Marginal Tax Rate", fontsize=12)
        ax.set_title("Tax Schedule Convergence Across Seeds", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_ylim(0, 0.75)
        ax.set_xlim(-2, 80)
        ax.legend(fontsize=11, loc="lower right")

        out = plots_dir / "fig1_tax_schedule_convergence.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: Economy Dashboard (2x2 with mean ± SD ribbons)
# ---------------------------------------------------------------------------
def fig2_economy_dashboard(
    all_metrics: pd.DataFrame, plots_dir: Path,
) -> Path:
    """2x2: production, revenue, gini, effective tax rate — all mean ± SD."""
    grouped = all_metrics.groupby("epoch")

    epochs = np.array(sorted(all_metrics["epoch"].unique()))
    panels = [
        ("total_production", "Total Production", COLORS.PRODUCTIVITY),
        ("total_tax_revenue", "Tax Revenue", COLORS.REVENUE),
        ("gini_coefficient", "Gini Coefficient", COLORS.EVASION),
        ("mean_effective_tax_rate", "Effective Tax Rate", COLORS.PLANNER),
    ]

    with swarm_theme():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for ax, (col, title, color) in zip(axes.flat, panels, strict=False):
            mean = grouped[col].mean().reindex(epochs).values
            std = grouped[col].std().reindex(epochs).values
            _ribbon_plot(ax, epochs, mean, std, color, "Mean ± 1 SD")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.legend(fontsize=9, loc="upper left")

        fig.suptitle("Economy Dashboard (Multi-Seed)", fontsize=16,
                     y=0.98, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out = plots_dir / "fig2_economy_dashboard_multiseed.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: Wealth by Type (boxplot/violin with significance brackets)
# ---------------------------------------------------------------------------
def fig3_wealth_by_type(
    sweep_df: pd.DataFrame, stats_path: Path, plots_dir: Path,
) -> Path:
    """Boxplot: 4 agent types × N seeds, with significance brackets."""
    types = ["honest", "gaming", "evasive", "collusive"]
    type_labels = ["Honest", "Gaming", "Evasive", "Collusive"]
    colors = [_AGENT_TYPE_COLORS[t] for t in types]

    data = [sweep_df[f"{t}_mean_wealth"].dropna().values for t in types]

    # Load stats for significance info
    sig_pairs = []
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        for h in stats.get("hypotheses", []):
            if h.get("significant") and "wealth" in h.get("hypothesis", "").lower():
                hyp = h["hypothesis"]
                if "honest" in hyp.lower() and "collusive" in hyp.lower():
                    sig_pairs.append((0, 3, f"p={h['p_value']:.2e}"))
                elif "honest" in hyp.lower() and "evasive" in hyp.lower():
                    sig_pairs.append((0, 2, f"p={h['p_value']:.2e}"))
                elif "evasive" in hyp.lower() and "collusive" in hyp.lower():
                    sig_pairs.append((2, 3, f"p={h['p_value']:.2e}"))

    with swarm_theme():
        fig, ax = plt.subplots(figsize=(10, 7))

        parts = ax.violinplot(data, positions=range(len(types)),
                              showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(to_rgba(colors[i], 0.3))
            pc.set_edgecolor(colors[i])
        for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color(COLORS.TEXT_PRIMARY)

        # Overlay scatter
        for i, d in enumerate(data):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(d))
            ax.scatter(np.full(len(d), i) + jitter, d,
                       color=colors[i], alpha=0.7, s=30, zorder=5)

        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(type_labels, fontsize=12)
        ax.set_ylabel("Mean Total Wealth", fontsize=12)
        ax.set_title("Agent Wealth by Policy Type (Multi-Seed)",
                     fontsize=14, fontweight="bold", pad=12)

        # Significance brackets
        ymax = max(max(d) for d in data if len(d) > 0)
        for idx, (i, j, label) in enumerate(sig_pairs):
            y = ymax + (idx + 1) * ymax * 0.08
            _significance_bracket(ax, i, j, y, label)

        # Legend
        patches = [Patch(facecolor=to_rgba(c, 0.3), edgecolor=c, label=lab)
                   for c, lab in zip(colors, type_labels, strict=True)]
        ax.legend(handles=patches, fontsize=10, loc="upper right")

        out = plots_dir / "fig3_wealth_by_type.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: Enforcement (2-panel)
# ---------------------------------------------------------------------------
def fig4_enforcement(
    all_metrics: pd.DataFrame, sweep_df: pd.DataFrame, plots_dir: Path,
) -> Path:
    """2-panel: audit/fines timeline (mean ± SD) + evasive vs honest net income."""
    grouped = all_metrics.groupby("epoch")
    epochs = np.array(sorted(all_metrics["epoch"].unique()))

    with swarm_theme():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Audit/fines timeline
        audit_mean = grouped["total_audits"].mean().reindex(epochs).values
        audit_std = grouped["total_audits"].std().reindex(epochs).values
        fine_mean = grouped["total_fines"].mean().reindex(epochs).values
        fine_std = grouped["total_fines"].std().reindex(epochs).values

        _ribbon_plot(ax1, epochs, audit_mean, audit_std,
                     COLORS.WELFARE, "Audits")
        ax1r = ax1.twinx()
        _ribbon_plot(ax1r, epochs, fine_mean, fine_std,
                     COLORS.TOXICITY, "Fines ($)")
        ax1r.set_ylabel("Fine Amount", fontsize=11, color=COLORS.TOXICITY)
        ax1r.spines["right"].set_visible(True)
        ax1r.spines["right"].set_color(COLORS.ACCENT_BORDER)

        ax1.set_title("Enforcement Activity", fontsize=13, fontweight="bold", pad=10)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Audit Count", fontsize=11)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1r.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

        # Panel 2: Evasive vs honest wealth comparison
        types = ["honest", "evasive"]
        labels = ["Honest", "Evasive"]
        colors = [_AGENT_TYPE_COLORS["honest"], _AGENT_TYPE_COLORS["evasive"]]
        means = [sweep_df[f"{t}_mean_wealth"].mean() for t in types]
        stds = [sweep_df[f"{t}_mean_wealth"].std() for t in types]

        bars = ax2.bar(labels, means, yerr=stds, capsize=8,
                       color=[to_rgba(c, 0.7) for c in colors],
                       edgecolor=colors, linewidth=1.5)
        ax2.set_ylabel("Mean Total Wealth", fontsize=11)
        ax2.set_title("Evasive vs Honest Net Wealth", fontsize=13,
                      fontweight="bold", pad=10)

        # Annotate values
        for bar, m, s in zip(bars, means, stds, strict=True):
            ax2.text(bar.get_x() + bar.get_width() / 2, m + s + 20,
                     f"{m:.0f}±{s:.0f}", ha="center", fontsize=10,
                     color=COLORS.TEXT_PRIMARY)

        fig.suptitle("Enforcement & Evasion Cost", fontsize=16,
                     y=1.02, fontweight="bold")
        fig.tight_layout()

        out = plots_dir / "fig4_enforcement.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5: Collusion (2-panel)
# ---------------------------------------------------------------------------
def fig5_collusion(
    all_metrics: pd.DataFrame, sweep_df: pd.DataFrame, plots_dir: Path,
) -> Path:
    """2-panel: collusion events timeline + collusive vs honest wealth."""
    grouped = all_metrics.groupby("epoch")
    epochs = np.array(sorted(all_metrics["epoch"].unique()))

    with swarm_theme():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Collusion events (mean ± SD)
        coll_mean = grouped["collusion_events_detected"].mean().reindex(epochs).values
        coll_std = grouped["collusion_events_detected"].std().reindex(epochs).values
        susp_mean = grouped["collusion_suspicion_mean"].mean().reindex(epochs).values
        susp_std = grouped["collusion_suspicion_mean"].std().reindex(epochs).values

        _ribbon_plot(ax1, epochs, coll_mean, coll_std,
                     COLORS.ADVERSARIAL, "Events Detected")

        ax1r = ax1.twinx()
        _ribbon_plot(ax1r, epochs, susp_mean, susp_std,
                     COLORS.PLANNER, "Suspicion Score")
        ax1r.set_ylabel("Mean Suspicion", fontsize=11, color=COLORS.PLANNER)
        ax1r.spines["right"].set_visible(True)
        ax1r.spines["right"].set_color(COLORS.ACCENT_BORDER)

        ax1.set_title("Collusion Detection", fontsize=13, fontweight="bold", pad=10)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Events per Epoch", fontsize=11)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1r.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

        # Panel 2: Collusive vs honest wealth
        types = ["honest", "collusive"]
        labels = ["Honest", "Collusive"]
        colors = [_AGENT_TYPE_COLORS["honest"], _AGENT_TYPE_COLORS["collusive"]]
        means = [sweep_df[f"{t}_mean_wealth"].mean() for t in types]
        stds = [sweep_df[f"{t}_mean_wealth"].std() for t in types]

        bars = ax2.bar(labels, means, yerr=stds, capsize=8,
                       color=[to_rgba(c, 0.7) for c in colors],
                       edgecolor=colors, linewidth=1.5)
        ax2.set_ylabel("Mean Total Wealth", fontsize=11)
        ax2.set_title("Collusive vs Honest Wealth", fontsize=13,
                      fontweight="bold", pad=10)

        for bar, m, s in zip(bars, means, stds, strict=True):
            ax2.text(bar.get_x() + bar.get_width() / 2, m + s + 20,
                     f"{m:.0f}±{s:.0f}", ha="center", fontsize=10,
                     color=COLORS.TEXT_PRIMARY)

        fig.suptitle("Collusion Dynamics", fontsize=16,
                     y=1.02, fontweight="bold")
        fig.tight_layout()

        out = plots_dir / "fig5_collusion.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate multi-seed AI Economist GTB figures"
    )
    parser.add_argument("sweep_dir", type=Path, help="Sweep output directory")
    parser.add_argument("--copy-to-docs", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy figures to docs/papers/figures/")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_metrics = pd.read_csv(sweep_dir / "all_metrics.csv")
    sweep_df = pd.read_csv(sweep_dir / "sweep_results.csv")
    stats_path = sweep_dir / "aggregated_stats.json"

    print(f"Loaded {len(sweep_df)} seeds, {len(all_metrics)} metric rows")

    # Generate figures
    outputs = []
    outputs.append(fig1_tax_schedule(sweep_dir, sweep_df, plots_dir))
    outputs.append(fig2_economy_dashboard(all_metrics, plots_dir))
    outputs.append(fig3_wealth_by_type(sweep_df, stats_path, plots_dir))
    outputs.append(fig4_enforcement(all_metrics, sweep_df, plots_dir))
    outputs.append(fig5_collusion(all_metrics, sweep_df, plots_dir))

    for p in outputs:
        print(f"  -> {p}")

    # Copy to docs
    if args.copy_to_docs:
        docs_dir = Path("docs/papers/figures/ai_economist_gtb")
        docs_dir.mkdir(parents=True, exist_ok=True)
        for p in outputs:
            if p.exists():
                shutil.copy2(p, docs_dir / p.name)
        print(f"\nCopied to {docs_dir}/")

    print(f"\nAll plots saved to {plots_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
