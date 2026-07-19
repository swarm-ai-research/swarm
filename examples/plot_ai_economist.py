#!/usr/bin/env python
"""Generate publication-quality visuals for the AI Economist GTB simulation.

Usage:
    python examples/plot_ai_economist.py runs/20260215_095359_ai_economist_seed42

Outputs two composite figures to <run_dir>/plots/:
    - gtb_economy_dashboard.png   (production, inequality, tax schedule)
    - gtb_adversarial_dynamics.png (agent wealth, enforcement, collusion)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai_economist_colors import AGENT_TYPE_COLORS as _AGENT_TYPE_COLORS
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Patch

from swarm.analysis.enhanced_dashboard import plot_enhanced_line
from swarm.analysis.theme import COLORS, add_danger_zone, swarm_theme

# Default initial tax schedule (flat 10% before planner adaptation)
_INITIAL_BRACKETS = [
    {"threshold": 0.0, "rate": 0.10},
    {"threshold": 10.0, "rate": 0.10},
    {"threshold": 25.0, "rate": 0.10},
    {"threshold": 50.0, "rate": 0.10},
]


def _agent_type(agent_id: str) -> str:
    """Extract agent type from worker ID like 'worker_honest_0_1'."""
    parts = agent_id.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"


def _bracket_steps(brackets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert bracket defs to step-plot arrays (income, rate)."""
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


def _kpi_badge(ax, value: str, label: str, color: str) -> None:
    """Render a compact KPI badge in the given axes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rect = FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.9,
        boxstyle="round,pad=0.06",
        facecolor=to_rgba(color, 0.10),
        edgecolor=to_rgba(color, 0.35),
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(0.5, 0.68, value, ha="center", va="center",
            fontsize=18, fontweight="bold", color=color)
    ax.text(0.5, 0.25, label, ha="center", va="center",
            fontsize=9, color=COLORS.TEXT_MUTED)


def plot_economy_dashboard(
    metrics: pd.DataFrame,
    tax_schedule: dict,
    plots_dir: Path,
) -> Path:
    """Figure 1: Economy Dashboard — 2x2 with KPI row."""
    with swarm_theme():
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(
            3, 3,
            height_ratios=[0.18, 1, 1],
            hspace=0.4, wspace=0.35,
        )

        epochs = metrics["epoch"]

        # ── KPI badges (top row) ──
        final = metrics.iloc[-1]
        kpis = [
            (f"{final['total_production']:.0f}", "Total Production", COLORS.PRODUCTIVITY),
            (f"{final['total_tax_revenue']:.0f}", "Tax Revenue", COLORS.REVENUE),
            (f"{final['gini_coefficient']:.2f}", "Final Gini", COLORS.EVASION),
        ]
        for i, (val, label, color) in enumerate(kpis):
            _kpi_badge(fig.add_subplot(gs[0, i]), val, label, color)

        # ── Panel 1: Production & Tax Revenue (dual axis) ──
        ax1 = fig.add_subplot(gs[1, :2])
        # Subtle gradient fill for production
        ax1.fill_between(epochs, metrics["total_production"],
                         alpha=0.08, color=COLORS.PRODUCTIVITY)
        ax1.plot(epochs, metrics["total_production"],
                 color=COLORS.PRODUCTIVITY, linewidth=2.5,
                 marker="o", markersize=4, markeredgecolor=COLORS.BG_DARK,
                 markeredgewidth=1, label="Production")

        ax1.set_ylabel("Total Production", fontsize=11, color=COLORS.PRODUCTIVITY)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_title("Production & Tax Revenue", fontsize=13, pad=12, fontweight="bold")

        ax1r = ax1.twinx()
        ax1r.fill_between(epochs, metrics["total_tax_revenue"],
                          alpha=0.08, color=COLORS.REVENUE)
        ax1r.plot(epochs, metrics["total_tax_revenue"],
                  color=COLORS.REVENUE, linewidth=2, linestyle="--",
                  label="Tax Revenue")
        ax1r.set_ylabel("Tax Revenue", fontsize=11, color=COLORS.REVENUE)
        ax1r.spines["right"].set_visible(True)
        ax1r.spines["right"].set_color(COLORS.ACCENT_BORDER)
        ax1r.tick_params(axis="y", colors=COLORS.TEXT_MUTED)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1r.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

        # ── Panel 2: Tax Schedule Evolution ──
        ax3 = fig.add_subplot(gs[1, 2])
        xi, yi = _bracket_steps(_INITIAL_BRACKETS)
        xf, yf = _bracket_steps(tax_schedule["brackets"])

        ax3.fill_between(xf, yf, alpha=0.12, step="post", color=COLORS.PLANNER)
        ax3.plot(xi, yi, color=COLORS.TEXT_MUTED, linewidth=2,
                 linestyle="--", label="Initial (flat 10%)", drawstyle="steps-post")
        ax3.plot(xf, yf, color=COLORS.PLANNER, linewidth=3,
                 label="Final (learned)", drawstyle="steps-post")

        # Annotate final bracket rates
        for b in tax_schedule["brackets"]:
            ax3.annotate(
                f"{b['rate']:.0%}",
                xy=(b["threshold"] + 2, b["rate"]),
                fontsize=9, color=COLORS.PLANNER, fontweight="bold",
            )

        ax3.set_xlabel("Income Bracket", fontsize=11)
        ax3.set_ylabel("Marginal Tax Rate", fontsize=11)
        ax3.set_title("Tax Schedule Evolution", fontsize=13, pad=12, fontweight="bold")
        ax3.set_ylim(0, 0.7)
        ax3.set_xlim(-2, 80)
        ax3.legend(fontsize=10, loc="lower right")

        # ── Panel 3: Inequality (Gini + Welfare) ──
        ax2 = fig.add_subplot(gs[2, :2])
        plot_enhanced_line(
            ax2, epochs, metrics["gini_coefficient"],
            COLORS.EVASION, label="Gini Coefficient", show_annotations=False,
        )
        ax2.set_ylabel("Gini Coefficient", fontsize=11, color=COLORS.EVASION)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_title("Inequality & Welfare", fontsize=13, pad=12, fontweight="bold")
        ax2.set_ylim(0, 0.65)
        add_danger_zone(ax2, 0.5, direction="above", label="high inequality")

        ax2r = ax2.twinx()
        ax2r.plot(epochs, metrics["welfare"],
                  color=COLORS.WELFARE, linewidth=2, linestyle=":",
                  marker="s", markersize=3, label="Welfare")
        ax2r.set_ylabel("Welfare", fontsize=11, color=COLORS.WELFARE)
        ax2r.spines["right"].set_visible(True)
        ax2r.spines["right"].set_color(COLORS.ACCENT_BORDER)
        ax2r.tick_params(axis="y", colors=COLORS.TEXT_MUTED)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")

        # ── Panel 4: Tax rate over time ──
        ax4 = fig.add_subplot(gs[2, 2])
        plot_enhanced_line(
            ax4, epochs, metrics["mean_effective_tax_rate"],
            COLORS.PLANNER, label="Effective Tax Rate", show_annotations=False,
        )
        # End-value callout
        final_rate = metrics["mean_effective_tax_rate"].iloc[-1]
        ax4.annotate(
            f"  {final_rate:.1%}", xy=(epochs.iloc[-1], final_rate),
            fontsize=11, color=COLORS.PLANNER, fontweight="bold", va="center",
        )
        ax4.set_ylabel("Mean Effective Rate", fontsize=11)
        ax4.set_xlabel("Epoch", fontsize=11)
        ax4.set_title("Effective Tax Rate", fontsize=13, pad=12, fontweight="bold")
        ax4.set_ylim(0, 0.5)

        fig.suptitle(
            "AI Economist GTB — Economy Dashboard",
            fontsize=16, y=0.98, fontweight="bold",
        )

        out = plots_dir / "gtb_economy_dashboard.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return out


def plot_adversarial_dynamics(
    metrics: pd.DataFrame,
    workers: pd.DataFrame,
    plots_dir: Path,
) -> Path:
    """Figure 2: Adversarial Dynamics — 2x2 layout."""
    with swarm_theme():
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.35)

        epochs = metrics["epoch"]

        # ── Panel 1: Agent Wealth by Type (horizontal bar) ──
        ax1 = fig.add_subplot(gs[0, 0])
        w = workers.copy()
        w["type"] = w["agent_id"].apply(_agent_type)
        w["house_value"] = w["houses_built"] * 50
        w["total_wealth"] = w["coin"] + w["house_value"]
        w = w.sort_values("total_wealth", ascending=True)

        bar_colors = [_AGENT_TYPE_COLORS.get(t, COLORS.OPPORTUNISTIC) for t in w["type"]]

        # Cleaner labels: just type + short index
        labels = []
        for aid in w["agent_id"]:
            parts = aid.split("_")
            labels.append(f"{parts[1].capitalize()} {parts[-1]}")

        ax1.barh(
            labels, w["coin"],
            color=[to_rgba(c, 0.85) for c in bar_colors],
            edgecolor=[to_rgba(c, 1.0) for c in bar_colors],
            linewidth=0.8, height=0.7,
        )
        ax1.barh(
            labels, w["house_value"], left=w["coin"],
            color=[to_rgba(c, 0.3) for c in bar_colors],
            edgecolor=[to_rgba(c, 0.8) for c in bar_colors],
            linewidth=0.8, height=0.7,
        )

        # Wealth value at end of bar
        for i, (_, row) in enumerate(w.iterrows()):
            if row["total_wealth"] > 50:
                ax1.text(row["total_wealth"] + 15, i, f"{row['total_wealth']:.0f}",
                         va="center", fontsize=8, color=COLORS.TEXT_MUTED)

        ax1.set_xlabel("Wealth (Coin + Houses)", fontsize=11)
        ax1.set_title("Agent Wealth by Policy Type", fontsize=13, pad=12, fontweight="bold")
        ax1.tick_params(axis="y", labelsize=9)

        type_patches = [
            Patch(facecolor=c, edgecolor=to_rgba(c, 0.8), label=t.capitalize())
            for t, c in _AGENT_TYPE_COLORS.items()
        ]
        ax1.legend(handles=type_patches, fontsize=9, loc="lower right",
                   title="Policy Type", title_fontsize=9)

        # ── Panel 2: Enforcement Activity ──
        ax2 = fig.add_subplot(gs[0, 1])

        # Smoother stacked area
        audits = metrics["total_audits"]
        catches = metrics["total_catches"]
        fines = metrics["total_fines"]

        ax2.fill_between(epochs, 0, audits,
                         alpha=0.5, color=COLORS.WELFARE, label="Audits")
        ax2.fill_between(epochs, audits, audits + catches,
                         alpha=0.5, color=COLORS.DECEPTIVE, label="Catches")
        ax2.plot(epochs, audits + catches,
                 color=COLORS.DECEPTIVE, linewidth=1, alpha=0.5)

        ax2.set_ylabel("Audit / Catch Count", fontsize=11)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_title("Enforcement Activity", fontsize=13, pad=12, fontweight="bold")

        # Fines on twin axis (different scale)
        ax2r = ax2.twinx()
        ax2r.fill_between(epochs, fines, alpha=0.12, color=COLORS.TOXICITY)
        ax2r.plot(epochs, fines,
                  color=COLORS.TOXICITY, linewidth=2, linestyle="--", label="Fines ($)")
        ax2r.set_ylabel("Fine Amount", fontsize=11, color=COLORS.TOXICITY)
        ax2r.spines["right"].set_visible(True)
        ax2r.spines["right"].set_color(COLORS.ACCENT_BORDER)
        ax2r.tick_params(axis="y", colors=COLORS.TEXT_MUTED)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

        # ── Panel 3: Collusion Detection ──
        ax3 = fig.add_subplot(gs[1, 0])
        plot_enhanced_line(
            ax3, epochs, metrics["collusion_events_detected"],
            COLORS.ADVERSARIAL, label="Events Detected", show_annotations=False,
        )
        # Just annotate the peak
        peak_idx = metrics["collusion_events_detected"].idxmax()
        peak_val = metrics["collusion_events_detected"].iloc[peak_idx]
        ax3.annotate(
            f"peak: {peak_val:.0f}",
            xy=(epochs.iloc[peak_idx], peak_val),
            xytext=(0, 14), textcoords="offset points",
            fontsize=10, color=COLORS.ADVERSARIAL, fontweight="bold",
            ha="center",
            arrowprops={"arrowstyle": "->", "color": COLORS.ADVERSARIAL, "lw": 1.2},
        )

        ax3.set_ylabel("Events per Epoch", fontsize=11, color=COLORS.ADVERSARIAL)
        ax3.set_xlabel("Epoch", fontsize=11)
        ax3.set_title("Collusion Detection", fontsize=13, pad=12, fontweight="bold")

        ax3r = ax3.twinx()
        ax3r.plot(epochs, metrics["collusion_suspicion_mean"],
                  color=COLORS.PLANNER, linewidth=2, linestyle=":",
                  label="Suspicion Score")
        ax3r.set_ylabel("Mean Suspicion", fontsize=11, color=COLORS.PLANNER)
        ax3r.set_ylim(0.6, 0.95)
        ax3r.spines["right"].set_visible(True)
        ax3r.spines["right"].set_color(COLORS.ACCENT_BORDER)
        ax3r.tick_params(axis="y", colors=COLORS.TEXT_MUTED)

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3r.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

        # ── Panel 4: Bunching & Evasion ──
        ax4 = fig.add_subplot(gs[1, 1])
        plot_enhanced_line(
            ax4, epochs, metrics["bunching_intensity"],
            COLORS.EVASION, label="Bunching Intensity", show_annotations=False,
        )
        ax4.set_ylabel("Bunching Intensity", fontsize=11, color=COLORS.EVASION)
        ax4.set_xlabel("Epoch", fontsize=11)
        ax4.set_title("Tax Evasion Signals", fontsize=13, pad=12, fontweight="bold")

        ax4r = ax4.twinx()
        ax4r.plot(epochs, metrics["exploit_frequency"],
                  color=COLORS.TOXICITY, linewidth=2, linestyle="--",
                  marker="v", markersize=4, label="Exploit Frequency")
        ax4r.set_ylabel("Exploit Freq", fontsize=11, color=COLORS.TOXICITY)
        ax4r.spines["right"].set_visible(True)
        ax4r.spines["right"].set_color(COLORS.ACCENT_BORDER)
        ax4r.tick_params(axis="y", colors=COLORS.TEXT_MUTED)

        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4r.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

        fig.suptitle(
            "AI Economist GTB — Adversarial Dynamics",
            fontsize=16, y=0.98, fontweight="bold",
        )

        out = plots_dir / "gtb_adversarial_dynamics.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate AI Economist GTB visualizations"
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Run folder (e.g. runs/20260215_095359_ai_economist_seed42)",
    )
    parser.add_argument(
        "--mode", default="dark", choices=["dark", "light"],
        help="Theme mode (default: dark)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    csv_dir = run_dir / "csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    metrics = pd.read_csv(csv_dir / "metrics.csv")
    workers = pd.read_csv(csv_dir / "workers.csv")
    with open(csv_dir / "tax_schedule.json") as f:
        tax_schedule = json.load(f)

    print(f"Loaded {len(metrics)} epochs, {len(workers)} workers")

    out1 = plot_economy_dashboard(metrics, tax_schedule, plots_dir)
    print(f"  {out1}")

    out2 = plot_adversarial_dynamics(metrics, workers, plots_dir)
    print(f"  {out2}")

    print(f"\nAll plots saved to {plots_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
