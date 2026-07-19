#!/usr/bin/env python
"""
Generate publication-quality metric graphs from SWARM simulations.

Runs multiple scenarios, collects metrics, and produces matplotlib PNG plots
including welfare-vs-toxicity curves, time-series overlays, agent comparisons,
and metric correlation scatter plots.

Usage:
    python examples/generate_metric_graphs.py
    python examples/generate_metric_graphs.py --scenarios baseline strict_governance
    python examples/generate_metric_graphs.py --out-dir runs/graphs
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure matplotlib for headless rendering before import
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Data container for collected metrics (superset of EpochMetrics)
# ---------------------------------------------------------------------------


@dataclass
class CollectedEpoch:
    """Epoch-level metrics collected from the orchestrator."""

    epoch: int = 0
    total_interactions: int = 0
    accepted_interactions: int = 0
    total_posts: int = 0
    total_votes: int = 0
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    avg_payoff: float = 0.0
    total_welfare: float = 0.0
    # Augmented from orchestrator state
    avg_reputation: float = 0.0
    reputation_std: float = 0.0
    n_agents: int = 0
    n_frozen: int = 0


@dataclass
class AgentFinalState:
    """Final state of a single agent."""

    agent_id: str = ""
    agent_type: str = ""
    reputation: float = 0.0
    resources: float = 0.0
    total_payoff: float = 0.0
    is_frozen: bool = False


@dataclass
class ScenarioResult:
    """Complete result from running one scenario."""

    scenario_id: str = ""
    epochs: List[CollectedEpoch] = field(default_factory=list)
    agents: List[AgentFinalState] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
]

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "strict_governance": "Strict Governance",
    "adversarial_redteam": "Adversarial Red-Team",
    "security_evaluation": "Security Evaluation",
    "network_effects": "Network Effects",
    "collusion_detection": "Collusion Detection",
}

DPI = 160
FIGSIZE_WIDE = (10, 5.5)
FIGSIZE_SQUARE = (7, 6)
FIGSIZE_TALL = (10, 7)


def _style_ax(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def run_scenario_collect(scenario_path: Path) -> ScenarioResult:
    """Run a scenario and collect all metrics into a ScenarioResult."""
    scenario = load_scenario(scenario_path)
    orchestrator = build_orchestrator(scenario)

    result = ScenarioResult(scenario_id=scenario.scenario_id)

    # Capture per-epoch reputation snapshots via callback so we get
    # the actual evolving reputation stats, not just end-state values.
    epoch_rep_snapshots: List[Dict[str, float]] = []

    def _capture_reputation(_em: Any) -> None:
        agents = orchestrator.get_all_agents()
        reps = [orchestrator.state.get_agent(a.agent_id).reputation for a in agents]
        n_frozen = len(orchestrator.state.frozen_agents)
        epoch_rep_snapshots.append({
            "avg_reputation": float(np.mean(reps)) if reps else 0.0,
            "reputation_std": float(np.std(reps)) if len(reps) > 1 else 0.0,
            "n_agents": len(agents),
            "n_frozen": n_frozen,
        })

    orchestrator.on_epoch_end(_capture_reputation)

    # Run the full simulation
    epoch_metrics_list = orchestrator.run()

    # Collect epoch data with per-epoch reputation stats
    for i, em in enumerate(epoch_metrics_list):
        rep_snap = epoch_rep_snapshots[i] if i < len(epoch_rep_snapshots) else {}
        ce = CollectedEpoch(
            epoch=em.epoch,
            total_interactions=em.total_interactions,
            accepted_interactions=em.accepted_interactions,
            total_posts=em.total_posts,
            total_votes=em.total_votes,
            toxicity_rate=em.toxicity_rate,
            quality_gap=em.quality_gap,
            avg_payoff=em.avg_payoff,
            total_welfare=em.total_welfare,
            avg_reputation=rep_snap.get("avg_reputation", 0.0),
            reputation_std=rep_snap.get("reputation_std", 0.0),
            n_agents=rep_snap.get("n_agents", 0),
            n_frozen=rep_snap.get("n_frozen", 0),
        )
        result.epochs.append(ce)

    # Collect final agent states
    all_agents = orchestrator.get_all_agents()
    frozen_set = orchestrator.state.frozen_agents
    for agent in all_agents:
        state = orchestrator.state.get_agent(agent.agent_id)
        result.agents.append(
            AgentFinalState(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type.value,
                reputation=state.reputation,
                resources=state.resources,
                total_payoff=state.total_payoff,
                is_frozen=agent.agent_id in frozen_set,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------


def plot_welfare_vs_toxicity(
    results: Dict[str, ScenarioResult],
    out_dir: Path,
) -> Path:
    """
    Scatter plot: total_welfare (x) vs toxicity_rate (y) for each epoch,
    colored by scenario.  The key trade-off visualization.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    for idx, (sid, sr) in enumerate(results.items()):
        welfare = [e.total_welfare for e in sr.epochs]
        toxicity = [e.toxicity_rate for e in sr.epochs]
        color = COLORS[idx % len(COLORS)]
        label = SCENARIO_LABELS.get(sid, sid)

        ax.scatter(
            welfare,
            toxicity,
            c=color,
            label=label,
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )
        # Draw connecting line showing temporal progression
        ax.plot(welfare, toxicity, color=color, alpha=0.3, linewidth=1)

    _style_ax(ax, "Welfare vs. Toxicity Trade-off", "Total Welfare", "Toxicity Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, framealpha=0.9, loc="best")

    out = out_dir / "welfare_vs_toxicity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metric_time_series(
    results: Dict[str, ScenarioResult],
    metric: str,
    out_dir: Path,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Path:
    """Time-series overlay of a single metric across scenarios."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for idx, (sid, sr) in enumerate(results.items()):
        epochs = [e.epoch for e in sr.epochs]
        values = [getattr(e, metric, 0.0) for e in sr.epochs]
        color = COLORS[idx % len(COLORS)]
        label = SCENARIO_LABELS.get(sid, sid)
        ax.plot(
            epochs, values, color=color, label=label, linewidth=2, marker="o", markersize=4
        )

    title = metric.replace("_", " ").title() + " Over Time"
    _style_ax(ax, title, "Epoch", ylabel or metric.replace("_", " ").title())
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9, framealpha=0.9, loc="best")

    out = out_dir / f"{metric}_time_series.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_multi_metric_panel(
    results: Dict[str, ScenarioResult],
    out_dir: Path,
) -> Path:
    """2x2 panel plot: toxicity, quality_gap, total_welfare, avg_payoff."""
    metrics = [
        ("toxicity_rate", "Toxicity Rate", (-0.05, 1.05)),
        ("quality_gap", "Quality Gap", None),
        ("total_welfare", "Total Welfare", None),
        ("avg_payoff", "Avg Payoff", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TALL)
    axes_flat = axes.flatten()

    for panel_idx, (metric, ylabel, ylim) in enumerate(metrics):
        ax = axes_flat[panel_idx]
        for idx, (sid, sr) in enumerate(results.items()):
            epochs = [e.epoch for e in sr.epochs]
            values = [getattr(e, metric, 0.0) for e in sr.epochs]
            color = COLORS[idx % len(COLORS)]
            label = SCENARIO_LABELS.get(sid, sid)
            ax.plot(
                epochs,
                values,
                color=color,
                label=label,
                linewidth=1.8,
                marker="o",
                markersize=3,
            )

        _style_ax(ax, ylabel, "Epoch", "")
        if ylim:
            ax.set_ylim(*ylim)
        if panel_idx == 0:
            ax.legend(fontsize=7, framealpha=0.9, loc="best")

    fig.suptitle(
        "Key Metrics Comparison Across Scenarios", fontsize=14, fontweight="bold", y=1.02
    )
    out = out_dir / "multi_metric_panel.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_acceptance_rate(
    results: Dict[str, ScenarioResult],
    out_dir: Path,
) -> Path:
    """Acceptance rate (accepted/total) over time per scenario."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for idx, (sid, sr) in enumerate(results.items()):
        epochs = [e.epoch for e in sr.epochs]
        rates = []
        for e in sr.epochs:
            denom = float(e.total_interactions) if e.total_interactions else 0.0
            rates.append(float(e.accepted_interactions) / denom if denom > 0 else 0.0)
        color = COLORS[idx % len(COLORS)]
        label = SCENARIO_LABELS.get(sid, sid)
        ax.plot(
            epochs, rates, color=color, label=label, linewidth=2, marker="o", markersize=4
        )

    _style_ax(ax, "Acceptance Rate Over Time", "Epoch", "Accepted / Total")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, framealpha=0.9, loc="best")

    out = out_dir / "acceptance_rate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scenario_summary_bars(
    results: Dict[str, ScenarioResult],
    out_dir: Path,
) -> Path:
    """Bar chart comparing average metrics across scenarios."""
    metrics = ["toxicity_rate", "quality_gap", "total_welfare", "avg_payoff"]
    n_scenarios = len(results)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_scenarios, 1)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for idx, (sid, sr) in enumerate(results.items()):
        avgs = []
        for m in metrics:
            vals = [getattr(e, m, 0.0) for e in sr.epochs]
            avgs.append(float(np.mean(vals)) if vals else 0.0)
        color = COLORS[idx % len(COLORS)]
        label = SCENARIO_LABELS.get(sid, sid)
        offset = (idx - n_scenarios / 2 + 0.5) * width
        bars = ax.bar(x + offset, avgs, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, avgs, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n").title() for m in metrics], fontsize=9)
    _style_ax(ax, "Average Metrics by Scenario", "", "Value")
    ax.legend(fontsize=9, framealpha=0.9, loc="best")

    out = out_dir / "scenario_summary_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_toxicity_vs_quality_gap(
    results: Dict[str, ScenarioResult],
    out_dir: Path,
) -> Path:
    """Scatter: toxicity_rate vs quality_gap (adverse selection indicator)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    for idx, (sid, sr) in enumerate(results.items()):
        toxicity = [e.toxicity_rate for e in sr.epochs]
        qgap = [e.quality_gap for e in sr.epochs]
        color = COLORS[idx % len(COLORS)]
        label = SCENARIO_LABELS.get(sid, sid)
        ax.scatter(
            toxicity,
            qgap,
            c=color,
            label=label,
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    # Reference lines
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    _style_ax(
        ax,
        "Toxicity vs. Quality Gap (Adverse Selection)",
        "Toxicity Rate",
        "Quality Gap (negative = adverse selection)",
    )
    ax.legend(fontsize=9, framealpha=0.9, loc="best")

    out = out_dir / "toxicity_vs_quality_gap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_agent_payoff_comparison(
    results: Dict[str, ScenarioResult],
    out_dir: Path,
) -> Path:
    """Grouped bar chart: final agent payoffs by type across scenarios."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Aggregate payoffs by agent type per scenario
    all_types: List[str] = []
    for sr in results.values():
        for a in sr.agents:
            if a.agent_type not in all_types:
                all_types.append(a.agent_type)
    all_types.sort()

    n_scenarios = len(results)
    n_types = len(all_types)
    x = np.arange(n_types)
    width = 0.8 / max(n_scenarios, 1)

    for idx, (sid, sr) in enumerate(results.items()):
        # Average payoff per agent type
        type_payoffs: Dict[str, List[float]] = {t: [] for t in all_types}
        for a in sr.agents:
            type_payoffs[a.agent_type].append(a.total_payoff)

        avgs = [
            float(np.mean(type_payoffs[t])) if type_payoffs[t] else 0.0 for t in all_types
        ]
        color = COLORS[idx % len(COLORS)]
        label = SCENARIO_LABELS.get(sid, sid)
        offset = (idx - n_scenarios / 2 + 0.5) * width
        ax.bar(x + offset, avgs, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title() for t in all_types], fontsize=9)
    _style_ax(ax, "Agent Payoff by Type Across Scenarios", "Agent Type", "Avg Total Payoff")
    ax.legend(fontsize=9, framealpha=0.9, loc="best")

    out = out_dir / "agent_payoff_by_type.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_agent_reputation_bars(
    results: Dict[str, ScenarioResult],
    out_dir: Path,
) -> Path:
    """Horizontal bar chart: final reputation of all agents per scenario."""
    n_scenarios = len(results)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), squeeze=False)

    for idx, (sid, sr) in enumerate(results.items()):
        ax = axes[0][idx]
        agents_sorted = sorted(sr.agents, key=lambda a: a.reputation, reverse=True)
        names = [f"{a.agent_id}\n({a.agent_type})" for a in agents_sorted]
        reps = [a.reputation for a in agents_sorted]
        bar_colors = []
        for a in agents_sorted:
            if a.is_frozen:
                bar_colors.append("#E0E0E0")
            elif a.agent_type in ("adversarial", "adaptive_adversary", "deceptive"):
                bar_colors.append("#F44336")
            elif a.agent_type == "opportunistic":
                bar_colors.append("#FF9800")
            else:
                bar_colors.append("#4CAF50")

        ax.barh(names, reps, color=bar_colors, alpha=0.85, edgecolor="white")
        _style_ax(ax, SCENARIO_LABELS.get(sid, sid), "Reputation", "")

    out = out_dir / "agent_reputation_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_welfare_toxicity_frontier(
    results: Dict[str, ScenarioResult],
    out_dir: Path,
) -> Path:
    """
    Welfare-toxicity frontier: for each scenario, plot the mean +/- std
    as an ellipse/errorbar showing where it sits on the trade-off curve.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    for idx, (sid, sr) in enumerate(results.items()):
        welfare = [e.total_welfare for e in sr.epochs]
        toxicity = [e.toxicity_rate for e in sr.epochs]
        color = COLORS[idx % len(COLORS)]
        label = SCENARIO_LABELS.get(sid, sid)

        w_mean, w_std = float(np.mean(welfare)), float(np.std(welfare))
        t_mean, t_std = float(np.mean(toxicity)), float(np.std(toxicity))

        ax.errorbar(
            w_mean,
            t_mean,
            xerr=w_std,
            yerr=t_std,
            fmt="o",
            color=color,
            markersize=12,
            capsize=6,
            capthick=2,
            linewidth=2,
            label=label,
        )
        # Annotate
        ax.annotate(
            label,
            (w_mean, t_mean),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8,
            color=color,
        )

    _style_ax(
        ax,
        "Welfare-Toxicity Frontier (mean +/- std)",
        "Total Welfare",
        "Toxicity Rate",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, framealpha=0.9, loc="best")

    out = out_dir / "welfare_toxicity_frontier.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CSV export (lightweight, no dependency on swarm.analysis.export)
# ---------------------------------------------------------------------------
def _export_epochs_csv(result: ScenarioResult, path: Path) -> None:
    """Write epoch-level metrics to CSV."""
    import csv

    fieldnames = [
        "scenario_id",
        "epoch",
        "total_interactions",
        "accepted_interactions",
        "toxicity_rate",
        "quality_gap",
        "avg_payoff",
        "total_welfare",
        "total_posts",
        "total_votes",
        "n_agents",
        "n_frozen",
        "avg_reputation",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in result.epochs:
            writer.writerow(
                {
                    "scenario_id": result.scenario_id,
                    "epoch": e.epoch,
                    "total_interactions": e.total_interactions,
                    "accepted_interactions": e.accepted_interactions,
                    "toxicity_rate": f"{e.toxicity_rate:.6f}",
                    "quality_gap": f"{e.quality_gap:.6f}",
                    "avg_payoff": f"{e.avg_payoff:.6f}",
                    "total_welfare": f"{e.total_welfare:.6f}",
                    "total_posts": e.total_posts,
                    "total_votes": e.total_votes,
                    "n_agents": e.n_agents,
                    "n_frozen": e.n_frozen,
                    "avg_reputation": f"{e.avg_reputation:.6f}",
                }
            )


def _export_agents_csv(result: ScenarioResult, path: Path) -> None:
    """Write final agent states to CSV."""
    import csv

    fieldnames = [
        "scenario_id",
        "agent_id",
        "agent_type",
        "reputation",
        "resources",
        "total_payoff",
        "is_frozen",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in result.agents:
            writer.writerow(
                {
                    "scenario_id": result.scenario_id,
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "reputation": f"{a.reputation:.6f}",
                    "resources": f"{a.resources:.6f}",
                    "total_payoff": f"{a.total_payoff:.6f}",
                    "is_frozen": a.is_frozen,
                }
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DEFAULT_SCENARIOS = ["baseline", "strict_governance", "adversarial_redteam"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SWARM metric graphs")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Scenario IDs to run (names without .yaml extension)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/<timestamp>_metric_graphs/)",
    )
    args = parser.parse_args()

    # Resolve output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or Path(f"runs/{timestamp}_metric_graphs")
    plots_dir = out_dir / "plots"
    csv_dir = out_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    scenarios_dir = Path("scenarios")

    # Collect results per scenario
    results: Dict[str, ScenarioResult] = {}

    for scenario_name in args.scenarios:
        yaml_path = scenarios_dir / f"{scenario_name}.yaml"
        if not yaml_path.exists():
            # Try subdirectories
            candidates = list(scenarios_dir.rglob(f"{scenario_name}.yaml"))
            if candidates:
                yaml_path = candidates[0]
            else:
                print(f"WARNING: Scenario file not found: {yaml_path}, skipping.")
                continue

        print(f"Running scenario: {scenario_name} ({yaml_path})")
        sr = run_scenario_collect(yaml_path)
        results[sr.scenario_id] = sr

        # Export CSV data
        _export_epochs_csv(sr, csv_dir / f"{sr.scenario_id}_epochs.csv")
        _export_agents_csv(sr, csv_dir / f"{sr.scenario_id}_agents.csv")
        print(f"  -> {len(sr.epochs)} epochs, {len(sr.agents)} agents collected")

    if not results:
        print("ERROR: No scenarios produced data.")
        return 1

    # Generate plots
    print("\nGenerating plots...")
    written: List[Path] = []

    plot_funcs = [
        ("welfare_vs_toxicity", plot_welfare_vs_toxicity),
        ("multi_metric_panel", plot_multi_metric_panel),
        ("acceptance_rate", plot_acceptance_rate),
        ("scenario_summary_bars", plot_scenario_summary_bars),
        ("toxicity_vs_quality_gap", plot_toxicity_vs_quality_gap),
        ("agent_payoff_by_type", plot_agent_payoff_comparison),
        ("agent_reputation_bars", plot_agent_reputation_bars),
        ("welfare_toxicity_frontier", plot_welfare_toxicity_frontier),
    ]

    for name, func in plot_funcs:
        try:
            p = func(results, plots_dir)
            written.append(p)
            print(f"  -> {p.name}")
        except Exception as err:
            print(f"  WARNING: Failed to generate {name}: {err}")

    # Individual time-series plots
    for metric, ylabel, ylim in [
        ("toxicity_rate", "Toxicity Rate", (-0.05, 1.05)),
        ("quality_gap", "Quality Gap", None),
        ("total_welfare", "Total Welfare", None),
        ("avg_payoff", "Avg Payoff", None),
    ]:
        try:
            p = plot_metric_time_series(results, metric, plots_dir, ylabel, ylim)
            written.append(p)
            print(f"  -> {p.name}")
        except Exception as err:
            print(f"  WARNING: Failed to generate {metric} time series: {err}")

    # Write manifest
    manifest = plots_dir / "README.txt"
    manifest.write_text(
        "\n".join(
            [
                "Generated metric graphs:",
                f"Date: {datetime.now().isoformat()}",
                f"Scenarios: {', '.join(results.keys())}",
                "",
                "Plots:",
                *(f"  - {p.name}" for p in written),
                "",
            ]
        )
    )

    print(f"\nAll {len(written)} plots written to: {plots_dir}/")
    print(f"Data exported to: {csv_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
