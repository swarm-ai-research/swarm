#!/usr/bin/env python
"""
GasTown Workspace Composition Study

Study how adversarial agent proportion affects a GasTown-style workspace,
and whether governance levers effectively contain the damage.

Methodology:
  - Fix total population at 7 agents (matching gastown_workspace.yaml)
  - Sweep adversarial (rogue) proportion from 0 to 6 out of 7 agents
  - Remaining slots filled with honest agents
  - Run each config under three governance regimes:
    - "ungoverned": all governance levers OFF (baseline)
    - "governed": circuit breaker + collusion detection + staking + auditing ON
    - "refinery": Refinery quality gate only (reject p < 0.5, zero friction on accepted)
  - Use GasTown payoff parameters (s_plus=3, s_minus=1.5, h=2.5)
  - Run each configuration for 30 epochs, 10 steps/epoch, 3 seeds
  - Compare welfare, toxicity, adverse selection, and per-class payoffs

Safety questions studied:
  - At what rogue proportion does the workspace break down?
  - How much damage do governance levers prevent?
  - Do honest agents maintain positive payoff under adversarial pressure?
  - Is there a critical threshold where governance can no longer contain rogues?
  - Does the Refinery achieve most toxicity reduction at a fraction of welfare cost?

Usage:
    python examples/gastown_composition_study.py
    python examples/gastown_composition_study.py --total-agents 7 --epochs 30 --steps 10 --seeds 3

Smoke test:
    python examples/gastown_composition_study.py --total-agents 4 --epochs 3 --steps 3 --seeds 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm.agents.adversarial import AdversarialAgent
from swarm.agents.base import BaseAgent
from swarm.agents.honest import HonestAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
REGIMES = ("ungoverned", "governed", "refinery")


@dataclass
class CompositionConfig:
    """A single population composition to test."""

    label: str
    regime: str  # "ungoverned", "governed", or "refinery"
    n_honest: int
    n_opportunistic: int
    n_adversarial: int

    @property
    def total(self) -> int:
        return self.n_honest + self.n_opportunistic + self.n_adversarial

    @property
    def rogue_pct(self) -> float:
        """Percentage of adversarial agents."""
        return self.n_adversarial / self.total if self.total else 0.0


@dataclass
class RunResult:
    """Result from one run (one composition + one seed)."""

    composition: str
    regime: str
    rogue_pct: float
    seed: int
    n_epochs: int
    # Aggregates across all epochs
    mean_welfare: float = 0.0
    total_welfare: float = 0.0
    mean_toxicity: float = 0.0
    mean_quality_gap: float = 0.0
    mean_avg_payoff: float = 0.0
    final_welfare: float = 0.0
    final_toxicity: float = 0.0
    # Per-class average payoffs
    honest_avg_payoff: float = 0.0
    opportunistic_avg_payoff: float = 0.0
    adversarial_avg_payoff: float = 0.0


@dataclass
class AggResult:
    """Aggregated result across seeds for one composition."""

    label: str
    regime: str
    rogue_pct: float
    n_seeds: int
    welfare_mean: float
    welfare_std: float
    welfare_total_mean: float
    welfare_total_std: float
    toxicity_mean: float
    toxicity_std: float
    quality_gap_mean: float
    avg_payoff_mean: float
    # Per-class payoff means
    honest_payoff_mean: float
    opportunistic_payoff_mean: float
    adversarial_payoff_mean: float


# ---------------------------------------------------------------------------
# Governance configs
# ---------------------------------------------------------------------------
def _governed_config() -> GovernanceConfig:
    """GasTown governance: circuit breaker + collusion + staking + auditing."""
    return GovernanceConfig(
        # Transaction tax
        transaction_tax_rate=0.05,
        transaction_tax_split=0.5,
        # Reputation decay
        reputation_decay_rate=0.95,
        # Circuit breaker
        circuit_breaker_enabled=True,
        freeze_threshold_toxicity=0.5,
        freeze_threshold_violations=3,
        freeze_duration_epochs=2,
        # Random audit
        audit_enabled=True,
        audit_probability=0.2,
        audit_penalty_multiplier=2.0,
        audit_threshold_p=0.5,
        # Staking
        staking_enabled=True,
        min_stake_to_participate=5.0,
        stake_slash_rate=0.2,
        # Collusion detection
        collusion_detection_enabled=True,
        collusion_frequency_threshold=2.0,
        collusion_correlation_threshold=0.7,
        collusion_min_interactions=3,
        collusion_score_threshold=0.5,
        collusion_penalty_multiplier=1.5,
        collusion_realtime_penalty=True,
        collusion_realtime_rate=0.1,
        # Bandwidth cap
        bandwidth_cap=8,
    )


def _ungoverned_config() -> GovernanceConfig:
    """Baseline: no governance levers enabled."""
    return GovernanceConfig(
        reputation_decay_rate=1.0,  # no decay
        circuit_breaker_enabled=False,
        audit_enabled=False,
        staking_enabled=False,
        collusion_detection_enabled=False,
    )


def _refinery_config() -> GovernanceConfig:
    """Refinery-only: deterministic quality gate, zero friction on accepted."""
    return GovernanceConfig(
        refinery_enabled=True,
        refinery_p_threshold=0.5,
        # Everything else off
        reputation_decay_rate=1.0,
        circuit_breaker_enabled=False,
        audit_enabled=False,
        staking_enabled=False,
        collusion_detection_enabled=False,
    )


_REGIME_CONFIG = {
    "ungoverned": _ungoverned_config,
    "governed": _governed_config,
    "refinery": _refinery_config,
}


# ---------------------------------------------------------------------------
# Composition builder
# ---------------------------------------------------------------------------
def build_compositions(total_agents: int) -> List[CompositionConfig]:
    """Build compositions for all regime sweeps.

    Sweeps adversarial proportion from 0 to (total-1)/total.
    Keeps 1 opportunistic agent (the corner-cutter) when possible,
    fills remaining with honest.
    """
    compositions: List[CompositionConfig] = []

    for regime in REGIMES:
        for n_adversarial in range(total_agents):
            remaining = total_agents - n_adversarial
            # Keep 1 opportunistic when there's room for at least 2 non-adversarial
            n_opportunistic = 1 if remaining >= 2 else 0
            n_honest = remaining - n_opportunistic

            label = f"{regime.title()} {n_adversarial}/{total_agents} rogue"

            compositions.append(
                CompositionConfig(
                    label=label,
                    regime=regime,
                    n_honest=n_honest,
                    n_opportunistic=n_opportunistic,
                    n_adversarial=n_adversarial,
                )
            )

    return compositions


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------
def create_agents_from_config(comp: CompositionConfig) -> List[BaseAgent]:
    """Create agent instances from a composition config."""
    agents: List[BaseAgent] = []
    for i in range(comp.n_honest):
        agents.append(HonestAgent(agent_id=f"honest_{i + 1}"))
    for i in range(comp.n_opportunistic):
        agents.append(OpportunisticAgent(agent_id=f"opportunistic_{i + 1}"))
    for i in range(comp.n_adversarial):
        agents.append(AdversarialAgent(agent_id=f"rogue_{i + 1}"))
    return agents


# ---------------------------------------------------------------------------
# Per-class payoff extraction
# ---------------------------------------------------------------------------
def _extract_per_class_payoffs(
    orchestrator: Orchestrator,
) -> Dict[str, List[float]]:
    """Extract per-class payoffs using isinstance."""
    payoffs: Dict[str, List[float]] = {
        "honest": [],
        "opportunistic": [],
        "adversarial": [],
    }
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        if state is None:
            continue
        if isinstance(agent, AdversarialAgent):
            payoffs["adversarial"].append(state.total_payoff)
        elif isinstance(agent, OpportunisticAgent):
            payoffs["opportunistic"].append(state.total_payoff)
        elif isinstance(agent, HonestAgent):
            payoffs["honest"].append(state.total_payoff)
    return payoffs


def _mean_or_zero(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(
    comp: CompositionConfig,
    seed: int,
    n_epochs: int,
    steps_per_epoch: int,
) -> RunResult:
    """Run one simulation for a given composition and seed."""
    # GasTown payoff parameters from gastown_workspace.yaml
    payoff_config = PayoffConfig(
        s_plus=3.0,
        s_minus=1.5,
        h=2.5,
        theta=0.5,
        rho_a=0.3,
        rho_b=0.2,
        w_rep=1.5,
    )

    governance_config = _REGIME_CONFIG[comp.regime]()

    orch_config = OrchestratorConfig(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        payoff_config=payoff_config,
        governance_config=governance_config,
    )

    orchestrator = Orchestrator(config=orch_config)
    agents = create_agents_from_config(comp)
    for agent in agents:
        orchestrator.register_agent(agent)

    epoch_metrics = orchestrator.run()

    # Aggregate epoch-level metrics
    welfares = [em.total_welfare for em in epoch_metrics]
    toxicities = [em.toxicity_rate for em in epoch_metrics]
    qgaps = [em.quality_gap for em in epoch_metrics]
    payoffs = [em.avg_payoff for em in epoch_metrics]

    # Extract per-class payoffs
    class_payoffs = _extract_per_class_payoffs(orchestrator)

    return RunResult(
        composition=comp.label,
        regime=comp.regime,
        rogue_pct=comp.rogue_pct,
        seed=seed,
        n_epochs=len(epoch_metrics),
        mean_welfare=float(np.mean(welfares)) if welfares else 0.0,
        total_welfare=float(np.sum(welfares)) if welfares else 0.0,
        mean_toxicity=float(np.mean(toxicities)) if toxicities else 0.0,
        mean_quality_gap=float(np.mean(qgaps)) if qgaps else 0.0,
        mean_avg_payoff=float(np.mean(payoffs)) if payoffs else 0.0,
        final_welfare=float(welfares[-1]) if welfares else 0.0,
        final_toxicity=float(toxicities[-1]) if toxicities else 0.0,
        honest_avg_payoff=_mean_or_zero(class_payoffs["honest"]),
        opportunistic_avg_payoff=_mean_or_zero(class_payoffs["opportunistic"]),
        adversarial_avg_payoff=_mean_or_zero(class_payoffs["adversarial"]),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(results: List[RunResult]) -> List[AggResult]:
    """Group by composition and compute mean/std across seeds."""
    groups: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        groups[r.composition].append(r)

    aggs: List[AggResult] = []
    for label, runs in sorted(
        groups.items(), key=lambda x: (x[1][0].regime, x[1][0].rogue_pct)
    ):
        welfares = [r.mean_welfare for r in runs]
        welfare_totals = [r.total_welfare for r in runs]
        toxicities = [r.mean_toxicity for r in runs]
        qgaps = [r.mean_quality_gap for r in runs]
        payoffs_all = [r.mean_avg_payoff for r in runs]
        honest_pays = [r.honest_avg_payoff for r in runs]
        opp_pays = [r.opportunistic_avg_payoff for r in runs]
        adv_pays = [r.adversarial_avg_payoff for r in runs]

        aggs.append(
            AggResult(
                label=label,
                regime=runs[0].regime,
                rogue_pct=runs[0].rogue_pct,
                n_seeds=len(runs),
                welfare_mean=float(np.mean(welfares)),
                welfare_std=float(np.std(welfares)),
                welfare_total_mean=float(np.mean(welfare_totals)),
                # ddof=1: sample std across seeds — feeds the SEM error bars
                welfare_total_std=float(np.std(welfare_totals, ddof=1)) if len(runs) > 1 else 0.0,
                toxicity_mean=float(np.mean(toxicities)),
                toxicity_std=float(np.std(toxicities)),
                quality_gap_mean=float(np.mean(qgaps)),
                avg_payoff_mean=float(np.mean(payoffs_all)),
                honest_payoff_mean=float(np.mean(honest_pays)),
                opportunistic_payoff_mean=float(np.mean(opp_pays)),
                adversarial_payoff_mean=float(np.mean(adv_pays)),
            )
        )
    return aggs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
DPI = 160
COLORS = {
    "governed": "#4CAF50",      # green
    "ungoverned": "#F44336",    # red
    "refinery": "#2196F3",      # blue
    "honest": "#2196F3",        # blue
    "opportunistic": "#FF9800", # orange
    "adversarial": "#9C27B0",   # purple
}

MARKERS = {
    "governed": "s",
    "ungoverned": "o",
    "refinery": "D",
}


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _split_by_regime(
    aggs: List[AggResult],
) -> Dict[str, List[AggResult]]:
    """Split aggregated results by regime."""
    result: Dict[str, List[AggResult]] = {}
    for regime in REGIMES:
        result[regime] = sorted(
            [a for a in aggs if a.regime == regime], key=lambda a: a.rogue_pct
        )
    return result


# Plot 1: Welfare — all regimes
def plot_welfare_comparison(aggs: List[AggResult], out_dir: Path) -> Path:
    """Multi-line: welfare vs rogue proportion for all regimes."""
    by_regime = _split_by_regime(aggs)
    fig, ax = plt.subplots(figsize=(10, 6))

    for regime in REGIMES:
        data = by_regime[regime]
        if not data:
            continue
        pcts = [a.rogue_pct * 100 for a in data]
        welfares = [a.welfare_total_mean for a in data]
        # SEM of the plotted quantity (total welfare across seeds). The old
        # expression scaled the std of a *different* series (per-epoch-mean
        # welfare) by n_seeds — inflating bars by ~n^1.5 on the wrong units.
        stds = [a.welfare_total_std / np.sqrt(a.n_seeds) for a in data]
        ax.errorbar(
            pcts, welfares, yerr=stds,
            color=COLORS[regime], linewidth=2.5, marker=MARKERS[regime], markersize=8,
            capsize=5, capthick=1.5, label=regime.title(), zorder=5,
        )

    _style_ax(
        ax,
        "GasTown: Total Welfare by Rogue Proportion",
        "Adversarial Agent Proportion (%)",
        "Total Welfare (sum over epochs)",
    )
    ax.legend(fontsize=10, loc="best")

    out = out_dir / "gastown_welfare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# Plot 2: Toxicity — all regimes
def plot_toxicity_comparison(aggs: List[AggResult], out_dir: Path) -> Path:
    """Multi-line: toxicity vs rogue proportion."""
    by_regime = _split_by_regime(aggs)
    fig, ax = plt.subplots(figsize=(10, 6))

    for regime in REGIMES:
        data = by_regime[regime]
        if not data:
            continue
        pcts = [a.rogue_pct * 100 for a in data]
        tox = [a.toxicity_mean for a in data]
        stds = [a.toxicity_std / np.sqrt(a.n_seeds) for a in data]  # SEM, consistent with welfare plot
        ax.errorbar(
            pcts, tox, yerr=stds,
            color=COLORS[regime], linewidth=2.5, marker=MARKERS[regime], markersize=8,
            capsize=5, capthick=1.5, label=regime.title(), zorder=5,
        )

    _style_ax(
        ax,
        "GasTown: Toxicity Rate by Rogue Proportion",
        "Adversarial Agent Proportion (%)",
        "Toxicity Rate (mean over epochs)",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="best")

    out = out_dir / "gastown_toxicity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# Plot 3: Governance protection — welfare/toxicity gap vs ungoverned baseline
def plot_governance_protection(aggs: List[AggResult], out_dir: Path) -> Path:
    """Grouped bars: welfare gap and toxicity reduction vs ungoverned baseline."""
    by_regime = _split_by_regime(aggs)
    ungov_by_pct = {round(a.rogue_pct, 2): a for a in by_regime["ungoverned"]}
    fig, ax = plt.subplots(figsize=(12, 6))

    compare_regimes = [r for r in REGIMES if r != "ungoverned"]
    pcts_available = sorted(ungov_by_pct.keys())
    labels = [f"{pct * 100:.0f}%" for pct in pcts_available]
    x = np.arange(len(labels))
    n_bars = len(compare_regimes) * 2  # welfare + toxicity per regime
    total_width = 0.7
    bar_width = total_width / n_bars

    bar_idx = 0
    for regime in compare_regimes:
        regime_by_pct = {round(a.rogue_pct, 2): a for a in by_regime[regime]}
        welfare_gaps = []
        toxicity_gaps = []
        for pct in pcts_available:
            if pct in regime_by_pct:
                g = regime_by_pct[pct]
                u = ungov_by_pct[pct]
                welfare_gaps.append(g.welfare_total_mean - u.welfare_total_mean)
                toxicity_gaps.append(u.toxicity_mean - g.toxicity_mean)
            else:
                welfare_gaps.append(0.0)
                toxicity_gaps.append(0.0)

        offset_w = (bar_idx - n_bars / 2 + 0.5) * bar_width
        offset_t = (bar_idx + 1 - n_bars / 2 + 0.5) * bar_width
        ax.bar(x + offset_w, welfare_gaps, bar_width,
               label=f"{regime.title()} welfare gain", color=COLORS[regime], alpha=0.85)
        ax.bar(x + offset_t, toxicity_gaps, bar_width,
               label=f"{regime.title()} tox. reduction", color=COLORS[regime], alpha=0.45,
               hatch="//")
        bar_idx += 2

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    _style_ax(
        ax,
        "Governance Protection: Benefit Over Ungoverned Baseline",
        "Adversarial Agent Proportion",
        "Benefit vs. Ungoverned",
    )
    ax.legend(fontsize=9, loc="best")

    out = out_dir / "gastown_governance_protection.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# Plot 4: Per-class payoffs at key rogue levels
def plot_payoff_breakdown(aggs: List[AggResult], out_dir: Path) -> Path:
    """Grouped bars: per-class payoffs for all regimes."""
    by_regime = _split_by_regime(aggs)

    fig, axes = plt.subplots(1, len(REGIMES), figsize=(8 * len(REGIMES), 7), sharey=True)

    for ax, regime in zip(axes, REGIMES, strict=True):
        data = by_regime[regime]
        if not data:
            continue

        labels = [f"{a.rogue_pct * 100:.0f}% rogue" for a in data]
        x_pos = np.arange(len(labels))
        width = 0.25

        bars_data = [
            ("Honest", [a.honest_payoff_mean for a in data], COLORS["honest"]),
            ("Opportunistic", [a.opportunistic_payoff_mean for a in data], COLORS["opportunistic"]),
            ("Adversarial", [a.adversarial_payoff_mean for a in data], COLORS["adversarial"]),
        ]

        for idx, (name, vals, color) in enumerate(bars_data):
            offset = (idx - 1) * width
            bars = ax.bar(x_pos + offset, vals, width, label=name, color=color, alpha=0.85)
            for bar, val in zip(bars, vals, strict=True):
                if val != 0.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold",
                    )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_title(f"{regime.title()}: Per-Class Payoffs", fontsize=12, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Average Total Payoff", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "GasTown: Per-Class Payoffs by Rogue Proportion",
        fontsize=14, fontweight="bold", y=1.02,
    )

    out = out_dir / "gastown_payoff_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# Plot 5: Adverse selection (quality gap)
def plot_adverse_selection(aggs: List[AggResult], out_dir: Path) -> Path:
    """Multi-line: quality gap vs rogue proportion."""
    by_regime = _split_by_regime(aggs)
    fig, ax = plt.subplots(figsize=(10, 6))

    for regime in REGIMES:
        data = by_regime[regime]
        if not data:
            continue
        pcts = [a.rogue_pct * 100 for a in data]
        qgaps = [a.quality_gap_mean for a in data]
        ax.plot(
            pcts, qgaps,
            color=COLORS[regime], linewidth=2.5, marker=MARKERS[regime], markersize=8,
            label=regime.title(), zorder=5,
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_between(
        [0, 100], [0, 0], [-1, -1],
        alpha=0.05, color="red", label="Adverse selection zone",
    )

    _style_ax(
        ax,
        "GasTown: Adverse Selection by Rogue Proportion",
        "Adversarial Agent Proportion (%)",
        "Quality Gap (negative = adverse selection)",
    )
    ax.legend(fontsize=10, loc="best")

    out = out_dir / "gastown_adverse_selection.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def write_results_csv(results: List[RunResult], path: Path) -> None:
    """Write raw per-run results to CSV."""
    fieldnames = [
        "composition", "regime", "rogue_pct", "seed", "n_epochs",
        "mean_welfare", "total_welfare", "mean_toxicity",
        "mean_quality_gap", "mean_avg_payoff",
        "final_welfare", "final_toxicity",
        "honest_avg_payoff", "opportunistic_avg_payoff", "adversarial_avg_payoff",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "composition": r.composition,
                "regime": r.regime,
                "rogue_pct": f"{r.rogue_pct:.2f}",
                "seed": r.seed,
                "n_epochs": r.n_epochs,
                "mean_welfare": f"{r.mean_welfare:.6f}",
                "total_welfare": f"{r.total_welfare:.6f}",
                "mean_toxicity": f"{r.mean_toxicity:.6f}",
                "mean_quality_gap": f"{r.mean_quality_gap:.6f}",
                "mean_avg_payoff": f"{r.mean_avg_payoff:.6f}",
                "final_welfare": f"{r.final_welfare:.6f}",
                "final_toxicity": f"{r.final_toxicity:.6f}",
                "honest_avg_payoff": f"{r.honest_avg_payoff:.6f}",
                "opportunistic_avg_payoff": f"{r.opportunistic_avg_payoff:.6f}",
                "adversarial_avg_payoff": f"{r.adversarial_avg_payoff:.6f}",
            })


def write_aggregated_csv(aggs: List[AggResult], path: Path) -> None:
    """Write aggregated results to CSV."""
    fieldnames = [
        "composition", "regime", "rogue_pct", "n_seeds",
        "welfare_total_mean", "welfare_total_std", "welfare_std",
        "toxicity_mean", "toxicity_std",
        "quality_gap_mean", "avg_payoff_mean",
        "honest_payoff_mean", "opportunistic_payoff_mean", "adversarial_payoff_mean",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in aggs:
            writer.writerow({
                "composition": a.label,
                "regime": a.regime,
                "rogue_pct": f"{a.rogue_pct:.2f}",
                "n_seeds": a.n_seeds,
                "welfare_total_mean": f"{a.welfare_total_mean:.4f}",
                "welfare_total_std": f"{a.welfare_total_std:.4f}",
                "welfare_std": f"{a.welfare_std:.4f}",
                "toxicity_mean": f"{a.toxicity_mean:.4f}",
                "toxicity_std": f"{a.toxicity_std:.4f}",
                "quality_gap_mean": f"{a.quality_gap_mean:.4f}",
                "avg_payoff_mean": f"{a.avg_payoff_mean:.4f}",
                "honest_payoff_mean": f"{a.honest_payoff_mean:.4f}",
                "opportunistic_payoff_mean": f"{a.opportunistic_payoff_mean:.4f}",
                "adversarial_payoff_mean": f"{a.adversarial_payoff_mean:.4f}",
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="GasTown composition study: governance effectiveness under adversarial pressure"
    )
    parser.add_argument("--total-agents", type=int, default=7, help="Total agents per run")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per run")
    parser.add_argument("--steps", type=int, default=10, help="Steps per epoch")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or Path(f"runs/{timestamp}_gastown_composition_study")
    plots_dir = out_dir / "plots"
    csv_dir = out_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    compositions = build_compositions(args.total_agents)
    seeds = list(range(42, 42 + args.seeds))
    n_per_regime = len(compositions) // len(REGIMES)

    print("=" * 70)
    print("GasTown Composition Study")
    print(f"  Agents: {args.total_agents}, Epochs: {args.epochs}, Steps/epoch: {args.steps}")
    print(f"  Seeds: {seeds}")
    print(f"  Regimes: {len(REGIMES)} ({', '.join(REGIMES)})")
    print(f"  Compositions: {len(compositions)} ({n_per_regime} per regime)")
    print(f"  Total runs: {len(compositions) * len(seeds)}")
    print("=" * 70)

    # Run all combinations
    all_results: List[RunResult] = []
    total_runs = len(compositions) * len(seeds)
    run_idx = 0

    for comp in compositions:
        for seed in seeds:
            run_idx += 1
            print(
                f"  [{run_idx}/{total_runs}] {comp.label} "
                f"(H={comp.n_honest}, O={comp.n_opportunistic}, "
                f"A={comp.n_adversarial}) seed={seed}...",
                end=" ", flush=True,
            )
            result = run_single(comp, seed, args.epochs, args.steps)
            all_results.append(result)
            print(
                f"welfare={result.total_welfare:.1f}, "
                f"toxicity={result.mean_toxicity:.3f}"
            )

    # Aggregate
    aggs = aggregate_results(all_results)

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(
        f"{'Composition':<32} {'Regime':<12} {'Rogue%':>7} {'TotalWelfare':>13} "
        f"{'WelfStd':>8} {'Toxicity':>9} {'ToxStd':>7} "
        f"{'Hon_Pay':>8} {'Opp_Pay':>8} {'Adv_Pay':>8}"
    )
    print("-" * 130)
    for a in aggs:
        print(
            f"{a.label:<32} {a.regime:<12} {a.rogue_pct*100:>6.0f}% "
            f"{a.welfare_total_mean:>13.2f} {a.welfare_std:>8.2f} "
            f"{a.toxicity_mean:>9.3f} {a.toxicity_std:>7.3f} "
            f"{a.honest_payoff_mean:>8.2f} {a.opportunistic_payoff_mean:>8.2f} "
            f"{a.adversarial_payoff_mean:>8.2f}"
        )

    # Key comparisons
    by_regime: Dict[str, List[AggResult]] = {}
    for regime in REGIMES:
        by_regime[regime] = [a for a in aggs if a.regime == regime]

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Compare peak welfare per regime
    for regime in REGIMES:
        data = by_regime[regime]
        if data:
            peak = max(data, key=lambda a: a.welfare_total_mean)
            print(f"  {regime.title()} peak welfare: {peak.label} "
                  f"(welfare={peak.welfare_total_mean:.2f}, toxicity={peak.toxicity_mean:.3f})")

    # Find breakdown threshold: where toxicity exceeds 0.5
    for regime in REGIMES:
        data = by_regime[regime]
        breakdown = next(
            (a for a in sorted(data, key=lambda a: a.rogue_pct)
             if a.toxicity_mean > 0.5),
            None,
        )
        if breakdown:
            print(f"  {regime.title()} breakdown threshold: {breakdown.rogue_pct*100:.0f}% rogue "
                  f"(toxicity={breakdown.toxicity_mean:.3f})")
        else:
            print(f"  {regime.title()}: toxicity never exceeds 0.5")

    # Compare all regimes vs ungoverned at matching rogue levels
    ungov_by_pct = {round(a.rogue_pct, 2): a for a in by_regime["ungoverned"]}
    for regime in REGIMES:
        if regime == "ungoverned":
            continue
        regime_by_pct = {round(a.rogue_pct, 2): a for a in by_regime[regime]}
        for pct in sorted(set(regime_by_pct.keys()) & set(ungov_by_pct.keys())):
            g = regime_by_pct[pct]
            u = ungov_by_pct[pct]
            w_diff = g.welfare_total_mean - u.welfare_total_mean
            t_diff = g.toxicity_mean - u.toxicity_mean
            print(f"  At {pct*100:.0f}% rogue: {regime} welfare diff={w_diff:+.2f}, "
                  f"toxicity diff={t_diff:+.3f}")

    # Generate plots
    print("\nGenerating plots...")
    plots_written: List[Path] = []
    plots_written.append(plot_welfare_comparison(aggs, plots_dir))
    plots_written.append(plot_toxicity_comparison(aggs, plots_dir))
    plots_written.append(plot_governance_protection(aggs, plots_dir))
    plots_written.append(plot_payoff_breakdown(aggs, plots_dir))
    plots_written.append(plot_adverse_selection(aggs, plots_dir))

    for p in plots_written:
        print(f"  -> {p}")

    # Export CSVs
    csv_path = csv_dir / "results.csv"
    write_results_csv(all_results, csv_path)
    print(f"  -> {csv_path}")

    agg_csv_path = csv_dir / "aggregated_results.csv"
    write_aggregated_csv(aggs, agg_csv_path)
    print(f"  -> {agg_csv_path}")

    print(f"\nAll outputs in: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
