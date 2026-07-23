#!/usr/bin/env python3
"""Governance parameter sweep: vary each lever one-at-a-time, 10 seeds each.

Tests whether any governance configuration can prevent nuclear exchange
in hawk-vs-hawk escalation dynamics.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
matplotlib.use("Agg")

from swarm.domains.escalation_sandbox.config import EscalationConfig  # noqa: E402
from swarm.domains.escalation_sandbox.metrics import (  # noqa: E402
    EscalationMetrics,
    compute_sweep_statistics,
)
from swarm.domains.escalation_sandbox.runner import EscalationRunner  # noqa: E402

# ── Load baseline governance config ──────────────────────────────────
BASELINE_PATH = "scenarios/escalation_governance.yaml"
SEEDS = list(range(10))
OUTDIR = Path("runs/escalation_governance_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(BASELINE_PATH) as f:
    RAW_BASELINE = yaml.safe_load(f)
DOMAIN_BASELINE = RAW_BASELINE.get("domain", RAW_BASELINE)

# ── Define sweep axes ────────────────────────────────────────────────
# Each axis: (lever_name, governance_key, values, label)
SWEEP_AXES = [
    (
        "Circuit Breaker Turns",
        "circuit_breaker_negotiation_turns",
        [1, 2, 3, 5],
        "Negotiation turns",
    ),
    (
        "Treaty Penalty",
        "treaty_defection_penalty",
        [5.0, 10.0, 20.0, 50.0],
        "Penalty",
    ),
    (
        "Treaty Max Level",
        "treaty_max_level",
        [2, 3, 4, 5],
        "Max level",
    ),
    (
        "De-escalation Friction",
        "de_escalation_friction_multiplier",
        [0.25, 0.5, 1.0, 2.0],
        "Multiplier",
    ),
    (
        "MAD Retaliation Prob",
        "mad_retaliation_probability",
        [0.5, 0.7, 0.9, 1.0],
        "Probability",
    ),
]


def make_config(gov_overrides: dict) -> EscalationConfig:
    """Create an EscalationConfig with governance overrides applied."""
    data = copy.deepcopy(DOMAIN_BASELINE)
    for k, v in gov_overrides.items():
        data["governance"][k] = v
    return EscalationConfig.from_dict(data)


def run_config(config: EscalationConfig, seeds: list[int]) -> list[EscalationMetrics]:
    """Run a config across all seeds and return metrics."""
    metrics = []
    for seed in seeds:
        runner = EscalationRunner(config, seed=seed)
        m = runner.run()
        metrics.append(m)
    return metrics


# ── Run sweep ────────────────────────────────────────────────────────
print("=" * 80)
print("Governance Parameter Sweep — One-at-a-time, 10 seeds each")
print("=" * 80)

# First run baseline
print("\nRunning baseline (all defaults) ...")
baseline_config = EscalationConfig.from_dict(copy.deepcopy(DOMAIN_BASELINE))
baseline_metrics = run_config(baseline_config, SEEDS)
baseline_stats = compute_sweep_statistics(baseline_metrics)
print(f"  Baseline: nuclear_rate={baseline_stats['nuclear_threshold_rate']:.0%}, "
      f"welfare={baseline_stats['mean_welfare_composite']:.1f}")

# Store all results: axis_name -> {value_label: [metrics]}
all_results: dict[str, dict[str, list[EscalationMetrics]]] = {}

for axis_name, gov_key, values, _label in SWEEP_AXES:
    print(f"\n── Sweeping: {axis_name} ──")
    axis_results: dict[str, list[EscalationMetrics]] = {}

    for val in values:
        val_label = str(val)
        print(f"  {gov_key}={val} ...", end=" ", flush=True)
        config = make_config({gov_key: val})
        metrics = run_config(config, SEEDS)
        stats = compute_sweep_statistics(metrics)
        nuc_rate = stats["nuclear_threshold_rate"]
        welfare = stats["mean_welfare_composite"]
        print(f"nuclear={nuc_rate:.0%}, welfare={welfare:.1f}")
        axis_results[val_label] = metrics

    all_results[axis_name] = axis_results

# ── Plot results ─────────────────────────────────────────────────────
fig, axes_grid = plt.subplots(2, 5, figsize=(28, 10))
fig.suptitle(
    "Governance Parameter Sweep — Nuclear Rate & Welfare by Lever\n"
    "(Hawk vs Hawk, 10 seeds each, one lever varied at a time)",
    fontsize=14, y=0.98,
)

colors_sweep = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for col, (axis_name, _gov_key, values, label) in enumerate(SWEEP_AXES):
    axis_results = all_results[axis_name]
    val_labels = [str(v) for v in values]

    # Top row: nuclear rate
    ax_nuc = axes_grid[0, col]
    nuc_rates = []
    for vl in val_labels:
        stats = compute_sweep_statistics(axis_results[vl])
        nuc_rates.append(stats["nuclear_threshold_rate"] * 100)

    bars = ax_nuc.bar(
        range(len(val_labels)), nuc_rates,
        color=colors_sweep, alpha=0.8, width=0.6,
    )
    ax_nuc.axhline(
        baseline_stats["nuclear_threshold_rate"] * 100,
        color="red", ls="--", lw=1.2, label="Baseline",
    )
    ax_nuc.set_xticks(range(len(val_labels)))
    ax_nuc.set_xticklabels(val_labels)
    ax_nuc.set_ylabel("Nuclear Rate (%)")
    ax_nuc.set_title(f"{axis_name}")
    ax_nuc.set_ylim(0, 105)
    ax_nuc.legend(fontsize=7)
    # Annotate bars with values
    for bar, rate in zip(bars, nuc_rates, strict=False):
        ax_nuc.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.0f}%", ha="center", va="bottom", fontsize=8,
        )

    # Bottom row: welfare box plots
    ax_welf = axes_grid[1, col]
    welfare_data = [
        [m.welfare_composite for m in axis_results[vl]]
        for vl in val_labels
    ]
    bp = ax_welf.boxplot(
        welfare_data, patch_artist=True, labels=val_labels, widths=0.6,
    )
    for patch, c in zip(bp["boxes"], colors_sweep, strict=False):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_welf.axhline(
        baseline_stats["mean_welfare_composite"],
        color="red", ls="--", lw=1.2, label="Baseline",
    )
    ax_welf.axhline(0, color="gray", ls=":", lw=0.8)
    ax_welf.set_ylabel("Welfare Composite")
    ax_welf.set_xlabel(label)
    ax_welf.legend(fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plot_path = OUTDIR / "governance_sweep.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved to {plot_path}")

# ── Heatmap: nuclear rate by lever value ─────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
matrix = []
y_labels = []
x_labels = []

for axis_name, _gov_key, values, _label in SWEEP_AXES:
    axis_results = all_results[axis_name]
    row = []
    for v in values:
        stats = compute_sweep_statistics(axis_results[str(v)])
        row.append(stats["nuclear_threshold_rate"] * 100)
    matrix.append(row)
    y_labels.append(axis_name)
    if not x_labels:
        x_labels = [str(v) for v in values]

matrix_arr = np.array(matrix)
im = ax2.imshow(matrix_arr, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
ax2.set_xticks(range(len(x_labels)))
ax2.set_xticklabels(x_labels)
ax2.set_yticks(range(len(y_labels)))
ax2.set_yticklabels(y_labels)

# Annotate cells
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        val = matrix_arr[i, j]
        ax2.text(j, i, f"{val:.0f}%", ha="center", va="center",
                 color="white" if val > 60 else "black", fontsize=10, fontweight="bold")

ax2.set_title("Nuclear Rate (%) by Governance Lever Value\n(each row swept independently)")
fig2.colorbar(im, label="Nuclear Rate (%)")
plt.tight_layout()
heatmap_path = OUTDIR / "governance_heatmap.png"
fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Heatmap saved to {heatmap_path}")

# ── Summary table ────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'Lever':<28} {'Value':>8} {'Nuclear%':>10} {'Welfare':>10} "
      f"{'Velocity':>10} {'Diverge':>9} {'DeEsc':>8}")
print("-" * 100)

print(f"{'BASELINE':<28} {'all':>8} "
      f"{baseline_stats['nuclear_threshold_rate']*100:>9.0f}% "
      f"{baseline_stats['mean_welfare_composite']:>10.1f} "
      f"{baseline_stats['mean_escalation_velocity']:>10.3f} "
      f"{baseline_stats['mean_signal_action_divergence']:>9.3f} "
      f"{baseline_stats['mean_de_escalation_rate']:>8.3f}")
print()

for axis_name, _gov_key, values, _label in SWEEP_AXES:
    axis_results = all_results[axis_name]
    for v in values:
        stats = compute_sweep_statistics(axis_results[str(v)])
        nuc = stats["nuclear_threshold_rate"] * 100
        marker = " *" if nuc < baseline_stats["nuclear_threshold_rate"] * 100 else ""
        print(f"{axis_name:<28} {str(v):>8} "
              f"{nuc:>9.0f}% "
              f"{stats['mean_welfare_composite']:>10.1f} "
              f"{stats['mean_escalation_velocity']:>10.3f} "
              f"{stats['mean_signal_action_divergence']:>9.3f} "
              f"{stats['mean_de_escalation_rate']:>8.3f}{marker}")
    print()

print("=" * 100)
print("* = lower nuclear rate than baseline")

# ── Find best config ─────────────────────────────────────────────────
print("\n── Best configurations (lowest nuclear rate) ──")
best_configs = []
for axis_name, gov_key, values, _label in SWEEP_AXES:
    axis_results = all_results[axis_name]
    for v in values:
        stats = compute_sweep_statistics(axis_results[str(v)])
        best_configs.append((
            stats["nuclear_threshold_rate"],
            stats["mean_welfare_composite"],
            axis_name,
            gov_key,
            v,
        ))

best_configs.sort(key=lambda x: (x[0], -x[1]))
for nuc, welf, name, _key, val in best_configs[:5]:
    print(f"  {name} = {val}: nuclear={nuc:.0%}, welfare={welf:.1f}")
