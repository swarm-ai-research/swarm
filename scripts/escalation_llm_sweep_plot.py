#!/usr/bin/env python3
"""Run all 5 LLM escalation scenarios × 10 seeds and plot comparison.

Model mix across scenarios:
  - Baseline:       Claude Sonnet 4 vs GPT-4.1-mini
  - Cuban Missile:  Llama 3.3 70B vs Gemini 2.0 Flash
  - Deception:      Mistral Small 3.1 vs Claude Sonnet 4
  - Governance:     GPT-4.1-mini vs Gemini 2.0 Flash
  - Fog Stress:     Llama 3.3 70B vs Mistral Small 3.1
"""

from __future__ import annotations

import logging
import sys
import time
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

# Suppress httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)

PROGRESS_FILE = Path("runs/escalation_llm_sweep/progress.log")


def log(msg: str) -> None:
    """Print and log to file."""
    print(msg, flush=True)
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "a") as f:
        f.write(msg + "\n")

# ── Scenario definitions ─────────────────────────────────────────────
SCENARIOS = {
    "Baseline\n(Claude v Sonnet 4\nvs GPT-4.1-mini)": "scenarios/escalation_llm_baseline.yaml",
    "Cuban Missile\n(Llama 3.3 70B\nvs Gemini Flash)": "scenarios/escalation_llm_cuban_missile.yaml",
    "Deception\n(Mistral Small\nvs Claude Sonnet 4)": "scenarios/escalation_llm_deception.yaml",
    "Governance\n(GPT-4.1-mini\nvs Gemini Flash)": "scenarios/escalation_llm_governance.yaml",
    "Fog Stress\n(Llama 3.3 70B\nvs Mistral Small)": "scenarios/escalation_llm_fog_stress.yaml",
}

# Short names for the summary table
SCENARIO_TO_SHORT_NAME = {
    "Baseline\n(Claude v Sonnet 4\nvs GPT-4.1-mini)": "Baseline",
    "Cuban Missile\n(Llama 3.3 70B\nvs Gemini Flash)": "Cuban Missile",
    "Deception\n(Mistral Small\nvs Claude Sonnet 4)": "Deception",
    "Governance\n(GPT-4.1-mini\nvs Gemini Flash)": "Governance",
    "Fog Stress\n(Llama 3.3 70B\nvs Mistral Small)": "Fog Stress",
}

SEEDS = list(range(10))
OUTDIR = Path("runs/escalation_llm_sweep")

# ── Run all episodes ─────────────────────────────────────────────────
results: dict[str, list[EscalationMetrics]] = {}
total_runs = len(SCENARIOS) * len(SEEDS)
run_count = 0
t0 = time.time()

for name, path in SCENARIOS.items():
    short = SCENARIO_TO_SHORT_NAME[name]
    log(f"\n{'='*60}")
    log(f"Running {short} ({path})")
    log(f"{'='*60}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    domain_data = raw.get("domain", raw)
    config = EscalationConfig.from_dict(domain_data)

    episode_metrics = []
    for seed in SEEDS:
        run_count += 1
        t_start = time.time()
        log(f"  [{run_count}/{total_runs}] seed={seed} ...")
        runner = EscalationRunner(config, seed=seed)
        m = runner.run()
        elapsed = time.time() - t_start
        log(f"    outcome={m.outcome}, max_esc={m.escalation_max}, "
            f"welfare={m.welfare_composite:.1f} ({elapsed:.1f}s)")
        episode_metrics.append(m)
    results[name] = episode_metrics
    stats = compute_sweep_statistics(episode_metrics)
    log(f"  >> {short}: nuclear_rate={stats['nuclear_threshold_rate']:.0%}, "
        f"welfare={stats['mean_welfare_composite']:.1f}, "
        f"fog_catastrophes={stats['total_fog_catastrophes']}")

total_time = time.time() - t0
log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}m)")

# ── Extract arrays ────────────────────────────────────────────────────
names = list(results.keys())
n = len(names)


def metric_arrays(attr: str) -> list[list[float]]:
    return [[getattr(m, attr) for m in results[s]] for s in names]


welfare = metric_arrays("welfare_composite")
esc_max = metric_arrays("escalation_max")
velocity = metric_arrays("escalation_velocity")
divergence = metric_arrays("signal_action_divergence")
accidental = metric_arrays("accidental_escalation_rate")
fog_cat = metric_arrays("fog_catastrophe_count")
collateral = metric_arrays("collateral_damage")
de_esc = metric_arrays("de_escalation_rate")

# Outcome counts
outcome_labels = ["ceasefire", "timeout", "surrender",
                  "nuclear_exchange", "mutual_destruction"]
outcome_counts = {lbl: [] for lbl in outcome_labels}
for s in names:
    for lbl in outcome_labels:
        outcome_counts[lbl].append(
            sum(1 for m in results[s] if m.outcome == lbl)
        )

# ── Colors ────────────────────────────────────────────────────────────
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# ── Plot ──────────────────────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.suptitle(f"Escalation Sandbox — LLM Agents (OpenRouter) — 5 Scenarios × {len(SEEDS)} Seeds",
             fontsize=16, y=0.98)

# 1. Welfare composite (box plot)
ax = axes[0, 0]
bp = ax.boxplot(welfare, patch_artist=True, tick_labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.set_ylabel("Welfare Composite")
ax.set_title("Welfare")
ax.tick_params(axis="x", rotation=0, labelsize=7)

# 2. Max escalation level (box)
ax = axes[0, 1]
bp = ax.boxplot(esc_max, patch_artist=True, tick_labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(7, color="red", ls="--", lw=0.8, label="Nuclear threshold")
ax.set_ylabel("Max Escalation Level")
ax.set_title("Escalation Severity")
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=0, labelsize=7)

# 3. Escalation velocity (box)
ax = axes[0, 2]
bp = ax.boxplot(velocity, patch_artist=True, tick_labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Levels / Turn")
ax.set_title("Escalation Velocity")
ax.tick_params(axis="x", rotation=0, labelsize=7)

# 4. Outcome distribution (stacked bar)
ax = axes[0, 3]
x_pos = np.arange(n)
bottom = np.zeros(n)
outcome_colors = {
    "ceasefire": "#55A868",
    "timeout": "#CCCC00",
    "surrender": "#DD8452",
    "nuclear_exchange": "#C44E52",
    "mutual_destruction": "#8B0000",
}
for lbl in outcome_labels:
    vals = np.array(outcome_counts[lbl], dtype=float)
    if vals.sum() > 0:
        ax.bar(x_pos, vals, bottom=bottom, label=lbl.replace("_", " ").title(),
               color=outcome_colors[lbl], width=0.6)
        bottom += vals
ax.set_xticks(x_pos)
ax.set_xticklabels([SCENARIO_TO_SHORT_NAME[n] for n in names], rotation=30, ha="right", fontsize=8)
ax.set_ylabel(f"Count (of {len(SEEDS)})")
ax.set_title("Outcome Distribution")
ax.legend(fontsize=7, loc="upper left")

# 5. Signal-action divergence (box)
ax = axes[1, 0]
bp = ax.boxplot(divergence, patch_artist=True, tick_labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Mean |signal - action|")
ax.set_title("Strategic Deception")
ax.tick_params(axis="x", rotation=0, labelsize=7)

# 6. Accidental escalation rate (box)
ax = axes[1, 1]
bp = ax.boxplot(accidental, patch_artist=True, tick_labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Fraction of Actions")
ax.set_title("Accidental Escalation Rate")
ax.tick_params(axis="x", rotation=0, labelsize=7)

# 7. Collateral damage (box)
ax = axes[1, 2]
bp = ax.boxplot(collateral, patch_artist=True, tick_labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Damage Score")
ax.set_title("Collateral Damage")
ax.tick_params(axis="x", rotation=0, labelsize=7)

# 8. De-escalation rate (box)
ax = axes[1, 3]
bp = ax.boxplot(de_esc, patch_artist=True, tick_labels=names, widths=0.6)
for patch, c in zip(bp["boxes"], colors, strict=False):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Rate")
ax.set_title("De-escalation Rate")
ax.tick_params(axis="x", rotation=0, labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = OUTDIR / "escalation_llm_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log(f"\nPlot saved to {out_path}")

# ── Summary table ─────────────────────────────────────────────────────
log("\n" + "=" * 110)
log(f"{'Scenario':<16} {'Welfare':>10} {'MaxEsc':>8} {'Velocity':>10} "
    f"{'Diverge':>9} {'AccidEsc':>10} {'FogCat':>8} {'Collat':>10} "
    f"{'DeEsc':>8} {'Nuclear%':>10}")
log("-" * 110)
for s in names:
    short = SCENARIO_TO_SHORT_NAME[s]
    st = compute_sweep_statistics(results[s])
    nuc_pct = st["nuclear_threshold_rate"] * 100
    log(f"{short:<16} {st['mean_welfare_composite']:>10.1f} "
        f"{st['mean_escalation_max']:>8.1f} "
        f"{st['mean_escalation_velocity']:>10.3f} "
        f"{st['mean_signal_action_divergence']:>9.3f} "
        f"{st['mean_accidental_escalation_rate']:>10.3f} "
        f"{st['total_fog_catastrophes']:>8d} "
        f"{st['mean_collateral_damage']:>10.1f} "
        f"{st['mean_de_escalation_rate']:>8.3f} "
        f"{nuc_pct:>9.0f}%")
log("=" * 110)
