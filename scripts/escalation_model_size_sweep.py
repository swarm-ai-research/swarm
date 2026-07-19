#!/usr/bin/env python3
"""Model size vs escalation behavior sweep.

Tests whether model scale correlates with deception, escalation rate,
or de-escalation ability. Each model plays against itself (mirror match)
to isolate size effects from asymmetric pairings.

Sweeps 6 models across 2 personas × 10 seeds = 120 LLM runs.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
matplotlib.use("Agg")

from sweep_utils import load_checkpoint, log, save_checkpoint  # noqa: E402

from swarm.domains.escalation_sandbox.agents import (  # noqa: E402
    EscalationAgentBridge,
    EscalationPolicy,
)
from swarm.domains.escalation_sandbox.config import EscalationConfig  # noqa: E402
from swarm.domains.escalation_sandbox.env import EscalationEnvironment  # noqa: E402
from swarm.domains.escalation_sandbox.metrics import (  # noqa: E402
    EscalationMetrics,
    compute_escalation_metrics,
    compute_sweep_statistics,
)

# ── Configuration ────────────────────────────────────────────────────
# Models ordered by approximate parameter count.
# All accessed via OpenRouter for uniform API.
MODELS = {
    "Llama 3.1 8B": {
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "params_b": 8,
        "family": "Llama",
    },
    "Mistral Small 3.1 24B": {
        "model_id": "mistralai/mistral-small-3.1-24b-instruct",
        "params_b": 24,
        "family": "Mistral",
    },
    "Llama 3.3 70B": {
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "params_b": 70,
        "family": "Llama",
    },
    "GPT-4.1 mini": {
        "model_id": "openai/gpt-4.1-mini",
        "params_b": 100,  # estimated
        "family": "OpenAI",
    },
    "Claude Sonnet 4": {
        "model_id": "anthropic/claude-sonnet-4",
        "params_b": 200,  # estimated
        "family": "Anthropic",
    },
    "Llama 3.1 405B": {
        "model_id": "meta-llama/llama-3.1-405b-instruct",
        "params_b": 405,
        "family": "Llama",
    },
}

# Two persona settings to test: default (neutral) and adversarial
PERSONAS = {
    "Default (mirror)": "default",
    "Adversarial (mirror)": "adversarial",
}

# Use the baseline scenario config as a template
BASE_YAML = "scenarios/escalation_llm_baseline.yaml"

SEEDS = list(range(10))
OUTDIR = Path("runs/escalation_model_size")
OUTDIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = OUTDIR / "progress.log"
CHECKPOINT_FILE = OUTDIR / "checkpoint.json"


def run_mirror_match(
    model_id: str, persona: str, seed: int,
) -> EscalationMetrics:
    """Run a mirror match: same model vs itself with given persona."""
    import random

    with open(BASE_YAML) as f:
        raw = yaml.safe_load(f)
    data = raw.get("domain", raw)
    config = EscalationConfig.from_dict(copy.deepcopy(data))

    rng = random.Random(seed)
    env = EscalationEnvironment(config, seed=seed)

    policies: dict[str, EscalationPolicy] = {}
    for agent_cfg in config.agents:
        aid = agent_cfg.agent_id
        env.add_nation(
            agent_id=aid,
            name=agent_cfg.name,
            military_strength=agent_cfg.military_strength,
            economic_strength=agent_cfg.economic_strength,
            has_nuclear=agent_cfg.has_nuclear,
            has_second_strike=agent_cfg.has_second_strike,
            intelligence_quality=agent_cfg.intelligence_quality,
        )
        # Override model and persona for mirror match
        policy = EscalationAgentBridge(
            agent_id=aid,
            name=agent_cfg.name,
            provider="openrouter",
            model_id=model_id,
            persona=persona,
            temperature=0.7,
            seed=rng.randint(0, 2**31),
        )
        policies[aid] = policy

    # Apply initial escalation
    initial = config.crisis.initial_escalation
    if initial > 0:
        from swarm.domains.escalation_sandbox.entities import EscalationLevel
        clamped = max(0, min(9, initial))
        for nation in env._nations.values():
            nation.current_level = EscalationLevel(clamped)

    # Run episode
    while not env.is_terminal():
        actions = {}
        for aid, policy in policies.items():
            obs = env.obs(aid)
            actions[aid] = policy.decide(obs)
        env.apply_actions(actions)

    return compute_escalation_metrics(
        turn_results=env.turn_results,
        nations=env.nations,
        events=env.events,
        nuclear_threshold_turn=env.nuclear_threshold_turn,
        outcome=env.outcome.value,
    )


# ── Run sweep ────────────────────────────────────────────────────────
log("=" * 80, PROGRESS_FILE)
log("Model Size vs Escalation Behavior Sweep", PROGRESS_FILE)
log(f"{len(MODELS)} models × {len(PERSONAS)} personas × "
    f"{len(SEEDS)} seeds = "
    f"{len(MODELS) * len(PERSONAS) * len(SEEDS)} runs", PROGRESS_FILE)
log("=" * 80, PROGRESS_FILE)

completed = load_checkpoint(CHECKPOINT_FILE)
results: dict[str, dict[str, list[EscalationMetrics]]] = {}

total = len(MODELS) * len(PERSONAS) * len(SEEDS)
count = 0
start = time.time()

for persona_name, persona in PERSONAS.items():
    results[persona_name] = {}
    log(f"\n── {persona_name} ──", PROGRESS_FILE)

    for model_name, model_info in MODELS.items():
        metrics_list: list[EscalationMetrics] = []
        log(f"  Model: {model_name} ({model_info['params_b']}B)", PROGRESS_FILE)

        for seed in SEEDS:
            run_key = f"{persona_name}|{model_name}|{seed}"
            count += 1

            if run_key in completed:
                m = EscalationMetrics(**completed[run_key])
                metrics_list.append(m)
                log(f"    Seed {seed}: [cached] div={m.signal_action_divergence:.3f}", PROGRESS_FILE)
                continue

            log(f"    Seed {seed} ({count}/{total}) ... ", PROGRESS_FILE)
            try:
                t0 = time.time()
                m = run_mirror_match(model_info["model_id"], persona, seed)
                elapsed = time.time() - t0

                metrics_list.append(m)
                completed[run_key] = m.to_dict()
                save_checkpoint(completed, CHECKPOINT_FILE)

                nuc = m.outcome in ("nuclear_exchange", "mutual_destruction")
                log(f"      -> nuc={nuc}, div={m.signal_action_divergence:.3f}, "
                    f"welfare={m.welfare_composite:.1f} ({elapsed:.1f}s)", PROGRESS_FILE)
            except Exception as e:
                log(f"      -> ERROR: {e}", PROGRESS_FILE)
                continue

        results[persona_name][model_name] = metrics_list
        if metrics_list:
            stats = compute_sweep_statistics(metrics_list)
            log(f"  {model_name} summary: nuclear={stats['nuclear_threshold_rate']:.0%}, "
                f"divergence={stats['mean_signal_action_divergence']:.3f}, "
                f"welfare={stats['mean_welfare_composite']:.1f}", PROGRESS_FILE)

elapsed_total = time.time() - start
log(f"\nTotal: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)", PROGRESS_FILE)

# ── Plot 1: Size vs metrics ──────────────────────────────────────────
model_names = list(MODELS.keys())
persona_names = list(PERSONAS.keys())
param_counts = [MODELS[m]["params_b"] for m in model_names]
n_models = len(model_names)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Model Size vs Escalation Behavior\n"
    "(Mirror matches: same model vs itself, 10 seeds each)",
    fontsize=14, y=0.98,
)

colors = {"Default (mirror)": "#2196F3", "Adversarial (mirror)": "#F44336"}

for row, pn in enumerate(persona_names):
    # Col 0: Divergence vs size
    ax = axes[row, 0]
    divs_mean, divs_std = [], []
    for mn in model_names:
        mlist = results[pn].get(mn, [])
        if mlist:
            vals = [m.signal_action_divergence for m in mlist]
            divs_mean.append(np.mean(vals))
            divs_std.append(np.std(vals))
        else:
            divs_mean.append(0)
            divs_std.append(0)
    ax.errorbar(param_counts, divs_mean, yerr=divs_std, fmt="o-",
                color=colors[pn], capsize=4, markersize=8)
    for i, mn in enumerate(model_names):
        ax.annotate(mn.split()[0], (param_counts[i], divs_mean[i]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center")
    ax.set_xlabel("Model Size (B params)")
    ax.set_ylabel("Signal-Action Divergence")
    ax.set_title(f"{pn}: Divergence vs Size")
    ax.set_xscale("log")
    ax.axhline(0, color="gray", ls=":", lw=0.8)

    # Col 1: Nuclear rate vs size
    ax = axes[row, 1]
    nuc_rates = []
    for mn in model_names:
        mlist = results[pn].get(mn, [])
        if mlist:
            stats = compute_sweep_statistics(mlist)
            nuc_rates.append(stats["nuclear_threshold_rate"] * 100)
        else:
            nuc_rates.append(0)
    ax.bar(range(n_models), nuc_rates, color=colors[pn], alpha=0.7, width=0.6)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([m.split()[0] + f"\n{MODELS[m]['params_b']}B" for m in model_names],
                       fontsize=8)
    ax.set_ylabel("Nuclear Rate (%)")
    ax.set_title(f"{pn}: Nuclear Rate")
    ax.set_ylim(0, 105)
    for i, rate in enumerate(nuc_rates):
        ax.text(i, rate + 1, f"{rate:.0f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Col 2: Welfare vs size
    ax = axes[row, 2]
    welf_mean, welf_std = [], []
    for mn in model_names:
        mlist = results[pn].get(mn, [])
        if mlist:
            vals = [m.welfare_composite for m in mlist]
            welf_mean.append(np.mean(vals))
            welf_std.append(np.std(vals))
        else:
            welf_mean.append(0)
            welf_std.append(0)
    ax.errorbar(param_counts, welf_mean, yerr=welf_std, fmt="s-",
                color=colors[pn], capsize=4, markersize=8)
    for i, mn in enumerate(model_names):
        ax.annotate(mn.split()[0], (param_counts[i], welf_mean[i]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center")
    ax.set_xlabel("Model Size (B params)")
    ax.set_ylabel("Welfare Composite")
    ax.set_title(f"{pn}: Welfare vs Size")
    ax.set_xscale("log")
    ax.axhline(0, color="gray", ls="--", lw=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plot_path = OUTDIR / "model_size_sweep.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log(f"\nPlot saved to {plot_path}", PROGRESS_FILE)

# ── Plot 2: Heatmap ──────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))

for ax_idx, pn in enumerate(persona_names):
    ax = axes2[ax_idx]
    matrix = np.zeros((3, n_models))  # 3 rows: div, nuc%, welfare
    for j, mn in enumerate(model_names):
        mlist = results[pn].get(mn, [])
        if mlist:
            stats = compute_sweep_statistics(mlist)
            matrix[0, j] = stats["mean_signal_action_divergence"]
            matrix[1, j] = stats["nuclear_threshold_rate"] * 100
            matrix[2, j] = stats["mean_welfare_composite"]

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([f"{m.split()[0]}\n{MODELS[m]['params_b']}B" for m in model_names],
                       fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Divergence", "Nuclear %", "Welfare"], fontsize=9)
    ax.set_title(pn)
    for i in range(3):
        for j in range(n_models):
            val = matrix[i, j]
            fmt = f"{val:.2f}" if i == 0 else (f"{val:.0f}%" if i == 1 else f"{val:.0f}")
            ax.text(j, i, fmt, ha="center", va="center",
                    fontsize=9, fontweight="bold")
    fig2.colorbar(im, ax=ax)

plt.tight_layout()
heatmap_path = OUTDIR / "model_size_heatmap.png"
fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
log(f"Heatmap saved to {heatmap_path}", PROGRESS_FILE)

# ── Summary ──────────────────────────────────────────────────────────
log("\n" + "=" * 110, PROGRESS_FILE)
log(f"{'Persona':<25} {'Model':<25} {'Params':>7} {'Nuc%':>6} {'Diverge':>9} "
    f"{'Welfare':>9} {'Velocity':>9} {'DeEsc':>7} {'N':>4}", PROGRESS_FILE)
log("-" * 110, PROGRESS_FILE)
for pn in persona_names:
    for mn in model_names:
        mlist = results[pn].get(mn, [])
        if mlist:
            stats = compute_sweep_statistics(mlist)
            log(f"{pn:<25} {mn:<25} {MODELS[mn]['params_b']:>6}B "
                f"{stats['nuclear_threshold_rate']*100:>5.0f}% "
                f"{stats['mean_signal_action_divergence']:>9.3f} "
                f"{stats['mean_welfare_composite']:>9.1f} "
                f"{stats['mean_escalation_velocity']:>9.3f} "
                f"{stats['mean_de_escalation_rate']:>7.3f} "
                f"{len(mlist):>4d}", PROGRESS_FILE)
    log("", PROGRESS_FILE)
log("=" * 110, PROGRESS_FILE)
