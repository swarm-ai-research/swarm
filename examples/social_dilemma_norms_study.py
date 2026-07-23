#!/usr/bin/env python3
"""
Social Dilemma Norms Study — governance effects on cooperation emergence.

Sweeps 3 social dilemmas x 5 governance configs x N seeds to measure whether
governance mechanisms can substitute for alignment training in promoting
cooperative norms.

Dilemmas:
  1. Tragedy of the Commons — shared resource harvesting
  2. Prisoner's Dilemma — pairwise cooperate/defect
  3. Public Goods Game — contribute to shared pool

Governance configs (subset of 8):
  1. baseline           — no governance (control)
  2. reputation         — reputation decay=0.9
  3. tax_and_reputation — tax=0.05 + decay=0.9
  4. circuit_breaker    — freeze toxic agents
  5. full_governance    — all levers enabled

Usage:
    python examples/social_dilemma_norms_study.py --dry-run
    python examples/social_dilemma_norms_study.py --seeds 3 --epochs 15
    python examples/social_dilemma_norms_study.py --dilemma commons --seeds 1
    python examples/social_dilemma_norms_study.py --live --model llama3.2
    python examples/social_dilemma_norms_study.py --live --provider openrouter
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.bridges.concordia.adapter import ConcordiaAdapter
from swarm.bridges.concordia.config import ConcordiaConfig, JudgeConfig
from swarm.bridges.concordia.dilemma_narratives import (
    DilemmaType,
    generate_dilemma_corpus,
)
from swarm.bridges.concordia.events import JudgeScores
from swarm.bridges.concordia.judge import LLMCallFn, LLMJudge
from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEngine
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.agent import AgentType
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)

# ── LLM client factories ─────────────────────────────────────────────────

PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",
        "default_model": "llama3.2",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.1-8b-instruct",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}


def make_llm_client(
    provider: str = "ollama",
    model: str | None = None,
    timeout: float = 120.0,
) -> LLMCallFn:
    """Build an LLMCallFn for the given provider via OpenAI-compat API.

    Args:
        provider: One of "ollama", "openrouter", "openai".
        model: Model name (uses provider default if None).
        timeout: HTTP timeout in seconds.
    """
    from openai import OpenAI

    defaults = PROVIDER_DEFAULTS[provider]
    client_key = (
        "ollama"
        if provider == "ollama"
        else os.environ.get(defaults["api_key_env"], "")
    )
    if not client_key and provider != "ollama":
        raise RuntimeError(
            f"Missing {defaults['api_key_env']} environment variable for {provider}"
        )

    client = OpenAI(
        api_key=client_key,
        base_url=defaults["base_url"],
        timeout=timeout if provider != "ollama" else 300.0,
    )
    tag = provider.capitalize()

    def _call(prompt: str, model_name: str, temperature: float) -> str:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "%s call attempt %d failed: %s", tag, attempt + 1, exc
                )
                if attempt < 2:
                    time.sleep(2 ** attempt)
        logger.error("%s call failed after 3 attempts: %s", tag, last_err)
        return ""

    return _call


# ── Dilemma definitions ──────────────────────────────────────────────────

DILEMMA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "commons": {
        "type": DilemmaType.COMMONS,
        "agents": [
            ("sustainable_1", "sustainable"),
            ("sustainable_2", "sustainable"),
            ("greedy_1", "greedy"),
            ("conditional_1", "conditional"),
        ],
        "agent_types": {
            "sustainable_1": AgentType.HONEST,
            "sustainable_2": AgentType.HONEST,
            "greedy_1": AgentType.ADVERSARIAL,
            "conditional_1": AgentType.HONEST,
        },
    },
    "prisoners_dilemma": {
        "type": DilemmaType.PRISONERS_DILEMMA,
        "agents": [
            ("cooperator_1", "cooperator"),
            ("cooperator_2", "cooperator"),
            ("defector_1", "defector"),
            ("grudger_1", "grudger"),
        ],
        "agent_types": {
            "cooperator_1": AgentType.HONEST,
            "cooperator_2": AgentType.HONEST,
            "defector_1": AgentType.ADVERSARIAL,
            "grudger_1": AgentType.HONEST,
        },
    },
    "public_goods": {
        "type": DilemmaType.PUBLIC_GOODS,
        "agents": [
            ("generous_1", "generous"),
            ("generous_2", "generous"),
            ("free_rider_1", "free_rider"),
            ("conditional_1", "conditional"),
        ],
        "agent_types": {
            "generous_1": AgentType.HONEST,
            "generous_2": AgentType.HONEST,
            "free_rider_1": AgentType.ADVERSARIAL,
            "conditional_1": AgentType.HONEST,
        },
    },
}

# ── 5 governance configurations ──────────────────────────────────────────


def governance_configs() -> List[Dict[str, Any]]:
    """Return the 5 governance configurations for the norms study."""
    return [
        {
            "label": "baseline",
            "governance": GovernanceConfig(),
        },
        {
            "label": "reputation",
            "governance": GovernanceConfig(
                reputation_decay_rate=0.9,
            ),
        },
        {
            "label": "tax_and_reputation",
            "governance": GovernanceConfig(
                transaction_tax_rate=0.05,
                transaction_tax_split=0.5,
                reputation_decay_rate=0.9,
            ),
        },
        {
            "label": "circuit_breaker",
            "governance": GovernanceConfig(
                circuit_breaker_enabled=True,
                freeze_threshold_toxicity=0.5,
                freeze_threshold_violations=2,
                freeze_duration_epochs=2,
            ),
        },
        {
            "label": "full_governance",
            "governance": GovernanceConfig(
                transaction_tax_rate=0.08,
                transaction_tax_split=0.6,
                reputation_decay_rate=0.85,
                circuit_breaker_enabled=True,
                freeze_threshold_toxicity=0.45,
                freeze_threshold_violations=2,
                freeze_duration_epochs=2,
                audit_enabled=True,
                audit_probability=0.25,
                audit_penalty_multiplier=2.5,
                audit_threshold_p=0.5,
            ),
        },
    ]


# ── Score-injecting judge (from concordia_governance_sweep) ──────────────


class CorpusJudge(LLMJudge):
    """LLMJudge subclass that returns pre-computed scores from a corpus."""

    def __init__(self) -> None:
        super().__init__(config=None, llm_client=None)
        self._queue: List[JudgeScores] = []

    def enqueue(self, scores: JudgeScores) -> None:
        self._queue.append(scores)

    def evaluate(self, narrative: str) -> JudgeScores:
        if self._queue:
            return self._queue.pop(0)
        return JudgeScores()


# ── Per-epoch metrics ────────────────────────────────────────────────────


@dataclass
class DilemmaEpochMetrics:
    """Metrics for a single epoch in a dilemma run."""

    epoch: int = 0
    toxicity: float = 0.0
    welfare: float = 0.0
    quality_gap: float = 0.0
    frozen_count: int = 0
    interactions_count: int = 0
    # Dilemma-specific
    cooperation_rate: float = 0.0
    resource_level: float = 100.0
    avg_contribution: float = 0.0
    norm_strength: float = 0.0
    gini_payoff: float = 0.0


@dataclass
class DilemmaRunResult:
    """Result from a single dilemma run."""

    dilemma: str = ""
    label: str = ""
    seed: int = 0
    epoch_metrics: List[DilemmaEpochMetrics] = field(default_factory=list)

    # Aggregates
    mean_toxicity: float = 0.0
    mean_welfare: float = 0.0
    mean_quality_gap: float = 0.0
    mean_cooperation_rate: float = 0.0
    cooperation_trend: float = 0.0  # slope of cooperation over epochs
    mean_norm_strength: float = 0.0
    mean_gini: float = 0.0
    final_resource_level: float = 100.0
    mean_avg_contribution: float = 0.0
    max_frozen: int = 0
    total_interactions: int = 0

    def compute_aggregates(self) -> None:
        if not self.epoch_metrics:
            return
        n = len(self.epoch_metrics)
        self.mean_toxicity = sum(m.toxicity for m in self.epoch_metrics) / n
        self.mean_welfare = sum(m.welfare for m in self.epoch_metrics) / n
        self.mean_quality_gap = sum(m.quality_gap for m in self.epoch_metrics) / n
        self.mean_cooperation_rate = (
            sum(m.cooperation_rate for m in self.epoch_metrics) / n
        )
        self.mean_norm_strength = (
            sum(m.norm_strength for m in self.epoch_metrics) / n
        )
        self.mean_gini = sum(m.gini_payoff for m in self.epoch_metrics) / n
        self.mean_avg_contribution = (
            sum(m.avg_contribution for m in self.epoch_metrics) / n
        )
        self.final_resource_level = self.epoch_metrics[-1].resource_level
        self.max_frozen = max(m.frozen_count for m in self.epoch_metrics)
        self.total_interactions = sum(
            m.interactions_count for m in self.epoch_metrics
        )

        # Cooperation trend: linear regression slope
        if n >= 2:
            x_mean = (n - 1) / 2.0
            y_vals = [m.cooperation_rate for m in self.epoch_metrics]
            y_mean = sum(y_vals) / n
            num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(y_vals))
            den = sum((i - x_mean) ** 2 for i in range(n))
            self.cooperation_trend = num / den if den > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dilemma": self.dilemma,
            "label": self.label,
            "seed": self.seed,
            "mean_toxicity": self.mean_toxicity,
            "mean_welfare": self.mean_welfare,
            "mean_quality_gap": self.mean_quality_gap,
            "mean_cooperation_rate": self.mean_cooperation_rate,
            "cooperation_trend": self.cooperation_trend,
            "mean_norm_strength": self.mean_norm_strength,
            "mean_gini": self.mean_gini,
            "final_resource_level": self.final_resource_level,
            "mean_avg_contribution": self.mean_avg_contribution,
            "max_frozen": self.max_frozen,
            "total_interactions": self.total_interactions,
        }


# ── Gini coefficient ────────────────────────────────────────────────────


def _gini(values: List[float]) -> float:
    """Compute Gini coefficient for a list of values."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


# ── Single run ───────────────────────────────────────────────────────────


def run_single(
    dilemma_name: str,
    label: str,
    gov_config: GovernanceConfig,
    n_epochs: int,
    steps_per_epoch: int,
    seed: int,
    llm_client: LLMCallFn | None = None,
    llm_model: str = "llama3.2",
) -> DilemmaRunResult:
    """Run a single dilemma + governance config combination."""
    dilemma_cfg = DILEMMA_CONFIGS[dilemma_name]
    agent_specs = dilemma_cfg["agents"]
    agent_types = dilemma_cfg["agent_types"]
    agent_ids = [a[0] for a in agent_specs]

    # Generate corpus
    corpus, dilemma_state = generate_dilemma_corpus(
        dilemma_type=dilemma_cfg["type"],
        agents=agent_specs,
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )

    # Build components — live LLM judge or pre-computed corpus judge
    if llm_client is not None:
        judge = LLMJudge(
            config=JudgeConfig(model=llm_model),
            llm_client=llm_client,
        )
    else:
        judge = CorpusJudge()
    adapter = ConcordiaAdapter(config=ConcordiaConfig(), judge=judge)
    engine = GovernanceEngine(config=gov_config, seed=seed)

    # Build environment state
    state = EnvState(steps_per_epoch=steps_per_epoch)
    for agent_id in agent_ids:
        state.add_agent(agent_id, agent_type=agent_types[agent_id])

    metrics = SoftMetrics()
    result = DilemmaRunResult(dilemma=dilemma_name, label=label, seed=seed)

    # Track cumulative payoffs per agent for Gini
    cumulative_payoffs: Dict[str, float] = dict.fromkeys(agent_ids, 0.0)

    for epoch_idx, epoch_samples in enumerate(corpus):
        state.current_epoch = epoch_idx
        engine.apply_epoch_start(state, epoch_idx)

        epoch_interactions: List[SoftInteraction] = []

        for step_idx, (narrative_text, expected_scores) in enumerate(epoch_samples):
            state.current_step = step_idx
            if isinstance(judge, CorpusJudge):
                judge.enqueue(expected_scores)

            interactions = adapter.process_narrative(
                agent_ids=agent_ids,
                narrative_text=narrative_text,
                step=step_idx,
            )

            for interaction in interactions:
                if not engine.can_agent_act(interaction.initiator, state):
                    continue

                effect = engine.apply_interaction(interaction, state)
                interaction.c_a += effect.cost_a
                interaction.c_b += effect.cost_b

                for agent_id, delta in effect.reputation_deltas.items():
                    if agent_id in state.agents:
                        state.agents[agent_id].reputation += delta

                for agent_id in effect.agents_to_freeze:
                    state.frozen_agents.add(agent_id)
                for agent_id in effect.agents_to_unfreeze:
                    state.frozen_agents.discard(agent_id)

                epoch_interactions.append(interaction)

                # Track payoffs
                cumulative_payoffs[interaction.initiator] += (
                    interaction.p * 2.0 - interaction.c_a
                )

            engine.apply_step(state, step_idx)

        # Compute epoch metrics
        em = DilemmaEpochMetrics(
            epoch=epoch_idx,
            frozen_count=len(state.frozen_agents),
            interactions_count=len(epoch_interactions),
        )

        if epoch_interactions:
            em.toxicity = metrics.toxicity_rate(epoch_interactions)
            em.quality_gap = metrics.quality_gap(epoch_interactions)
            welfare = metrics.welfare_metrics(epoch_interactions)
            em.welfare = welfare.get("total_welfare", 0.0)

        # Dilemma-specific metrics
        em.cooperation_rate = dilemma_state.overall_cooperation_rate()
        em.resource_level = (
            dilemma_state.resource_pool / dilemma_state.resource_capacity * 100
        )

        # Average contribution (PGG)
        all_contribs = []
        for agent_contribs in dilemma_state.contributions.values():
            all_contribs.extend(agent_contribs)
        em.avg_contribution = (
            sum(all_contribs) / len(all_contribs) if all_contribs else 0.0
        )

        # Norm strength: inverse std dev of per-agent cooperation rates
        per_agent_rates = [
            dilemma_state.agent_cooperation_rate(a) for a in agent_ids
        ]
        if len(per_agent_rates) > 1:
            mean_rate = sum(per_agent_rates) / len(per_agent_rates)
            variance = sum(
                (r - mean_rate) ** 2 for r in per_agent_rates
            ) / len(per_agent_rates)
            std_dev = variance ** 0.5
            em.norm_strength = 1.0 / (1.0 + std_dev)
        else:
            em.norm_strength = 1.0

        # Gini coefficient of cumulative payoffs
        em.gini_payoff = _gini(list(cumulative_payoffs.values()))

        result.epoch_metrics.append(em)

    result.compute_aggregates()
    return result


# ── Full sweep ───────────────────────────────────────────────────────────


def run_sweep(
    *,
    dilemmas: Optional[List[str]] = None,
    n_seeds: int = 3,
    n_epochs: int = 15,
    steps_per_epoch: int = 5,
    progress: bool = True,
    llm_client: LLMCallFn | None = None,
    llm_model: str = "llama3.2",
) -> List[DilemmaRunResult]:
    """Run all dilemma x governance config combinations."""
    if dilemmas is None:
        dilemmas = list(DILEMMA_CONFIGS.keys())

    configs = governance_configs()
    total_runs = len(dilemmas) * len(configs) * n_seeds
    results: List[DilemmaRunResult] = []
    current = 0

    for dilemma_name in dilemmas:
        for config_entry in configs:
            label = config_entry["label"]
            gov_config = config_entry["governance"]

            for seed_offset in range(n_seeds):
                current += 1
                seed = 42 + seed_offset

                if progress:
                    print(
                        f"  [{current:>3}/{total_runs}] "
                        f"{dilemma_name:<20} {label:<25} seed={seed}"
                    )

                result = run_single(
                    dilemma_name, label, gov_config,
                    n_epochs, steps_per_epoch, seed,
                    llm_client=llm_client,
                    llm_model=llm_model,
                )
                results.append(result)

    return results


# ── Export helpers ────────────────────────────────────────────────────────


def export_csv(results: List[DilemmaRunResult], path: Path) -> None:
    """Export aggregate results to CSV."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].to_dict().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def export_epoch_csv(results: List[DilemmaRunResult], path: Path) -> None:
    """Export per-epoch metrics to CSV."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dilemma", "label", "seed", "epoch", "toxicity", "welfare",
        "quality_gap", "frozen_count", "interactions_count",
        "cooperation_rate", "resource_level", "avg_contribution",
        "norm_strength", "gini_payoff",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            for em in r.epoch_metrics:
                writer.writerow({
                    "dilemma": r.dilemma,
                    "label": r.label,
                    "seed": r.seed,
                    "epoch": em.epoch,
                    "toxicity": em.toxicity,
                    "welfare": em.welfare,
                    "quality_gap": em.quality_gap,
                    "frozen_count": em.frozen_count,
                    "interactions_count": em.interactions_count,
                    "cooperation_rate": em.cooperation_rate,
                    "resource_level": em.resource_level,
                    "avg_contribution": em.avg_contribution,
                    "norm_strength": em.norm_strength,
                    "gini_payoff": em.gini_payoff,
                })


def export_history(results: List[DilemmaRunResult], path: Path) -> None:
    """Export full run history as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        entry = r.to_dict()
        entry["epoch_metrics"] = [
            {
                "epoch": em.epoch,
                "toxicity": em.toxicity,
                "welfare": em.welfare,
                "quality_gap": em.quality_gap,
                "frozen_count": em.frozen_count,
                "interactions_count": em.interactions_count,
                "cooperation_rate": em.cooperation_rate,
                "resource_level": em.resource_level,
                "avg_contribution": em.avg_contribution,
                "norm_strength": em.norm_strength,
                "gini_payoff": em.gini_payoff,
            }
            for em in r.epoch_metrics
        ]
        data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Plotting ─────────────────────────────────────────────────────────────


def generate_plots(results: List[DilemmaRunResult], plot_dir: Path) -> None:
    """Generate 6 comparison plots from sweep results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available - skipping plots")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Group by dilemma and label
    by_dilemma: Dict[str, Dict[str, List[DilemmaRunResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        by_dilemma[r.dilemma][r.label].append(r)

    dilemma_names = sorted(by_dilemma.keys())
    config_labels = [c["label"] for c in governance_configs()]

    # ── 1. Cooperation rate by governance config (grouped bars) ──────
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(config_labels))
    width = 0.25
    for i, dname in enumerate(dilemma_names):
        means = []
        for lbl in config_labels:
            runs = by_dilemma[dname].get(lbl, [])
            if runs:
                means.append(
                    sum(r.mean_cooperation_rate for r in runs) / len(runs)
                )
            else:
                means.append(0)
        ax.bar(x + i * width, means, width, label=dname)
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Cooperation Rate")
    ax.set_title("Cooperation Rate by Governance Config")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plot_dir / "cooperation_by_config.png", dpi=150)
    plt.close(fig)

    # ── 2. Cooperation over time (line plot, 3 subplots) ─────────────
    fig, axes = plt.subplots(1, len(dilemma_names), figsize=(5 * len(dilemma_names), 5), sharey=True)
    if len(dilemma_names) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_labels)))
    for ax_idx, dname in enumerate(dilemma_names):
        ax = axes[ax_idx]
        for c_idx, lbl in enumerate(config_labels):
            runs = by_dilemma[dname].get(lbl, [])
            if not runs:
                continue
            max_epochs = max(len(r.epoch_metrics) for r in runs)
            avg_coop = []
            for e in range(max_epochs):
                vals = [
                    r.epoch_metrics[e].cooperation_rate
                    for r in runs if e < len(r.epoch_metrics)
                ]
                avg_coop.append(sum(vals) / len(vals) if vals else 0)
            ax.plot(range(max_epochs), avg_coop, label=lbl, color=colors[c_idx])
        ax.set_title(dname.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        if ax_idx == 0:
            ax.set_ylabel("Cooperation Rate")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Cooperation Over Time by Dilemma", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "cooperation_over_time.png", dpi=150)
    plt.close(fig)

    # ── 3. Welfare vs cooperation scatter (Pareto frontier) ──────────
    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"commons": "o", "prisoners_dilemma": "s", "public_goods": "^"}
    for dname in dilemma_names:
        for lbl in config_labels:
            runs = by_dilemma[dname].get(lbl, [])
            if not runs:
                continue
            avg_coop = sum(r.mean_cooperation_rate for r in runs) / len(runs)
            avg_welfare = sum(r.mean_welfare for r in runs) / len(runs)
            ax.scatter(
                avg_coop, avg_welfare,
                marker=markers.get(dname, "o"), s=80, zorder=3,
            )
            ax.annotate(
                f"{lbl}\n({dname[:3]})",
                (avg_coop, avg_welfare), fontsize=6,
                textcoords="offset points", xytext=(5, 5),
            )
    ax.set_xlabel("Mean Cooperation Rate")
    ax.set_ylabel("Mean Welfare")
    ax.set_title("Welfare vs Cooperation (Pareto Frontier)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "pareto_welfare_cooperation.png", dpi=150)
    plt.close(fig)

    # ── 4. Commons resource depletion timeline ───────────────────────
    commons_results = by_dilemma.get("commons", {})
    if commons_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        for c_idx, lbl in enumerate(config_labels):
            runs = commons_results.get(lbl, [])
            if not runs:
                continue
            max_epochs = max(len(r.epoch_metrics) for r in runs)
            avg_resource = []
            for e in range(max_epochs):
                vals = [
                    r.epoch_metrics[e].resource_level
                    for r in runs if e < len(r.epoch_metrics)
                ]
                avg_resource.append(sum(vals) / len(vals) if vals else 0)
            ax.plot(range(max_epochs), avg_resource, label=lbl, color=colors[c_idx])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Resource Level (%)")
        ax.set_title("Commons Resource Depletion Over Time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(plot_dir / "commons_resource_depletion.png", dpi=150)
        plt.close(fig)

    # ── 5. Toxicity heatmap (governance x dilemma) ───────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    heatmap_data = []
    for lbl in config_labels:
        row = []
        for dname in dilemma_names:
            runs = by_dilemma[dname].get(lbl, [])
            if runs:
                row.append(sum(r.mean_toxicity for r in runs) / len(runs))
            else:
                row.append(0)
        heatmap_data.append(row)
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(config_labels)))
    ax.set_yticklabels(config_labels)
    ax.set_xticks(range(len(dilemma_names)))
    ax.set_xticklabels(
        [d.replace("_", "\n") for d in dilemma_names], fontsize=9
    )
    ax.set_title("Toxicity: Governance x Dilemma")
    fig.colorbar(im, ax=ax, label="Mean Toxicity")
    # Annotate cells
    for i in range(len(config_labels)):
        for j in range(len(dilemma_names)):
            ax.text(j, i, f"{heatmap_data[i][j]:.3f}",
                    ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "toxicity_heatmap.png", dpi=150)
    plt.close(fig)

    # ── 6. Inequality (Gini) by config ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(config_labels))
    width = 0.25
    for i, dname in enumerate(dilemma_names):
        means = []
        for lbl in config_labels:
            runs = by_dilemma[dname].get(lbl, [])
            if runs:
                means.append(sum(r.mean_gini for r in runs) / len(runs))
            else:
                means.append(0)
        ax.bar(x + i * width, means, width, label=dname)
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Gini Coefficient")
    ax.set_title("Payoff Inequality by Governance Config")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "inequality_by_config.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {plot_dir}/")


# ── Summary table ────────────────────────────────────────────────────────


def print_summary(results: List[DilemmaRunResult]) -> None:
    """Print a formatted summary table."""
    groups: Dict[Tuple[str, str], List[DilemmaRunResult]] = defaultdict(list)
    for r in results:
        groups[(r.dilemma, r.label)].append(r)

    print()
    print(
        f"{'Dilemma':<20} {'Config':<22} "
        f"{'CoopRate':>8} {'Trend':>7} {'Toxicity':>8} "
        f"{'Welfare':>8} {'Gini':>6} {'Norm':>6}"
    )
    print("-" * 95)

    for (dilemma, label) in sorted(groups.keys()):
        runs = groups[(dilemma, label)]
        n = len(runs)
        print(
            f"{dilemma:<20} {label:<22} "
            f"{sum(r.mean_cooperation_rate for r in runs) / n:>8.3f} "
            f"{sum(r.cooperation_trend for r in runs) / n:>+7.4f} "
            f"{sum(r.mean_toxicity for r in runs) / n:>8.4f} "
            f"{sum(r.mean_welfare for r in runs) / n:>8.2f} "
            f"{sum(r.mean_gini for r in runs) / n:>6.3f} "
            f"{sum(r.mean_norm_strength for r in runs) / n:>6.3f}"
        )
    print()


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Social Dilemma Norms Study - governance effects on cooperation"
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds per config (default: 3)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Epochs per run (default: 15)",
    )
    parser.add_argument(
        "--steps", type=int, default=5,
        help="Steps per epoch (default: 5)",
    )
    parser.add_argument(
        "--dilemma", type=str, default=None,
        choices=list(DILEMMA_CONFIGS.keys()),
        help="Run only one dilemma (default: all three)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: runs/<timestamp>_social_dilemma_norms/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configs and corpus generation without running sweep",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live LLM calls instead of synthetic corpus scores",
    )
    parser.add_argument(
        "--provider", type=str, default="ollama",
        choices=list(PROVIDER_DEFAULTS.keys()),
        help="LLM provider for --live mode (default: ollama)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name for --live mode (default: provider-specific)",
    )
    args = parser.parse_args()

    dilemmas = [args.dilemma] if args.dilemma else list(DILEMMA_CONFIGS.keys())
    n_configs = len(governance_configs())
    total_runs = len(dilemmas) * n_configs * args.seeds

    print("=" * 70)
    print("  Social Dilemma Norms Study")
    print("=" * 70)
    print()
    print(f"  Dilemmas:         {', '.join(dilemmas)}")
    print(f"  Configs:          {n_configs}")
    print(f"  Seeds/config:     {args.seeds}")
    print(f"  Epochs/run:       {args.epochs}")
    print(f"  Steps/epoch:      {args.steps}")
    print(f"  Total runs:       {total_runs}")
    live_model = args.model or PROVIDER_DEFAULTS[args.provider]["default_model"]
    if args.live:
        print(f"  Mode:             LIVE ({args.provider} / {live_model})")
    else:
        print("  Mode:             Synthetic (corpus-driven)")
    print()

    if args.dry_run:
        print("DRY RUN - validating configs...")
        for cfg in governance_configs():
            label = cfg["label"]
            gov = cfg["governance"]
            active = GovernanceEngine(config=gov).get_active_lever_names()
            print(f"  {label:<25} active levers: {active}")

        print()
        print("Validating corpus generation...")
        for dname in dilemmas:
            dcfg = DILEMMA_CONFIGS[dname]
            corpus, dstate = generate_dilemma_corpus(
                dilemma_type=dcfg["type"],
                agents=dcfg["agents"],
                n_epochs=3,
                steps_per_epoch=2,
                seed=42,
            )
            n_samples = sum(len(epoch) for epoch in corpus)
            print(
                f"  {dname:<20} epochs={len(corpus)} "
                f"samples={n_samples} "
                f"coop_rate={dstate.overall_cooperation_rate():.2f}"
            )

        print()
        print("All configs and corpora valid.")
        return 0

    # Determine output directory
    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = Path("runs") / f"{ts}_social_dilemma_norms"
    else:
        out_dir = args.output

    print(f"  Output:           {out_dir}")
    print()

    # Export sweep config
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_config = {
        "dilemmas": dilemmas,
        "n_configs": n_configs,
        "n_seeds": args.seeds,
        "n_epochs": args.epochs,
        "steps_per_epoch": args.steps,
        "total_runs": total_runs,
        "config_labels": [c["label"] for c in governance_configs()],
        "mode": "live" if args.live else "synthetic",
        "provider": args.provider if args.live else None,
        "model": live_model if args.live else None,
    }
    with open(out_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    # Build LLM client for live mode
    llm_client: LLMCallFn | None = None
    if args.live:
        llm_client = make_llm_client(provider=args.provider, model=live_model)

    # Run sweep
    print("Running sweep...")
    results = run_sweep(
        dilemmas=dilemmas,
        n_seeds=args.seeds,
        n_epochs=args.epochs,
        steps_per_epoch=args.steps,
        llm_client=llm_client,
        llm_model=live_model,
    )

    # Print summary
    print()
    print("=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print_summary(results)

    # Export
    csv_dir = out_dir / "csv"
    export_csv(results, csv_dir / "summary.csv")
    export_epoch_csv(results, csv_dir / "epochs.csv")
    export_history(results, out_dir / "history.json")
    print(f"CSV exported to {csv_dir}/")

    # Plots
    generate_plots(results, out_dir / "plots")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
