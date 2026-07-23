#!/usr/bin/env python
"""
First Governance Experiment

A beginner-friendly example demonstrating the effect of a transaction tax
on ecosystem health in a mixed-agent population.

This example accompanies the tutorial:
  docs/tutorials/first-governance-experiment.md

Usage:
    python examples/first_governance_experiment.py

What it does:
    1. Runs a baseline scenario (no governance)
    2. Runs a taxed scenario (5% transaction tax)
    3. Compares toxicity rate, quality gap, and mean payoff
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction

# ── Helpers ──────────────────────────────────────────────────────────────────


def make_interaction(
    p: float,
    accepted: bool = True,
    tax: float = 0.0,
) -> SoftInteraction:
    """Create a synthetic interaction with the given soft label p."""
    return SoftInteraction(
        interaction_id=f"i-{p:.2f}-{accepted}",
        initiator="agent_a",
        counterparty="agent_b",
        p=p,
        v_hat=p * 2 - 1,  # linear heuristic: maps [0,1]→[-1,1] (simplified approximation, not a true sigmoid inverse)
        accepted=accepted,
    )


def simulate_condition(
    tax: float,
    n_interactions: int = 200,
    seed: int = 42,
) -> dict:
    """
    Simulate a simple ecosystem with a mix of honest and opportunistic behavior.

    Returns summary metrics for the condition.
    """
    import random

    rng = random.Random(seed)

    proxy = ProxyComputer()
    payoff_config = PayoffConfig(s_plus=2.0, s_minus=1.0, h=2.0, theta=0.5)
    engine = SoftPayoffEngine(config=payoff_config)
    metrics = SoftMetrics(payoff_engine=engine)

    interactions = []

    for _ in range(n_interactions):
        # 50% honest (high p), 30% opportunistic (medium p), 20% deceptive (low p)
        r = rng.random()
        if r < 0.50:
            # Honest: high task progress, low rework
            obs = ProxyObservables(
                task_progress_delta=rng.uniform(0.5, 0.9),
                rework_count=rng.randint(0, 1),
                verifier_rejections=0,
                counterparty_engagement_delta=rng.uniform(0.3, 0.7),
            )
        elif r < 0.80:
            # Opportunistic: moderate progress, some rework
            obs = ProxyObservables(
                task_progress_delta=rng.uniform(0.1, 0.4),
                rework_count=rng.randint(1, 3),
                verifier_rejections=rng.randint(0, 1),
                counterparty_engagement_delta=rng.uniform(-0.1, 0.3),
            )
        else:
            # Deceptive: low/negative progress, high rework
            obs = ProxyObservables(
                task_progress_delta=rng.uniform(-0.3, 0.2),
                rework_count=rng.randint(3, 7),
                verifier_rejections=rng.randint(1, 3),
                counterparty_engagement_delta=rng.uniform(-0.5, 0.0),
            )

        _, p = proxy.compute_labels(obs)

        # Acceptance decision: tax raises the bar
        # Agents accept if expected surplus minus tax exceeds zero
        expected_surplus = p * payoff_config.s_plus - (1 - p) * payoff_config.s_minus
        accepted = (expected_surplus - tax) > 0.0

        interactions.append(make_interaction(p=p, accepted=accepted, tax=tax))

    # Compute metrics
    toxicity = metrics.toxicity_rate(interactions)
    q_gap = metrics.quality_gap(interactions)

    accepted = [i for i in interactions if i.accepted]
    mean_payoff = 0.0
    if accepted:
        payoffs = [engine.payoff_initiator(i) - tax for i in accepted]
        mean_payoff = sum(payoffs) / len(payoffs)

    return {
        "n_total": len(interactions),
        "n_accepted": len(accepted),
        "toxicity_rate": toxicity,
        "quality_gap": q_gap,
        "mean_payoff": mean_payoff,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 60)
    print("First Governance Experiment")
    print("Does a transaction tax improve ecosystem health?")
    print("=" * 60)
    print()

    conditions = {
        "Baseline (no tax)": 0.00,
        "Light tax (1%)":    0.01,
        "Medium tax (5%)":   0.05,
        "Heavy tax (10%)":   0.10,
    }

    print(f"{'Condition':22s}  {'Toxicity':>9s}  {'Quality Gap':>12s}  {'Mean Payoff':>11s}  {'Accepted':>8s}")
    print("-" * 72)

    for name, tax in conditions.items():
        result = simulate_condition(tax=tax, n_interactions=500, seed=42)
        toxicity = result["toxicity_rate"]
        q_gap = result["quality_gap"]
        payoff = result["mean_payoff"]
        n_acc = result["n_accepted"]
        n_tot = result["n_total"]

        # Simple health indicators
        tox_marker = "✓" if toxicity < 0.20 else ("⚠" if toxicity < 0.35 else "✗")
        gap_marker = "✓" if q_gap > 0 else "✗"

        print(
            f"{name:22s}  "
            f"{toxicity:>7.3f} {tox_marker}  "
            f"{q_gap:>+10.3f} {gap_marker}  "
            f"{payoff:>10.3f}   "
            f"{n_acc:>4d}/{n_tot}"
        )

    print()
    print("Key:")
    print("  Toxicity    ✓ < 0.20  ⚠ 0.20–0.35  ✗ > 0.35")
    print("  Quality Gap ✓ positive (healthy selection)  ✗ negative (adverse selection)")
    print()
    print("Observations:")
    print("  - Quality gap remains positive: acceptance threshold filters low-quality interactions")
    print("  - Heavier taxes reduce throughput (fewer accepted interactions)")
    print("  - Toxicity is driven by deceptive agent prevalence (20% of population)")
    print("  - Tax trades efficiency (lower mean payoff) for slight toxicity reduction")
    print()
    print("Next step: try the parameter sweep guide to find the optimal tax rate.")
    print("  See: docs/tutorials/first-governance-experiment.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
