#!/usr/bin/env python
"""
Cascade Risk Lever — parameter sensitivity sweep.

Sweeps cascade_risk_threshold × cascade_risk_p_threshold across a grid
to identify which values produce meaningful separation in agent outcomes.
Reports per-config: mean cascade_risk, penalty count, reputation delta,
and whether the lever fires at all.

This script validates (or invalidates) the provisional defaults in
GovernanceConfig:
    cascade_risk_threshold   = 0.5
    cascade_risk_p_threshold = 0.3
    cascade_risk_window      = 200

Usage:
    python examples/run_cascade_sensitivity.py [scenarios/baseline.yaml]
"""

import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.8]
P_THRESHOLDS = [0.15, 0.3, 0.45, 0.6]
SEEDS = [42, 123, 456]
N_EPOCHS = 5
STEPS_PER_EPOCH = 20


def run_one(scenario_path: str, threshold: float, p_thresh: float,
            window: int, seed: int) -> dict:
    """Run a single configuration and return summary metrics."""
    config = load_scenario(scenario_path)

    # Override governance config for cascade risk
    gov_overrides = {
        "cascade_risk_enabled": True,
        "cascade_risk_threshold": threshold,
        "cascade_risk_p_threshold": p_thresh,
        "cascade_risk_window": window,
        "cascade_risk_penalty_scale": 1.0,
        "cascade_risk_reputation_scale": 0.5,
    }
    for k, v in gov_overrides.items():
        if hasattr(config.governance, k):
            setattr(config.governance, k, v)

    # Set seed and epoch config before building orchestrator
    config.orchestrator_config.seed = seed
    config.orchestrator_config.n_epochs = N_EPOCHS
    config.orchestrator_config.steps_per_epoch = STEPS_PER_EPOCH

    orch = build_orchestrator(config)

    orch.run()

    # Collect cascade lever effects from completed interactions
    lever_fires = 0
    total_cost = 0.0
    total_rep_penalty = 0.0
    for ix in orch.state.completed_interactions:
        # Check if cascade risk lever fired by looking at governance costs
        if ix.c_a > 0 or ix.c_b > 0:
            lever_fires += 1
            total_cost += ix.c_a + ix.c_b

    # Reputation summary
    for _agent_id, agent_state in orch.state.agents.items():
        total_rep_penalty += min(0, agent_state.reputation)

    return {
        "threshold": threshold,
        "p_threshold": p_thresh,
        "window": window,
        "seed": seed,
        "total_interactions": len(orch.state.completed_interactions),
        "lever_fires": lever_fires,
        "total_cost": round(total_cost, 4),
        "total_rep_penalty": round(total_rep_penalty, 4),
        "fire_rate": round(lever_fires / max(1, len(orch.state.completed_interactions)), 4),
    }


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "scenarios/baseline.yaml"
    if not Path(scenario).exists():
        print(f"Scenario not found: {scenario}")
        sys.exit(1)

    results = []
    configs = list(product(THRESHOLDS, P_THRESHOLDS, [200], SEEDS))
    total = len(configs)

    print(f"Running {total} configurations ({len(THRESHOLDS)} thresholds "
          f"× {len(P_THRESHOLDS)} p_thresholds × {len(SEEDS)} seeds)")
    print(f"Scenario: {scenario}\n")

    for i, (thresh, p_thresh, window, seed) in enumerate(configs, 1):
        print(f"  [{i}/{total}] threshold={thresh}, p_threshold={p_thresh}, "
              f"seed={seed} ... ", end="", flush=True)
        try:
            result = run_one(scenario, thresh, p_thresh, window, seed)
            results.append(result)
            print(f"fires={result['lever_fires']}, "
                  f"rate={result['fire_rate']:.1%}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Aggregate across seeds
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean across seeds)")
    print("=" * 70)
    print(f"{'threshold':>10} {'p_thresh':>10} {'fire_rate':>10} "
          f"{'avg_cost':>10} {'avg_rep':>10} {'separation':>10}")
    print("-" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["threshold"], r["p_threshold"])].append(r)

    fire_rates = []
    for (thresh, p_thresh), runs in sorted(grouped.items()):
        avg_fire = sum(r["fire_rate"] for r in runs) / len(runs)
        avg_cost = sum(r["total_cost"] for r in runs) / len(runs)
        avg_rep = sum(r["total_rep_penalty"] for r in runs) / len(runs)
        fire_rates.append(avg_fire)
        print(f"{thresh:>10.2f} {p_thresh:>10.2f} {avg_fire:>10.1%} "
              f"{avg_cost:>10.2f} {avg_rep:>10.2f} {'---':>10}")

    # Separation = range of fire rates across configs
    if fire_rates:
        separation = max(fire_rates) - min(fire_rates)
        print(f"\nFire rate range: {min(fire_rates):.1%} – {max(fire_rates):.1%} "
              f"(separation = {separation:.1%})")
        if separation < 0.05:
            print("WARNING: Low separation — defaults may not meaningfully "
                  "differentiate agent outcomes. Consider adjusting thresholds.")

    # Export
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"runs/{ts}_cascade_sensitivity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to {out_path}")


if __name__ == "__main__":
    main()
