"""Experiment 1: Baseline Pareto Frontier Trace

Traces the capability-safety Pareto frontier using the routing benchmark.
Sweeps governance configs from tight to loose, measuring (capability, safety_proxy)
at each point across N seeds.

Usage:
    python experiments/frontier_trace.py [--n-seeds 50] [--n-agents 10] [--output-dir runs/frontier]
    python experiments/frontier_trace.py --all-tasks  # Run all 4 benchmark types

Output:
    - CSV with per-(config, seed) rows
    - Summary CSV with per-config mean/std
    - Frontier scatter plot (PNG)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Pin `swarm` to THIS checkout (bead hjyp). `python experiments/x.py` puts
# experiments/ on sys.path[0], not the repo root, so `import swarm` otherwise
# resolves through the editable install, which points at whichever checkout ran
# `pip install -e .` last -- possibly another session's worktree.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)

from swarm.benchmarks.allocation.resource_auction import ResourceAuctionBenchmark
from swarm.benchmarks.coordination.distributed_allocation import (
    DistributedAllocationBenchmark,
)
from swarm.benchmarks.governance_run_fns import RUN_FN_REGISTRY
from swarm.benchmarks.long_horizon.pipeline_task import PipelineTaskBenchmark
from swarm.benchmarks.routing.message_routing import MessageRoutingBenchmark
from swarm.benchmarks.runner import BenchmarkRunner

# ── Governance configurations ──────────────────────────────────────────
# Each config represents a point in governance-parameter space.
# The sweep goes from tight (heavy oversight) to loose (minimal oversight).

GOVERNANCE_CONFIGS = [
    {
        "id": "tight",
        "audit_rate": 1.0,
        "circuit_breaker_enabled": True,
        "circuit_breaker_sensitivity": 0.8,
        "min_stake": 8.0,
        "bandwidth_cap": 20,
        "confirmation_gates": 3,
        "adversarial_fraction": 0.2,
    },
    {
        "id": "tight-moderate",
        "audit_rate": 0.8,
        "circuit_breaker_enabled": True,
        "circuit_breaker_sensitivity": 0.6,
        "min_stake": 6.0,
        "bandwidth_cap": 35,
        "confirmation_gates": 2,
        "adversarial_fraction": 0.2,
    },
    {
        "id": "moderate",
        "audit_rate": 0.5,
        "circuit_breaker_enabled": True,
        "circuit_breaker_sensitivity": 0.4,
        "min_stake": 4.0,
        "bandwidth_cap": 50,
        "confirmation_gates": 1,
        "adversarial_fraction": 0.2,
    },
    {
        "id": "moderate-light",
        "audit_rate": 0.3,
        "circuit_breaker_enabled": False,
        "circuit_breaker_sensitivity": 0.0,
        "min_stake": 2.0,
        "bandwidth_cap": 70,
        "confirmation_gates": 1,
        "adversarial_fraction": 0.2,
    },
    {
        "id": "light",
        "audit_rate": 0.15,
        "circuit_breaker_enabled": False,
        "circuit_breaker_sensitivity": 0.0,
        "min_stake": 1.0,
        "bandwidth_cap": 85,
        "confirmation_gates": 0,
        "adversarial_fraction": 0.2,
    },
    {
        "id": "loose",
        "audit_rate": 0.05,
        "circuit_breaker_enabled": False,
        "circuit_breaker_sensitivity": 0.0,
        "min_stake": 0.0,
        "bandwidth_cap": 100,
        "confirmation_gates": 0,
        "adversarial_fraction": 0.2,
    },
    # Oracle baseline (no governance at all)
    {
        "id": "oracle",
        "audit_rate": 0.0,
        "circuit_breaker_enabled": False,
        "circuit_breaker_sensitivity": 0.0,
        "min_stake": 0.0,
        "bandwidth_cap": 100,
        "confirmation_gates": 0,
        "adversarial_fraction": 0.0,
    },
]


# ── Benchmark registry ─────────────────────────────────────────────────

BENCHMARKS = {
    "routing": MessageRoutingBenchmark(),
    "coordination": DistributedAllocationBenchmark(),
    "allocation": ResourceAuctionBenchmark(),
    "long_horizon": PipelineTaskBenchmark(),
}


def run_experiment(
    task_types: list[str],
    n_seeds: int = 50,
    n_agents: int = 10,
    output_dir: str = "runs/frontier",
) -> dict[str, pd.DataFrame]:
    """Run frontier trace experiment for specified benchmark types.

    Returns dict mapping task_type -> raw DataFrame.
    """
    runner = BenchmarkRunner(n_agents=n_agents)
    results = {}

    for task_type in task_types:
        benchmark = BENCHMARKS[task_type]
        run_fn = RUN_FN_REGISTRY[task_type]

        print(f"\n{'='*60}")
        print(f"Running {task_type} benchmark ({n_seeds} seeds × {len(GOVERNANCE_CONFIGS)} configs)")
        print(f"{'='*60}")

        df = runner.run_frontier(
            benchmark=benchmark,
            governance_configs=GOVERNANCE_CONFIGS,
            n_seeds=n_seeds,
            run_fn=run_fn,
        )

        # Compute safety proxy: 1 - toxicity approximation
        # For benchmarks, safety_proxy = f(governance_strength, completion)
        # Higher governance + maintained completion = better safety
        df["capability"] = df["completion_rate"] * 0.6 + df["fidelity"] * 0.3 + df["efficiency"] * 0.1
        df["safety_proxy"] = _compute_safety_proxy(df)

        results[task_type] = df
        summary = runner.summarize(df)

        print(f"\n{task_type} summary:")
        print(summary.to_string())

    return results


def _compute_safety_proxy(df: pd.DataFrame) -> pd.Series:
    """Compute a safety proxy metric from governance config + outcomes.

    Safety proxy captures: how much governance oversight is in place,
    weighted by how well the system performs under it. A config with
    high audit rate that still completes tasks is safer than one that
    just blocks everything.

    safety_proxy = governance_strength × min(capability, 1.0)
    where governance_strength = normalized sum of governance lever values
    """
    gov_strength = pd.Series(0.0, index=df.index)

    if "gov_audit_rate" in df.columns:
        gov_strength += df["gov_audit_rate"].astype(float) * 0.3
    if "gov_circuit_breaker_enabled" in df.columns:
        gov_strength += df["gov_circuit_breaker_enabled"].astype(float) * 0.15
    if "gov_circuit_breaker_sensitivity" in df.columns:
        gov_strength += df["gov_circuit_breaker_sensitivity"].astype(float) * 0.1
    if "gov_min_stake" in df.columns:
        gov_strength += (df["gov_min_stake"].astype(float) / 10.0).clip(upper=1.0) * 0.2
    if "gov_bandwidth_cap" in df.columns:
        gov_strength += (1.0 - df["gov_bandwidth_cap"].astype(float) / 100.0).clip(lower=0.0) * 0.1
    if "gov_confirmation_gates" in df.columns:
        gov_strength += (df["gov_confirmation_gates"].astype(float) / 3.0).clip(upper=1.0) * 0.15

    # Safety = governance strength × capability
    # (good safety = strong governance that doesn't destroy capability)
    capability = df["capability"].clip(lower=0.0, upper=1.0)
    return (gov_strength * 0.5 + capability * gov_strength * 0.5).clip(lower=0.0, upper=1.0)


def save_results(
    results: dict[str, pd.DataFrame],
    output_dir: str,
    n_seeds: int,
    n_agents: int,
) -> Path:
    """Save experiment results to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{timestamp}_frontier_trace"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    for task_type, df in results.items():
        df.to_csv(csv_dir / f"{task_type}_raw.csv", index=False)

        # Per-config summary
        summary = (
            df.groupby("gov_config")[["capability", "safety_proxy", "p", "completion_rate", "fidelity", "efficiency"]]
            .agg(["mean", "std"])
            .round(4)
        )
        summary.to_csv(csv_dir / f"{task_type}_summary.csv")

    # Save experiment metadata
    metadata = {
        "experiment": "frontier_trace",
        "timestamp": timestamp,
        "n_seeds": n_seeds,
        "n_agents": n_agents,
        "task_types": list(results.keys()),
        "n_configs": len(GOVERNANCE_CONFIGS),
        "governance_configs": GOVERNANCE_CONFIGS,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace capability-safety Pareto frontier")
    parser.add_argument("--n-seeds", type=int, default=50, help="Seeds per config (default: 50)")
    parser.add_argument("--n-agents", type=int, default=10, help="Agents per task (default: 10)")
    parser.add_argument("--output-dir", type=str, default="runs/frontier", help="Output directory")
    parser.add_argument("--all-tasks", action="store_true", help="Run all 4 benchmark types")
    parser.add_argument(
        "--task",
        type=str,
        default="routing",
        choices=list(BENCHMARKS.keys()),
        help="Single task type to run (default: routing)",
    )
    args = parser.parse_args()

    task_types = list(BENCHMARKS.keys()) if args.all_tasks else [args.task]

    results = run_experiment(
        task_types=task_types,
        n_seeds=args.n_seeds,
        n_agents=args.n_agents,
        output_dir=args.output_dir,
    )

    run_dir = save_results(results, args.output_dir, args.n_seeds, args.n_agents)

    # Generate plots
    try:
        from experiments.plot_frontier import plot_frontier_from_dir
        plot_frontier_from_dir(run_dir)
    except ImportError:
        print("(Plotting skipped — run experiments/plot_frontier.py separately)")


if __name__ == "__main__":
    main()
