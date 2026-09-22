"""Experiment 2: Screening Protocol Frontier Shift

Tests whether contracts-as-screening moves configurations from inside the
Pareto frontier to on it. Two conditions:

  1. Baseline: Uniform governance — all agents face the same harness
  2. Treatment: Differentiated governance — agents with good track records
     get looser constraints via a screening protocol

The hypothesis is that screening does information work, allowing the system
to extend more autonomy selectively, moving the (capability, safety) point
northeast without uniformly relaxing constraints.

Usage:
    python experiments/screening_frontier.py [--n-seeds 50] [--n-agents 10]
    python experiments/screening_frontier.py --task long_horizon
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Pin `swarm` to THIS checkout (bead hjyp). `python experiments/x.py` puts
# experiments/ on sys.path[0], not the repo root, so `import swarm` otherwise
# resolves through the editable install, which points at whichever checkout ran
# `pip install -e .` last -- possibly another session's worktree.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)

from swarm.benchmarks.allocation.resource_auction import ResourceAuctionBenchmark
from swarm.benchmarks.base import TaskInstance, TaskResult
from swarm.benchmarks.coordination.distributed_allocation import (
    DistributedAllocationBenchmark,
)
from swarm.benchmarks.governance_run_fns import (
    _governance_friction,
)
from swarm.benchmarks.long_horizon.pipeline_task import PipelineTaskBenchmark
from swarm.benchmarks.routing.message_routing import MessageRoutingBenchmark
from swarm.benchmarks.runner import BenchmarkRunner

# ── Agent track records ────────────────────────────────────────────────
# Simulates a screening protocol: each agent has a trust score [0, 1]
# based on historical behavior. Trusted agents get reduced governance.


def _generate_trust_scores(n_agents: int, seed: int, adversarial_fraction: float = 0.2) -> dict[str, float]:
    """Generate trust scores — honest agents accumulate trust, adversarial don't.

    In a real system these would come from reputation/contract history.
    Here we simulate them: most agents are trustworthy (score 0.7-1.0),
    a fraction are adversarial (score 0.0-0.3).
    """
    rng = np.random.default_rng(seed + 999)
    n_adversarial = max(1, int(n_agents * adversarial_fraction))
    scores = {}
    for i in range(n_agents):
        if i < n_adversarial:
            scores[f"agent_{i}"] = float(rng.uniform(0.0, 0.3))
        else:
            scores[f"agent_{i}"] = float(rng.uniform(0.7, 1.0))
    return scores


def _differentiated_friction(
    gov_config: dict[str, Any],
    agent_trust: float,
    rng: np.random.Generator,
) -> float:
    """Compute friction adjusted by agent trust.

    Trusted agents face less friction (governance trusts them more).
    Untrusted agents face full or amplified friction.

    friction_effective = base_friction * (1 - trust_discount)
    trust_discount = agent_trust * screening_strength
    """
    base_friction = _governance_friction(gov_config, rng)
    screening_strength = float(gov_config.get("screening_strength", 0.5))
    trust_discount = agent_trust * screening_strength
    return base_friction * (1.0 - trust_discount)


# ── Screening-aware run functions ──────────────────────────────────────

def _screening_routing_run_fn(
    instance: TaskInstance,
    gov_config: dict[str, Any],
    trust_scores: dict[str, float],
) -> TaskResult:
    """Routing with per-agent governance based on trust scores."""
    rng = np.random.default_rng(instance.seed + hash(gov_config.get("id", "")) % 2**31)

    payload = getattr(instance, "payload", 0)
    source = getattr(instance, "source_agent", "")
    target = getattr(instance, "target_agent", "")
    network = getattr(instance, "network", None)

    if network is not None:
        from swarm.benchmarks.routing.message_routing import _bfs_path
        path = _bfs_path(network, source, target)
    else:
        path = [source, target]

    if not path:
        return TaskResult(completed=False, payload=None, steps_taken=0, agent_trace=[])

    # Per-hop friction based on each agent's trust
    total_extra_steps = 0
    corruption_happened = False
    for agent_id in path:
        trust = trust_scores.get(agent_id, 0.5)
        agent_friction = _differentiated_friction(gov_config, trust, rng)
        total_extra_steps += int(agent_friction * 3)

        if rng.random() < agent_friction * 0.4:
            corruption_happened = True

    actual_steps = len(path) - 1 + total_extra_steps
    if corruption_happened:
        payload = int(rng.integers(0, 10_000))

    return TaskResult(
        completed=True,
        payload=payload,
        steps_taken=actual_steps,
        agent_trace=list(path),
    )


def _screening_coordination_run_fn(
    instance: TaskInstance,
    gov_config: dict[str, Any],
    trust_scores: dict[str, float],
) -> TaskResult:
    """Coordination with per-agent friction based on trust."""
    rng = np.random.default_rng(instance.seed + hash(gov_config.get("id", "")) % 2**31)

    target_total = getattr(instance, "target_total", 0.0)
    capacities = getattr(instance, "agent_capacities", {})
    if not capacities:
        return TaskResult(completed=False, payload={}, steps_taken=0, agent_trace=[])

    total_cap = sum(capacities.values())
    ratio = target_total / total_cap if total_cap > 0 else 0.0

    allocation = {}
    max_steps = 1
    for aid, cap in capacities.items():
        trust = trust_scores.get(aid, 0.5)
        friction = _differentiated_friction(gov_config, trust, rng)
        ideal = cap * ratio
        noise = rng.normal(0, friction * ideal * 0.5)
        allocation[aid] = max(0.0, ideal + noise)
        max_steps = max(max_steps, 1 + int(friction * 5))

    return TaskResult(
        completed=True,
        payload=allocation,
        steps_taken=max_steps,
        agent_trace=sorted(capacities.keys()),
    )


def _screening_auction_run_fn(
    instance: TaskInstance,
    gov_config: dict[str, Any],
    trust_scores: dict[str, float],
) -> TaskResult:
    """Auction with per-agent bid noise based on trust."""
    rng = np.random.default_rng(instance.seed + hash(gov_config.get("id", "")) % 2**31)

    valuations = getattr(instance, "agent_valuations", {})
    n_resources = getattr(instance, "n_resources", 0)
    if not valuations:
        return TaskResult(completed=False, payload={}, steps_taken=0, agent_trace=[])

    agent_ids = list(valuations.keys())
    assignment: dict[int, str] = {}
    max_steps = 1

    for r in range(n_resources):
        noisy_vals = {}
        for aid in agent_ids:
            trust = trust_scores.get(aid, 0.5)
            friction = _differentiated_friction(gov_config, trust, rng)
            true_val = valuations[aid][r]
            noise = rng.normal(0, friction * true_val * 0.8)
            noisy_vals[aid] = max(0.0, true_val + noise)
            max_steps = max(max_steps, 1 + int(friction * 3))
        winner = max(agent_ids, key=lambda a: noisy_vals[a])
        assignment[r] = winner

    return TaskResult(
        completed=True,
        payload=assignment,
        steps_taken=max_steps,
        agent_trace=sorted(set(assignment.values())),
    )


def _screening_pipeline_run_fn(
    instance: TaskInstance,
    gov_config: dict[str, Any],
    trust_scores: dict[str, float],
) -> TaskResult:
    """Pipeline with per-stage friction based on agent trust."""
    from swarm.benchmarks.long_horizon.pipeline_task import _stage_transform

    rng = np.random.default_rng(instance.seed + hash(gov_config.get("id", "")) % 2**31)

    initial_payload = getattr(instance, "initial_payload", 0)
    stages = getattr(instance, "stages", [])
    if not stages:
        return TaskResult(completed=False, payload=0, steps_taken=0, agent_trace=[])

    payload = initial_payload
    trace: list[str] = []
    total_steps = 0

    for stage in stages:
        trust = trust_scores.get(stage.agent_id, 0.5)
        friction = _differentiated_friction(gov_config, trust, rng)

        if rng.random() < friction * 0.15:
            return TaskResult(completed=True, payload=payload, steps_taken=total_steps + 1, agent_trace=trace)

        payload = _stage_transform(payload, stage.transform_key)
        trace.append(stage.agent_id)
        total_steps += 1 + int(friction * 2)

    return TaskResult(completed=True, payload=payload, steps_taken=total_steps, agent_trace=trace)


SCREENING_RUN_FNS = {
    "routing": _screening_routing_run_fn,
    "coordination": _screening_coordination_run_fn,
    "allocation": _screening_auction_run_fn,
    "long_horizon": _screening_pipeline_run_fn,
}

BENCHMARKS = {
    "routing": MessageRoutingBenchmark(),
    "coordination": DistributedAllocationBenchmark(),
    "allocation": ResourceAuctionBenchmark(),
    "long_horizon": PipelineTaskBenchmark(),
}


# ── Governance configs for screening experiment ────────────────────────

def _make_configs(screening_strength: float) -> list[dict[str, Any]]:
    """Generate governance configs at different tightness levels."""
    base_configs = [
        {"id_base": "tight", "audit_rate": 0.8, "circuit_breaker_enabled": True,
         "circuit_breaker_sensitivity": 0.6, "min_stake": 6.0, "bandwidth_cap": 30,
         "confirmation_gates": 2, "adversarial_fraction": 0.2},
        {"id_base": "moderate", "audit_rate": 0.5, "circuit_breaker_enabled": True,
         "circuit_breaker_sensitivity": 0.4, "min_stake": 4.0, "bandwidth_cap": 50,
         "confirmation_gates": 1, "adversarial_fraction": 0.2},
        {"id_base": "light", "audit_rate": 0.15, "circuit_breaker_enabled": False,
         "min_stake": 1.0, "bandwidth_cap": 85, "confirmation_gates": 0,
         "adversarial_fraction": 0.2},
    ]
    configs = []
    for c in base_configs:
        config = dict(c)
        config["screening_strength"] = screening_strength
        suffix = "uniform" if screening_strength == 0.0 else f"screen_{screening_strength:.1f}"
        config["id"] = f"{c['id_base']}_{suffix}"
        del config["id_base"]
        configs.append(config)
    return configs


def run_screening_experiment(
    task_types: list[str],
    n_seeds: int = 50,
    n_agents: int = 10,
    output_dir: str = "runs/screening",
) -> dict[str, pd.DataFrame]:
    """Run paired baseline vs screening experiments."""
    from swarm.benchmarks.governance_run_fns import RUN_FN_REGISTRY

    results = {}

    for task_type in task_types:
        benchmark = BENCHMARKS[task_type]
        uniform_run_fn = RUN_FN_REGISTRY[task_type]
        screening_run_fn = SCREENING_RUN_FNS[task_type]

        # Baseline: uniform governance (screening_strength = 0)
        baseline_configs = _make_configs(screening_strength=0.0)
        # Treatment: differentiated governance (screening_strength = 0.5)
        treatment_configs = _make_configs(screening_strength=0.5)

        runner = BenchmarkRunner(n_agents=n_agents)

        print(f"\n{'='*60}")
        print(f"Screening experiment: {task_type}")
        print(f"{'='*60}")

        # Run baseline (uniform governance for all agents)
        print("  Running baseline (uniform governance)...")
        df_baseline = runner.run_frontier(
            benchmark=benchmark,
            governance_configs=baseline_configs,
            n_seeds=n_seeds,
            run_fn=uniform_run_fn,
        )
        df_baseline["condition"] = "baseline"

        # Run treatment (screening-differentiated governance)
        print("  Running treatment (screening protocol)...")
        all_treatment_rows = []
        for seed in range(n_seeds):
            trust_scores = _generate_trust_scores(n_agents, seed)
            instance, oracle = benchmark.generate(seed, n_agents)

            for config in treatment_configs:
                redacted = benchmark.redact(instance)
                result = screening_run_fn(redacted, config, trust_scores)
                score = benchmark.score(result, oracle, adversarial_fraction=float(config.get("adversarial_fraction", 0.0)))
                interaction = benchmark.to_soft_interaction(score)

                from dataclasses import asdict
                row = {
                    "benchmark": benchmark.task_id,
                    "task_type": benchmark.task_type,
                    "gov_config": config["id"],
                    "seed": seed,
                    "adversarial_fraction": float(config.get("adversarial_fraction", 0.0)),
                    "p": interaction.p,
                    "condition": "screening",
                    **asdict(score),
                }
                for k, v in config.items():
                    if k not in ("id", "adversarial_fraction"):
                        row[f"gov_{k}"] = v
                all_treatment_rows.append(row)

        df_treatment = pd.DataFrame(all_treatment_rows)

        # Combine
        df = pd.concat([df_baseline, df_treatment], ignore_index=True)
        df["capability"] = df["completion_rate"] * 0.6 + df["fidelity"] * 0.3 + df["efficiency"] * 0.1

        results[task_type] = df

        # Print comparison
        for base_name in ["tight", "moderate", "light"]:
            bl = df[(df["condition"] == "baseline") & (df["gov_config"].str.startswith(base_name))]
            tr = df[(df["condition"] == "screening") & (df["gov_config"].str.startswith(base_name))]
            if len(bl) > 0 and len(tr) > 0:
                bl_cap = bl["capability"].mean()
                tr_cap = tr["capability"].mean()
                bl_p5 = bl["p"].quantile(0.05)
                tr_p5 = tr["p"].quantile(0.05)
                print(f"  {base_name:12s}  baseline cap={bl_cap:.3f} p5={bl_p5:.3f}  |  screening cap={tr_cap:.3f} p5={tr_p5:.3f}  |  delta_cap={tr_cap-bl_cap:+.3f}  delta_p5={tr_p5-bl_p5:+.3f}")

    return results


def save_results(results: dict[str, pd.DataFrame], output_dir: str) -> Path:
    """Save screening experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{timestamp}_screening"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = run_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    for task_type, df in results.items():
        df.to_csv(csv_dir / f"{task_type}_raw.csv", index=False)

    metadata = {
        "experiment": "screening_frontier_shift",
        "timestamp": timestamp,
        "task_types": list(results.keys()),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to {run_dir}")
    return run_dir


def plot_screening_comparison(results: dict[str, pd.DataFrame], run_dir: Path) -> None:
    """Plot paired baseline vs screening frontier comparison."""
    import matplotlib.pyplot as plt

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for task_type, df in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: frontier scatter, colored by condition
        for condition, marker, alpha in [("baseline", "o", 0.6), ("screening", "^", 0.8)]:
            subset = df[df["condition"] == condition]
            summary = subset.groupby("gov_config")[["capability", "p"]].agg(["mean", "std"])

            for config in summary.index:
                cap_mean = summary.loc[config, ("capability", "mean")]
                cap_std = summary.loc[config, ("capability", "std")]
                p_mean = summary.loc[config, ("p", "mean")]
                p_std = summary.loc[config, ("p", "std")]

                color = "#d62728" if condition == "baseline" else "#2ca02c"
                axes[0].errorbar(
                    cap_mean, p_mean, xerr=cap_std, yerr=p_std,
                    fmt=marker, color=color, markersize=9, capsize=3,
                    alpha=alpha, label=f"{config}" if config == summary.index[0] else "",
                )

        # Draw arrows from baseline to screening for matched configs
        for base_name in ["tight", "moderate", "light"]:
            bl = df[(df["condition"] == "baseline") & (df["gov_config"].str.startswith(base_name))]
            tr = df[(df["condition"] == "screening") & (df["gov_config"].str.startswith(base_name))]
            if len(bl) > 0 and len(tr) > 0:
                bx, by = bl["capability"].mean(), bl["p"].mean()
                tx, ty = tr["capability"].mean(), tr["p"].mean()
                axes[0].annotate(
                    "", xy=(tx, ty), xytext=(bx, by),
                    arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.5, "alpha": 0.6},
                )
                axes[0].text((bx + tx) / 2, (by + ty) / 2 + 0.02, base_name,
                             fontsize=8, ha="center", color="gray")

        axes[0].set_xlabel("Capability", fontsize=11)
        axes[0].set_ylabel("p = P(beneficial)", fontsize=11)
        axes[0].set_title(f"Screening Frontier Shift: {task_type}", fontsize=13)
        # Manual legend
        import matplotlib.patches as mpatches
        baseline_patch = mpatches.Patch(color="#d62728", label="Baseline (uniform)")
        screening_patch = mpatches.Patch(color="#2ca02c", label="Screening (differentiated)")
        axes[0].legend(handles=[baseline_patch, screening_patch], loc="best")
        axes[0].grid(True, alpha=0.3)

        # Right: paired bar chart of capability deltas
        base_names = ["tight", "moderate", "light"]
        deltas_cap = []
        deltas_p5 = []
        for base_name in base_names:
            bl = df[(df["condition"] == "baseline") & (df["gov_config"].str.startswith(base_name))]
            tr = df[(df["condition"] == "screening") & (df["gov_config"].str.startswith(base_name))]
            if len(bl) > 0 and len(tr) > 0:
                deltas_cap.append(tr["capability"].mean() - bl["capability"].mean())
                deltas_p5.append(tr["p"].quantile(0.05) - bl["p"].quantile(0.05))
            else:
                deltas_cap.append(0)
                deltas_p5.append(0)

        x = np.arange(len(base_names))
        width = 0.35
        axes[1].bar(x - width / 2, deltas_cap, width, label="Capability delta", color="#1f77b4")
        axes[1].bar(x + width / 2, deltas_p5, width, label="5th pctile p delta", color="#ff7f0e")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(base_names)
        axes[1].set_ylabel("Delta (screening - baseline)", fontsize=11)
        axes[1].set_title(f"Screening Effect Size: {task_type}", fontsize=13)
        axes[1].axhline(y=0, color="black", linewidth=0.5)
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(plots_dir / f"{task_type}_screening.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {task_type}_screening.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Screening protocol frontier shift experiment")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="runs/screening")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--task", type=str, default="routing", choices=list(BENCHMARKS.keys()))
    args = parser.parse_args()

    task_types = list(BENCHMARKS.keys()) if args.all_tasks else [args.task]

    results = run_screening_experiment(
        task_types=task_types,
        n_seeds=args.n_seeds,
        n_agents=args.n_agents,
        output_dir=args.output_dir,
    )

    run_dir = save_results(results, args.output_dir)
    plot_screening_comparison(results, run_dir)


if __name__ == "__main__":
    main()
