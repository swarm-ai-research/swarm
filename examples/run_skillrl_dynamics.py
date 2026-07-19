#!/usr/bin/env python
"""Run SkillRL dynamics study across multiple seeds and generate plots.

Loads the skillrl scenario, runs it with extended epochs across multiple seeds,
and captures per-epoch snapshots of skill evolution dynamics (library growth,
type composition, GRPO baseline, threshold drift, payoffs by agent type).

Usage:
    python examples/run_skillrl_dynamics.py --seeds 10
    python examples/run_skillrl_dynamics.py --seeds 5 --epochs 30 --steps 15
    python examples/run_skillrl_dynamics.py --from-json runs/<dir>/snapshots.json

Outputs:
    runs/<timestamp>_skillrl_dynamics/snapshots.json
    runs/<timestamp>_skillrl_dynamics/plots/*.png (6 plots)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.scenarios import build_orchestrator, load_scenario


def _snapshot_epoch(orchestrator, epoch: int) -> Dict[str, Any]:
    """Capture per-epoch snapshot of all agent states and skill metrics."""
    snapshot: Dict[str, Any] = {"epoch": epoch, "agents": []}

    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        # SkillRL agents have a skill_summary method, so detect by attribute
        is_skillrl = hasattr(agent, "skill_summary")
        effective_type = "skillrl" if is_skillrl else agent.agent_type.value

        agent_data: Dict[str, Any] = {
            "agent_id": agent.agent_id,
            "agent_type": effective_type,
            "total_payoff": state.total_payoff if state else 0.0,
            "reputation": state.reputation if state else 0.0,
        }

        # SkillRL-specific metrics
        if is_skillrl:
            summary = agent.skill_summary()
            agent_data["skill_summary"] = summary

            # Compute mean acceptance_threshold_delta across active skills
            deltas = []
            for skill in agent.skill_library.all_skills:
                d = skill.effect.get("acceptance_threshold_delta", 0.0)
                deltas.append(d)
            agent_data["mean_threshold_delta"] = (
                sum(deltas) / len(deltas) if deltas else 0.0
            )

            # Avg skill effectiveness from library performance data
            effs = []
            for skill in agent.skill_library.all_skills:
                perf = agent.skill_library.get_performance(skill.skill_id)
                if perf and perf.invocations > 0:
                    effs.append(perf.effectiveness)
            agent_data["avg_effectiveness"] = (
                sum(effs) / len(effs) if effs else 0.0
            )

        snapshot["agents"].append(agent_data)

    return snapshot


def run_single_seed(
    scenario_path: Path, seed: int, n_epochs: int, steps_per_epoch: int
) -> List[Dict[str, Any]]:
    """Run one seed and return list of per-epoch snapshots."""
    scenario = load_scenario(scenario_path)

    # Override config for this seed
    scenario.orchestrator_config.seed = seed
    scenario.orchestrator_config.n_epochs = n_epochs
    scenario.orchestrator_config.steps_per_epoch = steps_per_epoch

    orchestrator = build_orchestrator(scenario)
    snapshots: List[Dict[str, Any]] = []

    # Register epoch-end callback to capture snapshots
    def on_epoch(epoch_metrics):
        snap = _snapshot_epoch(orchestrator, epoch_metrics.epoch)
        snapshots.append(snap)

    orchestrator.on_epoch_end(on_epoch)

    # Run
    orchestrator.run()
    return snapshots


def run_dynamics_study(
    scenario_path: Path,
    seeds: List[int],
    n_epochs: int,
    steps_per_epoch: int,
    output_dir: Path,
) -> Path:
    """Run across all seeds and save combined snapshots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    all_seed_data: Dict[str, Any] = {
        "scenario": str(scenario_path),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "timestamp": datetime.now().isoformat(),
        "runs": {},
    }

    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i + 1}/{len(seeds)})...", end=" ", flush=True)
        t0 = time.time()
        snapshots = run_single_seed(scenario_path, seed, n_epochs, steps_per_epoch)
        elapsed = time.time() - t0
        all_seed_data["runs"][str(seed)] = snapshots
        print(f"done ({elapsed:.1f}s, {len(snapshots)} epochs)")

    # Save
    snapshots_path = output_dir / "snapshots.json"
    with open(snapshots_path, "w") as f:
        json.dump(all_seed_data, f, indent=2, default=str)
    print(f"\nSnapshots saved to {snapshots_path}")

    return snapshots_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SkillRL dynamics study with multi-seed plots"
    )
    parser.add_argument(
        "--seeds", type=int, default=10, help="Number of seeds (default: 10)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Epochs per run (default: 50)"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Steps per epoch (default: 10)"
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("scenarios/skillrl.yaml"),
        help="Scenario YAML (default: scenarios/skillrl.yaml)",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Skip simulation, plot from saved snapshots.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/<timestamp>_skillrl_dynamics)",
    )
    args = parser.parse_args()

    # Determine output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / f"{ts}_skillrl_dynamics"

    if args.from_json:
        snapshots_path = args.from_json
        output_dir = snapshots_path.parent
        print(f"Loading saved snapshots from {snapshots_path}")
    else:
        # Validate scenario
        if not args.scenario.exists():
            print(f"Error: Scenario not found: {args.scenario}")
            return 1

        seeds = list(range(args.seeds))
        print("=" * 60)
        print("SkillRL Dynamics Study")
        print("=" * 60)
        print(f"  Scenario: {args.scenario}")
        print(f"  Seeds: {len(seeds)}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Steps/epoch: {args.steps}")
        print(f"  Output: {output_dir}")
        print()

        snapshots_path = run_dynamics_study(
            args.scenario, seeds, args.epochs, args.steps, output_dir
        )

    # Generate plots
    print("\nGenerating plots...")
    from examples.plot_skillrl_dynamics import plot_all_from_json

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = plot_all_from_json(snapshots_path, plots_dir)
    for p in generated:
        print(f"  -> {p}")

    print(f"\nAll outputs in {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
