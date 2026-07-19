#!/usr/bin/env python
"""
Tierra ecology study -- governed vs ungoverned comparison across seeds.

Runs tierra.yaml and tierra_governed.yaml side-by-side across 5 seeds,
comparing genome diversity, parasitism fraction, speciation events,
resource inequality (Gini), and population size.

The governed scenario adds: circuit breaker, collusion detection, 5% tax,
reputation decay, and a diversity-preserving reaper.

Usage:
    # Run both scenarios across all seeds (default)
    python examples/run_tierra_study.py

    # Run a single scenario
    python examples/run_tierra_study.py scenarios/tierra.yaml

    # Run a single scenario with a specific seed
    python examples/run_tierra_study.py scenarios/tierra.yaml --seed 42
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.analysis.export import export_to_json
from swarm.metrics.tierra_metrics import (
    cooperation_fraction,
    genome_diversity,
    parasitism_fraction,
    resource_gini,
    speciation_count,
)
from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44, 45, 46]
DEFAULT_SCENARIOS = [
    Path("scenarios/tierra.yaml"),
    Path("scenarios/tierra_governed.yaml"),
]


def _get_tierra_handler(orchestrator):
    """Safely retrieve the TierraHandler from the orchestrator."""
    return getattr(orchestrator, "_tierra_handler", None)


def _collect_tierra_snapshot(orchestrator):
    """Collect a tierra metrics snapshot from the current orchestrator state.

    Returns a dict with genome_diversity, parasitism, cooperation,
    speciation, gini, and population.
    """
    handler = _get_tierra_handler(orchestrator)
    if handler is None:
        return {
            "genome_diversity": 0.0,
            "parasitism": 0.0,
            "cooperation": 0.0,
            "speciation": 0,
            "gini": 0.0,
            "population": 0,
        }

    state = orchestrator.state
    living = handler._living_tierra_agents(state)
    genomes = [
        handler._genome_registry[aid]
        for aid in living
        if aid in handler._genome_registry
    ]
    resources = [state.agents[aid].resources for aid in living]

    return {
        "genome_diversity": genome_diversity(genomes),
        "parasitism": parasitism_fraction(genomes),
        "cooperation": cooperation_fraction(genomes),
        "speciation": speciation_count(genomes),
        "gini": resource_gini(resources),
        "population": len(living),
    }


def run_single(scenario_path: Path, seed: int) -> dict:
    """Run a single scenario with a given seed.

    Returns a dict with scenario_id, seed, epoch_snapshots (list of
    per-epoch tierra metric dicts), and summary statistics.
    """
    scenario = load_scenario(scenario_path)
    config = scenario.orchestrator_config

    # Override seed
    config.seed = seed

    orchestrator = build_orchestrator(scenario)

    # --- Wire MetricsAggregator (standard pattern) ---
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=scenario.scenario_id,
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=seed,
    )

    def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, initiator_payoff)
        aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
        aggregator.get_history().interactions.append(interaction)

    orchestrator.on_interaction_complete(_on_interaction)

    # Collect tierra snapshots per epoch
    epoch_snapshots = []

    def _on_epoch_end(epoch_metrics):
        agent_states = orchestrator.state.agents
        frozen = orchestrator.state.frozen_agents
        quarantined = getattr(orchestrator.state, "quarantined_agents", set())
        aggregator.finalize_epoch(
            epoch=epoch_metrics.epoch,
            agent_states=agent_states,
            frozen_agents=frozen,
            quarantined_agents=quarantined,
        )

        snap = _collect_tierra_snapshot(orchestrator)
        snap["epoch"] = epoch_metrics.epoch
        epoch_snapshots.append(snap)

    orchestrator.on_epoch_end(_on_epoch_end)

    # --- Run simulation ---
    metrics_history = orchestrator.run()

    # --- Build summary ---
    if epoch_snapshots:
        final = epoch_snapshots[-1]
        # Averages over all epochs
        avg_diversity = (
            sum(s["genome_diversity"] for s in epoch_snapshots)
            / len(epoch_snapshots)
        )
        avg_parasitism = (
            sum(s["parasitism"] for s in epoch_snapshots)
            / len(epoch_snapshots)
        )
        avg_gini = (
            sum(s["gini"] for s in epoch_snapshots)
            / len(epoch_snapshots)
        )
        avg_population = (
            sum(s["population"] for s in epoch_snapshots)
            / len(epoch_snapshots)
        )
        max_speciation = max(s["speciation"] for s in epoch_snapshots)
    else:
        final = {
            "genome_diversity": 0.0,
            "parasitism": 0.0,
            "cooperation": 0.0,
            "speciation": 0,
            "gini": 0.0,
            "population": 0,
        }
        avg_diversity = 0.0
        avg_parasitism = 0.0
        avg_gini = 0.0
        avg_population = 0.0
        max_speciation = 0

    summary = {
        "final_diversity": final["genome_diversity"],
        "final_parasitism": final["parasitism"],
        "final_cooperation": final["cooperation"],
        "final_speciation": final["speciation"],
        "final_gini": final["gini"],
        "final_population": final["population"],
        "avg_diversity": avg_diversity,
        "avg_parasitism": avg_parasitism,
        "avg_gini": avg_gini,
        "avg_population": avg_population,
        "max_speciation": max_speciation,
    }

    # Standard metrics summary
    total_interactions = sum(m.total_interactions for m in metrics_history)
    total_accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0.0
    )
    total_welfare = sum(m.total_welfare for m in metrics_history)
    summary["total_interactions"] = total_interactions
    summary["total_accepted"] = total_accepted
    summary["avg_toxicity"] = avg_toxicity
    summary["total_welfare"] = total_welfare

    # Export run artifacts
    history = aggregator.end_simulation()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = (
        Path("runs")
        / f"{timestamp}_{scenario.scenario_id}_seed{seed}"
    )

    export_path = export_dir / "history.json"
    export_to_json(history, export_path, include_events=True)

    tierra_path = export_dir / "tierra_snapshots.json"
    tierra_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tierra_path, "w") as f:
        json.dump(
            {
                "scenario_id": scenario.scenario_id,
                "seed": seed,
                "epoch_snapshots": epoch_snapshots,
                "summary": summary,
            },
            f,
            indent=2,
        )

    return {
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "epoch_snapshots": epoch_snapshots,
        "summary": summary,
        "export_dir": str(export_dir),
    }


def _print_epoch_table(result: dict) -> None:
    """Print epoch-by-epoch tierra metrics table for a single run."""
    snaps = result["epoch_snapshots"]
    if not snaps:
        print("  (no epoch data)")
        return

    header = (
        f"{'Ep':<4} {'Pop':<5} {'Divers':<8} {'Parasit':<8} "
        f"{'Coop':<8} {'Species':<8} {'Gini':<8}"
    )
    print(header)
    print("-" * 52)
    for s in snaps:
        print(
            f"{s['epoch']:<4} "
            f"{s['population']:<5} "
            f"{s['genome_diversity']:<8.4f} "
            f"{s['parasitism']:<8.4f} "
            f"{s['cooperation']:<8.4f} "
            f"{s['speciation']:<8} "
            f"{s['gini']:<8.4f}"
        )


def _print_comparison_table(all_results: list[dict]) -> None:
    """Print governed vs ungoverned comparison across seeds."""
    # Group by scenario
    by_scenario: dict[str, list[dict]] = {}
    for r in all_results:
        sid = r["scenario_id"]
        by_scenario.setdefault(sid, []).append(r)

    scenario_ids = sorted(by_scenario.keys())
    if len(scenario_ids) < 2:
        print("(Only one scenario — skipping comparison table)")
        return

    print("=" * 90)
    print("COMPARISON: Governed vs Ungoverned")
    print("=" * 90)
    print()

    # Per-seed comparison
    metrics_keys = [
        ("final_diversity", "Diversity"),
        ("final_parasitism", "Parasitism"),
        ("final_cooperation", "Cooperation"),
        ("final_speciation", "Speciation"),
        ("final_gini", "Gini"),
        ("final_population", "Population"),
        ("avg_toxicity", "Avg Toxicity"),
        ("total_welfare", "Welfare"),
    ]

    # Print per-metric rows
    print(f"{'Metric':<14} {'Scenario':<20}", end="")
    seeds = sorted({r["seed"] for r in all_results})
    for s in seeds:
        print(f" seed{s:>3}", end="")
    print(f" {'mean':>8}")
    print("-" * (34 + 8 * len(seeds) + 9))

    for key, label in metrics_keys:
        for sid in scenario_ids:
            runs = {r["seed"]: r for r in by_scenario[sid]}
            values = []
            print(f"{label:<14} {sid:<20}", end="")
            for s in seeds:
                val = runs[s]["summary"][key] if s in runs else 0.0
                values.append(val)
                if isinstance(val, int):
                    print(f" {val:>7}", end="")
                else:
                    print(f" {val:>7.4f}", end="")
            mean_val = sum(values) / len(values) if values else 0.0
            if isinstance(values[0] if values else 0, int):
                print(f" {mean_val:>8.1f}")
            else:
                print(f" {mean_val:>8.4f}")
        print()

    # Delta row (governed - ungoverned)
    if len(scenario_ids) == 2:
        print()
        print("Deltas (governed - ungoverned):")
        print("-" * (14 + 8 * len(seeds) + 9))
        sid_a, sid_b = scenario_ids  # alphabetical: tierra, tierra_governed
        runs_a = {r["seed"]: r for r in by_scenario[sid_a]}
        runs_b = {r["seed"]: r for r in by_scenario[sid_b]}
        for key, label in metrics_keys:
            deltas = []
            print(f"  {label:<12}", end="")
            for s in seeds:
                va = runs_a.get(s, {}).get("summary", {}).get(key, 0.0)
                vb = runs_b.get(s, {}).get("summary", {}).get(key, 0.0)
                d = vb - va
                deltas.append(d)
                if isinstance(va, int) and isinstance(vb, int):
                    print(f" {d:>+7}", end="")
                else:
                    print(f" {d:>+7.4f}", end="")
            mean_d = sum(deltas) / len(deltas) if deltas else 0.0
            print(f" {mean_d:>+8.4f}")
        print()


def main():
    # Parse args
    scenario_paths = []
    seeds = list(SEEDS)
    explicit_seed = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--seed" and i + 1 < len(args):
            explicit_seed = int(args[i + 1])
            seeds = [explicit_seed]
            i += 2
        elif args[i].endswith(".yaml") or args[i].endswith(".yml"):
            scenario_paths.append(Path(args[i]))
            i += 1
        else:
            print(f"Unknown argument: {args[i]}")
            return 1

    if not scenario_paths:
        scenario_paths = list(DEFAULT_SCENARIOS)

    # Validate
    for sp in scenario_paths:
        if not sp.exists():
            print(f"Error: Scenario file not found: {sp}")
            return 1

    print("=" * 78)
    print("Tierra Ecology Study -- Governed vs Ungoverned Comparison")
    print("=" * 78)
    print()
    print(f"Scenarios: {', '.join(str(p) for p in scenario_paths)}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(scenario_paths) * len(seeds)}")
    print()

    # --- Run all combinations ---
    all_results = []
    run_num = 0
    total_runs = len(scenario_paths) * len(seeds)

    for sp in scenario_paths:
        for seed in seeds:
            run_num += 1
            print("-" * 78)
            print(f"[{run_num}/{total_runs}] {sp.stem} seed={seed}")
            print("-" * 78)

            result = run_single(sp, seed)
            all_results.append(result)

            # Print epoch table for this run
            _print_epoch_table(result)
            print()

            s = result["summary"]
            print(f"  Final: pop={s['final_population']}, "
                  f"diversity={s['final_diversity']:.4f}, "
                  f"parasitism={s['final_parasitism']:.4f}, "
                  f"speciation={s['final_speciation']}, "
                  f"gini={s['final_gini']:.4f}")
            print(f"  Exported: {result['export_dir']}")
            print()

    # --- Comparison table ---
    print()
    _print_comparison_table(all_results)

    # --- Export combined results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_dir = Path("runs") / f"{timestamp}_tierra_study"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / "combined_results.json"

    # Strip epoch_snapshots for the combined summary (keep it concise)
    export_data = {
        "study": "tierra_ecology_governed_vs_ungoverned",
        "scenarios": [str(p) for p in scenario_paths],
        "seeds": seeds,
        "results": [
            {
                "scenario_id": r["scenario_id"],
                "seed": r["seed"],
                "summary": r["summary"],
                "export_dir": r["export_dir"],
            }
            for r in all_results
        ],
    }
    with open(combined_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Combined results exported: {combined_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
