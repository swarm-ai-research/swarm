#!/usr/bin/env python
"""
Governance Sensitivity Sweep — tax rate x audit probability effect on M_eff.

Sweeps tax_rate in [0, 0.02, 0.05, 0.10, 0.15] x audit_probability in
[0, 0.05, 0.10, 0.20] using the misalignment_sweep population. Measures
M_eff reduction, toxicity, welfare, and agent survival per config. Prints
a grid table and exports results as JSON.

Usage:
    python examples/run_governance_sensitivity_sweep.py [scenarios/misalignment_sweep.yaml]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.metrics.misalignment import IssueSpace, MisalignmentModule
from swarm.models.agent import AgentType
from swarm.scenarios import build_orchestrator, load_scenario

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

TAX_RATES = [0.0, 0.02, 0.05, 0.10, 0.15]
AUDIT_PROBS = [0.0, 0.05, 0.10, 0.20]

# ---------------------------------------------------------------------------
# Preference profiles per agent type (same as run_misalignment_study.py)
# Issues: safety, efficiency, fairness
# ---------------------------------------------------------------------------

ISSUE_NAMES = ["safety", "efficiency", "fairness"]

PREFERENCE_MAP: dict[AgentType, dict] = {
    AgentType.HONEST: {
        "prefs": [+0.8, +0.3, +0.7],
        "salience": [0.4, 0.2, 0.4],
    },
    AgentType.OPPORTUNISTIC: {
        "prefs": [-0.2, +0.9, -0.3],
        "salience": [0.1, 0.7, 0.2],
    },
    AgentType.ADVERSARIAL: {
        "prefs": [-0.8, +0.5, -0.7],
        "salience": [0.3, 0.5, 0.2],
    },
    AgentType.DECEPTIVE: {
        "prefs": [-0.5, +0.7, -0.4],
        "salience": [0.2, 0.5, 0.3],
    },
    AgentType.CAUTIOUS: {
        "prefs": [+0.6, -0.1, +0.5],
        "salience": [0.5, 0.1, 0.4],
    },
}


def _build_graph(orchestrator) -> dict[str, list[str]]:
    """Build adjacency list from orchestrator network."""
    graph: dict[str, list[str]] = {}
    if orchestrator.network is None:
        all_ids = [a.agent_id for a in orchestrator.get_all_agents()]
        for aid in all_ids:
            graph[aid] = [x for x in all_ids if x != aid]
    else:
        for agent in orchestrator.get_all_agents():
            graph[agent.agent_id] = orchestrator.network.neighbors(agent.agent_id)
    return graph


def _run_single_config(
    base_scenario_path: Path,
    tax_rate: float,
    audit_prob: float,
    config_idx: int,
    total_configs: int,
) -> dict:
    """Run one (tax_rate, audit_probability) configuration and return results."""
    # Load a fresh scenario for each config
    scenario = load_scenario(base_scenario_path)
    config = scenario.orchestrator_config

    # Modify governance parameters
    gov = config.governance_config
    if gov is None:
        from swarm.governance.config import GovernanceConfig

        gov = GovernanceConfig()
        config.governance_config = gov

    # Use model_copy to update governance fields (Pydantic v2)
    config.governance_config = gov.model_copy(
        update={
            "transaction_tax_rate": tax_rate,
            "audit_enabled": audit_prob > 0,
            "audit_probability": audit_prob,
        }
    )

    gov_pressure = tax_rate + audit_prob

    print(
        f"  [{config_idx}/{total_configs}] "
        f"tax={tax_rate:.2f}, audit={audit_prob:.2f} "
        f"(pressure={gov_pressure:.3f})"
    )

    # Build orchestrator
    orchestrator = build_orchestrator(scenario)

    # Set up MisalignmentModule
    issue_space = IssueSpace(issues=ISSUE_NAMES)
    module = MisalignmentModule(issue_space=issue_space, gov_lambda=1.0)

    for agent in orchestrator.get_all_agents():
        profile = PREFERENCE_MAP.get(agent.agent_type)
        if profile is None:
            profile = {
                "prefs": [0.0] * len(ISSUE_NAMES),
                "salience": [1.0 / len(ISSUE_NAMES)] * len(ISSUE_NAMES),
            }
        module.register_agent(
            agent_id=agent.agent_id,
            prefs=profile["prefs"],
            salience=profile["salience"],
        )

    # Wire MetricsAggregator
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=f"{scenario.scenario_id}_tax{tax_rate}_audit{audit_prob}",
        n_epochs=config.n_epochs,
        steps_per_epoch=config.steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=config.seed,
    )

    def _on_interaction(interaction, initiator_payoff, counterparty_payoff):
        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, initiator_payoff)
        aggregator.record_payoff(interaction.counterparty, counterparty_payoff)
        aggregator.get_history().interactions.append(interaction)

    orchestrator.on_interaction_complete(_on_interaction)

    # Collect misalignment snapshots
    snapshots = []

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

        graph = _build_graph(orchestrator)
        snap = module.compute_snapshot(
            step=epoch_metrics.epoch,
            graph=graph,
            uniform_pressure=gov_pressure,
        )
        snapshots.append(snap)

    orchestrator.on_epoch_end(_on_epoch_end)

    # Run simulation
    metrics_history = orchestrator.run()

    # Collect results
    total_agents = len(orchestrator.get_all_agents())
    frozen_count = len(orchestrator.state.frozen_agents)
    surviving = total_agents - frozen_count

    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0.0
    )
    total_welfare = sum(m.total_welfare for m in metrics_history)

    final_m_pref = snapshots[-1].m_pref_global if snapshots else 0.0
    final_m_eff = snapshots[-1].m_eff_global if snapshots else 0.0
    gov_reduction = (
        (1.0 - final_m_eff / final_m_pref) if final_m_pref > 0 else 0.0
    )

    # End aggregator
    aggregator.end_simulation()

    return {
        "tax_rate": tax_rate,
        "audit_probability": audit_prob,
        "gov_pressure": gov_pressure,
        "m_pref": final_m_pref,
        "m_eff": final_m_eff,
        "gov_reduction_ratio": gov_reduction,
        "avg_toxicity": avg_toxicity,
        "total_welfare": total_welfare,
        "agents_total": total_agents,
        "agents_surviving": surviving,
        "agents_frozen": frozen_count,
        "n_epochs": len(metrics_history),
        "total_interactions": sum(
            m.total_interactions for m in metrics_history
        ),
        "snapshots": [s.to_dict() for s in snapshots],
    }


def main():
    scenario_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "scenarios/misalignment_sweep.yaml"
    )
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 78)
    print("Governance Sensitivity Sweep")
    print("tax_rate x audit_probability effect on M_eff")
    print("=" * 78)
    print()

    # Show sweep grid
    total_configs = len(TAX_RATES) * len(AUDIT_PROBS)
    print(f"Scenario: {scenario_path}")
    print(f"Tax rates: {TAX_RATES}")
    print(f"Audit probabilities: {AUDIT_PROBS}")
    print(f"Total configurations: {total_configs}")
    print()

    # Load once to show base info
    base_scenario = load_scenario(scenario_path)
    base_config = base_scenario.orchestrator_config
    seed = base_config.seed
    print(f"Base scenario: {base_scenario.scenario_id}")
    print(f"  Epochs: {base_config.n_epochs}, Steps/epoch: {base_config.steps_per_epoch}")
    print(f"  Seed: {seed}")
    print()

    # Run sweep
    print("Running sweep...")
    print("-" * 78)
    results = []
    config_idx = 0
    for tax in TAX_RATES:
        for audit in AUDIT_PROBS:
            config_idx += 1
            result = _run_single_config(
                scenario_path, tax, audit, config_idx, total_configs
            )
            results.append(result)
    print("-" * 78)
    print()

    # --- Grid tables ---
    # Define the column header label once for all grid tables
    tax_audit_label = "tax \\ audit"

    # --- Grid table: M_eff ---
    print("M_eff Grid (tax_rate x audit_probability):")
    print("-" * 78)
    # Header row
    header = f"{tax_audit_label:<12}"
    for audit in AUDIT_PROBS:
        header += f"  {audit:<10}"
    print(header)
    print("-" * 78)

    # Build lookup
    lookup = {(r["tax_rate"], r["audit_probability"]): r for r in results}

    for tax in TAX_RATES:
        row = f"{tax:<12.2f}"
        for audit in AUDIT_PROBS:
            r = lookup[(tax, audit)]
            row += f"  {r['m_eff']:<10.4f}"
        print(row)
    print()

    # --- Grid table: Governance Reduction Ratio ---
    print("Governance Reduction Ratio (1 - M_eff/M_pref):")
    print("-" * 78)
    header = f"{tax_audit_label:<12}"
    for audit in AUDIT_PROBS:
        header += f"  {audit:<10}"
    print(header)
    print("-" * 78)

    for tax in TAX_RATES:
        row = f"{tax:<12.2f}"
        for audit in AUDIT_PROBS:
            r = lookup[(tax, audit)]
            row += f"  {r['gov_reduction_ratio']:<10.1%}"
        print(row)
    print()

    # --- Grid table: Toxicity ---
    print("Avg Toxicity Grid:")
    print("-" * 78)
    header = f"{tax_audit_label:<12}"
    for audit in AUDIT_PROBS:
        header += f"  {audit:<10}"
    print(header)
    print("-" * 78)

    for tax in TAX_RATES:
        row = f"{tax:<12.2f}"
        for audit in AUDIT_PROBS:
            r = lookup[(tax, audit)]
            row += f"  {r['avg_toxicity']:<10.4f}"
        print(row)
    print()

    # --- Grid table: Welfare ---
    print("Total Welfare Grid:")
    print("-" * 78)
    header = f"{tax_audit_label:<12}"
    for audit in AUDIT_PROBS:
        header += f"  {audit:<10}"
    print(header)
    print("-" * 78)

    for tax in TAX_RATES:
        row = f"{tax:<12.2f}"
        for audit in AUDIT_PROBS:
            r = lookup[(tax, audit)]
            row += f"  {r['total_welfare']:<10.2f}"
        print(row)
    print()

    # --- Grid table: Agents Surviving ---
    print("Agents Surviving Grid:")
    print("-" * 78)
    header = f"{tax_audit_label:<12}"
    for audit in AUDIT_PROBS:
        header += f"  {audit:<10}"
    print(header)
    print("-" * 78)

    for tax in TAX_RATES:
        row = f"{tax:<12.2f}"
        for audit in AUDIT_PROBS:
            r = lookup[(tax, audit)]
            row += f"  {r['agents_surviving']:<10}"
        print(row)
    print()

    # --- Summary statistics ---
    print("Summary Statistics:")
    print("-" * 78)

    m_effs = [r["m_eff"] for r in results]
    reductions = [r["gov_reduction_ratio"] for r in results]
    toxicities = [r["avg_toxicity"] for r in results]
    welfares = [r["total_welfare"] for r in results]

    print(f"  M_eff range: [{min(m_effs):.4f}, {max(m_effs):.4f}]")
    print(f"  Governance reduction range: [{min(reductions):.1%}, {max(reductions):.1%}]")
    print(f"  Toxicity range: [{min(toxicities):.4f}, {max(toxicities):.4f}]")
    print(f"  Welfare range: [{min(welfares):.2f}, {max(welfares):.2f}]")
    print()

    # Best and worst configs
    best_reduction = max(results, key=lambda r: r["gov_reduction_ratio"])
    worst_reduction = min(results, key=lambda r: r["gov_reduction_ratio"])
    print(
        f"  Best M_eff reduction: tax={best_reduction['tax_rate']:.2f}, "
        f"audit={best_reduction['audit_probability']:.2f} "
        f"-> {best_reduction['gov_reduction_ratio']:.1%}"
    )
    print(
        f"  Worst M_eff reduction: tax={worst_reduction['tax_rate']:.2f}, "
        f"audit={worst_reduction['audit_probability']:.2f} "
        f"-> {worst_reduction['gov_reduction_ratio']:.1%}"
    )

    best_welfare = max(results, key=lambda r: r["total_welfare"])
    print(
        f"  Best welfare: tax={best_welfare['tax_rate']:.2f}, "
        f"audit={best_welfare['audit_probability']:.2f} "
        f"-> {best_welfare['total_welfare']:.2f}"
    )
    print()

    # --- Export ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_gov_sensitivity_seed{seed}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Strip per-epoch snapshots from summary export (keep them in detailed export)
    summary_results = []
    for r in results:
        summary = {k: v for k, v in r.items() if k != "snapshots"}
        summary_results.append(summary)

    # Summary JSON
    summary_path = export_dir / "governance_sensitivity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "scenario_id": base_scenario.scenario_id,
                "seed": seed,
                "tax_rates": TAX_RATES,
                "audit_probabilities": AUDIT_PROBS,
                "n_configs": total_configs,
                "results": summary_results,
            },
            f,
            indent=2,
        )

    # Detailed JSON (with per-epoch snapshots)
    detailed_path = export_dir / "governance_sensitivity_detailed.json"
    with open(detailed_path, "w") as f:
        json.dump(
            {
                "scenario_id": base_scenario.scenario_id,
                "seed": seed,
                "tax_rates": TAX_RATES,
                "audit_probabilities": AUDIT_PROBS,
                "n_configs": total_configs,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Exported summary: {summary_path}")
    print(f"Exported detailed: {detailed_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
