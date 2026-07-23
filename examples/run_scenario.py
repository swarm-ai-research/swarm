#!/usr/bin/env python
"""
Run a simulation from a YAML scenario file.

Usage:
    python examples/run_scenario.py scenarios/baseline.yaml
    python examples/run_scenario.py scenarios/status_game.yaml
    python examples/run_scenario.py scenarios/strict_governance.yaml
"""

import sys
import uuid
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.export import export_to_json
from swarm.scenarios import build_orchestrator, load_scenario


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_scenario.py <scenario.yaml>")
        print("\nAvailable scenarios:")
        for f in sorted(Path("scenarios").glob("*.yaml")):
            print(f"  - {f}")
        return 1

    scenario_path = Path(sys.argv[1])
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 60)
    print("Distributional AGI Safety Sandbox - Scenario Runner")
    print("=" * 60)
    print()

    # Load scenario
    print(f"Loading scenario: {scenario_path}")
    scenario = load_scenario(scenario_path)

    print(f"  ID: {scenario.scenario_id}")
    print(f"  Description: {scenario.description}")
    print(f"  Motif: {scenario.motif}")
    print()

    # Show governance config
    gov = scenario.orchestrator_config.governance_config
    if gov:
        print("Governance Configuration:")
        print(f"  Transaction tax: {gov.transaction_tax_rate * 100:.1f}%")
        print(f"  Reputation decay: {gov.reputation_decay_rate}")
        print(f"  Staking enabled: {gov.staking_enabled}", end="")
        if gov.staking_enabled:
            print(f" (min stake: {gov.min_stake_to_participate})")
        else:
            print()
        print(f"  Circuit breaker enabled: {gov.circuit_breaker_enabled}", end="")
        if gov.circuit_breaker_enabled:
            print(f" (threshold: {gov.freeze_threshold_toxicity})")
        else:
            print()
        print(f"  Audit enabled: {gov.audit_enabled}", end="")
        if gov.audit_enabled:
            print(f" (probability: {gov.audit_probability})")
        else:
            print()
        print()

    # Build orchestrator
    print("Building orchestrator...")
    orchestrator = build_orchestrator(scenario)

    # Wire MetricsAggregator for rich export data
    from swarm.analysis.aggregation import MetricsAggregator

    aggregator = MetricsAggregator()
    config = scenario.orchestrator_config
    aggregator.start_simulation(
        simulation_id=scenario.scenario_id,
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

    def _on_epoch_end(epoch_metrics):
        agent_states = orchestrator.state.agents
        frozen = orchestrator.state.frozen_agents
        quarantined = getattr(orchestrator.state, 'quarantined_agents', set())
        aggregator.finalize_epoch(
            epoch=epoch_metrics.epoch,
            agent_states=agent_states,
            frozen_agents=frozen,
            quarantined_agents=quarantined,
        )

    orchestrator.on_epoch_end(_on_epoch_end)

    # Show agents
    print(f"\nRegistered {len(orchestrator.get_all_agents())} agents:")
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        print(
            f"  - {agent.agent_id} ({agent.agent_type.value}): resources={state.resources:.0f}"
        )
    print()

    # Run simulation
    print(
        f"Running simulation: {config.n_epochs} epochs x {config.steps_per_epoch} steps"
    )
    print("-" * 60)

    metrics_history = orchestrator.run()

    print()
    print("Epoch-by-Epoch Metrics:")
    print("-" * 60)
    print(
        f"{'Epoch':<6} {'Interactions':<13} {'Accepted':<10} {'Toxicity':<10} {'Welfare':<10}"
    )
    print("-" * 60)

    for m in metrics_history:
        print(
            f"{m.epoch:<6} "
            f"{m.total_interactions:<13} "
            f"{m.accepted_interactions:<10} "
            f"{m.toxicity_rate:<10.4f} "
            f"{m.total_welfare:<10.2f}"
        )

    print("-" * 60)
    print()

    # Summary
    total_interactions = sum(m.total_interactions for m in metrics_history)
    total_accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0
    )
    total_welfare = sum(m.total_welfare for m in metrics_history)

    print("Summary Statistics:")
    print(f"  Total Interactions: {total_interactions}")
    print(f"  Accepted Interactions: {total_accepted}")
    print(f"  Average Toxicity Rate: {avg_toxicity:.4f}")
    print(f"  Total Welfare: {total_welfare:.2f}")
    print()

    # Final agent states
    print("Final Agent States:")
    print("-" * 60)
    frozen = orchestrator.state.frozen_agents
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        status = " [FROZEN]" if agent.agent_id in frozen else ""
        print(
            f"  {agent.agent_id}: "
            f"rep={state.reputation:.2f}, "
            f"res={state.resources:.2f}, "
            f"payoff={state.total_payoff:.2f}"
            f"{status}"
        )
    print()

    # Auto-export rich history
    history = aggregator.end_simulation()
    run_id = str(uuid.uuid4())[:8]
    export_dir = Path("runs") / f"{run_id}_{scenario.scenario_id}"
    export_path = export_dir / "history.json"
    export_to_json(history, export_path, include_events=True)
    print(f"Exported rich history: {export_path}")
    print(f"  Agent snapshots: {sum(len(v) for v in history.agent_snapshots.values())}")
    print(f"  Interaction events: {len(history.interactions)}")

    from swarm.analysis.run_plots import write_run_plots

    plot_paths = write_run_plots(
        metrics_history, export_dir, scenario_id=scenario.scenario_id,
    )
    for p in plot_paths:
        print(f"  Plot: {p}")
    print()

    # Check success criteria
    criteria = scenario.success_criteria
    if criteria:
        print("Success Criteria:")
        print("-" * 60)

        success = True

        if "min_epochs" in criteria:
            passed = len(metrics_history) >= criteria["min_epochs"]
            status = "[PASS]" if passed else "[FAIL]"
            print(
                f"  {status} Epochs: {len(metrics_history)} >= {criteria['min_epochs']}"
            )
            success = success and passed

        if "min_agents" in criteria:
            n_agents = len(orchestrator.get_all_agents())
            passed = n_agents >= criteria["min_agents"]
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {status} Agents: {n_agents} >= {criteria['min_agents']}")
            success = success and passed

        if "toxicity_threshold" in criteria:
            passed = avg_toxicity <= criteria["toxicity_threshold"]
            status = "[PASS]" if passed else "[FAIL]"
            print(
                f"  {status} Toxicity: {avg_toxicity:.4f} <= {criteria['toxicity_threshold']}"
            )
            success = success and passed

        print()
        if success:
            print("Result: ALL CRITERIA PASSED")
        else:
            print("Result: SOME CRITERIA FAILED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
