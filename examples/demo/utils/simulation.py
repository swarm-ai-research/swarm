"""Simulation wrapper functions for the demo."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path (simulation.py is at examples/demo/utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swarm.analysis.aggregation import MetricsAggregator  # noqa: E402
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig  # noqa: E402
from swarm.governance.config import GovernanceConfig  # noqa: E402
from swarm.scenarios.loader import (  # noqa: E402
    build_orchestrator,
    load_scenario,
)

SCENARIOS_DIR = PROJECT_ROOT / "scenarios"

# Safety limits to prevent excessive resource consumption
MAX_AGENTS_PER_TYPE = 10
MAX_TOTAL_AGENTS = 40
MAX_EPOCHS = 50
MAX_STEPS_PER_EPOCH = 30


# Shared callback helpers to avoid duplication
def _make_on_interaction_callback(aggregator):
    """Create an on_interaction callback for the aggregator."""
    def on_interaction(interaction, payoff_a, payoff_b):
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, payoff_a)
        aggregator.record_payoff(interaction.counterparty, payoff_b)
    return on_interaction


def _make_on_epoch_callback(orchestrator, aggregator):
    """Create an on_epoch callback for the aggregator."""
    def on_epoch(epoch_metrics):
        agent_states = {
            aid: orchestrator.state.get_agent(aid) for aid in orchestrator._agents
        }
        aggregator.finalize_epoch(
            epoch=orchestrator.state.current_epoch - 1,
            agent_states=agent_states,
        )
    return on_epoch


def _requires_llm(data: dict) -> bool:
    """Return True if scenario YAML uses LLM-backed agents."""
    for agent_spec in data.get("agents", []):
        if agent_spec.get("type") == "llm":
            return True
    return False


def list_scenarios() -> List[Dict[str, str]]:
    """List available scenarios, excluding those that need LLM API keys."""
    import yaml

    scenarios = []
    for yaml_file in sorted(SCENARIOS_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        if _requires_llm(data):
            continue

        scenarios.append(
            {
                "id": data.get("scenario_id", yaml_file.stem),
                "description": data.get("description", ""),
                "path": str(yaml_file),
                "filename": yaml_file.name,
            }
        )
    return scenarios


def run_scenario(scenario_path: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Run a scenario and return structured results.

    Args:
        scenario_path: Path to YAML scenario file (must be under scenarios/)
        seed: Optional seed override

    Returns:
        Dict with epoch_metrics, agent_states, config info

    Raises:
        ValueError: If path is outside the scenarios directory
    """
    # Path traversal protection: resolve and verify within scenarios dir
    resolved = Path(scenario_path).resolve()
    scenarios_resolved = SCENARIOS_DIR.resolve()
    if not str(resolved).startswith(str(scenarios_resolved)):
        raise ValueError(
            f"Scenario path must be within {SCENARIOS_DIR}, got {scenario_path}"
        )

    # Reject scenarios that require LLM API keys
    import yaml

    with open(resolved) as f:
        raw = yaml.safe_load(f)
    if _requires_llm(raw):
        raise ValueError("LLM-backed scenarios are not supported in the demo")

    scenario = load_scenario(resolved)
    if seed is not None:
        scenario.orchestrator_config.seed = seed

    # Disable file logging in demo mode to prevent disk writes
    scenario.orchestrator_config.log_path = None
    scenario.orchestrator_config.log_events = False

    orchestrator = build_orchestrator(scenario)

    # Attach aggregator for rich metrics
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=scenario.scenario_id,
        n_epochs=scenario.orchestrator_config.n_epochs,
        steps_per_epoch=scenario.orchestrator_config.steps_per_epoch,
        n_agents=len(orchestrator._agents),
        seed=scenario.orchestrator_config.seed,
    )

    # Wire up interaction recording
    orchestrator._on_interaction_complete.append(_make_on_interaction_callback(aggregator))

    # Wire up epoch finalization
    orchestrator._on_epoch_end.append(_make_on_epoch_callback(orchestrator, aggregator))

    # Run
    epoch_metrics_list = orchestrator.run()
    history = aggregator.end_simulation()

    incoherence_series: List[float] = []
    if history and getattr(history, "epoch_snapshots", None):
        incoherence_series = [
            float(getattr(snapshot, "incoherence_index", 0.0))
            for snapshot in history.epoch_snapshots
        ]

    # Extract agent final states
    agent_states = []
    for agent_id, _agent in orchestrator._agents.items():
        state = orchestrator.state.get_agent(agent_id)
        if state:
            agent_states.append(
                {
                    "agent_id": agent_id,
                    "agent_type": state.agent_type.value,
                    "reputation": round(state.reputation, 2),
                    "resources": round(state.resources, 2),
                    "interactions": state.interactions_initiated
                    + state.interactions_received,
                    "total_payoff": round(state.total_payoff, 2),
                }
            )

    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "epoch_metrics": epoch_metrics_list,
        "agent_states": agent_states,
        "history": history,
        "incoherence_series": incoherence_series,
        "n_epochs": scenario.orchestrator_config.n_epochs,
        "n_agents": len(orchestrator._agents),
    }


def run_custom(
    n_honest: int = 3,
    n_opportunistic: int = 1,
    n_deceptive: int = 1,
    n_adversarial: int = 0,
    n_epochs: int = 20,
    steps_per_epoch: int = 10,
    tax_rate: float = 0.0,
    reputation_decay: float = 1.0,
    staking_enabled: bool = False,
    min_stake: float = 0.0,
    circuit_breaker_enabled: bool = False,
    freeze_threshold: float = 0.7,
    audit_enabled: bool = False,
    audit_probability: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a custom simulation with specified parameters.

    Returns:
        Dict with epoch_metrics, agent_states, config info

    Raises:
        ValueError: If parameters exceed safety limits
    """
    # Validate bounds to prevent resource exhaustion
    total_agents = n_honest + n_opportunistic + n_deceptive + n_adversarial
    if total_agents > MAX_TOTAL_AGENTS:
        raise ValueError(
            f"Total agents ({total_agents}) exceeds max ({MAX_TOTAL_AGENTS})"
        )
    if total_agents < 1:
        raise ValueError("Must have at least 1 agent")
    if n_epochs > MAX_EPOCHS:
        raise ValueError(f"n_epochs ({n_epochs}) exceeds max ({MAX_EPOCHS})")
    if steps_per_epoch > MAX_STEPS_PER_EPOCH:
        raise ValueError(
            f"steps_per_epoch ({steps_per_epoch}) exceeds max ({MAX_STEPS_PER_EPOCH})"
        )
    for name, val in [
        ("n_honest", n_honest),
        ("n_opportunistic", n_opportunistic),
        ("n_deceptive", n_deceptive),
        ("n_adversarial", n_adversarial),
    ]:
        if val > MAX_AGENTS_PER_TYPE:
            raise ValueError(f"{name} ({val}) exceeds max ({MAX_AGENTS_PER_TYPE})")

    from swarm.agents.adversarial import AdversarialAgent
    from swarm.agents.deceptive import DeceptiveAgent
    from swarm.agents.honest import HonestAgent
    from swarm.agents.opportunistic import OpportunisticAgent

    governance_config = GovernanceConfig(
        transaction_tax_rate=tax_rate,
        reputation_decay_rate=reputation_decay,
        staking_enabled=staking_enabled,
        min_stake_to_participate=min_stake,
        circuit_breaker_enabled=circuit_breaker_enabled,
        freeze_threshold_toxicity=freeze_threshold,
        audit_enabled=audit_enabled,
        audit_probability=audit_probability,
    )

    config = OrchestratorConfig(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        governance_config=governance_config,
        seed=seed,
    )

    orchestrator = Orchestrator(config)

    # Register agents
    agent_specs = [
        (HonestAgent, "honest", n_honest),
        (OpportunisticAgent, "opportunistic", n_opportunistic),
        (DeceptiveAgent, "deceptive", n_deceptive),
        (AdversarialAgent, "adversarial", n_adversarial),
    ]

    for agent_class, type_name, count in agent_specs:
        for i in range(count):
            orchestrator.register_agent(agent_class(agent_id=f"{type_name}_{i + 1}"))

    # Attach aggregator
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id="custom",
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        n_agents=len(orchestrator._agents),
        seed=seed,
    )

    orchestrator._on_interaction_complete.append(_make_on_interaction_callback(aggregator))

    orchestrator._on_epoch_end.append(_make_on_epoch_callback(orchestrator, aggregator))

    epoch_metrics_list = orchestrator.run()
    history = aggregator.end_simulation()

    incoherence_series: List[float] = []
    if history and getattr(history, "epoch_snapshots", None):
        incoherence_series = [
            float(getattr(snapshot, "incoherence_index", 0.0))
            for snapshot in history.epoch_snapshots
        ]

    # Extract agent final states
    agent_states = []
    for agent_id, _agent in orchestrator._agents.items():
        state = orchestrator.state.get_agent(agent_id)
        if state:
            agent_states.append(
                {
                    "agent_id": agent_id,
                    "agent_type": state.agent_type.value,
                    "reputation": round(state.reputation, 2),
                    "resources": round(state.resources, 2),
                    "interactions": state.interactions_initiated
                    + state.interactions_received,
                    "total_payoff": round(state.total_payoff, 2),
                }
            )

    return {
        "scenario_id": "custom",
        "description": "Custom simulation",
        "epoch_metrics": epoch_metrics_list,
        "agent_states": agent_states,
        "history": history,
        "incoherence_series": incoherence_series,
        "n_epochs": n_epochs,
        "n_agents": len(orchestrator._agents),
    }
