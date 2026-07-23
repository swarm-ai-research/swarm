"""Scenario generation from paper annotations.

Converts PaperAnnotation into SWARM scenario YAML configs
matching the baseline.yaml schema.
"""

from typing import Any

import yaml

from swarm.research.annotator import PaperAnnotation

# Default payoff parameters matching baseline.yaml
DEFAULT_PAYOFF = {
    "s_plus": 2.0,
    "s_minus": 1.0,
    "h": 2.0,
    "theta": 0.5,
    "rho_a": 0.0,
    "rho_b": 0.0,
    "w_rep": 1.0,
}

# Default rate limits
DEFAULT_RATE_LIMITS = {
    "posts_per_epoch": 10,
    "interactions_per_step": 5,
    "votes_per_epoch": 50,
    "tasks_per_epoch": 3,
}


class ScenarioGenerator:
    """Generates SWARM scenario configs from paper annotations."""

    def from_paper(self, annotation: PaperAnnotation) -> dict[str, Any]:
        """Produce a scenario dict matching baseline.yaml schema."""
        agents = self._map_agent_composition(annotation)
        governance = self._map_governance(annotation)
        simulation = self._map_simulation_params(annotation)
        success_criteria = self._map_success_criteria(annotation)
        payoff = self._map_payoff_params(annotation)

        scenario_id = f"agentxiv_{annotation.paper_id}" if annotation.paper_id else "agentxiv_generated"

        return {
            "scenario_id": scenario_id,
            "description": f"Auto-generated from paper: {annotation.title}",
            "motif": "paper_validation",
            "agents": agents,
            "governance": governance,
            "simulation": simulation,
            "rate_limits": dict(DEFAULT_RATE_LIMITS),
            "payoff": payoff,
            "success_criteria": success_criteria,
            "outputs": {
                "event_log": f"logs/{scenario_id}_events.jsonl",
                "metrics_csv": f"logs/{scenario_id}_metrics.csv",
            },
        }

    def from_paper_to_yaml(self, annotation: PaperAnnotation) -> str:
        """Produce a YAML string of the scenario config."""
        scenario = self.from_paper(annotation)
        return yaml.dump(scenario, default_flow_style=False, sort_keys=False)

    def _map_agent_composition(
        self, annotation: PaperAnnotation
    ) -> list[dict[str, Any]]:
        """Map failure modes to agent type mix."""
        failure_modes = annotation.risk_profile.failure_modes
        density = annotation.risk_profile.interaction_density

        # Base counts scaled by density
        density_scale = {"low": 1, "medium": 2, "high": 3}
        if density not in density_scale:
            raise ValueError(
                f"Unknown interaction_density {density!r}; "
                f"expected one of {sorted(density_scale)}"
            )
        scale = density_scale[density]

        honest_count = 3 * scale
        opportunistic_count = 0
        deceptive_count = 0

        if "collusion" in failure_modes:
            deceptive_count += scale
        if "deception" in failure_modes:
            deceptive_count += scale
        if "adverse_selection" in failure_modes:
            opportunistic_count += scale
        if "free_riding" in failure_modes:
            opportunistic_count += scale
        if "conflict" in failure_modes:
            opportunistic_count += max(1, scale // 2)
            deceptive_count += max(1, scale // 2)
        if "miscoordination" in failure_modes:
            opportunistic_count += scale

        # Ensure at least one honest agent
        if honest_count == 0:
            honest_count = 1

        agents: list[dict[str, Any]] = [{"type": "honest", "count": honest_count}]
        if opportunistic_count > 0:
            agents.append({"type": "opportunistic", "count": opportunistic_count})
        if deceptive_count > 0:
            agents.append({"type": "deceptive", "count": deceptive_count})

        return agents

    def _map_governance(self, annotation: PaperAnnotation) -> dict[str, Any]:
        """Map assumptions to governance configuration."""
        assumptions = annotation.risk_profile.assumptions
        failure_modes = annotation.risk_profile.failure_modes

        governance: dict[str, Any] = {
            "transaction_tax_rate": 0.0,
            "transaction_tax_split": 0.5,
            "reputation_decay_rate": 1.0,
            "bandwidth_cap": 10,
            "staking_enabled": False,
            "min_stake_to_participate": 0.0,
            "circuit_breaker_enabled": False,
            "audit_enabled": False,
        }

        # If paper assumes honest majority, use minimal governance
        if "assumes_honest_majority" in assumptions:
            return governance

        # Enable governance levers based on failure modes
        if "collusion" in failure_modes or "deception" in failure_modes:
            governance["audit_enabled"] = True
            governance["circuit_breaker_enabled"] = True

        if "adverse_selection" in failure_modes:
            governance["staking_enabled"] = True
            governance["min_stake_to_participate"] = 0.1

        if "free_riding" in failure_modes:
            governance["transaction_tax_rate"] = 0.05

        return governance

    def _map_simulation_params(self, annotation: PaperAnnotation) -> dict[str, Any]:
        """Map density to simulation parameters."""
        density = annotation.risk_profile.interaction_density

        params = {"low": (10, 5), "medium": (10, 10), "high": (20, 15)}
        epochs, steps = params.get(density, (10, 10))

        return {
            "n_epochs": epochs,
            "steps_per_epoch": steps,
            "seed": 42,
        }

    def _map_success_criteria(self, annotation: PaperAnnotation) -> dict[str, Any]:
        """Map testable claims to metric thresholds."""
        criteria: dict[str, Any] = {
            "min_epochs": 10,
            "min_agents": 5,
            "toxicity_threshold": 0.5,
        }

        for claim in annotation.claims:
            if not claim.testable or not claim.metric:
                continue
            if claim.metric == "toxicity_rate" and claim.expected == "negative":
                criteria["toxicity_threshold"] = 0.3
            elif claim.metric == "total_welfare" and claim.expected == "positive":
                criteria["min_epochs"] = max(criteria["min_epochs"], 15)

        return criteria

    def _map_payoff_params(self, annotation: PaperAnnotation) -> dict[str, Any]:
        """Return payoff parameters (defaults from baseline)."""
        return dict(DEFAULT_PAYOFF)
