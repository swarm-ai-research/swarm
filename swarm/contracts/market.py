"""Contract market: pre-play signing stage and interaction routing.

The ContractMarket sits between the orchestrator and the agent-to-agent
interaction layer. Each round (or epoch):
1. Governance publishes available contracts
2. Each agent chooses Sign(contract) or Refuse (-> default market)
3. Interactions are routed through the chosen contract's protocol
4. Outcomes are logged for separation/welfare/safety metrics

Agent decision logic uses a utility-based model:
- Good agents sign if expected_counterpart_quality * surplus_bonus > signing_cost
- Opportunistic agents sign if profitable (positive expected payoff)
- Malicious agents sign only if infiltration is cheaper than default pool attacks

The decision model uses the agent's beliefs about pool composition and
their own type-dependent cost/benefit calculation.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, field_validator

from swarm.contracts.contract import (
    Contract,
    ContractDecision,
    ContractType,
    DefaultMarket,
    _validated_copy,
)
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import SoftInteraction


class ContractMarketConfig(BaseModel):
    """Configuration for the contract market."""

    # Whether agents can switch contracts between epochs
    allow_switching: bool = True

    # Switching friction: cost multiplier when changing contracts
    switching_cost_multiplier: float = 0.5

    # Belief update rate: how fast agents update pool quality beliefs
    belief_update_rate: float = 0.3

    # Agent type -> contract preference weights
    # Higher weight = more likely to prefer that contract type
    honest_truthful_preference: float = 0.8
    honest_fair_preference: float = 0.6
    opportunistic_truthful_preference: float = 0.3
    opportunistic_fair_preference: float = 0.4
    adversarial_truthful_preference: float = 0.1
    adversarial_fair_preference: float = 0.15

    @field_validator(
        "switching_cost_multiplier",
        "belief_update_rate",
        "honest_truthful_preference",
        "honest_fair_preference",
        "opportunistic_truthful_preference",
        "opportunistic_fair_preference",
        "adversarial_truthful_preference",
        "adversarial_fair_preference",
    )
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v


class ContractMarket:
    """Orchestrates the pre-play contract signing stage and routes interactions.

    Each epoch:
    1. run_signing_stage() - agents choose contracts based on beliefs
    2. route_interaction() - called per-interaction to apply contract protocol
       and record outcomes for metric computation
    """

    def __init__(
        self,
        contracts: Optional[List[Contract]] = None,
        config: Optional[ContractMarketConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the contract market.

        Args:
            contracts: Available contracts (default: all three types).
            config: Market configuration.
            seed: Random seed for reproducible agent decisions.
        """
        self.config = config or ContractMarketConfig()
        self._rng = random.Random(seed)

        # Set up available contracts
        if contracts is None:
            from swarm.contracts.contract import (
                FairDivisionContract,
                TruthfulAuctionContract,
            )

            contracts = [
                TruthfulAuctionContract(),
                FairDivisionContract(),
            ]

        self._default_market = DefaultMarket()
        self._contracts: Dict[str, Contract] = {c.name: c for c in contracts}
        self._contracts[self._default_market.name] = self._default_market

        # Agent -> contract mapping (current epoch)
        self._memberships: Dict[str, str] = {}

        # History of all signing decisions
        self._decision_history: List[ContractDecision] = []

        # Per-contract interaction outcomes for metric computation
        self._contract_interactions: Dict[str, List[SoftInteraction]] = {
            name: [] for name in self._contracts
        }

        # Agent beliefs about pool quality (updated each epoch)
        self._pool_quality_beliefs: Dict[str, float] = dict.fromkeys(
            self._contracts, 0.5
        )

        # Track which agents are in which pool for routing
        self._pool_agents: Dict[str, List[str]] = {
            name: [] for name in self._contracts
        }

        # Dedup: prevent double-recording the same interaction
        self._seen_interaction_ids: Set[str] = set()

    @property
    def contracts(self) -> Dict[str, Contract]:
        """Available contracts (read-only view)."""
        return dict(self._contracts)

    @property
    def memberships(self) -> Dict[str, str]:
        """Current agent->contract membership mapping (read-only view)."""
        return dict(self._memberships)

    @property
    def decision_history(self) -> List[ContractDecision]:
        """Full history of signing decisions."""
        return list(self._decision_history)

    def run_signing_stage(
        self,
        agents: List[AgentState],
        epoch: int,
    ) -> Dict[str, str]:
        """Run the pre-play contract signing stage.

        Each agent evaluates available contracts and chooses one based on
        their type-dependent utility calculation.

        Args:
            agents: All agents participating this epoch.
            epoch: Current epoch number.

        Returns:
            Mapping of agent_id -> contract_name.
        """
        new_memberships: Dict[str, str] = {}

        for agent in agents:
            contract_name, cost_paid, reason = self._agent_choose_contract(
                agent, epoch
            )
            new_memberships[agent.agent_id] = contract_name

            decision = ContractDecision(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                contract_chosen=self._contracts[contract_name].contract_type,
                signing_cost_paid=cost_paid,
                epoch=epoch,
                reason=reason,
            )
            self._decision_history.append(decision)

        # Update memberships and pool tracking
        self._memberships = new_memberships
        self._pool_agents = {name: [] for name in self._contracts}
        for agent_id, contract_name in new_memberships.items():
            self._pool_agents[contract_name].append(agent_id)

        return new_memberships

    def route_interaction(
        self,
        interaction: SoftInteraction,
    ) -> SoftInteraction:
        """Route an interaction through the appropriate contract protocol.

        Routing logic (agents default to default_market if not in membership):
        - If both agents are in the same contract -> use that contract
        - If agents are in different contracts or either agent lacks membership
          -> use default market (conservative: don't grant protections the other
          party didn't sign up for)

        Args:
            interaction: The raw interaction.

        Returns:
            Modified interaction with contract protocol applied.
        """
        initiator_contract = self._memberships.get(
            interaction.initiator, self._default_market.name
        )
        counterparty_contract = self._memberships.get(
            interaction.counterparty, self._default_market.name
        )

        # Determine which contract governs this interaction
        if initiator_contract == counterparty_contract:
            contract_name = initiator_contract
        else:
            # Cross-pool interaction: use default market (no protections
            # unless both parties opted in)
            contract_name = self._default_market.name

        contract = self._contracts[contract_name]
        modified = contract.execute(interaction)

        # Check for audits — penalize both parties symmetrically
        if contract.should_audit(modified, self._rng):
            penalty = contract.penalize(
                interaction.initiator, interaction.p
            )
            if penalty > 0:
                modified = _validated_copy(
                    modified,
                    {
                        "c_a": modified.c_a + penalty,
                        "c_b": modified.c_b + penalty,
                        "metadata": {
                            **modified.metadata,
                            "audit_triggered": True,
                            "audit_penalty": penalty,
                        },
                    },
                )

        # Record for metrics (dedup by interaction_id)
        if modified.interaction_id not in self._seen_interaction_ids:
            self._seen_interaction_ids.add(modified.interaction_id)
            self._contract_interactions[contract_name].append(modified)

        return modified

    def update_beliefs(self) -> Dict[str, float]:
        """Update pool quality beliefs based on observed outcomes.

        Returns:
            Updated pool quality beliefs (contract_name -> avg p).
        """
        for name, interactions in self._contract_interactions.items():
            if interactions:
                observed_quality = sum(i.p for i in interactions) / len(
                    interactions
                )
                # Exponential moving average
                old = self._pool_quality_beliefs.get(name, 0.5)
                alpha = self.config.belief_update_rate
                self._pool_quality_beliefs[name] = (
                    alpha * observed_quality + (1 - alpha) * old
                )

        return dict(self._pool_quality_beliefs)

    def reset_epoch(self) -> None:
        """Reset per-epoch state (call at epoch start)."""
        self._contract_interactions = {
            name: [] for name in self._contracts
        }
        self._seen_interaction_ids = set()

    def get_pool_composition(self) -> Dict[str, Dict[str, int]]:
        """Get the type composition of each contract pool.

        Returns:
            {contract_name: {agent_type: count}}.
        """
        composition: Dict[str, Dict[str, int]] = {
            name: {} for name in self._contracts
        }

        # Use current memberships + agent type from last decisions
        agent_types: Dict[str, AgentType] = {}
        for decision in self._decision_history:
            agent_types[decision.agent_id] = decision.agent_type

        for agent_id, contract_name in self._memberships.items():
            atype = agent_types.get(agent_id, AgentType.HONEST)
            type_name = atype.value
            composition[contract_name][type_name] = (
                composition[contract_name].get(type_name, 0) + 1
            )

        return composition

    def get_contract_interactions(self) -> Dict[str, List[SoftInteraction]]:
        """Get interactions grouped by contract."""
        return {
            name: list(interactions)
            for name, interactions in self._contract_interactions.items()
        }

    def _agent_choose_contract(
        self,
        agent: AgentState,
        epoch: int,
    ) -> Tuple[str, float, str]:
        """Agent decision logic for contract selection.

        Uses a utility-based model where the agent evaluates:
        utility(contract) = preference_weight * pool_quality_belief - signing_cost

        Returns:
            (contract_name, cost_paid, reason_string)
        """
        effective_type = agent.true_type or agent.agent_type

        best_contract = self._default_market.name
        best_utility = 0.0  # Default market has utility 0 (baseline)
        best_cost = 0.0
        reason = "default_fallback"

        # If switching is disabled and the agent already has a membership,
        # lock them into their current contract.
        prev_contract = self._memberships.get(agent.agent_id)
        if not self.config.allow_switching and prev_contract is not None:
            return prev_contract, 0.0, "locked_in (switching disabled)"

        for name, contract in self._contracts.items():
            if name == self._default_market.name:
                continue

            cost = contract.signing_cost(agent)

            # Can the agent afford it?
            if cost > agent.resources:
                continue

            # Type-dependent preference weight
            preference = self._get_preference_weight(
                effective_type, contract.contract_type
            )

            # Pool quality belief
            quality_belief = self._pool_quality_beliefs.get(name, 0.5)

            # Switching cost if agent was in a different contract last epoch
            switching_penalty = 0.0
            if prev_contract is not None and prev_contract != name:
                switching_penalty = cost * self.config.switching_cost_multiplier

            # Utility calculation
            # Good agents value quality highly; adversarial agents see less value
            utility = (
                preference * quality_belief * agent.resources
                - cost
                - switching_penalty
            )

            # Add noise for non-determinism
            utility += self._rng.gauss(0, 0.1 * abs(utility) + 0.01)

            if utility > best_utility:
                best_utility = utility
                best_contract = name
                best_cost = cost
                reason = (
                    f"utility={utility:.3f} "
                    f"(quality_belief={quality_belief:.2f}, "
                    f"cost={cost:.2f})"
                )

        return best_contract, best_cost, reason

    def _get_preference_weight(
        self,
        agent_type: AgentType,
        contract_type: ContractType,
    ) -> float:
        """Get type-dependent preference weight for a contract.

        Higher weights mean the agent type finds more value in the contract.
        """
        if agent_type in (AgentType.HONEST,):
            if contract_type == ContractType.TRUTHFUL_AUCTION:
                return self.config.honest_truthful_preference
            elif contract_type == ContractType.FAIR_DIVISION:
                return self.config.honest_fair_preference
        elif agent_type == AgentType.OPPORTUNISTIC:
            if contract_type == ContractType.TRUTHFUL_AUCTION:
                return self.config.opportunistic_truthful_preference
            elif contract_type == ContractType.FAIR_DIVISION:
                return self.config.opportunistic_fair_preference
        elif agent_type in (
            AgentType.DECEPTIVE,
            AgentType.ADVERSARIAL,
        ):
            if contract_type == ContractType.TRUTHFUL_AUCTION:
                return self.config.adversarial_truthful_preference
            elif contract_type == ContractType.FAIR_DIVISION:
                return self.config.adversarial_fair_preference

        # Default for unknown types (RLM, CREWAI, etc.)
        return 0.3
