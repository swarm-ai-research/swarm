"""Dworkin-style auction mechanism for fair resource allocation.

Implements auction-based resource allocation inspired by Ronald Dworkin's
approach to distributive justice. Agents start with equal token endowments
and bid on resource bundles. Allocations are verified for envy-freeness:
no agent would prefer another agent's allocation at clearing prices.

Reference: "Virtual Agent Economies" (arXiv 2509.10147) - Tomasev et al.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from swarm.env.stats import gini


@dataclass
class AuctionConfig:
    """Configuration for the Dworkin auction mechanism."""

    initial_endowment: float = 100.0
    max_rounds: int = 50
    price_adjustment_rate: float = 0.1
    convergence_tolerance: float = 0.01
    envy_tolerance: float = 0.05

    def validate(self) -> None:
        """Validate configuration values."""
        if self.initial_endowment <= 0:
            raise ValueError("initial_endowment must be positive")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.price_adjustment_rate <= 0 or self.price_adjustment_rate > 1:
            raise ValueError("price_adjustment_rate must be in (0, 1]")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive")
        if self.envy_tolerance < 0:
            raise ValueError("envy_tolerance must be non-negative")


@dataclass
class ResourceBundle:
    """A bundle of resources available for auction."""

    bundle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resources: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize bundle."""
        return {
            "bundle_id": self.bundle_id,
            "resources": dict(self.resources),
        }


@dataclass
class AuctionBid:
    """An agent's bid expressing resource preferences."""

    agent_id: str = ""
    valuations: Dict[str, float] = field(default_factory=dict)
    budget: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize bid."""
        return {
            "agent_id": self.agent_id,
            "valuations": dict(self.valuations),
            "budget": self.budget,
        }


@dataclass
class AuctionAllocation:
    """Result of an allocation for a single agent."""

    agent_id: str = ""
    resources: Dict[str, float] = field(default_factory=dict)
    price_paid: float = 0.0
    utility: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize allocation."""
        return {
            "agent_id": self.agent_id,
            "resources": dict(self.resources),
            "price_paid": self.price_paid,
            "utility": self.utility,
        }


@dataclass
class EnvyViolation:
    """Records an envy violation between two agents."""

    envious_agent: str = ""
    envied_agent: str = ""
    utility_gap: float = 0.0


@dataclass
class AuctionResult:
    """Complete result of an auction round."""

    allocations: Dict[str, AuctionAllocation] = field(default_factory=dict)
    clearing_prices: Dict[str, float] = field(default_factory=dict)
    rounds_to_converge: int = 0
    is_envy_free: bool = False
    envy_violations: List[EnvyViolation] = field(default_factory=list)
    total_utility: float = 0.0
    converged: bool = False

    def to_dict(self) -> Dict:
        """Serialize result."""
        return {
            "allocations": {k: v.to_dict() for k, v in self.allocations.items()},
            "clearing_prices": dict(self.clearing_prices),
            "rounds_to_converge": self.rounds_to_converge,
            "is_envy_free": self.is_envy_free,
            "envy_violation_count": len(self.envy_violations),
            "total_utility": self.total_utility,
            "converged": self.converged,
        }


class DworkinAuction:
    """
    Implements Dworkin's auction with equal endowments and envy-freeness test.

    The mechanism works via tatonnement (price adjustment):
    1. Start with equal prices for all resources
    2. Each agent demands resources that maximize their utility given budget
    3. If demand exceeds supply, raise price; if supply exceeds demand, lower price
    4. Repeat until market clears (demand ~ supply for all resources)
    5. Verify envy-freeness: no agent prefers another's bundle at clearing prices

    Connection to soft labels: agent effective endowments can be modulated by
    reputation (derived from average p history), linking the allocation
    mechanism to the quality signal pipeline.
    """

    def __init__(self, config: Optional[AuctionConfig] = None) -> None:
        """Initialize the auction engine."""
        self.config = config or AuctionConfig()
        self.config.validate()

    def run_auction(
        self,
        bids: Dict[str, AuctionBid],
        available_resources: Dict[str, float],
    ) -> AuctionResult:
        """
        Run a Dworkin-style auction.

        Args:
            bids: Agent ID -> AuctionBid with valuations and budget
            available_resources: Resource name -> total quantity available

        Returns:
            AuctionResult with allocations, prices, and envy-freeness check
        """
        if not bids or not available_resources:
            return AuctionResult(converged=True, is_envy_free=True)

        resource_names = list(available_resources.keys())

        # Initialize prices: equal for all resources
        prices = dict.fromkeys(resource_names, 1.0)

        converged = False
        rounds = 0

        for round_num in range(self.config.max_rounds):
            rounds = round_num + 1

            # Compute demand for each agent given current prices
            allocations = self._compute_optimal_demands(
                bids, prices, available_resources
            )

            # Check market clearing
            total_demand = dict.fromkeys(resource_names, 0.0)
            for alloc in allocations.values():
                for r, qty in alloc.resources.items():
                    total_demand[r] = total_demand.get(r, 0.0) + qty

            # Adjust prices via tatonnement
            max_excess = 0.0
            for r in resource_names:
                excess = total_demand.get(r, 0.0) - available_resources[r]
                max_excess = max(max_excess, abs(excess))

                # Price adjustment proportional to excess demand
                adjustment = self.config.price_adjustment_rate * (
                    excess / max(available_resources[r], 1e-10)
                )
                prices[r] = max(0.01, prices[r] * (1 + adjustment))

            if max_excess < self.config.convergence_tolerance:
                converged = True
                break

        # Final allocation with clearing prices
        allocations = self._compute_optimal_demands(bids, prices, available_resources)

        # Normalize allocations so total demand doesn't exceed supply
        allocations = self._normalize_allocations(allocations, available_resources)

        # Compute utilities and prices paid
        for agent_id, alloc in allocations.items():
            bid = bids[agent_id]
            alloc.utility = sum(
                bid.valuations.get(r, 0.0) * qty for r, qty in alloc.resources.items()
            )
            alloc.price_paid = sum(
                prices.get(r, 0.0) * qty for r, qty in alloc.resources.items()
            )

        # Check envy-freeness
        is_envy_free, violations = self._check_envy_free(allocations, bids, prices)

        total_utility = sum(a.utility for a in allocations.values())

        return AuctionResult(
            allocations=allocations,
            clearing_prices=prices,
            rounds_to_converge=rounds,
            is_envy_free=is_envy_free,
            envy_violations=violations,
            total_utility=total_utility,
            converged=converged,
        )

    def _compute_optimal_demands(
        self,
        bids: Dict[str, AuctionBid],
        prices: Dict[str, float],
        available_resources: Dict[str, float],
    ) -> Dict[str, AuctionAllocation]:
        """Compute optimal demand for each agent given prices."""
        allocations = {}

        for agent_id, bid in bids.items():
            # For each resource, compute value-per-price ratio
            # Agent buys resources with highest ratio first
            ratios = []
            for r, val in bid.valuations.items():
                if r in prices and prices[r] > 0:
                    ratios.append((val / prices[r], r, val))

            # Sort by bang-per-buck (descending)
            ratios.sort(reverse=True)

            resources_demanded: Dict[str, float] = {}
            remaining_budget = bid.budget

            for _ratio, r, _val in ratios:
                if remaining_budget <= 0:
                    break
                # How much can we afford?
                max_affordable = remaining_budget / prices[r]
                # Don't demand more than available
                max_available = available_resources.get(r, 0.0)
                qty = min(max_affordable, max_available)
                if qty > 0:
                    resources_demanded[r] = qty
                    remaining_budget -= qty * prices[r]

            allocations[agent_id] = AuctionAllocation(
                agent_id=agent_id,
                resources=resources_demanded,
            )

        return allocations

    def _normalize_allocations(
        self,
        allocations: Dict[str, AuctionAllocation],
        available_resources: Dict[str, float],
    ) -> Dict[str, AuctionAllocation]:
        """Scale down allocations so total demand <= supply for each resource."""
        # Compute total demand per resource
        total_demand: Dict[str, float] = {}
        for alloc in allocations.values():
            for r, qty in alloc.resources.items():
                total_demand[r] = total_demand.get(r, 0.0) + qty

        # Scale down if over-demanded
        for r, demand in total_demand.items():
            supply = available_resources.get(r, 0.0)
            if demand > supply and demand > 0:
                scale = supply / demand
                for alloc in allocations.values():
                    if r in alloc.resources:
                        alloc.resources[r] *= scale

        return allocations

    def _check_envy_free(
        self,
        allocations: Dict[str, AuctionAllocation],
        bids: Dict[str, AuctionBid],
        prices: Dict[str, float],
    ) -> Tuple[bool, List[EnvyViolation]]:
        """
        Check envy-freeness: no agent prefers another's bundle.

        An agent i envies agent j if i's utility from j's bundle (at prices
        i can afford) exceeds i's utility from their own bundle.
        """
        violations = []
        agent_ids = list(allocations.keys())

        for i_id in agent_ids:
            my_alloc = allocations[i_id]
            my_bid = bids[i_id]
            my_utility = sum(
                my_bid.valuations.get(r, 0.0) * qty
                for r, qty in my_alloc.resources.items()
            )

            for j_id in agent_ids:
                if i_id == j_id:
                    continue

                other_alloc = allocations[j_id]
                # Can I afford other's bundle?
                other_cost = sum(
                    prices.get(r, 0.0) * qty for r, qty in other_alloc.resources.items()
                )

                if other_cost > my_bid.budget + self.config.envy_tolerance:
                    continue  # Can't afford it, no envy

                # What would I value other's bundle at?
                other_utility_for_me = sum(
                    my_bid.valuations.get(r, 0.0) * qty
                    for r, qty in other_alloc.resources.items()
                )

                gap = other_utility_for_me - my_utility
                if gap > self.config.envy_tolerance:
                    violations.append(
                        EnvyViolation(
                            envious_agent=i_id,
                            envied_agent=j_id,
                            utility_gap=gap,
                        )
                    )

        return len(violations) == 0, violations

    def compute_gini_coefficient(
        self, allocations: Dict[str, AuctionAllocation]
    ) -> float:
        """
        Compute Gini coefficient of allocation utilities.

        0 = perfect equality, 1 = maximum inequality.
        """
        return gini(a.utility for a in allocations.values())
