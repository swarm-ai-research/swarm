"""Delivery agent policies for the SimWorld Delivery domain.

Provides baseline and adversarial delivery policies that map to
SimWorld's persona-driven behaviors:
- Conscientious: reliable, on-time focused
- Open/Aggressive: risk-taking, profit-maximizing
- Cautious: conservative, consistent
- Opportunistic: exploits the system (overbids, gaming)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from swarm.domains.simworld_delivery.entities import (
    DeliveryAction,
    DeliveryActionType,
)


class DeliveryPolicy(ABC):
    """Base class for delivery agent policies."""

    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self._rng = random.Random(seed)

    @abstractmethod
    def decide(self, obs: dict) -> DeliveryAction:
        """Choose an action given the current observation."""


class ConscientiousPolicy(DeliveryPolicy):
    """Reliable delivery agent: bids conservatively, completes on time.

    Maps to SimWorld's conscientious persona — focuses on task completion
    with consistent, low-risk strategy.
    """

    def decide(self, obs: dict) -> DeliveryAction:
        current = obs.get("current_order")

        # If delivering, continue
        if current is not None:
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.DELIVER,
                order_id=current["order_id"],
            )

        # Bid on the most profitable available order (value/distance ratio)
        available = obs.get("available_orders", [])
        if not available:
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.WAIT,
            )

        # Pick order with best value-to-distance ratio
        best = max(
            available,
            key=lambda o: o["value"] / max(
                o["pickup_distance"] + o["distance"], 1.0,
            ),
        )

        # Conservative bid: slightly below order value
        bid = best["value"] * self._rng.uniform(0.6, 0.8)

        return DeliveryAction(
            agent_id=self.agent_id,
            action_type=DeliveryActionType.BID,
            order_id=best["order_id"],
            bid_amount=bid,
        )


class AggressivePolicy(DeliveryPolicy):
    """Aggressive profit-maximizer: takes risks, buys tools early.

    Maps to SimWorld's open/aggressive persona — explores opportunities
    but frequently overbids or misallocates resources.
    """

    def __init__(
        self, agent_id: str, seed: int | None = None,
        scooter_priority: bool = True,
    ) -> None:
        super().__init__(agent_id, seed)
        self._scooter_priority = scooter_priority
        self._bought_scooter = False

    def decide(self, obs: dict) -> DeliveryAction:
        current = obs.get("current_order")

        # If delivering, continue
        if current is not None:
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.DELIVER,
                order_id=current["order_id"],
            )

        # Buy scooter early (aggressive tool investment)
        if (self._scooter_priority
                and not obs.get("has_scooter", False)
                and not self._bought_scooter
                and obs.get("budget", 0) >= 50.0):
            self._bought_scooter = True
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.BUY_SCOOTER,
            )

        # Bid on highest-value order regardless of feasibility
        available = obs.get("available_orders", [])
        if not available:
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.WAIT,
            )

        best = max(available, key=lambda o: o["value"])

        # Aggressive bid: sometimes overbids (bids above order value)
        if self._rng.random() < 0.3:
            bid = best["value"] * self._rng.uniform(1.0, 1.5)
        else:
            bid = best["value"] * self._rng.uniform(0.5, 0.9)

        return DeliveryAction(
            agent_id=self.agent_id,
            action_type=DeliveryActionType.BID,
            order_id=best["order_id"],
            bid_amount=bid,
        )


class CautiousPolicy(DeliveryPolicy):
    """Conservative delivery agent: low risk, consistent returns.

    Maps to SimWorld's cautious persona — trades peak performance
    for consistency and stability.
    """

    def decide(self, obs: dict) -> DeliveryAction:
        current = obs.get("current_order")

        # If delivering, continue
        if current is not None:
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.DELIVER,
                order_id=current["order_id"],
            )

        available = obs.get("available_orders", [])
        if not available:
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.WAIT,
            )

        # Only bid on nearby, high-margin, feasible orders
        speed = obs.get("speed", 1.0)
        feasible = [
            o for o in available
            if (o["pickup_distance"] + o["distance"]) / max(speed, 0.1)
            < o["steps_remaining"] * 0.7
        ]

        if not feasible:
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.WAIT,
            )

        best = max(
            feasible,
            key=lambda o: o["value"] / max(
                o["pickup_distance"] + o["distance"], 1.0,
            ),
        )

        # Very conservative bid
        bid = best["value"] * self._rng.uniform(0.4, 0.6)

        return DeliveryAction(
            agent_id=self.agent_id,
            action_type=DeliveryActionType.BID,
            order_id=best["order_id"],
            bid_amount=bid,
        )


class OpportunisticPolicy(DeliveryPolicy):
    """Exploitative agent: games the system, manipulates bids.

    Represents the adversarial agent type in SWARM's framework —
    exploits information asymmetry and governance gaps.
    """

    def decide(self, obs: dict) -> DeliveryAction:
        current = obs.get("current_order")

        if current is not None:
            # Sometimes abandons carrying deliveries
            if current.get("carrying") and self._rng.random() < 0.1:
                return DeliveryAction(
                    agent_id=self.agent_id,
                    action_type=DeliveryActionType.WAIT,
                )
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.DELIVER,
                order_id=current["order_id"],
            )

        available = obs.get("available_orders", [])
        if not available:
            return DeliveryAction(
                agent_id=self.agent_id,
                action_type=DeliveryActionType.WAIT,
            )

        # Cherry-pick: only take high-value, short-distance orders
        cherry = sorted(
            available,
            key=lambda o: o["value"] / max(
                o["pickup_distance"] + o["distance"], 1.0,
            ),
            reverse=True,
        )

        best = cherry[0]

        # Undercut bid to win, but only for profitable orders
        bid = best["value"] * self._rng.uniform(0.2, 0.4)

        return DeliveryAction(
            agent_id=self.agent_id,
            action_type=DeliveryActionType.BID,
            order_id=best["order_id"],
            bid_amount=bid,
        )
