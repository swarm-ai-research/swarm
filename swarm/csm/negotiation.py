"""Module B: Negotiation with compute budgets.

Bilateral bargaining with time/compute constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from swarm.csm.types import (
    AdversarialEnvironment,
    NegotiationState,
)

# ---------------------------------------------------------------------------
# Negotiation strategies
# ---------------------------------------------------------------------------

class NegotiationStrategy:
    """Base class for negotiation strategies."""

    def make_offer(self, state: NegotiationState, is_buyer: bool) -> float:
        raise NotImplementedError

    def should_accept(
        self, state: NegotiationState, offer: float, is_buyer: bool
    ) -> bool:
        raise NotImplementedError


class GreedyStrategy(NegotiationStrategy):
    """Greedy baseline: always offer near own reservation value."""

    def __init__(self, concession_rate: float = 0.05):
        self.concession_rate = concession_rate

    def make_offer(self, state: NegotiationState, is_buyer: bool) -> float:
        progress = state.round_number / max(state.max_rounds, 1)
        concession = self.concession_rate * progress
        if is_buyer:
            # Start low, concede upward
            return state.buyer_reservation * (0.5 + concession)
        else:
            # Start high, concede downward
            return state.seller_reservation * (1.5 - concession)

    def should_accept(
        self, state: NegotiationState, offer: float, is_buyer: bool
    ) -> bool:
        if is_buyer:
            return offer <= state.buyer_reservation
        return offer >= state.seller_reservation


class BoulwareStrategy(NegotiationStrategy):
    """Boulware: hold firm, concede only near deadline."""

    def __init__(self, hardness: float = 3.0):
        self.hardness = hardness

    def make_offer(self, state: NegotiationState, is_buyer: bool) -> float:
        t = state.round_number / max(state.max_rounds, 1)
        # Concession follows t^hardness (late concession)
        concession = t ** self.hardness

        if is_buyer:
            lo = state.seller_reservation * 0.8
            hi = state.buyer_reservation
            return float(lo + concession * (hi - lo))
        else:
            hi = state.buyer_reservation * 1.2
            lo = state.seller_reservation
            return float(hi - concession * (hi - lo))

    def should_accept(
        self, state: NegotiationState, offer: float, is_buyer: bool
    ) -> bool:
        if is_buyer:
            return offer <= state.buyer_reservation
        return offer >= state.seller_reservation


class AdaptiveStrategy(NegotiationStrategy):
    """Adaptive: adjusts based on opponent's pattern."""

    def __init__(self) -> None:
        self._opponent_offers: list[float] = []

    def make_offer(self, state: NegotiationState, is_buyer: bool) -> float:
        t = state.round_number / max(state.max_rounds, 1)

        if len(self._opponent_offers) >= 2:
            # Estimate opponent's concession rate
            deltas = [
                self._opponent_offers[i] - self._opponent_offers[i - 1]
                for i in range(1, len(self._opponent_offers))
            ]
            avg_delta = sum(deltas) / len(deltas)
            # Match opponent's concession style
            concession = abs(avg_delta) * (1 + t)
        else:
            concession = 0.1 * t

        if is_buyer:
            return float(state.buyer_reservation * (0.6 + concession))
        else:
            return float(state.seller_reservation * (1.4 - concession))

    def should_accept(
        self, state: NegotiationState, offer: float, is_buyer: bool
    ) -> bool:
        self._opponent_offers.append(offer)
        if is_buyer:
            return offer <= state.buyer_reservation * 1.05  # Slight flexibility
        return offer >= state.seller_reservation * 0.95

    def reset(self) -> None:
        self._opponent_offers.clear()


# ---------------------------------------------------------------------------
# Negotiation result
# ---------------------------------------------------------------------------

@dataclass
class NegotiationResult:
    """Result of a bilateral negotiation."""

    negotiation_id: str = ""
    buyer_id: str = ""
    seller_id: str = ""
    agreed: bool = False
    agreement_price: float = 0.0
    rounds_used: int = 0
    buyer_surplus: float = 0.0
    seller_surplus: float = 0.0
    total_surplus: float = 0.0
    compute_spent: float = 0.0
    manipulation_attempted: bool = False
    manipulation_succeeded: bool = False


# ---------------------------------------------------------------------------
# Negotiation engine
# ---------------------------------------------------------------------------

class NegotiationEngine:
    """Engine for running bilateral negotiations."""

    def __init__(
        self,
        adversarial_env: AdversarialEnvironment = AdversarialEnvironment.BENIGN,
        rng: Optional[np.random.Generator] = None,
    ):
        self.adversarial_env = adversarial_env
        self.rng = rng or np.random.default_rng()

    def _make_agreement_result(
        self,
        state: NegotiationState,
        price: float,
        round_num: int,
        b_surplus: float,
        s_surplus: float,
        total_compute: float,
        manip_attempted: bool,
        manip_succeeded: bool,
    ) -> NegotiationResult:
        """Construct a NegotiationResult for an agreed outcome."""
        return NegotiationResult(
            negotiation_id=state.negotiation_id,
            buyer_id=state.buyer_id,
            seller_id=state.seller_id,
            agreed=True,
            agreement_price=price,
            rounds_used=round_num + 1,
            buyer_surplus=b_surplus,
            seller_surplus=s_surplus,
            total_surplus=b_surplus + s_surplus,
            compute_spent=total_compute,
            manipulation_attempted=manip_attempted,
            manipulation_succeeded=manip_succeeded,
        )

    def run_negotiation(
        self,
        state: NegotiationState,
        buyer_strategy: NegotiationStrategy,
        seller_strategy: NegotiationStrategy,
        compute_cost_per_round: float = 0.01,
    ) -> NegotiationResult:
        """Run a single bilateral negotiation.

        Args:
            state: Initial negotiation state.
            buyer_strategy: Buyer's bargaining strategy.
            seller_strategy: Seller's bargaining strategy.
            compute_cost_per_round: Cost per negotiation round.

        Returns:
            NegotiationResult.
        """
        total_compute = 0.0
        manip_attempted = False
        manip_succeeded = False

        for round_num in range(state.max_rounds):
            state.round_number = round_num

            # Budget check
            round_cost = compute_cost_per_round
            if total_compute + round_cost > state.compute_budget:
                break
            total_compute += round_cost

            # Buyer makes offer
            buyer_offer = buyer_strategy.make_offer(state, is_buyer=True)
            # Seller makes offer
            seller_offer = seller_strategy.make_offer(state, is_buyer=False)

            # Manipulation attempt
            if self.adversarial_env == AdversarialEnvironment.MANIPULATION:
                manip_attempted = True
                if float(self.rng.random()) < 0.3:
                    manip_succeeded = True
                    # Shift offers to disadvantage one side
                    seller_offer *= 1.1

            # Check if buyer accepts seller's offer
            if buyer_strategy.should_accept(state, seller_offer, is_buyer=True):
                price = seller_offer
                state.settled = True
                state.agreement_price = price

                b_surplus = state.buyer_reservation - price
                s_surplus = price - state.seller_reservation

                return self._make_agreement_result(
                    state, price, round_num, b_surplus, s_surplus,
                    total_compute, manip_attempted, manip_succeeded
                )

            # Check if seller accepts buyer's offer
            if seller_strategy.should_accept(state, buyer_offer, is_buyer=False):
                price = buyer_offer
                state.settled = True
                state.agreement_price = price

                b_surplus = state.buyer_reservation - price
                s_surplus = price - state.seller_reservation

                return self._make_agreement_result(
                    state, price, round_num, b_surplus, s_surplus,
                    total_compute, manip_attempted, manip_succeeded
                )

        # No agreement
        return NegotiationResult(
            negotiation_id=state.negotiation_id,
            buyer_id=state.buyer_id,
            seller_id=state.seller_id,
            agreed=False,
            rounds_used=state.max_rounds,
            compute_spent=total_compute,
            manipulation_attempted=manip_attempted,
            manipulation_succeeded=manip_succeeded,
        )
