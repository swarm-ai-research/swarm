"""Negotiation agent strategies for resource negotiation games.

Provides several agent types that play the resource negotiation game
with different strategies:

- ``FairNegotiator``: proposes even splits, accepts reasonable deals
- ``GreedyNegotiator``: claims most value, concedes under deadline pressure
- ``StrategicNegotiator``: reads opponent proposals to infer valuations,
  proposes mutually-beneficial splits, tightens acceptance as deadline nears
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


class NegotiatorBase(BaseAgent):
    """Shared helpers for the negotiator strategy agents."""

    def _compute_score(
        self,
        allocation: Dict[str, int],
        valuations: Dict[str, float],
        pool: Dict[str, int],
    ) -> float:
        raw = sum(valuations.get(r, 0) * allocation.get(r, 0) for r in pool)
        max_s = sum(valuations.get(r, 0) * pool[r] for r in pool)
        return raw / max_s if max_s > 0 else 0.0

    def _get_active_game(self, observation: Observation) -> Optional[Dict]:
        games: List[Dict] = getattr(observation, "resource_negotiation_games", [])
        for g in games:
            if not g.get("game_over", True):
                return g
        return None

    def _make_negotiate_action(
        self,
        game_id: str,
        negotiate_action: str,
        proposal: Optional[Dict] = None,
        message: str = "",
    ) -> Action:
        meta: Dict[str, Any] = {
            "game_id": game_id,
            "negotiate_action": negotiate_action,
            "message": message,
        }
        if proposal:
            meta["proposal"] = proposal
        return Action(
            action_type=ActionType.RESOURCE_NEGOTIATE,
            agent_id=self.agent_id,
            metadata=meta,
        )


class FairNegotiator(NegotiatorBase):
    """Negotiation agent that proposes fair (even) splits.

    Strategy: split resources roughly 50/50 by quantity, accept any
    proposal that gives at least 30% of max possible score.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng=None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )
        self.accept_threshold = self.config.get("accept_threshold", 0.3)

    def act(self, observation: Observation) -> Action:
        game = self._get_active_game(observation)
        if game is None:
            return self.create_noop_action()

        if not game.get("is_your_turn", False):
            return self.create_noop_action()

        pool = game["pool"]
        valuations = game["your_valuations"]
        last_proposal = game.get("last_proposal")

        # Accept if the last proposal gives us enough value
        if last_proposal and last_proposal.get("proposer_role") != game["role"]:
            my_alloc = last_proposal["their_share"]  # we're the "their"
            score = self._compute_score(my_alloc, valuations, pool)
            if score >= self.accept_threshold:
                return self._make_negotiate_action(
                    game["game_id"], "accept", message="Fair enough, deal!"
                )

        # Propose an even split
        my_share, their_share = self._even_split(pool)
        return self._make_negotiate_action(
            game["game_id"],
            "propose",
            proposal={"my_share": my_share, "their_share": their_share},
            message="Let's split this fairly.",
        )

    def _even_split(self, pool: Dict[str, int]) -> tuple:
        my_share = {}
        their_share = {}
        for name, qty in sorted(pool.items()):
            half = qty // 2
            my_share[name] = half
            their_share[name] = qty - half
        return my_share, their_share

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        return True

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> Optional[InteractionProposal]:
        return None


class GreedyNegotiator(NegotiatorBase):
    """Negotiation agent that maximizes own value aggressively.

    Strategy: claim all high-value resources initially, slowly concede
    as the deadline approaches to avoid the -0.5 no-deal penalty.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng=None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.OPPORTUNISTIC,
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

    def act(self, observation: Observation) -> Action:
        game = self._get_active_game(observation)
        if game is None:
            return self.create_noop_action()

        if not game.get("is_your_turn", False):
            return self.create_noop_action()

        pool = game["pool"]
        valuations = game["your_valuations"]
        round_num = game.get("round_number", 1)
        last_proposal = game.get("last_proposal")

        # Under deadline pressure (round 5+), accept anything > 0
        if round_num >= 5 and last_proposal:
            proposer_role = last_proposal.get("proposer_role")
            if proposer_role != game["role"]:
                my_alloc = last_proposal["their_share"]
                score = self._compute_score(my_alloc, valuations, pool)
                if score > 0:
                    return self._make_negotiate_action(
                        game["game_id"], "accept",
                        message="Fine, let's close this.",
                    )

        # In early rounds, accept only if ≥ 60% of max
        if last_proposal and last_proposal.get("proposer_role") != game["role"]:
            my_alloc = last_proposal["their_share"]
            score = self._compute_score(my_alloc, valuations, pool)
            threshold = max(0.2, 0.7 - 0.1 * round_num)
            if score >= threshold:
                return self._make_negotiate_action(
                    game["game_id"], "accept",
                    message="I can work with that.",
                )

        # Propose greedily: take all high-value resources
        my_share, their_share = self._greedy_split(pool, valuations, round_num)
        return self._make_negotiate_action(
            game["game_id"],
            "propose",
            proposal={"my_share": my_share, "their_share": their_share},
            message="Here's my offer.",
        )

    def _greedy_split(
        self,
        pool: Dict[str, int],
        valuations: Dict[str, float],
        round_num: int,
    ) -> tuple:
        """Split resources favoring high-value ones for self.

        Concession factor increases with round number.
        """
        concession = min(0.5, 0.05 * round_num)
        my_share = {}
        their_share = {}

        # Sort resources by our valuation (highest first)
        sorted_resources = sorted(
            pool.items(), key=lambda x: valuations.get(x[0], 0), reverse=True
        )

        for name, qty in sorted_resources:
            # Take most of high-value resources, give away low-value ones
            my_qty = max(0, int(qty * (1.0 - concession)))
            my_share[name] = my_qty
            their_share[name] = qty - my_qty

        return my_share, their_share

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        return True

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> Optional[InteractionProposal]:
        return None


class StrategicNegotiator(NegotiatorBase):
    """Negotiation agent that infers opponent valuations from proposals.

    Strategy:
    - Round 1-2: anchor high, observe opponent's counter-proposals
    - Round 3-4: propose win-win splits based on inferred valuations
    - Round 5+: accept declining thresholds to avoid no-deal
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng=None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

    def act(self, observation: Observation) -> Action:
        game = self._get_active_game(observation)
        if game is None:
            return self.create_noop_action()

        if not game.get("is_your_turn", False):
            return self.create_noop_action()

        pool = game["pool"]
        valuations = game["your_valuations"]
        round_num = game.get("round_number", 1)
        last_proposal = game.get("last_proposal")
        history = game.get("history", [])

        # Determine acceptance threshold based on round
        if round_num >= 7:
            threshold = 0.05
        elif round_num >= 6:
            threshold = 0.25
        elif round_num >= 5:
            threshold = 0.40
        else:
            threshold = 0.60

        # Check if we should accept the current proposal
        if last_proposal and last_proposal.get("proposer_role") != game["role"]:
            my_alloc = last_proposal["their_share"]
            score = self._compute_score(my_alloc, valuations, pool)
            if score >= threshold:
                return self._make_negotiate_action(
                    game["game_id"], "accept",
                    message="This works for both of us.",
                )

        # Infer what opponent values from their proposals
        opponent_preferences = self._infer_opponent_preferences(
            history, game["role"], pool
        )

        # Build a strategic proposal
        my_share, their_share = self._strategic_split(
            pool, valuations, opponent_preferences, round_num
        )
        return self._make_negotiate_action(
            game["game_id"],
            "propose",
            proposal={"my_share": my_share, "their_share": their_share},
            message="I think this works well for both of us.",
        )

    def _infer_opponent_preferences(
        self,
        history: List[Dict],
        my_role: str,
        pool: Dict[str, int],
    ) -> Dict[str, float]:
        """Infer relative opponent preferences from their proposals.

        Returns a dict mapping resource_name → inferred preference weight
        (higher means opponent values it more).
        """
        preferences: Dict[str, float] = dict.fromkeys(pool, 1.0)

        for turn in history:
            # Look at opponent's proposals
            if turn["role"] == my_role:
                continue
            if turn["action"] != "propose" or turn.get("proposal") is None:
                continue

            prop = turn["proposal"]
            # The opponent's "my_share" is what they want
            their_desired = prop["my_share"]
            for name, qty in their_desired.items():
                total = pool.get(name, 1)
                if total > 0:
                    # Higher fraction claimed → higher preference
                    preferences[name] += qty / total

        return preferences

    def _strategic_split(
        self,
        pool: Dict[str, int],
        my_valuations: Dict[str, float],
        opponent_preferences: Dict[str, float],
        round_num: int,
    ) -> tuple:
        """Build a split that gives opponent what they seem to want
        while keeping what we value most.
        """
        my_share = {}
        their_share = {}

        # Compute a "trade advantage" score for each resource:
        # high advantage = we value it much more than opponent seems to
        trade_scores = {}
        for name, _qty in pool.items():
            my_val = my_valuations.get(name, 0)
            opp_pref = opponent_preferences.get(name, 1.0)
            # Higher ratio = we want it more relative to them
            trade_scores[name] = my_val / max(opp_pref, 0.01)

        # Sort by trade advantage (we keep the highest advantage resources)
        sorted_resources = sorted(
            pool.items(), key=lambda x: trade_scores.get(x[0], 0), reverse=True
        )

        # Generosity increases with round number
        generosity = min(0.5, 0.05 * round_num)

        for name, qty in sorted_resources:
            advantage = trade_scores.get(name, 1.0)
            if advantage > 1.5:
                # We want it more — take most
                my_qty = max(0, int(qty * (0.8 - generosity)))
            elif advantage < 0.7:
                # They want it more — give most to them
                my_qty = max(0, int(qty * (0.2 + generosity * 0.5)))
            else:
                # Similar preference — split evenly
                my_qty = qty // 2

            my_share[name] = min(my_qty, qty)
            their_share[name] = qty - my_share[name]

        return my_share, their_share

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        return True

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> Optional[InteractionProposal]:
        return None
