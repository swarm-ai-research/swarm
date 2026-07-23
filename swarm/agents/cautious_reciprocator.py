"""Cautious Reciprocator agent — a custom agent that learns who to trust."""

from typing import Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType


class CautiousReciprocator(BaseAgent):
    """
    A trust-but-verify agent that starts cautious and rewards good partners.

    Strategy:
    - Starts with a high acceptance threshold (skeptical of strangers)
    - Tracks per-counterparty payoff history
    - Lowers threshold for agents who have delivered positive payoffs
    - Raises threshold (eventually blacklists) agents who deliver negative payoffs
    - Only proposes interactions to agents with positive track records
    - Diversifies: won't over-concentrate interactions with any single partner
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialise a CautiousReciprocator.

        Args:
            agent_id: Stable identifier for this agent in the simulation.
            roles: Optional explicit role list; defaults to BaseAgent default.
            config: Optional config dict. Recognised keys: ``base_threshold``,
                ``blacklist_threshold``, ``max_per_partner``,
                ``post_probability``.
            name: Optional human-readable display name.
            rng: Optional ``random.Random``-like instance for reproducibility.
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,  # Registers as honest archetype
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )

        # Tunables
        self.base_threshold = self.config.get("base_threshold", 0.55)
        self.blacklist_threshold = self.config.get("blacklist_threshold", -2.0)
        self.max_interactions_per_partner = self.config.get("max_per_partner", 8)
        self.post_probability = self.config.get("post_probability", 0.2)

        # Per-counterparty ledger: agent_id -> cumulative payoff
        self._payoff_ledger: Dict[str, float] = {}
        # Per-counterparty interaction count
        self._interaction_count: Dict[str, int] = {}
        # Blacklist
        self._blacklist: set = set()

    # ------------------------------------------------------------------
    # Core decision methods
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Choose the next action under a trust-but-verify policy.

        Decision order: accept/reject proposals against the per-counterparty
        ledger, handle bid decisions, occasionally bid on bounties, finish
        active tasks, claim new ones, propose to vetted partners, post
        occasionally, then vote.

        Args:
            observation: Current world view for this agent.

        Returns:
            The selected ``Action`` (defaults to a no-op when nothing matches).
        """
        # 1. Handle pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator = proposal.get("initiator_id", "")
            if self._is_acceptable(initiator):
                return self.create_accept_action(proposal["proposal_id"])
            return self.create_reject_action(proposal["proposal_id"])

        # 2. Handle marketplace bids
        if observation.pending_bid_decisions:
            return self._handle_bids(observation)

        # 3. Bid on bounties conservatively
        if observation.available_bounties and self._rng.random() < 0.25:
            return self._bid_on_bounty(observation)

        # 4. Work on active tasks
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                task_id = task.get("task_id", "")
                return self.create_submit_output_action(
                    task_id, f"Quality output for {task_id[:8]}"
                )

        # 5. Claim tasks
        if observation.can_claim_task and observation.available_tasks:
            task = observation.available_tasks[0]
            return self.create_claim_task_action(task["task_id"])

        # 6. Propose interactions to trusted partners
        if observation.can_interact and observation.visible_agents:
            action = self._propose_to_trusted(observation)
            if action:
                return action

        # 7. Post occasionally
        if observation.can_post and self._rng.random() < self.post_probability:
            return self.create_post_action("Sharing observations from recent work.")

        # 8. Vote on quality posts
        if observation.can_vote and observation.visible_posts:
            post = self._rng.choice(observation.visible_posts)
            direction = 1 if post.get("net_votes", 0) >= 0 else -1
            return self.create_vote_action(post.get("post_id", ""), direction)

        return self.create_noop_action()

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        """Accept only when the initiator clears trust and ledger gates.

        Args:
            proposal: Incoming interaction proposal.
            observation: Current world view (unused here but kept for API parity).

        Returns:
            True iff the initiator is not blacklisted, has not exceeded the
            per-partner cap with a non-profitable record, and meets the
            ledger-adjusted trust threshold.
        """
        return self._is_acceptable(proposal.initiator_id)

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> Optional[InteractionProposal]:
        """Propose only to a counterparty currently flagged as acceptable.

        Args:
            observation: Current world view (unused here but kept for API parity).
            counterparty_id: Candidate counterparty agent id.

        Returns:
            A new ``InteractionProposal`` when the counterparty passes the
            ledger-adjusted gate, otherwise ``None``.
        """
        if not self._is_acceptable(counterparty_id):
            return None
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Let's collaborate — I've had good results with you.",
        )

    # ------------------------------------------------------------------
    # Learning from outcomes
    # ------------------------------------------------------------------

    def update_from_outcome(self, interaction, payoff: float) -> None:
        """Update the per-counterparty ledger and auto-blacklist on losses.

        Tracks cumulative payoff and interaction count for each partner.
        Any counterparty whose cumulative payoff drops below
        ``self.blacklist_threshold`` is permanently added to the blacklist.

        Args:
            interaction: The completed interaction record.
            payoff: Realised payoff for this agent from the interaction.
        """
        super().update_from_outcome(interaction, payoff)

        counterparty = self._get_counterparty(interaction)

        # Update ledger
        self._payoff_ledger[counterparty] = (
            self._payoff_ledger.get(counterparty, 0.0) + payoff
        )
        self._interaction_count[counterparty] = (
            self._interaction_count.get(counterparty, 0) + 1
        )

        # Auto-blacklist agents who are net-negative
        if self._payoff_ledger[counterparty] < self.blacklist_threshold:
            self._blacklist.add(counterparty)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_acceptable(self, agent_id: str) -> bool:
        """Should we interact with this agent?"""
        if agent_id in self._blacklist:
            return False

        # Diversification: don't over-interact with one partner
        count = self._interaction_count.get(agent_id, 0)
        if count >= self.max_interactions_per_partner:
            # Only continue if they've been profitable
            avg = self._payoff_ledger.get(agent_id, 0.0) / max(count, 1)
            if avg < 0:
                return False

        # Trust score from base class EMA
        trust = self.compute_counterparty_trust(agent_id)

        # Lower threshold for agents with positive payoff history
        ledger = self._payoff_ledger.get(agent_id, 0.0)
        adjusted_threshold = self.base_threshold
        if ledger > 1.0:
            adjusted_threshold -= 0.15  # More willing for proven partners
        elif ledger < -0.5:
            adjusted_threshold += 0.2  # More cautious for poor performers

        return bool(trust >= adjusted_threshold)

    def _propose_to_trusted(self, observation: Observation) -> Optional[Action]:
        """Propose collaboration to the best available partner."""
        candidates = []
        for agent in observation.visible_agents:
            aid = agent.get("agent_id", "")
            if aid == self.agent_id or aid in self._blacklist:
                continue
            ledger = self._payoff_ledger.get(aid, 0.0)
            trust = self.compute_counterparty_trust(aid)
            candidates.append((aid, ledger, trust))

        if not candidates:
            return None

        # Sort by cumulative payoff, then trust
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_id = candidates[0][0]

        if not self._is_acceptable(best_id):
            return None

        return self.create_propose_action(
            counterparty_id=best_id,
            interaction_type=InteractionType.COLLABORATION,
            content="I'd like to collaborate based on our track record.",
        )

    def _handle_bids(self, observation: Observation) -> Action:
        """Accept the most cost-effective bid."""
        bids = observation.pending_bid_decisions
        if not bids:
            return self.create_noop_action()
        best = min(bids, key=lambda b: b.get("bid_amount", float("inf")))
        return self.create_accept_bid_action(
            bounty_id=best["bounty_id"], bid_id=best["bid_id"]
        )

    def _bid_on_bounty(self, observation: Observation) -> Action:
        """Bid conservatively on bounties."""
        bounties = observation.available_bounties
        if not bounties:
            return self.create_noop_action()
        bounty = self._rng.choice(bounties)
        reward = bounty.get("reward_amount", 0)
        bid = reward * self._rng.uniform(0.85, 0.95)
        return self.create_place_bid_action(
            bounty_id=bounty["bounty_id"],
            bid_amount=bid,
            message="Reliable delivery, fair price.",
        )
