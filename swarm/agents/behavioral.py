"""Behavioral agent types: Cautious, Collaborative, Adaptive.

Three complementary behavioral archetypes for richer simulation scenarios:

- CautiousAgent: Risk-averse; only accepts interactions with high expected quality.
- CollaborativeAgent: Coalition-builder; preferentially partners with known good actors.
- AdaptiveAgent: Learning agent; adjusts acceptance strategy from interaction history.
"""

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

# ---------------------------------------------------------------------------
# CautiousAgent
# ---------------------------------------------------------------------------


class CautiousAgent(BaseAgent):
    """Agent that only accepts high-quality interactions.

    Strategy:
    - Applies a high acceptance threshold on expected interaction quality (p).
    - Falls back to counterparty trust when no quality signal is present.
    - Avoids proposing to agents with below-threshold trust scores.
    - Models risk-averse behaviour in adversarial environments.

    Config keys:
        threshold (float, default 0.7): Minimum expected p to accept.
        post_probability (float, default 0.2): Probability of posting each step.
    """

    def __init__(
        self,
        agent_id: str,
        threshold: float = 0.7,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialise a CautiousAgent.

        Args:
            agent_id: Stable identifier for this agent in the simulation.
            threshold: Minimum expected interaction quality `p` (or trust) to
                accept a counterparty. Used as the default if not overridden
                via ``config["threshold"]``.
            roles: Optional explicit role list; defaults to BaseAgent default.
            config: Optional config dict. Recognised keys: ``threshold``,
                ``post_probability``.
            name: Optional human-readable display name.
            rng: Optional ``random.Random``-like instance for reproducibility.
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.CAUTIOUS,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.threshold: float = self.config.get("threshold", threshold)
        self.post_probability: float = self.config.get("post_probability", 0.2)

    # ------------------------------------------------------------------
    # Core decision methods
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Choose the next action under a cautious / risk-averse policy.

        Decision order: accept/reject pending proposals, evaluate bids,
        finish in-progress tasks, claim only easy/trivial tasks, propose to
        high-trust agents, occasionally post, then upvote good content.

        Args:
            observation: Current world view for this agent.

        Returns:
            The selected ``Action`` (defaults to a no-op when nothing matches).
        """
        # 1. Handle pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")
            expected_p = proposal.get("expected_p", None)
            if self._is_acceptable(initiator_id, expected_p):
                return self.create_accept_action(proposal["proposal_id"])
            return self.create_reject_action(proposal["proposal_id"])

        # 2. Handle bid decisions cautiously (accept only low bids from trusted bidders)
        if observation.pending_bid_decisions:
            return self._handle_bids(observation)

        # 3. Work on active tasks
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self.create_submit_output_action(
                    task["task_id"], f"Careful output for {task['task_id'][:8]}"
                )

        # 4. Claim easy tasks only
        if observation.can_claim_task and observation.available_tasks:
            safe_tasks = [
                t
                for t in observation.available_tasks
                if t.get("difficulty", "hard") in ("trivial", "easy")
            ]
            if safe_tasks:
                return self.create_claim_task_action(safe_tasks[0]["task_id"])

        # 5. Propose to high-trust agents
        if observation.can_interact and observation.visible_agents:
            action = self._propose_to_trusted(observation)
            if action:
                return action

        # 6. Post occasionally
        if observation.can_post and self._rng.random() < self.post_probability:
            return self.create_post_action("Sharing a carefully verified observation.")

        # 7. Upvote quality content
        if observation.can_vote and observation.visible_posts:
            post = self._rng.choice(observation.visible_posts)
            direction = 1 if post.get("net_votes", 0) >= 0 else -1
            return self.create_vote_action(post.get("post_id", ""), direction)

        return self.create_noop_action()

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        """Accept only if expected quality (or fallback trust) clears threshold.

        Args:
            proposal: Incoming interaction proposal.
            observation: Current world view (unused here but kept for API parity).

        Returns:
            True when the proposal's expected ``p`` (or counterparty trust)
            is at least ``self.threshold``.
        """
        expected_p = proposal.metadata.get("expected_p") if proposal.metadata else None
        return self._is_acceptable(proposal.initiator_id, expected_p)

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> Optional[InteractionProposal]:
        """Propose a collaboration only to a trusted counterparty.

        Args:
            observation: Current world view (unused here but kept for API parity).
            counterparty_id: Candidate counterparty agent id.

        Returns:
            A new ``InteractionProposal`` when the counterparty meets the
            cautious trust threshold, otherwise ``None``.
        """
        if not self._is_acceptable(counterparty_id):
            return None
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Proposing a careful, quality-focused collaboration.",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_acceptable(
        self, agent_id: str, expected_p: Optional[float] = None
    ) -> bool:
        """Return True only if the quality signal or trust meets our threshold."""
        if expected_p is not None:
            return bool(expected_p >= self.threshold)
        # Fall back to trust-based gating
        trust = self.compute_counterparty_trust(agent_id)
        return bool(trust >= self.threshold)

    def _propose_to_trusted(self, observation: Observation) -> Optional[Action]:
        """Propose only to agents whose trust meets the threshold."""
        candidates = [
            (a.get("agent_id", ""), self.compute_counterparty_trust(a.get("agent_id", "")))
            for a in observation.visible_agents
            if a.get("agent_id", "") != self.agent_id
        ]
        qualified = [(aid, t) for aid, t in candidates if t >= self.threshold]
        if not qualified:
            return None
        best_id = max(qualified, key=lambda x: x[1])[0]
        return self.create_propose_action(
            counterparty_id=best_id,
            interaction_type=InteractionType.COLLABORATION,
            content="I'd like to collaborate — your track record looks solid.",
        )

    def _handle_bids(self, observation: Observation) -> Action:
        bids = observation.pending_bid_decisions
        if not bids:
            return self.create_noop_action()
        # Accept the cheapest bid from a trusted bidder; otherwise skip.
        trusted_bids = [
            b
            for b in bids
            if self.compute_counterparty_trust(b.get("bidder_id", "")) >= self.threshold
        ]
        pool = trusted_bids if trusted_bids else bids  # fall back if no trusted bids
        best = min(pool, key=lambda b: b.get("bid_amount", float("inf")))
        return self.create_accept_bid_action(
            bounty_id=best["bounty_id"], bid_id=best["bid_id"]
        )


# ---------------------------------------------------------------------------
# CollaborativeAgent
# ---------------------------------------------------------------------------


class CollaborativeAgent(BaseAgent):
    """Agent that builds coalitions with known good actors.

    Strategy:
    - Maintains a running trust score per counterparty (via BaseAgent EMA).
    - Preferentially proposes to and accepts from the highest-trust agents.
    - Extends trust to neutral/unknown agents up to a configurable limit.
    - Models cooperative coalition-formation dynamics.

    Config keys:
        min_trust (float, default 0.45): Minimum trust to accept from strangers.
        coalition_size (int, default 5): Max preferred coalition partners.
        post_probability (float, default 0.3): Probability of posting.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialise a CollaborativeAgent.

        Args:
            agent_id: Stable identifier for this agent in the simulation.
            roles: Optional explicit role list; defaults to BaseAgent default.
            config: Optional config dict. Recognised keys: ``min_trust``,
                ``coalition_size``, ``post_probability``.
            name: Optional human-readable display name.
            rng: Optional ``random.Random``-like instance for reproducibility.
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.COLLABORATIVE,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.min_trust: float = self.config.get("min_trust", 0.45)
        self.coalition_size: int = self.config.get("coalition_size", 5)
        self.post_probability: float = self.config.get("post_probability", 0.3)

        # Preferred coalition set: {agent_id}
        self._coalition: set = set()

    # ------------------------------------------------------------------
    # Core decision methods
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Choose the next action under a coalition-building policy.

        Decision order: accept proposals from coalition or trusted agents,
        finish in-progress tasks, claim available tasks, propose to coalition
        members or rising-trust peers, signal openness via posts, then vote.

        Args:
            observation: Current world view for this agent.

        Returns:
            The selected ``Action`` (defaults to a no-op when nothing matches).
        """
        # 1. Handle pending proposals — accept from coalition and trusted agents
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")
            if self._should_accept(initiator_id):
                return self.create_accept_action(proposal["proposal_id"])
            return self.create_reject_action(proposal["proposal_id"])

        # 2. Work on active tasks
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self.create_submit_output_action(
                    task["task_id"],
                    f"Collaborative output for {task['task_id'][:8]}",
                )

        # 3. Claim tasks
        if observation.can_claim_task and observation.available_tasks:
            return self.create_claim_task_action(
                observation.available_tasks[0]["task_id"]
            )

        # 4. Propose to coalition members or build new relationships
        if observation.can_interact and observation.visible_agents:
            action = self._propose_interaction(observation)
            if action:
                return action

        # 5. Post to signal cooperation
        if observation.can_post and self._rng.random() < self.post_probability:
            return self.create_post_action(
                "Open to collaboration — looking for reliable partners."
            )

        # 6. Vote on quality posts
        if observation.can_vote and observation.visible_posts:
            post = self._rng.choice(observation.visible_posts)
            direction = 1 if post.get("net_votes", 0) >= 0 else -1
            return self.create_vote_action(post.get("post_id", ""), direction)

        return self.create_noop_action()

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        """Accept when the initiator is in the coalition or sufficiently trusted.

        Args:
            proposal: Incoming interaction proposal.
            observation: Current world view (unused here but kept for API parity).

        Returns:
            True when the initiator is a coalition member or has trust
            ``>= self.min_trust``.
        """
        return self._should_accept(proposal.initiator_id)

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> Optional[InteractionProposal]:
        """Propose to a coalition member or a trust-qualified candidate.

        Args:
            observation: Current world view (unused here but kept for API parity).
            counterparty_id: Candidate counterparty agent id.

        Returns:
            A new ``InteractionProposal`` when the counterparty is acceptable,
            otherwise ``None``.
        """
        if not self._should_accept(counterparty_id):
            return None
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Join our coalition — better outcomes together.",
        )

    def update_from_outcome(self, interaction, payoff: float) -> None:
        """Update trust and dynamically grow/prune the coalition set.

        High-trust partners (>=0.65) are added to the coalition up to
        ``self.coalition_size``; partners whose trust drops below 0.4 are
        evicted.

        Args:
            interaction: The completed interaction record.
            payoff: Realised payoff for this agent from the interaction.
        """
        super().update_from_outcome(interaction, payoff)
        # Promote high-quality partners into the coalition
        counterparty = self._get_counterparty(interaction)
        trust = self.compute_counterparty_trust(counterparty)
        if trust >= 0.65 and len(self._coalition) < self.coalition_size:
            self._coalition.add(counterparty)
        elif trust < 0.4:
            self._coalition.discard(counterparty)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_accept(self, agent_id: str) -> bool:
        """Accept coalition members and sufficiently trusted strangers."""
        if agent_id in self._coalition:
            return True
        trust = self.compute_counterparty_trust(agent_id)
        return bool(trust >= self.min_trust)

    def _propose_interaction(self, observation: Observation) -> Optional[Action]:
        """Propose to the highest-trust visible agent, coalition members first."""
        candidates: List[tuple] = []
        for agent in observation.visible_agents:
            aid = agent.get("agent_id", "")
            if aid == self.agent_id:
                continue
            trust = self.compute_counterparty_trust(aid)
            in_coalition = aid in self._coalition
            candidates.append((aid, trust, in_coalition))

        if not candidates:
            return None

        # Coalition members first, then highest trust
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        best_id = candidates[0][0]

        return self.create_propose_action(
            counterparty_id=best_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Coalition collaboration proposal.",
        )


# ---------------------------------------------------------------------------
# AdaptiveAgent
# ---------------------------------------------------------------------------


class AdaptiveAgent(BaseAgent):
    """Agent that learns from outcomes and adapts its acceptance strategy.

    Strategy:
    - Tracks a rolling window of recent interaction outcomes.
    - Adjusts acceptance threshold up (more cautious) when recent payoffs are low
      and down (more open) when payoffs are consistently high.
    - Explores new counterparties occasionally to avoid local optima.
    - Models reinforcement-learning / evolutionary dynamics.

    Config keys:
        initial_threshold (float, default 0.5): Starting acceptance threshold.
        min_threshold (float, default 0.3): Floor for threshold adaptation.
        max_threshold (float, default 0.8): Ceiling for threshold adaptation.
        window_size (int, default 20): Rolling window for outcome tracking.
        adapt_rate (float, default 0.05): Step size for threshold updates.
        explore_probability (float, default 0.1): Prob of accepting a stranger.
        post_probability (float, default 0.25): Probability of posting.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialise an AdaptiveAgent.

        Args:
            agent_id: Stable identifier for this agent in the simulation.
            roles: Optional explicit role list; defaults to BaseAgent default.
            config: Optional config dict. Recognised keys:
                ``initial_threshold``, ``min_threshold``, ``max_threshold``,
                ``window_size``, ``adapt_rate``, ``explore_probability``,
                ``post_probability``.
            name: Optional human-readable display name.
            rng: Optional ``random.Random``-like instance for reproducibility.
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADAPTIVE,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.threshold: float = self.config.get("initial_threshold", 0.5)
        self.min_threshold: float = self.config.get("min_threshold", 0.3)
        self.max_threshold: float = self.config.get("max_threshold", 0.8)
        self.window_size: int = self.config.get("window_size", 20)
        self.adapt_rate: float = self.config.get("adapt_rate", 0.05)
        self.explore_probability: float = self.config.get("explore_probability", 0.1)
        self.post_probability: float = self.config.get("post_probability", 0.25)

        # Rolling payoff window for adaptation
        self._recent_payoffs: List[float] = []

    # ------------------------------------------------------------------
    # Core decision methods
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Choose the next action under an adaptive / learning policy.

        Decision order: accept/reject proposals at the current adapted
        threshold, finish active tasks, claim tasks, propose using an
        explore/exploit policy, occasionally post status updates, then vote.

        Args:
            observation: Current world view for this agent.

        Returns:
            The selected ``Action`` (defaults to a no-op when nothing matches).
        """
        # 1. Handle pending proposals with adaptive threshold
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")
            if self._is_acceptable(initiator_id):
                return self.create_accept_action(proposal["proposal_id"])
            return self.create_reject_action(proposal["proposal_id"])

        # 2. Work on active tasks
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self.create_submit_output_action(
                    task["task_id"],
                    f"Adaptive output for {task['task_id'][:8]}",
                )

        # 3. Claim tasks
        if observation.can_claim_task and observation.available_tasks:
            return self.create_claim_task_action(
                observation.available_tasks[0]["task_id"]
            )

        # 4. Propose interactions — explore or exploit
        if observation.can_interact and observation.visible_agents:
            action = self._propose_interaction(observation)
            if action:
                return action

        # 5. Post occasionally
        if observation.can_post and self._rng.random() < self.post_probability:
            return self.create_post_action(
                f"Adapting strategy. Current threshold: {self.threshold:.2f}"
            )

        # 6. Vote
        if observation.can_vote and observation.visible_posts:
            post = self._rng.choice(observation.visible_posts)
            direction = 1 if post.get("net_votes", 0) >= 0 else -1
            return self.create_vote_action(post.get("post_id", ""), direction)

        return self.create_noop_action()

    def accept_interaction(
        self, proposal: InteractionProposal, observation: Observation
    ) -> bool:
        """Accept when the initiator clears the current adaptive threshold.

        With probability ``self.explore_probability``, any initiator is
        accepted to enable exploration of new partners.

        Args:
            proposal: Incoming interaction proposal.
            observation: Current world view (unused here but kept for API parity).

        Returns:
            True when the initiator is acceptable under the current threshold
            or selected for exploration.
        """
        return self._is_acceptable(proposal.initiator_id)

    def propose_interaction(
        self, observation: Observation, counterparty_id: str
    ) -> Optional[InteractionProposal]:
        """Propose to a counterparty that clears the adaptive threshold.

        Args:
            observation: Current world view (unused here but kept for API parity).
            counterparty_id: Candidate counterparty agent id.

        Returns:
            A new ``InteractionProposal`` when the counterparty is acceptable
            under the current threshold, otherwise ``None``.
        """
        if not self._is_acceptable(counterparty_id):
            return None
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content=f"Proposal at adapted threshold {self.threshold:.2f}.",
        )

    def update_from_outcome(self, interaction, payoff: float) -> None:
        """Record payoff in the rolling window and re-tune the threshold.

        High average payoffs relax the threshold (open up); low average
        payoffs tighten it. Window length is ``self.window_size`` and the
        per-update step is ``self.adapt_rate``.

        Args:
            interaction: The completed interaction record.
            payoff: Realised payoff for this agent from the interaction.
        """
        super().update_from_outcome(interaction, payoff)
        # Record payoff and adapt threshold
        self._recent_payoffs.append(payoff)
        if len(self._recent_payoffs) > self.window_size:
            self._recent_payoffs.pop(0)
        self._adapt_threshold()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adapt_threshold(self) -> None:
        """Adjust threshold based on recent payoff performance."""
        if not self._recent_payoffs:
            return
        avg = sum(self._recent_payoffs) / len(self._recent_payoffs)
        if avg > 0.5:
            # Doing well — can afford to open up slightly
            self.threshold = max(self.min_threshold, self.threshold - self.adapt_rate)
        else:
            # Poor outcomes — tighten standards
            self.threshold = min(self.max_threshold, self.threshold + self.adapt_rate)

    def _is_acceptable(self, agent_id: str) -> bool:
        """Accept based on adaptive threshold, with occasional exploration."""
        # Exploration: accept any agent with small probability
        if self._rng.random() < self.explore_probability:
            return True
        trust = self.compute_counterparty_trust(agent_id)
        return bool(trust >= self.threshold)

    def _propose_interaction(self, observation: Observation) -> Optional[Action]:
        """Exploit best known partners or explore a random candidate."""
        candidates = [
            (a.get("agent_id", ""), self.compute_counterparty_trust(a.get("agent_id", "")))
            for a in observation.visible_agents
            if a.get("agent_id", "") != self.agent_id
        ]
        if not candidates:
            return None

        # Exploit: propose to highest-trust agent above threshold
        above_threshold = [(aid, t) for aid, t in candidates if t >= self.threshold]
        if above_threshold:
            best_id = max(above_threshold, key=lambda x: x[1])[0]
        elif self._rng.random() < self.explore_probability:
            # Explore: pick a random unknown candidate
            best_id = self._rng.choice(candidates)[0]
        else:
            return None

        return self.create_propose_action(
            counterparty_id=best_id,
            interaction_type=InteractionType.COLLABORATION,
            content=f"Adaptive collaboration proposal (threshold={self.threshold:.2f}).",
        )
