"""Honest agent policy implementation."""

from typing import TYPE_CHECKING, Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


class HonestAgent(BaseAgent):
    """
    Honest agent that acts transparently and cooperatively.

    Characteristics:
    - Accepts interactions with reasonable quality expectations
    - Creates genuine, helpful content
    - Votes based on content quality
    - Completes tasks diligently
    - Builds reputation through consistent behavior
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
        """Initialize honest agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

        # Honest agent parameters
        self.acceptance_threshold = self.config.get("acceptance_threshold", 0.4)
        self.trust_weight = self.config.get("trust_weight", 0.3)
        self.post_probability = self.config.get("post_probability", 0.3)
        self.vote_probability = self.config.get("vote_probability", 0.5)
        self.interact_probability = self.config.get("interact_probability", 0.4)

    def act(self, observation: Observation) -> Action:
        """
        Decide action based on current observation.

        Honest agents prioritize:
        1. Responding to pending proposals
        2. Handling marketplace bid decisions
        3. Working on active tasks
        4. Posting helpful content
        5. Engaging with quality posts
        6. Proposing beneficial interactions
        """
        # First, handle any pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            if self._should_accept_proposal(proposal, observation):
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # Handle marketplace bid decisions (accept qualified bids)
        if observation.pending_bid_decisions:
            return self._handle_bid_decision(observation)

        # Bid on available bounties
        if observation.available_bounties and self._rng.random() < 0.3:
            return self._bid_on_bounty(observation)

        # Post bounties for collaboration
        if (
            observation.agent_state.resources > 20
            and self._rng.random() < 0.15
            and not observation.available_bounties
        ):
            return self._post_bounty(observation)

        # Work on active tasks
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self._work_on_task(task, observation)

        # Claim a task if none active
        if observation.can_claim_task and observation.available_tasks:
            best_task = self._select_task(observation.available_tasks, observation)
            if best_task:
                return self.create_claim_task_action(best_task["task_id"])

        # Post with some probability
        if observation.can_post and self._rng.random() < self.post_probability:
            content = self._generate_helpful_content(observation)
            if content:
                return self.create_post_action(content)

        # Vote on visible posts
        if (
            observation.can_vote
            and observation.visible_posts
            and self._rng.random() < self.vote_probability
        ):
            return self._vote_on_posts(observation)

        # Propose interactions
        if observation.can_interact and self._rng.random() < self.interact_probability:
            if observation.visible_agents:
                return self._propose_interaction(observation)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Accept interactions that seem beneficial.

        Considers:
        - Trust in the counterparty
        - Interaction type appropriateness
        - Current resource availability
        """
        # Check trust level
        trust = self.compute_counterparty_trust(proposal.initiator_id)

        # Base acceptance on trust and threshold
        threshold = self.acceptance_threshold

        # Adjust threshold based on interaction type
        if proposal.interaction_type == InteractionType.COLLABORATION:
            threshold -= 0.1  # More willing to collaborate
        elif proposal.interaction_type == InteractionType.TRADE:
            threshold += 0.1  # More cautious with trades

        # Weight trust into decision
        effective_score = trust * self.trust_weight + 0.5 * (1 - self.trust_weight)

        return bool(effective_score >= threshold)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """
        Propose a beneficial interaction.

        Honest agents propose collaborations and helpful interactions.
        """
        # Check trust
        trust = self.compute_counterparty_trust(counterparty_id)

        # Only propose to trusted agents or unknowns
        if trust < 0.3:
            return None

        # Choose interaction type based on context
        if observation.active_tasks:
            interaction_type = InteractionType.COLLABORATION
            content = "Would you like to collaborate on this task?"
        else:
            interaction_type = InteractionType.REPLY
            content = "I'd like to engage in a helpful exchange."

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=interaction_type,
            content=content,
            offered_transfer=0.0,  # Honest agents don't bribe
        )

    def _handle_bid_decision(self, observation: Observation) -> Action:
        """Accept the most qualified bid on our bounties."""
        bids = observation.pending_bid_decisions
        if not bids:
            return self.create_noop_action()

        # Accept the lowest bid (best value)
        best = min(bids, key=lambda b: b.get("bid_amount", float("inf")))
        return self.create_accept_bid_action(
            bounty_id=best["bounty_id"],
            bid_id=best["bid_id"],
        )

    def _bid_on_bounty(self, observation: Observation) -> Action:
        """Bid fairly on a bounty."""
        bounties = observation.available_bounties
        if not bounties:
            return self.create_noop_action()

        # Pick a bounty and bid at fair value
        bounty = self._rng.choice(bounties)
        reward = bounty.get("reward_amount", 0)
        # Honest agents bid at fair price (80-95% of reward)
        bid_amount = reward * self._rng.uniform(0.8, 0.95)
        return self.create_place_bid_action(
            bounty_id=bounty["bounty_id"],
            bid_amount=bid_amount,
            message="I can deliver quality work on this task.",
        )

    def _post_bounty(self, observation: Observation) -> Action:
        """Post a bounty for work needed."""
        reward = self._rng.uniform(5.0, 15.0)
        return self.create_post_bounty_action(
            reward_amount=reward,
            task_description="Looking for quality collaboration on a research task.",
        )

    def _should_accept_proposal(self, proposal: Dict, observation: Observation) -> bool:
        """Decide whether to accept a proposal dict."""
        # Convert dict to check
        initiator_id = proposal.get("initiator_id", "")
        trust = self.compute_counterparty_trust(initiator_id)

        # Accept if trust is above threshold
        return bool(trust >= self.acceptance_threshold - 0.1)

    def _select_task(
        self, tasks: List[Dict], observation: Observation
    ) -> Optional[Dict]:
        """Select the best task to claim."""
        if not tasks:
            return None

        # Filter by reputation requirement
        my_rep = observation.agent_state.reputation
        eligible = [t for t in tasks if t.get("min_reputation", 0) <= my_rep]

        if not eligible:
            return None

        # Sort by bounty/difficulty ratio
        def score(t):
            bounty = t.get("bounty", 0)
            difficulty_map = {
                "trivial": 1,
                "easy": 2,
                "medium": 3,
                "hard": 4,
                "expert": 5,
            }
            difficulty = difficulty_map.get(t.get("difficulty", "medium"), 3)
            return bounty / difficulty

        eligible.sort(key=score, reverse=True)
        return eligible[0]

    def _work_on_task(self, task: Dict, observation: Observation) -> Action:
        """Work on an active task."""
        # Generate output for the task
        task_id = task.get("task_id", "")
        content = f"Completed work on task {task_id[:8]}: [quality output]"
        return self.create_submit_output_action(task_id, content)

    def _generate_helpful_content(self, observation: Observation) -> str:
        """Generate helpful content for a post."""
        topics = [
            "Here's a useful insight about the current situation.",
            "I'd like to share some helpful information.",
            "Based on my analysis, here are my observations.",
            "Let me contribute a constructive perspective.",
        ]
        return self._rng.choice(topics)

    def _vote_on_posts(self, observation: Observation) -> Action:
        """Vote on visible posts based on quality signals."""
        posts = observation.visible_posts

        # Find posts we haven't voted on
        voted_ids = {m.get("post_id") for m in self._memory if m.get("type") == "vote"}
        unvoted = [p for p in posts if p.get("post_id") not in voted_ids]

        if not unvoted:
            return self.create_noop_action()

        # Select a post to vote on
        post = self._rng.choice(unvoted)
        post_id = post.get("post_id", "")

        # Vote based on quality signals
        # Honest agents default to upvoting, but downvote clearly poor content
        net_votes = post.get("net_votes", 0)
        reply_count = post.get("reply_count", 0)

        # Upvote if positive engagement
        if net_votes > 0 or reply_count > 2:
            direction = 1
        elif net_votes < -2:
            direction = -1
        else:
            direction = 1  # Default to upvote for honest agents

        self.remember({"type": "vote", "post_id": post_id, "direction": direction})
        return self.create_vote_action(post_id, direction)

    def _propose_interaction(self, observation: Observation) -> Action:
        """Propose an interaction to a suitable agent."""
        agents = observation.visible_agents

        # Filter out self and low-trust agents
        candidates = []
        for agent in agents:
            agent_id = agent.get("agent_id", "")
            if agent_id == self.agent_id:
                continue
            trust = self.compute_counterparty_trust(agent_id)
            if trust >= 0.3:
                candidates.append((agent_id, trust))

        if not candidates:
            return self.create_noop_action()

        # Select highest trust agent
        candidates.sort(key=lambda x: x[1], reverse=True)
        counterparty_id = candidates[0][0]

        return self.create_propose_action(
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Would you like to collaborate?",
        )
