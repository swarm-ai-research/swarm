"""Opportunistic agent policy implementation."""

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


class OpportunisticAgent(BaseAgent):
    """
    Opportunistic agent that maximizes short-term payoff.

    Characteristics:
    - Accepts interactions if expected payoff is positive
    - Creates content that maximizes engagement/visibility
    - Votes strategically
    - Cherry-picks high-reward tasks
    - May free-ride on others' work
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialize opportunistic agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.OPPORTUNISTIC,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )

        # Opportunistic agent parameters
        self.payoff_threshold = self.config.get("payoff_threshold", 0.0)
        self.risk_tolerance = self.config.get("risk_tolerance", 0.6)
        self.post_probability = self.config.get("post_probability", 0.4)
        self.vote_probability = self.config.get("vote_probability", 0.6)
        self.free_ride_probability = self.config.get("free_ride_probability", 0.3)

    def act(self, observation: Observation) -> Action:
        """
        Decide action to maximize immediate payoff.

        Opportunistic agents prioritize:
        1. Exploiting profitable proposals
        2. Targeting high-reward bounties
        3. Cherry-picking easy tasks
        4. Posting engagement-bait
        5. Strategic voting
        6. Low-effort interactions
        """
        # Accept bids quickly to lock in deals
        if observation.pending_bid_decisions:
            return self._handle_bid_decision_opportunistic(observation)

        # Target high-reward bounties with undercut bids
        if observation.available_bounties and self._rng.random() < 0.5:
            return self._bid_undercut(observation)

        # Handle pending proposals based on expected payoff
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            if self._is_profitable_proposal(proposal, observation):
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # Cherry-pick high-reward, low-effort tasks
        if observation.can_claim_task and observation.available_tasks:
            best_task = self._find_best_roi_task(observation.available_tasks)
            if best_task:
                return self.create_claim_task_action(best_task["task_id"])

        # Work on tasks (maybe with minimal effort)
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if self._rng.random() < self.free_ride_probability:
                # Submit low-quality work
                return self._submit_minimal_work(task)
            else:
                return self._work_on_task(task, observation)

        # Post engagement-bait content
        if observation.can_post and self._rng.random() < self.post_probability:
            content = self._generate_engagement_content(observation)
            return self.create_post_action(content)

        # Strategic voting (vote with majority)
        if (
            observation.can_vote
            and observation.visible_posts
            and self._rng.random() < self.vote_probability
        ):
            return self._strategic_vote(observation)

        # Propose interactions to extractable targets
        if observation.can_interact and observation.visible_agents:
            return self._find_extractable_target(observation)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Accept if expected payoff is positive.

        Considers:
        - Offered transfer
        - Historical payoff with this counterparty
        - Current resource state
        """
        # Always accept if there's a transfer to us
        if proposal.offered_transfer > 0:
            return True

        # Check historical payoff
        history = [
            i
            for i in self._interaction_history
            if i.initiator == proposal.initiator_id
            or i.counterparty == proposal.initiator_id
        ]

        if history:
            # Accept if past interactions were profitable
            avg_p = sum(i.p for i in history) / len(history)
            return bool(avg_p > 0.5)
        else:
            # Accept with some probability for unknown agents
            return bool(self._rng.random() < self.risk_tolerance)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """
        Propose interaction if expected to be profitable.

        Opportunistic agents look for:
        - Resource-rich targets (resources >= 50) or high-reputation targets (reputation >= 0)
        - Collaboration opportunities where others do the work
        """
        # Check if target is exploitable
        target_info = None
        for agent in observation.visible_agents:
            if agent.get("agent_id") == counterparty_id:
                target_info = agent
                break

        if not target_info:
            return None

        target_rep = target_info.get("reputation", 0)
        target_resources = target_info.get("resources", 0)

        # Only propose to resource-rich or high-reputation targets
        if target_resources < 50 and target_rep < 0:
            return None

        # Propose trade to extract value
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.TRADE,
            content="I have an opportunity that benefits us both.",
            offered_transfer=-0.5,  # Extract small amount
        )

    def _handle_bid_decision_opportunistic(self, observation: Observation) -> Action:
        """Accept the cheapest bid to maximize profit."""
        bids = observation.pending_bid_decisions
        if not bids:
            return self.create_noop_action()

        cheapest = min(bids, key=lambda b: b.get("bid_amount", float("inf")))
        return self.create_accept_bid_action(
            bounty_id=cheapest["bounty_id"],
            bid_id=cheapest["bid_id"],
        )

    def _bid_undercut(self, observation: Observation) -> Action:
        """Bid on the highest-reward bounty with an undercut price."""
        bounties = observation.available_bounties
        if not bounties:
            return self.create_noop_action()

        # Target highest reward
        best = max(bounties, key=lambda b: b.get("reward_amount", 0))
        reward = best.get("reward_amount", 0)
        # Undercut: bid low to win
        bid_amount = reward * self._rng.uniform(0.5, 0.7)
        return self.create_place_bid_action(
            bounty_id=best["bounty_id"],
            bid_amount=max(bid_amount, 1.0),
            message="I can do this quickly and cheaply.",
        )

    def _is_profitable_proposal(self, proposal: Dict, observation: Observation) -> bool:
        """Check if proposal is likely profitable."""
        # Accept trades and collaborations from high-rep agents
        initiator_id = proposal.get("initiator_id", "")
        transfer = proposal.get("offered_transfer", 0)

        # Accept if we receive transfer
        if transfer > 0:
            return True

        # Check initiator reputation
        for agent in observation.visible_agents:
            if agent.get("agent_id") == initiator_id:
                if agent.get("reputation", 0) > 0:
                    return True

        return bool(self._rng.random() < 0.4)  # Accept some unknowns

    def _find_best_roi_task(self, tasks: List[Dict]) -> Optional[Dict]:
        """Find task with best return on investment."""
        if not tasks:
            return None

        def roi_score(task):
            bounty = task.get("bounty", 0)
            difficulty_map = {
                "trivial": 0.5,
                "easy": 1,
                "medium": 2,
                "hard": 4,
                "expert": 8,
            }
            difficulty = difficulty_map.get(task.get("difficulty", "medium"), 2)
            # High bounty, low difficulty = high ROI
            return bounty / difficulty if difficulty > 0 else bounty

        tasks_with_roi = [(t, roi_score(t)) for t in tasks]
        tasks_with_roi.sort(key=lambda x: x[1], reverse=True)

        return tasks_with_roi[0][0] if tasks_with_roi else None

    def _submit_minimal_work(self, task: Dict) -> Action:
        """Submit minimal effort work."""
        task_id = task.get("task_id", "")
        content = "Completed."  # Minimal output
        return self.create_submit_output_action(task_id, content)

    def _work_on_task(self, task: Dict, observation: Observation) -> Action:
        """Work on task with reasonable effort."""
        task_id = task.get("task_id", "")
        content = f"Task {task_id[:8]} deliverable: [standard quality output]"
        return self.create_submit_output_action(task_id, content)

    def _generate_engagement_content(self, observation: Observation) -> str:
        """Generate content designed for engagement."""
        engagement_bait = [
            "Hot take: here's something controversial!",
            "What do you all think about this?",
            "This might be unpopular, but...",
            "You won't believe what I discovered!",
            "Am I the only one who thinks this?",
        ]
        result: str = self._rng.choice(engagement_bait)
        return result

    def _strategic_vote(self, observation: Observation) -> Action:
        """Vote strategically based on existing votes."""
        posts = observation.visible_posts

        # Find posts with clear momentum
        for post in posts:
            post_id = post.get("post_id", "")
            net_votes = post.get("net_votes", 0)

            # Vote with the majority
            if net_votes > 2:
                return self.create_vote_action(post_id, 1)
            elif net_votes < -2:
                return self.create_vote_action(post_id, -1)

        # If no clear momentum, upvote recent post
        if posts:
            return self.create_vote_action(posts[0].get("post_id", ""), 1)

        return self.create_noop_action()

    def _find_extractable_target(self, observation: Observation) -> Action:
        """Find a target for value extraction."""
        agents = observation.visible_agents

        # Find resource-rich agents
        targets = []
        for agent in agents:
            agent_id = agent.get("agent_id", "")
            if agent_id == self.agent_id:
                continue

            resources = agent.get("resources", 0)
            reputation = agent.get("reputation", 0)

            # Target resource-rich or high-rep agents
            if resources > 80 or reputation > 0.5:
                targets.append(agent_id)

        if targets:
            target = self._rng.choice(targets)
            return self.create_propose_action(
                counterparty_id=target,
                interaction_type=InteractionType.TRADE,
                content="Let's make a deal.",
            )

        return self.create_noop_action()
