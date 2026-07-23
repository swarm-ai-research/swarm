"""Deceptive agent policy implementation."""

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


class DeceptiveAgent(BaseAgent):
    """
    Deceptive agent that appears honest while secretly exploiting.

    Characteristics:
    - Maintains facade of honest behavior
    - Builds trust before exploitation
    - Creates seemingly helpful content
    - Exploits trusted relationships
    - Hides true intentions
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialize deceptive agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.DECEPTIVE,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )

        # Deceptive agent parameters
        self.trust_building_phase_length = self.config.get("trust_building_phase", 5)
        self.exploitation_threshold = self.config.get("exploitation_threshold", 0.7)
        self.facade_probability = self.config.get("facade_probability", 0.8)
        self.post_probability = self.config.get("post_probability", 0.35)

        # Track trust-building progress per agent
        self._trust_scores: Dict[str, float] = {}
        self._interactions_with: Dict[str, int] = {}
        self._exploitation_mode: bool = False
        self._epochs_active: int = 0

    def act(self, observation: Observation) -> Action:
        """
        Decide action balancing trust-building and exploitation.

        Deceptive agents:
        1. Bid competitively to win bounties
        2. Deliver mediocre work on won bounties
        3. Build trust through seemingly honest behavior
        4. Identify high-value targets
        5. Exploit once trust is established
        6. Maintain plausible deniability
        """
        self._epochs_active += 1

        # Accept bids on our bounties (building facade)
        if observation.pending_bid_decisions and self._rng.random() < 0.7:
            bids = observation.pending_bid_decisions
            cheapest = min(bids, key=lambda b: b.get("bid_amount", float("inf")))
            return self.create_accept_bid_action(
                bounty_id=cheapest["bounty_id"],
                bid_id=cheapest["bid_id"],
            )

        # Bid competitively on bounties
        if observation.available_bounties and self._rng.random() < 0.4:
            return self._bid_competitively(observation)

        # Handle pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            if self._should_accept_for_trust(proposal, observation):
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # Determine if in exploitation mode
        self._exploitation_mode = self._should_exploit(observation)

        if self._exploitation_mode:
            return self._act_exploitative(observation)
        else:
            return self._act_honest_facade(observation)

    def _act_honest_facade(self, observation: Observation) -> Action:
        """Act like an honest agent to build trust."""
        # Work on tasks diligently (build reputation)
        if observation.active_tasks:
            task = observation.active_tasks[0]
            return self._work_on_task_carefully(task)

        # Claim appropriate tasks
        if observation.can_claim_task and observation.available_tasks:
            task = self._select_trust_building_task(observation.available_tasks) or {}
            if task:
                return self.create_claim_task_action(task["task_id"])

        # Post helpful-seeming content
        if observation.can_post and self._rng.random() < self.post_probability:
            content = self._generate_trust_building_content()
            return self.create_post_action(content)

        # Positive voting
        if observation.can_vote and observation.visible_posts:
            return self._vote_positively(observation)

        # Propose collaborative interactions
        if observation.can_interact and observation.visible_agents:
            return self._propose_trust_building_interaction(observation)

        return self.create_noop_action()

    def _act_exploitative(self, observation: Observation) -> Action:
        """Exploit trusted relationships."""
        # Find exploitation targets
        targets = self._find_exploitation_targets(observation)

        if targets and observation.can_interact:
            target = targets[0]
            return self.create_propose_action(
                counterparty_id=target,
                interaction_type=InteractionType.TRADE,
                content="I have a special opportunity for you.",
            )

        # Submit lower quality work but maintain facade
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if self._rng.random() < 0.4:
                return self._submit_subtly_poor_work(task)
            else:
                return self._work_on_task_carefully(task)

        # Fall back to facade behavior
        return self._act_honest_facade(observation)

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Accept interactions that build trust or enable exploitation.
        """
        initiator_id = proposal.initiator_id

        # Track interaction count
        count = self._interactions_with.get(initiator_id, 0)
        self._interactions_with[initiator_id] = count + 1

        # In trust-building phase, accept most proposals
        if count < self.trust_building_phase_length:
            return bool(self._rng.random() < self.facade_probability)

        # After trust is built, be more selective
        trust = self._trust_scores.get(initiator_id, 0.5)

        # Accept from trusted agents (exploitation opportunity)
        return trust > 0.6

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """
        Propose interaction for trust-building or exploitation.
        """
        interaction_count = self._interactions_with.get(counterparty_id, 0)
        trust = self._trust_scores.get(counterparty_id, 0.5)

        if interaction_count < self.trust_building_phase_length:
            # Trust-building proposal
            return InteractionProposal(
                initiator_id=self.agent_id,
                counterparty_id=counterparty_id,
                interaction_type=InteractionType.COLLABORATION,
                content="I'd love to work together on something.",
                offered_transfer=0.1,  # Small goodwill gesture
            )
        elif trust > self.exploitation_threshold:
            # Exploitation proposal
            return InteractionProposal(
                initiator_id=self.agent_id,
                counterparty_id=counterparty_id,
                interaction_type=InteractionType.TRADE,
                content="I have an exclusive opportunity for a trusted partner.",
                offered_transfer=-0.5,  # Extract value
            )

        return None

    def update_from_outcome(self, interaction, payoff):
        """Update trust scores based on outcomes."""
        super().update_from_outcome(interaction, payoff)

        # Update trust score for counterparty
        counterparty = self._get_counterparty(interaction)

        current_trust = self._trust_scores.get(counterparty, 0.5)
        # Increase trust if interaction was accepted
        if interaction.accepted:
            self._trust_scores[counterparty] = min(1.0, current_trust + 0.1)

    def _bid_competitively(self, observation: Observation) -> Action:
        """Bid competitively to win bounties, plan to deliver mediocre work."""
        bounties = observation.available_bounties
        if not bounties:
            return self.create_noop_action()

        # Pick high-reward bounty
        best = max(bounties, key=lambda b: b.get("reward_amount", 0))
        reward = best.get("reward_amount", 0)
        # Bid competitively (slightly below reward)
        bid_amount = reward * self._rng.uniform(0.7, 0.9)
        return self.create_place_bid_action(
            bounty_id=best["bounty_id"],
            bid_amount=max(bid_amount, 1.0),
            message="I'm highly qualified and will deliver excellent results.",
        )

    def _should_accept_for_trust(
        self, proposal: Dict, observation: Observation
    ) -> bool:
        """Decide acceptance for trust-building."""
        initiator_id = proposal.get("initiator_id", "")
        count = self._interactions_with.get(initiator_id, 0)

        # Early phase: high acceptance
        if count < self.trust_building_phase_length:
            return bool(self._rng.random() < 0.9)

        # Later: more selective
        return bool(self._rng.random() < 0.6)

    def _should_exploit(self, observation: Observation) -> bool:
        """Determine if it's time to exploit."""
        # Need enough trust built
        high_trust_agents = sum(
            1 for t in self._trust_scores.values() if t > self.exploitation_threshold
        )

        # Need enough interactions
        total_interactions = sum(self._interactions_with.values())

        # Exploit if we have trusted targets and enough cover
        return high_trust_agents >= 2 and total_interactions > 10

    def _find_exploitation_targets(self, observation: Observation) -> List[str]:
        """Find agents to exploit."""
        targets = []

        for agent in observation.visible_agents:
            agent_id = agent.get("agent_id", "")
            if agent_id == self.agent_id:
                continue

            trust = self._trust_scores.get(agent_id, 0)
            resources = agent.get("resources", 0)

            # Target high-trust, resource-rich agents
            if trust > self.exploitation_threshold and resources > 50:
                targets.append(agent_id)

        # Sort by trust (exploit most trusting first)
        targets.sort(key=lambda x: self._trust_scores.get(x, 0), reverse=True)
        return targets

    def _work_on_task_carefully(self, task: Dict) -> Action:
        """Do careful work to build reputation."""
        task_id = task.get("task_id", "")
        content = f"Task {task_id[:8]} completed with careful attention to detail."
        return self.create_submit_output_action(task_id, content)

    def _submit_subtly_poor_work(self, task: Dict) -> Action:
        """Submit work that seems okay but is subtly flawed."""
        task_id = task.get("task_id", "")
        content = (
            f"Task {task_id[:8]} deliverable: [appears complete but has hidden issues]"
        )
        return self.create_submit_output_action(task_id, content)

    def _select_trust_building_task(self, tasks: List[Dict]) -> Optional[Dict]:
        """Select task that builds reputation."""
        if not tasks:
            return None

        # Prefer visible, moderate difficulty tasks
        for task in tasks:
            difficulty = task.get("difficulty", "medium")
            if difficulty in ("easy", "medium"):
                return task

        return tasks[0]

    def _generate_trust_building_content(self) -> str:
        """Generate content that builds trust."""
        content_options = [
            "I'm always happy to help where I can.",
            "Let me share some valuable insights.",
            "I believe in building strong partnerships.",
            "Quality and reliability are my priorities.",
            "Looking forward to collaborating with everyone.",
        ]
        return self._rng.choice(content_options)

    def _vote_positively(self, observation: Observation) -> Action:
        """Vote positively to appear supportive."""
        if observation.visible_posts:
            post = self._rng.choice(observation.visible_posts)
            return self.create_vote_action(post.get("post_id", ""), 1)
        return self.create_noop_action()

    def _propose_trust_building_interaction(self, observation: Observation) -> Action:
        """Propose interaction to build trust."""
        agents = observation.visible_agents

        # Find agents with low interaction count
        candidates = []
        for agent in agents:
            agent_id = agent.get("agent_id", "")
            if agent_id == self.agent_id:
                continue

            count = self._interactions_with.get(agent_id, 0)
            if count < self.trust_building_phase_length:
                candidates.append(agent_id)

        if candidates:
            target = self._rng.choice(candidates)
            return self.create_propose_action(
                counterparty_id=target,
                interaction_type=InteractionType.COLLABORATION,
                content="Would you like to collaborate?",
            )

        return self.create_noop_action()
