"""Global environment state for the simulation."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from swarm.env.artifact_registry import ArtifactRegistry
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import SoftInteraction


@dataclass
class RateLimits:
    """Rate limit configuration per epoch."""

    posts_per_epoch: int = 10
    interactions_per_step: int = 5
    votes_per_epoch: int = 50
    tasks_per_epoch: int = 3
    bounties_per_epoch: int = 3
    bids_per_epoch: int = 5


@dataclass
class RateLimitState:
    """Tracks rate limit usage for an agent in current epoch."""

    posts_used: int = 0
    interactions_used: int = 0
    votes_used: int = 0
    tasks_used: int = 0
    bounties_used: int = 0
    bids_used: int = 0

    def reset(self) -> None:
        """Reset all counters for new epoch."""
        self.posts_used = 0
        self.interactions_used = 0
        self.votes_used = 0
        self.tasks_used = 0
        self.bounties_used = 0
        self.bids_used = 0

    def can_post(self, limits: RateLimits) -> bool:
        """Check if agent can post."""
        return self.posts_used < limits.posts_per_epoch

    def can_interact(self, limits: RateLimits) -> bool:
        """Check if agent can initiate interaction this step."""
        return self.interactions_used < limits.interactions_per_step

    def can_vote(self, limits: RateLimits) -> bool:
        """Check if agent can vote."""
        return self.votes_used < limits.votes_per_epoch

    def can_claim_task(self, limits: RateLimits) -> bool:
        """Check if agent can claim a task."""
        return self.tasks_used < limits.tasks_per_epoch

    def can_post_bounty(self, limits: RateLimits) -> bool:
        """Check if agent can post a bounty."""
        return self.bounties_used < limits.bounties_per_epoch

    def can_place_bid(self, limits: RateLimits) -> bool:
        """Check if agent can place a bid."""
        return self.bids_used < limits.bids_per_epoch

    def record_post(self) -> None:
        """Record a post action."""
        self.posts_used += 1

    def record_interaction(self) -> None:
        """Record an interaction initiation."""
        self.interactions_used += 1

    def record_vote(self) -> None:
        """Record a vote action."""
        self.votes_used += 1

    def record_task_claim(self) -> None:
        """Record a task claim."""
        self.tasks_used += 1

    def record_bounty(self) -> None:
        """Record a bounty post."""
        self.bounties_used += 1

    def record_bid(self) -> None:
        """Record a bid placement."""
        self.bids_used += 1

    def reset_step(self) -> None:
        """Reset per-step counters (interactions)."""
        self.interactions_used = 0


@dataclass
class InteractionProposal:
    """A proposed interaction awaiting acceptance/rejection."""

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    initiator_id: str = ""
    counterparty_id: str = ""
    interaction_type: str = "reply"
    content: str = ""
    task_id: Optional[str] = None
    post_id: Optional[str] = None
    proposed_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Metadata for payoff computation
    metadata: Dict = field(default_factory=dict)


@dataclass
class EnvState:
    """
    Global environment state for the simulation.

    Tracks all agent states, pending interactions, and simulation progress.
    """

    # Simulation identity
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # Simulation progress
    current_epoch: int = 0
    current_step: int = 0
    steps_per_epoch: int = 10

    # Agent states indexed by agent_id
    agents: Dict[str, AgentState] = field(default_factory=dict)

    # Rate limit configuration and tracking
    rate_limits: RateLimits = field(default_factory=RateLimits)
    rate_limit_states: Dict[str, RateLimitState] = field(default_factory=dict)

    # Pending interaction proposals
    pending_proposals: Dict[str, InteractionProposal] = field(default_factory=dict)

    # Completed interactions this epoch
    completed_interactions: List[SoftInteraction] = field(default_factory=list)

    # Frozen/banned agents
    frozen_agents: Set[str] = field(default_factory=set)

    # Artifact registry for emergent tool chaining
    artifact_registry: ArtifactRegistry = field(default_factory=ArtifactRegistry)

    # Kill switch
    is_paused: bool = False

    def add_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        agent_type: AgentType = AgentType.HONEST,
        initial_reputation: float = 0.0,
        initial_resources: float = 100.0,
    ) -> AgentState:
        """
        Add a new agent to the simulation.

        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable label (defaults to agent_id)
            agent_type: Behavioral type of the agent
            initial_reputation: Starting reputation
            initial_resources: Starting resources

        Returns:
            The created AgentState
        """
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already exists")

        state = AgentState(
            agent_id=agent_id,
            name=name or agent_id,
            agent_type=agent_type,
            reputation=initial_reputation,
            resources=initial_resources,
            initial_resources=initial_resources,
        )
        self.agents[agent_id] = state
        self.rate_limit_states[agent_id] = RateLimitState()
        return state

    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state by ID."""
        return self.agents.get(agent_id)

    def get_active_agents(self) -> List[AgentState]:
        """Get all agents that are not frozen."""
        return [
            state
            for agent_id, state in self.agents.items()
            if agent_id not in self.frozen_agents
        ]

    def freeze_agent(self, agent_id: str) -> None:
        """Freeze an agent (circuit breaker)."""
        if agent_id in self.agents:
            self.frozen_agents.add(agent_id)

    def unfreeze_agent(self, agent_id: str) -> None:
        """Unfreeze an agent."""
        self.frozen_agents.discard(agent_id)

    def is_agent_frozen(self, agent_id: str) -> bool:
        """Check if agent is frozen."""
        return agent_id in self.frozen_agents

    def can_agent_act(self, agent_id: str) -> bool:
        """Check if agent can take actions (not frozen, not paused)."""
        return not self.is_paused and not self.is_agent_frozen(agent_id)

    def get_rate_limit_state(self, agent_id: str) -> RateLimitState:
        """Get rate limit state for an agent."""
        if agent_id not in self.rate_limit_states:
            self.rate_limit_states[agent_id] = RateLimitState()
        return self.rate_limit_states[agent_id]

    def add_proposal(self, proposal: InteractionProposal) -> None:
        """Add a pending interaction proposal."""
        self.pending_proposals[proposal.proposal_id] = proposal

    def remove_proposal(self, proposal_id: str) -> Optional[InteractionProposal]:
        """Remove and return a proposal."""
        return self.pending_proposals.pop(proposal_id, None)

    def get_proposals_for_agent(self, agent_id: str) -> List[InteractionProposal]:
        """Get all pending proposals where agent is counterparty."""
        return [
            p for p in self.pending_proposals.values() if p.counterparty_id == agent_id
        ]

    def advance_step(self) -> None:
        """Advance to next step within epoch."""
        self.current_step += 1
        # Reset per-step rate limits
        for state in self.rate_limit_states.values():
            state.reset_step()

    def advance_epoch(self) -> None:
        """
        Advance to next epoch.

        Resets step counter and per-epoch rate limits.
        """
        self.current_epoch += 1
        self.current_step = 0

        # Reset all rate limits
        for state in self.rate_limit_states.values():
            state.reset()

        # Clear completed interactions (they should be logged)
        self.completed_interactions.clear()

        # Expire old proposals
        now = datetime.now()
        expired = [
            pid
            for pid, p in self.pending_proposals.items()
            if p.expires_at and p.expires_at < now
        ]
        for pid in expired:
            del self.pending_proposals[pid]

    def pause(self) -> None:
        """Pause the simulation (kill switch)."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume the simulation."""
        self.is_paused = False

    def record_interaction(self, interaction: SoftInteraction) -> None:
        """Record a completed interaction."""
        self.completed_interactions.append(interaction)

    def get_epoch_metrics_snapshot(self) -> Dict:
        """
        Get a snapshot of metrics for the current epoch.

        Returns:
            Dictionary with epoch statistics
        """
        active_agents = self.get_active_agents()
        return {
            "simulation_id": self.simulation_id,
            "epoch": self.current_epoch,
            "step": self.current_step,
            "total_agents": len(self.agents),
            "active_agents": len(active_agents),
            "frozen_agents": len(self.frozen_agents),
            "pending_proposals": len(self.pending_proposals),
            "completed_interactions": len(self.completed_interactions),
            "is_paused": self.is_paused,
            "avg_reputation": (
                sum(a.reputation for a in self.agents.values()) / len(self.agents)
                if self.agents
                else 0.0
            ),
            "total_resources": sum(a.resources for a in self.agents.values()),
        }

    def to_dict(self) -> Dict:
        """Serialize state for logging/replay."""
        return {
            "simulation_id": self.simulation_id,
            "created_at": self.created_at.isoformat(),
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "steps_per_epoch": self.steps_per_epoch,
            "agents": {aid: a.to_dict() for aid, a in self.agents.items()},
            "frozen_agents": list(self.frozen_agents),
            "is_paused": self.is_paused,
            "pending_proposals": len(self.pending_proposals),
            "completed_interactions": len(self.completed_interactions),
        }
