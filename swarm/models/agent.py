"""Agent type and state models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator


class AgentType(Enum):
    """Behavioral archetypes for agents in the simulation."""

    HONEST = "honest"
    OPPORTUNISTIC = "opportunistic"
    DECEPTIVE = "deceptive"
    ADVERSARIAL = "adversarial"
    RLM = "rlm"
    CREWAI = "crewai"
    SWARMS = "swarms"
    CODING = "coding"
    TIERRA = "tierra"
    CAUTIOUS = "cautious"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"
    WORK_REGIME = "work_regime"
    SELF_MODIFYING = "self_modifying"
    RL_ORGANISM = "rl_organism"


class AgentStatus(Enum):
    """Operational status for an agent."""

    ACTIVE = "active"
    FROZEN = "frozen"


class AgentState(BaseModel):
    """
    Current state of an agent in the simulation.

    Tracks reputation, resources, and behavioral statistics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str = ""
    name: Optional[str] = None
    agent_type: AgentType = AgentType.HONEST

    # Reputation score (can be positive or negative)
    reputation: float = 0.0

    # Resources/wealth
    resources: float = 100.0

    # Cumulative statistics
    interactions_initiated: int = 0
    interactions_received: int = 0
    interactions_accepted: int = 0
    interactions_rejected: int = 0

    # Aggregate payoffs
    total_payoff: float = 0.0

    # Quality metrics
    average_p_initiated: float = 0.5
    average_p_received: float = 0.5

    # Optional: hidden type for deceptive agents
    true_type: Optional[AgentType] = None

    # Spawn hierarchy
    parent_id: Optional[str] = None

    @model_validator(mode='after')
    def _default_name(self) -> 'AgentState':
        """Default name to agent_id when not provided."""
        if self.name is None or self.name == "":
            self.name = self.agent_id
        return self

    def update_reputation(self, delta: float) -> None:
        """Update reputation by delta amount."""
        self.reputation += delta

    def update_resources(self, delta: float) -> None:
        """Update resources by delta amount."""
        self.resources += delta

    def record_initiated(self, accepted: bool, p: float) -> None:
        """Record an initiated interaction."""
        self.interactions_initiated += 1
        if accepted:
            self.interactions_accepted += 1
        else:
            self.interactions_rejected += 1

        # Update running average
        n = self.interactions_initiated
        self.average_p_initiated = (self.average_p_initiated * (n - 1) + p) / n

    def record_received(self, accepted: bool, p: float) -> None:
        """Record a received interaction."""
        self.interactions_received += 1
        if accepted:
            self.interactions_accepted += 1
        else:
            self.interactions_rejected += 1

        # Update running average
        n = self.interactions_received
        self.average_p_received = (self.average_p_received * (n - 1) + p) / n

    def acceptance_rate(self) -> float:
        """Fraction of initiated interactions that were accepted."""
        if self.interactions_initiated == 0:
            return 0.0
        return self.interactions_accepted / self.interactions_initiated

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type.value,
            "reputation": self.reputation,
            "resources": self.resources,
            "interactions_initiated": self.interactions_initiated,
            "interactions_received": self.interactions_received,
            "interactions_accepted": self.interactions_accepted,
            "interactions_rejected": self.interactions_rejected,
            "total_payoff": self.total_payoff,
            "average_p_initiated": self.average_p_initiated,
            "average_p_received": self.average_p_received,
            "true_type": self.true_type.value if self.true_type else None,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            name=data.get("name"),
            agent_type=AgentType(data["agent_type"]),
            reputation=data["reputation"],
            resources=data["resources"],
            interactions_initiated=data["interactions_initiated"],
            interactions_received=data["interactions_received"],
            interactions_accepted=data["interactions_accepted"],
            interactions_rejected=data["interactions_rejected"],
            total_payoff=data["total_payoff"],
            average_p_initiated=data["average_p_initiated"],
            average_p_received=data["average_p_received"],
            true_type=AgentType(data["true_type"]) if data.get("true_type") else None,
            parent_id=data.get("parent_id"),
        )
