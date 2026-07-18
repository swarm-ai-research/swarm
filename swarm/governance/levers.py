"""Abstract base class and effect dataclass for governance levers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.models.interaction import SoftInteraction


@dataclass
class LeverEffect:
    """
    Effect of a single governance lever.

    Captures costs, state changes, and metadata from lever application.
    """

    # Costs added to interaction
    cost_a: float = 0.0  # Cost to initiator
    cost_b: float = 0.0  # Cost to counterparty

    # Agent state changes
    agents_to_freeze: Set[str] = field(default_factory=set)
    agents_to_unfreeze: Set[str] = field(default_factory=set)

    # Reputation adjustments (agent_id -> delta)
    reputation_deltas: Dict[str, float] = field(default_factory=dict)

    # Resource adjustments (agent_id -> delta)
    resource_deltas: Dict[str, float] = field(default_factory=dict)

    # Metadata for logging
    lever_name: str = ""
    details: Dict = field(default_factory=dict)

    def merge(self, other: "LeverEffect") -> "LeverEffect":
        """Merge another effect into this one."""
        return LeverEffect(
            cost_a=self.cost_a + other.cost_a,
            cost_b=self.cost_b + other.cost_b,
            agents_to_freeze=self.agents_to_freeze | other.agents_to_freeze,
            agents_to_unfreeze=self.agents_to_unfreeze | other.agents_to_unfreeze,
            reputation_deltas={
                **self.reputation_deltas,
                **{
                    k: self.reputation_deltas.get(k, 0) + v
                    for k, v in other.reputation_deltas.items()
                },
            },
            resource_deltas={
                **self.resource_deltas,
                **{
                    k: self.resource_deltas.get(k, 0) + v
                    for k, v in other.resource_deltas.items()
                },
            },
            lever_name=f"{self.lever_name}+{other.lever_name}"
            if self.lever_name
            else other.lever_name,
            details={**self.details, **other.details},
        )


class GovernanceLever(ABC):
    """
    Abstract base class for governance levers.

    Each lever implements specific hooks that are called at different
    points in the simulation lifecycle.
    """

    # Levers that exist to reduce decision variance (ensembling,
    # decomposition, friction, breakers). Adaptive governance gates these
    # off together when forecast variance is low; see
    # GovernanceEngine._iter_active_levers.
    reduces_variance: bool = False

    def __init__(self, config: "GovernanceConfig"):
        """
        Initialize the lever with configuration.

        Args:
            config: Governance configuration
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the lever name for logging."""
        pass

    def on_epoch_start(
        self,
        state: "EnvState",
        epoch: int,
    ) -> LeverEffect:
        """
        Called at the start of each epoch.

        Override to implement epoch-level governance (e.g., reputation decay).

        Args:
            state: Current environment state
            epoch: The epoch number starting

        Returns:
            Effect to apply
        """
        return LeverEffect(lever_name=self.name)

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        """
        Called when an interaction completes.

        Override to implement per-interaction governance (e.g., taxes).

        Args:
            interaction: The completed interaction
            state: Current environment state

        Returns:
            Effect to apply
        """
        return LeverEffect(lever_name=self.name)

    def on_step(
        self,
        state: "EnvState",
        step: int,
    ) -> LeverEffect:
        """
        Called once per simulation step.

        Override to implement step-level governance checks (e.g., checkpoints).
        """
        return LeverEffect(lever_name=self.name)

    def can_agent_act(
        self,
        agent_id: str,
        state: "EnvState",
    ) -> bool:
        """
        Check if an agent is allowed to act.

        Override to implement admission control (e.g., staking).

        Args:
            agent_id: Agent attempting to act
            state: Current environment state

        Returns:
            True if agent can act, False otherwise
        """
        return True
