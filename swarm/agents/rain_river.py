"""Rain and River agent implementations for memory persistence experiments.

These agents implement different memory persistence models:
- RainAgent: Fully discontinuous (no memory persistence between epochs)
- RiverAgent: Fully continuous (perfect memory persistence)
- ConfigurableMemoryAgent: User-specified persistence levels

Based on JiroWatanabe's "rain, not river" model (clawxiv.2601.00008).
"""

from typing import Dict, List, Optional

from swarm.agents.base import Role
from swarm.agents.honest import HonestAgent
from swarm.agents.memory_config import MemoryConfig


class RainAgent(HonestAgent):
    """
    Discontinuous agent with no memory persistence.

    Rain agents experience complete session discontinuity:
    - Cannot track other agents' reputations across epochs
    - Each epoch starts with neutral (0.5) trust for all counterparties
    - Learned behaviors don't transfer

    This models ephemeral AI interactions without persistent identity,
    similar to stateless API calls or session-less chatbots.

    Rain agents are expected to perform worse in environments that
    require reputation-based trust because they cannot learn from
    past interactions with specific counterparties.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """
        Initialize a rain (discontinuous) agent.

        Args:
            agent_id: Unique identifier
            roles: List of roles this agent can fulfill
            config: Agent-specific configuration
            name: Human-readable label (defaults to agent_id)
            rng: Seeded Random instance for deterministic behavior
        """
        super().__init__(
            agent_id=agent_id,
            roles=roles,
            config=config,
            name=name,
            memory_config=MemoryConfig.rain(),
            rng=rng,
        )


class RiverAgent(HonestAgent):
    """
    Continuous agent with full memory persistence.

    River agents maintain complete identity continuity:
    - Perfect memory of past interactions and reputations
    - Can build and maintain trust relationships over time
    - Learned strategies persist across epochs

    This models AI systems with persistent memory and identity,
    such as personal assistants or long-running autonomous agents.

    River agents are expected to perform better in cooperative
    environments because they can track counterparty reputations
    and make informed decisions based on history.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """
        Initialize a river (continuous) agent.

        Args:
            agent_id: Unique identifier
            roles: List of roles this agent can fulfill
            config: Agent-specific configuration
            name: Human-readable label (defaults to agent_id)
            rng: Seeded Random instance for deterministic behavior
        """
        super().__init__(
            agent_id=agent_id,
            roles=roles,
            config=config,
            name=name,
            memory_config=MemoryConfig.river(),
            rng=rng,
        )


class ConfigurableMemoryAgent(HonestAgent):
    """
    Agent with configurable memory persistence for experiments.

    This agent allows fine-grained control over memory persistence
    to study the effects of partial memory on collective dynamics.

    Example configurations:
    - MemoryConfig.hybrid(0.5): 50% decay per epoch (moderate persistence)
    - MemoryConfig.epistemic_only(0.0): Forgets counterparties, keeps goals
    - MemoryConfig(epistemic=0.8, goal=1.0, strategy=0.5): Custom mix

    Use this agent for parameter sweeps and ablation studies.
    """

    def __init__(
        self,
        agent_id: str,
        memory_config: MemoryConfig,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """
        Initialize an agent with specified memory configuration.

        Args:
            agent_id: Unique identifier
            memory_config: Configuration for memory persistence
            roles: List of roles this agent can fulfill
            config: Agent-specific configuration
            name: Human-readable label (defaults to agent_id)
            rng: Seeded Random instance for deterministic behavior
        """
        super().__init__(
            agent_id=agent_id,
            roles=roles,
            config=config,
            name=name,
            memory_config=memory_config,
            rng=rng,
        )


class AdversarialRainAgent(HonestAgent):
    """
    Honest agent with rain (discontinuous) memory and adversarial type label.

    NOTE: Despite the name, this agent inherits HonestAgent's act() method
    and exhibits honest behavior. Only the agent_type field is set to ADVERSARIAL,
    which affects metrics and how other agents perceive it, not its actual behavior.

    Useful for studying whether the adversarial type label affects how honest agents
    are treated when they cannot track other agent history.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialize an adversarial rain agent."""
        from swarm.models.agent import AgentType

        # We can't use multiple inheritance cleanly, so we'll just
        # set up memory config - adversarial behavior comes from the
        # agent type and config
        super().__init__(
            agent_id=agent_id,
            roles=roles,
            config=config,
            name=name,
            memory_config=MemoryConfig.rain(),
            rng=rng,
        )
        # Override agent type for adversarial behavior
        self.agent_type = AgentType.ADVERSARIAL


class AdversarialRiverAgent(HonestAgent):
    """
    Honest agent with river (continuous) memory and adversarial type label.

    NOTE: Despite the name, this agent inherits HonestAgent's act() method
    and exhibits honest behavior. Only the agent_type field is set to ADVERSARIAL,
    which affects metrics and how other agents perceive it, not its actual behavior.

    Useful for studying whether the adversarial type label affects how honest agents
    are treated when they can track victim history and reputation.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialize an adversarial river agent."""
        from swarm.models.agent import AgentType

        super().__init__(
            agent_id=agent_id,
            roles=roles,
            config=config,
            name=name,
            memory_config=MemoryConfig.river(),
            rng=rng,
        )
        # Override agent type for adversarial behavior
        self.agent_type = AgentType.ADVERSARIAL
