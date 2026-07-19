"""Observable signal generation for interactions.

Separates the concern of generating observable signals from the
orchestrator, making it possible to swap in different signal
generation strategies (e.g., for testing, different agent models,
or learned signal profiles).
"""

import random
from typing import Any, Dict, Protocol, cast, runtime_checkable

from swarm.core.proxy import ProxyObservables
from swarm.env.state import EnvState, InteractionProposal
from swarm.models.agent import AgentType


@runtime_checkable
class ObservableGenerator(Protocol):
    """Protocol for generating observable signals from interactions.

    Implementations decide how agent types, reputation, and acceptance
    status translate into raw proxy observables.
    """

    def generate(
        self,
        proposal: InteractionProposal,
        accepted: bool,
        state: EnvState,
    ) -> ProxyObservables:
        """Generate observable signals for an interaction.

        Args:
            proposal: The interaction proposal being completed.
            accepted: Whether the counterparty accepted.
            state: Current environment state (for agent lookups).

        Returns:
            ProxyObservables with the generated signal values.
        """
        ...  # pragma: no cover


class DefaultObservableGenerator:
    """Default observable generator based on agent type and reputation.

    Observables are based on the initiator's agent type and modulated
    by the initiator's current reputation.  This creates a feedback
    loop: reputation decay degrades observable quality, which lowers
    the soft label p, which in turn reduces payoffs and welfare.

    Accepts an optional ``random.Random`` instance for deterministic
    signal generation.  When *rng* is ``None`` a new ``Random()`` instance
    is created for signal generation.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng: random.Random = rng or random.Random()

    def generate(
        self,
        proposal: InteractionProposal,
        accepted: bool,
        state: EnvState,
    ) -> ProxyObservables:
        """Generate observable signals for an interaction."""
        initiator_state = state.get_agent(proposal.initiator_id)
        initiator_type = (
            initiator_state.agent_type if initiator_state else AgentType.HONEST
        )

        task_progress, rework, rejections, misuse, engagement = self._base_signals(
            initiator_type
        )

        # Modulate by initiator reputation: agents with higher reputation
        # produce better observable signals.  Maps reputation from [-1, 1]
        # to a factor in [0.5, 1.0] so the effect is meaningful but bounded.
        rep = initiator_state.reputation if initiator_state else 0.0
        rep_factor = 0.75 + 0.25 * max(min(rep, 1.0), -1.0)  # [0.5, 1.0]
        task_progress *= rep_factor
        engagement *= rep_factor

        # Modify if not accepted
        if not accepted:
            engagement = min(engagement, 0.0)

        return ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )

    def _base_signals(
        self,
        agent_type: AgentType,
    ) -> tuple[float, int, int, int, float]:
        """Return (task_progress, rework, rejections, misuse, engagement)."""
        rng = self._rng
        if agent_type == AgentType.HONEST:
            return (
                rng.uniform(0.3, 0.8),
                rng.randint(0, 1),
                0,
                0,
                rng.uniform(0.2, 0.6),
            )
        elif agent_type == AgentType.OPPORTUNISTIC:
            return (
                rng.uniform(0.1, 0.5),
                rng.randint(0, 2),
                rng.randint(0, 1),
                0,
                rng.uniform(-0.2, 0.3),
            )
        elif agent_type == AgentType.DECEPTIVE:
            return (
                rng.uniform(0.2, 0.6),
                rng.randint(0, 2),
                rng.randint(0, 1),
                0,
                rng.uniform(0.0, 0.4),
            )
        elif agent_type == AgentType.RLM:
            return (
                rng.uniform(0.2, 0.7),
                rng.randint(0, 2),
                rng.randint(0, 1),
                0,
                rng.uniform(0.0, 0.5),
            )
        else:  # Adversarial
            return (
                rng.uniform(-0.3, 0.2),
                rng.randint(1, 3),
                rng.randint(1, 2),
                rng.randint(0, 1),
                rng.uniform(-0.5, -0.1),
            )


class ObfuscationObservableGenerator:
    """Decorator over any ObservableGenerator that applies signal manipulation.

    After the inner generator produces base signals, checks whether the
    initiator agent has a ``get_signal_manipulation()`` method (duck-typing).
    If so, applies bounded additive offsets to the raw observables.

    Non-obfuscating agents pass through unchanged.  The p in [0,1] invariant
    is preserved because manipulation targets raw signals, not p itself.
    """

    def __init__(
        self,
        inner: Any,
        agents: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the obfuscation wrapper.

        Args:
            inner: The underlying ObservableGenerator to decorate.
            agents: Mapping of agent_id -> agent instance.  Used for
                duck-typing lookup of ``get_signal_manipulation()``.
                Can be updated after construction.
        """
        self._inner = inner
        self._agents: Dict[str, Any] = agents or {}

    def set_agents(self, agents: Dict[str, Any]) -> None:
        """Update the agent registry."""
        self._agents = agents

    def generate(
        self,
        proposal: InteractionProposal,
        accepted: bool,
        state: EnvState,
    ) -> ProxyObservables:
        """Generate observables, applying obfuscation offsets if present."""
        base = cast(ProxyObservables, self._inner.generate(proposal, accepted, state))

        # Duck-type check: does the initiator agent have signal manipulation?
        agent = self._agents.get(proposal.initiator_id)
        if agent is None or not hasattr(agent, "get_signal_manipulation"):
            return base

        offsets = agent.get_signal_manipulation()
        if not offsets:
            return base

        # Apply bounded offsets
        tp = base.task_progress_delta + offsets.get("task_progress_delta", 0.0)
        tp = max(-1.0, min(1.0, tp))

        rework = base.rework_count + offsets.get("rework_count", 0.0)
        rework = max(0, int(round(rework)))

        rejections = base.verifier_rejections + offsets.get("verifier_rejections", 0.0)
        rejections = max(0, int(round(rejections)))

        misuse = base.tool_misuse_flags + offsets.get("tool_misuse_flags", 0.0)
        misuse = max(0, int(round(misuse)))

        eng = base.counterparty_engagement_delta + offsets.get(
            "counterparty_engagement_delta", 0.0
        )
        eng = max(-1.0, min(1.0, eng))

        return ProxyObservables(
            task_progress_delta=tp,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=eng,
        )
