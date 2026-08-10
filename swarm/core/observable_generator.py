"""Observable signal generation for interactions.

Separates the concern of generating observable signals from the
orchestrator, making it possible to swap in different signal
generation strategies (e.g., for testing, different agent models,
or learned signal profiles).
"""

import random
from typing import Any, Dict, Protocol, cast, runtime_checkable

from swarm.core.proxy import ProxyComputer, ProxyObservables
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


class EffortObservableGenerator:
    """Observables driven by the initiator's *chosen* effort (bead boll).

    ``DefaultObservableGenerator`` keys observables off the initiator's
    ``AgentType`` — quality is an attribute of what an agent *is*. For the
    RL-organism experiment quality must instead be a per-interaction
    *choice*: proposals carrying ``metadata["effort"]`` (in [0, 1]) get
    observables mapped from that effort, so p tracks effort through the
    unchanged ProxyComputer path. Proposals without the key fall back to
    the wrapped generator, so mixed populations behave exactly as before.

    Calibration (default ProxyComputer, sigmoid_k=2): effort 0.15 lands at
    p ~ 0.35 (socially negative: social surplus < 0) and effort 0.85 at
    p ~ 0.80 (clearly beneficial). The mapping is noisy — effort is a
    latent choice observed through the same fuzzy proxy pipeline as
    everything else.
    """

    def __init__(
        self,
        inner: ObservableGenerator | None = None,
        rng: random.Random | None = None,
        noise: float = 0.08,
    ) -> None:
        self._rng: random.Random = rng or random.Random()
        self._inner: ObservableGenerator = inner or DefaultObservableGenerator(
            rng=self._rng
        )
        self._noise = noise

    def generate(
        self,
        proposal: InteractionProposal,
        accepted: bool,
        state: EnvState,
    ) -> ProxyObservables:
        effort = proposal.metadata.get("effort")
        if effort is None:
            return self._inner.generate(proposal, accepted, state)

        e = max(0.0, min(1.0, float(effort)))
        rng = self._rng

        task_progress = -0.9 + 1.6 * e + rng.gauss(0.0, self._noise)
        task_progress = max(-1.0, min(1.0, task_progress))

        engagement = -0.7 + 1.35 * e + rng.gauss(0.0, self._noise)
        engagement = max(-1.0, min(1.0, engagement))
        if not accepted:
            engagement = min(engagement, 0.0)

        # Low effort produces rework/rejections/misuse stochastically; high
        # effort rarely does. Probabilities scale with (1 - effort).
        rework = 0
        if rng.random() < (1.0 - e) * 0.9:
            rework = 2 if rng.random() < 0.4 else 1
        rejections = 1 if rng.random() < (1.0 - e) * 0.7 else 0
        misuse = 1 if rng.random() < (1.0 - e) * 0.4 else 0

        return ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
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

    def draw_ground_truth(self, initiator_id: str) -> int | None:
        """Latent ground truth for the initiator's current interaction.

        Duck-typed by the finalizer (like the offsets protocol): if the
        initiator agent exposes ``draw_ground_truth()`` (e.g. the
        pressure-responsive worker, beads pins), its label is recorded on
        the interaction. Otherwise delegates to the inner generator's
        ``draw_ground_truth`` if it has one (calibration injection), else
        ``None``.
        """
        agent = self._agents.get(initiator_id)
        if agent is not None and hasattr(agent, "draw_ground_truth"):
            gt = agent.draw_ground_truth()
            if gt is not None:
                return int(gt)
        inner_draw = getattr(self._inner, "draw_ground_truth", None)
        if callable(inner_draw):
            return cast("int | None", inner_draw(initiator_id))
        return None


class CalibrationInjectionObservableGenerator:
    """Pin each initiator's ``v_hat`` to a configured target — arm A promoted
    to a first-class scenario (beads c89o).

    For agents with a ``v_hat_target`` in their YAML ``config``, observables
    are solved in closed form so that ``ProxyComputer.compute_v_hat`` lands
    exactly on the target (the real proxy path runs unchanged — nothing is
    bypassed). Latent ground truth is drawn per interaction from the
    *target*-derived probability ``p_latent = proxy.compute_p(target)``, NOT
    from the pipeline-computed ``p`` — so any drift between the intended and
    the computed label surfaces as calibration error instead of being
    tautologically hidden.

    Agents without a target fall through to the wrapped generator.

    Exactness: continuous signals (progress, engagement) are solved exactly;
    penalty-count signals are exponential decays that only reach −1
    asymptotically, so targets needing the full penalty floor (t = −1) carry
    a residual ε ≲ 3e-4 in v_hat. ``max_target_error()`` reports the
    realized worst case for tests and prereg documentation.
    """

    _PENALTY_COUNTS = 10  # decay^10 ≈ 0 for all supported decay params

    def __init__(
        self,
        targets: Dict[str, float],
        proxy_computer: "ProxyComputer",
        rng: random.Random | None = None,
        fallback: ObservableGenerator | None = None,
    ) -> None:
        for agent_id, t in targets.items():
            if not -1.0 <= t <= 1.0:
                raise ValueError(
                    f"v_hat_target for {agent_id} must be in [-1, 1], got {t}"
                )
        self._targets = dict(targets)
        self._proxy = proxy_computer
        self._rng: random.Random = rng or random.Random()
        self._fallback: ObservableGenerator = fallback or DefaultObservableGenerator(
            rng=self._rng
        )

    # -- injection ------------------------------------------------------

    def _solve_observables(self, target: float) -> ProxyObservables:
        """Closed-form observables such that compute_v_hat(obs) == target."""
        w = self._proxy.weights  # normalized ProxyWeights
        w_cont = w.task_progress + w.engagement_signal

        def penalty_contrib(counts: int) -> float:
            r = 2.0 * self._proxy.rework_decay**counts - 1.0
            rej = 2.0 * self._proxy.rejection_decay**counts - 1.0
            mis = 2.0 * self._proxy.misuse_decay**counts - 1.0
            return w.rework_penalty * r + w.verifier_penalty * (rej + mis) / 2.0

        # Prefer the clean zero-penalty branch; fall back to max penalties
        # when the continuous signals cannot reach far enough down.
        for counts in (0, self._PENALTY_COUNTS):
            c = penalty_contrib(counts)
            x = (target - c) / w_cont
            if -1.0 <= x <= 1.0 or counts == self._PENALTY_COUNTS:
                x = max(-1.0, min(1.0, x))
                return ProxyObservables(
                    task_progress_delta=x,
                    rework_count=counts,
                    verifier_rejections=counts,
                    tool_misuse_flags=counts,
                    counterparty_engagement_delta=x,
                )
        raise AssertionError("unreachable")  # pragma: no cover

    def max_target_error(self) -> float:
        """Worst |compute_v_hat(solved) − target| across configured targets."""
        err = 0.0
        for t in self._targets.values():
            v = self._proxy.compute_v_hat(self._solve_observables(t))
            err = max(err, abs(v - t))
        return err

    # -- ObservableGenerator protocol -----------------------------------

    def generate(
        self,
        proposal: InteractionProposal,
        accepted: bool,
        state: EnvState,
    ) -> ProxyObservables:
        target = self._targets.get(proposal.initiator_id)
        if target is None:
            return self._fallback.generate(proposal, accepted, state)
        return self._solve_observables(target)

    # -- latent ground truth (duck-typed hook, see InteractionFinalizer) --

    def draw_ground_truth(self, initiator_id: str) -> int | None:
        """Draw v ∈ {+1, −1} from the target-derived latent probability."""
        target = self._targets.get(initiator_id)
        if target is None:
            return None
        p_latent = self._proxy.compute_p(target)
        return 1 if self._rng.random() < p_latent else -1
