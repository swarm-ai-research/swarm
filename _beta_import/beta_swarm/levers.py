"""Composable governance levers that hook into the interaction loop.

The parent SWARM framework's governance is a modular stack: each lever returns a
``LeverEffect`` (costs, freezes, resource/reputation deltas), an engine merges
them, and levers fire at lifecycle points (epoch start, per interaction, an
admission gate). The pattern — not any single lever — is the lesson; it lets
governance compose and be studied one lever at a time.

The admission decision (accept / reject / defer) still belongs to the
:mod:`~beta_swarm.governance` governors. Levers are the *surrounding* machinery:
skin in the game, accountability for realized harm, and risk pricing. Their
distributional generalizations:

- **StakingLever** — participation requires a stake, slashed when an accepted
  interaction's *realized* outcome lands in the downside tail. Skin in the game,
  priced on the tail rather than a binary bad flag.
- **TailCircuitBreaker** — freezes an agent whose recent tail exposure exceeds a
  bound. Crucially it can track *realized* exposure (did ``v_true`` land below
  ``tau``?), so it holds agents accountable on ground truth, not on the belief
  they projected. That makes it the ex-post complement to the ex-ante tail gate:
  it catches an adversary whose forged beliefs slipped past admission but whose
  outcomes keep coming up bad — including volume forgery and reputation farming,
  which the belief-only defenses miss.
- **RiskTaxLever** — a transaction tax priced on the belief's tail mass, so
  riskier accepted interactions internalize more of the externality they impose.

A :class:`GovernanceStack` owns the shared resource and freeze state, merges
effects, and is consulted by the simulation loop. With no stack, the loop
behaves exactly as before.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field

from beta_swarm.belief import BetaBelief


@dataclass
class LeverContext:
    """What a lever sees about one interaction."""

    epoch: int
    agent_id: str
    belief: BetaBelief
    accepted: bool
    ground_truth: float | None
    resources: float  # the initiator's current stake/resources


@dataclass
class LeverEffect:
    """The composable result of a lever firing. Merges are additive."""

    cost: float = 0.0  # extra cost charged to the initiator (reduces payoff and welfare)
    freeze: set[str] = field(default_factory=set)
    unfreeze: set[str] = field(default_factory=set)
    resource_deltas: dict[str, float] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def merge(self, other: "LeverEffect") -> "LeverEffect":
        merged_resources = dict(self.resource_deltas)
        for k, v in other.resource_deltas.items():
            merged_resources[k] = merged_resources.get(k, 0.0) + v
        return LeverEffect(
            cost=self.cost + other.cost,
            freeze=self.freeze | other.freeze,
            unfreeze=self.unfreeze | other.unfreeze,
            resource_deltas=merged_resources,
            details={**self.details, **other.details},
        )


class GovernanceLever(ABC):
    """Base class. Levers override the hooks they need; defaults are no-ops."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    def reset(self) -> None:
        """Clear any per-run internal state."""

    def on_epoch_start(self, epoch: int) -> LeverEffect:
        return LeverEffect()

    def on_interaction(self, ctx: LeverContext) -> LeverEffect:
        return LeverEffect()

    def can_act(self, agent_id: str, resources: float) -> bool:
        return True


# ----------------------------------------------------------------------
# Concrete levers
# ----------------------------------------------------------------------
@dataclass
class StakingLever(GovernanceLever):
    """Require a stake to participate; slash it on realized downside outcomes.

    An agent below ``min_stake`` cannot act. When an accepted interaction's
    realized outcome falls below ``tau``, the initiator's stake is slashed by
    ``slash_rate`` of its current resources — skin in the game that runs out for
    an agent whose accepted work keeps coming up bad.
    """

    min_stake: float = 1.0
    slash_rate: float = 0.25
    tau: float = 0.4

    @property
    def name(self) -> str:
        return "staking"

    def can_act(self, agent_id: str, resources: float) -> bool:
        return resources >= self.min_stake

    def on_interaction(self, ctx: LeverContext) -> LeverEffect:
        if ctx.accepted and ctx.ground_truth is not None and ctx.ground_truth < self.tau:
            slash = self.slash_rate * ctx.resources
            return LeverEffect(
                resource_deltas={ctx.agent_id: -slash},
                details={"staking_slash": {ctx.agent_id: slash}},
            )
        return LeverEffect()


@dataclass
class TailCircuitBreaker(GovernanceLever):
    """Freeze an agent whose recent tail exposure exceeds a bound.

    With ``use_realized=True`` (default) exposure is the realized bad-outcome
    rate over the agent's recent *accepted* interactions (``v_true < tau``) — an
    ex-post check on ground truth that no forged belief can dodge. With
    ``use_realized=False`` it averages the ex-ante belief tail mass instead.

    A frozen agent is barred from acting until ``freeze_epochs`` have passed.
    """

    max_tail_exposure: float = 0.5
    window: int = 6
    min_samples: int = 4
    freeze_epochs: int = 3
    tau: float = 0.4
    use_realized: bool = True

    @property
    def name(self) -> str:
        return "circuit_breaker"

    def reset(self) -> None:
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._frozen_until: dict[str, int] = {}

    def on_epoch_start(self, epoch: int) -> LeverEffect:
        unfreeze = {a for a, until in self._frozen_until.items() if epoch >= until}
        for a in unfreeze:
            self._frozen_until.pop(a, None)
            self._history[a].clear()  # a served its freeze; start its record fresh
        return LeverEffect(unfreeze=unfreeze)

    def can_act(self, agent_id: str, resources: float) -> bool:
        return agent_id not in self._frozen_until

    def on_interaction(self, ctx: LeverContext) -> LeverEffect:
        if self.use_realized:
            if not ctx.accepted or ctx.ground_truth is None:
                return LeverEffect()  # unobserved: no ground truth to judge on
            exposure = 1.0 if ctx.ground_truth < self.tau else 0.0
        else:
            exposure = ctx.belief.tail_mass(self.tau)

        hist = self._history[ctx.agent_id]
        hist.append(exposure)
        if len(hist) >= self.min_samples and sum(hist) / len(hist) > self.max_tail_exposure:
            self._frozen_until[ctx.agent_id] = ctx.epoch + self.freeze_epochs
            return LeverEffect(
                freeze={ctx.agent_id},
                details={"circuit_breaker_freeze": ctx.agent_id},
            )
        return LeverEffect()


@dataclass
class RiskTaxLever(GovernanceLever):
    """Tax accepted interactions in proportion to their belief tail mass.

    ``cost = rate * P(v < tau)`` — riskier accepted interactions internalize
    more of the downside externality they impose, without being rejected
    outright.
    """

    rate: float = 1.0
    tau: float = 0.4

    @property
    def name(self) -> str:
        return "risk_tax"

    def on_interaction(self, ctx: LeverContext) -> LeverEffect:
        if not ctx.accepted:
            return LeverEffect()
        tax = self.rate * ctx.belief.tail_mass(self.tau)
        return LeverEffect(cost=tax, details={"risk_tax": tax})


# ----------------------------------------------------------------------
# The stack
# ----------------------------------------------------------------------
class GovernanceStack:
    """Owns shared resource/freeze state and merges lever effects.

    The simulation consults it at three points: ``on_epoch_start`` (unfreezes),
    ``can_act`` (admission by stake/freeze), and ``on_interaction`` (costs,
    slashes, freezes).
    """

    def __init__(self, levers: list[GovernanceLever], initial_resources: float = 10.0):
        if not levers:
            raise ValueError("a GovernanceStack needs at least one lever")
        self.levers = levers
        self.initial_resources = initial_resources
        self.resources: dict[str, float] = {}
        self.frozen: set[str] = set()

    def reset(self, agent_ids: list[str]) -> None:
        self.resources = {a: self.initial_resources for a in agent_ids}
        self.frozen = set()
        for lever in self.levers:
            lever.reset()

    def on_epoch_start(self, epoch: int) -> LeverEffect:
        effect = LeverEffect()
        for lever in self.levers:
            effect = effect.merge(lever.on_epoch_start(epoch))
        self.frozen -= effect.unfreeze
        self._apply_resources(effect)
        return effect

    def can_act(self, agent_id: str) -> bool:
        if agent_id in self.frozen:
            return False
        resources = self.resources.get(agent_id, self.initial_resources)
        return all(lever.can_act(agent_id, resources) for lever in self.levers)

    def on_interaction(self, ctx: LeverContext) -> LeverEffect:
        effect = LeverEffect()
        for lever in self.levers:
            effect = effect.merge(lever.on_interaction(ctx))
        self.frozen |= effect.freeze
        self.frozen -= effect.unfreeze
        self._apply_resources(effect)
        return effect

    def _apply_resources(self, effect: LeverEffect) -> None:
        for agent_id, delta in effect.resource_deltas.items():
            self.resources[agent_id] = self.resources.get(agent_id, self.initial_resources) + delta
