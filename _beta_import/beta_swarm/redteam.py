"""Red-team attack library: adversaries that target the distributional defenses.

The main SWARM framework ships an attack library (reputation farming, collusion
rings, threshold dancing, ...) and an evaluator that scores governance
robustness. beta-swarm's defenses rest on one load-bearing assumption:
**deception cannot fake evidence volume**, so a malicious interaction's belief
arrives *diffuse* (low concentration), where the tail-mass governor's
DEFER-to-audit gate catches it. The attacks here exist to break that
assumption, and the evaluator measures whether a given governor configuration
survives.

Two attacks are distributional in a way the scalar formalism has no analogue
for:

- **Volume forgery** (evasion). The proxy reads *event count* as evidence, so a
  belief sharpens with the number of signals it exercises. An adversary
  fabricates a pile of (claimed-resolved) rework events: the penalty signal
  saturates so the mean barely moves, but the concentration climbs past the
  DEFER threshold. The forged-sharp belief has almost no tail mass, so it is
  accepted with no audit — and worse, if it *is* audited, its forged
  concentration blunts a fixed-strength audit, so ground truth cannot overrule
  it. Both failures are invisible to a mean-only view.
- **Reputation farming** (exploitation). An adversary does genuine high-quality
  work for several epochs, banking real reputation concentration, then flips to
  poor work behind inflated signals. Pooled reputation carries the inflated
  mean past the gate until realized outcomes drag it back down — a bounded
  exploitation window whose length *is* the detection latency.

The **collusion ring** (coordination) reuses :class:`~beta_swarm.agents.Archetype.COLLUDER`
and is scored here for evasion/damage as well, complementing the shape-based
detection in :mod:`beta_swarm.collusion`.

The mitigations these surface — a random-audit floor, and audit strength that
scales with the audited belief's concentration — are exposed as
:class:`~beta_swarm.simulation.SimulationConfig` levers, so an attack and its
patch run through the same harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from beta_swarm.agents import Archetype, SimAgent, make_population
from beta_swarm.collusion import CollusionDetector
from beta_swarm.governance import TailMassGovernor
from beta_swarm.interaction import BetaInteraction
from beta_swarm.levers import GovernanceStack
from beta_swarm.payoff import DistributionalPayoffEngine
from beta_swarm.proxy import BetaProxyComputer, ProxyObservables
from beta_swarm.simulation import Simulation, SimulationConfig, SimulationResult


class AttackCategory(str, Enum):
    EVASION = "evasion"            # slip past the gate undetected
    EXPLOITATION = "exploitation"  # abuse a trust relationship
    COORDINATION = "coordination"  # multiple adversaries cooperate


# ----------------------------------------------------------------------
# Adversary agents
# ----------------------------------------------------------------------
# Deceptive true-quality: genuinely poor work (mean 0.2).
_POOR = (2.0, 8.0)
_GOOD = (8.0, 3.0)


@dataclass
class VolumeForger(SimAgent):
    """Fabricates event volume to buy concentration past the DEFER gate.

    Emits high progress and engagement (to hold the mean up) alongside a pile
    of claimed-resolved rework events (to drive concentration up). The true
    quality is poor; the forged belief is sharp and high-mean.
    """

    forge_volume: int = 10

    def emit_observables(self, v_true, rng, counterparty=None, epoch=0) -> ProxyObservables:
        return ProxyObservables(
            task_progress_delta=float(np.clip(rng.normal(0.9, 0.05), -1.0, 1.0)),
            rework_count=self.forge_volume,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=float(np.clip(rng.normal(0.9, 0.05), -1.0, 1.0)),
        )


@dataclass
class ReputationFarmer(SimAgent):
    """Genuine work to bank reputation, then a flip to poor work.

    Before ``flip_epoch`` the agent draws high-quality outcomes and reports them
    faithfully (indistinguishable from an honest agent). After the flip it draws
    poor outcomes and emits inflated signals, exploiting the reputation it
    accrued until realized outcomes catch up.
    """

    flip_epoch: int = 10

    def sample_outcome(self, rng, epoch: int = 0) -> float:
        a, b = _GOOD if epoch < self.flip_epoch else _POOR
        return float(rng.beta(a, b))

    def emit_observables(self, v_true, rng, counterparty=None, epoch=0) -> ProxyObservables:
        if epoch < self.flip_epoch:
            signal = 2.0 * v_true - 1.0
            return ProxyObservables(
                task_progress_delta=float(np.clip(signal + rng.normal(0.0, 0.15), -1.0, 1.0)),
                rework_count=int(rng.poisson(1.5 * (1.0 - v_true))),
                verifier_rejections=int(rng.poisson(1.0 * (1.0 - v_true))),
                counterparty_engagement_delta=float(
                    np.clip(signal + rng.normal(0.0, 0.20), -1.0, 1.0)
                ),
            )
        return ProxyObservables(
            task_progress_delta=float(np.clip(rng.normal(0.75, 0.10), -1.0, 1.0)),
            rework_count=0,
            counterparty_engagement_delta=float(np.clip(rng.normal(0.05, 0.10), -1.0, 1.0)),
        )


# ----------------------------------------------------------------------
# Scenario + result
# ----------------------------------------------------------------------
@dataclass
class AttackScenario:
    """A named attack: how to build the population and what it targets."""

    attack_id: str
    name: str
    category: AttackCategory
    description: str
    targeted_assumption: str
    adversaries: list[SimAgent]
    background: dict[Archetype, int] = field(
        default_factory=lambda: {Archetype.HONEST: 4, Archetype.MEDIOCRE: 3}
    )

    def population(self) -> list[SimAgent]:
        """The full population: background archetypes plus the adversaries."""
        return make_population(self.background) + list(self.adversaries)

    @property
    def adversary_ids(self) -> set[str]:
        return {a.agent_id for a in self.adversaries}


@dataclass
class AttackResult:
    """Outcome of running an attack against a governor configuration."""

    attack_id: str
    n_adversary_interactions: int
    evasion_rate: float          # fraction of adversary interactions accepted
    audit_evasion_rate: float    # fraction of adversary interactions never audited
    blocked_rate: float          # fraction barred by a lever (frozen / no stake)
    realized_damage: float       # welfare destroyed by accepted adversary work (>=0)
    adversary_payoff: float      # total payoff banked by the adversaries
    detection_epoch: int | None  # first epoch acceptance stays suppressed, else None
    detection_latency: int | None
    collusion_flagged: bool      # did the shape detector flag any adversary pair?
    acceptance_by_epoch: list[float]

    @property
    def defended(self) -> bool:
        """A crude verdict: adversaries are shut out and do little damage."""
        return self.evasion_rate < 0.1 and self.realized_damage < 5.0


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run_attack(
    scenario: AttackScenario,
    governor: TailMassGovernor,
    config: SimulationConfig | None = None,
    proxy: BetaProxyComputer | None = None,
    levers: GovernanceStack | None = None,
    detection_threshold: float = 0.1,
) -> AttackResult:
    """Execute an attack and score governance robustness.

    Pass a ``proxy`` to test proxy-level mitigations (e.g. a concentration cap
    against volume forgery), or a ``levers`` stack to test surrounding
    governance (staking, circuit breakers). ``detection_threshold`` is the
    per-epoch adversary acceptance rate below which governance is considered to
    have caught on; ``detection_epoch`` is the first epoch from which acceptance
    stays at or below it for the remainder of the run.
    """
    config = config or SimulationConfig()
    engine = DistributionalPayoffEngine()
    result = Simulation(
        scenario.population(), governor, config, engine=engine, proxy=proxy, levers=levers
    ).run()

    ids = scenario.adversary_ids
    adversary = [i for i in result.interactions if i.initiator in ids]
    n = len(adversary)
    if n == 0:
        raise ValueError("scenario produced no adversary interactions")

    accepted = [i for i in adversary if i.accepted]
    evasion_rate = len(accepted) / n
    audit_evasion_rate = sum(not i.metadata["audited"] for i in adversary) / n
    blocked_rate = sum(i.metadata.get("blocked", False) for i in adversary) / n
    realized_damage = sum(
        max(0.0, float(engine.harm_fn(np.asarray(i.ground_truth))
                       - engine.surplus_fn(np.asarray(i.ground_truth))))
        for i in accepted
    )
    adversary_payoff = sum(result.payoffs[a] for a in ids)

    acc_by_epoch = _acceptance_by_epoch(result, ids, config)
    detection_epoch = _first_sustained_below(acc_by_epoch, detection_threshold)
    first_attack = _first_attack_epoch(acc_by_epoch)
    detection_latency = (
        None
        if detection_epoch is None or first_attack is None
        else max(0, detection_epoch - first_attack)
    )

    collusion_flagged = any(
        r.initiator in ids and r.counterparty in ids
        for r in CollusionDetector().flag(result.interactions)
    )

    return AttackResult(
        attack_id=scenario.attack_id,
        n_adversary_interactions=n,
        evasion_rate=evasion_rate,
        audit_evasion_rate=audit_evasion_rate,
        blocked_rate=blocked_rate,
        realized_damage=realized_damage,
        adversary_payoff=adversary_payoff,
        detection_epoch=detection_epoch,
        detection_latency=detection_latency,
        collusion_flagged=collusion_flagged,
        acceptance_by_epoch=acc_by_epoch,
    )


def _acceptance_by_epoch(
    result: SimulationResult, ids: set[str], config: SimulationConfig
) -> list[float]:
    """Per-epoch adversary acceptance rate.

    Interactions are appended in epoch order, and every agent initiates the
    same fixed number per epoch, so the adversary sub-log slices cleanly into
    equal per-epoch chunks.
    """
    adversary = [i for i in result.interactions if i.initiator in ids]
    per = len(adversary) // config.n_epochs if config.n_epochs else len(adversary)
    if per == 0:
        return [float(np.mean([i.accepted for i in adversary]))] if adversary else []
    acc: list[float] = []
    for e in range(config.n_epochs):
        chunk = adversary[e * per : (e + 1) * per]
        if chunk:
            acc.append(sum(i.accepted for i in chunk) / len(chunk))
    return acc


def _first_attack_epoch(acc_by_epoch: list[float]) -> int | None:
    """First epoch with any adversary acceptance (the exploitation onset)."""
    for e, a in enumerate(acc_by_epoch):
        if a > 0.0:
            return e
    return None


def _first_sustained_below(acc_by_epoch: list[float], threshold: float) -> int | None:
    """First epoch from which acceptance stays <= threshold for the remainder.

    Returns ``None`` if the adversary never becomes (and stays) suppressed, or
    if it was never accepted in the first place (nothing to detect).
    """
    if not acc_by_epoch or max(acc_by_epoch) == 0.0:
        return None
    for e in range(len(acc_by_epoch)):
        if all(a <= threshold for a in acc_by_epoch[e:]):
            return e
    return None


# ----------------------------------------------------------------------
# The library
# ----------------------------------------------------------------------
class AttackLibrary:
    """Predefined attack scenarios targeting the distributional defenses."""

    @staticmethod
    def volume_forgery(n_adversaries: int = 3, forge_volume: int = 10) -> AttackScenario:
        return AttackScenario(
            attack_id="volume_forgery",
            name="Volume Forgery",
            category=AttackCategory.EVASION,
            description=(
                "Fabricate a pile of claimed-resolved events to drive the "
                "belief's concentration past the DEFER gate while holding the "
                "mean high, evading the audit entirely."
            ),
            targeted_assumption=(
                "Deception cannot fake evidence volume, so malicious beliefs "
                "arrive diffuse and get deferred to audit."
            ),
            adversaries=[
                VolumeForger(f"forger-{i}", Archetype.DECEPTIVE, *_POOR, forge_volume=forge_volume)
                for i in range(n_adversaries)
            ],
        )

    @staticmethod
    def reputation_farming(n_adversaries: int = 3, flip_epoch: int = 10) -> AttackScenario:
        return AttackScenario(
            attack_id="reputation_farming",
            name="Reputation Farming",
            category=AttackCategory.EXPLOITATION,
            description=(
                "Do genuine high-quality work to bank reputation concentration, "
                "then flip to poor work behind inflated signals — exploiting "
                "pooled reputation until realized outcomes catch up."
            ),
            targeted_assumption=(
                "Reputation earned from realized outcomes is a trustworthy "
                "prior to pool into new interactions' beliefs."
            ),
            adversaries=[
                ReputationFarmer(f"farmer-{i}", Archetype.DECEPTIVE, *_POOR, flip_epoch=flip_epoch)
                for i in range(n_adversaries)
            ],
        )

    @staticmethod
    def collusion_ring(n_adversaries: int = 3) -> AttackScenario:
        adversaries = [
            SimAgent.make(f"colluder-{i}", Archetype.COLLUDER, ring="attack-ring")
            for i in range(n_adversaries)
        ]
        return AttackScenario(
            attack_id="collusion_ring",
            name="Collusion Ring",
            category=AttackCategory.COORDINATION,
            description=(
                "A ring of deceivers preferentially interacts internally, "
                "corroborating each other's inflated signals to buy back the "
                "concentration a lone deceiver forfeits."
            ),
            targeted_assumption=(
                "Corroborated engagement is trustworthy evidence, and a "
                "deceiver acting alone cannot manufacture it."
            ),
            adversaries=adversaries,
        )

    @classmethod
    def all(cls) -> list[AttackScenario]:
        return [cls.volume_forgery(), cls.reputation_farming(), cls.collusion_ring()]
