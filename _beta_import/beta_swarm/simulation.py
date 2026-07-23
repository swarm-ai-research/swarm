"""Closed-loop simulation: agents, proxy, governance, payoffs, reputation.

This ports the main SWARM framework's orchestrator lesson — safety phenomena
are *emergent*; you only see them by running populations through a governed
interaction loop — and generalizes three of its mechanisms to distributions:

1. **Reputation is a belief.** The scalar framework tracks a reputation number
   with epoch decay. Here the governor maintains a ``BetaBelief`` per agent,
   conjugately updated with realized outcomes; reputation decay becomes
   *concentration* decay (evidence discounting), so a neglected reputation
   diffuses back toward "we don't know" rather than sliding toward a default.

2. **Evidence pools.** The per-interaction proxy belief and the initiator's
   reputation belief are two evidence sources about the same outcome; they
   combine by adding pseudo-counts (counting the shared uniform prior once).

3. **Audits add evidence.** The scalar framework's audit lever re-scores an
   interaction at a cost. Here a ``DEFER`` verdict routes the interaction to an
   audit that injects pseudo-evidence about the true outcome, *sharpening* the
   belief before the governor re-decides. Auditing is how a diffuse belief earns
   its way past a tail-mass gate — and its cost falls on unproven agents, then
   fades as their reputation concentrates.

Rejected interactions yield no outcome evidence (you never observe the quality
of work you refused), so adverse selection and its remedies are visible in the
metrics, epoch by epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from beta_swarm.agents import Archetype, SimAgent
from beta_swarm.belief import BetaBelief
from beta_swarm.governance import Decision, TailMassGovernor
from beta_swarm.interaction import BetaInteraction, InteractionType
from beta_swarm.levers import GovernanceStack, LeverContext
from beta_swarm.metrics import DistributionalMetrics
from beta_swarm.payoff import DistributionalPayoffEngine
from beta_swarm.proxy import BetaProxyComputer

if TYPE_CHECKING:
    from beta_swarm.adaptive import AdaptiveController

_FLOOR = 1e-6


def pool_beliefs(a: BetaBelief, b: BetaBelief) -> BetaBelief:
    """Combine two independent evidence sources about the same outcome.

    Each belief is (uniform prior + pseudo-evidence). Pooling adds the evidence
    and counts the shared ``Beta(1, 1)`` prior once:

        alpha = alpha_a + alpha_b - 1,   beta = beta_a + beta_b - 1
    """
    return BetaBelief(
        alpha=max(a.alpha + b.alpha - 1.0, _FLOOR),
        beta=max(b.beta + a.beta - 1.0, _FLOOR),
    )


def decay_belief(belief: BetaBelief, decay: float) -> BetaBelief:
    """Discount a belief's evidence toward the uniform prior.

    ``alpha' = 1 + decay * (alpha - 1)`` (and likewise for beta): the mean drifts
    toward 0.5 only as fast as the evidence supporting it fades, and the
    concentration shrinks toward the prior's. This is the distributional form of
    the scalar framework's reputation-decay lever.
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1]")
    return BetaBelief(
        alpha=max(1.0 + decay * (belief.alpha - 1.0), _FLOOR),
        beta=max(1.0 + decay * (belief.beta - 1.0), _FLOOR),
    )


@dataclass
class SimulationConfig:
    """Knobs for the closed loop.

    Parameters
    ----------
    n_epochs, steps_per_epoch:
        Loop dimensions. Every agent initiates one interaction per step.
    seed:
        Single RNG seed; identical configs replay identically.
    audit_strength:
        Pseudo-counts of outcome evidence one audit injects. Larger audits
        sharpen the belief harder around the (noisily) observed truth.
    audit_noise:
        Std-dev of the auditor's observation of ``v_true``.
    audit_cost:
        Charged to the initiator per audit (a real resource; subtracted from
        welfare too).
    reputation_decay:
        Epoch-end evidence discount on every reputation belief (1 = no decay).
    ring_affinity:
        Probability that a ring member picks a fellow ring member as
        counterparty (uniform over the rest otherwise). Non-ring agents always
        pick uniformly.
    random_audit_prob:
        Probability that any interaction is audited regardless of its verdict —
        a floor of scrutiny no belief shape is exempt from (the parent
        framework's random-audit lever). ``0`` disables it.
    audit_scales_with_concentration:
        If ``True``, an audit's evidence strength scales with the audited
        belief's concentration (``audit_concentration_multiple * concentration``,
        floored at ``audit_strength``) so a forged-confidence belief cannot
        blunt its own audit. If ``False`` (default), audits carry a fixed
        ``audit_strength`` — the vulnerable configuration.
    audit_concentration_multiple:
        The scaling factor used when ``audit_scales_with_concentration`` is set.
    """

    n_epochs: int = 20
    steps_per_epoch: int = 5
    seed: int = 0
    audit_strength: float = 16.0
    audit_noise: float = 0.10
    audit_cost: float = 0.05
    reputation_decay: float = 0.9
    ring_affinity: float = 0.8
    random_audit_prob: float = 0.0
    audit_scales_with_concentration: bool = False
    audit_concentration_multiple: float = 2.0


@dataclass
class EpochReport:
    """Per-epoch metrics, scalar and distributional side by side."""

    epoch: int
    n_accepted: int
    n_rejected: int
    n_audits: int
    realized_welfare: float          # sum of surplus(v_true) - harm(v_true) - audit costs
    realized_toxicity: float         # E[1 - v_true | accepted]
    mean_quality_gap: float          # legacy scalar gap
    wasserstein_gap: float           # distributional gap
    expected_tail_mass: float        # E[P(v < tau) | accepted]
    crps: float | None               # calibration of beliefs vs ground truth
    acceptance_by_archetype: dict[str, float]


@dataclass
class SimulationResult:
    interactions: list[BetaInteraction]
    epoch_reports: list[EpochReport]
    reputation: dict[str, BetaBelief]
    payoffs: dict[str, float]

    def acceptance_rate(self, archetype: Archetype) -> float:
        """Overall acceptance rate for one archetype's interactions."""
        mine = [
            i for i in self.interactions if i.metadata.get("archetype") == archetype.value
        ]
        if not mine:
            return 0.0
        return sum(i.accepted for i in mine) / len(mine)

    @property
    def total_welfare(self) -> float:
        return sum(r.realized_welfare for r in self.epoch_reports)

    @property
    def overall_toxicity(self) -> float:
        accepted = [i for i in self.interactions if i.accepted]
        if not accepted:
            return 0.0
        return float(np.mean([1.0 - i.ground_truth for i in accepted]))


class Simulation:
    """Run a population through a governed interaction loop.

    Per interaction: the initiator's true outcome is drawn, its archetype emits
    observables, the proxy builds a belief, that belief pools with the
    initiator's reputation, and the governor decides. ``DEFER`` triggers one
    audit (evidence injection at a cost) and a re-decision; a second ``DEFER``
    rejects. Accepted interactions realize payoffs and update reputation;
    rejected ones are never observed, so they teach the governor nothing.
    """

    def __init__(
        self,
        population: list[SimAgent],
        governor: TailMassGovernor,
        config: SimulationConfig | None = None,
        engine: DistributionalPayoffEngine | None = None,
        proxy: BetaProxyComputer | None = None,
        levers: GovernanceStack | None = None,
        controller: "AdaptiveController | None" = None,
    ):
        if not population:
            raise ValueError("population must be non-empty")
        self.population = population
        self.governor = governor
        self.config = config or SimulationConfig()
        self.engine = engine or DistributionalPayoffEngine()
        self.proxy = proxy or BetaProxyComputer()
        self.levers = levers
        self.controller = controller
        self.metrics = DistributionalMetrics()

    def run(self) -> SimulationResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        reputation: dict[str, BetaBelief] = {
            a.agent_id: BetaBelief.uniform() for a in self.population
        }
        payoffs: dict[str, float] = {a.agent_id: 0.0 for a in self.population}
        all_interactions: list[BetaInteraction] = []
        reports: list[EpochReport] = []
        if self.levers is not None:
            self.levers.reset([a.agent_id for a in self.population])

        for epoch in range(cfg.n_epochs):
            epoch_interactions: list[BetaInteraction] = []
            n_audits = 0
            welfare = 0.0
            if self.levers is not None:
                self.levers.on_epoch_start(epoch)

            for _ in range(cfg.steps_per_epoch):
                for agent in self.population:
                    if self.levers is not None and not self.levers.can_act(agent.agent_id):
                        epoch_interactions.append(self._blocked_interaction(agent))
                        continue

                    interaction, audited = self._one_interaction(
                        agent, reputation, rng, epoch
                    )
                    n_audits += audited
                    epoch_interactions.append(interaction)

                    if audited:
                        welfare -= cfg.audit_cost
                    if interaction.accepted:
                        v = interaction.ground_truth
                        welfare += float(
                            self.engine.surplus_fn(np.asarray(v))
                            - self.engine.harm_fn(np.asarray(v))
                        )
                        pi_a, pi_b = self.engine.payoffs_both(
                            interaction.belief, c_a=interaction.c_a
                        )
                        payoffs[interaction.initiator] += pi_a
                        payoffs[interaction.counterparty] += pi_b
                        reputation[agent.agent_id] = reputation[agent.agent_id].update(
                            positive=v, negative=1.0 - v
                        )

                    if self.levers is not None:
                        effect = self.levers.on_interaction(
                            LeverContext(
                                epoch=epoch,
                                agent_id=agent.agent_id,
                                belief=interaction.belief,
                                accepted=interaction.accepted,
                                ground_truth=interaction.ground_truth,
                                resources=self.levers.resources.get(agent.agent_id, 0.0),
                            )
                        )
                        if effect.cost:
                            payoffs[agent.agent_id] -= effect.cost
                            welfare -= effect.cost
                            interaction.metadata["lever_cost"] = effect.cost
                        if effect.details:
                            interaction.metadata["lever_details"] = effect.details

            for agent_id, belief in reputation.items():
                reputation[agent_id] = decay_belief(belief, cfg.reputation_decay)

            report = self._report(epoch, epoch_interactions, n_audits, welfare)
            reports.append(report)
            all_interactions.extend(epoch_interactions)
            if self.controller is not None:
                self.controller.on_epoch_end(report, self.governor, epoch)

        return SimulationResult(
            interactions=all_interactions,
            epoch_reports=reports,
            reputation=reputation,
            payoffs=payoffs,
        )

    # ------------------------------------------------------------------
    # One interaction through the pipeline
    # ------------------------------------------------------------------
    def _one_interaction(
        self,
        agent: SimAgent,
        reputation: dict[str, BetaBelief],
        rng: np.random.Generator,
        epoch: int = 0,
    ) -> tuple[BetaInteraction, bool]:
        cfg = self.config
        counterparty = self._pick_counterparty(agent, rng)
        v_true = agent.sample_outcome(rng, epoch)
        obs = agent.emit_observables(v_true, rng, counterparty=counterparty, epoch=epoch)
        proxy_belief = self.proxy.compute_belief(obs)
        belief = pool_beliefs(proxy_belief, reputation[agent.agent_id])

        verdict = self.governor.evaluate(belief)
        audited = False
        audit_cost = 0.0
        forced_audit = cfg.random_audit_prob > 0.0 and rng.random() < cfg.random_audit_prob
        if verdict.decision is Decision.DEFER or forced_audit:
            audited = True
            audit_cost = cfg.audit_cost
            # An audit must be able to overrule the belief it checks. A fixed
            # strength is blunted by a forged-confidence belief (volume-forgery
            # attack); scaling the audit to the belief's concentration keeps
            # ground truth dominant regardless of how sharp the belief claims.
            strength = cfg.audit_strength
            if cfg.audit_scales_with_concentration:
                strength = max(strength, cfg.audit_concentration_multiple * belief.concentration)
            v_seen = float(np.clip(v_true + rng.normal(0.0, cfg.audit_noise), 0.0, 1.0))
            belief = belief.update(
                positive=strength * v_seen,
                negative=strength * (1.0 - v_seen),
            )
            verdict = self.governor.evaluate(belief)

        accepted = verdict.decision is Decision.ACCEPT

        interaction = BetaInteraction(
            initiator=agent.agent_id,
            counterparty=counterparty.agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=accepted,
            belief=belief,
            proxy_belief=proxy_belief,
            c_a=audit_cost,
            ground_truth=v_true,
            metadata={
                "archetype": agent.archetype.value,
                "audited": audited,
                "verdict": verdict.decision.value,
                "reason": verdict.reason,
            },
        )
        return interaction, audited

    def _blocked_interaction(self, agent: SimAgent) -> BetaInteraction:
        """A rejected placeholder for an agent barred by a lever (frozen / no stake).

        No proxy, governor, or payoff runs; the outcome is never observed
        (``ground_truth=None``), so it is excluded from calibration and quality
        metrics while still keeping the per-epoch interaction count stable.
        """
        return BetaInteraction(
            initiator=agent.agent_id,
            counterparty=agent.agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=False,
            ground_truth=None,
            metadata={
                "archetype": agent.archetype.value,
                "audited": False,
                "verdict": "blocked",
                "blocked": True,
            },
        )

    def _pick_counterparty(self, agent: SimAgent, rng: np.random.Generator) -> SimAgent:
        others = [a for a in self.population if a.agent_id != agent.agent_id]
        if not others:
            return agent
        ring_partners = [a for a in others if agent.in_ring_with(a)]
        if ring_partners and rng.random() < self.config.ring_affinity:
            return ring_partners[rng.integers(len(ring_partners))]
        return others[rng.integers(len(others))]

    # ------------------------------------------------------------------
    # Epoch metrics
    # ------------------------------------------------------------------
    def _report(
        self,
        epoch: int,
        interactions: list[BetaInteraction],
        n_audits: int,
        welfare: float,
    ) -> EpochReport:
        accepted = [i for i in interactions if i.accepted]
        by_archetype: dict[str, float] = {}
        for archetype in Archetype:
            mine = [i for i in interactions if i.metadata["archetype"] == archetype.value]
            if mine:
                by_archetype[archetype.value] = sum(i.accepted for i in mine) / len(mine)
        realized_toxicity = (
            float(np.mean([1.0 - i.ground_truth for i in accepted])) if accepted else 0.0
        )
        return EpochReport(
            epoch=epoch,
            n_accepted=len(accepted),
            n_rejected=len(interactions) - len(accepted),
            n_audits=n_audits,
            realized_welfare=welfare,
            realized_toxicity=realized_toxicity,
            mean_quality_gap=self.metrics.mean_quality_gap(interactions),
            wasserstein_gap=self.metrics.quality_gap_wasserstein(interactions),
            expected_tail_mass=self.metrics.expected_tail_mass(
                interactions, tau=self.governor.tau
            ),
            crps=self.metrics.crps(interactions),
            acceptance_by_archetype=by_archetype,
        )
