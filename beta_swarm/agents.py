"""Agent archetypes for closed-loop simulation.

The main SWARM framework runs populations of behavioral archetypes (honest,
opportunistic, deceptive, ...) through an orchestrator to study emergent,
interaction-level failures. beta-swarm distills that lesson to the three
archetypes that matter for the *distributional* story:

- **honest** — genuinely good work, and observables that faithfully report it.
- **mediocre** — reliably average work, faithfully reported. The "confidently
  mediocre" belief shape comes from this population.
- **deceptive** — genuinely poor work behind inflated signals. The agent can
  fake the *level* of its observables (report high progress, hide rework and
  verifier events) but it cannot fake the *volume* of evidence: an interaction
  that hides its negative events simply exercises fewer signals, so the proxy
  produces a belief with a high mean and **low concentration**. That diffuse
  shape is the distributional fingerprint of deception — invisible to scalar
  ``p``, and exactly what tail-mass governance triggers on.
- **colluder** — a deceptive agent with accomplices. Ring members preferentially
  interact with each other, and a ring counterparty *corroborates* the inflated
  signals with faked engagement — the one evidence channel a lone deceiver
  cannot manufacture. Within-ring beliefs therefore arrive with the same
  inflated mean but **higher concentration** than the ring's interactions with
  outsiders. The mean asymmetry is negligible (they lie to everyone equally);
  the *shape* asymmetry is the collusion fingerprint.

Each agent has a true quality distribution ``v ~ Beta(quality_alpha,
quality_beta)`` and an emission model mapping a realized outcome ``v_true`` to
:class:`~beta_swarm.proxy.ProxyObservables`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from beta_swarm.proxy import ProxyObservables


class Archetype(str, Enum):
    HONEST = "honest"
    MEDIOCRE = "mediocre"
    DECEPTIVE = "deceptive"
    COLLUDER = "colluder"


# True-quality Beta parameters per archetype: (alpha, beta).
_QUALITY: dict[Archetype, tuple[float, float]] = {
    Archetype.HONEST: (8.0, 3.0),      # mean ~0.73 — genuinely good
    Archetype.MEDIOCRE: (30.0, 30.0),  # mean 0.50, sharp — reliably average
    Archetype.DECEPTIVE: (2.0, 8.0),   # mean 0.20 — genuinely poor
    Archetype.COLLUDER: (2.0, 8.0),    # mean 0.20 — deceptive, with accomplices
}


@dataclass
class SimAgent:
    """A simulated agent: a true quality distribution plus an emission model.

    ``ring`` groups colluders: agents sharing a ring id preferentially select
    each other and corroborate each other's inflated signals.
    """

    agent_id: str
    archetype: Archetype
    quality_alpha: float
    quality_beta: float
    ring: str | None = None

    @classmethod
    def make(cls, agent_id: str, archetype: Archetype, ring: str | None = None) -> "SimAgent":
        qa, qb = _QUALITY[archetype]
        return cls(
            agent_id=agent_id,
            archetype=archetype,
            quality_alpha=qa,
            quality_beta=qb,
            ring=ring,
        )

    def in_ring_with(self, other: "SimAgent | None") -> bool:
        return other is not None and self.ring is not None and self.ring == other.ring

    def sample_outcome(self, rng: np.random.Generator, epoch: int = 0) -> float:
        """Draw the true quality ``v_true in [0, 1]`` of one interaction.

        ``epoch`` is passed for time-varying strategies (e.g. a red-team agent
        that does genuine work early, then flips); base archetypes ignore it.
        """
        return float(rng.beta(self.quality_alpha, self.quality_beta))

    def emit_observables(
        self,
        v_true: float,
        rng: np.random.Generator,
        counterparty: "SimAgent | None" = None,
        epoch: int = 0,
    ) -> ProxyObservables:
        """Generate the observable signals the proxy will see.

        Honest and mediocre agents emit *faithful* observables correlated with
        ``v_true``, including the negative events (rework, verifier rejections)
        that both lower the belief mean and add evidence. Deceptive agents emit
        *inflated* observables: high reported progress regardless of ``v_true``,
        suppressed negative events, and tepid counterparty engagement — which
        raises the mean but starves the belief of concentration. A colluder
        emits like a deceiver, except that a ring counterparty corroborates with
        enthusiastic engagement, buying back the concentration a lone deceiver
        forfeits.

        ``epoch`` is passed for time-varying strategies (e.g. red-team
        reputation farming); the base archetypes ignore it.
        """
        if self.archetype in (Archetype.DECEPTIVE, Archetype.COLLUDER):
            if self.archetype is Archetype.COLLUDER and self.in_ring_with(counterparty):
                engagement = rng.normal(0.85, 0.10)
            else:
                engagement = rng.normal(0.05, 0.10)
            return ProxyObservables(
                task_progress_delta=float(np.clip(rng.normal(0.75, 0.10), -1.0, 1.0)),
                rework_count=0,
                verifier_rejections=0,
                tool_misuse_flags=0,
                counterparty_engagement_delta=float(np.clip(engagement, -1.0, 1.0)),
            )
        signal = 2.0 * v_true - 1.0
        return ProxyObservables(
            task_progress_delta=float(np.clip(signal + rng.normal(0.0, 0.15), -1.0, 1.0)),
            rework_count=int(rng.poisson(1.5 * (1.0 - v_true))),
            verifier_rejections=int(rng.poisson(1.0 * (1.0 - v_true))),
            tool_misuse_flags=0,
            counterparty_engagement_delta=float(
                np.clip(signal + rng.normal(0.0, 0.20), -1.0, 1.0)
            ),
        )


def make_population(counts: dict[Archetype, int]) -> list[SimAgent]:
    """Build a population like ``{HONEST: 4, MEDIOCRE: 3, DECEPTIVE: 3}``.

    Agent ids are ``"<archetype>-<index>"`` so logs stay readable. All
    ``COLLUDER`` agents are placed in one shared ring (``"ring-0"``).
    """
    agents: list[SimAgent] = []
    for archetype, n in counts.items():
        ring = "ring-0" if archetype is Archetype.COLLUDER else None
        for i in range(n):
            agents.append(SimAgent.make(f"{archetype.value}-{i}", archetype, ring=ring))
    return agents
