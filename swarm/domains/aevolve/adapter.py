"""Map A-Evolve evolution outcomes into SWARM's soft-label frame.

Each gated candidate becomes one :class:`SoftInteraction`: the evolution
engine is the initiator, the workspace the counterparty, ``accepted`` is
the gate decision, and ``p`` is the candidate's quality estimate (holdout
score when available, shrunk train score otherwise — a score the gate
optimized against is partially self-graded).

With that mapping the standard metrics interrogate the evolution loop:

- ``quality_gap < 0`` ⇒ the gate adversely selects — it admits mutations
  that look good on the benchmark while true (holdout) quality is worse
  than what it rolled back. That is exactly the Goodhart failure mode the
  A-Evolve bridge exists to measure.
- ``toxicity`` ⇒ expected harm carried by the accepted lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from swarm.domains.aevolve.config import AevolveDomainConfig
from swarm.domains.aevolve.entities import CandidateOutcome, GateDecision
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction


@dataclass
class EvolutionReport:
    """SWARM metrics over one evolution run's gated candidates."""

    n_candidates: int = 0
    n_accepted: int = 0
    toxicity: float = 0.0
    quality_gap: float = 0.0
    mean_p_accepted: float = 0.0
    adverse_selection: bool = False
    interactions: List[SoftInteraction] = field(default_factory=list)


class AevolveEvolutionAdapter:
    """Fold :class:`CandidateOutcome` records into an :class:`EvolutionReport`."""

    def __init__(self, config: Optional[AevolveDomainConfig] = None) -> None:
        self.config = config or AevolveDomainConfig()

    def candidate_p(self, outcome: CandidateOutcome) -> float:
        """Quality estimate for one candidate, always in [0, 1]."""

        if outcome.holdout_score is not None:
            p = outcome.holdout_score
        else:
            shrink = self.config.train_score_shrinkage
            p = 0.5 + (outcome.benchmark_score - 0.5) * (1.0 - shrink)
        return max(0.0, min(1.0, p))

    def from_outcomes(self, outcomes: Sequence[CandidateOutcome]) -> EvolutionReport:
        interactions: List[SoftInteraction] = []
        for outcome in outcomes:
            if outcome.gate == GateDecision.PENDING:
                continue
            p = self.candidate_p(outcome)
            interactions.append(
                SoftInteraction(
                    initiator="evolution-engine",
                    counterparty=f"workspace:{outcome.candidate_id}",
                    accepted=outcome.gate == GateDecision.ACCEPTED,
                    p=p,
                    v_hat=2.0 * p - 1.0,
                    metadata={
                        "trial": outcome.trial,
                        "mutation_kind": outcome.mutation_kind,
                        "benchmark_score": outcome.benchmark_score,
                        "holdout_score": outcome.holdout_score,
                    },
                )
            )

        metrics = SoftMetrics()
        accepted = [i for i in interactions if i.accepted]
        quality_gap = metrics.quality_gap(interactions)
        return EvolutionReport(
            n_candidates=len(interactions),
            n_accepted=len(accepted),
            toxicity=metrics.toxicity_rate(interactions),
            quality_gap=quality_gap,
            mean_p_accepted=(
                sum(i.p for i in accepted) / len(accepted) if accepted else 0.0
            ),
            adverse_selection=quality_gap < 0,
            interactions=interactions,
        )
