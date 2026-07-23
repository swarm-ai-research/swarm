"""Collusion detection on belief distributions.

The main SWARM framework's ``CollusionDetector`` flags rings with two scalar
signals: unusual pair interaction frequency, and quality asymmetry — high ``p``
inside the group, low ``p`` toward outsiders. A competent ring defeats the
second signal by lying to *everyone*: it inflates observables toward outsiders
too, so the belief means inside and outside the ring are nearly identical.

The distributional fingerprint survives, because corroboration is evidence.
A ring counterparty actively confirms the inflated signals (faked engagement),
which a lone deceiver cannot manufacture — so within-ring beliefs arrive with
the same mean but systematically **higher concentration** than the same
initiator's beliefs toward outsiders. Shape-aware signals see what the mean
does not:

- **Wasserstein-1** between the within-pair and outside belief mixtures
  (sensitive to spread, not just location),
- **concentration lift** — the ratio of within-pair to outside mean
  concentration,
- **tail asymmetry** — how much downside tail mass the within-pair beliefs
  are missing relative to the initiator's outside interactions.

Combined with frequency lift (the graph signal, carried over from the scalar
detector), suspicious pairs separate cleanly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from beta_swarm.belief import BetaBelief
from beta_swarm.interaction import BetaInteraction
from beta_swarm.metrics import wasserstein1_mixtures


@dataclass(frozen=True)
class PairReport:
    """Collusion signals for one ordered pair (initiator -> counterparty)."""

    initiator: str
    counterparty: str
    n_interactions: int
    frequency_lift: float      # observed / expected count under uniform mixing
    mean_gap: float            # within-pair mean minus outside mean (the scalar signal)
    w1_gap: float              # Wasserstein-1 between within and outside mixtures
    concentration_lift: float  # within-pair mean concentration / outside
    tail_asymmetry: float      # outside tail mass minus within-pair tail mass
    suspicion: float           # frequency_lift * w1_gap — 0 when either is absent

    def __repr__(self) -> str:
        return (
            f"PairReport({self.initiator} -> {self.counterparty}: "
            f"n={self.n_interactions}, freq x{self.frequency_lift:.1f}, "
            f"w1={self.w1_gap:.3f}, conc x{self.concentration_lift:.2f}, "
            f"suspicion={self.suspicion:.3f})"
        )


@dataclass
class CollusionDetector:
    """Score ordered pairs for collusion from an interaction log.

    Parameters
    ----------
    tau:
        Bad-outcome threshold for the tail-asymmetry signal.
    min_interactions:
        Pairs with fewer within-pair interactions are not scored (too noisy).
    min_frequency_lift, min_w1_gap:
        Flagging thresholds for :meth:`flag`: a pair must interact more often
        than uniform mixing predicts *and* look distributionally different
        toward each other than toward outsiders.
    """

    tau: float = 0.4
    min_interactions: int = 5
    min_frequency_lift: float = 2.0
    min_w1_gap: float = 0.03

    def analyze(self, interactions: Sequence[BetaInteraction]) -> list[PairReport]:
        """Score every qualifying ordered pair, most suspicious first."""
        by_initiator: dict[str, list[tuple[str, BetaBelief]]] = defaultdict(list)
        agents: set[str] = set()
        for i in interactions:
            # Score on the per-interaction evidence belief when available: the
            # pooled belief mixes in a reputation prior that is identical
            # toward every partner, diluting exactly the contrast we measure.
            belief = i.proxy_belief if i.proxy_belief is not None else i.belief
            by_initiator[i.initiator].append((i.counterparty, belief))
            agents.add(i.initiator)
            agents.add(i.counterparty)
        n_agents = len(agents)
        if n_agents < 3:
            return []

        reports: list[PairReport] = []
        for initiator, mine in by_initiator.items():
            by_partner: dict[str, list[BetaBelief]] = defaultdict(list)
            for counterparty, belief in mine:
                by_partner[counterparty].append(belief)
            expected = len(mine) / (n_agents - 1)

            # Baseline: the initiator's arm's-length traffic — partners that
            # are NOT themselves overrepresented. A ring member's baseline must
            # exclude its fellow ring members, or their corroborated beliefs
            # contaminate the very contrast being measured.
            lifts = {p: len(bs) / expected for p, bs in by_partner.items()}

            for partner, within in by_partner.items():
                if len(within) < self.min_interactions:
                    continue
                baseline = [
                    b
                    for p, bs in by_partner.items()
                    if p != partner and lifts[p] < self.min_frequency_lift
                    for b in bs
                ] or [b for p, bs in by_partner.items() if p != partner for b in bs]
                if len(baseline) < self.min_interactions:
                    continue
                reports.append(
                    self._score_pair(initiator, partner, within, baseline, expected)
                )
        reports.sort(key=lambda r: r.suspicion, reverse=True)
        return reports

    def flag(self, interactions: Sequence[BetaInteraction]) -> list[PairReport]:
        """The pairs whose graph *and* shape signals both exceed thresholds."""
        return [
            r
            for r in self.analyze(interactions)
            if r.frequency_lift >= self.min_frequency_lift
            and r.w1_gap >= self.min_w1_gap
        ]

    def _score_pair(
        self,
        initiator: str,
        partner: str,
        within: list[BetaBelief],
        outside: list[BetaBelief],
        expected_count: float,
    ) -> PairReport:
        frequency_lift = len(within) / expected_count if expected_count > 0 else 0.0
        w1_gap = wasserstein1_mixtures(within, outside)
        mean_gap = float(
            np.mean([b.mean for b in within]) - np.mean([b.mean for b in outside])
        )
        conc_out = float(np.mean([b.concentration for b in outside]))
        concentration_lift = (
            float(np.mean([b.concentration for b in within])) / conc_out
            if conc_out > 0
            else 0.0
        )
        tail_asymmetry = float(
            np.mean([b.tail_mass(self.tau) for b in outside])
            - np.mean([b.tail_mass(self.tau) for b in within])
        )
        return PairReport(
            initiator=initiator,
            counterparty=partner,
            n_interactions=len(within),
            frequency_lift=frequency_lift,
            mean_gap=mean_gap,
            w1_gap=w1_gap,
            concentration_lift=concentration_lift,
            tail_asymmetry=tail_asymmetry,
            suspicion=frequency_lift * w1_gap,
        )
