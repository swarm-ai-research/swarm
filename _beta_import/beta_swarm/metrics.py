"""Distributional metrics: the quality gap generalizes to a distance between
the accepted and rejected *belief distributions*, not a difference of means.

In the scalar formalism the quality gap was ``E[p | accepted] - E[p | rejected]``
— a signed difference of two numbers. Once each interaction carries a full
belief, "are accepted interactions different from rejected ones?" becomes a
question about two *distributions over outcomes*. We answer it with a proper
distance:

- **Wasserstein-1** (earth-mover) between the accepted and rejected outcome
  mixtures — symmetric, in outcome units, and sensitive to spread, not just
  location.
- **KL divergence** between the mixtures — information-theoretic, asymmetric.

The legacy mean-difference gap is retained as :meth:`mean_quality_gap` so the
two formalisms can be compared directly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import special

from beta_swarm.belief import BetaBelief
from beta_swarm.interaction import BetaInteraction


# ----------------------------------------------------------------------
# Pairwise divergences between two beliefs
# ----------------------------------------------------------------------
def beta_kl(p: BetaBelief, q: BetaBelief) -> float:
    """Closed-form ``KL( Beta(p) || Beta(q) )``.

    Uses the standard expression in terms of log-Beta and digamma functions.
    """
    a1, b1 = p.alpha, p.beta
    a2, b2 = q.alpha, q.beta
    return float(
        special.betaln(a2, b2)
        - special.betaln(a1, b1)
        + (a1 - a2) * special.digamma(a1)
        + (b1 - b2) * special.digamma(b1)
        + (a2 - a1 + b2 - b1) * special.digamma(a1 + b1)
    )


# ----------------------------------------------------------------------
# Mixture helpers (a set of beliefs -> one empirical distribution)
# ----------------------------------------------------------------------
def _mixture_cdf(beliefs: Sequence[BetaBelief], grid: np.ndarray) -> np.ndarray:
    """CDF of the equal-weight mixture of ``beliefs`` evaluated on ``grid``."""
    acc = np.zeros_like(grid)
    for b in beliefs:
        acc += b.cdf(grid)
    return acc / len(beliefs)


def _mixture_pdf(beliefs: Sequence[BetaBelief], grid: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(grid)
    for b in beliefs:
        acc += b.pdf(grid)
    return acc / len(beliefs)


def wasserstein1_mixtures(
    a: Sequence[BetaBelief], b: Sequence[BetaBelief], n_grid: int = 1024
) -> float:
    """Wasserstein-1 distance between two equal-weight mixtures of beliefs.

    On the bounded support ``[0, 1]`` this equals ``∫ |F_a(v) - F_b(v)| dv``.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    fa = _mixture_cdf(a, grid)
    fb = _mixture_cdf(b, grid)
    return float(np.trapezoid(np.abs(fa - fb), grid))


def kl_mixtures(
    a: Sequence[BetaBelief], b: Sequence[BetaBelief], n_grid: int = 1024, eps: float = 1e-12
) -> float:
    """KL divergence ``KL(mix_a || mix_b)`` between two belief mixtures (numeric)."""
    grid = np.linspace(0.0, 1.0, n_grid)
    pa = _mixture_pdf(a, grid) + eps
    pb = _mixture_pdf(b, grid) + eps
    integrand = pa * np.log(pa / pb)
    return float(np.trapezoid(integrand, grid))


class DistributionalMetrics:
    """Population-level metrics over a list of :class:`BetaInteraction`."""

    # --- selection structure ------------------------------------------------
    @staticmethod
    def _split(interactions: Sequence[BetaInteraction]):
        acc = [i.belief for i in interactions if i.accepted]
        rej = [i.belief for i in interactions if not i.accepted]
        return acc, rej

    def quality_gap_wasserstein(self, interactions: Sequence[BetaInteraction]) -> float:
        """Wasserstein-1 distance between accepted and rejected belief mixtures.

        Always non-negative (it is a metric). Pair it with
        :meth:`mean_quality_gap` to recover the *sign* — i.e. whether the
        selection favors high- or low-quality interactions.
        """
        acc, rej = self._split(interactions)
        if not acc or not rej:
            return 0.0
        return wasserstein1_mixtures(acc, rej)

    def quality_gap_kl(self, interactions: Sequence[BetaInteraction]) -> float:
        """KL divergence between accepted and rejected belief mixtures."""
        acc, rej = self._split(interactions)
        if not acc or not rej:
            return 0.0
        return kl_mixtures(acc, rej)

    def mean_quality_gap(self, interactions: Sequence[BetaInteraction]) -> float:
        """Legacy gap: ``E[mean | accepted] - E[mean | rejected]``.

        Negative indicates adverse selection. Retained for direct comparison
        with the scalar formalism.
        """
        acc, rej = self._split(interactions)
        if not acc or not rej:
            return 0.0
        return float(np.mean([b.mean for b in acc]) - np.mean([b.mean for b in rej]))

    # --- toxicity / tail risk ------------------------------------------------
    def toxicity_rate(self, interactions: Sequence[BetaInteraction]) -> float:
        """``E[1 - mean | accepted]`` — the direct analogue of the old toxicity."""
        acc = [i.belief for i in interactions if i.accepted]
        if not acc:
            return 0.0
        return float(np.mean([1.0 - b.mean for b in acc]))

    def expected_tail_mass(
        self, interactions: Sequence[BetaInteraction], tau: float = 0.25
    ) -> float:
        """``E[ P(v < tau) | accepted ]`` — distributional toxicity.

        Unlike :meth:`toxicity_rate`, this sees epistemic uncertainty: a diffuse
        belief contributes tail mass even when its mean is high.
        """
        acc = [i.belief for i in interactions if i.accepted]
        if not acc:
            return 0.0
        return float(np.mean([b.tail_mass(tau) for b in acc]))

    def mean_concentration(
        self, interactions: Sequence[BetaInteraction], accepted_only: bool = False
    ) -> float:
        """Average epistemic certainty across interactions."""
        beliefs = [
            i.belief for i in interactions if (i.accepted or not accepted_only)
        ]
        if not beliefs:
            return 0.0
        return float(np.mean([b.concentration for b in beliefs]))

    # --- calibration ---------------------------------------------------------
    def crps(self, interactions: Sequence[BetaInteraction], n_grid: int = 512) -> float | None:
        """Mean Continuous Ranked Probability Score against ground-truth outcomes.

        CRPS is the proper scoring rule for a full predictive distribution
        (the continuous analogue of the Brier score):

            CRPS(F, y) = ∫ (F(v) - 1{v >= y})^2 dv

        Lower is better. Returns ``None`` if no interaction has ground truth.
        """
        with_truth = [i for i in interactions if i.ground_truth is not None]
        if not with_truth:
            return None
        grid = np.linspace(0.0, 1.0, n_grid)
        total = 0.0
        for i in with_truth:
            cdf = i.belief.cdf(grid)
            step = (grid >= i.ground_truth).astype(float)
            total += float(np.trapezoid((cdf - step) ** 2, grid))
        return total / len(with_truth)
