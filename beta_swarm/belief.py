"""Beta belief: a parameterized distribution over a continuous outcome v in [0, 1].

This is the core primitive of beta-swarm. Where the original soft-label
formalism carried a single scalar ``p = P(v = +1)``, an agent here carries a
full ``Beta(alpha, beta)`` belief over the continuous quality ``v in [0, 1]``.

That belief decomposes into two interpretable quantities:

- **mean**     ``alpha / (alpha + beta)`` — recovers the old point estimate ``p``.
- **concentration** ``alpha + beta`` — epistemic certainty. Large concentration
  means a sharp belief; small concentration means a diffuse one.

The whole point of carrying both is to separate *"confident the interaction is
mediocre"* (mean ~ 0.5, high concentration) from *"no idea whether it's great or
terrible"* (mean ~ 0.5, low concentration). The scalar-``p`` formalism collapses
these to the same number; here they have very different tail mass ``P(v < tau)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

# Floor for alpha/beta. Beta(alpha, beta) requires both > 0; we keep a small
# positive floor so a belief is always a proper distribution and digamma/betaln
# stay finite.
_PARAM_FLOOR = 1e-6


@dataclass(frozen=True)
class BetaBelief:
    """An agent's belief about a continuous outcome ``v in [0, 1]``.

    Immutable. Conjugate updates (:meth:`update`) and reparameterizations return
    new instances rather than mutating in place, which keeps event logs
    replayable.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or not np.isfinite(self.beta):
            raise ValueError(f"alpha, beta must be finite; got ({self.alpha}, {self.beta})")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"alpha, beta must be positive (Beta is undefined otherwise); "
                f"got ({self.alpha}, {self.beta})"
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_mean_concentration(cls, mean: float, concentration: float) -> "BetaBelief":
        """Build a belief from a ``mean in (0, 1)`` and ``concentration > 0``.

        This is the natural parameterization for callers who think in terms of
        "what do I expect" (mean) and "how sure am I" (concentration):

            alpha = mean * concentration
            beta  = (1 - mean) * concentration
        """
        if not 0.0 < mean < 1.0:
            mean = float(np.clip(mean, _PARAM_FLOOR, 1.0 - _PARAM_FLOOR))
        if concentration <= 0:
            raise ValueError(f"concentration must be positive; got {concentration}")
        return cls(alpha=mean * concentration, beta=(1.0 - mean) * concentration)

    @classmethod
    def from_p(cls, p: float, concentration: float = 2.0) -> "BetaBelief":
        """Lift a legacy scalar label ``p`` into a belief with the given concentration.

        As ``concentration -> infinity`` the belief collapses to a point mass at
        ``p``, recovering the original soft-label formalism exactly. The default
        ``concentration = 2.0`` corresponds to a uniform-prior + one pseudo-count
        of evidence on each side scaled to ``p``.
        """
        return cls.from_mean_concentration(mean=p, concentration=concentration)

    @classmethod
    def uniform(cls) -> "BetaBelief":
        """The maximally uncertain belief, ``Beta(1, 1)`` — flat over ``[0, 1]``."""
        return cls(alpha=1.0, beta=1.0)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    @property
    def mean(self) -> float:
        """``alpha / (alpha + beta)`` — the analogue of the old point estimate ``p``."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def concentration(self) -> float:
        """``alpha + beta`` — epistemic certainty / total pseudo-evidence."""
        return self.alpha + self.beta

    @property
    def variance(self) -> float:
        """Variance of the belief. Decreases as concentration grows."""
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def mode(self) -> float | None:
        """The MAP outcome, or ``None`` when the belief is not unimodal interior.

        Beta is bimodal-at-the-edges when both params < 1, so there is no
        interior mode to report in that regime.
        """
        a, b = self.alpha, self.beta
        if a > 1 and b > 1:
            return (a - 1.0) / (a + b - 2.0)
        return None

    # ------------------------------------------------------------------
    # Distribution functions
    # ------------------------------------------------------------------
    def pdf(self, v: float | np.ndarray) -> np.ndarray:
        """Probability density at ``v``."""
        return stats.beta.pdf(v, self.alpha, self.beta)

    def cdf(self, v: float | np.ndarray) -> np.ndarray:
        """Cumulative distribution ``P(V <= v)``.

        Equal to the regularized incomplete beta function ``I_v(alpha, beta)``.
        """
        return stats.beta.cdf(v, self.alpha, self.beta)

    def tail_mass(self, tau: float) -> float:
        """``P(v < tau)`` — the downside-risk lever.

        This is what governance triggers on instead of a point estimate. A belief
        that is "confidently mediocre" (mean 0.5, high concentration) puts little
        mass below a low ``tau``; a belief that is "uncertain great-or-terrible"
        (mean 0.5, low concentration) puts a lot there. Same mean, very different
        tail mass — that distinction is invisible to scalar ``p``.
        """
        return float(special.betainc(self.alpha, self.beta, np.clip(tau, 0.0, 1.0)))

    def upper_tail_mass(self, tau: float) -> float:
        """``P(v > tau)`` — the upside complement of :meth:`tail_mass`."""
        return 1.0 - self.tail_mass(tau)

    def quantile(self, q: float) -> float:
        """Inverse CDF: the outcome ``v`` such that ``P(V <= v) = q``."""
        return float(stats.beta.ppf(q, self.alpha, self.beta))

    # ------------------------------------------------------------------
    # Expectations
    # ------------------------------------------------------------------
    def expect(self, func, n_grid: int = 512) -> float:
        """``E_F[func(v)]`` for an arbitrary scalar ``func`` over ``[0, 1]``.

        Uses fixed-grid trapezoidal quadrature against the density. For linear or
        low-degree-polynomial payoffs prefer the analytic helpers on the payoff
        engine; this is the general fallback for arbitrary surplus shapes.
        """
        v = np.linspace(0.0, 1.0, n_grid)
        w = self.pdf(v)
        fv = np.asarray(func(v), dtype=float)
        integrand = fv * w
        return float(np.trapezoid(integrand, v))

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------
    def update(self, positive: float = 0.0, negative: float = 0.0) -> "BetaBelief":
        """Conjugate update: add ``positive`` to alpha and ``negative`` to beta.

        ``positive``/``negative`` are (possibly fractional) pseudo-counts of
        good/bad evidence. Returns a new, more concentrated belief.
        """
        if positive < 0 or negative < 0:
            raise ValueError("evidence counts must be non-negative")
        return BetaBelief(alpha=self.alpha + positive, beta=self.beta + negative)

    def sample(self, rng: np.random.Generator, size: int | None = None) -> float | np.ndarray:
        """Draw outcome(s) ``v`` from the belief."""
        return rng.beta(self.alpha, self.beta, size=size)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, data: dict) -> "BetaBelief":
        return cls(alpha=float(data["alpha"]), beta=float(data["beta"]))

    def __repr__(self) -> str:
        return (
            f"BetaBelief(alpha={self.alpha:.4g}, beta={self.beta:.4g} | "
            f"mean={self.mean:.3f}, conc={self.concentration:.3g})"
        )
