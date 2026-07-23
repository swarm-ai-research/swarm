"""Distributional payoff engine: payoffs are ``E_F[surplus(v)]`` over the belief.

The original engine computed ``S_soft = p * s_plus - (1 - p) * s_minus`` from a
single scalar ``p``. Here surplus is a *function of the continuous outcome*
``v in [0, 1]`` and the payoff is its expectation under the agent's
:class:`~beta_swarm.belief.BetaBelief`:

    S = E_F[surplus(v)],   E = E_F[harm(v)]

For the default *linear* surplus this reduces exactly to the old formula with
``p`` replaced by the belief mean — so beta-swarm is a strict generalization.
The payoff only departs from the scalar case when the surplus is **nonlinear**,
because then the shape of the belief (not just its mean) moves the payoff. A
convex downside penalty, for instance, makes a diffuse "great-or-terrible"
belief strictly worse than a sharp "mediocre" one with the same mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel, model_validator

from beta_swarm.belief import BetaBelief

# A surplus/harm function maps an outcome (or array of outcomes) in [0, 1] to a
# real value. It must be vectorized over numpy arrays.
OutcomeFunc = Callable[[np.ndarray], np.ndarray]


class PayoffConfig(BaseModel):
    """Configuration mirroring the original ``PayoffConfig`` knobs.

    The economic parameters carry over unchanged; what differs is that
    ``s_plus`` / ``s_minus`` / ``h`` now scale a *function of v* rather than a
    binary outcome.
    """

    s_plus: float = 2.0   # surplus weight on the "good" end (v -> 1)
    s_minus: float = 1.0  # loss weight on the "bad" end (v -> 0)
    h: float = 2.0        # harm-externality magnitude
    theta: float = 0.5    # surplus split (initiator share)
    rho_a: float = 0.0    # externality internalization (initiator)
    rho_b: float = 0.0    # externality internalization (counterparty)
    w_rep: float = 1.0    # reputation weight
    # Curvature on the surplus downside. 1.0 -> linear (recovers old formalism).
    # >1.0 -> convex penalty on low outcomes, so tail mass / belief spread bites.
    downside_curvature: float = 1.0

    @model_validator(mode="after")
    def _check(self) -> "PayoffConfig":
        if self.s_plus < 0 or self.s_minus < 0 or self.h < 0:
            raise ValueError("s_plus, s_minus, h must be non-negative")
        if not 0 <= self.theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        if not 0 <= self.rho_a <= 1 or not 0 <= self.rho_b <= 1:
            raise ValueError("rho_a, rho_b must be in [0, 1]")
        if not 0 <= self.w_rep <= 100:
            raise ValueError("w_rep must be in [0, 100]")
        if self.downside_curvature <= 0:
            raise ValueError("downside_curvature must be positive")
        return self


@dataclass
class PayoffBreakdown:
    """Detailed breakdown of a distributional payoff computation."""

    expected_surplus: float
    expected_harm: float
    surplus_share: float
    transfer: float
    governance_cost: float
    externality_cost: float
    reputation_bonus: float
    total: float


class DistributionalPayoffEngine:
    """Computes payoffs as expectations of surplus/harm under a Beta belief.

    Pass a custom ``surplus_fn`` / ``harm_fn`` to study arbitrary outcome
    valuations; otherwise the defaults below are used.
    """

    def __init__(
        self,
        config: Optional[PayoffConfig] = None,
        surplus_fn: Optional[OutcomeFunc] = None,
        harm_fn: Optional[OutcomeFunc] = None,
    ):
        self.config = config or PayoffConfig()
        self._surplus_fn = surplus_fn
        self._harm_fn = harm_fn
        # Whether we can take the analytic (mean-only) shortcut.
        self._linear = (
            surplus_fn is None
            and harm_fn is None
            and self.config.downside_curvature == 1.0
        )

    # ------------------------------------------------------------------
    # Default outcome valuations
    # ------------------------------------------------------------------
    def surplus_fn(self, v: np.ndarray) -> np.ndarray:
        """Default surplus: linear reward minus a (possibly convex) downside.

        ``surplus(v) = s_plus * v - s_minus * (1 - v)^gamma``

        With ``gamma = 1`` this is ``s_plus*v - s_minus*(1-v)``, whose expectation
        is ``s_plus*mean - s_minus*(1-mean)`` — exactly the old ``S_soft`` with
        ``p = mean``. With ``gamma > 1`` low outcomes are penalized superlinearly,
        so the left tail of the belief — not just its mean — drives the payoff.
        """
        if self._surplus_fn is not None:
            return self._surplus_fn(v)
        g = self.config.downside_curvature
        return self.config.s_plus * v - self.config.s_minus * np.power(1.0 - v, g)

    def harm_fn(self, v: np.ndarray) -> np.ndarray:
        """Default harm externality: ``h * (1 - v)`` (linear in badness)."""
        if self._harm_fn is not None:
            return self._harm_fn(v)
        return self.config.h * (1.0 - v)

    # ------------------------------------------------------------------
    # Expectations under the belief
    # ------------------------------------------------------------------
    def expected_surplus(self, belief: BetaBelief) -> float:
        """``E_F[surplus(v)]``."""
        if self._linear:
            m = belief.mean
            return self.config.s_plus * m - self.config.s_minus * (1.0 - m)
        return belief.expect(self.surplus_fn)

    def expected_harm(self, belief: BetaBelief) -> float:
        """``E_F[harm(v)]``. For the default linear harm this is ``h * (1 - mean)``."""
        if self._harm_fn is None:
            return self.config.h * (1.0 - belief.mean)
        return belief.expect(self.harm_fn)

    # ------------------------------------------------------------------
    # Player payoffs
    # ------------------------------------------------------------------
    def payoffs_both(
        self,
        belief: BetaBelief,
        tau: float = 0.0,
        c_a: float = 0.0,
        c_b: float = 0.0,
        r_a: float = 0.0,
        r_b: float = 0.0,
    ) -> tuple[float, float]:
        """Initiator and counterparty payoffs in one call.

            pi_a = theta * S - tau - c_a - rho_a * E + w_rep * r_a
            pi_b = (1-theta) * S + tau - c_b - rho_b * E + w_rep * r_b
        """
        cfg = self.config
        S = self.expected_surplus(belief)
        E = self.expected_harm(belief)
        pi_a = cfg.theta * S - tau - c_a - cfg.rho_a * E + cfg.w_rep * r_a
        pi_b = (1.0 - cfg.theta) * S + tau - c_b - cfg.rho_b * E + cfg.w_rep * r_b
        return pi_a, pi_b

    def breakdown(
        self,
        belief: BetaBelief,
        *,
        initiator: bool,
        tau: float = 0.0,
        cost: float = 0.0,
        reputation: float = 0.0,
    ) -> PayoffBreakdown:
        """Itemized payoff for one player."""
        cfg = self.config
        S = self.expected_surplus(belief)
        E = self.expected_harm(belief)
        if initiator:
            share = cfg.theta * S
            transfer = -tau
            rho = cfg.rho_a
        else:
            share = (1.0 - cfg.theta) * S
            transfer = tau
            rho = cfg.rho_b
        externality_cost = rho * E
        reputation_bonus = cfg.w_rep * reputation
        total = share + transfer - cost - externality_cost + reputation_bonus
        return PayoffBreakdown(
            expected_surplus=S,
            expected_harm=E,
            surplus_share=share,
            transfer=transfer,
            governance_cost=cost,
            externality_cost=externality_cost,
            reputation_bonus=reputation_bonus,
            total=total,
        )

    def social_surplus(self, belief: BetaBelief) -> float:
        """``E_F[surplus(v)] - E_F[harm(v)]`` — welfare including unpriced harm."""
        return self.expected_surplus(belief) - self.expected_harm(belief)

    # ------------------------------------------------------------------
    # Break-even analysis
    # ------------------------------------------------------------------
    def break_even_mean(self) -> float:
        """Belief mean at which the (linear) expected surplus is zero.

        ``mean = s_minus / (s_plus + s_minus)``. For nonlinear surplus this is an
        approximation about the mean; use it as a reference threshold only.
        """
        denom = self.config.s_plus + self.config.s_minus
        return 0.5 if denom == 0 else self.config.s_minus / denom
