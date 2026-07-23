"""Tests for the distributional payoff engine, including reduction to the scalar case."""

import numpy as np
import pytest

from beta_swarm.belief import BetaBelief
from beta_swarm.payoff import DistributionalPayoffEngine, PayoffConfig


def _scalar_S_soft(p, cfg):
    return p * cfg.s_plus - (1 - p) * cfg.s_minus


def test_linear_surplus_reduces_to_scalar_formalism():
    """With linear surplus, E_F[surplus] == old S_soft with p = mean."""
    cfg = PayoffConfig()
    eng = DistributionalPayoffEngine(cfg)
    for mean in [0.1, 0.5, 0.9]:
        b = BetaBelief.from_mean_concentration(mean, 8.0)
        assert eng.expected_surplus(b) == pytest.approx(_scalar_S_soft(mean, cfg))


def test_linear_harm_reduces_to_scalar():
    cfg = PayoffConfig()
    eng = DistributionalPayoffEngine(cfg)
    b = BetaBelief.from_mean_concentration(0.6, 5.0)
    assert eng.expected_harm(b) == pytest.approx(cfg.h * (1 - 0.6))


def test_convex_downside_penalizes_spread_at_equal_mean():
    """The distributional payoff: a diffuse belief is worse than a sharp one
    with the same mean, when the downside is convex."""
    cfg = PayoffConfig(downside_curvature=2.0)
    eng = DistributionalPayoffEngine(cfg)
    sharp = BetaBelief.from_mean_concentration(0.5, 200.0)
    diffuse = BetaBelief.from_mean_concentration(0.5, 2.0)
    assert sharp.mean == pytest.approx(diffuse.mean)
    # Convex downside: more spread -> lower expected surplus.
    assert eng.expected_surplus(diffuse) < eng.expected_surplus(sharp)


def test_linear_is_spread_insensitive():
    """Sanity check the converse: with linear surplus, spread does not matter."""
    eng = DistributionalPayoffEngine(PayoffConfig(downside_curvature=1.0))
    sharp = BetaBelief.from_mean_concentration(0.5, 200.0)
    diffuse = BetaBelief.from_mean_concentration(0.5, 2.0)
    assert eng.expected_surplus(sharp) == pytest.approx(eng.expected_surplus(diffuse))


def test_payoffs_both_split_and_transfer():
    eng = DistributionalPayoffEngine(PayoffConfig(theta=0.5))
    b = BetaBelief.from_mean_concentration(0.8, 10.0)
    pi_a, pi_b = eng.payoffs_both(b, tau=0.3)
    S = eng.expected_surplus(b)
    assert pi_a == pytest.approx(0.5 * S - 0.3)
    assert pi_b == pytest.approx(0.5 * S + 0.3)


def test_breakdown_total_consistent():
    eng = DistributionalPayoffEngine()
    b = BetaBelief.from_mean_concentration(0.7, 6.0)
    bd = eng.breakdown(b, initiator=True, tau=0.2, cost=0.1, reputation=0.5)
    expected = (
        bd.surplus_share + bd.transfer - bd.governance_cost
        - bd.externality_cost + bd.reputation_bonus
    )
    assert bd.total == pytest.approx(expected)


def test_custom_surplus_fn_used():
    eng = DistributionalPayoffEngine(surplus_fn=lambda v: np.ones_like(v))
    b = BetaBelief(2.0, 3.0)
    assert eng.expected_surplus(b) == pytest.approx(1.0, abs=1e-4)


def test_break_even_mean():
    eng = DistributionalPayoffEngine(PayoffConfig(s_plus=3.0, s_minus=1.0))
    assert eng.break_even_mean() == pytest.approx(0.25)
