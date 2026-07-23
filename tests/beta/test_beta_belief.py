"""Tests for the BetaBelief primitive."""

import numpy as np
import pytest

from beta_swarm.belief import BetaBelief


def test_mean_concentration_roundtrip():
    b = BetaBelief.from_mean_concentration(mean=0.7, concentration=10.0)
    assert b.mean == pytest.approx(0.7)
    assert b.concentration == pytest.approx(10.0)
    assert b.alpha == pytest.approx(7.0)
    assert b.beta == pytest.approx(3.0)


def test_from_p_recovers_mean():
    b = BetaBelief.from_p(0.3, concentration=4.0)
    assert b.mean == pytest.approx(0.3)


def test_high_concentration_collapses_variance():
    diffuse = BetaBelief.from_mean_concentration(0.5, 2.0)
    sharp = BetaBelief.from_mean_concentration(0.5, 200.0)
    assert sharp.variance < diffuse.variance
    assert sharp.mean == pytest.approx(diffuse.mean)


def test_confident_mediocre_vs_uncertain_have_same_mean_different_tails():
    """The whole motivation: same mean, very different downside tail mass."""
    confident_mediocre = BetaBelief.from_mean_concentration(0.5, 200.0)
    uncertain = BetaBelief.from_mean_concentration(0.5, 2.0)
    assert confident_mediocre.mean == pytest.approx(uncertain.mean)
    tau = 0.25
    assert confident_mediocre.tail_mass(tau) < uncertain.tail_mass(tau)


def test_tail_mass_matches_cdf():
    b = BetaBelief(2.0, 5.0)
    assert b.tail_mass(0.3) == pytest.approx(float(b.cdf(0.3)))


def test_tail_and_upper_tail_sum_to_one():
    b = BetaBelief(3.0, 4.0)
    assert b.tail_mass(0.4) + b.upper_tail_mass(0.4) == pytest.approx(1.0)


def test_expect_linear_matches_mean():
    b = BetaBelief(4.0, 6.0)
    # E[v] should equal the mean
    assert b.expect(lambda v: v) == pytest.approx(b.mean, abs=1e-4)


def test_update_increases_concentration_and_shifts_mean():
    b = BetaBelief.uniform()
    updated = b.update(positive=8.0, negative=2.0)
    assert updated.concentration > b.concentration
    assert updated.mean > b.mean


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        BetaBelief(0.0, 1.0)
    with pytest.raises(ValueError):
        BetaBelief(-1.0, 2.0)


def test_quantile_inverts_cdf():
    b = BetaBelief(3.0, 3.0)
    q = b.quantile(0.4)
    assert float(b.cdf(q)) == pytest.approx(0.4, abs=1e-4)


def test_serialization_roundtrip():
    b = BetaBelief(2.5, 7.5)
    assert BetaBelief.from_dict(b.to_dict()) == b


def test_sampling_is_in_unit_interval():
    rng = np.random.default_rng(0)
    samples = BetaBelief(2.0, 2.0).sample(rng, size=1000)
    assert np.all((samples >= 0) & (samples <= 1))
