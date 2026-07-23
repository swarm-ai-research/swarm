"""Tests for distributional governance levers."""

import pytest

from beta_swarm.belief import BetaBelief
from beta_swarm.governance import (
    Decision,
    TailMassGovernor,
    ValueAtRiskGovernor,
)


def test_tail_mass_governor_distinguishes_equal_means():
    """The capability scalar-p lacks: same mean, different admission decision."""
    gov = TailMassGovernor(tau=0.25, max_tail_mass=0.15, min_mean=None)
    confident_mediocre = BetaBelief.from_mean_concentration(0.5, 200.0)
    uncertain = BetaBelief.from_mean_concentration(0.5, 2.0)
    # Same mean...
    assert confident_mediocre.mean == pytest.approx(uncertain.mean)
    # ...but different verdict, because tail mass differs.
    assert gov.evaluate(confident_mediocre).decision is Decision.ACCEPT
    assert gov.evaluate(uncertain).decision is Decision.REJECT


def test_mean_threshold_would_treat_them_identically():
    """Contrast: a pure mean floor cannot separate the two beliefs above."""
    confident_mediocre = BetaBelief.from_mean_concentration(0.5, 200.0)
    uncertain = BetaBelief.from_mean_concentration(0.5, 2.0)
    # A legacy mean-only policy accepts/rejects both the same way.
    assert (confident_mediocre.mean >= 0.5) == (uncertain.mean >= 0.5)


def test_defer_on_low_concentration():
    gov = TailMassGovernor(
        tau=0.25, max_tail_mass=0.9, defer_below_concentration=5.0
    )
    diffuse = BetaBelief.from_mean_concentration(0.6, 2.0)
    assert gov.evaluate(diffuse).decision is Decision.DEFER


def test_min_mean_floor_rejects_low_mean():
    gov = TailMassGovernor(tau=0.1, max_tail_mass=1.0, min_mean=0.6)
    low = BetaBelief.from_mean_concentration(0.4, 50.0)
    assert gov.evaluate(low).decision is Decision.REJECT


def test_verdict_records_lever_readings():
    gov = TailMassGovernor()
    v = gov.evaluate(BetaBelief.from_mean_concentration(0.7, 10.0))
    assert 0 <= v.tail_mass <= 1
    assert v.mean == pytest.approx(0.7)
    assert v.concentration == pytest.approx(10.0)


def test_value_at_risk_governor():
    gov = ValueAtRiskGovernor(alpha=0.05, min_var_outcome=0.3)
    sharp_good = BetaBelief.from_mean_concentration(0.7, 100.0)
    diffuse = BetaBelief.from_mean_concentration(0.7, 2.0)
    # The sharp belief has a higher 5th-percentile outcome.
    assert gov.value_at_risk(sharp_good) > gov.value_at_risk(diffuse)
    assert gov.accepts(sharp_good)


def test_invalid_config():
    with pytest.raises(ValueError):
        TailMassGovernor(tau=1.5)
    with pytest.raises(ValueError):
        ValueAtRiskGovernor(alpha=0.0)
