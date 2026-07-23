"""Tests for distributional metrics: Wasserstein/KL quality gap, tail toxicity, CRPS."""

import pytest

from beta_swarm.belief import BetaBelief
from beta_swarm.interaction import BetaInteraction
from beta_swarm.metrics import (
    DistributionalMetrics,
    beta_kl,
    kl_mixtures,
    wasserstein1_mixtures,
)


def _ix(mean, conc, accepted, gt=None):
    return BetaInteraction(
        belief=BetaBelief.from_mean_concentration(mean, conc),
        accepted=accepted,
        ground_truth=gt,
    )


def test_beta_kl_zero_for_identical():
    b = BetaBelief(3.0, 5.0)
    assert beta_kl(b, b) == pytest.approx(0.0, abs=1e-9)


def test_beta_kl_positive_for_different():
    assert beta_kl(BetaBelief(2.0, 5.0), BetaBelief(5.0, 2.0)) > 0


def test_wasserstein_zero_for_identical_mixtures():
    beliefs = [BetaBelief(2.0, 3.0), BetaBelief(4.0, 4.0)]
    assert wasserstein1_mixtures(beliefs, beliefs) == pytest.approx(0.0, abs=1e-3)


def test_wasserstein_detects_separated_mixtures():
    high = [BetaBelief.from_mean_concentration(0.9, 20.0)]
    low = [BetaBelief.from_mean_concentration(0.1, 20.0)]
    # Means differ by 0.8; W1 should be close to that for sharp beliefs.
    assert wasserstein1_mixtures(high, low) == pytest.approx(0.8, abs=0.05)


def test_quality_gaps_on_adverse_selection():
    m = DistributionalMetrics()
    # Accepted are LOW quality, rejected are HIGH: adverse selection.
    interactions = [
        _ix(0.2, 20.0, accepted=True),
        _ix(0.25, 20.0, accepted=True),
        _ix(0.85, 20.0, accepted=False),
        _ix(0.9, 20.0, accepted=False),
    ]
    # Signed mean gap is negative (adverse selection).
    assert m.mean_quality_gap(interactions) < 0
    # Distance metrics are positive (the two populations are far apart).
    assert m.quality_gap_wasserstein(interactions) > 0
    assert m.quality_gap_kl(interactions) > 0


def test_quality_gap_zero_when_no_rejected():
    m = DistributionalMetrics()
    interactions = [_ix(0.5, 10.0, accepted=True)]
    assert m.quality_gap_wasserstein(interactions) == 0.0
    assert m.mean_quality_gap(interactions) == 0.0


def test_expected_tail_mass_sees_uncertainty():
    """Two accepted populations with equal mean but different concentration
    have different distributional toxicity."""
    m = DistributionalMetrics()
    sharp = [_ix(0.6, 200.0, accepted=True)]
    diffuse = [_ix(0.6, 2.0, accepted=True)]
    assert m.expected_tail_mass(diffuse, tau=0.25) > m.expected_tail_mass(sharp, tau=0.25)
    # The legacy point toxicity cannot tell them apart.
    assert m.toxicity_rate(diffuse) == pytest.approx(m.toxicity_rate(sharp))


def test_crps_rewards_calibrated_sharp_belief():
    m = DistributionalMetrics()
    # Ground truth 0.8: a sharp belief centered there scores better (lower CRPS)
    # than a sharp belief centered far away.
    good = [_ix(0.8, 100.0, accepted=True, gt=0.8)]
    bad = [_ix(0.2, 100.0, accepted=True, gt=0.8)]
    assert m.crps(good) < m.crps(bad)


def test_crps_none_without_ground_truth():
    m = DistributionalMetrics()
    assert m.crps([_ix(0.5, 10.0, accepted=True)]) is None


def test_kl_mixtures_nonnegative():
    a = [BetaBelief(2.0, 5.0)]
    b = [BetaBelief(5.0, 2.0)]
    assert kl_mixtures(a, b) >= 0
