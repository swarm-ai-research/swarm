"""Tests for distributional calibration."""

import numpy as np
import pytest

from beta_swarm.agents import Archetype
from beta_swarm.belief import BetaBelief
from beta_swarm.calibration import (
    calibrate,
    calibrate_interactions,
    crps,
    fidelity_run,
    pit_values,
    sweep_evidence_scale,
    tail_reliability_bins,
)
from beta_swarm.interaction import BetaInteraction


def _self_consistent(n: int = 2000, seed: int = 0):
    """Beliefs whose outcomes are drawn from the beliefs themselves.

    By construction this is perfectly calibrated (up to sampling noise).
    """
    rng = np.random.default_rng(seed)
    beliefs, outcomes = [], []
    for _ in range(n):
        b = BetaBelief.from_mean_concentration(rng.uniform(0.2, 0.8), rng.uniform(3, 30))
        beliefs.append(b)
        outcomes.append(float(b.sample(rng)))
    return beliefs, outcomes


def test_self_consistent_beliefs_are_calibrated():
    beliefs, outcomes = _self_consistent()
    report = calibrate(beliefs, outcomes)
    assert report.pit_deviation < 0.05
    assert report.tail_ece < 0.03
    # PIT is roughly uniform: every decile holds roughly a tenth of the mass.
    assert max(report.pit_freqs) < 0.15


def test_overconfidence_and_underconfidence_raise_pit_deviation():
    rng = np.random.default_rng(1)
    truth = [BetaBelief.from_mean_concentration(0.5, 10.0) for _ in range(1500)]
    outcomes = [float(b.sample(rng)) for b in truth]
    honest = calibrate(truth, outcomes).pit_deviation
    sharper = calibrate(
        [BetaBelief.from_mean_concentration(0.5, 80.0)] * len(truth), outcomes
    ).pit_deviation
    vaguer = calibrate(
        [BetaBelief.from_mean_concentration(0.5, 2.0)] * len(truth), outcomes
    ).pit_deviation
    assert sharper > honest + 0.1
    assert vaguer > honest + 0.1


def test_tail_bins_track_reality_when_calibrated():
    beliefs, outcomes = _self_consistent()
    for b in tail_reliability_bins(beliefs, outcomes, tau=0.4):
        if b.n >= 100:
            assert b.observed_rate == pytest.approx(b.mean_predicted, abs=0.08)


def test_crps_rewards_sharp_truthful_beliefs():
    outcome = 0.7
    sharp = BetaBelief.from_mean_concentration(0.7, 100.0)
    diffuse = BetaBelief.from_mean_concentration(0.7, 2.0)
    wrong = BetaBelief.from_mean_concentration(0.2, 100.0)
    assert crps([sharp], [outcome]) < crps([diffuse], [outcome]) < crps([wrong], [outcome])


def test_calibrate_interactions_requires_ground_truth():
    no_truth = [BetaInteraction(initiator="a", counterparty="b")]
    assert calibrate_interactions(no_truth) is None


def test_calibrate_interactions_can_score_proxy_beliefs():
    sharp = BetaBelief.from_mean_concentration(0.7, 50.0)
    diffuse = BetaBelief.from_mean_concentration(0.3, 2.0)
    log = [
        BetaInteraction(
            initiator="a", counterparty="b",
            belief=sharp, proxy_belief=diffuse, ground_truth=0.7,
        )
        for _ in range(20)
    ]
    decision = calibrate_interactions(log)
    raw = calibrate_interactions(log, use_proxy_belief=True)
    assert decision.crps < raw.crps
    assert decision.sharpness > raw.sharpness


def test_fidelity_run_is_deterministic():
    a = fidelity_run(200, seed=5)
    b = fidelity_run(200, seed=5)
    assert a[1] == b[1]
    assert [x.alpha for x in a[0]] == [x.alpha for x in b[0]]


def test_deceptive_traffic_destroys_calibration():
    faithful = calibrate(*fidelity_run(1500, seed=0))
    adversarial = calibrate(
        *fidelity_run(1500, archetypes=[Archetype.DECEPTIVE], seed=0)
    )
    assert adversarial.crps > 3 * faithful.crps
    assert adversarial.pit_deviation > faithful.pit_deviation + 0.3
    # The signature: outcomes land below the belief's support, PIT piles at 0.
    assert adversarial.pit_freqs[0] > 0.8


def test_default_evidence_scale_is_underconfident():
    """The concentration knob's headline: default 1.5 is far from the optimum."""
    swept = dict(sweep_evidence_scale([1.5, 14.0], n=1500, seed=0))
    assert swept[14.0].crps < swept[1.5].crps
    assert swept[14.0].pit_deviation < swept[1.5].pit_deviation
    assert swept[14.0].tail_ece < swept[1.5].tail_ece
    # Paired design: both reports scored the same number of identical draws.
    assert swept[14.0].n_total == swept[1.5].n_total


def test_pit_values_range_and_length():
    beliefs, outcomes = _self_consistent(n=50)
    pit = pit_values(beliefs, outcomes)
    assert len(pit) == 50
    assert np.all((pit >= 0.0) & (pit <= 1.0))
    with pytest.raises(ValueError):
        pit_values(beliefs, outcomes[:-1])
