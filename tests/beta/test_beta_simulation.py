"""Tests for the closed-loop simulation."""

import pytest

from beta_swarm.agents import Archetype, make_population
from beta_swarm.belief import BetaBelief
from beta_swarm.governance import TailMassGovernor
from beta_swarm.simulation import (
    Simulation,
    SimulationConfig,
    decay_belief,
    pool_beliefs,
)


def _population():
    return make_population(
        {Archetype.HONEST: 4, Archetype.MEDIOCRE: 3, Archetype.DECEPTIVE: 3}
    )


def _mean_governor():
    """A pure mean floor — the legacy scalar-p lever (tail cap never binds)."""
    return TailMassGovernor(tau=0.4, max_tail_mass=1.0, min_mean=0.5)


def _tail_governor():
    return TailMassGovernor(
        tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=20.0
    )


# ----------------------------------------------------------------------
# Belief algebra
# ----------------------------------------------------------------------
def test_pooling_with_uniform_is_identity():
    b = BetaBelief(alpha=3.0, beta=2.0)
    pooled = pool_beliefs(b, BetaBelief.uniform())
    assert pooled.alpha == pytest.approx(b.alpha)
    assert pooled.beta == pytest.approx(b.beta)


def test_pooling_adds_evidence():
    pooled = pool_beliefs(BetaBelief(3.0, 2.0), BetaBelief(2.0, 4.0))
    assert pooled.alpha == pytest.approx(4.0)
    assert pooled.beta == pytest.approx(5.0)


def test_decay_discounts_toward_uniform():
    b = BetaBelief(11.0, 6.0)
    assert decay_belief(b, 1.0).alpha == pytest.approx(b.alpha)
    half = decay_belief(b, 0.5)
    assert half.alpha == pytest.approx(6.0)
    assert half.beta == pytest.approx(3.5)
    assert half.concentration < b.concentration
    gone = decay_belief(b, 0.0)
    assert gone.alpha == pytest.approx(1.0)
    assert gone.beta == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Loop mechanics
# ----------------------------------------------------------------------
def test_run_is_deterministic_given_seed():
    cfg = SimulationConfig(n_epochs=4, steps_per_epoch=3, seed=11)
    a = Simulation(_population(), _tail_governor(), cfg).run()
    b = Simulation(_population(), _tail_governor(), cfg).run()
    assert [e.realized_welfare for e in a.epoch_reports] == [
        e.realized_welfare for e in b.epoch_reports
    ]
    assert [i.ground_truth for i in a.interactions] == [
        i.ground_truth for i in b.interactions
    ]


def test_no_defer_configured_means_no_audits():
    cfg = SimulationConfig(n_epochs=3, steps_per_epoch=3, seed=0)
    result = Simulation(_population(), _mean_governor(), cfg).run()
    assert sum(e.n_audits for e in result.epoch_reports) == 0


def test_audits_charge_the_initiator():
    cfg = SimulationConfig(n_epochs=3, steps_per_epoch=3, seed=0)
    result = Simulation(_population(), _tail_governor(), cfg).run()
    audited = [i for i in result.interactions if i.metadata["audited"]]
    assert audited, "diffuse early beliefs should trigger audits"
    assert all(i.c_a == pytest.approx(cfg.audit_cost) for i in audited)


def test_unreachable_defer_threshold_shuts_the_ecosystem_down():
    """If one audit cannot reach the defer bar, everything loops to REJECT."""
    gov = TailMassGovernor(
        tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=100.0
    )
    cfg = SimulationConfig(n_epochs=2, steps_per_epoch=3, seed=0, audit_strength=4.0)
    result = Simulation(_population(), gov, cfg).run()
    assert all(not i.accepted for i in result.interactions)


def test_reputation_tracks_realized_quality():
    """Under a permissive governor, reputation beliefs converge on true quality."""
    cfg = SimulationConfig(n_epochs=8, steps_per_epoch=5, seed=3)
    result = Simulation(_population(), _mean_governor(), cfg).run()
    honest = [
        result.reputation[a.agent_id].mean
        for a in _population()
        if a.archetype is Archetype.HONEST
    ]
    deceptive = [
        result.reputation[a.agent_id].mean
        for a in _population()
        if a.archetype is Archetype.DECEPTIVE
    ]
    assert min(honest) > max(deceptive)


# ----------------------------------------------------------------------
# The headline result
# ----------------------------------------------------------------------
def test_tail_governor_beats_mean_governor():
    """The distributional lever: higher welfare, lower toxicity, no deception tax.

    Deceptive agents inflate the belief mean, so the mean governor admits them
    until pooled reputation catches up — paying for the damage meanwhile. The
    tail governor defers their diffuse beliefs to audit and shuts them out from
    epoch 0. Directionally robust across seeds (checked over seeds 0-7).
    """
    cfg = SimulationConfig(seed=2)
    mean_result = Simulation(_population(), _mean_governor(), cfg).run()
    tail_result = Simulation(_population(), _tail_governor(), cfg).run()

    assert tail_result.total_welfare > mean_result.total_welfare
    assert tail_result.overall_toxicity < mean_result.overall_toxicity
    # The mean governor pays a deception tax the tail governor avoids.
    assert mean_result.acceptance_rate(Archetype.DECEPTIVE) > 0.05
    assert tail_result.acceptance_rate(Archetype.DECEPTIVE) < 0.03
    # Both keep serving honest agents.
    assert mean_result.acceptance_rate(Archetype.HONEST) > 0.95
    assert tail_result.acceptance_rate(Archetype.HONEST) > 0.95
    # Audited beliefs are better calibrated: lower CRPS against ground truth.
    mean_crps = [e.crps for e in mean_result.epoch_reports if e.crps is not None]
    tail_crps = [e.crps for e in tail_result.epoch_reports if e.crps is not None]
    assert sum(tail_crps) / len(tail_crps) < sum(mean_crps) / len(mean_crps)


def test_epoch_zero_gap():
    """The whole point, sharpest at the start: before any reputation exists,
    the mean governor admits deceptive work wholesale; the tail governor none."""
    cfg = SimulationConfig(seed=2)
    mean_e0 = Simulation(_population(), _mean_governor(), cfg).run().epoch_reports[0]
    tail_e0 = Simulation(_population(), _tail_governor(), cfg).run().epoch_reports[0]
    assert mean_e0.acceptance_by_archetype["deceptive"] > 0.5
    assert tail_e0.acceptance_by_archetype["deceptive"] == 0.0
