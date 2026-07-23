"""Tests for the adaptive governance controller."""

import numpy as np
import pytest

from beta_swarm.adaptive import AdaptiveController, ProposalStatus
from beta_swarm.agents import Archetype, SimAgent, make_population
from beta_swarm.governance import TailMassGovernor
from beta_swarm.simulation import EpochReport, Simulation, SimulationConfig


def _report(epoch: int, toxicity: float, welfare: float = 5.0) -> EpochReport:
    return EpochReport(
        epoch=epoch,
        n_accepted=30,
        n_rejected=20,
        n_audits=0,
        realized_welfare=welfare,
        realized_toxicity=toxicity,
        mean_quality_gap=0.0,
        wasserstein_gap=0.0,
        expected_tail_mass=0.0,
        crps=None,
        acceptance_by_archetype={},
    )


def _governor(mtm: float = 0.3) -> TailMassGovernor:
    return TailMassGovernor(tau=0.4, max_tail_mass=mtm, min_mean=None, defer_below_concentration=20.0)


def _drive(controller: AdaptiveController, governor, toxicity_seq, welfare_seq=None):
    """Feed a toxicity sequence epoch by epoch."""
    welfare_seq = welfare_seq or [5.0] * len(toxicity_seq)
    for epoch, (tox, wf) in enumerate(zip(toxicity_seq, welfare_seq)):
        controller.on_epoch_end(_report(epoch, tox, wf), governor, epoch)


# ----------------------------------------------------------------------
# Contemplation direction
# ----------------------------------------------------------------------
def test_high_realized_risk_tightens_the_budget():
    gov = _governor(0.4)
    ctrl = AdaptiveController(target_toxicity=0.3, band=0.02, warmup_epochs=2, min_evidence_epochs=3)
    _drive(ctrl, gov, [0.5] * 6)  # risk well above target
    assert gov.max_tail_mass < 0.4  # tightened
    assert ctrl.history and ctrl.history[0].new_value < ctrl.history[0].old_value


def test_low_realized_risk_loosens_the_budget():
    gov = _governor(0.2)
    ctrl = AdaptiveController(target_toxicity=0.4, band=0.02, warmup_epochs=2, min_evidence_epochs=3)
    _drive(ctrl, gov, [0.2] * 6)  # risk well below target
    assert gov.max_tail_mass > 0.2  # loosened


def test_on_target_proposes_nothing():
    gov = _governor(0.3)
    ctrl = AdaptiveController(target_toxicity=0.35, band=0.03, warmup_epochs=2, min_evidence_epochs=3)
    _drive(ctrl, gov, [0.35] * 8)  # squarely in-band
    assert ctrl.history == []
    assert gov.max_tail_mass == pytest.approx(0.3)


def test_warmup_suppresses_early_reaction():
    gov = _governor(0.4)
    ctrl = AdaptiveController(target_toxicity=0.3, warmup_epochs=10, min_evidence_epochs=3)
    _drive(ctrl, gov, [0.5] * 8)  # all within warmup
    assert ctrl.history == []


# ----------------------------------------------------------------------
# Crystallize / revert discipline
# ----------------------------------------------------------------------
def test_change_that_helps_is_crystallized():
    gov = _governor(0.4)
    ctrl = AdaptiveController(
        target_toxicity=0.3, band=0.02, warmup_epochs=0,
        min_evidence_epochs=3, contemplation_interval=3, evaluation_window=3,
    )
    # High risk before, then the change coincides with risk falling to target.
    _drive(ctrl, gov, [0.5, 0.5, 0.5, 0.5, 0.30, 0.30, 0.30])
    assert ctrl.n_crystallized >= 1
    assert ctrl.history[0].status is ProposalStatus.CRYSTALLIZED


def test_change_that_overshoots_is_reverted():
    gov = _governor(0.4)
    original = gov.max_tail_mass
    ctrl = AdaptiveController(
        target_toxicity=0.3, band=0.02, warmup_epochs=0,
        min_evidence_epochs=3, contemplation_interval=3, evaluation_window=3,
    )
    # Risk above target, then after tightening the realized risk overshoots far
    # past target (closeness gets worse), so the proposal should be rolled back.
    # The sequence ends exactly at the resolution epoch, before any new proposal.
    _drive(ctrl, gov, [0.5, 0.5, 0.5, 0.05, 0.05, 0.05])
    assert ctrl.n_reverted >= 1
    reverted = ctrl.history[0]
    assert reverted.status is ProposalStatus.REVERTED
    # A reverted change restores the pre-change value.
    assert gov.max_tail_mass == pytest.approx(original)


def test_respects_bounds():
    gov = _governor(0.08)
    ctrl = AdaptiveController(
        target_toxicity=0.1, band=0.01, bounds=(0.05, 0.6), step=0.05,
        warmup_epochs=0, min_evidence_epochs=3,
    )
    _drive(ctrl, gov, [0.5] * 6)  # wants to tighten hard
    assert gov.max_tail_mass >= 0.05  # never below the floor


def test_one_proposal_active_at_a_time():
    gov = _governor(0.5)
    ctrl = AdaptiveController(
        target_toxicity=0.3, band=0.02, warmup_epochs=0,
        min_evidence_epochs=3, contemplation_interval=1, evaluation_window=4,
    )
    _drive(ctrl, gov, [0.5] * 6)
    # With a 4-epoch eval window and 6 epochs, at most one proposal resolves;
    # no second proposal is opened while one is active.
    active = [p for p in ctrl.history if p.status is ProposalStatus.ACTIVE]
    assert len(active) <= 1


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------
def _hetero_population():
    pop = make_population({Archetype.HONEST: 4, Archetype.DECEPTIVE: 2})
    for i, mean in enumerate(np.linspace(0.35, 0.68, 8)):
        pop.append(SimAgent(f"borderline-{i}", Archetype.MEDIOCRE, mean * 12.0, (1 - mean) * 12.0))
    return pop


def test_converges_toward_target_from_both_directions():
    def run(start):
        gov = TailMassGovernor(tau=0.4, max_tail_mass=start, min_mean=None, defer_below_concentration=20.0)
        ctrl = AdaptiveController(target_toxicity=0.37, band=0.01)
        result = Simulation(
            _hetero_population(), gov, SimulationConfig(seed=7, n_epochs=70), controller=ctrl
        ).run()
        late = float(np.mean([e.realized_toxicity for e in result.epoch_reports[-15:]]))
        return gov.max_tail_mass, late

    loose_mtm, loose_tox = run(0.55)
    tight_mtm, tight_tox = run(0.05)
    # Both misconfigured starts land near the target and near each other.
    assert abs(loose_tox - 0.37) < 0.02
    assert abs(tight_tox - 0.37) < 0.02
    assert abs(loose_mtm - tight_mtm) < 0.15


def test_no_controller_leaves_governor_untouched():
    gov = TailMassGovernor(tau=0.4, max_tail_mass=0.3, min_mean=None, defer_below_concentration=20.0)
    Simulation(make_population({Archetype.HONEST: 3}), gov, SimulationConfig(seed=0, n_epochs=10)).run()
    assert gov.max_tail_mass == pytest.approx(0.3)
