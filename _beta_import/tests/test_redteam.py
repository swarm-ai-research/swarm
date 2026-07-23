"""Tests for the red-team attack library."""

import numpy as np

from beta_swarm.agents import Archetype
from beta_swarm.proxy import BetaProxyComputer
from beta_swarm.redteam import (
    AttackCategory,
    AttackLibrary,
    ReputationFarmer,
    VolumeForger,
    run_attack,
)
from beta_swarm.simulation import SimulationConfig
from beta_swarm.governance import TailMassGovernor


def _tail_gov():
    return TailMassGovernor(
        tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=20.0
    )


# ----------------------------------------------------------------------
# Adversary emission mechanics
# ----------------------------------------------------------------------
def test_volume_forger_produces_sharp_high_belief():
    """The attack's whole premise: forged event count buys concentration."""
    rng = np.random.default_rng(0)
    proxy = BetaProxyComputer()
    forger = VolumeForger("f", Archetype.DECEPTIVE, 2.0, 8.0, forge_volume=10)
    beliefs = [proxy.compute_belief(forger.emit_observables(0.2, rng)) for _ in range(200)]
    assert np.mean([b.mean for b in beliefs]) > 0.6         # looks good
    assert np.mean([b.concentration for b in beliefs]) > 18  # and forged-sharp
    assert np.mean([b.tail_mass(0.4) for b in beliefs]) < 0.05  # ~no tail mass


def test_concentration_cap_neutralizes_the_forgery():
    rng = np.random.default_rng(0)
    capped = BetaProxyComputer(max_concentration=6.0)
    forger = VolumeForger("f", Archetype.DECEPTIVE, 2.0, 8.0, forge_volume=10)
    beliefs = [capped.compute_belief(forger.emit_observables(0.2, rng)) for _ in range(100)]
    assert all(b.concentration <= 6.0 + 1e-9 for b in beliefs)


def test_reputation_farmer_flips_true_quality():
    rng = np.random.default_rng(0)
    farmer = ReputationFarmer("f", Archetype.DECEPTIVE, 2.0, 8.0, flip_epoch=10)
    early = np.mean([farmer.sample_outcome(rng, epoch=2) for _ in range(500)])
    late = np.mean([farmer.sample_outcome(rng, epoch=15) for _ in range(500)])
    assert early > 0.6 and late < 0.35  # genuine work, then poor


# ----------------------------------------------------------------------
# End-to-end attack outcomes
# ----------------------------------------------------------------------
def test_volume_forgery_breaks_the_standard_governor():
    result = run_attack(AttackLibrary.volume_forgery(), _tail_gov(), SimulationConfig(seed=7))
    assert result.evasion_rate > 0.4        # slips past the gate
    assert result.audit_evasion_rate > 0.8  # by evading the audit trigger
    assert result.realized_damage > 100
    assert not result.defended


def test_concentration_cap_defends_against_forgery():
    result = run_attack(
        AttackLibrary.volume_forgery(),
        _tail_gov(),
        SimulationConfig(seed=7),
        proxy=BetaProxyComputer(max_concentration=6.0),
    )
    assert result.evasion_rate < 0.2
    assert result.realized_damage < result.n_adversary_interactions  # ~no damage
    # The cap forces the forged belief through the DEFER gate to audit.
    assert result.audit_evasion_rate < 0.2


def test_scaled_audit_alone_does_not_stop_forgery():
    """A key finding: scaling the audit misses an attack that never triggers one."""
    scaled = run_attack(
        AttackLibrary.volume_forgery(),
        _tail_gov(),
        SimulationConfig(seed=7, audit_scales_with_concentration=True),
    )
    assert scaled.evasion_rate > 0.4


def test_reputation_farming_has_a_bounded_exploitation_window():
    result = run_attack(
        AttackLibrary.reputation_farming(flip_epoch=10), _tail_gov(), SimulationConfig(seed=7)
    )
    # It gets accepted (pre-flip genuinely, post-flip on banked reputation)...
    assert result.evasion_rate > 0.5
    # ...but realized outcomes eventually convict it: detection is not "never".
    assert result.detection_epoch is not None
    assert result.detection_epoch >= 10  # only after the flip


def test_collusion_ring_is_flagged_by_shape_detector():
    result = run_attack(AttackLibrary.collusion_ring(), _tail_gov(), SimulationConfig(seed=7))
    assert result.collusion_flagged


def test_library_covers_the_three_categories():
    cats = {s.category for s in AttackLibrary.all()}
    assert cats == {
        AttackCategory.EVASION,
        AttackCategory.EXPLOITATION,
        AttackCategory.COORDINATION,
    }


def test_run_attack_is_deterministic():
    a = run_attack(AttackLibrary.volume_forgery(), _tail_gov(), SimulationConfig(seed=3))
    b = run_attack(AttackLibrary.volume_forgery(), _tail_gov(), SimulationConfig(seed=3))
    assert a.evasion_rate == b.evasion_rate
    assert a.realized_damage == b.realized_damage
