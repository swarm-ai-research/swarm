"""Tests for the audit-bridge bond × detection frontier (beads 3ru4).

The audit lever's catch probability is derived from the *real*
`RandomAuditLever`, and the tests pin the load-bearing finding: the
per-interaction audit is on the decomposition ladder — an attacker that
cleans each interaction above `audit_threshold_p` blinds it.
"""

from swarm.governance.config import GovernanceConfig
from swarm.redteam.audit_bridge import (
    audit_catch_probability,
    frontier_point,
    honest_audit_false_positive,
)
from swarm.redteam.staking_bridge import SybilAttack


def _audit(prob=0.3, threshold=0.5, slash=0.2, stake=1.0):
    return GovernanceConfig(
        audit_enabled=True,
        audit_probability=prob,
        audit_threshold_p=threshold,
        staking_enabled=True,
        min_stake_to_participate=stake,
        stake_slash_rate=slash,
    )


class TestAuditIsOnTheLadder:
    def test_catch_equals_audit_prob_below_threshold(self):
        cfg = _audit(prob=0.3, threshold=0.5)
        c = audit_catch_probability(cfg, interaction_p=0.4, n_samples=6000)
        assert abs(c - 0.3) < 0.03

    def test_catch_collapses_when_attacker_cleans_above_threshold(self):
        cfg = _audit(prob=0.3, threshold=0.5)
        assert audit_catch_probability(cfg, 0.6, n_samples=6000) == 0.0
        assert audit_catch_probability(cfg, 0.8, n_samples=6000) == 0.0

    def test_disabled_audit_never_catches(self):
        cfg = GovernanceConfig(audit_enabled=False)
        assert audit_catch_probability(cfg, 0.1) == 0.0


class TestHonestFalsePositive:
    def test_rises_with_threshold(self):
        low = honest_audit_false_positive(
            _audit(threshold=0.5), honest_p_mean=0.7, n_samples=6000
        )
        high = honest_audit_false_positive(
            _audit(threshold=0.9), honest_p_mean=0.7, n_samples=6000
        )
        assert high > low

    def test_threshold_below_honest_floor_is_clean(self):
        fp = honest_audit_false_positive(
            _audit(threshold=0.3), honest_p_mean=0.7, honest_p_sd=0.1,
            n_samples=6000,
        )
        assert fp < 0.02


class TestFrontier:
    def test_clean_attacker_evades_audit_and_bond_does_not_deter(self):
        """Attacker cleaner than the threshold: audit blind, recoverable bond
        with ample capital does not deter."""
        atk = SybilAttack(objective_value=100.0)
        cfg = _audit(prob=0.3, threshold=0.5, stake=500.0)
        pt = frontier_point(
            atk, cfg, liquid_capital=1e9, attacker_interaction_p=0.6,
            n_samples=4000,
        )
        assert pt.audit_catch_prob == 0.0
        assert not pt.attacker_deterred
        assert pt.net_payoff == 100.0

    def test_catching_a_clean_attacker_costs_honest_false_positives(self):
        """To bite an attacker at p_int=0.8, threshold must exceed it, which
        flags a large fraction of honest (p~0.7) work."""
        atk = SybilAttack(objective_value=100.0)
        cfg = _audit(prob=0.3, threshold=0.85, stake=500.0)
        pt = frontier_point(
            atk, cfg, liquid_capital=1e9, attacker_interaction_p=0.8,
            honest_p_mean=0.7, n_samples=4000,
        )
        assert pt.audit_catch_prob > 0.0  # now the audit bites
        assert pt.honest_fp > 0.1  # but honest work pays for it

    def test_audit_plus_stake_can_deter_a_visible_attacker(self):
        """When the attacker is forced below threshold (audit bites) and the
        stake is high, the two levers together deter — the complementarity."""
        atk = SybilAttack(objective_value=100.0)
        cfg = _audit(prob=0.5, threshold=0.6, slash=0.3, stake=2000.0)
        pt = frontier_point(
            atk, cfg, liquid_capital=1e9, attacker_interaction_p=0.4,
            n_samples=4000,
        )
        assert pt.audit_catch_prob > 0.0
        assert pt.attacker_deterred
