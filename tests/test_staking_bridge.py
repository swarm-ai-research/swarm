"""Tests for the staking-bridge deterrence validation (beads zxkw).

The end-to-end finding, pinned here: a recoverable stake bond does NOT deter a
monitor-evading Sybil attacker by its amount; deterrence comes only from
capital scarcity or from slashing coupled to a residual catch probability.
"""

import math

from swarm.governance.config import GovernanceConfig
from swarm.redteam.staking_bridge import (
    SybilAttack,
    deterrence_stake_capital_channel,
    deterrence_stake_slash_channel,
    evaluate_sybil_attack,
    expected_slash_loss,
    identities_needed,
    usable_identities,
)


def _cfg(stake, slash=0.2):
    return GovernanceConfig(
        staking_enabled=True,
        min_stake_to_participate=float(stake),
        stake_slash_rate=float(slash),
    )


class TestLeverIsExercised:
    def test_usable_count_uses_real_gate(self):
        # capital 100, stake 25 -> 4 identities fundable and passing the gate
        assert usable_identities(_cfg(25), liquid_capital=100.0) == 4

    def test_gate_denies_when_stake_exceeds_funding(self):
        # a single identity funded at exactly `stake` passes; but capital 10
        # at stake 25 funds zero identities
        assert usable_identities(_cfg(25), liquid_capital=10.0) == 0

    def test_slash_uses_real_lever(self):
        cfg = _cfg(100, slash=0.3)
        # one identity, caught with certainty -> slashes 0.3 * 100
        loss = expected_slash_loss(cfg, n_identities=1, catch_prob=1.0)
        assert loss == 100.0 * 0.3

    def test_staking_disabled_is_unbounded(self):
        cfg = GovernanceConfig(staking_enabled=False)
        assert usable_identities(cfg, liquid_capital=1.0) > 1000


class TestRecoverableBondDoesNotDeter:
    def test_bond_size_never_deters_well_capitalized_evader(self):
        """The §7 correction: with no residual catch (pure recoverable bond)
        and ample capital, no stake level deters — the attacker recovers every
        bond."""
        atk = SybilAttack(objective_value=100.0)
        for stake in (10, 50, 100, 500, 5000):
            v = evaluate_sybil_attack(
                atk, _cfg(stake), liquid_capital=1e9, catch_prob=0.0
            )
            assert v.feasible
            assert not v.deterred
            assert v.net_payoff == 100.0


class TestCapitalScarcityChannel:
    def test_deterrence_threshold_is_capital_over_n(self):
        atk = SybilAttack(objective_mass=2.0, agent_threshold=1.0)  # n=3
        assert identities_needed(atk) == 3
        cap = 90.0
        thresh = deterrence_stake_capital_channel(atk, cap)
        assert thresh == 30.0  # 90 / 3
        # just below: can fund 3; just above: cannot
        below = evaluate_sybil_attack(atk, _cfg(29), cap, catch_prob=0.0)
        above = evaluate_sybil_attack(atk, _cfg(31), cap, catch_prob=0.0)
        assert not below.deterred
        assert above.deterred and not above.feasible

    def test_threshold_depends_on_capital_not_value(self):
        cheap = SybilAttack(objective_value=10.0)
        dear = SybilAttack(objective_value=10_000.0)
        # same n, same capital -> same capital-scarcity threshold
        assert deterrence_stake_capital_channel(
            cheap, 100.0
        ) == deterrence_stake_capital_channel(dear, 100.0)


class TestSlashChannel:
    def test_pure_bond_never_deters_via_slash(self):
        atk = SybilAttack()
        # catch_prob 0 or slash 0 -> infinite required stake
        assert deterrence_stake_slash_channel(
            atk, _cfg(1, slash=0.2), catch_prob=0.0
        ) == math.inf
        assert deterrence_stake_slash_channel(
            atk, _cfg(1, slash=0.0), catch_prob=0.2
        ) == math.inf

    def test_higher_catch_prob_lowers_required_stake(self):
        """Staking and detection are complements: better catch => less stake
        needed to deter."""
        atk = SybilAttack()
        s_low = deterrence_stake_slash_channel(atk, _cfg(1), catch_prob=0.05)
        s_high = deterrence_stake_slash_channel(atk, _cfg(1), catch_prob=0.3)
        assert s_high < s_low

    def test_slash_channel_deters_at_predicted_stake(self):
        atk = SybilAttack(objective_value=100.0)
        cfg_template = _cfg(1, slash=0.2)
        s = deterrence_stake_slash_channel(atk, cfg_template, catch_prob=0.3)
        # at that stake (ample capital), the attacker's net payoff is ~0
        v = evaluate_sybil_attack(
            atk, _cfg(s, slash=0.2), liquid_capital=1e12, catch_prob=0.3
        )
        assert abs(v.net_payoff) < 1e-6
        assert v.deterred
