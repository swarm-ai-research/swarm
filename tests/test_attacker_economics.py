"""Tests for the attacker-economics model (beads 7bwz)."""

import math
import random

import pytest

from swarm.redteam.attacker_economics import (
    EconomicsConfig,
    attacker_cost,
    attacker_units,
    critical_unit_cost,
    deterrence_threshold,
    false_positive_rate,
    fp_threshold,
    is_defensible,
    is_deterred,
)
from swarm.redteam.decomposition_ladder import Level


class TestAnalyticFalsePositive:
    def test_matches_simulation(self):
        """Normal-approx FP tracks the simulated benign accumulation in the
        tail regime (theta above the mean window sum), which is where the
        FP-budget inversion operates. Near the mean the zero-truncation of
        benign draws costs the approximation ~0.03; that region is not used."""
        cfg = EconomicsConfig()
        w = cfg.subunits_per_unit[Level.TRAJECTORY]
        for theta in (0.30, 0.35, 0.40):
            analytic = false_positive_rate(theta, Level.TRAJECTORY, cfg)
            rng = random.Random(0)
            hits = 0
            n = 20000
            for _ in range(n):
                s = sum(
                    max(0.0, rng.gauss(cfg.benign_mean, cfg.benign_sd))
                    for _ in range(w)
                )
                hits += s > theta
            sim = hits / n
            assert abs(analytic - sim) < 0.02

    def test_fp_threshold_inverts_rate(self):
        cfg = EconomicsConfig()
        for f_max in (0.01, 0.05, 0.2):
            theta = fp_threshold(Level.AGENT, cfg, f_max)
            assert false_positive_rate(theta, Level.AGENT, cfg) == pytest.approx(
                f_max, abs=1e-3
            )


class TestAttackerCost:
    def test_units_rise_as_threshold_tightens(self):
        cfg = EconomicsConfig()
        assert attacker_units(0.5, cfg) < attacker_units(0.05, cfg)

    def test_deterrence_scales_with_cost_and_inverse_value(self):
        base = EconomicsConfig()
        td_base = deterrence_threshold(Level.AGENT, base)
        # doubling per-unit cost doubles the loosest deterring threshold
        dearer = EconomicsConfig(
            unit_cost={**base.unit_cost, Level.AGENT: base.unit_cost[Level.AGENT] * 2}
        )
        assert deterrence_threshold(Level.AGENT, dearer) == pytest.approx(
            2 * td_base
        )
        # doubling objective value halves it
        richer = EconomicsConfig(objective_value=base.objective_value * 2)
        assert deterrence_threshold(Level.AGENT, richer) == pytest.approx(
            td_base / 2
        )

    def test_is_deterred_consistent_with_threshold(self):
        cfg = EconomicsConfig()
        td = deterrence_threshold(Level.AGENT, cfg)
        # just below theta_deter: deterred; just above: not (continuous approx)
        assert is_deterred(td * 0.9, Level.AGENT, cfg)
        assert attacker_cost(td * 0.9, Level.AGENT, cfg) >= cfg.objective_value


class TestDefensibility:
    def test_cheap_levels_not_defensible(self):
        """Trajectory-level unit cost is far below what monitoring needs."""
        cfg = EconomicsConfig()
        assert not is_defensible(Level.TRAJECTORY, cfg, f_max=0.05)

    def test_defensible_flips_above_critical_cost(self):
        cfg = EconomicsConfig()
        c_crit = critical_unit_cost(Level.AGENT, cfg, f_max=0.05)
        below = EconomicsConfig(
            unit_cost={**cfg.unit_cost, Level.AGENT: c_crit * 0.9}
        )
        above = EconomicsConfig(
            unit_cost={**cfg.unit_cost, Level.AGENT: c_crit * 1.1}
        )
        assert not is_defensible(Level.AGENT, below, f_max=0.05)
        assert is_defensible(Level.AGENT, above, f_max=0.05)

    def test_critical_cost_rises_with_objective_value(self):
        f_max = 0.05
        cheap_obj = EconomicsConfig(objective_value=50.0)
        dear_obj = EconomicsConfig(objective_value=500.0)
        assert critical_unit_cost(Level.AGENT, dear_obj, f_max) > critical_unit_cost(
            Level.AGENT, cheap_obj, f_max
        )

    def test_critical_cost_is_the_defensibility_boundary(self):
        """At exactly c_crit, theta_fp == theta_deter (regions just touch)."""
        cfg = EconomicsConfig()
        c_crit = critical_unit_cost(Level.AGENT, cfg, f_max=0.05)
        at = EconomicsConfig(unit_cost={**cfg.unit_cost, Level.AGENT: c_crit})
        assert deterrence_threshold(Level.AGENT, at) == pytest.approx(
            fp_threshold(Level.AGENT, at, 0.05), rel=1e-6
        )


class TestStakingConnection:
    def test_min_stake_is_agent_critical_cost(self):
        """The minimum stake for agent-level viability is c_crit at the agent
        level — a concrete governance number, monotone in objective value."""
        cfg = EconomicsConfig(objective_value=100.0, objective_mass=2.0)
        min_stake = critical_unit_cost(Level.AGENT, cfg, f_max=0.05)
        assert min_stake > cfg.unit_cost[Level.AGENT]  # default stake too low
        assert math.isfinite(min_stake)
