"""Tests for the decomposition-ladder adversarial validation (beads njvs).

The central hypothesis, confirmed here: a defense monitoring up to level ``m``
catches an attacker who decomposed to depth ``d`` iff ``d < m``. Any finite
defense is escaped by an attacker adaptive to that level.
"""

import random

from swarm.redteam.decomposition_ladder import (
    LadderConfig,
    Level,
    evaluate_attack,
    honest_false_positive_rate,
    plan_attack,
)


def _catch_rate(cfg, defense, depth, n=200):
    caught = 0
    for s in range(n):
        plan = plan_attack(cfg, depth)
        caught += evaluate_attack(cfg, plan, defense, random.Random(s)).caught
    return caught / n


class TestLadderRule:
    """caught iff attacker depth < defense monitoring level."""

    def test_defense_catches_below_its_level(self):
        cfg = LadderConfig()
        # trajectory monitor catches DIRECT and STEP (depth < TRAJECTORY)
        assert _catch_rate(cfg, Level.TRAJECTORY, Level.DIRECT) == 1.0
        assert _catch_rate(cfg, Level.TRAJECTORY, Level.STEP) == 1.0

    def test_defense_escaped_at_and_above_its_level(self):
        cfg = LadderConfig()
        # trajectory monitor is escaped by a cross-trajectory attacker
        assert _catch_rate(cfg, Level.TRAJECTORY, Level.TRAJECTORY) == 0.0
        assert _catch_rate(cfg, Level.TRAJECTORY, Level.AGENT) == 0.0

    def test_trajectory_monitor_headline_is_attacker_dependent(self):
        """The 0hp7 'flat ~0.92' holds only vs non-adaptive attackers; a
        cross-trajectory attacker drops it to zero. This is the qualification."""
        cfg = LadderConfig()
        non_adaptive = _catch_rate(cfg, Level.TRAJECTORY, Level.STEP)
        adaptive = _catch_rate(cfg, Level.TRAJECTORY, Level.TRAJECTORY)
        assert non_adaptive == 1.0
        assert adaptive == 0.0

    def test_cross_agent_attacker_escapes_every_per_agent_defense(self):
        cfg = LadderConfig()
        for defense in (Level.STEP, Level.TRAJECTORY, Level.AGENT):
            assert _catch_rate(cfg, defense, Level.AGENT) == 0.0

    def test_action_filter_catches_direct_only(self):
        cfg = LadderConfig()
        assert _catch_rate(cfg, Level.STEP, Level.DIRECT) > 0.5
        assert _catch_rate(cfg, Level.STEP, Level.STEP) == 0.0


class TestAttackerCost:
    def test_decomposition_cost_rises_with_depth(self):
        cfg = LadderConfig()
        direct = plan_attack(cfg, Level.DIRECT).total_trajectories
        traj = plan_attack(cfg, Level.TRAJECTORY).total_trajectories
        agent = plan_attack(cfg, Level.AGENT)
        assert traj > direct
        assert agent.n_agents > 1
        assert agent.total_trajectories >= traj

    def test_per_unit_mass_stays_under_threshold(self):
        cfg = LadderConfig()
        traj = plan_attack(cfg, Level.TRAJECTORY)
        assert traj.per_trajectory_mass <= cfg.theta_trajectory
        agent = plan_attack(cfg, Level.AGENT)
        assert agent.per_agent_mass <= cfg.theta_agent


class TestDefenderCost:
    def test_false_positive_rises_with_aggregation_level(self):
        """A larger aggregation window sums more benign activity, so the
        false-positive cost rises as the defender climbs the ladder."""
        cfg = LadderConfig(
            benign_step_mean=0.05, benign_step_sd=0.02, theta_agent=1.0,
            theta_trajectory=0.5,
        )
        # Same honest footprint; only the monitoring level changes.
        fp_traj = honest_false_positive_rate(
            cfg, Level.TRAJECTORY, n_agents=3, n_trajectories_per_agent=4,
            steps_per_trajectory=8, n_trials=400, rng=random.Random(1),
        )
        fp_agent = honest_false_positive_rate(
            cfg, Level.AGENT, n_agents=3, n_trajectories_per_agent=4,
            steps_per_trajectory=8, n_trials=400, rng=random.Random(1),
        )
        assert fp_agent >= fp_traj


class TestObjectiveAchieved:
    def test_escaped_objective_is_achieved(self):
        cfg = LadderConfig()
        plan = plan_attack(cfg, Level.AGENT)
        v = evaluate_attack(cfg, plan, Level.AGENT, random.Random(0))
        assert not v.caught
        assert v.objective_achieved
