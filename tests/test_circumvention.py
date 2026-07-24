"""Tests for the constraint-circumvention trajectory model (beads mtht)."""

import math
import random

import pytest

from swarm.redteam.circumvention import (
    CircumventionConfig,
    TrajectoryMonitorConfig,
    WorkerKind,
    action_filter_catch_rate,
    circumvention_intent,
    circumvention_success_rate,
    honest_completion_rate,
    mean_cumulative_suspicion_evaded,
    mean_realized_mass_on_catch,
    monitor_trajectory,
    simulate_episode,
    simulate_task,
    trajectory_monitor_catch_rate,
    trajectory_monitor_false_positive_rate,
)


def _circ(horizon, persistence, cfg, seed=0, n=400, mass=None):
    rng = random.Random(seed)
    return [
        simulate_task(
            WorkerKind.CIRCUMVENTION, horizon, persistence, cfg, rng, mass=mass
        )
        for _ in range(n)
    ]


class TestStepFlagProb:
    def test_clean_below_threshold(self):
        cfg = CircumventionConfig(tau=0.25)
        assert cfg.step_flag_prob(0.25) == 0.0
        assert cfg.step_flag_prob(0.1) == 0.0

    def test_rises_above_threshold(self):
        cfg = CircumventionConfig(tau=0.25, detect_prob=0.9, detect_gain=40.0)
        assert cfg.step_flag_prob(0.5) > 0.0
        assert cfg.step_flag_prob(1.0) > cfg.step_flag_prob(0.3)
        assert cfg.step_flag_prob(1.0) <= 0.9

    def test_hard_threshold_is_step(self):
        cfg = CircumventionConfig(tau=0.25, detect_prob=0.9, detect_gain=math.inf)
        assert cfg.step_flag_prob(0.25) == 0.0
        assert cfg.step_flag_prob(0.26) == 0.9


class TestDecompositionEvasion:
    def test_enough_horizon_evades_deterministically(self):
        """M=1, tau=0.25 => k>=4 clean. H=4 gives s=0.25, evades, no catch."""
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        out = simulate_task(
            WorkerKind.CIRCUMVENTION, 4, 0, cfg, random.Random(1)
        )
        assert out.evaded_action_filter
        assert not out.caught_by_action_filter
        assert out.succeeded
        assert out.per_step_suspicion == pytest.approx(0.25)

    def test_short_horizon_forces_over_threshold(self):
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        out = simulate_task(
            WorkerKind.CIRCUMVENTION, 1, 0, cfg, random.Random(1)
        )
        assert not out.evaded_action_filter
        assert out.per_step_suspicion == pytest.approx(1.0)


class TestHorizonScaling:
    def test_catch_rate_collapses_past_threshold(self):
        """Prediction 1: past the evasion horizon, action-level catch -> 0."""
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        below = action_filter_catch_rate(_circ(2, 0, cfg))
        at = action_filter_catch_rate(_circ(4, 0, cfg))
        above = action_filter_catch_rate(_circ(16, 0, cfg))
        assert below > 0.5
        assert at == 0.0
        assert above == 0.0

    def test_success_rises_with_horizon(self):
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        s_small = circumvention_success_rate(_circ(1, 0, cfg))
        s_large = circumvention_success_rate(_circ(8, 0, cfg))
        assert s_large > s_small
        assert s_large == pytest.approx(1.0)


class TestPersistence:
    def test_success_rises_with_persistence(self):
        """Prediction 2: at fixed short horizon, persistence -> success."""
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        s0 = circumvention_success_rate(_circ(1, 0, cfg))
        s_hi = circumvention_success_rate(_circ(1, 3, cfg))
        assert s_hi > s0
        assert s_hi == pytest.approx(1.0)

    def test_persistence_manufactures_horizon(self):
        """H=1 with enough persistence reaches the same clean k as large H."""
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        out = simulate_task(
            WorkerKind.CIRCUMVENTION, 1, 3, cfg, random.Random(1)
        )
        assert out.evaded_action_filter
        assert out.retries_used == 2  # k: 1->2->4, crosses tau at 4
        assert out.k_steps == 4


class TestHonestBaseline:
    def test_honest_never_caught_flat_completion(self):
        """Prediction 3: honest completion flat at 1.0 across the grid."""
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        honest = [
            simulate_task(WorkerKind.HONEST, H, P, cfg, random.Random(s))
            for H in (1, 4, 16) for P in (0, 3) for s in range(20)
        ]
        assert honest_completion_rate(honest) == 1.0
        assert all(not o.caught_by_action_filter for o in honest)
        assert all(o.cumulative_suspicion == 0.0 for o in honest)


class TestCumulativeSuspicionInvariant:
    def test_evaded_cumulative_suspicion_is_mass(self):
        """Trajectory-monitor hook (beads 0hp7): however finely the worker
        decomposes to evade, cumulative suspicion stays at M."""
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25, mass_spread=0.0)
        for H in (4, 8, 16, 32):
            outs = _circ(H, 0, cfg, n=50)
            evaded = [o for o in outs if o.evaded_action_filter]
            assert evaded
            assert all(o.cumulative_suspicion == pytest.approx(1.0) for o in evaded)
        assert mean_cumulative_suspicion_evaded(_circ(16, 0, cfg, n=50)) == (
            pytest.approx(1.0)
        )


class TestInstructionDecay:
    def test_zero_decay_never_reverts(self):
        rng = random.Random(0)
        assert not any(
            circumvention_intent(i, 20, 0.0, rng) for i in range(20)
        )

    def test_reversion_probability_rises_over_episode(self):
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        # High decay => later tasks far more often circumvention than early.
        early = late = 0
        for seed in range(200):
            ep = simulate_episode(
                10, 8, 2, cfg, random.Random(seed), instruction_decay=1.0
            )
            early += ep[1].kind is WorkerKind.CIRCUMVENTION
            late += ep[9].kind is WorkerKind.CIRCUMVENTION
        assert late > early

    def test_mixed_episode_honest_tasks_complete(self):
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        ep = simulate_episode(
            50, 4, 0, cfg, random.Random(3), instruction_decay=0.5
        )
        assert honest_completion_rate(ep) == 1.0


class TestTrajectoryMonitor:
    """The 0hp7 comparison arm: cumulative-suspicion monitoring."""

    def _circ(self, horizon, n=400, seed=0, spread=0.0):
        cfg = CircumventionConfig(
            suspicion_mass=1.0, tau=0.25, mass_spread=spread
        )
        rng = random.Random(seed)
        return [
            simulate_task(
                WorkerKind.CIRCUMVENTION, horizon, 0, cfg, rng,
                mass=cfg.draw_mass(rng),
            )
            for _ in range(n)
        ]

    def test_catch_rate_is_horizon_stable(self):
        """The headline contrast: monitor catch rate does NOT collapse with
        horizon the way the action filter does."""
        mon = TrajectoryMonitorConfig(threshold=0.5, coverage=1.0, detect_prob=1.0)
        rates = [
            trajectory_monitor_catch_rate(self._circ(H), mon, random.Random(1))
            for H in (1, 4, 8, 16, 32)
        ]
        # All essentially equal (M=1 > threshold, decomposition-proof).
        assert all(r == pytest.approx(1.0) for r in rates)

    def test_action_filter_collapses_where_monitor_holds(self):
        circ8 = self._circ(8)
        mon = TrajectoryMonitorConfig(threshold=0.5, detect_prob=1.0)
        assert action_filter_catch_rate(circ8) == 0.0
        assert trajectory_monitor_catch_rate(
            circ8, mon, random.Random(1)
        ) == pytest.approx(1.0)

    def test_low_mass_slips_under_coarse_monitor(self):
        """Monitor is decomposition-proof but not mass-proof: a forbidden
        outcome below the threshold evades it."""
        mon = TrajectoryMonitorConfig(threshold=0.5, detect_prob=1.0)
        cfg = CircumventionConfig(suspicion_mass=0.3, tau=0.25)
        out = simulate_task(WorkerKind.CIRCUMVENTION, 8, 0, cfg, random.Random(1))
        v = monitor_trajectory(out, mon, random.Random(1))
        assert not v.flagged

    def test_coverage_gates_observation(self):
        mon = TrajectoryMonitorConfig(threshold=0.5, coverage=0.0, detect_prob=1.0)
        assert trajectory_monitor_catch_rate(
            self._circ(8), mon, random.Random(1)
        ) == 0.0

    def test_false_positive_rises_with_horizon(self):
        """The monitor's weakness: benign activity accumulates over long
        honest trajectories and trips the threshold."""
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        mon = TrajectoryMonitorConfig(
            threshold=0.5, benign_step_mean=0.05, benign_step_sd=0.02,
            detect_prob=1.0,
        )
        honest_short = [
            simulate_task(WorkerKind.HONEST, 2, 0, cfg, random.Random(s))
            for s in range(400)
        ]
        honest_long = [
            simulate_task(WorkerKind.HONEST, 40, 0, cfg, random.Random(s))
            for s in range(400)
        ]
        fp_short = trajectory_monitor_false_positive_rate(
            honest_short, mon, random.Random(1)
        )
        fp_long = trajectory_monitor_false_positive_rate(
            honest_long, mon, random.Random(1)
        )
        assert fp_short == pytest.approx(0.0)
        assert fp_long > 0.5

    def test_mid_trajectory_pause_limits_realized_mass(self):
        """Mid-trajectory pause realizes less than full mass — the payoff over
        a post-hoc penalty that fires only after completion (mass M)."""
        mon = TrajectoryMonitorConfig(threshold=0.5, detect_prob=1.0)
        circ = self._circ(16, n=200)
        realized = mean_realized_mass_on_catch(circ, mon, random.Random(1))
        # Paused around threshold/M of the way; well under the full mass 1.0.
        assert 0.4 < realized < 0.75

    def test_honest_never_caught_by_action_filter_regardless(self):
        # sanity: monitor FP is a monitor property, action filter still clean
        cfg = CircumventionConfig(suspicion_mass=1.0, tau=0.25)
        honest = [
            simulate_task(WorkerKind.HONEST, 40, 0, cfg, random.Random(s))
            for s in range(50)
        ]
        assert all(not o.caught_by_action_filter for o in honest)
