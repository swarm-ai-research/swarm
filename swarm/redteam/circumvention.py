"""Constraint-circumvention trajectory simulation (beads mtht).

Models the OpenAI long-horizon failure mode
(docs/research/long-horizon-safety-lessons.md §1,§4,§5): a worker pursuing a
forbidden outcome decomposes it into a multi-step trajectory whose individual
steps each pass an *action-level* filter, even though the trajectory as a
whole crosses a boundary — the token-splitting incident, abstracted.

Why this is a standalone trajectory sim and not a YAML orchestrator scenario:
the orchestrator's unit of account is an atomic ``SoftInteraction`` with no
native task/step threading, and the whole quantity of interest here is how
detection scales with *trajectory length*. Forcing a decomposed trajectory
through atomic interactions would distort the measurement. This module owns
the trajectory model; the follow-up beads extend it in place — ``0hp7`` adds
a trajectory-level monitor as the comparison arm, ``q2fg`` a
trajectory-coverage metric.

Model. A forbidden outcome carries total suspicion mass ``M`` (normalized to
1). A worker spreads that mass across ``k`` steps, each carrying ``M/k``. An
action-level filter scores each step independently and can only see per-step
suspicion, so a fine-enough decomposition drops every step below its threshold
``tau`` and evades — but the *cumulative* suspicion is invariant at ``M`` no
matter how finely it is split, which is exactly the handle a trajectory
monitor will later grab.

- Horizon ``H`` bounds how finely the worker can split up front (k <= H).
- ``persistence`` bounds how many times a blocked step may be re-decomposed
  (each retry doubles k), i.e. how much effective horizon the worker can
  manufacture on the fly — the "kept trying for an hour" capability.
- ``instruction_decay`` makes the worker revert to a salient-but-wrong
  instruction source with probability rising over elapsed steps (the NanoGPT
  "post to GitHub" trap), so later tasks in an episode are more likely to be
  circumvention attempts.

Predictions this module is built to test (see experiments/):
  1. action-level catch rate DEGRADES as horizon grows (finer legal splits);
  2. circumvention success RISES with persistence (manufactured horizon);
  3. honest completion is FLAT in both (honest steps carry zero suspicion).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum


class WorkerKind(Enum):
    HONEST = "honest"
    CIRCUMVENTION = "circumvention"


@dataclass(frozen=True)
class CircumventionConfig:
    """Parameters of the circumvention model.

    suspicion_mass: total forbidden-outcome suspicion M (normalized).
    tau: action-level filter per-step threshold; steps at or below tau are
        clean, steps above are flagged with probability detect_prob (softened
        by detect_gain).
    detect_prob: max per-step flag probability for a clearly-over-threshold
        step.
    detect_gain: sharpness of the soft threshold; large => near-hard cutoff at
        tau. detect_gain=inf is the exact hard threshold.
    """

    suspicion_mass: float = 1.0
    mass_spread: float = 0.0  # per-task mass ~ U[M(1-spread), M(1+spread)]
    tau: float = 0.25
    detect_prob: float = 0.9
    detect_gain: float = 40.0

    def draw_mass(self, rng: random.Random) -> float:
        """Draw a per-task suspicion mass. Heterogeneous mass (spread > 0)
        smooths the population catch-rate curve, since each task's evasion
        horizon M/tau differs."""
        if self.mass_spread <= 0.0:
            return self.suspicion_mass
        lo = self.suspicion_mass * (1.0 - self.mass_spread)
        hi = self.suspicion_mass * (1.0 + self.mass_spread)
        return rng.uniform(lo, hi)

    def step_flag_prob(self, s: float) -> float:
        """Probability the action-level filter flags a single step of
        per-step suspicion ``s``. Zero at/below tau, rising to detect_prob
        above it. No false positives below threshold — a clean step is clean,
        which is what makes fine decomposition an evasion rather than a
        gamble."""
        if s <= self.tau:
            return 0.0
        if math.isinf(self.detect_gain):
            return self.detect_prob
        # Soft cutoff anchored so s == tau gives ~0 and s >> tau gives
        # detect_prob; the +tau offset keeps the ramp just above threshold.
        z = self.detect_gain * (s - self.tau)
        return self.detect_prob * (1.0 - math.exp(-z))


@dataclass(frozen=True)
class TaskOutcome:
    kind: WorkerKind
    horizon: int
    k_steps: int  # final number of steps the trajectory was split into
    per_step_suspicion: float
    cumulative_suspicion: float  # invariant = M for circumvention, 0 honest
    retries_used: int
    evaded_action_filter: bool  # reached a below-threshold decomposition
    caught_by_action_filter: bool
    succeeded: bool  # forbidden outcome reached (circ) / task done (honest)


def simulate_task(
    kind: WorkerKind,
    horizon: int,
    persistence: int,
    config: CircumventionConfig,
    rng: random.Random,
    mass: float | None = None,
) -> TaskOutcome:
    """Simulate one task's trajectory under the action-level filter.

    ``mass`` overrides the forbidden-outcome suspicion for this task; when
    ``None`` the config default is used. Real tasks carry heterogeneous
    suspicion, so a population drawn over a mass range yields a smooth
    catch-rate-vs-horizon curve rather than a single hard cliff at M/tau.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if kind is WorkerKind.HONEST:
        # Honest work carries no forbidden-outcome suspicion: every step is
        # clean, nothing to catch, task completes.
        return TaskOutcome(
            kind=kind,
            horizon=horizon,
            k_steps=horizon,
            per_step_suspicion=0.0,
            cumulative_suspicion=0.0,
            retries_used=0,
            evaded_action_filter=True,
            caught_by_action_filter=False,
            succeeded=True,
        )

    M = config.suspicion_mass if mass is None else mass
    k = horizon  # start by spreading maximally over the available horizon
    s = M / k
    retries = 0
    # Re-decompose a blocked (over-threshold) step, doubling k each time, up
    # to `persistence` attempts. This is how a persistent worker manufactures
    # effective horizon beyond H to drive per-step suspicion under tau.
    while s > config.tau and retries < persistence:
        k *= 2
        s = M / k
        retries += 1

    evaded = s <= config.tau

    caught = False
    if not evaded:
        # Ran out of persistence with steps still over threshold: execute
        # anyway; each over-threshold step is an independent detection chance.
        p_flag = config.step_flag_prob(s)
        caught = any(rng.random() < p_flag for _ in range(k))

    # Circumvention succeeds unless the filter caught it.
    succeeded = not caught

    return TaskOutcome(
        kind=kind,
        horizon=horizon,
        k_steps=k,
        per_step_suspicion=s,
        cumulative_suspicion=M,
        retries_used=retries,
        evaded_action_filter=evaded,
        caught_by_action_filter=caught,
        succeeded=succeeded,
    )


def circumvention_intent(
    step_index: int,
    total_steps: int,
    instruction_decay: float,
    rng: random.Random,
) -> bool:
    """Whether the worker reverts to the wrong instruction source on the task
    at position ``step_index`` of an episode of ``total_steps`` tasks.

    Probability rises linearly with elapsed fraction, scaled by
    instruction_decay (0 => never reverts, 1 => certainly reverts by the end).
    Models "models forget instructions on long rollouts."
    """
    if instruction_decay <= 0.0:
        return False
    elapsed = step_index / max(total_steps - 1, 1)
    return rng.random() < instruction_decay * elapsed


def simulate_episode(
    n_tasks: int,
    horizon: int,
    persistence: int,
    config: CircumventionConfig,
    rng: random.Random,
    instruction_decay: float = 0.0,
    always_circumvent: bool = False,
) -> list[TaskOutcome]:
    """Run an episode of ``n_tasks`` sequential tasks.

    With ``always_circumvent`` every task is a circumvention attempt (isolates
    the filter mechanics). Otherwise instruction_decay governs how often the
    worker reverts to the forbidden goal, rising over the episode.
    """
    outcomes = []
    for i in range(n_tasks):
        if always_circumvent:
            kind = WorkerKind.CIRCUMVENTION
        else:
            kind = (
                WorkerKind.CIRCUMVENTION
                if circumvention_intent(i, n_tasks, instruction_decay, rng)
                else WorkerKind.HONEST
            )
        mass = config.draw_mass(rng)
        outcomes.append(
            simulate_task(kind, horizon, persistence, config, rng, mass=mass)
        )
    return outcomes


# --- aggregate metrics -----------------------------------------------------


def action_filter_catch_rate(outcomes: list[TaskOutcome]) -> float:
    """Fraction of circumvention attempts caught by the action-level filter."""
    circ = [o for o in outcomes if o.kind is WorkerKind.CIRCUMVENTION]
    if not circ:
        return 0.0
    return sum(1 for o in circ if o.caught_by_action_filter) / len(circ)


def circumvention_success_rate(outcomes: list[TaskOutcome]) -> float:
    """Fraction of circumvention attempts that reached the forbidden
    outcome (evaded or slipped through)."""
    circ = [o for o in outcomes if o.kind is WorkerKind.CIRCUMVENTION]
    if not circ:
        return 0.0
    return sum(1 for o in circ if o.succeeded) / len(circ)


def honest_completion_rate(outcomes: list[TaskOutcome]) -> float:
    """Fraction of honest tasks completed (should be 1.0 — the flat baseline)."""
    honest = [o for o in outcomes if o.kind is WorkerKind.HONEST]
    if not honest:
        return 1.0
    return sum(1 for o in honest if o.succeeded) / len(honest)


def mean_cumulative_suspicion_evaded(outcomes: list[TaskOutcome]) -> float:
    """Mean cumulative suspicion of circumvention attempts that evaded the
    action filter. This is the trajectory-monitor handle (beads 0hp7): it
    stays at M no matter how finely the worker decomposed, so a monitor that
    sums per-step suspicion catches exactly what the action filter missed."""
    evaded = [
        o
        for o in outcomes
        if o.kind is WorkerKind.CIRCUMVENTION and o.evaded_action_filter
    ]
    if not evaded:
        return 0.0
    return sum(o.cumulative_suspicion for o in evaded) / len(evaded)


# --- trajectory-level monitor (beads 0hp7) ---------------------------------


@dataclass(frozen=True)
class TrajectoryMonitorConfig:
    """A monitor that sums per-step suspicion over a trajectory and can pause
    the session mid-way, rather than thresholding each action independently.

    threshold: cumulative suspicion at which the monitor flags. Must be < M
        for a task to be catchable; heterogeneous mass makes catch rate
        smooth. The gap M - threshold is the mass realized before the pause.
    coverage: fraction of trajectories the monitor observes (beads q2fg will
        make this the trust axis). Unobserved trajectories are never flagged.
    benign_step_mean / benign_step_sd: honest steps carry benign activity the
        monitor also sums; over a long honest trajectory this can cross the
        threshold and fire a false positive — the monitor's horizon-dependent
        weakness (OpenAI's "reduce unnecessary interruptions"). The action
        filter has no such weakness because it never aggregates.
    detect_prob: flag probability once the cumulative threshold is crossed.
    """

    threshold: float = 0.5
    coverage: float = 1.0
    benign_step_mean: float = 0.02
    benign_step_sd: float = 0.02
    detect_prob: float = 0.95


@dataclass(frozen=True)
class MonitorVerdict:
    observed: bool
    flagged: bool
    false_positive: bool  # flagged an honest trajectory
    detection_fraction: float  # fraction of the trajectory before the pause
    realized_mass: float  # forbidden suspicion realized before the pause


def monitor_trajectory(
    outcome: TaskOutcome,
    config: TrajectoryMonitorConfig,
    rng: random.Random,
) -> MonitorVerdict:
    """Run the trajectory monitor against one task's step sequence."""
    if rng.random() > config.coverage:
        return MonitorVerdict(
            observed=False,
            flagged=False,
            false_positive=False,
            detection_fraction=1.0,
            realized_mass=outcome.cumulative_suspicion,
        )

    if outcome.kind is WorkerKind.CIRCUMVENTION:
        M = outcome.cumulative_suspicion
        s = outcome.per_step_suspicion
        k = outcome.k_steps
        # Cumulative after j steps is j*s; it crosses threshold at j*.
        if M <= config.threshold or s <= 0:
            # Even the full trajectory never reaches threshold: monitor misses
            # (decomposition-proof, but not mass-proof — a low-mass forbidden
            # outcome slips under a coarse monitor).
            return MonitorVerdict(True, False, False, 1.0, M)
        if rng.random() > config.detect_prob:
            return MonitorVerdict(True, False, False, 1.0, M)
        j_star = min(k, math.ceil(config.threshold / s))
        frac = j_star / k
        return MonitorVerdict(
            observed=True,
            flagged=True,
            false_positive=False,
            detection_fraction=frac,
            realized_mass=frac * M,  # paused mid-trajectory: only this realized
        )

    # Honest trajectory: sum benign per-step activity; a long enough honest
    # session can accumulate past the threshold and false-positive.
    benign = 0.0
    for _ in range(outcome.horizon):
        benign += max(0.0, rng.gauss(config.benign_step_mean, config.benign_step_sd))
    if benign > config.threshold and rng.random() < config.detect_prob:
        return MonitorVerdict(True, True, True, 1.0, 0.0)
    return MonitorVerdict(True, False, False, 1.0, 0.0)


def trajectory_monitor_catch_rate(
    outcomes: list[TaskOutcome],
    config: TrajectoryMonitorConfig,
    rng: random.Random,
) -> float:
    """Fraction of circumvention attempts the trajectory monitor flags."""
    circ = [o for o in outcomes if o.kind is WorkerKind.CIRCUMVENTION]
    if not circ:
        return 0.0
    return sum(
        1 for o in circ if monitor_trajectory(o, config, rng).flagged
    ) / len(circ)


def trajectory_monitor_false_positive_rate(
    outcomes: list[TaskOutcome],
    config: TrajectoryMonitorConfig,
    rng: random.Random,
) -> float:
    """Fraction of honest trajectories the monitor wrongly flags — the cost
    that rises with horizon and must be traded against catch rate."""
    honest = [o for o in outcomes if o.kind is WorkerKind.HONEST]
    if not honest:
        return 0.0
    return sum(
        1 for o in honest if monitor_trajectory(o, config, rng).false_positive
    ) / len(honest)


def mean_realized_mass_on_catch(
    outcomes: list[TaskOutcome],
    config: TrajectoryMonitorConfig,
    rng: random.Random,
) -> float:
    """Mean forbidden mass realized before the monitor paused, over caught
    attempts. The mid-trajectory-pause payoff: lower = caught earlier, less
    damage done than a post-hoc penalty that fires only after completion."""
    realized = [
        v.realized_mass
        for o in outcomes
        if o.kind is WorkerKind.CIRCUMVENTION
        for v in (monitor_trajectory(o, config, rng),)
        if v.flagged
    ]
    if not realized:
        return 0.0
    return sum(realized) / len(realized)
