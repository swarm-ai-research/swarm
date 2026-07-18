"""The neural layer: continuous signals -> probabilistic atomic facts.

In a full neurosymbolic stack this layer is a learned network that perceives
raw input (positions, velocities, video, action logs) and emits *noisy*
probabilistic atoms such as ``near(agent, target)::0.8`` or
``moving_toward(a, b)::0.6``. The :class:`Perceiver` protocol is the seam where
such a network plugs in.

This module ships :class:`KinematicPerceiver`, a deterministic stand-in that
derives the same kind of probabilistic atoms from 2-D trajectories using smooth
logistic responses. It carries no learned weights, but it produces genuinely
*soft* facts (probabilities in ``(0, 1)``) so the Scallop layer can be exercised
and tested end-to-end without a GPU. Swap it for a learned model by
implementing :meth:`Perceiver.perceive`.

Emitted relations (one per timestep ``t``, plus temporal scaffolding):

==========================  ===================================================
``moving_toward(A,T,t)``    A's velocity points toward T at t
``closing(A,T,t)``          distance A->T strictly decreased into t
``increasing_distance(A,T,t)``  distance A->T strictly increased into t
``near(A,T,t)``             A is within the contact radius of T at t
``detected(A,T,t)``         A came within sensing radius of T at t (threat seen)
``searching(A,t)``          A is moving without a committed heading (wandering)
``approaching(A,T,t)``      A is both moving_toward and closing on T at t
``succ(t,t+1)``             discrete-time successor scaffold (probability 1)
==========================  ===================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence, Tuple

from swarm.neurosymbolic.engine import Program
from swarm.neurosymbolic.provenance import clamp01

Vec2 = Tuple[float, float]


def _logistic(x: float, k: float) -> float:
    """Numerically-stable logistic, clamped to ``[0, 1]``."""
    if x >= 0:
        z = math.exp(-k * x)
        return clamp01(1.0 / (1.0 + z))
    z = math.exp(k * x)
    return clamp01(z / (1.0 + z))


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def _dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


@dataclass
class Trajectory:
    """A named agent's positions over discrete time, plus optional targets.

    ``positions[i]`` is the agent's ``(x, y)`` at timestep ``i``. ``targets``
    maps a target name to *its* position series (same length); a static target
    can be given a length-1 series and is held constant.
    """

    agent: str
    positions: Sequence[Vec2]
    targets: Mapping[str, Sequence[Vec2]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.positions) < 2:
            raise ValueError("a trajectory needs at least two timesteps")

    def target_at(self, name: str, t: int) -> Vec2:
        series = self.targets[name]
        return series[t] if len(series) > 1 else series[0]


@dataclass
class PerceiverConfig:
    """Sharpness/scale knobs for the kinematic stand-in perceiver.

    Each ``*_sharpness`` is the logistic slope ``k``; larger = more confident
    (closer to hard 0/1) atoms. The radii set the distance scales at which
    ``near`` and ``detected`` cross probability 0.5.
    """

    contact_radius: float = 1.0
    sensing_radius: float = 5.0
    near_sharpness: float = 2.0
    detect_sharpness: float = 1.0
    heading_sharpness: float = 3.0
    distance_sharpness: float = 4.0
    wander_sharpness: float = 3.0


class Perceiver(Protocol):
    """Maps raw trajectories to probabilistic atoms loaded into a Program."""

    def perceive(self, traj: Trajectory, program: Optional[Program] = None) -> Program:
        """Emit probabilistic atoms for ``traj`` into ``program`` (or a new one)."""


class KinematicPerceiver:
    """Deterministic stand-in neural layer over 2-D kinematics.

    Produces soft facts via logistic responses on geometric quantities
    (heading alignment, distance deltas, proximity). Deterministic and seedless,
    so any pipeline built on it is reproducible from the trajectory alone.
    """

    def __init__(self, config: Optional[PerceiverConfig] = None) -> None:
        self.config = config or PerceiverConfig()

    def perceive(self, traj: Trajectory, program: Optional[Program] = None) -> Program:
        prog = program or Program()
        cfg = self.config
        positions = traj.positions
        n = len(positions)

        # Temporal scaffold: discrete successor relation.
        for t in range(n - 1):
            prog.fact("succ", t, t + 1, p=1.0)

        for t in range(n):
            vel = _sub(positions[t], positions[t - 1]) if t > 0 else (0.0, 0.0)
            speed = _norm(vel)

            for name in traj.targets:
                tgt = traj.target_at(name, t)
                to_tgt = _sub(tgt, positions[t])
                dist = _norm(to_tgt)

                # near / detected: proximity-based.
                prog.fact(
                    "near", traj.agent, name, t,
                    p=_logistic(cfg.contact_radius - dist, cfg.near_sharpness),
                )
                prog.fact(
                    "detected", traj.agent, name, t,
                    p=_logistic(cfg.sensing_radius - dist, cfg.detect_sharpness),
                )

                # moving_toward: velocity aligned with the direction to target.
                if speed > 1e-9 and dist > 1e-9:
                    cos = _dot(vel, to_tgt) / (speed * dist)
                else:
                    cos = 0.0
                p_toward = _logistic(cos, cfg.heading_sharpness)
                prog.fact("moving_toward", traj.agent, name, t, p=p_toward)

                # closing / increasing_distance: change in distance into t.
                if t > 0:
                    prev_tgt = traj.target_at(name, t - 1)
                    prev_dist = _norm(_sub(prev_tgt, positions[t - 1]))
                    delta = prev_dist - dist  # >0 means getting closer
                    p_close = _logistic(delta, cfg.distance_sharpness)
                    prog.fact("closing", traj.agent, name, t, p=p_close)
                    prog.fact(
                        "increasing_distance", traj.agent, name, t,
                        p=clamp01(1.0 - p_close),
                    )
                    # approaching = moving_toward AND closing (a soft conjunction).
                    prog.fact(
                        "approaching", traj.agent, name, t,
                        p=clamp01(p_toward * p_close),
                    )

            # searching: moving but not strongly committed to any heading.
            best_alignment = 0.0
            for name in traj.targets:
                tgt = traj.target_at(name, t)
                to_tgt = _sub(tgt, positions[t])
                d = _norm(to_tgt)
                if speed > 1e-9 and d > 1e-9:
                    best_alignment = max(best_alignment, _dot(vel, to_tgt) / (speed * d))
            moving = _logistic(speed - 0.1, cfg.wander_sharpness)
            uncommitted = _logistic(-best_alignment, cfg.heading_sharpness)
            prog.fact("searching", traj.agent, t, p=clamp01(moving * uncommitted))

        return prog
