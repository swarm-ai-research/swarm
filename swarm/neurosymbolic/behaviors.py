"""Behaviour classification: compose probabilistic atoms into behaviours.

This is the payload of the neurosymbolic stack. The neural layer
(:mod:`swarm.neurosymbolic.perceiver`) emits noisy per-timestep atoms; the
rules here — run on the :mod:`swarm.neurosymbolic.engine` — compose them into
temporally-extended behaviour classifications, with probabilities flowing
through every join and recursion.

Behaviours
----------
- **pursuing** — *repeatedly moving toward a target while closing distance.*
  A recursive ``run`` relation chains consecutive ``pursuit_step`` atoms; a
  sustained run (length ``>= min_run_length``) signals pursuit. The run's
  probability is the product of its steps, so a long, confident chase scores
  high and a one-off coincidence does not.
- **evading** — *increasing distance after detection.* Detection makes an agent
  ``alerted`` (a recursive forward-persisting relation); sustained
  distance-increase while alerted is an evasion run.
- **foraging** — *alternating search and approach.* A ``forage_cycle`` is a
  search step immediately followed by an approach; repeated cycles
  (``>= min_cycles``) signal foraging.

The read-out aggregations are deliberately conjunctive/idempotent so they stay
consistent with the engine's top-1-proof provenance: sustained behaviours use
``max`` over qualifying runs (most-confident episode); the repetition count for
foraging uses the product of the ``K`` strongest cycles (all ``K`` must hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from swarm.neurosymbolic.engine import Atom, FactSet, Program, Var
from swarm.neurosymbolic.perceiver import (
    KinematicPerceiver,
    Perceiver,
    Trajectory,
)
from swarm.neurosymbolic.provenance import DEFAULT_PROVENANCE, Provenance

# Convenience variable handles for readable rules.
_A, _T, _O, _S, _T1, _T2 = (
    Var("A"), Var("T"), Var("O"), Var("S"), Var("T1"), Var("T2")
)
_WILD = Var("_")


@dataclass
class BehaviorConfig:
    """Thresholds for promoting atom-level evidence to a behaviour label."""

    min_run_length: int = 3   # consecutive steps for pursuing/evading
    min_cycles: int = 2       # search->approach cycles for foraging


@dataclass
class Episode:
    """A supporting temporal episode for a behaviour, with its probability."""

    target: Optional[str]
    start: int
    end: int
    probability: float


@dataclass
class BehaviorScores:
    """Per-behaviour probabilities plus the evidence that produced them."""

    agent: str
    scores: Dict[str, float]
    episodes: Dict[str, List[Episode]] = field(default_factory=dict)

    def top(self) -> Tuple[str, float]:
        """The most-probable behaviour and its probability."""
        return max(self.scores.items(), key=lambda kv: kv[1])


def add_behavior_rules(program: Program) -> Program:
    """Add the pursuing/evading/foraging rule set to ``program`` in place."""
    # -- pursuing: repeated moving_toward while closing --------------------
    program.rule(
        Atom("pursuit_step", _A, _T, _T1),
        Atom("moving_toward", _A, _T, _T1),
        Atom("closing", _A, _T, _T1),
    )
    program.rule(  # base run: a single step is a run [t, t]
        Atom("pursuit_run", _A, _T, _T1, _T1),
        Atom("pursuit_step", _A, _T, _T1),
    )
    program.rule(  # extend a run across the successor relation
        Atom("pursuit_run", _A, _T, _S, _T2),
        Atom("pursuit_run", _A, _T, _S, _T1),
        Atom("succ", _T1, _T2),
        Atom("pursuit_step", _A, _T, _T2),
    )

    # -- evading: increasing distance after detection ----------------------
    program.rule(  # detection raises the alert
        Atom("alerted", _A, _O, _T1),
        Atom("detected", _A, _O, _T1),
    )
    program.rule(  # the alert persists forward in time
        Atom("alerted", _A, _O, _T2),
        Atom("alerted", _A, _O, _T1),
        Atom("succ", _T1, _T2),
    )
    program.rule(
        Atom("evade_step", _A, _O, _T1),
        Atom("alerted", _A, _O, _T1),
        Atom("increasing_distance", _A, _O, _T1),
    )
    program.rule(
        Atom("evade_run", _A, _O, _T1, _T1),
        Atom("evade_step", _A, _O, _T1),
    )
    program.rule(
        Atom("evade_run", _A, _O, _S, _T2),
        Atom("evade_run", _A, _O, _S, _T1),
        Atom("succ", _T1, _T2),
        Atom("evade_step", _A, _O, _T2),
    )

    # -- foraging: alternating search then approach ------------------------
    program.rule(
        Atom("forage_cycle", _A, _T1),
        Atom("searching", _A, _T1),
        Atom("succ", _T1, _T2),
        Atom("approaching", _A, _WILD, _T2),
    )
    return program


def _best_runs(
    db: FactSet, relation: str, agent: str, min_length: int
) -> List[Episode]:
    """Collect qualifying runs ``relation(agent, target, start, end)``.

    Keeps, per ``(target, start)``, the longest run that reaches at least
    ``min_length``, recording its probability.
    """
    best: Dict[Tuple[str, int], Episode] = {}
    for gt, p in db.get(relation).items():
        a, target_c, start_c, end_c = gt
        if a != agent:
            continue
        target, start, end = str(target_c), int(start_c), int(end_c)
        length = end - start + 1
        if length < min_length:
            continue
        key = (target, start)
        prev = best.get(key)
        if prev is None or end > prev.end:
            best[key] = Episode(target, start, end, float(p))
    return sorted(best.values(), key=lambda e: e.probability, reverse=True)


def classify_behaviors(
    traj: Trajectory,
    *,
    perceiver: Optional[Perceiver] = None,
    config: Optional[BehaviorConfig] = None,
    provenance: Optional[Provenance] = None,
) -> BehaviorScores:
    """Classify an agent's trajectory into behaviour probabilities.

    Runs the neural layer to emit probabilistic atoms, then the Scallop layer
    to compose them into pursuing/evading/foraging scores in ``[0, 1]``.
    """
    perceiver = perceiver or KinematicPerceiver()
    config = config or BehaviorConfig()
    prov = provenance or DEFAULT_PROVENANCE

    program = perceiver.perceive(traj)
    add_behavior_rules(program)
    db = program.run(prov)

    pursuit = _best_runs(db, "pursuit_run", traj.agent, config.min_run_length)
    evade = _best_runs(db, "evade_run", traj.agent, config.min_run_length)

    # Foraging: product of the K strongest distinct cycles (all K must hold).
    cycles = sorted(
        (
            Episode(None, int(t), int(t), float(p))
            for (a, t), p in db.get("forage_cycle").items()
            if a == traj.agent
        ),
        key=lambda e: e.probability,
        reverse=True,
    )
    if len(cycles) >= config.min_cycles:
        forage_p = 1.0
        for e in cycles[: config.min_cycles]:
            forage_p *= e.probability
    else:
        forage_p = 0.0

    scores = {
        "pursuing": pursuit[0].probability if pursuit else 0.0,
        "evading": evade[0].probability if evade else 0.0,
        "foraging": forage_p,
    }
    episodes = {
        "pursuing": pursuit,
        "evading": evade,
        "foraging": cycles[: config.min_cycles] if forage_p > 0 else [],
    }
    return BehaviorScores(agent=traj.agent, scores=scores, episodes=episodes)
