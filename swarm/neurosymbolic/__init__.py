"""Neurosymbolic behaviour classification (neural perception + Scallop rules).

A two-layer stack for classifying agent behaviour from raw signals:

1. **Neural layer** (:mod:`~swarm.neurosymbolic.perceiver`) perceives continuous
   input (positions, velocities, observations) and emits *probabilistic atomic
   facts* — ``near(a, t)::0.8``, ``moving_toward(a, b, t)::0.6``.
2. **Scallop layer** (:mod:`~swarm.neurosymbolic.engine` +
   :mod:`~swarm.neurosymbolic.behaviors`) composes those atoms into behaviour
   classifications via Datalog rules with recursion, propagating probabilities
   through every join.

The engine is a self-contained reimplementation of the probabilistic-Datalog
slice we need (no heavy dependency); :mod:`~swarm.neurosymbolic.scallop` bridges
to the real `Scallop <https://www.scallop-lang.org>`_ for those who have it.

Quick start::

    from swarm.neurosymbolic import Trajectory, classify_behaviors

    traj = Trajectory(
        agent="hunter",
        positions=[(0, 0), (1, 0), (2, 0), (3, 0)],
        targets={"prey": [(5, 0)]},
    )
    scores = classify_behaviors(traj)
    print(scores.top())  # ('pursuing', 0.87)
"""

from swarm.neurosymbolic.behaviors import (
    BehaviorConfig,
    BehaviorScores,
    Episode,
    add_behavior_rules,
    classify_behaviors,
)
from swarm.neurosymbolic.engine import (
    Atom,
    FactSet,
    Program,
    Rule,
    Var,
)
from swarm.neurosymbolic.perceiver import (
    KinematicPerceiver,
    Perceiver,
    PerceiverConfig,
    Trajectory,
)
from swarm.neurosymbolic.provenance import (
    DEFAULT_PROVENANCE,
    MaxTimesProvenance,
    Provenance,
    clamp01,
    noisy_or,
)
from swarm.neurosymbolic.scallop import (
    SCALLOP_BEHAVIOR_RULES,
    run_with_scallopy,
    scallopy_available,
    to_scallop_program,
)

__all__ = [
    # engine
    "Atom",
    "Var",
    "Rule",
    "Program",
    "FactSet",
    # provenance
    "Provenance",
    "MaxTimesProvenance",
    "DEFAULT_PROVENANCE",
    "noisy_or",
    "clamp01",
    # neural layer
    "Trajectory",
    "Perceiver",
    "PerceiverConfig",
    "KinematicPerceiver",
    # behaviours
    "BehaviorConfig",
    "BehaviorScores",
    "Episode",
    "add_behavior_rules",
    "classify_behaviors",
    # scallop bridge
    "to_scallop_program",
    "run_with_scallopy",
    "scallopy_available",
    "SCALLOP_BEHAVIOR_RULES",
]
