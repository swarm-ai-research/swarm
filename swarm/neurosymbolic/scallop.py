"""Optional bridge to the real Scallop (`scallop-lang.org`).

The in-repo :mod:`swarm.neurosymbolic.engine` reimplements the slice of Scallop
we need so the framework stays dependency-free and testable anywhere. This
module is the escape hatch to the *real* thing for users who have it installed:

- :func:`to_scallop_program` emits the behaviour rules as a ``.scl`` source
  string — useful for documentation, for running in the Scallop playground, or
  for handing to ``scallopy``. It requires no dependency.
- :func:`run_with_scallopy` executes a populated :class:`Program` on the
  ``scallopy`` backend if it is importable, raising a clear error otherwise.

The rule text mirrors :func:`swarm.neurosymbolic.behaviors.add_behavior_rules`
one-to-one. Keep the two in sync when editing either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from swarm.neurosymbolic.engine import Program

# The behaviour program as native Scallop. `min_run_length`/`min_cycles`
# thresholds are applied at read-out (Python side) rather than in the rules,
# matching the in-repo engine.
SCALLOP_BEHAVIOR_RULES = """\
// ---- neural-layer inputs (probabilistic facts emitted per timestep) ----
type moving_toward(agent: String, target: String, t: usize)
type closing(agent: String, target: String, t: usize)
type increasing_distance(agent: String, target: String, t: usize)
type near(agent: String, target: String, t: usize)
type detected(agent: String, target: String, t: usize)
type searching(agent: String, t: usize)
type approaching(agent: String, target: String, t: usize)
type succ(t1: usize, t2: usize)

// ---- pursuing: repeated moving_toward while closing ----
rel pursuit_step(a, tg, t) = moving_toward(a, tg, t) and closing(a, tg, t)
rel pursuit_run(a, tg, t, t) = pursuit_step(a, tg, t)
rel pursuit_run(a, tg, s, t2) =
    pursuit_run(a, tg, s, t1) and succ(t1, t2) and pursuit_step(a, tg, t2)

// ---- evading: increasing distance after detection ----
rel alerted(a, o, t) = detected(a, o, t)
rel alerted(a, o, t2) = alerted(a, o, t1) and succ(t1, t2)
rel evade_step(a, o, t) = alerted(a, o, t) and increasing_distance(a, o, t)
rel evade_run(a, o, t, t) = evade_step(a, o, t)
rel evade_run(a, o, s, t2) =
    evade_run(a, o, s, t1) and succ(t1, t2) and evade_step(a, o, t2)

// ---- foraging: alternating search then approach ----
rel forage_cycle(a, t1) =
    searching(a, t1) and succ(t1, t2) and approaching(a, _, t2)
"""


def to_scallop_program() -> str:
    """Return the behaviour rule set as a native Scallop (``.scl``) source string."""
    return SCALLOP_BEHAVIOR_RULES


def scallopy_available() -> bool:
    """True if the optional ``scallopy`` backend can be imported."""
    try:
        import scallopy  # noqa: F401
    except ImportError:
        return False
    return True


def run_with_scallopy(program: "Program", provenance: str = "topkproofs"):
    """Execute a populated :class:`Program` on the real ``scallopy`` backend.

    Loads the program's extensional facts and the shared behaviour rules into a
    ``scallopy`` context and runs it. Returns the ``scallopy`` context so the
    caller can read out relations. Raises :class:`ImportError` with install
    guidance if ``scallopy`` is not present.
    """
    try:
        import scallopy
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "scallopy is not installed. Install Scallop "
            "(https://www.scallop-lang.org) to use this backend, or use the "
            "built-in engine via Program.run()."
        ) from exc

    ctx = scallopy.ScallopContext(provenance=provenance)
    ctx.add_program(SCALLOP_BEHAVIOR_RULES)
    for relation, gt, p in program.edb.items():  # pragma: no cover - optional dep
        ctx.add_facts(relation, [(p, gt)])
    ctx.run()
    return ctx
