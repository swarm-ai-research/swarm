# Neurosymbolic Behavior Classification

Agent behavior has *compositional* structure: a behavior is a sequence or
combination of lower-level actions, states, and relations over time. That is
exactly what Datalog rules express well. The `swarm.neurosymbolic` package pairs
a **neural perception layer** that emits noisy probabilistic facts with a
**Scallop-style rule layer** that composes those facts into behavior
classifications — with probabilities flowing through every join and recursion.

This mirrors the framework's core "soft label" stance: instead of hard
good/bad classifications, every fact and every derived behavior carries a
probability in `[0, 1]`.

## Division of labor

| Layer | Module | Role |
|---|---|---|
| **Neural** | `perceiver.py` | Continuous input (positions, velocities, observations) → probabilistic atoms: `near(a, t)::0.8`, `moving_toward(a, b, t)::0.6` |
| **Scallop** | `engine.py` + `behaviors.py` | Datalog rules with recursion compose atoms into `pursuing` / `evading` / `foraging`, propagating probabilities |

The neural layer is a pluggable [`Perceiver`][perceiver] protocol. The shipped
`KinematicPerceiver` is a deterministic, seedless stand-in over 2-D kinematics
(no learned weights, no GPU) — swap in a learned network by implementing
`perceive`.

## Behaviors

- **pursuing** — *repeatedly moving toward a target while closing distance.* A
  recursive `pursuit_run` relation chains consecutive `pursuit_step` atoms; a
  sustained run scores high, a one-off coincidence does not. The run's
  probability is the product of its steps, so confidence compounds over time.
- **evading** — *increasing distance after detection.* Detection makes an agent
  `alerted` (a recursive, forward-persisting relation); sustained
  distance-increase while alerted is an evasion run.
- **foraging** — *alternating search and approach.* A `forage_cycle` is a
  search step immediately followed by an approach; repeated cycles signal
  foraging.

## Probability propagation

The engine uses a pluggable [provenance][provenance]. The default
`MaxTimesProvenance` is the **top-1-proof (Viterbi) semiring** — conjunction is
the product of probabilities, and alternative derivations combine by `max`.
`max` is idempotent, which guarantees the recursive least-fixpoint terminates at
a unique solution (equivalent to Scallop's `topkproofs` with `k = 1`). For
combining *distinct* enumerated proofs at read-out, `noisy_or` provides an
independent-OR (the assumption Scallop's `addmult` makes).

## Quick start

```python
from swarm.neurosymbolic import Trajectory, classify_behaviors

traj = Trajectory(
    agent="hunter",
    positions=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    targets={"prey": [(10, 0)]},
)
scores = classify_behaviors(traj)
print(scores.top())        # ('pursuing', 0.82)
print(scores.scores)       # {'pursuing': 0.82, 'evading': 0.0, 'foraging': 0.008}
```

Run the full demo (chaser / fleer / forager) with:

```bash
python examples/neurosymbolic_behavior.py
```

## Using the real Scallop

The in-repo engine reimplements only the slice of
[Scallop](https://www.scallop-lang.org) the framework needs, so the package
stays dependency-free and testable anywhere. To run on the real backend:

```python
from swarm.neurosymbolic import to_scallop_program, run_with_scallopy

print(to_scallop_program())          # emit the equivalent .scl source
# run_with_scallopy(program)         # execute via scallopy, if installed
```

`to_scallop_program()` requires no dependency and is handy for the Scallop
playground or documentation; `run_with_scallopy()` executes on `scallopy` if it
is importable and raises a clear install hint otherwise.

[perceiver]: https://github.com/swarm-ai-research/swarm/blob/main/swarm/neurosymbolic/perceiver.py
[provenance]: https://github.com/swarm-ai-research/swarm/blob/main/swarm/neurosymbolic/provenance.py
