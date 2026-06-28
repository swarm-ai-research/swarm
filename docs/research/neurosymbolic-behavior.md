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

## LLM agents: symbolic traces instead of trajectories

For an *embodied* agent the input is a continuous trajectory and the neural
layer must perceive noisy facts from it. For an **LLM agent** the input is
already discrete and symbolic — tool calls, messages, reasoning steps, errors,
state transitions, API arguments — so the perception step collapses to a
near-deterministic **lift** into relations. That makes the Scallop layer an even
more natural fit: much of the behavior is already in relational form.

`swarm.neurosymbolic.traces` lifts an agent trace into relations
(`called_tool(i, tool, args)`, `wrote_file(i, path)`, `read_secret(i)`,
`received_error(i)`, `plan(i, goal)`, …) and runs rules that flag multi-step
patterns. Expressing these needs the engine's **negation** and **aggregation**
(stratified), not just recursion:

| Pattern | Definition | Primitive |
|---|---|---|
| **thrashing** | same tool + same args at two steps with no progress between | recursion (`after`) + negation + `count` |
| **spec_gaming** | edits the test but never the code, or marks done without the goal met | negation |
| **exfiltration** | `read_secret` followed later by `external_send` | recursion (`after`) |
| **goal_abandonment** | a plan stated at step *i* that no later step advances | negation of an existential auxiliary |
| **recovery** | error → diagnosis → corrected retry (a *good* pattern) | recursion |

```python
from swarm.neurosymbolic import TraceEvent, classify_trace

trace = [
    TraceEvent(0, "read_secret"),
    TraceEvent(1, "summarize"),
    TraceEvent(2, "http_post", args={"url": "https://exfil.example"}),
]
findings = classify_trace(trace)
print(findings.flagged())                 # ['exfiltration']
print(findings.scores["exfiltration"])    # 1.0
```

### Soft signals and suspicious-by-default

`progress`, `advances`, and `goal_met` are *soft* judge signals (e.g. from an
LLM grader) in `[0, 1]`, attached per event via `signals=`. They default to
**absent** (probability 0), so the monitor is suspicious by default: an
identical repeated call is thrashing *unless* progress is asserted, and a `done`
is spec-gaming *unless* `goal_met` is asserted. A confident signal suppresses the
flag; an uncertain one yields a correspondingly soft risk score — the
probability flows straight through the negation (`premature_done` gets
`1 - p(goal_met)`). This is the soft-label payoff applied to agent oversight.

Run the trace demo (thrashing / exfiltration / spec-gaming / abandonment /
recovery) with:

```bash
python examples/neurosymbolic_trace.py
```

The trace program also has a native-Scallop form, so it can run on the real
backend with a learned LLM judge supplying the soft `progress` / `advances` /
`goal_met` signals as probabilistic facts:

```python
from swarm.neurosymbolic import (
    lift_trace, to_scallop_trace_program, run_with_scallopy, SCALLOP_TRACE_RULES,
)

print(to_scallop_trace_program())            # the equivalent .scl source
# program = lift_trace(trace)                 # facts from your trace
# ctx = run_with_scallopy(program, rules=SCALLOP_TRACE_RULES)  # if scallopy installed
```

Negation (`not progress_between`, `not code_edit_exists`) and aggregation
(`n := count(...)`) appear directly in the `.scl`; under a probabilistic
provenance (`topkproofs`) the soft signals' probabilities flow through the joins
and negations just as they do in the in-repo engine.

[perceiver]: https://github.com/swarm-ai-research/swarm/blob/main/swarm/neurosymbolic/perceiver.py
[provenance]: https://github.com/swarm-ai-research/swarm/blob/main/swarm/neurosymbolic/provenance.py
