#!/usr/bin/env python3
"""Inspiration spike (distributional-agi-safety-y6to.3).

Prototype expressing a SWARM study as a fabro-workflow-style DAG:

    baseline -> sweep -> red_team -> analysis -> report

with the two control-flow features SWARM studies actually use:

  1. a *fan-out* stage (seed / parameter sweep), and
  2. one *conditional gate* (did the sweep clear the toxicity/regression
     threshold? -> accept and report, else -> loop back and widen the sweep).

This is deliberately NOT a port of ~/dev/fabro/lib/crates/fabro-workflow.
It is a ~200-line, dependency-free Python sketch whose only purpose is to
let us *feel* what a declarative study DAG buys us over the current prose-in-
a-slash-command orchestration (`/full_study`). It runs in dry-run mode by
default and only prints the plan + a fake execution trace.

The accompanying memo (docs/research/fabro-workflow-dag-spike.md) uses this
prototype to justify a PARTIAL-adopt recommendation.

Run:
    python examples/spikes/experiment_dag_prototype.py
    python examples/spikes/experiment_dag_prototype.py --resume  # checkpoint demo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Tiny DAG core (the "borrow fabro's concepts, not its code" layer)
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """Fabro's Outcome, trimmed to what SWARM needs: a status + context delta."""

    status: str  # "succeeded" | "failed" | "retry"
    context: dict = field(default_factory=dict)
    # Optional routing hint: name of the edge label the node prefers.
    prefer: Optional[str] = None


@dataclass
class Node:
    name: str
    kind: str  # start | fanout | agent | gate | exit  (maps to fabro shapes)
    run: Callable[[dict], Outcome]


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    # A predicate over (outcome, context). Default edge (empty condition) is
    # the fallback taken when no conditional edge matches.
    condition: Optional[Callable[[Outcome, dict], bool]] = None


class StudyDAG:
    """Minimal executor: typed nodes, conditional edges, checkpoint/resume."""

    def __init__(self, goal: str):
        self.goal = goal
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add(self, node: Node) -> "StudyDAG":
        self.nodes[node.name] = node
        return self

    def link(self, src, dst, label="", condition=None) -> "StudyDAG":
        self.edges.append(Edge(src, dst, label, condition))
        return self

    def _next(self, current: str, outcome: Outcome, ctx: dict) -> Optional[str]:
        out = [e for e in self.edges if e.src == current]
        # Conditional edges first, then the unlabeled fallback (fabro semantics).
        for e in out:
            if e.condition is not None and e.condition(outcome, ctx):
                return e.dst
        for e in out:
            if e.condition is None:
                return e.dst
        return None

    def run(self, ckpt: Path, resume: bool = False) -> dict:
        if resume and ckpt.exists():
            state = json.loads(ckpt.read_text())
            ctx, current = state["context"], state["current"]
            print(f"↻ resuming from checkpoint at node '{current}'\n")
        else:
            ctx, current = {"goal": self.goal}, "start"

        while current is not None:
            node = self.nodes[current]
            print(f"▶ [{node.kind:6}] {node.name}")
            outcome = node.run(ctx)
            ctx.update(outcome.context)
            # Checkpoint AFTER each node so a crash resumes at the next one.
            nxt = self._next(current, outcome, ctx)
            ckpt.write_text(json.dumps({"context": ctx, "current": nxt}, indent=2))
            if outcome.status == "failed":
                print(f"  ✗ {node.name} failed — halting (checkpoint kept)")
                break
            current = nxt
        return ctx


# ---------------------------------------------------------------------------
# SWARM study stages (dry-run stand-ins for the real CLI calls)
# ---------------------------------------------------------------------------
# In a real implementation each of these would shell out to the existing
# entry points, e.g.:
#   baseline  -> python -m swarm run <scenario> --seed S
#   sweep     -> python examples/parameter_sweep.py --scenario ... --output ...
#   red_team  -> /red_team  (swarm/redteam)
#   analysis  -> python -m swarm analyze  (swarm/analysis)
#   report    -> /write_paper
# Here they just mutate context so we can watch routing work.

SCENARIO = "scenarios/baseline.yaml"
SEEDS = [42, 43, 44]
TOX_THRESHOLD = 0.5


def stage_start(ctx: dict) -> Outcome:
    return Outcome("succeeded", {"scenario": SCENARIO, "widen": ctx.get("widen", 0)})


def stage_baseline(ctx: dict) -> Outcome:
    # Single reference run to anchor the sweep.
    return Outcome("succeeded", {"baseline_welfare": 1.0})


def stage_sweep(ctx: dict) -> Outcome:
    # Fan-out over seeds (fabro `component` / join_policy=wait_all).
    widen = ctx.get("widen", 0)
    seeds = SEEDS + [45, 46][: widen]  # "widen the sweep" on the retry loop
    print(f"    fan-out over seeds={seeds}")
    # Fake toxicity that improves as we widen the sweep (more data -> tighter).
    toxicity = 0.62 - 0.10 * widen
    return Outcome("succeeded", {"seeds": seeds, "toxicity": round(toxicity, 3)})


def stage_red_team(ctx: dict) -> Outcome:
    return Outcome("succeeded", {"redteam_breaches": max(0, 2 - ctx.get("widen", 0))})


def stage_gate(ctx: dict) -> Outcome:
    tox = ctx["toxicity"]
    passed = tox <= TOX_THRESHOLD
    verdict = "PASS" if passed else "FAIL"
    print(f"    gate: toxicity={tox} vs threshold={TOX_THRESHOLD} -> {verdict}")
    return Outcome("succeeded", {"gate_passed": passed})


def stage_widen(ctx: dict) -> Outcome:
    widen = ctx.get("widen", 0) + 1
    print(f"    widening sweep (attempt {widen})")
    return Outcome("succeeded", {"widen": widen})


def stage_report(ctx: dict) -> Outcome:
    print(f"    ✓ report: toxicity={ctx['toxicity']}, "
          f"redteam_breaches={ctx['redteam_breaches']}, seeds={ctx['seeds']}")
    return Outcome("succeeded", {"report": "docs/papers/draft.md"})


def build_study() -> StudyDAG:
    dag = StudyDAG(goal="Is the governance regime toxicity-safe under sweep + red-team?")
    (
        dag.add(Node("start", "start", stage_start))
        .add(Node("baseline", "agent", stage_baseline))
        .add(Node("sweep", "fanout", stage_sweep))
        .add(Node("red_team", "agent", stage_red_team))
        .add(Node("gate", "gate", stage_gate))
        .add(Node("widen", "agent", stage_widen))
        .add(Node("report", "agent", stage_report))
        .add(Node("exit", "exit", lambda ctx: Outcome("succeeded")))
    )
    dag.link("start", "baseline")
    dag.link("baseline", "sweep")
    dag.link("sweep", "red_team")
    dag.link("red_team", "gate")
    # Conditional routing out of the gate (fabro `diamond`):
    dag.link("gate", "report", "pass",
             condition=lambda o, c: c.get("gate_passed"))
    # Bounded loop-back: widen at most twice, else give up and report anyway.
    dag.link("gate", "widen", "fail",
             condition=lambda o, c: not c.get("gate_passed") and c.get("widen", 0) < 2)
    dag.link("gate", "report")  # fallback: accept after max widen
    dag.link("widen", "sweep")
    dag.link("report", "exit")
    return dag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoint instead of starting fresh")
    ap.add_argument("--ckpt", default="/tmp/swarm_study_dag.json")
    args = ap.parse_args()

    dag = build_study()
    print(f"GOAL: {dag.goal}")
    print(f"NODES: {' -> '.join(n for n in dag.nodes)}\n")

    ctx = dag.run(Path(args.ckpt), resume=args.resume)
    print(f"\nFINAL CONTEXT:\n{json.dumps(ctx, indent=2)}")


if __name__ == "__main__":
    main()
