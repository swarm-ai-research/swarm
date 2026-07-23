# Spike memo: should SWARM adopt a fabro-workflow-style experiment DAG?

**Issue:** distributional-agi-safety-y6to.3 (child of EPIC y6to, "Adapt Fabro components into SWARM")
**Type:** inspiration-only spike — 1-page memo + toy prototype, not a port
**Date:** 2026-07-01
**Prototype:** `examples/spikes/experiment_dag_prototype.py`
**Reference:** `~/dev/fabro/lib/crates/fabro-workflow` (DOT-based Rust pipeline runner)

## Recommendation: **PARTIAL**

Adopt fabro-workflow's *concepts* (typed nodes, conditional edges, fan-out/fan-in,
checkpoint/resume) as a thin **study-orchestration layer above scenarios**. Do **not**
adopt DOT/Graphviz syntax, do **not** port the Rust crate, and do **not** evolve the
scenario YAML format into a DAG.

## What fabro-workflow is

A directed graph parsed from Graphviz DOT. Node shapes map to handler types
(`box`=agent, `diamond`=conditional, `component`=parallel, `hexagon`=human gate,
`Msquare`=exit). Edges carry `condition`/`label` for routing. A thread-safe `Context`
flows through stages; `Checkpoint`s enable crash recovery; `model_stylesheet` assigns
LLMs by CSS-like specificity. It is a genuinely good fit for *multi-stage, branching,
retryable* AI pipelines.

## What SWARM actually has today

The `baseline → sweep → red-team → analysis → report` pipeline exists — but as **prose
inside `.claude/commands/full_study.md`**, interpreted by an LLM at runtime, plus a
scatter of imperative `examples/*.py` scripts (`parameter_sweep.py`, per-study drivers).
Two distinct layers are worth separating:

| Layer | Artifact today | Nature |
|---|---|---|
| **Single run config** | `scenarios/*.yaml` | Declarative, single config (agents, governance, payoff). Reproducible from YAML+seed. |
| **Multi-stage study** | `/full_study` markdown + `examples/*.py` | Imperative, LLM-interpreted, *not* a declarative/inspectable/checkpointed artifact. |

The gap fabro addresses is entirely in the **second** layer.

## Why PARTIAL, tied to SWARM's actual complexity

1. **Scenario YAML should stay single-config.** Scenarios are the reproducibility unit
   (invariant: "reproducible from scenario YAML + seed + exported CSVs"). Overloading
   them with pipeline control flow would break that clean contract and make a scenario
   no longer a pure config. **Reject** DAG-in-scenario-YAML.

2. **A study DAG is genuinely useful, but SWARM's control-flow needs are small.** Across
   the study commands, the pipeline is ~linear with exactly two structured features:
   - **fan-out**: seed/parameter sweep (already parallel in `parameter_sweep.py`), and
   - **one conditional gate**: a regression/threshold check (e.g. toxicity ≤ threshold,
     or "did red-team breach?") that either accepts and reports, or loops back to widen
     the sweep / tighten params.

   That is comfortably inside fabro's model — but it is *far* below what justifies a DOT
   dependency, a Graphviz toolchain, or an FFI bridge to a Rust crate. A ~200-line
   pure-Python executor (see prototype) covers it with zero new dependencies.

3. **Checkpoint/resume is the highest-value borrow.** A full study is long and
   multi-process; today a crash midway loses orchestration state (individual runs are
   reproducible, the *study* is not). Fabro's per-node checkpoint → resume is the single
   feature most worth stealing. The prototype demonstrates it: kill after `sweep`,
   `--resume` picks up at the next node with context intact.

4. **DOT buys us little.** SWARM authors are Python-first; a Graphviz file is a second
   syntax to learn and lint, and our graphs are too small to benefit from DOT's
   visualization payoff. Keep the DAG as data (Python/dataclasses or a small YAML
   *study* schema distinct from scenario YAML).

5. **Human-gate / model-stylesheet map to things we already have.** `hexagon` human
   gates ≈ `/council_review`; `model_stylesheet` ≈ the fabro-llm work already tracked in
   sibling issues (u5fv.2). No need to re-import them here.

## What the prototype demonstrates

`examples/spikes/experiment_dag_prototype.py` (runs dependency-free, dry-run):

- Typed nodes (`start/fanout/agent/gate/exit`) mirroring fabro shapes.
- Conditional edges out of the gate: `pass → report`, `fail → widen` (bounded to 2
  retries), fallback `→ report` after max widen.
- Fan-out stage that grows the seed set on each widen loop.
- Per-node JSON checkpoint + `--resume`.

Observed run: gate FAILs at toxicity 0.62, widens twice (0.52 → 0.42), PASSes, reports.
This is exactly the `/full_study` flow — but now *declarative, inspectable, and
resumable* instead of prose an LLM re-derives each time.

## Concrete next steps (if we act on PARTIAL — not in scope for this spike)

1. Define a small **study DAG schema** (new file type, e.g. `studies/*.yaml`) — kept
   strictly separate from `scenarios/*.yaml`.
2. Wrap existing entry points as node handlers (`swarm run`, `parameter_sweep.py`,
   `swarm/redteam`, `swarm/analysis`, `/write_paper`).
3. Add checkpoint/resume to `/full_study` so a study is reproducible as a whole, not
   just per-run.
4. Leave DOT, Rust FFI, and scenario-YAML changes out of scope.

## One-line verdict

Steal fabro's **ideas** (typed DAG + conditional gate + fan-out + checkpoint/resume) into
a thin Python study layer above scenarios; skip DOT, skip the port, and do not touch the
scenario YAML contract.
