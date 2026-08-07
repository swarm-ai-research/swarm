---
description: "Shepherd's reversible execution traces: SWARM already confines writes at the OS level, but writes land by default — the delta is propose-then-apply, and prevention doesn't decompose the way detection does."
---

# Shepherd: reversible execution and the prevention/detection split

**Source:** [Shepherd: Enabling Programmable Meta-Agents via Reversible Agentic
Execution Traces](https://arxiv.org/abs/2605.10913) (arXiv:2605.10913v3,
submitted 11 May 2026, revised 24 Jun 2026) — Simon Yu, Derek Chong, Ananjan
Nandi, Dilara Soylu, Jiuding Sun, Christopher D. Manning, Weiyan Shi. Code:
[shepherd-agents/shepherd](https://github.com/shepherd-agents/shepherd)
(MIT, Python 3.11+, alpha). Studied 2026-08-06.

Companion to [ksi-knowledge-curation.md](ksi-knowledge-curation.md): that note
argues knowledge-centric self-improvement removes the substrate governance acts
on. Shepherd is a candidate substrate — and the closest external system to
`swarm/bridges/worktree/`, which makes the comparison unusually concrete.

## What the paper does

Meta-agents — higher-order agents that create, operate on, and manage other
agents — need to coordinate agents, halt risky actions before execution, and
repair failed runs. The paper's claim is that existing substrates expose only
transcripts and environment snapshots, so meta-agents build ad hoc tooling to
reconstruct execution state. Shepherd makes **an agent's execution a
first-class object** that a meta-agent can inspect and transform: fork, replay,
revert.

Two design choices carry the system:

**Permissions are the type signature.** A task is a Python function with a
signature and docstring but no body; the signature is the contract the agent
fulfills at runtime.

```python
def write_program(repo: sp.GitRepo, prompt: str, output_path: str = "program.py") -> None:
    """Write a self-contained Python program doing what prompt requests."""
```

`repo: sp.GitRepo` grants read-write; `May[GitRepo, ReadOnly]` restricts to
inspection. In the README's phrasing: *"Reading the signature is reading the
permission surface."* These compile to syscall-level enforcement via macOS
Seatbelt and Linux Landlock (the latter requiring a privileged container).

**Outputs are retained, not applied.** Agents do not modify live directories.
A run produces retained workspace outputs and per-binding changesets that are
inspected, tested, and then selectively applied or discarded
(`shepherd run select/apply/discard`), with three-way merge when the workspace
has moved on. Copy-on-write forking is reported ~5× faster than docker commit
and fork, with ~95% KV-cache reuse on replay.

### Reported results

- Supervisor meta-agent: pair-coding pass rate **28.8% → 54.7%** on CooperBench.
- Counterfactual optimization: beats MetaHarness on Terminal-Bench 2.0 by
  **12.8%** with **58% lower wall-clock** time.
- Training meta-agent **doubles GRPO's uplift** on Terminal-Bench 2.0.

Alpha status, explicitly unstable APIs; Linux enforcement needs a privileged
container, which is a real constraint for CI where jobs are already
containerized.

## What SWARM already has (and the correction it forces)

It is tempting to read Shepherd as "real enforcement, unlike advisory gates."
That reading is wrong about this repo. `swarm/bridges/worktree/sandbox_launch.py`
already implements OS-level confinement in the same family, and its docstring
states the same core insight independently:

> `cwd` is only a working directory, not a jail: an allowlisted `python` can
> still write anywhere on disk and open network sockets.

SWARM wraps argv in `sandbox-exec` (macOS SBPL) or `bwrap` (Linux bubblewrap):
writes confined to the sandbox subtree, network denied via an empty namespace,
plus a targeted deny-list on credential reads (`.ssh`, `.aws`, `.env`, `*.pem`).
`swarm/agentgit/capabilities.py` derives that allowlist from a *cryptographic
delegation chain*, deny-by-default, and its docstring makes the attestation
critique explicitly:

> verification alone is still advisory — it proves what an agent *was allowed*
> to do without stopping it from doing more.

So the enforcement layer is not the gap. Two real deltas remain.

**Delta 1 — where authority comes from.** SWARM derives permissions from a
signed `human → org → agent` delegation chain: authority is *granted* and
provenance-bearing. Shepherd derives them from the task function's type
signature: authority is *code*, which is far better ergonomics but carries no
notion of who granted it or whether the grant was in scope. These compose
rather than compete — Shepherd's signature-as-permission-surface is a better
authoring interface over exactly the capability set SWARM already computes.

**Delta 2 — when the write lands.** This is the substantive gap. SWARM's
sandboxed agent writes *into its worktree*, and governance evaluates afterward:
pre-commit hooks, the AgentGit policy gate, CI. The write is real the moment it
happens; the gate decides whether it survives. Shepherd's retained outputs
never touch the live workspace until explicitly selected.

That is **default-accept versus default-propose**, and it is the same structure
as [erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md) lesson 1: the
default-⚪ ledger, where claims enter unverified and upgrade only through
verification. Shepherd applies default-unverified to *code changes* rather than
to *claims*. This repo has a concrete exhibit of the failure the propose model
prevents: on 2026-07-27 a concurrent-commit race absorbed one session's staged
files into another session's commit under the wrong bead id (CLAUDE.md
§ Concurrency guard). Staged-but-unlanded changes were corruptible precisely
because the workspace was shared and writes were live. Under propose-then-apply
that race cannot corrupt provenance, because nothing lands unreviewed.

## The research claim worth testing

**Prevention does not decompose the way detection does.**

Bead `3ru4`'s bond × detection frontier found that the per-interaction
`RandomAuditLever` collapses to **exactly 0.000** catch probability once an
attacker keeps each interaction one epsilon above `audit_threshold_p` — the
same step-decomposition that blinds action filters and trajectory monitors.
Every detection lever in `swarm/governance/` shares that vulnerability: it
scores an action after the fact, so an attacker who decomposes below the
scoring threshold is invisible at any honest-false-positive budget.

A capability precondition has a different shape. A write outside the granted
binding is refused at the syscall regardless of how the attacker decomposes it,
because there is no threshold to slip under — the check is on *authority*, not
on *suspiciousness*. Decomposing an unauthorized write into a thousand small
unauthorized writes yields a thousand refusals, not zero detections.

If that holds, it is a sharp and useful asymmetry: detection levers degrade
gracefully-to-uselessly against decomposition, while prevention levers do not
degrade at all — but prevention only covers what can be expressed as a
capability boundary, which excludes most of what SWARM cares about (a *harmful*
change to a file you are authorized to edit). The interesting quantity is the
**coverage/robustness tradeoff**: prevention has narrow coverage and flat
robustness; detection has broad coverage and cliff-shaped robustness. SWARM
currently models only the detection side.

## Mapping to SWARM constructs

| Shepherd mechanism | SWARM analogue | Gap |
|---|---|---|
| Syscall enforcement (Seatbelt/Landlock) | `sandbox_launch.py` (`sandbox-exec`/`bwrap`) | None — equivalent layer |
| Signature-as-permission-surface | `capabilities.py` from a signed delegation chain | Authoring ergonomics; SWARM has provenance Shepherd lacks |
| Retained outputs, `select/apply/discard` | *No analogue* — worktree writes are live, judged afterward | **Default-accept vs default-propose** (bead `hpf3`) |
| Fork / replay / revert as first-class ops | Event logs are replayable; execution state is not forkable | Reversibility as a governance lever (bead `hpf3`) |
| Prevention as capability precondition | Detection levers (`RandomAuditLever`, monitors) | Prevention/detection robustness split (bead `vwj3`) |

## Caveats before adopting anything

- Alpha, APIs unstable between releases.
- Linux enforcement requires a **privileged** container — awkward inside CI.
- The benchmark numbers measure *capability* uplift from meta-agent
  supervision (CooperBench, Terminal-Bench 2.0), not safety under an adversary.
  As with KSI, there is no threat model: nothing here is evaluated against an
  agent trying to escape its bindings.
