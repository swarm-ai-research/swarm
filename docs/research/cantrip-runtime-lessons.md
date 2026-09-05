# Runtime lessons from Cantrip (deepfates)

**Source:** [deepfates/cantrip](https://github.com/deepfates/cantrip), read at
commit 38647c8 (2026-07-23), version 1.3.3 on Hex, studied 2026-09-04. An
Elixir/OTP runtime for language-model entities by deepfates (Grove Research).
Its vocabulary: a *circle* (environment) holds a *medium* (conversation, Elixir
code, or bash), *gates* (host capabilities), and *wards* (runtime limits); the
*loom* is the append-only record of every turn; the *entity* is whatever
emerges from running the loop. Action space is written `A = M ∪ G − W`.

Cantrip is Elixir and the rig is Python, so nothing here is code to import.
What transfers is a set of design rules, five of which land on problems this
rig has already hit. Each section names the rig artifact it touches.

## 1. Wards are structural, not aspirational

The spellbook (`docs/spellbook.md`): wards "are the shape of the body the
entity inhabits, not policy the entity is asked to respect." Two mechanisms
back that sentence up.

- **Monotone composition.** When a parent spawns a child, numeric wards take
  the `min` and boolean wards take the `or`. A child can narrow its own
  authority and never widen it. Ceiling wards reject a child that declares
  more; they do not silently rewrite it.
- **Bounded-by-construction.** `Cantrip.new/1` fails validation if the circle
  lacks a medium, a `done` gate, or a truncation ward. You cannot build an
  unbounded entity by omission.

**Rig mapping.** This is the retro corollary from
[dispatch-retro-2026-07-19](dispatch-retro-2026-07-19.md) (side-effect writes
get used, voluntary protocols do not) applied to agent authority rather than
coordination. The artifact-only DONE protocol in `/bv-dispatch` is the
`done`-gate half. The monotone-composition half has no rig equivalent yet:
a dispatched sub-agent today inherits whatever the prompt says. Candidate
bead: express per-track authority as a ward set and check narrowing at
spawn time.

## 2. Termination is a gate the entity must call

The `done` gate is the only way a cast terminates cleanly; a turn-limit
truncation returns `meta.terminated == false` and the eval rubric can require
`terminated: true` and `gate_used: <name>` separately.

**Rig mapping.** fm-agent-harness (swarm-ai-research/fm-agent-harness,
ladder run 2026-08-20) found the entire ∀-feedback payoff was the gate saying
*you are not finished*: `nudge` 10/10, `tests` 0/10, because tests opened
the gate on broken code every time. Cantrip's rubric separates "ended through
the expected path" from "ended," which is exactly the measurement that
harness needs and is the cheap proxy for gate quality. Its eval doc is also
honest in the same register as ours: threshold-only CI, no baseline
management, no inter-evaluator agreement, no cost control.

## 3. Single-writer storage, stated as a rule

`docs/architecture.md`: the JSONL loom "serializes appends through an in-BEAM
per-path lock, but it is still a single-writer file format across OS
processes. Use one writer per file; use Mnesia when multiple nodes need shared
durable state." The distributed doc then concedes Mnesia partitions produce
divergent disc copies with no automatic resolution, and recommends avoiding
multi-writer topologies for audit-trail looms.

**Rig mapping.** This is bead `urch` (shared git index across concurrent
sessions, staged files swept into the other session's commit) written as a
storage invariant. Our answer is worktrees; Cantrip's answer is the same
answer at the loom layer. Worth restating in CLAUDE.md as a rule about
*writers*, not about *git*, since `runs/runs.db` and `.beads/` have the same
shape.

## 4. Verification-shaped versus verified, applied to the sandbox

Cantrip's bash medium doc says the sandbox adapter contract "is empirical,
not aspirational": CI runs a real shell workload suite (`git`, `make`, `jq`,
`find`/`sed`/`grep` pipelines) under bubblewrap or Seatbelt, and "new shell
workload expectations should land as tests first so sandbox configuration
gaps surface in CI instead of in user sessions."

**Caveat they disclose.** The CI suite runs with `bash_network: :on` because
GitHub-hosted runners cannot create the network namespace bubblewrap uses for
default denial. Network denial is therefore pinned by a flag-shape test
(`--unshare-net` present in the command), not exercised. That is a
phantom-gate pattern of the kind
[erdos-1038-swarm-lessons](erdos-1038-swarm-lessons.md) lesson 5 audits for,
except disclosed in the doc rather than discovered after the fact. The
Auditor's step-6 phantom-gate check should treat "flag pinned, behavior not
exercised" as its own category, one notch above "job never dispatched."

## 5. Keep the judge's raw output next to the verdict

Judge criteria in the eval harness store the raw LLM response inside
`report.json` "so scoring can be audited later."

**Rig mapping.** [verify-subagent-findings] measured a ~30% false-finding
rate in fan-out review. loam's admission gate and the Auditor both need the
evidence kept adjacent to the score; Cantrip does it by default. Adopt the
same rule for any rubric-scored artifact in `runs/`: the judge transcript is
part of the result record, never a log line.

## Where the trust model is weaker than ours

- Every node in a Cantrip cluster is fully trusted; a peer with the Erlang
  cookie can bypass wards by operating below the API. The doc says do not
  cluster across trust domains. swarm-safety-gate exists to gate execution
  across exactly that boundary.
- The Familiar (the packaged coding entity) defaults to `sandbox:
  :unrestricted`, host-BEAM evaluation, for operator-local work. The safe
  port sandbox is the default only for hand-built code circles.
- The sandbox protects the host BEAM and denies ambient language
  capabilities. Mounts, network egress, CPU/memory quotas, and OS user
  isolation are explicitly the deployment's job via `port_runner`. Cantrip
  tests that the runner is invoked and does not verify its security
  properties.

## Smaller mechanisms worth noting

- **Errors are observations.** A failed gate returns `is_error: true` with a
  structured message on the next turn rather than raising. The entity reads
  the failure and adapts. Matches the rig's convention that a failed tool
  call is data for the agent, not a crash.
- **Folding is a view.** Context compression replaces old turns with a
  `[Folded: turns N..M]` marker in the prompt; the loom keeps every turn.
  Same split we want between what an agent sees and what the run record
  holds.
- **Streaming backpressure.** Opt-in barriers make a slow consumer slow the
  entity rather than grow an unbounded mailbox. Relevant if the observatory
  ever streams agent events.
- **Composition is code, not a workflow graph.** Children are spawned through
  the ordinary public API from inside the code medium, with the parent
  checking `max_depth` before any child starts. The doc calls this "the RLM
  pattern in package form" (recursive language models).
- **Process inventory as contract.** The architecture doc ends with a table of
  every process kind, its owner, restart strategy, and shutdown semantics,
  and says any new process must extend the table. Cheap discipline; the rig
  has no equivalent for its tmux/worktree/hook processes.

## Status

Section 1 adopted: `swarm/agentgit/wards.py` (`WardSet`, monotone
`compose`, INV-6) landed 2026-09-05 as commit 3422cbba; the `/claim` ward
gate and `WARDS:` dispatch stamps followed under bead illq.2 (epic illq).
Section 4's phantom-gate category is now named in bv-dispatch retro item 5.
Sections 3 and 5 remain candidate beads named inline. Docs read: `docs/spellbook.md`,
`docs/architecture.md`, `docs/port-isolated-runtime.md`,
`docs/distributed-familiar.md`, `docs/eval-harness.md`. Not yet read:
observability, acp-editor, public-api, signer-key-runbook, cleanup-status.
