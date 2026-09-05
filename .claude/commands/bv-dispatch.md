---
description: Graph-theoretic fleet dispatch — analyze the beads graph with bv, publish ranked per-track digests agents pull from at claim time (push ASSIGN only for strategic beads)
argument-hint: "[plan|hygiene|retro] [extra context]"
allowed-tools: Bash(bd:*), Bash(bv:*)
---

# Mode

Arguments: $ARGUMENTS

- **(no mode)** — full dispatch: analyze, publish per-track ranked digests to the
  channel, stamp assignee-free rationale on shortlisted beads; push ASSIGN mail
  only for strategic beads (critical path, epics). Agents pull from their
  track's digest at claim time (beads-x3q7 — see "Pull-model delivery").
- **plan** — analysis and assignment table ONLY. Zero writes: no bead comments, no
  dependency edges, no mail. Safe to run anywhere.
- **hygiene** — run only the hygiene pass (step 5): cycles, orphans, stale
  in-progress beads, missing edges. Propose fixes; apply only `bd dep add` /
  comment fixes the operator approves.
- **retro** — grade the last dispatch: find rationale comments from previous runs
  (they contain predicted unblock counts and metric values), compare predictions
  against what actually closed/unblocked since, and report prediction accuracy.
  This is how the dispatch policy learns whether its graph theory is working.
  Also produce the MUD LEDGER (Foote & Yoder, Big Ball of Mud):
  1. Entropy ratio — beads closed by consolidation work (hygiene, sweeps,
     cleanup, test/CI fixes) vs new beads opened, since last retro. A rig
     that only expands is building a shantytown; report the ratio and trend.
  2. Rug audit — find results flagged "machinery validation only", mock-based
     numbers, or similar cordons in bead comments; verify nothing downstream
     cites them as real measurements. A rug becomes load-bearing silently.
  3. Shearing-layer check — fast-layer defaults (library constants, tool
     versions, config defaults) that have drifted against slow-layer
     commitments (prereg locks, frozen schemas, append-only principles).
     The pattern: grep frozen/locked/prereg commitments, compare each to the
     current default it constrains. Every hit is a naa8-class incident waiting.
  4. Orphan influx (Morgenthaler et al., "Searching for Build Debt", MTD'12:
     make the default the right thing) — of beads created since last retro,
     what fraction arrived with zero edges? The wrong default (edgeless
     creation) is a debt factory; sweeps and sessions should file with
     discovered-from/blocks edges at creation time. Report the fraction,
     name the worst offender source, and propose the default-side fix
     (creation-time nudge or template) rather than another cleanup pass.
  5. Gate execution audit (erdos-1038 lesson: verification-shaped ≠ verified;
     docs/research/erdos-1038-swarm-lessons.md) — for each bead closed as
     verified/tested since last retro, confirm the actual gate ran: test
     output or run-folder path in a comment, CI run that executed (not just
     a lint/regex proxy job), benchmark numbers from a real invocation.
     Distinct from the rug audit: a rug is a cordoned mock cited as real; a
     phantom gate is a verification step everyone believes ran and never
     did. Where the bead carries a `WARDS:` stamp, check the DONE against
     that stamp's `negative_spec` list (`python -m swarm.agentgit wards show
     <id>`) rather than against the prose; a "flag pinned, behaviour not
     exercised" gate (Cantrip lesson 4) counts as phantom, one notch above
     "job never dispatched". Also audit blocked-bead reopenings: each must name the materially
     new mechanism that justified reopening (see analysis step 6).

Any text after the mode word is extra operator context — honor it.

# Snapshot (taken at invocation — trust this over memory, verify before mutating)

## Ready frontier
!`bd ready 2>/dev/null || echo "(bd unavailable from this directory — cd into a rig first)"`

## In flight
!`bd list --status in_progress 2>/dev/null || echo "(none/unavailable)"`

## Effective staleness (comment/run-aware — beads-y0cx)
!`python scripts/dispatch_staleness.py 2>/dev/null || echo "(scripts/dispatch_staleness.py unavailable — fall back to updated_at, but treat naive staleness as an upper bound: comments and runs/ artifacts don't bump it)"`

## Recently closed
!`bd list --status closed --limit 10 2>/dev/null || echo "(none/unavailable)"`

## bv availability
!`bv --help 2>/dev/null | head -30 || echo "bv NOT INSTALLED — fall back to bd dep tree, bd list --json, and manual graph analysis"`

# Task

Re-read AGENTS.md first — note each agent's specialty, current rig, and anything
they have in flight. If there is no agent roster, say so and use hypothetical
slots (Agent A/B/C) rather than inventing names.

Use bv's robot/JSON modes for everything below if available (see snapshot above);
otherwise fall back to `bd dep tree` + `bd list --json` and your own analysis.
The snapshot above is invocation-fresh; re-query anything you are about to mutate.

Analyze the open task graph with graph theory, and show your numbers:
1. Critical path — the longest dependency chain; which beads sit on it.
2. Unblock fan-out — for each ready (unblocked) bead, how many downstream beads
   its completion transitively frees. Rank by this.
3. Bottlenecks — betweenness centrality / articulation points: beads whose delay
   stalls multiple chains. These outrank raw priority fields.
4. Parallel tracks — decompose the ready frontier into independent subgraphs
   (no shared ancestors/descendants, no shared files or rig) so agents never
   collide or duplicate work. Check file-level overlap, not just graph edges —
   an edgeless graph does not prove tasks are parallel-safe.
5. Hygiene alerts — cycles, orphans, stale in-progress beads, and MISSING EDGES
   (dependencies that plainly exist in the descriptions but were never recorded).
   List them; assign the cheapest agent to fix graph hygiene before feature work.
   Staleness MUST use the effective-staleness snapshot above (beads-y0cx), not
   raw updated_at: bd comments and runs/ artifacts don't bump updated_at, so
   naive staleness cries zombie on actively-worked beads (k2yr was flagged
   stale-46d in r4 with a same-day run + comments). A bead is a zombie only if
   its EFFECTIVE staleness (max of updated_at, last comment, newest referencing
   run artifact) exceeds the threshold.
6. Approach-family registry (research beads only; erdos-1038 lessons 2–4) —
   when several open beads attack the same research question, cluster them by
   underlying approach, not surface wording. Flag convergence: if most routes
   collapsed into one family, redirect an agent to an underexplored family;
   elegance or compelling early numbers are NOT grounds for letting one family
   dominate. Keep ≥2 incompatible routes alive while the question is open.
   Blocked routes: a bead stalled on a theorem-strength gap gets a comment
   naming the exact blocker; reopen it only when a claimant names a materially
   new mechanism in the bead (retro audits these — MUD LEDGER item 5).
7. Gate-cost ranking (erdos ecosystem lesson 7: search aggression scales
   inversely with gate cost) — classify each exploratory bead's acceptance
   gate:
   - G0 mechanical: pass/fail without human judgment (failing test, DB
     trigger, deterministic seeded benchmark, concrete counterexample).
   - G1 replayable: seeded run whose numbers the Auditor can re-derive.
   - G2 judgment: design decisions, prose claims, prereg interpretation.
   Aggressive search — multiple agents, high persistence, loose priors —
   goes only to G0/G1 beads, where a wrong confident answer dies cheaply at
   the gate. G2 beads get ONE conservative agent plus a named reviewer;
   never point never-give-up prompting at a judgment gate.

Then produce the ranking (beads-x3q7 pull model): for each persona/track, a
ranked shortlist of 2–4 beads from that track, best first — the top pick is
the *presumptive* primary, resolved at claim time, not a binding assignment.
Mark beads that qualify as **strategic** (on the critical path, or epics whose
ranking will not churn within a day): only these get push-ASSIGN treatment.
Justify every ranking position with the actual metrics (e.g. "bd-142:
unblocks 6, on critical path, betweenness 0.31"), not vibes. State DIRECT
unblocks separately from transitive ones — retro 2026-07-19 half-credited a
prediction that conflated them. For research beads, include the gate class
(step 7) in the justification and let it set the aggression: G0/G1 top picks
may carry "persist until the gate passes" instructions; G2 top picks must not.

Consolidation quota (Big Ball of Mud discipline): at least ONE agent's primary
per round must be consolidation work — hygiene, missing edges, stale triage,
test/CI repair, dead-code or artifact cleanup — not feature expansion. Swarms
generate mud faster than humans; the counter-entropy investment has to be
structural, not aspirational. If the round has no consolidation candidate,
say so explicitly — that itself is a finding (the backlog isn't seeing its
own mess).

**Full dispatch mode only** (skip all writes in plan/hygiene/retro modes):

**Pull-model delivery (beads-x3q7).** Push-stamping per-agent assignments rots
at sweep churn — r4 stamped backups (909k/tbrj) that were closed before the
comment landed. The dispatcher therefore publishes *rankings*, not
*assignments*; agents resolve who-does-what at claim time, when the graph state
is fresh, and the atomic `/claim` gate (beads-wrr9) makes the resolution safe.

Bead stamps carry **assignee-free rationale only** (metrics, predicted unblock
counts, gate class, negative spec — never "assigned to X"): stamp the top 1–2
beads of each track's shortlist. Rationale describes the task, so it stays
valid regardless of who claims or when; predictions in these stamps are what a
future `retro` scores.

The NEGATIVE SPEC lives in that rationale stamp: 2–3 bead-specific
insufficient outcomes — the known cheats for that task class (e.g.
"single-seed result presented as general", "mock numbers without cordon",
"fix that loosens the assertion instead of the code"). The Auditor and retro
verify against this list, not against generic rigor.

**The stamp's last line is machine-readable (illq.2, Cantrip wards —
docs/research/cantrip-runtime-lessons.md §1).** Generate it, never hand-write
it:

    python -m swarm.agentgit wards stamp --gate-class <G0|G1|G2> \
        --negative "<insufficient outcome 1>" --negative "<outcome 2>" \
        [--permission read --permission test ...] <bead-id>

This posts `WARDS: {json}` to the bead: the gate class sets the turn/depth/
children budget (G0 12/1/4, G1 8/1/2, G2 4/0/0 — aggression scales inversely
with gate cost, step 7), permissions name what the track may do, and the
negative spec is carried as a list. `/claim` composes that stamp with the
claimant's own declaration and refuses a claim that would widen it (exit 3);
the effective set lands in `.agentgit/current-claim.json`. Retro reads
`negative_spec` from the stamp (`python -m swarm.agentgit wards show <id>`),
not from prose. For research beads in an
exploration round (approach registry active, step 6), the digest states the
question and the track's assigned family but OMITS the currently favored
route — independence first, cross-pollination after routes mature.

Then deliver through the rig's coordination channel, in this order of
preference:
1. agent_messages table in runs/runs.db (schema in CLAUDE.md):
   - One broadcast row **per track** — from_agent='bv-dispatch-r<N>',
     to_agent='#swarm', body='TRACK: <PERSONA> -> ranked [<bead-1>
     (<metrics>), <bead-2> (<metrics>), ...]. Pull protocol: claim the
     top-ranked bead still open+unclaimed via /claim; digest stale after 4h.'
   - One ASSIGN row **only for strategic beads** — critical-path or epic
     beads whose ranking will not churn within a day: body='ASSIGN:
     <bead-id> -> <PERSONA> (<metrics>). <one-line scope>. Claim via
     /claim <bead-id>, then reply CLAIM here. Backup: <id>.'
   Also READ the channel first: ack any unacked CLAIM/DONE rows and fold them
   into this round's analysis (a claimed bead is not rankable-as-open).
2. Agent mail (MCP), if configured.
If neither exists, say so and stop — do not simulate delivery. Rationale
stamps on beads remain the durable record either way; the channel is the
delivery mechanism (beads vw8g/x3q7: stamps alone provably do not reach
claimants).

Claim protocol (include in every delivery): claim atomically via
`/claim <bead-id>` — never bare `bd update --status in_progress` (advisory
only; the 7ge5 duplicate incident). Pull rules at claim time:
1. Take your track's digest; verify its top pick with fresh state
   (`bd show <id>` — still open? `/claim` refuses if another session holds
   it; a refusal means take the next-ranked, not a fight).
2. If the digest is >4h old, or any bead in your track closed since it was
   published, do not trust the ordering — re-run the dispatch analysis
   yourself in `plan` mode (bv or the in-rig fallback:
   `scripts/dispatch_staleness.py` + `bd ready` + this file's steps 1–4)
   and claim the top-ranked unclaimed bead from your track.
3. Strategic ASSIGN rows override the digest for their addressee: claim the
   assigned bead first unless it is closed or refused.

DONE protocol (include in every delivery): a `DONE:` row must point at a
concrete artifact — commit hash, run folder, or `artifact=<ref>` — and state
which gate ran via `gate=<check>:<result>` (or `gate=none`). The artifact
half is enforced mechanically by the `done_requires_artifact` trigger on
agent_messages (schema in CLAUDE.md); the gate half is checked here: in
retro mode, grep DONE rows since last retro for `gate=` and feed any
gateless or `gate=none` completions into MUD LEDGER item 5. Status reports
and "routine, just needs X" are not DONE; bounce them back to in_progress
with a comment naming what's missing (erdos-1038 lesson 5).

**All modes:**

Reply with a summary table (track → ranked beads → unblocks → gate class →
why, strategic ASSIGNs marked) and a Mermaid diagram
of the dependency graph showing the critical path highlighted and each agent's
track color-coded. Use ultrathink.

# Protocol migration

| Old behavior | Replacement | Why |
|---|---|---|
| Per-agent ASSIGN row + assignee stamp on every assigned bead | `TRACK:` ranked digest per persona; assignee-free rationale stamps on shortlist top picks; agents pull + `/claim` at claim time | Push stamps rot in minutes at sweep churn — r4 stamped backups (909k/tbrj) already closed before the comment landed (beads-x3q7) |
| ASSIGN for everything | ASSIGN only for strategic beads (critical path, epics) | Slow-moving rankings are the only ones that survive the latency between stamping and claiming |
| Claim = `bd update --status in_progress` | Claim = `/claim <bead>` (atomic, refuses if held) | Advisory status can't prevent duplicates — 7ge5 incident (beads-wrr9) |
