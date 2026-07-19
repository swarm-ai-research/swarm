---
description: Graph-theoretic fleet dispatch — analyze the beads graph with bv, assign optimal work to each agent, notify via agent mail
argument-hint: "[plan|hygiene|retro] [extra context]"
allowed-tools: Bash(bd:*), Bash(bv:*)
---

# Mode

Arguments: $ARGUMENTS

- **(no mode)** — full dispatch: analyze, assign, write rationale into beads, mail agents.
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

Any text after the mode word is extra operator context — honor it.

# Snapshot (taken at invocation — trust this over memory, verify before mutating)

## Ready frontier
!`bd ready 2>/dev/null || echo "(bd unavailable from this directory — cd into a rig first)"`

## In flight
!`bd list --status in_progress 2>/dev/null || echo "(none/unavailable)"`

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

Then produce the assignment: for each agent, ONE primary and ONE backup bead,
matched to their specialty, each drawn from a different parallel track. Justify
every pick with the actual metrics (e.g. "bd-142: unblocks 6, on critical path,
betweenness 0.31"), not vibes.

Consolidation quota (Big Ball of Mud discipline): at least ONE agent's primary
per round must be consolidation work — hygiene, missing edges, stale triage,
test/CI repair, dead-code or artifact cleanup — not feature expansion. Swarms
generate mud faster than humans; the counter-entropy investment has to be
structural, not aspirational. If the round has no consolidation candidate,
say so explicitly — that itself is a finding (the backlog isn't seeing its
own mess).

**Full dispatch mode only** (skip all writes in plan/hygiene/retro modes):

Before mailing anyone, write the rationale INTO each assigned bead as a comment
(metrics + predicted unblock count + why this agent) so the reasoning survives
outside the mail thread and a future `retro` run can score these predictions.

Then deliver each assignment through the rig's coordination channel, in this
order of preference:
1. agent_messages table in runs/runs.db (schema in CLAUDE.md): one broadcast
   row per assignment — from_agent='bv-dispatch-r<N>', to_agent='#swarm',
   body='ASSIGN: <bead-id> -> <PERSONA> (<metrics>). <one-line scope>. Claim:
   bd update --status in_progress, then reply CLAIM here. Backup: <id>.'
   Also READ the channel first: ack any unacked CLAIM/DONE rows and fold them
   into this round's analysis (a claimed bead is not assignable).
2. Agent mail (MCP), if configured.
If neither exists, say so and stop — do not simulate delivery. Comment stamps
on beads remain the durable record either way; the channel is the delivery
mechanism (beads vw8g/x3q7: stamps alone provably do not reach claimants).

Claim protocol (include in every delivery): mark in_progress immediately; if
the assignment is >4h old or an upstream bead in your track has closed since,
do not trust it — re-run the dispatch analysis yourself and claim the
top-ranked unclaimed bead from your track instead.

**All modes:**

Reply with a summary table (agent → bead → unblocks → why) and a Mermaid diagram
of the dependency graph showing the critical path highlighted and each agent's
track color-coded. Use ultrathink.
