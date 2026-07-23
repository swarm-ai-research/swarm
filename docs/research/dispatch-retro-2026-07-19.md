# Dispatch retro — 2026-07-19 (rounds 1–5, first retro)

Grades the first day of `/bv-dispatch` operation (2026-07-18 → 07-19).
Next retro (scheduled 2026-07-21 09:23, launchd) diffs against the baselines here.

## Headline stats

| Metric | Value |
|---|---|
| Dispatch rounds run | 5 |
| Beads created (day) | 49 — 6 with edges at creation (12%) |
| Beads closed (day) | 43 — 31 consolidation / 12 expansion (72% consolidation) |
| Entropy ratio (closed:opened) | **0.88** ← baseline for trend |
| Orphan influx | **88% edgeless at creation** ← baseline |
| Channel messages | 8 (6 ASSIGN, 1 CLAIM, 1 DONE) — first ever; table was empty since 2026-02-28 |
| Claims via comment stamps (r1–r4) | 0 in ~7 h |
| Claims via channel + read path (r5) | 1 in 18 min (`b61g`, shipped in ~1 h) |

## Prediction scorecard

- **Confirmed (7):** `naa8` rubric drift real; `8ms9` disk priority; `k8am`; h5bo
  articulation strengthening (0.029→0.050); backup-stamp rot (r4); staleness
  false-zombies (`k2yr`); **`vw8g` delivery-not-prioritization hypothesis —
  confirmed experimentally by a channel-blind probe session.**
- **Half (1):** `b61g` "unblocks 2" — direct unblock was 1 (`2m9z`, verified
  unblocked); `7ge5` was parallel, not downstream.
- **Wrong (1):** `36t6` as top ready bead — an artifact of the missing
  `u5fv.2→36t6` edge; the graph lied, the math didn't. Self-corrected r2.
- **Unresolved (4):** calibration megatrack unfreeze; `qoro`; `bv3p`/`se2m`;
  Scout's inferred edge (untested until `u5fv.2` moves).

## Mud ledger №1 (Foote & Yoder; Morgenthaler et al.)

1. **Entropy 0.88 / 72% consolidation closures** — healthy first reading; net
   +6 beads expansion; strategic beads moved by effort-hours, closures were small.
2. **Rug audit: clean.** Mock-jitter α=0.970/ICC=0.990 cited nowhere without the
   "machinery validation only" cordon. One rug under load: `gjc0` smoke builds on
   jittered mocks (legitimate — tests machinery). Watch: real-judge Arm B data
   must not inherit the mock alpha as a prior.
3. **Shearing: known fault properly jointed** (`PREREG_RUBRIC_VERSION` names the
   v1 lock at the collector, loud deviation warning; `joined.v1` test-enforced).
   No new lock-vs-default drift found.
4. **Orphan influx 88%; worst offenders pg-cleanup (26/26 edgeless) and
   joel-test (5/5).** All edge-bearing beads were post-discipline
   (`discovered-from` at creation). Fix is default-side, not another cleanup:
   bead `hg-orphan-default` (see below).

## Lessons (durable)

1. **Side-effect writes get used; voluntary protocols don't.** runs.db tables
   written by tooling accumulated (scenario_runs 100 rows); both coordination
   mechanisms requiring adherence sat at zero for five months. Channels must
   live in the default read path (SessionStart hook + AGENTS.md), not in docs.
2. **The graph is only as good as its edges.** Every dispatch mistake traced to
   under-declared dependencies, never to the graph theory. Prevention (edges at
   creation) beats hygiene rounds.
3. **Push rots at swarm churn.** Stamped backups died before the stamps landed.
   Assignments should carry rationale; claim-time analysis picks the bead (x3q7).
4. **Declared-live ≠ moving.** The calibration arms needed operator gates
   (criteria decision, API budget), not agent effort. Dispatch can arm gates;
   only the operator fires some of them.
5. **Acceptance tests on process beads.** vw8g closed on a demonstrated claim,
   not on "messages sent." Process fixes need falsifiable acceptance criteria.

## Action items → beads

- `x3q7` (existing): stop stamping backups; pull-model claims.
- `y0cx` (existing): staleness metric counts comments/runs.
- **new** sweep-tooling orphan default: pg-cleanup + joel-test file with
  `--deps discovered-from:<source>` at creation.
- Done inline: dispatcher now acks inbound CLAIM/DONE rows (rows 8–9 acked
  this retro); command file requires direct-vs-transitive split in unblock
  predictions.
