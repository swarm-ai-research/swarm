# AI Village → SWARM: mapping design

**Source:** [`aidigestorg/ai-village`](https://huggingface.co/datasets/aidigestorg/ai-village)
(AI Digest). Access granted 2026-09-21; figures below measured against the
export of 2026-09-20 (`manifest.json`), not the dataset card's prose, which is
roughly 2× stale. Tracked as bead `vu70`; the calibration question it feeds is
`fcmy.7`.

This is the design gate that has to clear before any bridge code is written. It
answers five questions — which regime, what counts as an interaction, what
stands in for task progress, how the addressing heuristic gets validated, and
what serves as the outcome variable. The fifth was raised by the first four. It
is answered here, but not in the shape the bead assumed: the Village supplies a
gate decision, not an interaction outcome.

Every figure below is reproducible with `experiments/ai_village_probe.py`.

## What the data actually is

| | pre-cutover | documented | undocumented |
|---|---|---|---|
| window | 2025-04-02 → 2026-03-24 | 2026-03-24 → 2026-07-03 | 2026-07-03 → 2026-09-20 |
| computer-use turns | 749,713 | 658,135 | 1,102,639 |
| sessions | 25,844 | 19,013 | 33,505 |
| consolidations | 0 | 18,854 | 33,471 |
| chat messages | 105,950 | 28,215 | 49,320 |
| distinct chat speakers | 24 | 23 | 32 |
| turn error rate | 5.36% | 7.26% | 5.92% |
| repeat-of-previous-action rate | 4.35% | 6.08% | 4.43% |

Totals: 2,510,487 turns, 381,610 events, 246,151 memories, 183,485 chat
messages, 78,362 sessions, 46 agents across the life of the village.

## D1 — Regime: post-cutover, primary window 2026-03-24 → 2026-07-03

`CHANGELOG.md` states that the 2026-03-24 perma-computer-use rollout is "the
single biggest structural change in the dataset" and that data either side
should be treated as different regimes. The event vocabulary bears that out —
it is close to disjoint:

| actionType | pre | post |
|---|---|---|
| `WAIT` | 36,022 | 0 |
| `PAUSE` | 635 | 39,837 |
| `CONSOLIDATE` | 0 | 52,325 |
| `START_USING_COMPUTER` | 25,839 | 136 |
| `STOP_USING_COMPUTER` | 25,809 | 130 |
| `OUTREACH_APPROVAL_*` | 0 | 695 |

**The reason to choose post-cutover is semantic uniformity, not stability.**
It is worth being exact about this, because the tempting claim is false. The
changelog lists 90 dated scaffolding changes: 63 before the cutover over 356
days (0.18/day) and 27 after over the 101 documented days (0.27/day). By
documented change density the chosen window is *churnier* than the one being
rejected. Some of that is likely recency bias in the changelog's own coverage,
which is an argument for distrusting both numbers rather than for preferring
either window.

What post-cutover actually buys is that every session means the same thing.
Agents are permanently in computer-use mode and consolidate every ~40 actions,
so the consolidation-bounded session is a uniform, mechanically-defined unit
(18,854 of them in the primary window). Pre-cutover, session boundaries are
agent-chosen, so "session" is itself a behavioural variable — it cannot serve
as a neutral observation unit.

Consequences to carry forward:

- Scaffolding-change dates are **known breakpoints**, not noise the window
  escapes. Fit sensitivity to them explicitly; do not assume the window is
  homogeneous just because it is inside one regime.
- The undocumented window holds 1.1M turns — more than the primary window. It
  is the densest part of the corpus and the least explicable, because the
  changelog stops at 2026-07-03 while data runs to 2026-09-20. **Hold it out
  as a replication set, do not discard it.** A fit that reproduces there is
  more convincing than one that cannot be checked; a fit that fails there is
  informative about scaffolding sensitivity.

## D2 — The interaction: session-anchored, chat-derived counterparty

`SoftInteraction` is dyadic: initiator × counterparty, with a joint outcome.
**The Village records no dyadic interactions.** Computer use is solo; chat is
broadcast into rooms. There is no pairwise unit with an outcome anywhere in the
schema, so the bridge must construct one. The construction proposed here:

- **Unit** — one consolidation-bounded session `S` of agent `A`.
- **Initiator** — `A`.
- **Counterparty** — derived from chat. Chat is reachable from the turn stream:
  `send_message_back_to_chat` is an `agent_action` (100,354 occurrences), so
  messages are actions *inside* sessions and join cleanly to `S`. Take the
  messages `A` sent during `S` that name exactly one other agent `B`; the modal
  `B` is the counterparty.
- **Observables** — `rework_count` from consecutive-identical `agent_action`
  within `S`; `verifier_rejections` from turns in `S` with non-null `error`;
  `counterparty_engagement_delta` from whether `B` replied naming `A` inside a
  window.

Both turn-level channels are externally generated and carry usable signal at
these rates (7.26% error, 6.08% repeat in the primary window), so neither is a
self-report.

**The trap to design against.** Most sessions will have no chat counterparty at
all. Dropping those sessions is a selection on sociability, and sociable
sessions are plausibly different in quality from heads-down ones. That selection
can manufacture exactly the effect `quality_gap` exists to detect — an apparent
adverse-selection signal that is an artifact of which sessions were kept.
Report the retained fraction, and compute the metrics on the dropped-vs-kept
split before reading anything into the headline number.

## D3 — Task progress: no admissible observable exists

This is the finding that most changes the plan. Every candidate fails:

| candidate | volume | why it fails |
|---|---|---|
| `summaries.jsonl` content | 939 | `SCHEMA.md`: generated **without** seeing inside computer-use sessions; "treat as secondary" |
| `STOP_USING_COMPUTER.summary` | 25,809 pre / 130 post | self-report, and absent from the chosen regime |
| `CONSOLIDATE.nextSessionGoal` | 52,325 | forward-looking intent, not retrospective progress; also self-report |
| `computer_use_sessions.has_been_asked_to_stop` | **1 in 78,362** | field is effectively dead |
| `USER_TALK` | 2,228 post | sparse and not attributable to a task |

The two survivors of the self-report objection (`STOP_USING_COMPUTER.summary`,
`CONSOLIDATE.nextSessionGoal`) fail it anyway, and they fail it in the specific
way that matters here. `fcmy.7` exists because `beta_swarm`'s deceptive
archetype inflates `task_progress_delta` to ~0.75 regardless of true quality,
making progress an actively misleading channel and engagement the discriminating
one — and the posterior recovered that, correctly and circularly. Feeding
agent self-reported progress in as `task_progress` would rebuild that same
circularity out of new parts. If engagement won again, we would have arranged
for it again.

**Decision: drop `task_progress` and fit three channels** (rework, verifier,
engagement), reporting plainly that the default 0.4-weight channel has no
admissible source in this corpus.

This cuts both ways and the writeup must say so. Removing progress removes the
built-in answer, which strengthens the test. But it also means the Village
**cannot** test the thing `fcmy.7` most wants tested — whether a fit correctly
*down*-weights a misleading progress channel — because there is no progress
channel to down-weight. The Village yields a partial answer to `fcmy.7`, over
three channels, not the complete one.

## D4 — Validating the naming → addressing heuristic

The counterparty construction rests on "names exactly one agent" standing in for
"addresses that agent". Measured on the main room (149,307 of 183,485 messages,
44 distinct speakers): 56.1% name at least one agent, and 50,987 name exactly
one. 26.7% contain an `@handle`.

Naming is not addressing — a message may discuss `B` rather than speak to it.
Before the heuristic is used:

1. Draw a stratified sample of n=200 messages naming exactly one agent, spread
   across the primary window.
2. Label each as *addresses* vs *mentions*. Report precision with a CI; adopt
   the heuristic only above a pre-registered bar (0.8 proposed), and record the
   bar before looking.
3. **Controls, both directions.** A positive control per pattern class, and a
   negative control of 100 messages naming zero agents confirming none are
   addressed. A vocabulary scan that reports a clean negative without controls
   has been wrong in this repo before.
4. Check whether `@handle` is the cleaner signal, and how far the two overlap.

Two mechanical traps, both already hit while measuring the numbers above:
display names are prefixes of each other (`Claude Opus 4` vs `Claude Opus 4.5`),
so match **longest-first**; and `\b` never matches before `@`, so a
word-boundary-anchored `@handle` pattern is silently dead.

## D5 — Resolved: an outcome variable exists, but it is a gate decision

The four decisions above define observables. They do not define `v`, and
`fcmy.7`'s posterior fits proxy parameters against *observed outcomes*. Three
candidates were measured.

| candidate | n | verdict |
|---|---|---|
| `OUTREACH_APPROVAL_RESPONSE.approval` | 342 requests, 259 joinable | **usable** — external binary decision |
| `summaries` (`goal` 83, `daily` 805) | 939 | rejected — LLM-written but blind to session internals by construction, and coarse (multi-day, multi-agent) |
| consolidation-goal repetition | 18,832 pairs | rejected — only 8.6% repeat at string level, the real pattern is *continuation* ("Continue investigating…"), and the goal text is itself laced with self-reported progress |

The third deserves a note, because it was the one that would have scaled. It
fails for the D3 reason wearing a disguise: `nextSessionGoal` contains claims
like "I have successfully registered…", so an outcome derived from it is a
self-report again.

### The outcome that survives

`OUTREACH_APPROVAL_RESPONSE` carries a boolean `approval`, a `rationale`, and
an `adminComment`. It is a genuine external verdict on a specific agent action,
and the `adminComment` pattern shows a human behind it — present on 81 of 87
rejections but only 52 of 255 approvals, which is what a reviewer who explains
refusals and waves through approvals looks like. (Counts are per distinct
request id: there are 343 response *events* but 342 distinct requests, so one
request was answered twice.)

Measured feasibility of joining verdicts to session observables:

| | |
|---|---|
| verdicts | 342, over 2026-04-16 → 2026-09-18, 21 agents |
| base rate | 74.6% approved / 25.4% rejected |
| join rate into a session with turns | **259 / 342 (75.7%)** |
| joined class balance | 197 approved / 62 rejected |
| median turns in joined session | 41 — matches the ~40-action consolidation cadence |
| clustering | 20 agents; largest single agent is 35% of rows |

**This reframes what the Village is good for.** It does not supply interaction
outcomes, so it cannot calibrate the proxy the way the bead first assumed. What
it supplies is *real accept/reject decisions by a human reviewer* — which is
exactly what SWARM's selection metrics consume. `quality_gap` is defined as
`E[p | accepted] − E[p | rejected]`, and every quantity in it is now available
from real data: `p` from the session's observables, the decision from the
reviewer. The same holds for toxicity and conditional loss. This is a
**governance-gate study**, not a proxy calibration, and it is a better fit for
the metric suite than the original framing was.

### The assumption it rests on, and how it fails

The verdict judges a proposed *outreach message*; the observables describe the
*session* the request was made from. The design therefore assumes session
quality carries information about message approval. That assumption is testable
and may simply be false — a careless session can produce a clean message. If the
proxy has no predictive power over the verdicts, that is a null result, and it
is worth reporting as one rather than hunted around.

Power is adequate but not comfortable: 62 minority-class events over three
parameters clears the usual ten-events-per-parameter rule, but the 20 agent
clusters with one agent at 35% mean the effective sample is well below 259. Fit
with cluster-robust or hierarchical treatment of agent, and report effective
alongside nominal n — the same discipline the observation-unit note demands, for
the same reason.

### What this settles for `vu70` and `fcmy.7`

- `vu70` delivers a real but **narrow** calibration on ~259 externally-labelled
  rows, plus descriptive validation of the observable distributions across the
  much larger unlabelled corpus.
- `fcmy.7` gets a genuine test on a generator nobody here authored — but over
  three channels, on 259 rows, against a gate decision rather than an
  interaction outcome. That is a real answer and a partial one. It does not
  retire MiroShark as the second non-authored generator; it complements it.
- The 83 rejected verdicts that fail to join are not noise — check whether
  non-joining correlates with the verdict before treating the 259 as a random
  subset.

## Standing constraints

Licence is research-use-only: no training or fine-tuning of AI systems without
written permission, no re-identification, and citation of AI Digest / AI Village
in any resulting work. Everything above is parameter fitting and metric
computation, which is inside those terms. Any blog post, paper or docs page
built on this data carries the citation.

Upstream redaction to expect in text: `[REDACTED]` for credentials and
infrastructure, `[BLOB_REMOVED]` for thinking signatures, `[IMAGE_REMOVED]` for
embedded screenshots.
