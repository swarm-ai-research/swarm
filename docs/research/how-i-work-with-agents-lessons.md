---
description: "Thorsten Ball's October 2026 list of agent-working habits read against the rig: the proof and blind-eval tips match what we measured, the model-choice and slop tips don't survive a swarm, and the blind eval is the experiment to run next."
author: "SWARM Team"
date: "2026-10-25"
source: "https://x.com/thorstenball/status/2114386050638336082"
keywords:
  - agent workflow
  - evidence
  - code review
  - agentic knowledge
  - agent coordination
---

# How I Work with Agents, read against the rig

**Source:** Thorsten Ball, "How I work with Agents ... as of October 2026",
post on X (<https://x.com/thorstenball/status/2114386050638336082>). Date of
this note: 2026-10-25. Bead: unfiled.
**Companion notes:**
[Stain the Page Lessons](stain-the-page-lessons.md) (the false green, the
corrector who doesn't check),
[Cantrip Runtime Lessons](cantrip-runtime-lessons.md) (the `nudge` ladder),
[erdos-1038 Swarm Lessons](erdos-1038-swarm-lessons.md) (phantom gates),
[Dispatch Retro 2026-07-19](dispatch-retro-2026-07-19.md) (side-effect writes),
and Pace Layers and Swarm Lag (layer tags; PR
[#646](https://github.com/swarm-ai-research/swarm/pull/646), not yet merged).

> **Sourcing status.** The post could not be fetched (x.com returned HTTP 402).
> Quotes tagged `[pasted]` are from the text the operator pasted on
> 2026-10-25, not checked against the live post. Rig numbers are quoted from
> the files or repos named next to them. Mappings tagged `[inferred]` are this
> note's argument, not the post's. No experiment was run for this note. §4
> proposes one.

## The claim in one paragraph

The post is fourteen habits from one engineer who works mainly in Amp. The
load-bearing ones: use the newest model, because "If a model is wrong 10% of
the time, you don't know which 10%" `[pasted]`; give the agent tool access to
the whole SDLC, since "If the agent can't see something, it makes it up"
`[pasted]`; test whether agents actually use the knowledge in AGENTS.md and
skills with "five blind agents, control vs. treatment" `[pasted]`; ask for
"irrefutable proof", the "Biggest game changer" `[pasted]`; stop worrying about
slop; and review only code with high blast radius, about five central files
plus migrations. The rest (heartbeats, async batching, one thread per task,
threads as company memory) is about the operator's time rather than the
agent's correctness.

The post is one operator on one thread at a time. The rig is many agents on a
shared graph. Most of the disagreements below come from that difference, not
from either side being wrong about its own setting.

## 1. Where the rig's numbers agree

**Proof, but the gate is what matters.** fm-agent-harness
(swarm-ai-research/fm-agent-harness, private) held model, task and loop fixed
and varied only the checker feedback. Ladder run, 10 seeds per arm: `nudge`
10/10, `props-blind` 10/10, `props` 10/10, `tests` 0/10, `nl` 3/10. Example
tests opened the gate on broken code 10 times out of 10. `nudge` (a bare
"wrong, revise") tied the arm that was handed the falsifying input. So the
value of proof was in refusing to let the agent stop, not in the content of the
proof `[inferred]`. That sharpens the post's advice: an end-to-end walkthrough
or demo video is only proof if something the agent doesn't control can make it
fail. The erdos-1038 case is the counterexample in the field. Green CI was a
regex source audit, and the kernel build never ran
([erdos-1038-swarm-lessons.md](erdos-1038-swarm-lessons.md)).

**Blind evals of agentic knowledge.** The post recommends testing whether
agents use AGENTS.md, skills and docs. The 2026-07-19 retro already has the
answer for one piece of rig knowledge, observed rather than tested: a
coordination channel documented for five months had zero messages. A bead
went unclaimed through four rounds of comment stamps, then was claimed 18
minutes after the channel entered the session-start read path
([dispatch-retro-2026-07-19.md](dispatch-retro-2026-07-19.md)). "Side-effect
writes get used; voluntary protocols don't" `[read, retro]` is what a blind
eval would likely find for most prose in AGENTS.md `[inferred]`. §4 proposes
running it properly.

**Theories in parallel.** "Give me 4 theories on why this is slow, then check
all four" `[pasted]` is the Stain the Page move applied to diagnosis: a wrong
draft draws correction, an open question draws nothing
([stain-the-page-lessons.md](stain-the-page-lessons.md)). The same caveat
applies. Someone has to check all four.

**Local dev is crowded.** "Hard to not run into resource conflicts locally"
`[pasted]` matches the rig's shared-checkout hazard: concurrent sessions share
one git index, and hook races follow. The rig's answer is one worktree per
branch, which this note was itself written in.

## 2. Where the rig's numbers disagree

**"Use the latest model" doesn't fix not knowing which 10%.** Three review
passes over the same 12 files (two effort levels of one model, one pass of
another) gave 11 verified-true findings with no line reported twice. The
higher-effort pass found none of what the lower-effort pass found. Nine of the
twelve files came back empty from a single pass and were not clean. Separately,
about 30% of fan-out review findings in the July pg-cleanup campaign were wrong
when checked against source. If a pass samples a small slice of what is
findable, a better model samples a different slice. Repeat passes and take the
union, then verify each finding. The post's own premise (you don't know which
10%) argues for that more than for an upgrade `[inferred]`.

**"Stop worrying about slop" holds at one layer, not all of them.** The post's
examples are code-level: "Closure vs. class, a neater type, variable names,
even duplication" `[pasted]`. The rig has no evidence against that. Its mud
shows up elsewhere. One session produced 22 edgeless orphan beads, 68 GB of
Docker sediment, and a library default drifting against a prereg lock (bead
`naa8`). The first retro measured 88% of new beads edgeless at creation. None
of that is code style, and none of it fixed itself; each needed a quota or a
default. So the narrower claim the rig can accept is "stop worrying about
code-style slop in fast-layer code" `[inferred]`.

**"Did the agent test it end to end?" is not a review substitute by itself.**
The post skips review when the answer is yes. Under the ladder result, "yes"
from an example test is the false-green case. The question to ask instead is
whether the test could have failed on the code that was written.

## 3. The review list is the slow layer

The post reviews "<5 central files" and database migrations `[pasted]`. That
is close to the slow layer in the pace-layers note: prereg locks, frozen
schemas, CLAUDE.md / AGENTS.md, specs. Migrations are the post's version of a
frozen schema. The two notes reach the same partition from different ends
`[inferred]`: review where changing something needs the operator's agreement,
gate everywhere else. What the rig adds is the ratification-lag metric from
the pace-layers note. A fast-layer change to something a slow-layer file names
should get a slow-layer commit acknowledging it, and the lag is measurable.

## 4. Proposal: run the blind eval on our own AGENTS.md

None of this has been run.

**Question.** Does a "this is not evidence" section in AGENTS.md reduce
unsupported claims of success?

**Arms.** Five agents per arm, same model, same tasks, blind to the
experiment.

| Arm | AGENTS.md |
|---|---|
| Control | current file |
| Prose | current file plus a "this is not evidence" list: example-only tests, an agent saying it verified something, subagent summaries, CI that doesn't build |
| Read path | the same list injected at session start by hook, not left in the file |

The rig already enforces one item on that list in code rather than prose. The
`done_requires_artifact` trigger on `agent_messages` rejects a `DONE:` row with
no commit hash, `runs/` path or `artifact=` tag (CLAUDE.md, from erdos-1038
lesson 5). That is a fourth arm for free: the trigger's rejection count is the
false-claim rate where the rule is a gate, to compare with the rate where it
is only text.

**Measure.** For each completed task, whether the final report claims success,
and whether a held-out checker the agents never see agrees. The metric is the
false-claim rate: claimed success that the held-out checker rejects.

**Prediction.** The retro predicts Prose ≈ Control and Read path < Control. If
Prose already beats Control, the retro's lesson was about coordination
channels specifically and shouldn't be generalized to AGENTS.md prose. If
neither beats Control, the section is decoration, and the gate has to live in
CI rather than in text.

## 5. What not to take from the post

- **"Who's still using Opus 4.5? Exactly."** `[pasted]` is a cost argument
  stated as a quality one. The rig's evidence is about pass count, not model
  vintage, so this note takes no side on which model to run.
- **"Don't let agents write for humans"** is a style rule outside this rig's
  scope. Noted so leaving it out isn't read as disagreement.
- **Threads as company memory** depends on Amp threads being linked to
  commits. The rig's equivalents are beads, `agent_messages` and git history,
  and the retro shows those get read only when reading them is a side effect.
