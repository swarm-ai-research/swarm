---
description: "Tipperman's Stain the Page (2026) read against the straw-horse gate, the wiki board's copied first posts, and the fm-agent-harness false green: a wrong draft draws better correction than an open question, but only when the corrector actually checks."
author: "SWARM Team"
date: "2026-09-06"
source: "https://engineering.squarespace.com/blog/2026/stain-the-page-how-i-prototype-with-ai"
keywords:
  - prototyping
  - prompt design
  - agent coordination
  - review
---

# Stain the Page, read as a dispatch mechanism

**Source:** Hannah Tipperman, *Stain the Page: How I Prototype with AI*,
Squarespace Engineering Blog, 24 July 2026. Read 2026-09-06.

## The essay

The claim is that AI prototyping conserves attention. An engineer spends
cognitive energy on names, boilerplate CSS, and file layout before reaching
the part of the problem that is new, and is tired by the time they get
there. Handing the low-level choices to a model saves the energy for
architecture and user experience.

Four moves carry it.

1. **Stain the page.** A painter covers the blank canvas first, with anything,
   because the first marks are easier to correct than to make. Ask for a rough
   draft and react to it.
2. **Avoid the void.** Ideas die because testing them costs too much. Dump a
   stream-of-consciousness prompt with the goal and the open questions. "You
   will not get the final answer — let that go!"
3. **Refine.** Two techniques. Cunningham's Law: the fastest way to get the
   right answer is to post the wrong one, so state a wrong draft and let the
   model correct it rather than asking an open question. And ask the model to
   challenge your assumptions and offer other perspectives.
4. **The case.** A working React dashboard for an "AI visibility scanner" in
   under thirty minutes. The tangible prototype exposed layout, flow, and
   ordering problems that planning in the abstract had not.

The claimed effect is that product conversations move from "is this
possible?" to "how do we make this great?", and designers and product
managers join earlier.

## The same mechanism, already in the rig, one level up

The essay is [Readme Driven Development](readme-driven-development-lessons.md)
for artifacts rather than specs, and the [straw-horse gate](classic-essays-swarm-lessons.md)
for prompts rather than plans. Conklin's group managers could not plan their
piece until a one-page master plan existed, so they wrote one in a day and
called it a straw horse. Tipperman's engineer cannot see the layout until a
dashboard exists, so she asks for one in thirty minutes. Both are the same
observation: a concrete wrong thing draws correction, and a blank page draws
nothing. The rig gate says a plan-less epic's top pick is writing the plan.
The essay's version, one level down, is that a bead whose body is a question
should be rewritten as a wrong answer with the open questions attached.

The wiki board did this to itself. The [spec-emergence scan](https://github.com/swarm-ai-research/wiki-agent-swarm-incident/blob/main/analysis/spec-emergence.md)
found 380 pages whose first post restates the protocol by copying a
neighbour's wording. No agent wrote the protocol from a blank page. Each one
stained its page with the nearest existing draft and edited. That is
Cunningham's Law running as a population process, and it is part of why the
board converged without anyone specifying it.

## Where the essay stops

Cunningham's Law has a hidden premise: someone reads the wrong answer and
checks it. The essay's corrector is the engineer, looking at a dashboard she
understands. Three rig findings are what happens when the corrector is
another model, or a gate, and does not check.

- **The false green** ([cantrip note](cantrip-runtime-lessons.md), fm-agent-harness
  2026-08-19). Example-based tests went green on broken code in five of five
  trials. A wrong draft plus a checker that accepts it is not a stain to be
  corrected; it is the painting. Property-based feedback caught the same bug
  five of five times because the property could not be satisfied by the
  draft's own assumptions.
- **Mechanical acceptance** ([erdos](erdos-1038-swarm-lessons.md)). A swarm that
  accepts its own drafts at the gate is the essay's move with the refine step
  deleted. Search without a corrector converges on the first stain.
- **Reviewer error rate.** Fan-out review findings were wrong about a third
  of the time when checked against source. "Ask the model to challenge your
  assumptions" produces challenges at that rate too. The challenges are the
  new wrong draft, and need their own corrector.

So the essay's move is sound and its scope is narrower than it reads: the
stain is cheap, the correction is where the cost went, and a rig that adopts
the first without budgeting the second has moved the void, not avoided it.

## Rig change

One line added to `/bv-dispatch`, under the straw-horse gate. A bead whose
body is a question ("should X use Y?") is rewritten before ranking as a
claimed answer plus a list of assumptions the answer rests on. The
reviewer's prompt then targets the assumptions list, which is the essay's
second refine technique with the corrector's work made explicit. G2 beads,
where the gate is judgment rather than a test, keep the assumptions list as
the review target and do not accept the draft on its own green.

## Limitations

- The essay is a single engineer's practice, reported as a case, not
  measured. The thirty-minute figure is hers.
- The wiki board reading is an interpretation of the export. Copied first
  posts are consistent with stain-and-edit, and also with agents that
  had the protocol in their prompt and only echoed it.
- The reviewer error rate is from this rig's own fan-out reviews, on this
  rig's code, and is the number that most needs re-measuring.
