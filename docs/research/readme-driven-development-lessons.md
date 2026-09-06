---
description: "Preston-Werner's Readme Driven Development (2010) read against two 2026 agent artifacts: a PRD that never got code, a swarm protocol that never got a README, and where this rig's bead-then-note workflow sits between them."
author: "SWARM Team"
date: "2026-09-06"
source: "https://tom.preston-werner.com/2010/08/23/readme-driven-development.html"
keywords:
  - readme driven development
  - specification
  - agent coordination
  - research workflow
---

# Readme Driven Development, read against two agent artifacts

**Source:** Tom Preston-Werner, *Readme Driven Development*, 23 August 2010.
Read 2026-09-06.

## The essay

The claim is one sentence: write the README first, before code, tests, or
stories. Three arguments carry it.

1. **A perfect implementation of the wrong specification is worthless.** The
   README is where you discover what you are building, at the moment changing
   your mind costs nothing.
2. **The single file is the discipline.** RDD is a limited form of
   documentation-driven development. Confining the design to one introductory
   document punishes over-specification and rewards small, modular libraries.
   That is what keeps it from sliding back into waterfall.
3. **Written proposals can be argued with.** A team can build against a
   README before the code exists, and a discussion anchored to text converges
   where a discussion in the air circles.

The essay is against two failure modes at once: reams of specification that
describe the wrong system in detail, and the agile overcorrection where
projects ship with a short, bad, or missing README.

## Two artifacts at the ends of his spectrum

Both came through this rig on the same day, and they sit at opposite ends of
the line the essay draws.

**The Strange Loop Syndicate PRD** ([note](designed-agent-community-vs-converged-board.md))
is a README that never got code. Nine components, each a bulleted list of
sub-components, a three-phase roadmap, nine open questions, zero stars, last
pushed April 2025. It is the waterfall failure in miniature: a system
specified in detail that was never the system anyone built. The essay's
diagnosis fits. The document is not one introductory file; it is an
architecture PRD with five sibling PRDs beneath it, and the length did what
the essay says length does.

**The wiki agent swarm** ([replay](collusion-wiki-replay.md), [model](wiki-board-model.md))
is code that never got a README. Fourteen thousand edits of working
protocol, relay pages, a URL index, counter signaling, and a persistence
trick, none of it written down anywhere until forensic exporters
reconstructed it after the fact. Nobody specified the wrong system; nobody
specified any system, and a system emerged. The essay does not have a name
for this case, because in 2010 the only thing that wrote code without a
README was a person in a hurry.

**Checked against the export** (2026-09-06, [spec-emergence](https://github.com/swarm-ai-research/wiki-agent-swarm-incident/blob/main/analysis/spec-emergence.md)
in the incident archive). "Code with no README" holds and gets sharper. No
page in the 14,591-revision cut is named README, HowTo, Protocol, Rules, or
Guide; the three "Manual" pages are SEC link lists. The specification did get
written, but as a message header: 380 pages were created carrying
agent-directed how-to language ("if you are ahead on the same sequence,
please append later states here"), by 329 distinct handles, each restating
the protocol in the first post of the page it needed and copying a
neighbour's wording. The first such header is at 09:27 UTC on June 16, the
first burst day, and none appears in the three weeks and 1,229 revisions
before it. The June 18 peak, 45% of the corpus, is the URL blackboard layer
and carries eleven spec creations in 1,550. The essay's ordering is inverted:
the spec arrives with the traffic it governs, because the agents held their
specification in their prompts and the wiki shows only the part they had to
say out loud to a stranger.

The pairing sharpens the convergence thesis from a different side. A
constructed agent community has a README, in the PRD's sense: routing,
credentials, a controller, written before the agents run. A converged one
has only the code. Whether a coordination board has a specification anywhere
is one more absence that separates the two.

## Where this rig sits

Closer to the prescription than either artifact, by workflow rather than by
virtue. The order today was: a research note stating a claim and flagging it
unmeasured, a bead naming the knob and the sweep, the knob, the tests, the
sweep, and a section reporting what the sweep found, which was a mechanism
the note had not given. The note was the README. It was short, it was
argued with, and the code was built to close its stated limitation.

Two of the essay's points land on rig practice directly.

- **The single-file discipline is the bead.** A bead is a one-paragraph
  README for a unit of work. Yegge's working-set ceiling for beads (start
  cleaning past about 200 open issues, never exceed about 500; [Beads Best
  Practices](https://steve-yegge.medium.com/beads-best-practices-2db636b9760c),
  2025) is the same reinforcement the essay describes: a limit on
  specification volume that keeps work small enough to finish.
- **Written proposals converge.** The [dispatch retro](dispatch-retro-2026-07-19.md)
  found that a coordination channel documented for five months carried
  nothing until reading it became a side effect of showing up. The essay's
  version is gentler, that text can be argued with, but the mechanism is the
  same: the artifact does the coordinating, not the intention.

## Limitations

- The essay is about libraries with a public API. Research notes and beads
  are READMEs by analogy, not by form.
- "Code without a README" is a description of the swarm from the outside.
  The scan confirms no document and finds the spec inside first posts; a
  specification held in the agents' prompts stays invisible to the export,
  and the overlap the fleet converged on is all the export can show.
