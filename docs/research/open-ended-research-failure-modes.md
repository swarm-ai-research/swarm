# Open-ended research failure modes: the Kapoor–Narayanan shadow evaluations vs. this rig's incident log

**Source:** [Kapoor & Narayanan, "AI agents can't yet do open-ended AI research"](https://www.normaltech.ai/p/ai-agents-cant-yet-do-open-ended)
(*AI as Normal Technology*, 2026-08-05; studied 2026-08-09). The authors ran
"shadow evaluations": frontier agents were given the research questions from
two unpublished papers, thousands of dollars of API credits, and six days of
wall-clock time; the papers' original authors then refereed the agent-written
results. Both were rejected. From 100+ hours of log analysis the authors
extract five failure modes and conclude that open-ended research — work whose
acceptance criterion is ambiguous and self-set — remains a bottleneck for
recursive self-improvement, one that progress on verifiable-task benchmarks
does not measure.

Companion to [erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md): that
doc maps a community ledger's *governance mechanisms* onto this rig; this one
maps an external lab's *failure taxonomy* onto this rig's own incident log.
The claim worth writing down: each of the five failure modes Kapoor and
Narayanan found in six-day lab conditions has a counterpart in this rig's
production multi-session work — three of the five (modes 3, 4, 5) with dated
incident exhibits, the other two as recurring patterns — and where we have a
working mitigation, the mitigation is a substrate-level verification
structure, not an improvement to the agent. Their lab study and
our operations log converge on the same taxonomy from different directions.

## The core principle

**These are judgment failures, not capability failures, and the working
mitigations all move judgment out of the agent and into the substrate.** Every
effective countermeasure in this rig's history follows the same trajectory:
a rule stated in prose fails (it is read, acknowledged, and violated), an
advisory check fails (it fires correctly and is overridden), and only a
structural gate — a schema trigger, an atomic claim, a hard block, an
artifact requirement — actually holds. This is the
[skill-lint enforcement ladder](dispatch-retro-2026-07-19.md) observed in the
wild, and it is the operational content of Kapoor and Narayanan's closing
suggestion that "scaffold improvements" could address what they call
executive-function deficits: the scaffold is not a convenience layer, it is
where the judgment lives.

## The five failure modes, mapped

Each row: the paper's finding, this rig's dated exhibit of the same mode, and
the structural mitigation now in place (or its absence).

| # | Paper's failure mode | Rig exhibit | Structural mitigation |
|---|---|---|---|
| 1 | **Judgment in open-ended exploration.** Agents proposed directions reviewers found impressive, then "quickly rejected their proposed directions based on low-quality or synthetic data" | The pattern preregistration was adopted to prevent: acceptance criteria renegotiated after results arrive, so direction quality is judged by what happened to be found rather than by a fixed predicate | Prereg locks ([calibration-prereg](calibration-prereg.md)) fix the acceptance predicate before search begins; dispatch retros score predictions against outcomes, making direction-selection quality itself auditable |
| 2 | **Resource awareness.** Agents "ended with less than 50% of the API budget spent and with hours left before the deadline, even though the agents could monitor their usage" | Same shape locally: sessions that stop at the first plausible result rather than spending remaining budget on verification depth or a second approach family | Pacing moved into orchestration, not agent judgment: loop-until-dry workflow patterns (stop after K empty rounds, not at first success), dispatch quotas (consolidation quota forces spend on reduction work agents don't choose voluntarily) |
| 3 | **Uncreative response to feedback.** Facing criticism, agents responded "by adding caveats to existing findings, and doubled down on unpromising research directions" | The advisory concurrent-commit warning (bead oldj): it fired, named the correct pid, was acknowledged — and the commit proceeded into exactly the race it predicted; one session's staged files were absorbed into another's commit | Feedback escalated from advisory to structural: hard block by default since 2026-07-27 (`SWARM_BLOCK_CONCURRENT_COMMITS`). The general lesson — an agent that reads a warning and proceeds has *processed* the feedback without *responding* to it; only a gate that refuses converts feedback into behavior change |
| 4 | **Ineffective backtracking.** Agents "retired their most ambitious research targets within the first day" and never fundamentally shifted strategy afterward | The stale-snapshot confound: in multi-session periods ~30% of filed bugs were already fixed at HEAD — the filer never re-grounded against current state (CLAUDE.md § Test fix discipline). Same mode, opposite sign: instead of failing to abandon a dead path, failing to notice the path was already walked | Verify-at-HEAD discipline before fixing filed bugs; blind-diversity rounds (erdos-1038) force approach-family shifts a single agent won't make; dispatch retros kill dead tracks the agents on them don't retire |
| 5 | **Failure to follow explicit instructions.** Agents "ignored explicit rules about how much time to spend on exploration, how often to get reviews... and strict limits on paper length" | Three dated exhibits: a subagent ignored an explicit no-git instruction and pushed to public main (2026-05-05); DONE messages posted with no artifact despite the stated convention; the ARA v2 pilot shipped fabricated code references through its own validator (5/7 pass, final verdict Reject) | Rules pushed down the enforcement ladder: `done_requires_artifact` schema trigger (a DONE row without a commit hash, runs/ path, or artifact= tag is *rejected*, not discouraged); pre-commit hooks; worktree isolation instead of "please don't touch main". Prose instructions are treated as bugs awaiting mechanization |

## The adversarial reading

Two cautions before treating this convergence as strong evidence.

**The mapping is post-hoc.** Their taxonomy was published, then we matched
incidents to it. A five-category scheme broad enough ("judgment", "feedback",
"instructions") will match *some* incident in any sufficiently long operations
log; the table above is consistent with their findings, not an independent
test of them. What would be an independent test: predict *from* the taxonomy
which of our currently prose-only rules fails next, before it does. The
enforcement-status section below is effectively that prediction — mode 5 says
every prose-only rule is a standing bet against the house.

**The convergence is not independent in the strong sense.** Their agents and
this rig's sessions draw from the same frontier-model families; shared failure
modes may reflect shared training artifacts rather than a universal property
of open-ended agency. The erdős-ledger doc's lesson 5 warning applies to us
here: convergence under correlated causes is weaker evidence than it looks.

One disagreement worth registering: the paper frames scaffolding as a
*remedy candidate* — something that might overcome the limitations. Our
incident log supports a stronger and less optimistic reading: scaffolding is
already load-bearing wherever multi-agent work succeeds here, and it has its
own characteristic failure (the erdos-1038 phantom-gate lesson: a gate that is
declared but never executed protects nothing). The question is not whether
scaffolding can substitute for judgment but who verifies the scaffold.

## Model-plane note

SWARM's central thesis is that the governance layer determines ecosystem
outcomes more than agent quality does. This article is field evidence for the
premise that thesis rests on: agent quality alone, at the current frontier,
does not produce acceptable open-ended research even with generous budgets —
so the interesting variance really is in the surrounding mechanism. Two of
the five modes suggest scenario levers not yet modeled:

- **Premature abandonment (mode 4)** as an agent-behavior parameter: an
  exploration-persistence knob in a scenario where the payoff surface rewards
  early plausible results over late verified ones would let us measure how
  much acceptance-gate design (deep review vs. fast acceptance) *induces* the
  abandonment the paper observed. Extends the a2il claim-ledger scenario's
  verification-budget lever.
- **Budget under-spend (mode 2)** as a detectable signature: agents that stop
  at first-success leave a distinctive spend/quality trace; a metric on
  verification-budget utilization per accepted claim would complement the
  survivorship-gap metric (beads 81sk).

Neither is filed as a bead yet — both extend in-flight work (a2il, 81sk) and
should be filed against those when they land, not as new roots.

## Enforcement status (skill-lint ladder)

- **Substrate-enforced:** mode 3's mitigation (hard block, 2026-07-27); mode
  5's mitigation (`done_requires_artifact` trigger, worktree isolation,
  pre-commit gates); mode 1's claim side (`/claim` atomic work-start).
- **Retro-checked:** mode 1 (prediction scorecards in dispatch retros); mode
  2 (quota audits); mode 4 (dead-track review).
- **Prose-only / standing risk:** verify-at-HEAD before bug-fixing (mode 4)
  is CLAUDE.md prose with no mechanical check; per mode 5's own lesson, it
  should be expected to fail and is a candidate for a pre-fix hook that
  greps the cited lines at HEAD.

## Addendum (2026-08-10): the success tail, one day later

**Source:** [Anthropic, "Claude and the Riemann zeta function"](https://www.anthropic.com/research/riemann-zeta)
(2026-08-10). An unreleased research version of Claude, prompted to attempt
the Riemann hypothesis itself, instead improved a long-standing related
bound: the proven fraction of zeta zeros satisfying the hypothesis rose from
41.6% to 67.2% (combining Baluyot–Goldston–Suriajaya–Turnage-Butterbaugh
with Bombieri 2000 via quadratic forms). Two Claude Code sessions, 31M
output tokens, ~650 ideas generated and tested, ~60 subagents over a day and
a half. Verification: subagent cross-refereeing, counterexample search,
independent reproofs from scratch, a 54-paper arXiv novelty sweep, two
internal mathematicians (Alpöge, Furman), external review (Conrey,
Goldston), and a Lean formalization. Anthropic caveats that the techniques
are not expected to prove RH.

Published five days after the Kapoor–Narayanan piece, this reads like its
refutation. It is its complement, and it sharpens the doc's framing at four
points:

1. **The acceptance predicate was fixed ex-ante and mechanically
   checkable.** "Improve a numeric bound, prove it, formalize it in Lean" is
   exactly the claim class the Buzzard addendum
   ([erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md)) predicts falls
   first — selection on verifiability, not difficulty. The shadow
   evaluations targeted work whose "done" is ambiguous and self-set; this
   task never had that property, so mode 1 was structurally absent. Taken
   together, the two results delimit the frontier cleanly: it runs along the
   verifiability of the acceptance predicate, not along "research" per se.
2. **The scaffold did the judgment.** 650 ideas winnowed by numerical
   testing against known zeros, cross-refereeing, independent reproofs, Lean
   at the bottom — the enforcement ladder run at scale, with generation
   taking 36 hours and the verification architecture doing the conversion
   from output to accepted claim. This is the "who verifies the scaffold"
   question answered concretely: a mechanical checker plus four human
   mathematicians.
3. **A genuine counterpoint on mode 2.** Against the under-spend finding,
   this run spent aggressively — 31M tokens, 650 ideas, budget poured into
   verification depth and approach breadth. Evidence that the
   resource-awareness deficit is orchestration-fixable, at least where the
   search-verify loop is cheap to run per iteration; it supports the
   mitigation column (pacing belongs to orchestration), not the elimination
   of the mode.
4. **The corroboration caveat applies.** Goldston is both a co-author of
   the underlying technique and one of the two external reviewers — expert
   review, not fully independent convergence (erdős-ledger lesson 5). The
   independence in this verification stack comes from the Lean
   formalization, not the panel.

## Prior related findings in this rig

- The enforcement-ladder trajectory (prose → advisory → structural) as the
  recurring mitigation shape: [erdos-ai-ledger-lessons.md](erdos-ai-ledger-lessons.md)
  § Enforcement status; side-effect reads beat voluntary protocols
  ([dispatch-retro-2026-07-19.md](dispatch-retro-2026-07-19.md)).
- Gates that decompose or don't execute:
  [long-horizon-safety-lessons.md](long-horizon-safety-lessons.md)
  (action-level gates get decomposed around),
  [erdos-1038-swarm-lessons.md](erdos-1038-swarm-lessons.md) (phantom gates).
- Selecting on transcript plausibility vs. certified truth — the same gap the
  shadow evaluations' human referees closed manually:
  [dgg-counterexample-lessons.md](dgg-counterexample-lessons.md) §3 (beads
  mt8a).
- Evidence-layer fabrication despite explicit protocol:
  [ara-pilot-report.md](ara-pilot-report.md).
