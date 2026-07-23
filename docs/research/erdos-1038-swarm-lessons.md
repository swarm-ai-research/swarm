# Swarm orchestration lessons from the Erdős-1038 proof swarm

**Source:** [ShouqiaoW/erdos](https://github.com/ShouqiaoW/erdos) (studied 2026-07-22).
A multi-agent AI system produced a proposed complete solution to Erdős problem
1038 (extremal sublevel-set measure of monic polynomials with roots in [-1,1]).
The repo contains the orchestration prompt (`1038/prompt.md`), the prose proof
(`1038/paper.tex`), and a 2,193-file Lean 4 formalization with a hardened
axiom-audit gate (`FinalAxiomAudit.lean`). It is the cleanest public example we
have found of an end-to-end pipeline: orchestration prompt → swarm search →
machine-checked acceptance.

## The core design principle

**Search aggressiveness must be a function of acceptance mechanicalness.**
The erdos prompt tells its swarm to assume the problem is solvable and to never
give up — normally a recipe for confident hallucination. It gets away with this
because nothing the agents believe matters until it passes the Lean kernel plus
an exact axiom allow-list. Unbounded optimism in search, zero trust in
acceptance; each enables the other. When our acceptance criterion is an agent
review (soft), generation prompts must stay conservative and claims graded
(Auditor scale). When acceptance is mechanical (pytest, prereg lock comparison,
deterministic benchmark), persistence can be cranked up and the tail of the
distribution mined.

## Lessons and their translation into this rig

| # | Erdős-swarm mechanism | Translation here |
|---|---|---|
| 1 | 17-item "insufficient approaches" list: a *negative spec* enumerating every known way an agent shortcuts this problem class (special cases, symmetry assumptions, local-for-global, numerics-as-proof, circular reduction) | Assignments and verification targets carry an explicit insufficient-outcomes list; the Auditor verifies against the enumerated cheats for the claim class, not generic rigor (`/bv-dispatch` delivery format; Auditor § Negative-spec verification) |
| 2 | Blind diversity early: agents are not told the currently favored approach; cross-pollination deferred until routes are developed | When multiple agents attack one research question, dispatch mail omits the favored route during exploration rounds (`/bv-dispatch` § assignment) |
| 3 | Approach-family registry: agents grouped by underlying mathematical idea, not surface wording; rebalanced away from overcrowded families; elegance/numerics explicitly not grounds for dominance | Dispatch analysis clusters same-question beads by approach family and flags convergence (`/bv-dispatch` analysis step 6) |
| 4 | Blocked-route bookkeeping: stalled routes marked blocked with the missing lemma named; reopened only on a "materially new mechanism" | Blocked research beads record their blocker; reopening requires naming the new mechanism in the bead, which retro audits (`/bv-dispatch` analysis step 6 + retro) |
| 5 | Artifact-only returns: "concrete lemmas, constructions, certified bounds, or counterexamples — reject status reports, vague optimism" | Enforced at the substrate: the `done_requires_artifact` trigger on `agent_messages` rejects DONE rows lacking a commit hash, `runs/` path, or `artifact=` tag at insert time (CLAUDE.md § Inter-Session Coordination); `gate=<check>:<result>` convention checked by `/bv-dispatch` retro |
| 6 | Verification-shaped ≠ verified: the repo's green CI was a regex source scan; the real kernel-build job existed but was **never dispatched**. All surface signals read "verified"; the load-bearing step never ran | MUD LEDGER item 5: gate execution audit — for claims closed as verified, confirm the actual gate ran, not its cheap proxy (`/bv-dispatch` retro) |

Lesson 6 is the cautionary one and compounds our existing rug audit: a rug is a
cordoned mock result cited as real; a phantom gate is worse — a *verification
step* everyone believes ran and never did. Both fail silently; only the ledger
catches them.

## Enforcement status (skill-lint ladder)

- **Substrate-enforced** (fails mechanically at write time): lesson 5 —
  `done_requires_artifact` trigger, applied to the live `runs/runs.db`
  2026-07-22 and part of the canonical schema in CLAUDE.md.
- **Retro-checked** (mechanical query inside a prose loop): lesson 6 (gate
  audit greps `gate=` in DONE rows + bead evidence), lesson 4 (reopen
  justification audit).
- **Prose-only** (model-followed, no check yet): lessons 1–3 (negative-spec
  authoring, blind rounds, family clustering). Candidate next rung: a
  dispatch-side check that every full-dispatch ASSIGN comment contains a
  `insufficient:` block before mail goes out.

## Prior related findings in this rig

- Rug audit and "machinery validation only" cordons — MUD LEDGER items 1–4
  (dispatch retro 2026-07-19).
- ~30% of fan-out review findings wrong; verify against source before applying
  (multi-session periods, CLAUDE.md § Test fix discipline).
- Side-effect reads beat voluntary protocols (coordination-channel finding,
  2026-07-19) — same spirit: make the right thing structural, not aspirational.
