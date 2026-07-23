# Swarm orchestration lessons from the Erdős-1038 proof swarm

**Source:** [ShouqiaoW/erdos](https://github.com/ShouqiaoW/erdos) (studied 2026-07-22).
A multi-agent AI system produced a proposed complete solution to Erdős problem
1038 (extremal sublevel-set measure of monic polynomials with roots in [-1,1]).
The repo contains the orchestration prompt (`1038/prompt.md`), the prose proof
(`1038/paper.tex`), and a 2,193-file Lean 4 formalization with a hardened
axiom-audit gate (`FinalAxiomAudit.lean`). It is the cleanest public example we
have found of an end-to-end pipeline: orchestration prompt → swarm search →
machine-checked acceptance.

**Official status check (2026-07-22):** the community ledger
([teorth/erdosproblems wiki, "AI contributions to Erdős problems"](https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems),
data cutoff 2026-06-30) lists 1038 as 🟡 **partial**, tier 1(d) — a
months-long human-AI collaboration (7 humans incl. Tao; 6 AI systems). The
ShouqiaoW repo and its Lean proof appear nowhere on the ledger. An unreviewed
mechanical proof whose gate never publicly ran is ⚪ claiming to be 🟢 — the
acceptance layer holding firm is lesson 6 operating at ecosystem scale.

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
  authoring, blind rounds, family clustering) and ecosystem lesson 7
  (gate-cost ranking — encoded as `/bv-dispatch` analysis step 7 with
  G0/G1/G2 classes modulating search aggression, but the classification
  itself is model judgment). Candidate next rungs: a dispatch-side check
  that every full-dispatch ASSIGN comment contains an `insufficient:`
  block before mail goes out, and a retro check that assignment tables
  carried a gate class for every research bead.

## Ecosystem lessons (the wider AI-on-Erdős corpus)

Sources: the [teorth/erdosproblems AI-contributions wiki](https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems)
(~527 entries as of 2026-06-30) and
[przchojecki/counterexamples](https://github.com/przchojecki/counterexamples)
(LLM-assisted counterexample archive). Read 2026-07-22. These generalize the
single-repo lessons above to the whole ecosystem; rig translations noted where
one exists.

1. **Provenance is load-bearing, not capability.** The ledger's 1(a)–1(d)
   taxonomy exists because "AI solved N open problems" headlines (late 2025)
   collapsed when the problems turned out to be database-open only — solved in
   literature the model had effectively retrieved. "Novel" is a property of
   the literature search, not the output. *Rig:* claims need a provenance
   field before a capability conclusion; Auditor grades vs prior art, not vs
   zero.
2. **The productive mode is colleague, not oracle.** Dominant tier is 1(d)
   human-AI collaboration (~140 entries, months-long episodes); standalone
   1(a) is a minority (~48). *Rig:* optimize the collaboration loop
   (dispatch → human review → redirect), not autonomous solve rate.
3. **Cross-model agreement is weak evidence.** Red entries 616 and 888:
   Claude, Gemini, and GPT *independently* produced flawed proofs of the same
   claims — overlapping training corpora share blind spots, so convergent
   plausibility ≈ correlated error. *Rig:* verification panels need
   perspective diversity (distinct lenses/failure modes), not redundant votes;
   prefer non-model verifiers wherever one exists.
4. **Verification is the bottleneck; formalization is the escape valve — with
   a trap.** Largest single category is 2(b) formalization (~180 entries):
   converting scarce expert-hours into free kernel-checks. The trap is
   verification theater (the 1038 phantom gate): the chain of trust must be
   end-to-end *executed*, not merely present and impressive.
5. **Acceptance is institutional, not just mechanical.** A kernel checks a
   proof; only the curated ledger confers "solved." It also logs failures with
   named systems (~7% 🔴) — the red rows carry more information than the green
   rate, because each failure mode becomes tomorrow's negative spec. *Rig:*
   the retro/MUD LEDGER is our acceptance layer; keep logging reds with
   attribution.
6. **The denominators are missing everywhere.** Every ledger entry is a
   survivor; failed attempts are never logged, so success rates computed from
   it are meaningless ("this page is not a benchmark"). The counterexamples
   archive shows the alternative: cleared search space and explicit
   still-open lists as first-class deliverables. *Rig:* sweeps report what
   was ruled out and what was not attempted (no-silent-caps), not just hits.

7. **Verification asymmetry decides where aggressive search pays.**
   ([Buzzard, "Human mathematicians are being outcounterexampled"](https://xenaproject.wordpress.com/2026/07/20/human-mathematicians-are-being-outcounterexampled/),
   2026-07-20.) Counterexamples fell first — unit distance (May 2026),
   Grothendieck group schemes and reportedly the Jacobian conjecture
   (July 2026) — because a counterexample has the best verification economics
   of any mathematical artifact: checking one concrete object against a
   definition is near-decidable (Grothendieck: a 1,076-line Lean file),
   while a general extremal proof needs the whole apparatus (1038: 2,193
   files). AI search wins exactly where verification is cheap relative to
   discovery. Buzzard's own acceptance policy is the pattern from § core
   design: "I am not reading AI-generated informal mathematics" — only
   compiled Lean, compiled by him. *Rig:* dispatch should rank exploratory
   beads by gate cost — point the most aggressive search (most agents,
   highest persistence, loosest priors) at claims with the cheapest
   mechanical gate (a failing test, a concrete object, a replayable run),
   and keep expensive-gate claims on conservative, human-reviewed tracks.
   (Incidents postdate our model knowledge cutoff; relayed on Buzzard's
   authority, gates not independently re-run here.)

Compressed: AI-on-math became real when the community built provenance
tracking, stratified verification, and institutional acceptance around it —
and every failure mode in the ledger is a variant of skipping one of the
three. The rig's analogues: negative specs + bead provenance, gate-execution
audits, retro-as-ledger. Lesson 7 adds the allocation rule the first six
imply: search aggression should scale inversely with gate cost.

## Prior related findings in this rig

- Rug audit and "machinery validation only" cordons — MUD LEDGER items 1–4
  (dispatch retro 2026-07-19).
- ~30% of fan-out review findings wrong; verify against source before applying
  (multi-session periods, CLAUDE.md § Test fix discipline).
- Side-effect reads beat voluntary protocols (coordination-channel finding,
  2026-07-19) — same spirit: make the right thing structural, not aspirational.
