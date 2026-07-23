# Ledger governance lessons from AI contributions to Erdős problems

**Source:** [teorth/erdosproblems wiki — "AI contributions to Erdős problems"](https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems)
(studied 2026-07-22). Terence Tao's curated ledger of AI work on Erdős
problems: outcomes classified 🟢 (full) / 🟡 (partial) / 🔴 (incorrect) /
⚪ (unverified), contributions split into primary (AI-generated mathematics)
and secondary (formalization, literature search, computation, rewrites), with
explicit disclaimers against reading the ledger as a benchmark.

Companion to [erdos-1038-swarm-lessons.md](erdos-1038-swarm-lessons.md):
that doc extracts *orchestration* lessons from one proof swarm's internals;
this one extracts *ecosystem governance* lessons from the community ledger
those swarms report into. The wiki is a live instance of the system SWARM
simulates — heterogeneous agents (humans + AI systems) emitting claims of
uncertain quality into a shared ledger, with a curation mechanism deciding
what gets accepted and how it gets labeled.

## The core principle

**When generation outpaces verification, the ledger's design — not the
generators' quality — determines ecosystem health.** Every mechanism on the
wiki page is compensation for the same asymmetry: AI can emit
plausible-looking proofs faster than the community can referee them. The
default-⚪ label, the 🔴 category, the Lean-formalization push, the
not-a-benchmark disclaimers — all are ledger-side controls on what SWARM
calls adverse selection: low-quality claims being preferentially accepted
because checking is expensive.

## Lessons and their translation

Each wiki mechanism maps onto two planes: the **model plane** (constructs in
the SWARM simulation — metrics, scenarios, payoff terms) and the **practice
plane** (how this rig itself operates — dispatch, beads, the runs ledger).

| # | Wiki mechanism | Model plane (SWARM sim) | Practice plane (this rig) |
|---|---|---|---|
| 1 | **Default-unverified ledger.** New claims enter ⚪; labels upgrade only through verification; Lean formalization converts social refereeing into mechanical checking | Acceptance gates with explicit verification cost; `plausibility_certificate_gap` (beads mt8a) measures the accepted set's drift from certified truth — the quantity the ⚪ default exists to bound | Artifact-only DONE trigger (`done_requires_artifact`); gate execution audit (erdos-1038 lesson 6); claims graded on the Auditor scale until a mechanical gate passes |
| 2 | **The denominator problem.** "Not a benchmark": only successes are logged, attempts are invisible, so the ledger systematically overstates capability | Dual reporting already exists (`toxicity_rate` vs `toxicity_rate_all`, `E[p\|accepted]` vs population); missing piece: a *survivorship gap* metric `E[p\|accepted] − E[p\|attempted]` reported per epoch — the amount a wins-only view inflates apparent quality (bead filed) | `runs/` ledger records attempts, not outcomes; prereg locks (calibration-prereg) commit to reporting before results exist; retros score predictions against the full attempt set |
| 3 | **Rediscovery confound.** An "AI solution" may recapitulate known work; detecting this needs exactly the literature scholarship AI was meant to replace. Corollary: obscurity ≠ difficulty | `scholar_bench` family (citation precision/recall, `citation_laundering.yaml`) models claims whose novelty must be validated against a corpus | Research Scout runs prior-work search before novelty claims; ~30% of filed bugs already fixed at HEAD (CLAUDE.md § Test fix discipline) is the same confound locally — verify against the current corpus before crediting a "find" |
| 4 | **Secondary contributions dominate.** Hundreds of formalizations vs. a handful of primary solutions; wiki insists both classes are "equally important" | Role heterogeneity where verifier/formalizer agents earn payoff by raising *others'* `p` — `synthesis_fraction` / `synthesis_depth` already measure this; payoff engines that reward only primary acceptance underprice the dominant value class | Dispatch credits verification, reproduction, and consolidation work (Reproducibility Sheriff quota) at parity with primary results; DONE artifacts include gate runs, not just features |
| 5 | **Independent convergence as verification.** Problem 90 solved independently by two labs' models: replication across systems with different failure modes substitutes partially for depth of review | Corroboration-weighted acceptance: independent agents converging on a claim raises effective `p` — but only under independence; the adversarial dual is collusion faking convergence (`swarm/metrics/collusion.py` measures exactly this). Convergence across *approach families* (erdos-1038 lesson 3) is stronger than within one | `/bv-dispatch` flags convergent bets (two personas independently spec'd qoro → priority signal); family clustering keeps convergence meaningful by preventing monoculture |
| 6 | **Credibility lives in the curation layer.** The ledger works because a trusted maintainer sets taxonomy, status codes, and submission norms — the governance layer *is* the artifact | This is SWARM's central thesis restated from the field: governance levers (acceptance thresholds, ρ internalization, reputation) determine outcomes more than agent quality; the wiki is a natural experiment in ledger design | beads + `runs/runs.db` + CLAUDE.md are our curation layer; schema triggers and retro audits are the maintainer role made mechanical |
| 7 | **Incentive design against Goodharting.** "Solving an Erdős problem doesn't automatically merit publication" — a pre-emptive strike against farming problem-count as prestige | Decouple acceptance from reward in the payoff engine: reputation terms that discount volume, penalties for retracted (🔴) claims; count-maximizing agents under a count-blind payoff is a ready-made scenario | Retro audits quota-gaming; closing a bead ≠ credit — the gate result is the credit; consolidation quotas reward reduction, not production |

## The adversarial reading

Lesson 5 deserves a warning label: independent convergence is only evidence
under actual independence. Two models trained on overlapping data, or two
agents sharing a favored route (the thing erdos-1038's blind-diversity rounds
exist to prevent), converge for correlated reasons — and a colluding
coalition can manufacture convergence deliberately. Any corroboration-weighted
acceptance rule creates an incentive to fake corroboration. A SWARM scenario
that prices this trade-off — verification budget spent on deep review of one
claim vs. shallow corroboration across many — would connect the collusion
metrics to the acceptance mechanism for the first time.

Lesson 2 has a local sting: our own dispatch retros are ledgers of a fleet's
work, and any retro that reads only DONE rows inherits the survivorship bias
the wiki warns about. The attempt denominator (assignments issued, claims
opened) is in the mail table and beads graph; retros should divide by it.

## Enforcement status (skill-lint ladder)

- **Substrate-enforced:** lesson 1's practice plane (artifact trigger, applied
  2026-07-22); lesson 6's practice plane (schema triggers on `agent_messages`).
- **Retro-checked:** lesson 7 (quota-gaming audit), lesson 2's practice plane
  (prereg comparison), lesson 5's dispatch flag (convergent-bet detection in
  `/bv-dispatch` analysis).
- **Prose-only / not yet built:** lesson 2's model plane (survivorship-gap
  metric), lesson 4's model plane (verifier-payoff parity scenario), lesson
  5's model plane (corroboration-vs-collusion acceptance scenario). Follow-up
  beads filed; see below.

## Follow-up beads

- Survivorship-gap metric: `E[p|accepted] − E[p|attempted]` per epoch in
  `SoftMetrics` + dual report in `MetricsReporter` (blocked on mt8a landing —
  `soft_metrics.py` is in flight).
- Claim-ledger scenario: scarce verification budget per epoch, default-⚪
  claims, acceptance via deep-review or corroboration; measures adverse
  selection as the verification queue saturates (`selection_saturation`).
- Corroboration-with-collusion-discount acceptance rule: wire
  `swarm/metrics/collusion.py` signals into the acceptance mechanism and
  measure how much fake convergence a corroboration-weighted gate admits.

## Addendum (2026-07-22): the success tail — formalization latency is collapsing

**Source:** [Buzzard, "Human mathematicians are being outcounterexampled"](https://xenaproject.wordpress.com/2026/07/20/human-mathematicians-are-being-outcounterexampled/)
(Xena Project, 2026-07-20). Three machine-found counterexamples accepted in
two months — Erdős unit distance (ChatGPT, Lean-formalized in a week; a later
Sol formalization ran 1.2M lines in three weeks, vs. mathlib's 2.3M over nine
years), Grothendieck's group-schemes question (Claude Fable; 1,076-line Lean
formalization in **four hours**), and the Jacobian conjecture (Claude Fable;
formalized into DeepMind's Formal Conjectures repo).

Three refinements to the table above:

1. **The verification budget is a moving target, not a constant.** Lesson 1
   treats verification cost as the fixed asymmetry the ledger must absorb;
   Buzzard's data shows the cost collapsing (weeks → hours) when the claim
   class has a mechanical checker and formalization itself is AI-assisted.
   This strengthens the case that the `a2il` claim-ledger scenario's
   per-epoch verification budget is the right lever to *sweep, not fix*: the
   interesting dynamics are in the transient where generation and
   verification capacity race each other.
2. **Counterexamples fall first because they are the cheapest claims to
   verify** — one finite certificate, no coverage of a proof space. This is
   selection on verifiability, not on difficulty, and it predicts which claim
   classes flip next in any domain: whatever the acceptance gate can check
   cheapest. (Same asymmetry dgg-counterexample-lessons.md is built on.)
3. **Pre-formalize the conjecture, not just the proof.** Buzzard's
   recommendation that conjectures be formally stated *before* AI attacks
   them is the ledger's default-⚪ made ex-ante: fix the acceptance predicate
   before search begins, so "solved" is never renegotiated after the fact.
   Our equivalent is the prereg lock — and the erdos-1038 phantom-gate lesson
   says to also verify the predicate actually runs.

The post also surfaces a stratification concern raised in its comments
(paywalled AI access splitting the research population into
capability tiers) — out of scope for the ledger mapping, but a natural
population-heterogeneity lever for the marketplace/economy scenario family.

## Prior related findings in this rig

- Orchestration-side dual of this doc: search aggressiveness must be a
  function of acceptance mechanicalness (erdos-1038-swarm-lessons.md).
- Plausibility–certificate gap as the measurable signature of selecting on
  transcript plausibility (dgg-counterexample-lessons.md §3, beads mt8a).
- Side-effect reads beat voluntary protocols (2026-07-19): the wiki's
  default-⚪ is the same move — unverified is the structural default, not an
  honor-system label.
