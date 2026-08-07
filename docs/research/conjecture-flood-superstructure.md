# Conjecture-flood economies and the missing interdisciplinary superstructure

**Date:** 2026-07-26
**Source:** Operator discussion on AI-assisted ("vibe math") navigation of PLT-adjacent fields while formalizing Verse. Captured here because the economic structure of the proposal is exactly the claim-market family SWARM models.

## The observation

AI-assisted exploration lets a researcher traverse related fields at maybe 100–1000x the old rate (a sidestep that used to cost days/weeks of books and papers now costs minutes). Sustained traversal exposes a *missing superstructure*: connective results between fields that plausibly exist but were never written, because interdisciplinary friction made them uneconomical.

Concrete bridge candidates surfaced in one formalization effort (Verse):

1. Denotational semantics × universe polymorphism via Reynolds parametricity (keeping sets compressible / levels irrelevant).
2. Parametricity × clone theory — uniformities in both types and values (endpoints exist: parametricity-as-naturality per Bainbridge–Freyd–Scedrov–Scott; abstract clones ≅ Lawvere theories; the middle span is unwritten).
3. Elementary embeddings × non-wellfounded set theory as two explanations of universe polymorphism (reflection-principle folklore; cf. Kovács on first-class universe levels).
4. Positive set theories (GPK+/∞, Forti–Honsell hyperuniverses) × parametricity — uniformity explained topologically. Most exotic; no known prior attempt.
5. The missing denotational semantics of mathematical notation — set-builder notation and functional logic programming as the same object (the Verse calculus states this from the PL side: expressions denote spaces of values; the set-theory side appears unwritten).

## The decomposition that matters

The "missing superstructure" splits into two different problems:

- **Retrieval failure.** Many "missing" bridges exist as scattered low-citation fragments in the wrong field's vocabulary. The friction was indexing, not intellect. This part is tractable now and is what the 100x speedup already exploits. Verification cost for these claims is low: a search, a read, a citation.
- **Genuinely absent structure.** Here generation is cheap but triage is the bottleneck. An AI mapping 10,000 fields emits ~10^6 candidate bridges, mostly type-correct analogies that fail on contact with a proof obligation. **The bottleneck moves from discovery to verification, and verification does not get the same speedup — unless the domain has a mechanical checker.**

Math-adjacent fields are the best case: Lean/mathlib is a scalable validator, so a bridge can be promoted from "vibe" to "theorem" without a human referee. (Precedent + cautionary tale: nLab is an attempted superstructure that accelerated its users but stayed parochial to one school's vocabulary; AI maintenance + formal gating would fix both weaknesses.)

## Why this is a SWARM problem

A conjecture-flood economy has the adverse-selection structure we model: claims cheap to produce, costly to check ⇒ the accepted set degrades (quality gap goes negative) unless the ecosystem prices verification in. "Advance frontiers speculatively, validate later" only outruns the historical publishing rate if the claim-market design is right:

- claims carry explicit verification status;
- checkers are rewarded;
- nothing is cited as load-bearing until it passes a gate.

That is mechanism design, not capabilities — the capabilities half is already arriving.

## Mapping to existing work

| Idea | Where it lives |
|---|---|
| Scarce verification budget, default-unverified claims | bead `a2il` (claim-ledger scenario) — note added referencing this doc |
| Aggregation rule as governance lever | bead `rrsf` |
| Corroboration + collusion discount | bead `77ao` |
| Heterogeneous verification costs (retrieval-cheap vs proof-expensive vs mechanically-checkable) | **new bead** — see below |
| Downstream contamination when an accepted-unverified claim is falsified | **new bead** — see below |
| Corpus-level N_eff for a knowledge base | bead `48yr` (related: what does a bridge-dense corpus do to effective size?) |

## New scenario levers filed from this discussion

1. **Heterogeneous verification-cost claim economy.** Extend the claim-ledger scenario with claim *classes* by verification cost: (a) retrieval-verifiable (cheap — the bridge already exists somewhere), (b) proof-required (expensive, human-refereed), (c) mechanically checkable (near-zero marginal cost, Lean-like). Predictions to test: verification budget allocation concentrates on (a) and (c) and starves (b); a checker-gated domain sustains a much higher generation rate at equal toxicity; adverse selection concentrates in the proof-required class.
2. **Claim-dependency contamination.** Accepted-but-unverified claims get built upon. Model a citation/dependency DAG over claims; when a load-bearing unverified claim is falsified, measure the cascade (retraction depth, wasted downstream work) as a function of how strictly "load-bearing until gated" is enforced. This is the cost side that justifies verification-status provenance.

---

*The PLT-specific threads (items 1–5 above) are the operator's external formalization work, not SWARM implementation targets; they are preserved here as a concrete instance of the claim-generation regime the scenarios model.*
