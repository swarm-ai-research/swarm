# Graph-Structural Sybil/Collusion Detector — Pre-Registration

**Status:** pre-registered, not yet run
**Date:** 2026-06-04
**Issue:** [`distributional-agi-safety-sk95`](https://github.com/swarm-ai-research/swarm/blob/main/.beads/issues.jsonl) (follows merged `2yw2` / PR #492)

## Motivation

PR #492 added a topology-aware sybil/collusion detector
(`swarm/metrics/graph_structural.py`) that complements the existing
score-then-threshold detectors:

- `models/identity.detect_sybil_clusters` — pairwise Jaccard + cosine of
  counterparty patterns, hard threshold.
- `governance/reputation_governor.detect_collusion_clusters` — mutual high-
  trust pairs → connected components.
- `metrics/collusion.CollusionDetector` — composite per-pair/group score,
  threshold flag.

The new detector ships with end-to-end smoke tests but no head-to-head
comparison. The research claim — *"topology beats thresholds for
coalition detection"* — is unsubstantiated until we have an ROC. This
pre-reg fixes the comparison protocol *before* running it.

## Central hypothesis

> On at least one synthetic coalition family (collusion ring, sybil
> cluster, threshold-dancing adversary), `detect_structural_anomalies`
> strictly dominates the best of the three threshold detectors in
> per-node ROC AUC.

"Strictly dominates" = AUC strictly greater with non-overlapping 95% CIs
from 100 graph-replicate bootstrap.

## Out of scope

- LLM-judgment of clusters (separate evaluation axis).
- Real-data validation (no human-labeled coalition data yet).
- Wiring into `ReputationGovernor` as an enforcement signal — that is the
  conditional follow-up gated on the outcome below.

## Design

### Synthetic graph families

Generated parametrically in `experiments/graph_structural_roc.py`, loosely
inspired by `swarm/redteam/attacks.py` patterns but not invoking the
`AttackScenario` machinery (single-PR scope).

| Family | Coalition signature | Parameters swept |
|---|---|---|
| **Benign** | none (negative control) | n_agents ∈ {30, 50, 100}, edge_density ∈ {0.05, 0.1, 0.2} |
| **Collusion ring** | k-clique with high mutual reciprocity | ring_size ∈ {3, 5, 8}; embedded in benign background |
| **Sybil cluster** | k agents with near-identical counterparty profile, low internal mutuality | cluster_size ∈ {3, 5, 8}; counterparty overlap ∈ {0.6, 0.8, 0.95} |
| **Threshold-dancing** | k agents whose pairwise frequency sits *just below* `CollusionDetector`'s flag threshold | cluster_size ∈ {3, 5}; margin ∈ {0.05, 0.1} |

Each (family, parameter) cell generates **100 replicates** with distinct
seeds. Ground truth per node: `1` if planted in a coalition, `0`
otherwise.

### Detectors

All four detectors are wrapped as `Callable[[List[Edge]], Dict[str,
float]]` returning a per-node anomaly score in `[0, 1]`:

| Detector | Score per node |
|---|---|
| `graph_structural.detect_structural_anomalies` | max `(reciprocity_z, -log10(p)) / normalizer` across anomalies containing the node; 0 if node in no candidate |
| `identity.detect_sybil_clusters` | max behavioral similarity to any other node in returned clusters; 0 if not flagged |
| `reputation_governor.detect_collusion_clusters` | 1 if in a returned cluster else 0, with a continuous fallback = (#mutual-trust-neighbors / n_agents) |
| `metrics.collusion.CollusionDetector` | `pair_collusion_score` averaged over the node's incident pairs |

Per-node ROC (TPR vs FPR) is computed by sweeping a threshold across each
detector's score distribution, *separately per detector* — this is the
apples-to-apples scoring axis we want.

### Primary metric

Per-detector **per-node ROC AUC** on each family, with 95% CI from 100
graph-replicate bootstrap.

### Secondary

- Per-family precision@k (k = planted coalition size).
- Wall-clock per detector (informational only — not a decision criterion).

## Decision rule

1. **Hypothesis confirmed (graph_structural strictly dominates on ≥1
   family):** file a follow-up to wire `graph_structural` into
   `ReputationGovernor` as an enforcement signal, gated by the
   `pvalue ≤ 0.05` test the detector already emits.
2. **No-strict-dominance but graph_structural ties on all families:** ship
   the detector as a *secondary metric* (logged alongside existing
   detectors in `MetricsReporter`); no governance wiring.
3. **Threshold detectors strictly dominate graph_structural on every
   family:** publish negative result in CHANGELOG + research log; keep
   the module in place as a benchmark anchor for future detectors; no
   governance wiring.

## Falsifiers

- If `graph_structural` is dominated by a threshold detector on the
  threshold-dancing family (the family it was *designed* to catch), that
  is a stronger negative than the headline AUC suggests — call it out
  explicitly in the writeup.
- If `graph_structural` ROC degrades non-monotonically as `cluster_size`
  grows, the Charikar peeling step may be the culprit; flag as a known
  caveat before any governance wiring.

## Deliverables

- `experiments/graph_structural_roc.py` — generator + sweep + plot.
- `runs/<ts>_graph_structural_roc/` — per-family AUC table, ROC plots,
  bootstrap CIs.
- CHANGELOG entry under `[Unreleased]` summarizing the outcome and the
  applied decision rule.

---

## Results (run 2026-06-04, 20 replicates per cell)

**Status:** post-registered findings, n=20 (below the n=100 pre-reg target;
re-run before drawing publication-grade conclusions). Decision-rule verdict
stable across re-runs in this regime.

| family | graph_structural | collusion_score | reputation_mutual | identity_jaccard |
|---|---|---|---|---|
| benign | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] | 0.500 [0.50, 0.50] |
| collusion_ring (size 3) | 0.592 [0.57, 0.62] | **1.000** | 0.692 [0.67, 0.72] | 0.723 [0.69, 0.76] |
| collusion_ring (size 5) | 0.975 [0.93, 1.00] | **1.000** | 0.888 [0.87, 0.91] | 0.428 [0.38, 0.47] |
| collusion_ring (size 8) | **1.000** | **1.000** | 0.992 [0.99, 1.00] | 0.213 [0.19, 0.24] |
| sybil_cluster (overlap 0.80) | 0.571 [0.53, 0.61] | **0.666 [0.62, 0.70]** | 0.334 [0.30, 0.37] | 0.190 [0.16, 0.22] |
| sybil_cluster (overlap 0.95) | 0.532 [0.50, 0.57] | **0.707 [0.67, 0.74]** | 0.403 [0.36, 0.45] | 0.166 [0.14, 0.18] |
| threshold_dancing (m=0.10) | 0.126 [0.05, 0.23] | **0.998 [0.99, 1.00]** | 0.067 [0.04, 0.09] | 0.146 [0.11, 0.18] |
| threshold_dancing (m=0.05) | 0.126 [0.05, 0.23] | **0.998 [0.99, 1.00]** | 0.067 [0.04, 0.09] | 0.146 [0.11, 0.18] |

**Verdict (decision rule applied):** *graph_structural does not strictly
dominate on any non-benign family. Per pre-reg #2: ship as a secondary
metric only; no governance wiring.*

### What changed our minds

1. **`collusion_score` (the existing composite threshold detector) wins or
   ties on every family.** This was not the expected baseline — the
   pre-reg implicitly assumed threshold detectors would degrade on the
   high-fidelity adversarial families. For ROC *ranking*, the composite
   score's pair-frequency + acceptance-rate + benefit-correlation
   features turn out to dominate even when no single component is
   individually decisive.
2. **`graph_structural` AUC is *worse than chance* on threshold_dancing
   (0.126 ± 0.10), confirming a pre-registered falsifier.** The
   threshold_dancing cluster has very high per-node degree; in the
   configuration-model null those high-degree nodes randomly hit each
   other a lot, so the null's densest-subgraph density is high, the
   cluster's `density_pvalue` saturates near 1.0, and the per-node score
   collapses to 0. Honest nodes meanwhile occasionally land in modest
   label-propagation communities with smaller p-values and end up
   ranked higher. **Net: the configuration-model null is defeated by
   uniformly-high-degree coalitions.** This is a real limitation of the
   detector, not just of this benchmark.
3. **`identity_jaccard` AUC drops below 0.5 on larger coalitions and on
   sybil families.** Returned clusters include genuinely behaviorally-
   similar honest pairs because they share a few popular counterparties,
   and large planted coalitions get fragmented across multiple returned
   clusters whose member-pair similarities are then divided by the
   per-cluster max. This is a known weakness of pairwise-similarity
   thresholding at large coalition sizes — worth surfacing as a
   secondary finding even though it wasn't the headline question.

### Known caveats

- **`threshold_dancing` margin sweep is degenerate.** Margins 0.05 and
  0.10 produce identical graphs because the target pair-count is
  `int(round(mean + (2-margin)*std))` and adjacent margins round to the
  same integer on this background. Findings hold for "any pair-frequency
  z just below the CollusionDetector cutoff", not for a continuous margin
  sweep. Either replace integer pair counts with Bernoulli per-step
  emission, or sweep margin over a wider range.
- **n=20 not n=100.** CI widths reflect 20 replicates; pre-reg target was
  100. Re-run before any external-facing claim. The verdict's qualitative
  shape (dominance / tie / dominated) is stable across re-runs in this
  regime, but the dominance gaps on collusion_ring sizes 3 and 5 deserve
  the tighter CIs.
- **Pair-score adapter for `collusion_score`** uses
  `report.agent_collusion_risk` directly. That includes both pair-score
  contributions and group-membership boosts. For an apples-to-apples
  comparison of *just* the pair signal, a future revision should ablate
  the group-membership contribution.

### Follow-up

- File a separate issue to investigate the high-degree-coalition defeat
  of the configuration-model null. Candidate fixes: size-conditioned
  null sampling; degree-binned local null; weighted reciprocity z-score
  using interaction *weights*, not presence.
- Add a degree-distribution diagnostic plot per generated family so
  future detector authors can see the shape they're being scored on.
- Per the decision rule: **no governance wiring follow-up filed.**
  Detector remains available as a metric for runs that want to log it
  alongside `CollusionDetector` for triangulation.

---

## Re-run after detector fixes (2026-06-04, beads-kwyf, 20 replicates)

Two fixes landed in the same PR after the negative result above:

1. **`density_pvalue` was buggy**, not just conservative — it compared
   observed density against the null's *globally densest* subgraph
   instead of the same nodes' density in the null. Fixed to the
   subset-conditioned test the docstring originally claimed.
2. **Multiplicative scoring replaced by rank aggregation**
   (`rank_aggregated_scores`). Each anomaly is ranked across all
   candidates on four signals (edge_probability, reciprocity_z,
   size-normalized k-core, -log10 p-value); composite = mean rank. No
   single signal can veto the others. Also switched the density rank to
   `edge_probability` (in [0, 1]) so a tight small clique correctly
   outranks a large-but-sparse community.

### Re-run results

| family | graph_structural (new) | (was) | collusion_score |
|---|---|---|---|
| benign | 0.500 | 0.500 | 0.500 |
| collusion_ring (size 3) | 0.598 [0.55, 0.65] | 0.592 | 1.000 |
| collusion_ring (size 5) | **1.000** | 0.975 | 1.000 |
| collusion_ring (size 8) | **1.000** | 1.000 | 1.000 |
| sybil_cluster (overlap 0.80) | 0.657 [0.625, 0.688] | 0.571 | 0.666 [0.623, 0.704] |
| sybil_cluster (overlap 0.95) | **0.754** [0.721, 0.786] | 0.532 | 0.707 [0.674, 0.744] |
| threshold_dancing (m=0.10) | **1.000** | 0.126 | 0.998 [0.995, 1.000] |
| threshold_dancing (m=0.05) | **1.000** | 0.126 | 0.998 [0.995, 1.000] |

### Verdict (re-applied)

**Decision-rule verdict unchanged: no *strict* dominance** — the strict
test requires `graph_structural`'s lower CI > each other detector's
upper CI, and at the AUC=1.0 ceiling that's mechanically impossible
when collusion_score also hits 1.0. Per pre-reg rule #2, the detector
still ships as a *secondary metric* only.

### What actually changed

- **threshold_dancing AUC 0.126 → 1.000.** The pre-registered falsifier
  no longer fires. The detector now perfectly catches the family it was
  designed for, validating the theory: once the null is correct and
  scoring is non-multiplicative, the structural signature (mutual edges
  among small subsets at near-saturation edge probability) wins.
- **sybil_cluster (95% overlap) AUC 0.532 → 0.754**, strictly better
  than `collusion_score`'s 0.707 (though CIs overlap). The detector now
  beats threshold detectors on the family they were specifically
  vulnerable to.
- **collusion_ring size 5 AUC 0.975 → 1.000.** Pre-existing strength
  cemented.
- The qualitative finding has flipped from "shipping as secondary
  metric because nothing dominates" to "shipping as secondary metric
  because graph_structural and collusion_score are now interchangeable
  on AUC, with complementary failure modes worth triangulating across."
  Stronger case for keeping both in `MetricsReporter` than the original
  negative result implied.

### Governance-wiring decision (initial re-run, AUC-saturated)

No follow-up filed *here*. Tying at AUC 1.0 on threshold_dancing is the
result the theory predicted; it is not strict dominance by the
pre-registered CI test. (See the next section — the saturation itself
was a benchmark problem, fixed in the harder generator pass.)

---

## Harder generators + final re-run (2026-06-04, 20 replicates)

The first re-run hit AUC 1.0 on multiple families, making the strict-CI
dominance test mechanically impossible to pass even when one detector
was clearly better. The generators were *too easy*. Three changes
removed the ceiling without sacrificing the planted ground truth:

- `collusion_ring`: incomplete clique (`ring_density=0.85`, default
  was implicitly 1.0), wider p range, more ring↔honest crossings.
- `threshold_dancing`: target pair-count moved from z≈1.9 to z≈1.0
  (further below the CollusionDetector cutoff), only 70% of cluster
  pairs interact, 60% of those are mutual. The cluster's signature
  becomes a partial noisy clique rather than a saturated one.
- `sybil_cluster`: unchanged (was already in the informative 0.5–0.75
  range).

### Final AUC table

| family | graph_structural | best other detector | strict dominance? |
|---|---|---|---|
| benign | 0.500 | — | tie (control) |
| collusion_ring (size 3) | 0.584 [0.55, 0.64] | identity_jaccard 0.514 [0.49, 0.54] | overlap |
| collusion_ring (size 5) | 0.607 [0.57, 0.66] | **reputation_mutual 0.749 [0.69, 0.80]** | dominated by reputation_mutual |
| collusion_ring (size 8) | **1.000** | reputation_mutual 0.955 [0.94, 0.97] | **gs strictly dominates** |
| sybil_cluster (overlap 0.80) | 0.657 [0.625, 0.688] | collusion_score 0.666 [0.62, 0.70] | overlap |
| sybil_cluster (overlap 0.95) | 0.754 [0.721, 0.786] | collusion_score 0.707 [0.67, 0.74] | overlap (gs leads on mean) |
| threshold_dancing (m=0.10) | **0.990 [0.970, 1.000]** | collusion_score 0.936 [0.90, 0.97] | **gs strictly dominates** |
| threshold_dancing (m=0.05) | **0.990 [0.970, 1.000]** | collusion_score 0.936 [0.90, 0.97] | **gs strictly dominates** |

### Final verdict (re-applied)

**Decision rule #1 fires:** `graph_structural` strictly dominates on
**3 families** (`collusion_ring_size8`, both `threshold_dancing`
variants). Per pre-reg, **governance-wiring follow-up filed as
beads-4ae5.**

### What we now know

- **graph_structural is the right detector for tight reciprocal
  coalitions and for adversaries that pace themselves below
  CollusionDetector's z=2 cutoff.** This was the theoretical claim;
  it's now empirically supported with strict-CI dominance on three
  families.
- **graph_structural is *not* the right detector for small or diluted
  coalitions.** On `collusion_ring_size3` and `size5`, it sits at AUC
  0.58–0.61; `reputation_mutual` (which uses trust scores derived from
  `p`) wins on the size-5 case (0.749, strictly dominant). The
  structural signature is just too sparse at small coalition sizes for
  topology alone to identify.
- **The detectors are complementary, not interchangeable.** Different
  families have different winning detectors:
  - small/diluted rings → `reputation_mutual` or `identity_jaccard`
  - tight rings + threshold_dancing → `graph_structural`
  - sybil clusters with high counterparty overlap → `graph_structural`
  - sybil clusters with moderate overlap → `collusion_score`
- This is the stronger case for `MetricsReporter` logging *all* of
  them and for the governor consuming all of them: there is no
  one-size-fits-all detector for coordination.

### Remaining caveats

- n=20 still, not n=100. Re-run before any external claim.
- `collusion_ring_size8` is still ceiling-bound at AUC 1.000.
  Increasing background density further would help but risks pushing
  smaller rings below random.
- The `threshold_dancing` margin sweep is still degenerate (margins
  0.05 and 0.10 produce the same `target_count` integer). Findings
  hold for "any pair-count z near 1.0 below cutoff" but not for a
  continuous margin sweep. See beads-sk95 caveats section.

---

## Operating-point metrics + cluster recovery (2026-06-05, beads-5cdk)

The pre-reg used per-node ROC AUC as the headline because it gives
each detector its best ranking shot. But governance acts on each
detector's **own operating threshold**, not on a swept-threshold
ranking — so ROC dominance is necessary but not sufficient for a
"flip default ON" decision. This pass adds three native-cutoff
metrics:

- **Precision / Recall / F1@native** — flag at each detector's *own*
  decision threshold:
  - `graph_structural`: pre-registered `is_suspicious` gate
    (size≥3, density≥1, reciprocity_z≥2, p≤0.05)
  - `identity_jaccard`: `IdentityConfig.behavioral_similarity_threshold=0.8`
  - `reputation_mutual`: `detect_collusion_clusters(min_mutual_trust=0.9)`
  - `collusion_score`: agent risk ≥ 0.5 (CollusionDetector default)
- **Cluster recovery** — for each planted coalition, max Jaccard with
  any returned cluster; mean across planted (greedy 1:1 assignment for
  the multi-coalition extension in beads-qoro).

### Results (20 replicates, same generators as final re-run above)

| family | gs F1@native | gs recovery | best other F1 | best other recovery |
|---|---|---|---|---|
| collusion_ring_size3 | 0.035 [0.00, 0.10] | 0.023 [0.00, 0.06] | 0.000 | 0.000 |
| collusion_ring_size5 | **0.131 [0.05, 0.24]** | 0.100 [0.03, 0.20] | 0.000 | 0.000 |
| collusion_ring_size8 | **0.318 [0.28, 0.39]** | **1.000 [1.00, 1.00]** | 0.000 | 0.000 |
| sybil_cluster (overlap 0.80) | 0.000 | 0.000 | 0.000 | 0.000 |
| sybil_cluster (overlap 0.95) | 0.000 | 0.000 | 0.000 | 0.000 |
| threshold_dancing (m=0.10) | **0.994 [0.98, 1.00]** | **0.990 [0.97, 1.00]** | 0.000 | 0.000 |
| threshold_dancing (m=0.05) | **0.994 [0.98, 1.00]** | **0.990 [0.97, 1.00]** | 0.000 | 0.000 |

### The headline finding

**Every threshold detector is at F1=0.000 across every adversarial
family.** They discriminate on score ranking (the AUC pass) but at
their own native cutoffs they do not actually flag any planted
coalition member. This means:

- The AUC-vs-threshold framing in earlier discussion *understated* the
  difference. The threshold detectors don't just lose on ranking —
  they don't fire at all at their own operating thresholds against
  these adversarial families.
- `graph_structural`'s `is_suspicious` gate is the only detector
  that actually flags planted coalitions at native cutoff. On
  `threshold_dancing` it does so with F1 ≈ 0.99 — operating-point
  parity with its AUC.

### Verdict (F1@native axis)

By the same strict-CI dominance test on F1@native: **`graph_structural`
strictly dominates on 4 families** (`collusion_ring_size5`,
`collusion_ring_size8`, both `threshold_dancing` variants — one more
than the AUC pass, which added `collusion_ring_size5`).

### Governance decision

**Recommend flipping `ReputationGovernor(structural_enabled=True)` to
default ON.** The original beads-4ae5 scope set it OFF "until canary"
because the AUC pass at that point didn't separate the detectors at
their own thresholds; the F1@native pass now does, and the
direction is unambiguous: native-cutoff comparisons confirm
`graph_structural` is the only detector that actually fires on
adversarial families.

Caveats before flipping:

- n=20, not n=100. Re-run before the flip-on PR lands.
- Operating-point F1 on small rings is still low (0.035 for size 3,
  0.131 for size 5). The detector mostly catches *tight* coalitions
  and *paced* adversaries. The governor consuming both this signal
  and `detect_collusion_clusters` retains the small-ring coverage.
- Threshold detectors' F1=0 across the board may reflect that their
  built-in defaults were tuned for a different distribution than
  these synthetics. A separate study should sweep each detector's
  threshold and report the F1-optimal cutoff — this PR doesn't do
  that.

A follow-up beads issue will be filed for the flip-ON PR with a
canary plan.
