# The structural collusion detector flags everyone

**Bead:** `19n0`. **Source:** [`aidigestorg/ai-village`](https://huggingface.co/datasets/aidigestorg/ai-village)
(AI Digest), post-cutover chat, 2026-03-24 → 2026-09-18. Reproduce with
`experiments/ai_village_detector_fp.py`.

We tune coalition detectors on synthetic planted coalitions — `qoro` traced
`graph_structural`'s ROC under overlap, `3ru4` measured a 35.6% honest
false-positive cost at the audit frontier. Neither had ever been run against
real multi-agent chat. This supplies that denominator, and the answer is worse
than a rate.

**On 28,476 real directed messages among 42 agents, `detect_structural_anomalies`
flags 14 of 14 candidate clusters (100%), in every month, under both edge
definitions — and flags a graph with randomly assigned partners just as
readily.**

## Result

| variant | msgs | nodes | density | flagged | cluster size / graph |
|---|---|---|---|---|---|
| A. names exactly one agent | 28,476 | 42 | 0.459 | 2/2 | 0.87 |
| B. `@handle` only (stricter) | 25,855 | 34 | 0.645 | 2/2 | 0.93 |
| C. **random-target null** | 28,476 | 42 | 0.756 | 1/1 | **1.00** |

Per month, real data: 2/2 flagged in all seven months — 14/14 pooled. The
random-target null is flagged 2/2 in all seven as well.

## Why this is not a rate but a defect

Three things have to be read together.

**The flagged coalitions are the entire village.** Flagged clusters average
87–93% of all nodes, and in the null 100%. A coalition comprising everyone is
not a finding about coordination; it is a restatement of the population. The
detector has no size prior — nothing in `detect_structural_anomalies` penalises
a cluster for containing every agent in the graph.

**Randomising partner choice does not reduce flagging.** Variant C assigns each
message's addressee uniformly at random while preserving who talks how much. It
destroys every trace of who-coordinates-with-whom, and it is flagged exactly as
often. Whatever the detector is responding to, it is not coordination structure.
This is the observation that turns a high false-positive rate into a statement
about the instrument.

**The mechanism is density, and real group chat is dense.** Candidates come from
Charikar densest-subgraph plus label-propagation communities. On a graph with
edge density 0.30–0.76, both return near-complete node sets; label propagation
collapses to a single giant community. The degree-preserving configuration null
then finds those sets significant, because it does not reproduce the reciprocity
of ordinary dialogue — in July, reciprocity 0.90 against `reciprocity_z` ≈ 7.5.
Agents in a group chat answer each other. The null model does not know that.

## What this does and does not say

It does **not** say there is no collusion in the AI Village. There are no labels,
so a positive is uninterpretable — that was the finding of
[`vu70`](ai-village-mapping-design.md), and it applies here. This measures
false-positive *behaviour* only.

It does say that `graph_structural`'s published ROC characteristics do not
transfer to dense observational chat, and that `qoro`'s conclusions should be
read as conditional on the sparsity of its synthetic graphs. A detector that
flags a uniformly random graph as a 42-agent conspiracy is not usable as a
deployment gate in this regime.

## Two fixes worth testing before it is used on observational data

1. **A size prior.** Reject or heavily discount clusters above some fraction of
   the graph. "Everyone is colluding" should cost more than it currently does.
2. **A null that preserves conversational reciprocity.** The configuration model
   preserves degree but destroys mutuality, so any dialogue corpus looks
   anomalous. A reciprocity-preserving null would ask the sharper question:
   is this pair mutual *beyond* what conversation already implies?

Whether signal survives beneath the vacuous global cluster — for instance with
cluster size capped at a third of the graph — is the obvious follow-up, and is
deliberately out of scope here.

## Update (bead `1a2w`): both fixes implemented, neither repairs it

The two fixes proposed above were built, tested and measured. **Neither fixes
the real-data false positives, and finding that out located a deeper defect.**

| configuration on the real graph | flagged |
|---|---|
| none (as published above) | 2/2 |
| size prior only (≤0.5 of nodes) | **0/0** |
| reciprocity-preserving null only | 2/2 |
| both | 0/0 |

**The size prior does not sharpen the detector, it silences it.** `0/0` means no
candidate survives the filter at all — candidate generation only ever produces
near-complete node sets, so capping size removes every candidate rather than
leaving the true ones behind. It suppresses the false positives by suppressing
the output.

**The reciprocity-preserving null changes nothing here** (2/2), despite being
correct on sparse graphs. Chasing why produced the actual finding.

### The real defect: the nulls lose edges

Both null models rewire by stub matching, and stubs collide or form self-loops.
The null graph therefore has systematically *fewer* edges than the observed one
— 8–23% fewer in direct measurement. The consequence is not subtle:

```
density_pvalue(WHOLE GRAPH) = 0.0196     under the configuration null
```

A graph cannot be denser than itself. Any candidate approaching the full node
set is guaranteed "significantly dense," which is precisely why every cluster is
flagged, why a randomly-wired graph is flagged too, and why a size prior appears
to help — it is removing the candidates that trip the artifact, not detecting
anything.

`reciprocity_preserving_null` is therefore built on double-edge swaps
(Maslov–Sneppen), which preserve degree *and* edge count exactly. On a sparse
graph it works as intended:

| | observed | configuration null | reciprocity null |
|---|---|---|---|
| edges | 97 | 85.5 (88%) | **97.0 (100%)** |
| reciprocity | 0.082 | destroyed | **0.082 (exact)** |
| `density_pvalue` (whole graph) | — | 0.0196 | **1.0000** |
| `reciprocity_z` (whole graph) | — | 2.52 | **−0.23** |

**And it does not carry to dense graphs.** At the density real group chat runs
(0.46), roughly 2% of edges are still lost to collisions that exist before any
swap is attempted, and 2% is enough to keep `density_pvalue` at the floor. That
is why the real corpus is still 2/2. Both behaviours are pinned in tests
(`TestEdgeCountArtifact`), including the failure.

### Where that leaves the detector

Unchanged for existing work: `max_size_fraction` defaults to `None` and `null`
defaults to `"configuration"`, so `qoro` and `3ru4` remain reproducible, and a
test asserts default output is identical.

For observational use the honest position is that `graph_structural` is still
not deployable on dense chat, and the remaining blocker is now specific and
fixable: make the rewiring collision-free on dense graphs, or replace the
density p-value with a statistic that is not degenerate when the candidate
approaches the whole graph. Fixing candidate generation — which currently cannot
propose anything smaller than most of the population — is the other half.

## Limits

- **Only `graph_structural` was exercised.** The composite score in
  `CollusionDetector` weights benefit-correlation 0.30, mutual-acceptance 0.20
  and quality 0.25, all of which need `tau` / `c_a` / `c_b` / `accepted` / `p`.
  Observational data carries none of them, so the composite's value would be an
  artifact of how the blanks are filled — setting `accepted=True` alone adds a
  flat 0.20 to every pair. It is not reported, and that is itself a finding: the
  composite is not computable on data of this kind.
- **Edges are inferred, not recorded.** Village chat has no reply, parent or
  addressee field; "names exactly one other agent" is a heuristic that
  [`vu70` D4](ai-village-mapping-design.md) left unvalidated. The result is
  robust to tightening it to `@handle` only and to replacing it with random
  targets, which is the strongest available answer short of a labelled sample.
- **p-values are floored** at 1/200 = 0.005 by `n_null_samples`, so `p<0.05` is
  saturated rather than finely resolved.
- **Candidate generation, not just the null, is the bottleneck** — only two
  candidates per window ever exist to be flagged.

Licence: research use only, no training without written permission, cite
AI Digest / AI Village.
