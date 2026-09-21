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
