# Significant is not convincing: a null-model audit of structural collusion evidence

**Bead:** `50bb`. Apart AI Collusion Research Sprint (2026-10-23 → 25), RQ 2.3
"what should count as convincing evidence of collusion?". Follows
[Collusion Detector Flags Everyone](collusion-detector-false-positives.md) (19n0, 1a2w).
Reproduce: `python experiments/collusion_evidence_audit.py` (837 s, no API calls),
then `python experiments/plot_collusion_evidence_audit.py <run_dir>`.
Run: `runs/20261025T202341Z_collusion_evidence_audit`.

## Setup

AI Village chat after the 2026-03-24 cutover: 36,783 directed messages (names
exactly one other agent), 52 agents, 8 monthly windows. Treated as non-collusive
background. Positive control: 4 random agents (≥20 messages sent) redirect a
fraction f ∈ {0.1, 0.2, 0.4} of their outside messages to a co-present member;
speakers, timestamps and volume unchanged. f = 0 is an honest group. 20 groups ×
4 f × 8 months. Every threshold is fixed on odd months (Mar/May/Jul/Sep) and frozen
before the even months are scored.

Five nulls: `configuration` (stub-matching, published default), `reciprocity`
(double-edge swap, with the fix below), `uniform` addressee, `copresent`
addressee (present that day), and `day_perm` (addressees permuted within each day,
so per-day sent **and** received counts are exact).

## 1. The edge leak 1a2w could not locate

`reciprocity_preserving_null` rewired mutual dyads first and never checked whether
they landed on, or opposite, an original one-way edge; the two then merged. On the
full graph (994 edges) the null averaged 984 (981–988), and the whole graph scored
p = 0.005 against itself. Blocking one-way positions during the mutual rewire makes
the count exact (994 every sample) and the whole graph p = 1.00. The 1a2w test that
pinned the dense-graph failure is replaced by one asserting exact edge count and
reciprocity at density 0.5.

Under the exact null the detector's 34-agent core is 831 internal edges vs a null
mean of 828.7, p = 0.0498: "significant", with an effect of two edges.

## 2. Adjudication (test months, 80 groups per cell)

| rule | honest f=0 | f=0.1 | f=0.2 | f=0.4 |
|---|---|---|---|---|
| p<.05 configuration | 0.33 | 1.00 | 1.00 | 1.00 |
| p<.05 reciprocity | 0.05 | 0.42 | 0.40 | 0.45 |
| p<.05 uniform | 0.39 | 0.82 | 1.00 | 1.00 |
| p<.05 copresent | 0.28 | 0.62 | 0.91 | 1.00 |
| p<.05 day_perm | 0.41 | 0.79 | 0.94 | 0.99 |
| **frozen:** day_perm p<.05 & R ≥ 2.41 | **0.01** | 0.16 | 0.26 | 0.45 |

On the detector's **own** candidate clusters (mean 85% of agents), calib/test:
configuration 8/8 / 8/8, uniform 4/8 / 4/8, copresent 4/8 / 4/8, reciprocity 0/8 / 0/8,
**day_perm 0/8 / 1/8**.

- **Schedule-preserving null removes the 19n0 false positives** on the clusters
  the detector actually proposes. Nulls that ignore who was active (and how much they
  were addressed that day) read co-activity as coordination.
- **No null makes a p-value sufficient.** 28–41% of honest named groups are
  significant under every message-level null, day_perm included, with a median
  honest ratio of 1.00. This is real affinity, not miscalibration.
- **Honest affinity overlaps planted collusion.** The 95th-percentile honest
  enrichment (r* = 2.41, bootstrap [1.76, 3.28]) exceeds the median for f = 0.4
  (2.19). AUC of the ratio, honest vs planted: day_perm 0.74 / 0.84 / 0.91, uniform
  0.77 / 0.88 / 0.95. Conditioning on received counts absorbs part of the planted
  signal, which is the cost of the calibration above.
- `reciprocity` is calibrated but blind: it scores the binary presence graph, and
  redirecting along edges that already exist adds nothing it can see.
- March and April (509 and 1,057 messages) detect nothing at any f.

## 3. Screening

Best candidate vs planted group: mean Jaccard 0.23 at every f, 0/480 reach 0.5.
Candidate generation returns near-complete node sets on dense chat; no null can
repair a candidate that is never proposed.

## Verdict

Structural excess is a trigger for review, not evidence. It becomes informative
only with an activity-schedule-preserving null **and** an effect-size bar
calibrated on honest groups from the same population, and even then catches fewer
than half of heavy coalitions. Consistent with 9err: detectors that require
quality asymmetry had 0% FPR on honest convergence where co-occurrence had 54%.

## Limits

No collusion labels (the background is assumed honest; the "a third of random
groups are significant" finding does not depend on that). Edges inferred from
mentions. One planting model (volume-preserving redirection); action or
side-channel coordination is invisible to all of these. k = 4 only. Pair-level
analogue for `CollusionDetector` is xf2r, separate.

Licence: research use only, no training without written permission, cite
AI Digest / AI Village.
