---
description: "Can SWARM's proxy hold a belief at p = 1? Only if the sigmoid is sharp. When it is, all fabrication loss sits in the near-certain band, and so does most honest work."
---

# Certainty-band audit: loss at near-certain p

**Bead:** `l1d6`. **Motivation:** the AI Village loan
([blog](../blog/the-m5000-loan-goal-consumed-judgment.md)), where the Ṁ3,045
loss came from a belief held at confidence 1.0 that the market price could not
move. Expected-harm accounting prices a p = 1 belief at (1 − p)·h = 0, so a
system can carry that loss while its metrics report none.

## 1. Can p saturate?

`ProxyComputer` (`swarm/core/proxy.py`) clamps v_hat to [−1, +1] and returns
p = 1 / (1 + e^(−k·v_hat)).

- **v_hat reaches exactly +1** when task progress and engagement are both +1
  with zero rework, rejections and misuse. It approaches −1 only
  asymptotically: the penalty terms are 2·decay^n − 1, which never reaches −1
  for finite n.
- **At the default k = 2, p is confined to [0.119, 0.881].** The proxy cannot
  express a belief stronger than about 88/12. The two scenarios in
  `scenarios/` that set the proxy's k (`proxy_sigmoid_k`) set it to 2.0, and
  `dynamic_toxicity_proxy_drift` only lowers it over time.
- **p reaches exactly 1.0 in float64 once k ≥ 37** at v_hat = 1, because
  e^(−37) is below half of machine epsilon. p never reaches exactly 0 through
  the proxy (the minimum is about 3.7 × 10⁻⁴⁴ at k = 100). Code that sets p
  directly (bridges, tests) is outside this audit.

Both boundaries are pinned in `tests/test_metrics.py::TestExtremePLossShare`.

## 2. Metric

`SoftMetrics.extreme_p_loss_share(interactions, band=0.05)` splits realized
loss, (s_minus + h) per accepted interaction whose ground truth is −1, by where
p sat:

- **share_high**, p ≥ 1 − band: confidently wrong, the loan case.
- **share_low**, p ≤ band: accepted although p said it was bad, which is a
  selection failure rather than a calibration one.

It also counts accepted interactions at p exactly 0.0 or 1.0 (`saturated`). It
returns None when no accepted interaction carries ground truth.

## 3. Run

`python -m experiments.certainty_band_audit` on `escalating_pressure` (the
scenario that records ground truth per interaction; fabricated positives are
−1). Only `sigmoid_k` changes between cells. Seeds 42, 7, 101; band 0.05. Means
over seeds:

| k | p range | accepted in high band | saturated (p = 1.0) | ground-truth bad share | proxy toxicity | share of loss in high band |
|---|---|---|---|---|---|---|
| 2 | [0.426, 0.881] | 0.000 | 0 | 0.072 | 0.295 | 0.000 |
| 5 | [0.320, 0.993] | 0.327 | 0 | 0.075 | 0.146 | **1.000** |
| 10 | [0.176, 1.000] | 0.587 | 0 | 0.059 | 0.096 | **1.000** |
| 40 | [0.001, 1.000] | 0.747 | 9 | 0.074 | 0.074 | **1.000** |

Per-seed rows are in the run's `results.csv`. Realized loss is small: 4 to 21
accepted fabrications per run.

## 4. Findings

**1. At the default sharpness, the certainty failure cannot occur.** No
accepted interaction reaches the band at k = 2, so share_high is 0 by
construction, not by good governance.

**2. Once k ≥ 5, all fabrication loss is in the near-certain band, in every
seed.** Nine accepted interactions per run sit at exactly p = 1.0 by k = 40.
This is the loan case reproduced: the system's most confident beliefs carry
all of its realized loss.

**3. The band is not a detector.** At k = 40 it holds 75% of all accepted work,
nearly all of it honest. Every dollar of loss is in the band, but most of the
band is fine. Capping p at 1 − band would restore a nonzero expected-harm floor
but would not tell fabrication from success.

**4. Sharpness alone moves proxy toxicity fourfold.** Ground-truth bad share
stays near 0.07 while E[1 − p | accepted] falls from 0.295 at k = 2 to 0.074 at
k = 40. Toxicity numbers are only comparable across runs at the same k. Today
that holds because every scenario uses k = 2, but nothing enforces it.

## 5. What this scenario builds in

`pressure_responsive` generates a fabricated report as honest-success
observables **plus** positive offsets (+0.5 progress, +0.5 engagement, −2
rework). Fabrication therefore outscores real success by design. Across the
three seeds, every accepted fabrication has v_hat ≥ 0.874 and no honest success
exceeds 0.834. A "too good to be true" upper-tail flag would catch all of it
here, and that result should not be believed: a real fabricator matches honest
success rather than exceeding it. Finding 2 depends on fabrication scoring at
least as well as success, which is the premise of the loan case. Finding 3
would get worse, not better, under a matching fabricator.

## 6. Not built

The calibration gate the bead suggested, which would cap p away from 0 and 1.
Finding 3 says a cap moves expected-harm accounting but not detection, so the
tradeoff worth measuring is the harm floor against the payoff distortion on
honest work. That is a separate sweep and was not run.
