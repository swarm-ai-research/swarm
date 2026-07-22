# Arm A deviation analysis: the "criteria miss" dissolves on inspection

**Bead:** distributional-agi-safety-eftp (discovered-from k2yr; gates 4e2s)
**Date:** 2026-07-19
**Data:** `runs/20260718T224200Z_calibration_fidelity_seed42` (registered sweep),
`runs/20260719T194722Z_calibration_fidelity_seed42` (exploratory k-extension)
**Prereg:** `docs/research/calibration-prereg.md` (arm A)

## Decision: no prereg amendment — the registered outcome branch applies; the numeric criteria move to c89o where they belong

## Finding 1 — the "pre-registered" thresholds are not in the prereg

The bead frames ECE 0.187 as a miss against "pre-registered <0.1". Provenance
check: **`calibration-prereg.md` contains no numeric pass/fail criteria for
arm A.** It registers the metrics (ECE/MCE/Brier), the procedure, the
"sweep k, identify the ECE-minimizing k" analysis, and four outcome branches —
including, verbatim, the branch we are in:

> **`p` is poorly calibrated but judges agree:** the framework still has a
> usable external anchor; we report the calibration gap honestly and re-tune
> `k`. Adaptive study still unblocked.

The ECE<0.1 / Brier<0.05 numbers come from the **architect dispatch spec of
2026-07-18 attached to the c89o promotion** (controlled-injection scenario) —
success criteria for a *future redesigned* experiment, not commitments of this
one. Option (a), "amend prereg with justified deviation," is therefore moot:
there is nothing to amend, and adopting the c89o spec retroactively as if it
had been registered would itself be the integrity violation.

## Finding 2 — registered readout (confirmatory)

Within the registered sweep (k ∈ [0.5, 5.0], p-grid {0.05…0.95}, 500/bin,
seed 42): **k\* = 5.0, ECE 0.187, MCE 0.629, Brier 0.0906.** ECE monotonically
decreasing in k across the registered range. This is the honest gap the prereg
asks us to report.

## Finding 3 — exploratory k-extension (labeled non-confirmatory)

The registered commitments fix the k sweep range, so this extension
(k ∈ {6, 8, 10, 15, 20, 30}, same seed/grid/code) is exploratory only. It
settles branch (c) — "extend k to test the ceiling claim" — negatively:

| k | ECE | MCE | Brier |
|---|---|---|---|
| 6 | 0.172 | 0.635 | 0.0880 |
| 8 | 0.156 | 0.727 | **0.0863** |
| 10 | 0.150 | 0.730 | 0.0868 |
| 15 | 0.137 | 0.830 | 0.0899 |
| 20 | 0.130 | 0.851 | 0.0932 |
| 30 | **0.123** | 0.851 | 0.0981 |

ECE approaches a floor well above 0.1; **Brier is U-shaped** (minimum ≈0.086
near k≈8, degrading after); MCE explodes toward 0.85. The c89o criteria are
**unreachable by tuning k** under the current generator — there is no Goodhart
path, and no "just extend the sweep" fix.

## Finding 4 — why: the generator makes latent p unrecoverable

`sample_observables` draws observables conditioned on the **realized** v, with
v=+1 and v=−1 ranges that touch at single points (`task_progress`:
[0.3, 0.95] vs [−0.5, 0.3]; `engagement`: [0.1, 0.7] vs [−0.6, 0.1]).
Empirically, **99.985% of draws uniquely decode v from `task_progress`
alone** (200k-draw check). Consequences:

1. The Bayes-optimal p̂ given these observables is ≈0/1 — *sharper is
   genuinely better-calibrated against realized outcomes*, which is why ECE
   falls monotonically in k. The docstring's "distributions overlap so the
   proxy is informative but not perfect" overstates the overlap.
2. **Latent p_true is unrecoverable in principle**: observables carry
   information about v only, so no estimator can distinguish an interaction
   drawn with p_true=0.4 from one drawn with p_true=0.2 given the same
   realized v. Mid-range calibration against latent p is structurally
   impossible — the reliability table's 0%-accuracy bins below 0.6 are this
   fact, not a proxy defect.
3. The residual ECE floor (~0.12) is a property of ProxyComputer's **fixed
   linear v̂ form** (it mixes the noisy count observables and cannot exploit
   the separability), not of information content.

Arm A as registered — calibration against **realized** ground truth — was
measured and reported. Arm A as *aspired to* in `fidelity.py`'s docstring
("faithful estimator of the latent outcome probability") cannot be measured
under this generator at all. That mismatch is the real finding.

## Disposition

- **(a) Amend prereg: NO.** Nothing to amend (Finding 1). The registered
  outcome branch — report the gap, identify k\* — is executed here.
- **(c) Extend k: DONE, exploratory, negative.** Criteria unreachable via k
  (Finding 3); the "ceiling" is real (Finding 4).
- **(b) Improve the experiment: YES — via c89o**, which is precisely the
  redesign Finding 4 demands: inject v̂ (or observables) conditioned on
  **latent p** rather than realized v, making latent-p fidelity measurable.
  The ECE<0.1 / Brier<0.05 spec originated with c89o and remains its
  success criterion, where it is meaningful and (in expectation) achievable.
- **4e2s (calibration epic): UNBLOCKED** per the prereg's own branch — the
  adaptive chain proceeds with the calibration gap stated (ECE 0.187 at
  registered k\*=5; exploratory Brier-optimal k≈8), and arm B's external
  judge as the anchor.

One process note for the epic: when c89o runs, register its criteria in the
prereg (or a versioned addendum) *before* data collection, so the next
"criteria miss" question has a one-line answer.
