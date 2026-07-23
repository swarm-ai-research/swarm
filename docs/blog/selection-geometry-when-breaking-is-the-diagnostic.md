---
date: 2026-06-06
description: "We added a projection-geometric adverse-selection metric to SWARM, ran it through three AUROC studies, and watched it break in exactly the way that turned out to be the diagnostic."
author: "SWARM Team"
keywords:
  - adverse selection
  - selection geometry
  - proxy gaming
  - AUROC
  - cause-3
claims:
  - metric: "toxicity AUROC under evasion"
    value: "0.712 → 0.812 → 0.912"
    description: "Raw toxicity discrimination rises monotonically with v̂ inflation"
  - metric: "selection_saturation AUROC under evasion"
    value: "0.685 → 0.657 → 0.629"
    description: "Saturation degrades under the same evasion — the breakage is the signature"
  - metric: "Cells run across three study versions"
    value: "81 000"
    description: "9 000 (v1) + 18 000 (v2) + 54 000 (v3)"
abstract: "A re-derivation of the quality gap as Cov(p,a)/Var(a) gives a Cauchy-Schwarz bound and a normalized metric, selection_saturation, that measures how much of the bound the governor is realizing. Three AUROC studies, each refuting the previous: v1 was too easy and saturated at 1.0; v2 showed toxicity and saturation are complementary across governor strictness; v3 added a v̂-inflating adversary and inverted the v2 ranking — but the inversion itself is a ground-truth-free proxy-gaming detector."
---

# Selection geometry: when a metric breaking is the diagnostic

## The setup

![Xiaohei cranking a winch to pull the proxy line between accepted and rejected interactions](figures/xiaohei-selection-geometry-01-capacity-winch.png)

If interactions have soft labels `p = P(v=+1 | observables)` and a governor accepts a subset `a ∈ {0,1}`, the quality gap

> `Q = E[p|a=1] − E[p|a=0]`

is the standard adverse-selection metric in SWARM. A negative `Q` means the governor is preferentially admitting low-quality interactions — the classic adverse-selection diagnosis.

If you rewrite `Q` as a covariance,

> `Q = Cov(p, a) / Var(a)`,

it becomes the regression coefficient of `p` on `a`. Cauchy-Schwarz immediately bounds it:

> `|Q| ≤ σ_p / √(αβ)`,    where α = P(accept), β = 1−α.

So there's a natural unit-normalized version — Pearson correlation `ρ(p, a) ∈ [−1, +1]` — and a *saturation* ratio that says what fraction of the C-S ceiling the governor is realizing:

> `selection_saturation = |Q| · √(αβ) / σ_p ∈ [0, 1]`.

A governor saturating its bound is doing all the sorting the proxy will let it. A governor near the ceiling can't be improved by tighter policy — it needs a better proxy.

The decomposition is also clean: `T = baseline_harm − selection_credit`, where `baseline_harm = 1−E[p]` is the population floor and `selection_credit = β·Q` is what governance can move.

That's the metric. Does it carry information beyond raw toxicity?

## Study v1: AUROC = 1.0 means nothing

![Xiaohei dropping every item through a too-wide hoop labeled AUROC 1.0](figures/xiaohei-selection-geometry-02-wide-hoop.png)

The first sweep was a 9 000-cell discrimination task: mix benign + adversarial fixture batches at fraction `f`, run them through a `v̂ > θ` governor, ask which metric best distinguishes adversarial populations (`f ≥ 0.4`) from benign ones (`f ≤ 0.1`).

Result: `baseline_harm`, `toxicity`, and `selection_saturation` all hit AUROC = 1.000 somewhere. The contrast was a 30-percentage-point f-gap with N=200 batches and a noiseless `v̂` — the discrimination task was too easy. Saturated AUROC means the experiment isn't measuring anything.

## Study v2: harder contrast, X-pattern

![Xiaohei balancing on crossed ropes between the lax side and the strict side, carrying toxicity and saturation buckets](figures/xiaohei-selection-geometry-03-tightrope.png)

So we tightened it. 5pp gap (`f ≤ 0.25 vs f ≥ 0.30`), N=50 batches, `v̂ + 𝒩(0, 0.25)` so the governor isn't a perfect oracle on `p`. 18 000 cells.

![AUROC across governor strictness](../research/figures/selection-geometry-auroc.png)

The middle panel (inflation=0.3) shows the v2 picture cleanly:

- **Lax governors (θ < 0):** raw `toxicity` wins (AUROC 0.89–0.98); saturation is near chance (0.47–0.50). Toxicity collapses to baseline harm when β → 1 — it just *is* the population mean.
- **Strict governors (θ > 0):** roles invert. Toxicity at chance (0.51–0.53); saturation peaks (0.87–0.88). Under heavy filtering the accepted slice is tiny and its `p` distribution narrow, so `|Q| / σ_p` saturates.
- **Crossover at θ ≈ 0.** Mean AUROC nearly tied (0.712 vs 0.685).

The two metrics aren't competing — they're complementary, sweeping the same AUROC budget across opposite halves of the governor space. The operational rule looked clean: report saturation when acceptance < 50%, toxicity when > 50%.

## Study v3: an evasive adversary inverts the ranking

![Xiaohei pumping more air into an inflated proxy balloon while the ground-truth weight stays anchored — the string decouples and the ranking flips](figures/xiaohei-selection-geometry-04-proxy-balloon.png)

But every realistic adversary is *adaptive*. They learn how the governor works and game it. The cheapest gaming strategy is to inflate `v̂` without changing real quality — the [cause-3 pattern from the arm-2 line](../research/adaptive-arm2-cause3-active-probe.md). So we added a `v̂`-inflation knob: adversarial interactions get `v̂ += inflation` while `p` stays low.

| inflation | toxicity mean AUROC | saturation mean AUROC | baseline_harm |
|---|---|---|---|
| 0.0 (no evasion) | 0.712 | 0.685 | 0.999 |
| 0.3 (mild) | **0.812** | 0.657 | 0.999 |
| 0.6 (strong) | **0.912** | 0.629 | 0.999 |

The v2 conclusion *reverses*. Toxicity gets stronger as evasion increases; saturation gets weaker.

Why? The adversary is specifically attacking the `(v̂, p)` coupling:

- **Toxicity** measures `E[1−p | a=1]`. It only cares whether low-`p` interactions get accepted. The adversary makes that happen — inflated `v̂` ferries low-`p` adversaries past the governor at every θ. Toxicity AUROC climbs because the signal it depends on gets *louder* under evasion.
- **Saturation** measures `|Q| · √(αβ) / σ_p`. Inflation pushes low-`p` adversaries into the accepted set, dragging `E[p|a=1]` down and shrinking `|Q|`. The ratio collapses because the proxy is no longer informative about `p`. Saturation is the metric most damaged by exactly the attack it's supposed to flag.

And `baseline_harm` doesn't move. The population is unchanged; only the proxy is gamed. `E[p]` doesn't care.

## The breakage is the diagnostic

![Xiaohei holding two tuning forks across a "proxy gaming" crack — toxicity up on one side, saturation down on the other](figures/xiaohei-selection-geometry-05-tuning-forks.png)

You can read this as bad news for `selection_saturation`. But the more useful reading is that **the pair `(toxicity, saturation)` carries information neither metric does alone**:

| pattern across a sweep | interpretation |
|---|---|
| both flat near chance | governor not sorting, no adversarial pressure |
| both rising | adversarial pressure, governor handling it |
| **toxicity rising, saturation falling** | **proxy gaming — adversary inflating `v̂` past the governor's calibration** |
| toxicity flat, saturation high | governor at proxy ceiling, benign population |
| toxicity flat, saturation low | governor lazy, no adversarial pressure |

Row three is the load-bearing case. The metric pair *diverges in opposite directions* exactly when the adversary is succeeding at proxy gaming. This is the cause-3 fingerprint, visible from observable metric geometry without ground truth, without a calibration anchor, without an audit log. A dashboard panel that plots `Δ(toxicity_AUROC, saturation_AUROC)` over time would flag evasion before any of the existing detectors fire.

What we set out to find was a better adverse-selection metric. What we got was that, but also: a metric that breaks under proxy gaming, and a *companion* metric that doesn't, and a diagnostic constructed from the gap between them. The first AUROC sweep gave the wrong answer because the experiment was too easy. The second gave a clean picture but missed the most important case. The third made the picture invert — and the invariance pinned down the new signal.

## Reproduce it

```bash
pip install swarm-safety
python -m experiments.selection_geometry_auroc --seed 2026
```

54 000 cells in ~36 s, single-core. Full findings doc with per-θ breakdowns and limits is in [`selection-geometry-auroc-findings.md`](../research/selection-geometry-auroc-findings.md).

---

*Disclaimer: This post uses financial market concepts (adverse selection, market microstructure) as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
