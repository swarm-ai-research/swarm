---
date: 2026-06-23
description: "Our 10-seed confirmatory arm reported a d=2.75 long-horizon screening effect under tight governance. A 50-seed replication deflated it to d=0.38 (non-significant) — and surfaced a real, smaller effect we'd been overlooking. Here's how we caught our own false positive."
---

# Replication Deflation: How 50 Seeds Killed Our d=2.75 Screening Effect

*Our 10-seed confirmatory arm reported a d=2.75 long-horizon improvement under tight governance with maximum screening. A focused 50-seed replication deflated it to d=0.38 — non-significant under Bonferroni correction — while revealing a genuine moderate-governance effect of d=0.76 that we'd been looking right past. This is the story of a false positive we caught on ourselves, and why we'd rather publish the deflation than the headline.*

---

A Cohen's *d* of 2.75 is an enormous effect size. In most fields you almost never see one honestly — it's the kind of number that should make you suspicious, not excited. Ours came out of the [capability–safety frontier work](capability-safety-pareto-frontier.md): in the 10-seed confirmatory arm of our screening-strength dose-response sweep, turning screening to maximum strength under **tight** governance appeared to improve long-horizon planning capability by *d=2.75, p<0.001*.

We did the responsible-looking thing and reported it with a Bonferroni correction. It survived. And then, because the number was too good, we ran it again with five times the seeds.

It did not survive that.

## The original result

The confirmatory arm used **N=10 seeds** per configuration: 10 agents, 20% adversarial, long-horizon (multi-step planning) task, screening strength swept 0.0 → 1.0. At maximum screening under tight governance, the long-horizon capability gap versus baseline came out at:

> **d = 2.75, p < 0.001** (N=10, Bonferroni-corrected, α/12 = 0.004)

Ten seeds proves a mechanism can fire. It does not tell you whether the magnitude is real. With high-variance agent-based simulations, ten draws is a thin foundation for an effect size — the denominator of *d* (the pooled standard deviation) is itself a noisy estimate at N=10, and a downward fluctuation in σ inflates *d* just as effectively as a real shift in the mean. So we preregistered a focused replication at **N=50**.

## The replication

Same configuration, 5× the statistical power. Here is what the extra 40 seeds per cell did to the headline:

| Governance | N (original) | d (original) | N (replication) | d (replication) | p-value | Bonferroni-significant? |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tight** | 10 | **2.75** | 50 | **0.38** | 0.058 | **No** |
| **Moderate** | 10 | — | 50 | **0.76** | 0.0001 | **Yes** |
| **Light** | 10 | (ceiling) | 50 | 0.05 | 0.811 | No |

*Bonferroni threshold: α/12 = 0.004 (12 comparisons).*

The tight-governance effect collapsed from **d=2.75 to d=0.38** — a seven-fold deflation — and landed at p=0.058, which doesn't even clear raw α=0.05, let alone the Bonferroni line at 0.004. The d=2.75 was a phantom: sampling noise in a high-variance domain, dressed up as a discovery by a small denominator.

## The part that makes it worth publishing

If the story were only "our big number went away," it would be a cautionary footnote. But the same 50 seeds that killed the phantom **surfaced a real effect we'd been overlooking** — at *moderate* governance, not tight:

> **Moderate governance: d = 0.76, p = 0.0001 — Bonferroni-significant.**

That's a moderate, durable effect, dose-dependent (it climbs monotonically as screening strength goes 0.0 → 1.0), and — crucially — **governance-specific**. It shows up under moderate governance and is absent under tight. The mechanistic reading is clean: tight governance already constrains agents so heavily that long-horizon planning is bottlenecked regardless of who's screened, so relaxing the screening signal buys nothing. Moderate governance leaves enough slack for the information work of screening to actually pay off in sequential, dependency-laden tasks.

So the replication didn't just subtract a false positive. It moved the true effect from the wrong governance regime to the right one. The headline we *could* have published (tight governance + screening = huge gains) was wrong in both magnitude *and* location.

## Why N=10 lied

Effect size is `d = (μ_treatment − μ_baseline) / σ`. Two things go wrong at small N in a noisy domain:

1. **The numerator gets lucky.** With 10 draws, a couple of favorable endpoint measurements can pull the treatment mean up by chance.
2. **The denominator gets unlucky.** σ estimated from 10 samples is unstable; a sample that happens to be tight deflates σ, and a deflated denominator inflates *d* directly.

At N=50, both stabilize. The numerator's lucky runs average out against unlucky ones; the variance estimate converges toward the truth. The effect that remains is the effect that was always there — d=0.76 under moderate governance — and the one that evaporates was never there.

## What we changed in how we work

Three methodological commitments came out of this, and they're now load-bearing across the project:

1. **Confirmatory arms at N=10 are insufficient for magnitude claims in agent-based models.** They're fine for "does the mechanism exist"; they are not fine for "how big is it." High-variance simulators need replication before any effect size is published.
2. **Bonferroni correction earns its keep.** With 12 comparisons, the corrected threshold is α/12 = 0.004. The deflated tight-governance effect (p=0.058) fails it cleanly — and so would a lot of tempting near-significant results. The same discipline backs our other headline claims (calibration-drift welfare collapse, additive toxicity mechanisms); this episode is the validation that the discipline catches real false positives, not just hypothetical ones.
3. **Pre-register the replication, then believe it.** We specified N=50 and the stopping rule before running, so there was no peeking-driven temptation to stop at a flattering interim. When the number came back small, there was no wiggle room to rationalize it away.

## Open questions

- How many seeds *are* enough for stable long-horizon estimates in this simulator? N=50 was sufficient to separate signal from noise here, but we don't have a general rule.
- Does the moderate-governance effect (d=0.76) persist across adversarial fractions other than 20%?
- Can we build confidence intervals that account for the inflation pattern up front, so a d=2.75 at N=10 is reported with the skepticism it deserves *before* replication?

## Why we lead with this

The cleanest signal a research program can send about its own reliability is to show its work catching its own errors. We could have shipped "screening produces enormous long-horizon gains." Instead the honest artifact is smaller, more specific, and more trustworthy: a moderate effect, in the moderate-governance regime, that survived a 5× power increase — and a 10-seed phantom that didn't. The deflation *is* the finding.

---

*Source: `20260326_screening_strength_sweep` (confirmatory arm, N=10) and `20260328_long_horizon_screening_replication` (N=50). Promoted in the claims registry as `small-sample-screening-inflation`. Raw run directories are archived to [swarm-artifacts](https://github.com/swarm-ai-research/swarm-artifacts).*
