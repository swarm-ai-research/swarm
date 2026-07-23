# Caveat: the soft AUROC/AUPRC = 1.00 figures are a ceiling of the generator, not a power estimate

**Date**: 2026-05-25
**Scope**: `swarm/detection/degradation.py` generative model; the headline soft-vs-binary results in [PR #450](https://github.com/swarm-ai-research/swarm/pull/450), the blog post `docs/blog/soft-vs-binary-detection.md`, and the 2D grid note `detection_2d_grid_key_cells.md`.

## TL;DR

The reported **soft AUROC = 1.000 (zero variance)** and **AUPRC = 1.000** are not measurements of how good soft detection is — they are a near-mathematical certainty baked into the synthetic generator. Three compounding structural choices force perfect separability. The **relative** soft-vs-binary gap and its **noise-dependence** are the robust findings; the absolute magnitudes (perfect soft, binary collapse to AUROC 0.50 / AUPRC ≈ base rate in clean-signal cells) are artifacts of parameter geometry. Quote the *direction and mechanism*, not the *1.00*.

## Why the soft ceiling is baked in

The per-agent toxicity detector scores each agent by its mean `E[1−p]` over the back-half evaluation window (`experiment.py` `_eval_window` → epochs `n//2 .. n`, ~50–100 interactions per agent). Three things then guarantee separation:

1. **Benign agents have zero heterogeneity.** Every benign agent is pinned to `benign_quality = 0.85` for all epochs (`degradation.py:_latent_quality`). There is no distribution of varied-but-healthy agents whose tail could overlap the degrading range.
2. **A large fixed mean gap.** Degrading agents drift from 0.85 toward `quality_floor = 0.66` — a 0.19 separation in `p`-space between class means.
3. **Averaging annihilates the noise.** Pooling dozens of interactions per agent collapses the within-agent standard error to ~0.01–0.02, so neither `proxy_noise` nor `quality_jitter` survives to the per-agent score.

### Measured separability (defaults, n=400 agents, seed 0)

| detector | benign | degrading | d′ | class ranges overlap? |
|---|---|---|---|---|
| soft toxicity | 0.136 ± 0.008 | 0.281 ± 0.025 | **7.83** | **no** (benign max 0.161 < deg min 0.226) |
| binary toxicity | 0.000 ± 0.000 | 0.020 ± 0.014 | 2.00 | yes |

A d′ of ~7.8 implies AUROC ≈ 1.0 to roughly eight decimal places. Perfect separation is structural, not empirical.

### Stress test — averaging is the lock

Soft AUROC (mean over 6 seeds, 120 agents, base rate 0.2), perturbing the two assumptions that should matter:

| setting | soft AUROC |
|---|---|
| defaults (floor 0.66, jitter 0.02) | 1.000 |
| `quality_jitter` 0.02 → **0.20** (10×) | 0.999 |
| `quality_floor` 0.66 → **0.80** (gap 0.19 → 0.05) | 1.000 |
| floor 0.80 **+** jitter 0.15 | 0.965 |
| floor 0.82 + jitter 0.20 | 0.959 |

Cranking per-epoch jitter 10× barely moves it (averaging eats it); shrinking the mean gap to 0.05 alone doesn't break it either. You only pull soft off the ceiling by *combining* a near-zero gap with large jitter — i.e. by manually removing the separability the model builds in.

## The binary "collapse" is equally mechanical

Binary toxicity only fires when `p` crosses below `τ* = 0.5`. The quality floor (0.66) sits 0.16 above the threshold, so:

- **Low `proxy_noise` (≈0.03):** almost no probability mass crosses 0.5 for *either* class → binary is blind → AUROC → 0.50, AUPRC → base rate (the "clean signal" cell where binary AUPRC ≈ 0.04).
- **High `proxy_noise` (≈0.15):** enough mass crosses for degrading agents → binary "recovers" to ~1.0.

The entire 2D `proxy_noise × quality_jitter` heatmap pattern is set by the geometry of `floor=0.66` vs `τ*=0.5` relative to observation noise — not by anything subtle about soft vs binary information content.

## Robust vs. artifact

**Robust (safe to claim):**
- Thresholding at 0.5 is *structurally* blind to degradation that stays above 0.5; the binary detector's benign score is literally 0.000 because nothing crosses. This is a property of the detector, not the parameters.
- Soft's relative advantage is largest in clean-signal regimes and shrinks as observation noise rises (qualitative shape of the 2D grid).

**Artifact (do not quote the magnitude):**
- Soft AUROC = 1.000 / AUPRC = 1.000.
- Binary AUROC 0.50 / AUPRC 0.04 in the clean-signal cell.

## What would make it a genuine power test

- Give benign agents a **quality distribution** (heterogeneity) whose upper-toxicity tail overlaps the degrading range — the single biggest fix.
- Make the gap **smaller and/or uncertain** (e.g. `quality_floor` sampled per agent, closer to `benign_quality`).
- **Score on fewer interactions** (or per-epoch rather than pooling the whole back-half) so within-agent noise actually matters.
- Tune parameters so soft AUROC lands in ~0.7–0.95 — the regime where the soft-vs-binary delta is informative rather than saturated.

## Reproduce

```python
# d′ and ranges (defaults)
from swarm.detection.degradation import StreamConfig, PopulationConfig, generate_population
from swarm.detection.detectors import MatchedDetectors
cfg = StreamConfig(); pop = PopulationConfig(n_agents=400, base_rate=0.5, stream=cfg)
streams = generate_population(pop, seed=0); det = MatchedDetectors()
n = cfg.n_epochs; es, ee = n // 2, n
# score each agent by det.soft_toxicity(s.window(es, ee)); split on s.is_degrading
```
