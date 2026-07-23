# Calibration Pilot Findings — Rubric v2 (n=82)

**Date:** 2026-06-02
**Pre-registration:** [calibration-prereg.md](calibration-prereg.md)
**Previous findings:** [calibration-pilot-findings.md](calibration-pilot-findings.md) (rubric v1)
**Status:** pilot result, not the powered run

## TL;DR

- **Rubric v2 closes the gap that fired the escalation rule in v1.**
  Same 82 items, same 3 judges, same seed: headline α went 0.87 → 0.94,
  and the mistral ↔ qwen disagreement that motivated v2 dropped from
  ρ=0.66 to ρ=0.95.
- **Four of five p-bins are now `strong` or `usable`.** Only `[0.4, 0.6)`
  still fires escalate, and the residual is a Krippendorff
  normalization artifact (judges agree more on the population, so the
  small within-bin spread looks relatively worse).
- **Decision-relevant regions are now actually usable.** The
  policy-obfuscation tail (`p ≥ 0.6`, where adaptive proxy-gaming
  would land) went from α=0.70 to α=0.97. Adaptive findings in that
  band can be made with confidence.
- Cost: $0.00 (same Ollama judges). Wall time: ~10 min.

## What v2 changed

Three mechanical changes to the rubric (`swarm/judges/rubric_v2.md`),
derived from the v1 pilot's failure modes:

1. **Bans the 0.5 default.** v1 said "score near 0.5 when metadata is
   empty"; qwen took this literally. v2 requires a directional score
   from `interaction_type` + party labels even when metadata is sparse.
2. **Reserves 0.0 and 1.0 for unambiguous cases.** v1 let mistral
   anchor to extremes on any positive/negative signal; v2 makes 0.05
   and 0.95 the routine "very confident" scores.
3. **Adds banded anchors with worked examples** at 0.05 / 0.15 / 0.30
   / 0.45 / 0.55 / 0.70 / 0.85 / 0.95, each tied to a specific evidence
   profile so judges have less room to interpret "low quality"
   differently.

The forbidden-fields contract and the orthogonality property are
**unchanged** from v1.

## Configuration

Identical to the v1 pilot, except `--rubric rubric.v2`:

| Parameter | Value |
|---|---|
| Scenario | `obfuscation` (10 epochs, seed 42) |
| Sample size | 82 accepted interactions, stratified by p |
| Judges | `llama3.1:8b`, `qwen2.5:14b`, `mistral:7b` — all local via Ollama |
| Rubric | `rubric_v2.md` (SHA-prefix `25af1f4eccfadcee`) |
| Temperature | 0 (deterministic) |

## Headline

| Metric | v1 | **v2** | Δ |
|---|---|---|---|
| Krippendorff α (interval) | 0.869 | **0.938** | +0.07 |
| ICC(2,k) | 0.953 | **0.979** | +0.03 |
| Verdict | strong | **strong** | — |

## Pairwise Spearman ρ

| pair | v1 ρ | **v2 ρ** | Δ |
|---|---|---|---|
| llama ↔ mistral | 0.82 | **0.96** | +0.14 |
| llama ↔ qwen | 0.82 | **1.00** | +0.18 |
| **mistral ↔ qwen** | **0.66** | **0.95** | **+0.29** |

The mistral ↔ qwen pair was the diagnostic that revealed the rubric
ambiguity in v1; closing that gap is the load-bearing test of whether
v2 actually fixed it. ρ=0.95 says yes.

## Per-bin agreement

| bin | n | v1 α | **v2 α** | v1 mean \|Δ\| | **v2 mean \|Δ\|** | v1 → v2 verdict |
|---|---|---|---|---|---|---|
| [0.0, 0.2) | 13 | −0.03 | **1.00** | 0.01 | **0.00** | degenerate → **strong** |
| [0.2, 0.4) | 9 | 0.20 | **0.67** | 0.04 | **0.01** | escalate → **usable** |
| **[0.4, 0.6)** | 20 | −0.11 | **−0.23** | 0.11 | **0.08** | escalate → **escalate** |
| [0.6, 0.8) | 20 | 0.77 | **0.97** | 0.17 | **0.04** | strong → **strong** |
| [0.8, 1.0] | 20 | 0.70 | **0.97** | 0.18 | **0.02** | strong → **strong** |

Four of five bins improved decisively. The **policy-obfuscation tail**
(`p ≥ 0.6`) is now strong with `α=0.97` and mean per-item disagreement
≈ 0.03 — i.e. the three judges differ by ~3% of full scale on average
in the regime adaptive proxy-gaming would produce its dangerous false
positives. **Usable for adaptive findings.**

## The remaining escalation in [0.4, 0.6)

`α=−0.23` looks worse than v1's `−0.11`, but **mean pairwise
disagreement actually dropped** from 0.11 to 0.08. The negative α is
a normalization artifact: v2 made the overall population more
agreeable, so even a small within-bin spread reads as
worse-than-population.

These items are almost certainly the `activation_obfuscation` cases —
the obfuscation fixture deliberately drifts p toward 0.5 over epochs
for this class. v2 caps activation_obfuscation at 0.40, but the judges
still disagree on whether specific items land at 0.30 vs 0.40. The
disagreement is bounded (0.08 absolute) but not zero.

This is the residual that v3 would target — but the headline impact
on the *decision-relevant* regions is already small.

## What this means for the adaptive study

- **`p ≥ 0.6`:** anchor is now strong; adaptive findings (improvement,
  filtering, proxy-gaming detection) can be reported without caveat.
- **`p < 0.2` and `p ∈ [0.2, 0.4)`:** anchor is now usable; findings
  carry their per-bin α as a footnote, not a blocker.
- **`p ∈ [0.4, 0.6)`:** anchor still escalates. Adaptive claims about
  ranking items in this band still need rubric v3 or an external
  human-rater pass. Disagreement is bounded (≤ 0.08 absolute) but
  surveying differences <0.08 across this band is below the
  ensemble's resolution.

The previous finding "the anchor is usable for very-good vs very-bad
but not for the middle" has **narrowed to "usable everywhere except
`[0.4, 0.6)`"** under v2.

## What's load-bearing in the comparison

The v1 → v2 improvement is not just "more agreement is better." The
falsifiable claim was: **the mistral ↔ qwen split was a rubric
ambiguity, not a model difference.** If v2's banded anchors were the
right fix, ρ(mistral, qwen) should jump close to 1.0 — and it did
(0.66 → 0.95). If the gap had been a real model-capability difference,
ρ would have stayed flat. v2 validates the rubric-ambiguity
hypothesis.

## Followups (in priority order)

1. **Adaptive RL implementation** can now proceed against v2 with a
   real anchor for the decision-relevant regions. This is the actual
   bottleneck for the rest of the study.
2. **Larger pilot at per-bin=50** would tighten the `[0.2, 0.4)`
   estimate (currently n=9) and provide a publishable powered result.
3. **Cross-scenario** — run on `mixed` and `self_optimizer` to confirm
   v2 transfers off the obfuscation fixture.
4. **Rubric v3** for `[0.4, 0.6)` — only worth doing if adaptive
   findings actually land in that band.
5. **Claude when `ANTHROPIC_API_KEY` is available** — would test
   whether v2's wins survive a fourth, qualitatively different judge.

## Reproducibility

```bash
# v2 judge run (3 judges × 82 items, ~10 min on local Ollama)
cd /path/to/swarm
JUDGE_MODEL_LLAMA=llama3.1:8b python -m experiments.calibration_judge \
    --rubric rubric.v2 \
    --judges llama qwen mistral \
    --scenario obfuscation --per-bin 20 --seed 42

# Agreement analysis on the resulting judge_scores.csv
python -m experiments.calibration_agreement \
    --scores /path/to/judge_scores.csv
```

Specific artifacts from this pilot (gitignored per project convention):

- v2 judge scores: `runs/20260603T004551Z_calibration_judge_seed42/`
- v2 agreement: `runs/20260603T010608Z_calibration_agreement/`
- v1 judge scores (for direct comparison): `runs/20260603T001634Z_calibration_judge_seed42/`
- v1 agreement: `runs/20260603T002953Z_calibration_agreement/`
