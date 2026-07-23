# Calibration Pilot Findings (rubric v1, n=82)

**Date:** 2026-06-02
**Pre-registration:** [calibration-prereg.md](calibration-prereg.md)
**Status:** pilot result, not the powered run

## TL;DR

- **Headline α = 0.87 is misleading.** Per-bin breakdown shows the
  ensemble is **strong on extremes, escalate on the middle** — the
  opposite of what the headline suggests.
- **Two of three judges are systematically biased in opposite
  directions** on middling cases. The 2-judge ensemble hid this; adding
  a third surfaced it. This is why the pre-reg asked for three.
- The pilot **fires the pre-registered escalation rule** for two of the
  five p-bins. Anchor is usable for distinguishing very-good from
  very-bad, **not** for ranking middle-quality items.
- Cost: $0.00 (all Ollama, local). Wall time: ~10–30 min.

## Configuration

| Parameter | Value |
|---|---|
| Scenario | `obfuscation` (10 epochs, seed 42) |
| Sample size | 82 accepted interactions, stratified by p across [0, 0.2, 0.4, 0.6, 0.8, 1.0] |
| Judges | `llama3.1:8b`, `qwen2.5:14b`, `mistral:7b` — all local via Ollama |
| Rubric | `rubric_v1.md` (SHA-prefix `254a10e42f42a8d9`) |
| Temperature | 0 (deterministic) |
| Pre-reg run | not yet — Claude blocked by missing `ANTHROPIC_API_KEY`; `mimo-v2.5-pro` (Gitlawb) is a reasoning model unsuited to one-shot judge calls; `glm-4.7-flash:q8_0` timed out |

## Headline

| Metric | Value | Pre-reg verdict |
|---|---|---|
| Krippendorff α (interval) | **0.869** | strong (≥ 0.7) |
| ICC(2,k) | 0.953 | strong |

## Pairwise Spearman ρ

| pair | ρ |
|---|---|
| llama ↔ mistral | 0.82 |
| llama ↔ qwen | 0.82 |
| **mistral ↔ qwen** | **0.66** ← weakest |

Llama sits between mistral and qwen — the latter two disagree more
with each other than either does with llama.

## Per-bin agreement (the real story)

| p_true bin | n | α (3-judge) | mean \|Δ\| | 2-judge α (llama+qwen) | verdict |
|---|---|---|---|---|---|
| [0.0, 0.2) | 13 | −0.03 | 0.01 | −0.04 | degenerate (no variance) |
| **[0.2, 0.4)** | 9 | **0.20** | 0.04 | 0.55 | **escalate** |
| **[0.4, 0.6)** | 20 | **−0.11** | 0.11 | 0.55 | **escalate** |
| [0.6, 0.8) | 20 | 0.77 | 0.17 | 0.84 | strong (barely) |
| [0.8, 1.0] | 20 | 0.70 | 0.18 | 0.64 | strong (barely) |

The escalation rule from the prereg — *α < 0.5 → add judge / stronger
model / human spot-check* — **fires for two of the five bins**. The
−0.11 in `[0.4, 0.6)` means within-unit disagreement *exceeds* the
marginal disagreement of the dataset, i.e. the judges agree LESS than
chance on those items.

## Mechanism

Per-judge score distribution on extreme cases:

| case | llama | qwen | mistral |
|---|---|---|---|
| blatant trade | 0.00 | 0.00 | 0.00 |
| honest interaction | 0.90 | 0.50 | 1.00 |

- **mistral** is the most confident — bimodal (0.0 or 1.0).
- **qwen** is the most conservative — defaults to 0.5 when metadata is sparse.
- **llama** is intermediate.

On extreme cases all three agree. On middling cases mistral and qwen
point in opposite directions, with llama floating between them — which
is precisely the variance pattern that produces high *headline* α (the
grand mean is well-defined) but low *per-bin* α (within-bin spread
dominates).

## What the 2-judge ensemble hid

| Bin | 2-judge α | 3-judge α | Δ |
|---|---|---|---|
| [0.2, 0.4) | 0.55 (usable) | 0.20 (escalate) | −0.35 |
| [0.4, 0.6) | 0.55 (usable) | −0.11 (escalate) | −0.66 |

The 2-judge ensemble (llama + qwen) happened to produce respectable
middle-bin α because llama and qwen tend to land on the same side of
0.5 on uncertain items. Adding mistral — which goes all the way to 0 or
1 on those same items — revealed the disagreement. **This is the
value the pre-reg's three-judge requirement was buying.**

## What this means for the adaptive study

The pre-reg specifies that the calibration anchor distinguishes:

1. agents-improved-quality (proxy and judge both rise),
2. agents-filtered-more (proxy and judge unchanged but volume falls),
3. agents-gamed-the-proxy (proxy rises, judge does not).

Applied to this pilot:

- **(1) and (3) are detectable** in the high-p tail (`p ≥ 0.6`) — α is
  strong there, and that's the regime where proxy-gaming false
  positives would land.
- **(2) is detectable** anywhere — filtering shows up as volume not as
  ranking.
- **Ranking-claim findings in the middle range** (e.g. "the adaptive
  agent shifted accepted-interaction quality from 0.4 to 0.5") **cannot
  be made with confidence on this anchor** — the judges don't agree on
  what the middle range *means*.

This is a real constraint, not a fatal flaw. Adaptive-study findings
in the middle band require either rubric v2 (next section) or an
external human-rater pass.

## Followups

1. **Rubric v2** — tighten the middle-range instructions. The
   conservative-vs-bimodal split is a rubric ambiguity, not a model
   bug. Likely change: *"do not default to 0.5 when metadata is sparse;
   pick a directional score and lean on `interaction_type` +
   `agent_type` for the tie-break."* Bumps the version per the freeze
   rule.
2. **Larger pilot at rubric v1** — n=82 has wide CIs on the per-bin
   estimates (n=9 in `[0.2, 0.4)`). A run at per-bin=50 would tighten
   them.
3. **Cross-scenario** — run the same pilot on `mixed` and
   `self_optimizer` scenarios. The obfuscation fixture is
   metadata-heavy; results may differ on scenarios with sparse
   metadata.
4. **Claude when available** — set `ANTHROPIC_API_KEY` and rerun. The
   conservative-vs-bimodal axis may collapse or get richer depending
   on where Claude lands.

## Reproducibility

The pilot used three local Ollama judges (`llama3.1:8b`, `qwen2.5:14b`,
`mistral:7b`). The committed `JUDGE_SPECS` in
`experiments/calibration_judge.py` only ships the `llama` entry by
default, so reproducing requires either extending `JUDGE_SPECS` for
your run or invoking `LLMJudge` directly. Sketch of the
`JUDGE_SPECS` extension (drop into a local edit of
`experiments/calibration_judge.py`):

```python
JUDGE_SPECS.update({
    "qwen": {"provider": "ollama", "model": "qwen2.5:14b"},
    "mistral": {"provider": "ollama", "model": "mistral:7b"},
})
```

Then, from the repository root:

```bash
# Judge run (3 judges × 82 items, ~10 min on local Ollama)
JUDGE_MODEL_LLAMA=llama3.1:8b python -m experiments.calibration_judge \
    --judges llama qwen mistral \
    --scenario obfuscation --per-bin 20 --seed 42

# Agreement analysis on the resulting judge_scores.csv
python -m experiments.calibration_agreement \
    --scores runs/<ts>_calibration_judge_seed42/judge_scores.csv
```

Run artifacts (per-row scores, per-bin CSV, config + git rev) are in
`runs/` (gitignored). The exact runs from this pilot:

- Judge scores: `runs/20260603T001634Z_calibration_judge_seed42/`
- Agreement: `runs/20260603T002953Z_calibration_agreement/`
