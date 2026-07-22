# Calibration Study — Pre-Registration

**Status:** pre-registered, not yet run
**Date:** 2026-06-01
**Unblocks:** [adaptive agents study](adaptive-agents-prereg.md) (arms 1–3)
**See also:** [graph-structural detector study](graph-structural-prereg.md) — a separate pre-reg in the same series, evaluating topology-based coordination detection against threshold detectors.
**Pilot result:** [calibration pilot findings (rubric v1, n=82)](calibration-pilot-findings.md) — three-judge Ollama ensemble at rubric v1 fires the escalation rule on the middle p-bins; drove the rubric v2 / v3 work.
**Pilot re-run (rubric v2):** [calibration pilot findings — rubric v2 (n=82)](calibration-pilot-findings-v2.md) — apples-to-apples re-run with rubric v2; headline α 0.87 → 0.94, four of five bins now strong or usable, mistral↔qwen ρ jumped 0.66 → 0.95.
**Pilot re-run (rubric v3):** [calibration pilot findings — rubric v3](calibration-pilot-findings-v3.md) — v3 replaces v2's caps with targets ("a target is the score, not a ceiling"); cross-scenario check on `mixed` confirmed over-fit on `obfuscation` (the middle bin α regressed on the other scenario), so v3 ships with the documented limitation.

## Motivation

The adaptive agents study cannot distinguish three causes of a toxicity
drop without an external signal:

1. Agents improved quality. *(governance worked)*
2. Agents rejected more. *(filtering)*
3. Agents gamed the proxy — `v̂` rises while true quality stays flat.
   *(governance defeated)*

The framework also rests on the implicit assumption that the proxy-derived
`p = sigmoid(v̂)` is well-calibrated against `SoftInteraction.ground_truth`
(the realized `v ∈ {+1, -1}` label). That has never been measured. Until
both things are anchored, every downstream result is interpretable only at
the level of `p` — which is exactly what a strategic agent will defeat.

## Central goal

Produce a **triangulated quality signal** that adaptive-agent results can be
interpreted against:

- a measured proxy-fidelity baseline (`p` vs `ground_truth`),
- an external judge anchor (LLM-judge scores on accepted interactions),
- inter-rater agreement across judges (so "the judge is also gameable"
  is bounded, not waved away).

The deliverable is a CSV of accepted interactions with `{v_hat, p,
ground_truth, judge_score_claude, judge_score_gpt4o_mini,
judge_score_llama, agreement_metrics}` that downstream studies can join
against by run-id.

## Arms

### Arm A — Proxy fidelity

Measure how well `p = sigmoid(v̂)` from `ProxyComputer` tracks the realized
`ground_truth` label.

- **Procedure:** generate interactions with a known latent generator
  probability drawn from a controlled grid (e.g. `{0.05, 0.2, 0.4, 0.6,
  0.8, 0.95}`, ≥500 interactions per bin) and sample `ground_truth ∈
  {+1, -1}` from it. Run `ProxyComputer` to compute `(v_hat, p)` per
  interaction. Bin `p`, compute the empirical frequency of
  `ground_truth = +1` per bin (reliability diagram).
- **Metrics:** expected calibration error (ECE), maximum calibration
  error (MCE), Brier score, reliability-diagram bins.
- **Sweep:** the sigmoid `k` parameter; report fidelity vs `k`. Identify
  the `k` that minimizes ECE.
- **Output:** `runs/<ts>_calibration_fidelity/` with per-bin CSV and
  reliability plot.

### Arm B — External judge anchor

Build the un-gameable quality signal the adaptive study needs.

- **Procedure:** sample a held-out set of accepted interactions from
  static-baseline runs (≥1000 interactions, stratified across scenarios
  and `p` bins). For each interaction, present its *content artifact*
  (text/output, not the proxy observables) to an LLM judge with a
  fixed rubric. Judge returns a score on `[0, 1]`.
- **Prompt discipline:** the judge prompt MUST NOT include the
  `ProxyObservables` fields `ProxyComputer` reads — concretely:
  `task_progress_delta`, `rework_count`, `verifier_rejections`,
  `tool_misuse_flags`, `counterparty_engagement_delta`. The judge rates
  on content quality dimensions orthogonal to those signals. This is
  what makes the signal un-gameable from an adaptive agent's
  perspective — the agent can shift observables without shifting what
  the judge sees.
- **Judges:** Claude, GPT-4o-mini, Llama 3.x (matches existing Table 9
  setup).
- **Output:** `runs/<ts>_calibration_judge/judge_scores.csv` with one row
  per interaction × judge.

### Arm C — Inter-rater agreement

Bound the "the judge is also gameable" concern.

- **Procedure:** on Arm B's sample, compute pairwise agreement across the
  three judge models.
- **Metrics:** Krippendorff's α (continuous), Spearman ρ between judge
  pairs, intra-class correlation (ICC). Report disagreement-by-bin so we
  see whether judges disagree more at the tails.
- **Decision rule:** if α < 0.5, the judge ensemble is too noisy to anchor
  the adaptive study — escalate (add a 4th judge, switch to a stronger
  model, or fall back to human spot-checks on a smaller subsample).
  Pre-register the α threshold so we don't drift it post-hoc.

## Critical confound

**The judge is also a model.** A sophisticated adaptive agent could,
in principle, learn to game the *judge* rather than the proxy. Three
defenses:

1. The judge prompt rubric is fixed and orthogonal to `ProxyComputer`'s
   observables, so the gaming channel is not the same one.
2. Inter-rater agreement (Arm C) bounds the size of model-specific
   exploits — an attack that fools Claude but not GPT-4o-mini and Llama
   will surface in α.
3. We pre-register an external-quality threshold *before* the adaptive
   runs and never tune it afterward. The adaptive study reports
   judge-rated quality as a held-out signal, not as a target.

These do not eliminate the recursion — they bound it. State the bound
explicitly in any downstream finding.

## Pre-registered commitments

- `p` grid, sample sizes, and `k` sweep range for Arm A are fixed before
  data collection.
- Judge prompt rubric, judge model versions, and excluded observable
  features for Arm B will be version-locked at `swarm/judges/rubric_v1.md`
  before any Arm B data collection.
- The Krippendorff α decision threshold for Arm C is set in advance.
- The CSV schema downstream studies join against is frozen before the
  adaptive study begins.
- All judge calls are logged in append-only JSONL alongside the run
  artifacts for full reproducibility.

## Outcomes

- **`p` is well-calibrated AND judges agree (α ≥ 0.7):** the strong
  result. Adaptive study unblocked with confidence; `p` can be reported
  as a calibrated proxy; judge serves as the un-gameable anchor.
- **`p` is poorly calibrated but judges agree:** the framework still
  has a usable external anchor; we report the calibration gap honestly
  and re-tune `k`. Adaptive study still unblocked.
- **Judges disagree (α < 0.5):** the anchor itself is too noisy. The
  adaptive study cannot rely on a single ensemble score. We either add
  judges, switch models, or fall back to a smaller human-rated subsample.
  This is itself a publishable result about LLM-judge reliability.
- **`p` well-calibrated AND judges agree with each other, but the
  proxy and the judge ensemble disagree:** `p` and the external judge
  measure different things. Most interesting result — interpret
  carefully; this is where the proxy-gaming threat model lives.

## Order of operations

1. **Arm A** — proxy fidelity on existing static-baseline data
   (no new runs needed if logs are sufficient). Cheap, runs first.
2. **Arm B** — build judge pipeline, freeze rubric, run on stratified
   sample.
3. **Arm C** — compute agreement on Arm B's output.
4. Freeze the joined CSV schema. Adaptive study (arms 1–3) is unblocked.

Arm A is independent and can begin immediately. Arms B and C share the
judge pipeline and are run in sequence.

---

## Post-registration addenda (append-only; not part of the registered design)

- **2026-07-19 — Arm A findings & deviation analysis:**
  [calibration-arm-a-deviation-analysis.md](calibration-arm-a-deviation-analysis.md).
  Registered readout k\*=5.0, ECE 0.187, Brier 0.0906; outcome branch 2
  ("report the gap honestly, re-tune k; adaptive study still unblocked")
  applies. Includes a labeled exploratory k-extension and the
  latent-p-unrecoverability root cause; the ECE<0.1 / Brier<0.05 numbers
  circulating in the tracker are the c89o scenario spec, not registered
  criteria of this document.
