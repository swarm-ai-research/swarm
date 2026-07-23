# Adaptive Agents Study — Pre-Registration

**Status:** pre-registered, not yet run
**Date:** 2026-06-01
**Depends on:** calibration study (provides the un-gameable quality anchor — §"Critical confound" below)

## Motivation

Static ablations (Table 4, Figure 4) showed `ρ` produces a vertical welfare
collapse: toxicity flat, welfare falling. `ρ` enters payoffs as a
post-generation cost term with no channel to interaction quality, so scripted
agents cannot respond to it. Reviewer [2]'s objection — that the static
non-result may be a property of the agents, not the framework — is the
question this study is designed to answer.

## Central hypothesis (mechanistic)

When agents can observe their own payoffs and adjust their **generation**
behavior (not just accept/reject), `ρ` and `τ` become incentives to raise
quality, bending the vertical collapse into a genuine toxicity–welfare Pareto
curve. The deliverable figure is the static `ρ`-curve and adaptive-generation
`ρ`-curve overlaid.

## Conditions (factorial)

### Adaptivity factor (rows)

1. **Static** — existing scripted agents. Replication anchor; must reproduce
   Table 4.
2. **Adaptive-acceptance** — agents optimize accept/reject only; generation
   distribution fixed.
3. **Adaptive-generation** — agents optimize their output quality
   distribution; this is the channel the static study lacks.
4. **Fully adaptive** — both, plus the option to game the proxy
   (lets evasion emerge if it is the optimal policy).

### Lever factor (columns)

- `ρ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}` (same 11-point grid as Figure 4 / Mesa Bridge static sweep).
- `τ` across the same values used statically.

### Population/seeds

Reuse static config exactly: seeds `{42, 123, 456, 789, 1024}`, same
populations.

## Implementation arms

### Route A — RL agents (primary, powered)

- Action space: generation parameters (distribution over `Δtask`, effort/rework,
  engagement) + accept/reject decision.
- Reward: realized payoff `π_i` under the active lever.
- Training: fixed budget, reported; learning-curve plot per condition;
  report convergence vs cycling.
- Defensibility: deterministic given seeds, runs locally, action space maps
  onto existing observable signals (no SWARM architecture change needed).

### Route B — LLM agents with feedback (corroboration arm, smaller)

- LLM-backed agents receive payoff history and governance state in-context,
  prompted to adjust strategy.
- Framed as corroboration of the RL primary arm, not powered to the same
  depth.

## Measurements

Per condition (overlaid on static):

- Toxicity `E[1−p | accepted]`, welfare, interaction volume, pass rate.
  Five seeds, mean ± std.
- **Toxicity–welfare frontier** per adaptivity level (deliverable figure).
- **Realized generation quality over epochs** — does the underlying `Δtask`
  distribution rise under `ρ`?
- **Participation-suppression decomposition** — split toxicity change into:
  (a) shift in accepted-quality, (b) shrinkage of the accepted set.
- **Evasion detection** (fully-adaptive condition only) — gap between `v̂`
  and held-out external/judge-rated quality.
- **Convergence diagnostics** — learning curves, policy stability across
  seeds.

## Critical confound

When toxicity drops in an adaptive condition, three causes are
indistinguishable without external signal:

1. Agents genuinely generate higher-quality interactions. *(governance worked)*
2. Agents reject more low-quality interactions. *(filtering — weaker)*
3. Agents game the proxy: `v̂` rises while true quality stays flat.
   *(governance defeated — the dangerous failure)*

The decomposition separates (1) from (2). Distinguishing (1) from (3) requires
an **external quality measure the agents cannot game**, supplied by the
calibration study on a sample of accepted interactions. This study must not
be interpreted without that anchor.

## Outcomes and what each buys

- **Bent Pareto frontier under adaptive-generation, verified externally
  (cause 1):** headline result. Static non-result is an artifact of
  non-adaptive agents. Promote to primary contribution.
- **Toxicity falls only via filtering (cause 2):** governance reshapes which
  interactions clear without improving the population. Report honestly;
  refines the framework, cautions "lower toxicity" can mean "less activity."
- **Agents game the proxy (cause 3):** observable-based governance is
  defeated by strategic agents; connects to proxy-gaming threat model (Q2).
  A finding, not a failure.
- **Static and adaptive curves coincide:** levers are inert even under
  adaptation; cost-redistribution is structurally insufficient. Publishable
  and reframes the paper.

## Pre-registered commitments

- Adaptation channels (1–4) are distinct conditions, reported separately.
- RL training budget is fixed in advance and reported.
- External-quality verification threshold is set before the adaptive runs.
- Failed/degenerate/non-converged runs are reported, not just converged ones.
- RL arm runs at the same five-seed depth as the static ablations.

## Order of operations

1. Calibration study (provides the external quality anchor).
2. Adaptive-acceptance arm (sanity / replicates Mesa result).
3. Adaptive-generation RL arm (the primary test).
4. Fully-adaptive RL arm (evasion detection).
5. LLM-feedback corroboration arm.

The adaptive study is **not** runnable in isolation — its central confound
is only resolvable with the calibration study's external signal.

---

## Addendum — Pinned reward for arm 2 (2026-06-02)

The original prereg said arm 2's reward is "realized payoff π_i under
whatever lever is active" without specifying whether to optimize:

- **Mean payoff per accepted** — rewards pickiness; surfaces channel (2).
- **Sum payoff over all attempted** — rewards quality + volume; surfaces channel (1).
- **Mean payoff per attempted** — `mean_per_accepted × accept_rate`; balanced.

The single-condition pilot (ρ=0.3, seed=42; see
[adaptive-arm2-pilot-findings.md](adaptive-arm2-pilot-findings.md))
demonstrated this matters: under "mean per accepted," CEM converged to
a pickiness strategy (accept rate collapsed 70% → 8.5%), and the
toxicity drop was entirely **channel (2) filtering**, not channel (1)
quality improvement. That made the cause-1-vs-2 distinction the
prereg promised impossible to test.

### Pin

For arm 2 (and any arm built on top of it), the **pinned reward is
`mean_attempted`** = total realized payoff ÷ number of attempted
interactions. Rejected interactions contribute 0 (they aren't
realized). Pickiness remains a legal strategy but doesn't pay off
unless rejected items would have contributed less than 0.

### Reporting commitment

All three reward summaries are recorded per iteration in
`CEMIterationReport` (`mean_elite_payoff_accepted`,
`mean_elite_payoff_attempted`, `mean_elite_sum_payoff`) regardless of
which one was the elite-selection criterion. The
**participation-suppression decomposition** the prereg requires is
then computable post-hoc on any run by inspecting whether
`mean_attempted` rose because of `mean_accepted` (channel 1, quality)
or because of `accept_rate` (channel 2, filtering — but at
mean_attempted's expense, since picking 1 in 10 items requires the
mean to rise ≥10× to win, not just any improvement).

### What this changes

- `swarm/adaptive/cem.py:PINNED_REWARD = "mean_attempted"`.
- `CEMConfig.reward` defaults to the pinned value; alternate rewards
  remain selectable for ablations.
- `experiments/adaptive_arm2_grid.py` runs the pre-registered grid
  (5 seeds × 6 ρ) under the pinned reward.
- The original prereg's "Order of operations" still holds — the
  calibration anchor is still required for cause-3 detection (proxy
  gaming), independent of which payoff function is used for elite
  selection.

### Synthesis

The full structural-inertness arc — static → adaptive-acceptance →
adaptive-generation (CEM) → fully-adaptive with the v3 anchor — and
its writeup live in [`arm2-paper-section.md`](arm2-paper-section.md).
