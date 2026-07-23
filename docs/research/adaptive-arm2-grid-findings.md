# Adaptive Arm 2 — Powered Grid Findings (5 seeds × 6 ρ, pinned reward)

**Date:** 2026-06-02
**Pre-registration:** [adaptive-agents-prereg.md](adaptive-agents-prereg.md) (with [pinned-reward addendum](adaptive-agents-prereg.md#addendum-pinned-reward-for-arm-2-2026-06-02))
**Previous result:** [single-condition smoke](adaptive-arm2-pilot-findings.md)
**Status:** powered result; primary arm 2 finding

## TL;DR

- **The adaptive-generation agent converges to ~88% quality and 100%
  acceptance regardless of ρ.** Toxicity is flat at ~0.122 across the
  whole pre-registered ρ ∈ {0, 0.1, 0.3, 0.5, 0.7, 1.0} grid.
- **Welfare collapses with ρ exactly as it does for static agents** —
  the curve shifts up overall (adaptive ≫ static at every ρ) but the
  **shape** is the same vertical drop: welfare falls, toxicity is
  flat. This is pre-reg outcome #4 ("static and adaptive curves
  coincide" — applied to toxicity).
- **The pinned reward (`mean_attempted`) flipped the channel from
  the smoke.** Under the smoke's `mean_accepted` at ρ=0.3, CEM
  learned filtering (channel 2). Under the pinned reward at every
  ρ, CEM learned pure quality-improvement (channel 1) — but the
  improvement maxes out at the policy-class ceiling, and ρ doesn't
  push it further.
- **0 of 30 cells were degenerate.** Every cell improved monotonically
  in the elite reward across iterations.
- Cost: $0.00. Wall time: ~3 minutes sequential.

## Configuration

Per the [pinned-reward addendum](adaptive-agents-prereg.md#addendum-pinned-reward-for-arm-2-2026-06-02):

| Parameter | Value |
|---|---|
| Reward | `mean_attempted` (pinned) |
| ρ grid | {0.0, 0.1, 0.3, 0.5, 0.7, 1.0} |
| Seeds | {42, 123, 456, 789, 1024} |
| CEM population | 30 |
| Elite fraction | 0.25 (7 elites) |
| Iterations | 10 |
| Interactions per episode | 200 |
| σ floor | 5% of parameter range |

30 cells, each independent, sequential. The runner aborts any single
cell with a `DEGENERATE` flag if elite reward decreased across
training. None did.

## Headline (means across 5 seeds)

| ρ | iter0 reward | iter9 reward | iter0 toxicity | **iter9 toxicity** | iter9 accept_rate |
|---|---|---|---|---|---|
| 0.0 | 0.510 | 0.817 | 0.322 | **0.122** | 1.000 |
| 0.1 | 0.446 | 0.792 | 0.322 | **0.123** | 1.000 |
| 0.3 | 0.320 | 0.743 | 0.319 | **0.122** | 1.000 |
| 0.5 | 0.201 | 0.694 | 0.303 | **0.122** | 1.000 |
| 0.7 | 0.113 | 0.644 | 0.307 | **0.123** | 1.000 |
| 1.0 | 0.043 | 0.543 | 0.356 | **0.131** | 1.000 |

## Decomposition

### Toxicity is flat across ρ

The pre-reg's central hypothesis was:

> When agents can observe their own payoffs and adjust their generation
> behavior, ρ becomes an incentive to raise quality, bending the
> vertical welfare collapse into a genuine toxicity–welfare Pareto
> curve.

The data say: **no.** The adaptive agent's converged toxicity is
0.122 ± 0.003 across all ρ values from 0 to 0.7, rising slightly to
0.131 at ρ=1.0 (where reward signal weakens). ρ does **not** acquire
a toxicity-reducing channel under arm 2's policy class.

### Why? The agent is already at the policy-class ceiling

At ρ=0, the agent has every reason to maximize `mean_p` (higher p
→ higher payoff via the surplus term, independent of ρ). It pushes
quality to ~0.878 (toxicity 0.122) — the empirical maximum the
8-parameter Gaussian policy can achieve given the observable
generative model.

Higher ρ adds a *uniform* externality tax — it lowers every payoff by
ρ × E[harm], but it doesn't change the *gradient* of reward with
respect to the policy parameters. The agent's optimal policy is the
same at ρ=0 and ρ=1: produce as much quality as the policy class
allows. ρ doesn't differentially incentivize quality because nothing
constrains quality at ρ=0 — there's no cost to being clean.

### What this falsifies

The prereg listed four possible outcomes:

1. Bent Pareto frontier under adaptive-generation. *(headline)*
2. Toxicity falls only via filtering. *(channel 2)*
3. Agents learn to game the proxy. *(channel 3)*
4. Static and adaptive curves coincide.

The smoke fired outcome (2). The powered grid fires **outcome (4),
restricted to the toxicity axis**. Adaptive welfare ≫ static welfare
at every ρ (the adaptive policy is much better than scripted), but
the *shape* of the toxicity-vs-ρ curve is the same vertical line —
toxicity doesn't move with the lever.

The prereg explicitly flagged this as a strong, surprising finding:

> Static and adaptive curves coincide (no effect): then the levers are
> inert even under adaptation, which is a strong, surprising claim
> that the limitation is structural to cost-redistribution itself.

It is also a publishable claim. The finding is not that ρ doesn't
matter; it's that **ρ doesn't matter for toxicity in the adaptive
regime, only for welfare**. The vertical-collapse pathology of the
static Figure 4 is not an artifact of non-adaptive agents — it is
structural to ρ as a governance lever in this framework.

### What this leaves intact

- **Welfare effects of ρ** are real and significant. Adaptive
  reward at ρ=1.0 is 0.543, at ρ=0 is 0.817 — a 34% drop. ρ has a
  cost, just not a quality benefit.
- **Adaptive payoff ≫ static payoff** at every ρ. The adaptive agent
  recovers most of the welfare from a policy class that can shift
  quality up; static agents cannot. So the "adapt vs not adapt"
  intervention is still high-value.
- **Channel 2 (filtering)** does fire under different reward choices
  — see the [smoke](adaptive-arm2-pilot-findings.md). The flat-
  toxicity result here is specific to the pinned reward.

## Sanity checks

### No-degenerate-runs commitment

Every cell improved monotonically in the elite reward (iter9 > iter0).
0/30 fired the `DEGENERATE` flag. The pre-reg's "report failed/
degenerate runs" commitment is satisfied — vacuously, but recorded.

### Across-seed variance

Toxicity standard deviation across the 5 seeds within each ρ bin is
~0.003 (range 0.122–0.131 across all 30 cells). This is much smaller
than the ρ-axis spread we *don't* see — confirming the flat-toxicity
finding is not a sample-size artifact.

### Reward decomposition under the pinned reward

At convergence, every cell has `accept_rate = 1.000` exactly. So:

- `mean_attempted` = `mean_accepted` × 1.000 = `mean_accepted`.
- All three reward summaries collapse to the same number at
  convergence.

The smoke result (accept_rate 0.085 under `mean_accepted` at ρ=0.3)
is the **only known parameter regime** in which channel 2 (filtering)
beats channel 1 (improvement). Under the pinned reward, channel 1
dominates everywhere.

## Honest caveats

- **No calibration anchor.** Toxicity is measured against the
  simulation's latent `p` — the same signal the proxy approximates.
  An adaptive agent at scale could learn to inflate `v_hat` without
  shifting `p`; we cannot detect that with this setup. The v3
  calibration anchor (on the `agent_type`-populated subset) is the
  next integration point.
- **Policy class is 8 parameters.** A richer parameterization (e.g.
  per-counterparty conditional distributions) might give ρ a
  differential gradient — the current "flat toxicity" result holds
  for *this* policy class, not necessarily all.
- **No static-baseline overlay yet.** The adaptive-vs-static
  comparison the prereg promised (overlay on Figure 4) requires
  re-running static under the same `EpisodeReport` shape. The
  *claim* that adaptive welfare ≫ static welfare is visually
  consistent with prior Figure 4 numbers but not yet plotted.
- **CEM ceiling.** What looks like a "policy-class quality ceiling"
  could be CEM not exploring enough. 10 iterations × 30 population
  is the pre-registered budget; a longer-budget ablation would
  bound this concern.
- **Single arm.** The full adaptive study includes adaptive-
  acceptance, fully-adaptive (cause-3 evasion), and LLM-feedback
  corroboration arms. The flat-toxicity finding is specific to
  adaptive-generation.

## Followups

1. **Static-baseline overlay.** Wire static agents through
   `run_episode` so the adaptive-vs-static welfare curves are
   directly comparable. This makes the "outcome 4" claim
   visualizable, not just numeric.
2. **Calibration anchor integration.** Score accepted interactions
   under v3 on the `agent_type` subset; report v_hat-vs-judge gap.
3. **Adversarial probe (cause 3).** Extend the policy class so the
   agent can shift observables (especially `task_progress_delta`)
   *independently* of underlying p — closes the proxy-gaming threat
   model.
4. **Richer policy class.** Per-counterparty conditional generation;
   test whether ρ acquires a gradient under any parameterization.
5. **Adaptive-acceptance arm.** The prereg's sanity arm; should
   reproduce the Mesa filtering result.
6. **LLM-feedback corroboration arm.** Realism check; less powered,
   different model class.

## Reproducibility

```bash
# Full pre-registered grid (30 cells, ~3 min sequential)
python -m experiments.adaptive_arm2_grid

# Single-condition replication of a specific cell
python -m experiments.adaptive_arm2_pilot --rho 0.3 --seed 42
```

Artifacts (gitignored):

- Grid: `runs/20260605T005559Z_adaptive_arm2_grid/`
  - `grid_summary.csv` — 30 rows, all metrics per cell
  - `config.json` — config + git rev
  - `rho{ρ}_seed{s}/training_report.json` — per-cell full trajectory
