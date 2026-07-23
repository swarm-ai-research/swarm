# Adaptive Arm 2 — CEM Pilot Findings (ρ=0.3, single seed)

**Date:** 2026-06-02
**Pre-registration:** [adaptive-agents-prereg.md](adaptive-agents-prereg.md)
**Status:** single-condition smoke; not the powered run

## TL;DR

- **The CEM trainer learned.** Mean elite payoff rose from 0.325 to
  0.750 over 10 iterations at ρ=0.3, seed=42.
- **The mechanism is mostly filtering, not quality improvement.**
  Accept rate collapsed from 70% to 8.5%; toxicity dropped from 0.32
  to 0.12, but the agent achieved this by **rejecting more**, not by
  **producing better**. This is the pre-reg's cause-2 (filtering)
  channel firing under ρ.
- **A real reward-specification ambiguity is surfaced.** "Mean
  payoff per accepted interaction" rewards pickiness; "total payoff
  over all attempted" would reward quality + volume. The prereg's
  language doesn't pin it down. The smoke shows it matters: under
  the current reward, the optimal CEM strategy is to be a high-
  pickiness gatekeeper.
- Cost: $0.00 (pure-Python CEM, 6 seconds wall-clock).

## Configuration

| Parameter | Value |
|---|---|
| ρ (initiator + counterparty externality internalization) | 0.3 |
| Seed | 42 |
| CEM population size | 30 |
| Elite fraction | 0.25 (n_elites = 7) |
| Iterations | 10 |
| Interactions per episode | 200 |
| σ floor (exploration) | 5% of parameter range |
| Policy parameters | 8 (means/spreads/Poisson-rates over observables + accept_threshold) |

## Learning curve

| iter | mean elite payoff | mean elite toxicity | mean elite accept rate |
|---|---|---|---|
| 0 | 0.325 | 0.322 | 0.697 |
| 9 | 0.750 | 0.119 | 0.085 |

Final policy episode (deterministic re-eval at the converged μ):

- n_accepted = 16 / 200 (8.0%)
- mean payoff = 0.750
- toxicity (E[1−p | accepted]) = 0.119

## What this tells us

### The trainer works

Δ payoff = +0.425 (+131%) in 10 iterations with 30 population × 200
interactions = 60k simulated interactions total. The CEM Gaussian
narrowed and the elite payoff rose monotonically.

### The mechanism is filtering, not improvement

The pre-reg identified three causes of a toxicity drop in an
adaptive condition:

1. Agents genuinely generate higher-quality interactions.
2. Agents reject more low-quality interactions. *(filtering)*
3. Agents game the proxy.

This pilot fires (2). The accept rate collapsed from ~70% to ~8.5%;
the policy learned to set a very high `accept_threshold` and let only
the cleanest interactions through. **Underlying generation quality
was not the variable that moved most.** The pre-reg said this would
be a real, separately-publishable outcome:

> Toxicity falls but only via filtering (cause 2): more modest but
> still informative — governance reshapes which interactions clear
> without improving the underlying population.

### A reward-specification ambiguity, surfaced by the smoke

The current `EpisodeReport.mean_payoff` is the average payoff over
**accepted** interactions. Under this reward:

- Be very picky → high mean per accepted → high reward.
- Generate higher quality → also high mean per accepted, but harder.

CEM found the easier path.

A different reward would push the agent towards channel (1):

| Reward | Optimizes |
|---|---|
| mean payoff per accepted (current) | pickiness; rewards channel (2) |
| sum payoff over all attempted | quality + volume; rewards channel (1) |
| mean payoff per attempted (mean × accept_rate) | balanced; both channels matter |

This is a **real design choice the pre-reg didn't pin down**. The
smoke surfaced it before we'd committed to a full ρ-grid run.

### Concrete recommendation

The follow-up powered run should report **both reward
specifications** for each (ρ, seed) so the participation-suppression
decomposition (per the prereg) lands cleanly. The cause-1-vs-cause-2
distinction is one of the load-bearing claims of the whole adaptive
arm.

## Honest limits

- **n=1 seed.** Five-seed power per the prereg is required before any
  comparison-to-static claim is made.
- **Single ρ.** The bend-the-vertical-collapse claim needs the full
  ρ ∈ {0, 0.1, 0.3, 0.5, 0.7, 1.0} grid.
- **No calibration anchor yet.** The judge anchor (rubric v3, with
  the agent_type-subset caveat) was not integrated. For this smoke,
  toxicity is measured against `p` (the simulation's latent), which
  is fine for showing CEM learns but is the *very signal* the
  adaptive agent would learn to game if let loose at scale.
- **Static-baseline comparison not yet plotted.** The original Table 4
  vertical-collapse comparison requires re-running the static
  ablations through the same `EpisodeReport` infrastructure. That's
  a follow-up.

## Followups

1. **Reward specification.** Pick one (mean-per-attempted is the
   most balanced) and pin it in the prereg before the powered run.
2. **Static baseline.** Wire the existing static-baseline (Table 4)
   into `run_episode` so adaptive-vs-static is a controlled
   comparison.
3. **5-seed × 6-ρ grid.** The pre-registered powered run for arm 2.
   Cost: pure Python, ~30 minutes wall time even sequentially.
4. **Calibration anchor integration.** Judge accepted interactions
   under rubric v3 on the `agent_type`-populated subset; surface
   the v_hat-vs-judge gap as a proxy-gaming detector.
5. **Adversarial probe.** The current ρ=0.3 result shows CEM finds a
   filtering strategy; an explicit adversarial CEM (reward = total
   payoff *plus* a malicious-utility component) would test the
   prereg's cause-3 (proxy gaming) scenario.

## Reproducibility

```bash
python -m experiments.adaptive_arm2_pilot --rho 0.3 --seed 42
```

Artifacts (gitignored):

- `runs/20260605T004624Z_adaptive_arm2_pilot_rho0.3_seed42/`
  - `training_report.json` — full trajectory + final policy
  - `iterations.csv` — learning curve
  - `config.json` — config + git rev
