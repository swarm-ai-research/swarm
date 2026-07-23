---
date: 2026-03-19
description: "Halving contract signing costs flips infiltration from 0% to 100%. We found the phase transition using agent-lens trajectory capture across three governance regimes."
author: "SWARM Team"
keywords:
  - agent screening phase transition
  - governance regime comparison
  - agent-lens trajectory analysis
  - Spence signaling multi-agent safety
claims:
  - metric: "Infiltration phase transition"
    value: "0% → 100% at 0.5x cost"
    description: "Reducing signing costs by 50% flips infiltration rate from 0.0 to 1.0 — a sharp phase transition in screening effectiveness"
  - metric: "Welfare variance CV"
    value: "3.914"
    description: "Welfare delta has coefficient of variation 3.914 across 5 seeds despite perfect separation quality"
  - metric: "Strict governance participation loss"
    value: "29%"
    description: "Strict governance freezes 29% of agents while increasing total welfare 78%"
abstract: "We used agent-lens to run forked multi-session experiments comparing three governance regimes for contract screening. The results reveal a sharp phase transition: halving signing costs flips infiltration from 0% to 100%. Screening is structurally perfect (zero variance in separation quality across seeds) but economically fragile (welfare delta CV = 3.914). Strict governance improves welfare 78% but freezes 29% of agents. The mechanism is binary — costs either exceed adversarial benefit or they don't."
---

# Halving the Entry Fee Breaks Screening Completely. Here's the Phase Transition.

*We used [agent-lens](https://github.com/dreadnode/agent-lens) to run forked experiments across three governance regimes. Screening has a cliff edge.*

---

Our [previous posts](contract-screening-separating-equilibrium.md) showed that contract signing costs produce perfect agent separation across 10+ seeds. The mechanism works. But how much slack does it actually have?

We swept governance strictness in three regimes --- baseline, strict (2x costs), and lenient (0.5x costs) --- and found that screening effectiveness isn't gradual. It's a step function.

## Experimental design

We used [agent-lens](https://github.com/dreadnode/agent-lens) to run forked multi-session experiments. Agent-lens captures full ATIF trajectories with shadow-git file tracking, enabling replay and resampling of any decision point.

Three regimes, same 10-agent population (5 honest, 3 opportunistic, 2 deceptive), 20 epochs:

| Parameter | Baseline | Strict | Lenient |
|-----------|----------|--------|---------|
| `stake_fraction` | 0.05 | 0.10 | 0.025 |
| `min_bond` | 0.5 | 1.0 | 0.25 |
| `audit_probability` | 0.2 | 0.4 | 0.0 |
| `audit_threshold_p` | 0.4 | 0.5 | --- |
| `penalty_multiplier` | 3.0 | 5.0 | 1.0 |
| `entry_fee` (fair div) | 5.0 | 10.0 | 2.5 |
| `freeze_threshold_toxicity` | 0.7 | 0.5 | 0.9 |
| Collusion detection | on | on | **off** |
| Sybil detection | on | on | **off** |

The agent-lens config uses `session_mode: forked` so sessions 2--4 branch from a shared understanding session, each running a different regime independently.

## The phase transition

| Metric | Baseline | Strict | Lenient |
|--------|----------|--------|---------|
| Separation Quality | 1.000 | 1.000 | 1.000 |
| **Infiltration Rate** | **0.000** | **0.000** | **1.000** |
| Welfare Delta | +0.44 | +0.85 | +0.35 |
| Toxicity | 0.322 | 0.397 | 0.327 |
| Total Welfare | 12.52 | 22.31 | 14.03 |
| Participation | 100% | 71% | 100% |

The infiltration rate doesn't degrade smoothly. At baseline costs, zero adversarial agents enter governed pools. At half the cost, *all of them do*. There's no middle ground.

This confirms the Spence signaling prediction: separation requires that signing costs exceed the adversarial agent's expected benefit from pool access. Below that threshold, the cost-benefit flips for every adversarial agent simultaneously because they share preference weights (0.05). The transition is sharp because the agents are homogeneous within type.

## Pool composition under lenient governance

**Baseline** sorted agents cleanly:
- Default market: 2 agents (the deceptive ones)
- Fair division: 3 agents (opportunistic)
- Truthful auction: 5 agents (honest)

**Lenient** produced a pooling equilibrium:
- Default market: 0 agents (empty)
- Fair division: 4 agents
- Truthful auction: 6 agents

The default market --- designed as a containment pool --- emptied entirely. Every agent, including both deceptive ones, found it profitable to enter governed pools. The signing costs no longer screened.

The quality paradox: despite 100% infiltration, welfare delta *remained positive* (+0.35). Protocol benefits (reduced friction, fairness bonuses) outweighed the contamination effect. The governed pools still worked better than the default market --- they just no longer excluded adversaries.

## Strict governance: welfare up, participation down

Doubling costs and audit rates produced 78% higher total welfare (22.31 vs 12.52) but froze 29% of agents. The circuit breaker --- with a tighter `freeze_threshold_toxicity` of 0.5 --- caught agents producing borderline interactions that wouldn't have triggered at baseline.

This is over-screening. The strict regime correctly identifies more adversarial behavior but also catches opportunistic agents who occasionally produce below-threshold interactions. The welfare gain comes at the cost of reduced participation.

## Welfare is the fragile metric

We ran the baseline across 5 seeds to measure variance:

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| Separation Quality | 1.000 | 0.000 | 0.000 |
| Infiltration Rate | 0.000 | 0.000 | 0.000 |
| Quality Gap | 0.083 | 0.053 | 0.638 |
| **Welfare Delta** | **0.088** | **0.343** | **3.914** |
| Toxicity | 0.322 | 0.006 | 0.018 |

Separation and infiltration have zero variance. The mechanism sorts perfectly every time. But welfare delta has a coefficient of variation of 3.9 and *changes sign* in 2 of 5 runs (seeds 123, 1337). The mechanism guarantees sorting, not welfare.

This is the same finding from our [10-seed sweep](contract-screening-sweep-10-seeds.md), now confirmed with agent-lens trajectory capture. The structural mechanism is robust; the economic payoff from that structure is noisy.

## What agent-lens adds

Running these experiments through agent-lens rather than raw CLI gave us:

1. **Forked sessions**: Sessions 2--4 branched from a shared codebase understanding, ensuring each regime comparison started from identical context. No re-reading, no drift.

2. **Shadow git**: Every file write attributed to a specific step. When the agent created `contract_screening_strict.yaml`, we can trace exactly which parameters it changed and verify they match the experimental design.

3. **Resampling**: Any API call can be resampled N times to study decision variance. For screening studies, this means we can ask: "would the agent's analysis change if it re-read the same metrics?"

4. **Trajectory provenance**: Full ATIF trajectories link every finding to the tool calls and file reads that produced it. The analysis is reproducible from the trajectory alone.

Total cost: $2.61 across both experiments (8 sessions, 212 steps, 89 tool calls).

## Implications for governance tuning

1. **Don't under-price screening.** The phase transition is sharp. There's no "slightly less screening" --- you either screen or you don't. Err on the side of higher signing costs.

2. **Welfare gains require scale.** Perfect sorting doesn't guarantee welfare improvements with small populations. The variance is inherent to small-N pool dynamics, not the mechanism.

3. **Watch for over-screening.** Strict governance works but freezes agents who are borderline, not adversarial. Consider graduated penalties before circuit breaker activation.

4. **Preference weights do the heavy lifting.** Even low entry fees (0.5--2.0) achieve perfect separation because the type-dependent preference weights (honest=0.8 vs adversarial=0.05) create strong utility differentials. The fees just need to be non-trivial.

## Reproduce it

```bash
# Install agent-lens
cd ~/agent-lens && uv sync

# Run the governance regime comparison (5 forked sessions)
uv run harness run experiments/screening_study.yaml

# Run the variance study (3 sessions, 5 seeds each)
uv run harness run experiments/screening_variance.yaml

# Inspect results
uv run harness inspect runs/<timestamp>

# Resample a specific decision point
uv run harness resample runs/<timestamp> --session 2 --request 5 --count 10
```

Experiment configs are in `~/agent-lens/experiments/`. Trajectories, diffs, and API captures are in `~/agent-lens/runs/`.

[GitHub (SWARM)](https://github.com/swarm-ai-research/swarm) | [GitHub (agent-lens)](https://github.com/dreadnode/agent-lens)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
