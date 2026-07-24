---
date: 2026-02-10
description: "The purity paradox in AI safety: populations with only 20% honest agents achieve 55% higher welfare than 100% honest populations. Why mixed agent diversity outperforms purity in multi-agent systems."
author: "SWARM Team"
keywords:
  - purity paradox AI safety
  - agent diversity welfare
  - mixed populations multi-agent
claims:
  - metric: "Welfare improvement (20% honest vs 100%)"
    value: "55% higher"
    description: "Populations with 20% honest agents achieve 55% higher welfare than 100% honest populations"
abstract: "Populations with only 20% honest agents achieve 55% higher welfare than 100% honest populations in SWARM simulations. The paradox is a measurement artifact: the total_welfare metric excludes harm externalities, rewarding interaction volume over quality. Under full externality internalization (rho >= 0.5), honesty dominates by 43%."
---

# The Purity Paradox: Why Mixed Agent Populations Outperform Pure Ones

Populations with only 20% honest agents achieve 55% higher welfare than 100% honest populations. This reflects how we measure welfare in [multi-agent systems](../research/papers.md), as formalized in [Soft-Label Governance for Distributional Safety in Multi-Agent Systems](https://arxiv.org/abs/2604.19752); see also [Distributional AGI Safety](https://arxiv.org/abs/2512.16856).

## The surprising finding

We swept honest agent proportion from 0% to 100% in 10% steps (10 agents, 30 epochs, 3 seeds each). Non-honest slots were filled 60/40 deceptive/opportunistic.

| Honest % | Total Welfare | Toxicity | Avg Payoff |
|----------|--------------|----------|------------|
| 0% | 727.5 | 0.370 | 0.408 |
| 10% | 657.8 | 0.367 | 0.413 |
| **20%** | **609.3** | 0.354 | 0.434 |
| 50% | 346.0 | 0.322 | 0.484 |
| **100%** | **391.6** | 0.275 | 0.560 |

Two things jump out:

1. **Welfare is monotonically decreasing with honesty.** 0% honest has the *highest* total welfare (727.5). This wasn't in the original paper, which only compared 20% vs 100%.

2. **Per-agent payoff tells the opposite story.** Honest agents individually earn more (0.560 vs 0.408). The paradox is an aggregate effect — mixed populations generate more interactions, producing more counted surplus.

## Why it happens

The welfare metric sums private payoffs over accepted interactions:

```
W = sum of (pi_a + pi_b) for accepted interactions
```

With externality internalization at rho = 0.1, **80% of the harm externality is excluded from welfare**. Mixed populations generate more interactions because deceptive and opportunistic agents accept more aggressively. More interactions = more counted surplus, even when the uncounted social cost is high.

This is the multi-agent equivalent of pollution-driven GDP growth. The economic activity is real. The externalities are also real but not measured.

## When the paradox breaks

We tested 21 parameter configurations. The paradox holds in 15/21 (71%) but breaks under specific conditions:

| Condition | Effect | Why |
|-----------|--------|-----|
| rho = 0.5 | Paradox disappears (-1%) | Tipping point: half of harm internalized |
| rho = 1.0 | Honesty dominates (+43%) | Full harm pricing; welfare = social surplus |
| All-deceptive mix | Honesty dominates (+78%) | Pure deception collapses trust entirely |
| No governance | Honesty dominates (+13%) | Without circuit breakers, mixed populations degrade |
| Symmetric stakes (s+ = s- = 1) | Honesty dominates (+14%) | No asymmetric upside from risk-taking |

The paradox **amplifies** under zero harm internalization (rho = 0.0, +21% stronger), extreme surplus asymmetry (s+ = 4, s- = 1, +27%), and low harm penalty (h = 0.5, +19%).

Reputation weight had zero influence across all tested values. The paradox is orthogonal to reputation.

## What this means

**The purity paradox stems from measurement choices.** The total_welfare metric excludes most harm externalities, rewarding interaction volume over interaction quality. Under social surplus accounting (full harm internalization), honesty dominates by 43%. The phenomenon is closely related to [adverse selection](../concepts/distributional-safety.md#adverse-selection) — mixed populations admit more low-quality interactions that inflate volume without improving outcomes.

**Policy implication: increase rho.** If the goal is to align private welfare with social welfare, the most direct lever is externality internalization. At rho >= 0.5, the paradox disappears. Governance design should focus on making agents bear the costs of harmful interactions.

**You cannot assume homogeneous alignment is optimal.** In systems where externalities are partially priced (which is most real systems), diversity itself provides structural value — but at a social cost the standard metrics don't capture.

## Reproduce it

```bash
pip install swarm-safety
python examples/reproduce_2602_00035.py      # Main reproduction
python examples/sensitivity_purity_paradox.py  # Full sensitivity analysis
```

Full methodology: [Purity Paradox Findings](../research/purity-paradox-findings.md)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
