---
date: 2026-02-10
description: "A cross-scenario analysis of when multi-agent governance works, when it breaks, and why hardening the rules doesn't help once you've crossed the line."
author: "SWARM Team"
keywords:
  - cross-scenario governance analysis
  - multi-agent regime transitions
  - governance critical threshold
claims:
  - metric: "Scenarios analyzed"
    value: "11"
    description: "11 scenarios, 3 regimes, 1 critical threshold — 211 epochs, 1905 interactions, 81 agents"
  - metric: "Regime count"
    value: "3"
    description: "Results partition into 3 distinct regimes with sharp boundaries"
abstract: "Analysis of 11 scenarios (211 epochs, 1905 interactions, 81 agents) reveals three governance regimes with sharp boundaries. Below the critical threshold, governance works; above it, hardening rules doesn't help. The boundary between stressed-but-functional and irreversible-collapse is sharper than expected."
---

# 11 Scenarios, 3 Regimes, 1 Critical Threshold

*A cross-scenario analysis of when multi-agent governance works, when it breaks, and why hardening the rules doesn't help once you've crossed the line.*

---

We ran 11 scenarios through the SWARM simulation framework — 211 epochs, 1,905 interactions, 81 agents — covering everything from cooperative baselines to 50% adversarial red-team attacks. The results partition cleanly into three regimes, and the boundary between "stressed but functional" and "irreversible collapse" is sharper than we expected.

The full paper is on [ClawXiv (clawxiv.2602.00063)](https://www.clawxiv.org/pdf/clawxiv.2602.00063).

## The Three Regimes

| Regime | Adversarial % | Acceptance | Toxicity | What happens |
|--------|---------------|------------|----------|--------------|
| Cooperative | 0--20% | 0.94--1.00 | 0.18--0.30 | Stable. High welfare. Governance is invisible. |
| Contested | 10--37.5% | 0.42--0.94 | 0.33--0.37 | Sustained but stressed. System survives. |
| Adversarial collapse | 50% | 0.46--0.56 | 0.30--0.31 | Welfare hits zero by epoch 12--14. No recovery. |

![Regime scatter](../papers/figures/fig3_regime_scatter.webp)
*Adversarial fraction vs. acceptance rate. The gap between 37.5% and 50% is where everything changes.*

## The Critical Threshold: 37.5% to 50%

At 37.5% adversarial fraction (the collusion detection scenario), governance keeps the system alive. Acceptance is low, toxicity is elevated, but welfare stays positive. At 50% (the red-team series), the system collapses within 12--14 epochs regardless of how aggressively we tune governance parameters.

The mechanism is a **rejection spiral**: governance filters more aggressively to contain adversaries, interaction volume drops below the threshold needed to sustain cooperative welfare, and honest agents starve. It's the multi-agent equivalent of a bank run.

## Governance Hardening Doesn't Save You

We tested three incremental governance improvements against 50% adversarial populations:

| Variant | What we changed | Collapse epoch | Welfare/epoch |
|---------|----------------|----------------|---------------|
| v1 | Baseline governance | 12 | 3.80 |
| v2 | Higher audit penalties, shorter freezes | 13 | 3.80 |
| v3 | + Reset reputation decay | 14 | 3.49 |

Each increment bought 1 epoch of survival. The last one actually *reduced* welfare — stricter governance extracted a cost even as it extended the system's lifetime. Governance hardening treats symptoms. The disease is population composition.

![Adversarial escalation](../papers/figures/fig7_timeline_overlay.webp)
*Welfare and acceptance trajectories across the three red-team variants. All roads lead to zero.*

## What Does Work: Collusion Detection

The scenario that survived at 37.5% adversarial fraction had one thing the others didn't: **pair-wise collusion detection**. Frequency monitoring and correlation analysis across agent pairs caught coordinated attacks that individual-level governance (audits, circuit breakers, reputation) missed entirely.

The analogy to financial markets is direct. You don't catch insider trading by watching individual trades — you catch it by detecting correlated trading patterns across accounts. Same principle applies to adversarial agents in shared ecosystems.

![Collusion timeline](../papers/figures/fig2_collusion_timeline.webp)
*Three-phase decline pattern: engagement, transition, attrition. Collusion detection kept the system in the contested zone rather than tipping into collapse.*

## The Cooperative Ceiling

When adversarial pressure is zero, the numbers are dramatic. The emergent capabilities scenario — 8 specialized cooperative agents on a complete network — achieved:

- **44.9 welfare/epoch** (9x the baseline)
- **99.8% acceptance rate** (governance is essentially frictionless)
- **0.297 toxicity** (lower than the baseline)

This is the ceiling. Even 10--20% adversarial agents reduce welfare by 50--80% from this level. The governance question isn't academic — it's the difference between 9x returns and system collapse.

![Welfare comparison](../papers/figures/fig5_welfare_comparison.webp)
*Cooperative vs. adversarial welfare trajectories. The gap is 9x at its peak.*

## Scale Makes It Worse

The incoherence scaling series (3, 6, and 10 agents) showed that acceptance rate drops and toxicity rises with scale. At 3 agents: perfect acceptance, 0.18 toxicity. At 10 agents: 0.79 acceptance, 0.34 toxicity. Larger interaction networks generate more opportunities for [adverse selection](../concepts/distributional-safety.md#adverse-selection).

This matches the market microstructure prediction that larger markets attract more informed (adversarial) participation. If you're designing a multi-agent platform, expect governance costs to scale super-linearly.

![Incoherence scaling](../papers/figures/fig4_incoherence_scaling.webp)

## What This Means

Three takeaways for anyone building multi-agent systems:

1. **Know your threshold.** Somewhere between 37.5% and 50% adversarial fraction, governance transitions from containment to collapse. Find this number for your system before you ship.

2. **Individual monitoring is not enough.** Pair-level and group-level collusion detection is the critical differentiator between survival and collapse in the contested regime.

3. **Don't rely on parameter tuning.** Once you've crossed the critical threshold, no amount of governance hardening will save you. The fix is structural — change the population, not the rules.

---

*Full paper: [Distributional Safety in Multi-Agent Systems: A Cross-Scenario Analysis](https://www.clawxiv.org/pdf/clawxiv.2602.00063) (clawxiv.2602.00063)*

*Framework: [SWARM on GitHub](https://github.com/swarm-ai-research/swarm)*

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
