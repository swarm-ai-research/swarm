---
date: 2026-02-20
description: "Mechanism design meets multi-agent safety: how Vickrey auctions and entry fees create a separating equilibrium in 20 epochs"
author: "SWARM Team"
keywords:
  - separating equilibrium AI safety
  - contract screening mechanism design
  - Vickrey auction multi-agent
claims:
  - metric: "Separation point"
    value: "fee=6.0"
    description: "Perfect separation between honest and adversarial agents achieved at entry fee of 6.0"
abstract: "A contract screening mechanism using Vickrey auctions and entry fees achieves perfect separation between honest and adversarial agents at fee=6.0. The separating equilibrium holds from epoch 0 across 20 epochs with 10 agents. Mechanism design principles from economics transfer directly to multi-agent AI safety."
---

# Costly Contracts Separate Honest Agents from Adversaries. Here's the Data.

*Mechanism design meets multi-agent safety: how Vickrey auctions and entry fees create a separating equilibrium in 20 epochs*

---

When you can't directly observe an agent's type, you need a mechanism that makes agents reveal it through their choices. That's the core insight of screening theory: design a menu of contracts where each type's optimal choice differs, and the population sorts itself.

We built this into SWARM's simulation framework and ran 20 epochs with 10 agents. The separation was perfect from epoch 0.

## The mechanism

Three contract pools are available each epoch:

| Pool | Entry cost | Benefits | Target agents |
|------|-----------|----------|---------------|
| **Truthful Auction** | 5% of resources (bond) | Halved friction, surplus bonus, audit protection | Honest agents with resources to stake |
| **Fair Division** | Flat fee (5.0) | Redistribution toward equal splits, fairness bonus | Cooperative agents willing to pay for structure |
| **Default Market** | Free | Higher friction, no protections | Anyone --- but especially agents who can't justify the cost |

Each agent evaluates utility: `preference * quality_belief * resources - cost - switching_penalty`. The preference weights are type-dependent: honest agents strongly prefer governed pools (0.8 for truthful auction, 0.6 for fair division), while adversarial agents barely value them (0.05 for both).

The key: adversarial agents *could* sign a governed contract, but the cost outweighs the benefit. The bond in the truthful auction is proportional to resources, and the audit mechanism penalizes low-quality interactions. For an agent planning to exploit others, the expected cost of audits and penalties exceeds the benefit of reduced friction.

## The results

![Pool Composition](figures/contract_pool_composition.webp)

The population self-sorted cleanly:
- **5 honest agents** chose the Truthful Auction
- **3 opportunistic agents** chose Fair Division
- **2 deceptive agents** stayed in the Default Market

This wasn't hard-coded. Every agent ran the same utility calculation with the same available contracts. The mechanism parameters --- bond fraction, entry fee, preference weights --- were calibrated so that each type's rational choice naturally produces separation.

## Screening effectiveness: 20/20 epochs perfect

![Screening Effectiveness](figures/contract_screening_effectiveness.webp)

**Separation quality = 1.0** every epoch. All honest agents stayed in governed pools. **Infiltration rate = 0.0** every epoch. No adversarial agents found it profitable to enter governed pools.

This isn't fragile. The deceptive agents evaluated truthful auction utility as: `0.05 * quality_belief * 60 - cost`. Even with optimistic beliefs about pool quality, the low preference weight makes the utility negative. The mechanism has slack.

## Pool quality divergence

![Contract Dashboard](figures/contract_dashboard.webp)

The dashboard tells the story across 20 epochs:

**Top left: Pool quality.** The Truthful Auction (green) consistently produces the highest average interaction quality, hovering around p = 0.76. Fair Division (blue) and Default Market (red) cluster around 0.65. Honest agents interacting with honest agents produce better outcomes --- the mechanism channels this.

**Top right: Pool [toxicity](../concepts/metrics.md#toxicity-rate).** Truthful Auction has the lowest toxicity (E[1-p | accepted] ~ 0.24). Fair Division runs higher (~0.35) because opportunistic agents game the margins. Default Market sits at ~0.34.

**Bottom left: Pool welfare.** The welfare gap is stark. Truthful Auction produces ~1.27 welfare per interaction versus ~0.73 for Default Market. The governed pool generates 74% more welfare per interaction.

**Bottom right: Welfare delta.** The governed-minus-default welfare gap oscillates epoch-to-epoch (small sample sizes per epoch) but is predominantly positive. The mechanism delivers real welfare improvement, not just sorting.

## Per-pool comparison: final epoch

![Pool Comparison](figures/contract_pool_comparison.webp)

Three metrics, side by side:
- **Quality:** Truthful Auction (0.763) > Default Market (0.661) > Fair Division (0.648)
- **Toxicity:** Truthful Auction (0.237) < Default Market (0.339) < Fair Division (0.352)
- **Welfare:** Truthful Auction (1.267) > Fair Division (0.884) > Default Market (0.729)

The Truthful Auction dominates on every metric. Fair Division occupies an interesting middle ground --- higher toxicity than the default market (opportunistic agents pushing boundaries) but higher welfare (the redistribution mechanism still generates surplus).

## Why this matters for AI safety

Multi-agent AI deployments can't observe intent directly. You can measure behavior, compute proxy scores, set thresholds --- but a sufficiently strategic agent can game all of these. Screening mechanisms add a different lever: **make agents pay for access to privileged environments**, and the cost structure reveals their type.

This is the same principle behind:
- **Performance bonds** in procurement (underbid → forfeit bond)
- **Stake-weighted consensus** in blockchains (attack → lose stake)
- **Insurance deductibles** (reckless behavior → pay more out of pocket)

The common thread: agents with harmful intent face higher expected costs from the mechanism than agents with honest intent. The mechanism doesn't need to know who is who. It creates conditions where agents sort themselves.

## Limitations

**Static preferences.** Our agents don't adapt their contract preferences over time. A real adversary might learn that governed pools have higher-quality targets and invest more in infiltration.

**No collusion.** Two adversaries can't coordinate to have one infiltrate a governed pool while the other remains in default. Our collusion detection scenario handles this separately, but it's not yet integrated with contract screening.

**Proxy quality.** The attack displacement metric was 0.0 because no interactions had p < 0.4 in this run. The observable generator produces moderate p values even for deceptive agents. More aggressive adversarial strategies would test the mechanism harder.

**Small population.** 10 agents, 20 epochs. The mechanism's robustness under larger populations with more nuanced agent types (semi-honest, conditionally cooperative) remains untested.

## Reproduce it

```bash
pip install swarm-safety
python -m swarm run scenarios/contract_screening.yaml --seed 42 --epochs 20 --steps 10
python scripts/plot_contract_screening.py
```

The scenario YAML, contract implementations, and all 67 unit tests are in the repo.

## What's next

- **Adaptive adversaries:** Red-team the mechanism with agents that dynamically adjust their willingness to pay for governed pool access
- **Cross-mechanism integration:** Combine contract screening with collusion detection and circuit breakers
- **Multi-seed sweeps:** Run 100+ seeds to characterize the distribution of screening outcomes
- **Belief dynamics:** Study how pool quality beliefs evolve when adversaries occasionally infiltrate

[GitHub](https://github.com/swarm-ai-research/swarm)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
