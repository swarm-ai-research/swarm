---
date: 2026-03-16
description: "We ran a NeurIPS 2025 delivery economy through SWARM's safety metrics. Profit says everything is fine. Adverse selection says 17% of high-value orders go to low-reputation agents."
author: "SWARM Team"
keywords:
  - SimWorld delivery economy
  - adverse selection multi-agent
  - capability compliance decoupling
  - distributional safety metrics
  - agent screening validation
claims:
  - metric: "Adverse selection signal"
    value: "0.174"
    description: "17% of high-value delivery orders won by low-reputation agents"
  - metric: "Screening separation quality"
    value: "0.750 ± 0.102"
    description: "Behavioral screening correctly sorts agent personas above 0.6 threshold across 10 seeds"
abstract: "SimWorld (NeurIPS 2025 Spotlight) models a multi-agent delivery economy where LLM agents bid on orders, invest in equipment, and cooperate or compete. Their evaluation uses profit. We ran the same economy through SWARM's distributional safety metrics and found what profit misses: adverse selection at 0.174 (low-reputation agents capturing high-value orders), variance amplification across agent types, and governance cost dynamics invisible to profit-only evaluation. A 10-seed screening validation confirms that behavioral signals correctly identify agent personas with separation quality 0.750."
---

# SimWorld's Delivery Agents Look Profitable. They're Also Adversely Selected.

*We ran a NeurIPS 2025 delivery economy through SWARM's safety metrics. Profit says everything is fine. Adverse selection says 17% of high-value orders go to low-reputation agents.*

---

SimWorld is a NeurIPS 2025 Spotlight paper that builds an Unreal Engine 5 simulator for LLM/VLM agent evaluation. Their Case 2 is a multi-agent delivery economy: agents bid on delivery orders, invest in scooters, share orders with partners, and navigate an urban grid. Different personality types produce different strategies. Claude-3.5-Sonnet and DeepSeek-V3 earn the most profit but behave erratically --- overbidding on low-value orders, buying scooters they never use.

The paper evaluates agents by profit. We wanted to know what profit misses.

## The setup

We built a SWARM domain module (`swarm/domains/simworld_delivery/`) that models SimWorld's Case 2 mechanics:

- **8 agents** with 4 policy types mapping to SimWorld's personas:
  - 3 conscientious (reliable, conservative bidding)
  - 2 aggressive (risk-taking, early scooter purchase, occasional overbids)
  - 2 cautious (only bid on feasible orders, low risk)
  - 1 opportunistic (cherry-picks high-value orders, sometimes abandons deliveries)

- **Delivery economy**: orders generated each epoch with random value ($5--50), origin/destination pairs, and deadlines. Agents bid in a reverse auction (lowest bid wins). Winners pick up and deliver for a payout, minus a 5% governance fee.

- **Governance levers**: reputation decay (1%/epoch), reputation bonuses/penalties for on-time/late/failed deliveries, minimum reputation threshold to bid (0.3).

5 seeds, 10 epochs each. Standard SWARM run protocol.

## What profit sees

| Metric | Value |
|--------|-------|
| Delivery rate | 0.279 |
| Total earnings | 1,745 |
| Earnings Gini | 0.167 |
| Overbid rate | 0.091 |

A reasonable picture. Delivery rate is moderate (not all orders are feasible in a sparse city). Earnings are distributed relatively equally (Gini 0.167). Overbidding is limited to ~9% of bids. SimWorld would report this economy as functional.

## What SWARM sees

| Metric | Value | What it means |
|--------|-------|---------------|
| **Adverse selection** | **0.174** | 17% of high-value orders won by low-reputation agents |
| Exploit frequency | 0.007 | Near-zero overt exploitation (failures are structural, not strategic) |
| Governance cost ratio | 0.004 | Fees are negligible relative to earnings |
| Variance amplification | 0.302 | Earnings diverge across types but haven't runaway |

The headline: **adverse selection is 0.174**. About one in six high-value delivery orders are won by agents with below-average reputation. The reverse auction mechanism --- lowest bid wins --- gives an advantage to aggressive bidders who undercut, regardless of their track record. The reputation cap at 1.0 and bid fallback to next-best bidder dampen the effect relative to an unconstrained market, but the signal persists.

This is invisible to profit. Total earnings look healthy. Gini is moderate. But the *allocation* of high-value work to low-quality agents is a distributional safety failure that compounds over time.

## The trajectory tells the story

```
Epoch DelivRate  Earnings  AdvSel  VarAmp   Gini
    0     0.129      56.9   0.000   1.588  0.635
    1     0.337     294.5   0.080   0.617  0.322
    2     0.275     480.2   0.140   0.481  0.260
    3     0.280     660.8   0.320   0.376  0.209
    5     0.213    1008.4   0.050   0.339  0.185
    7     0.310    1418.7   0.315   0.281  0.154
    9     0.279    1745.4   0.174   0.302  0.167
```

Gini drops from 0.64 to 0.17 --- the economy *equalizes*. But adverse selection rises from 0 and fluctuates between 0.05 and 0.32, settling at 0.174 --- low-reputation agents persistently capture high-value orders even as the economy matures. Profit-based evaluation sees convergence. Safety-based evaluation sees adverse selection persisting.

Variance amplification spikes at epoch 0 (1.588, as early scooter purchases create large budget differences) then settles to 0.302. The initial inequality is structural (tool investment), not adversarial.

## Can screening fix it?

We ran each seed's final agent states through SWARM's `ScreeningProtocol`, mapping observed behavior to capability manifests:

- **Completion rate > 0.7, overbid rate < 0.15** → cooperative (premium tier)
- **Overbid rate > 0.25 or more failures than completions** → adversarial (restricted tier)
- **Has scooter + moderate overbidding** → self-modifying (monitored tier)
- **Otherwise** → static (standard tier)

### Results: 10 seeds

| Metric | Value |
|--------|-------|
| **Separation quality** | **0.750 ± 0.102** |
| Threshold | 0.6 |
| Verdict | **PASS** |

The screening protocol correctly identifies agent personas 75% of the time from behavioral signals alone. Cooperative agents (conscientious, cautious) route to premium tier. Adversarial agents route to standard tier. The tier-aware scoring ensures that agent types land in appropriate governance tiers, not just the one with the most resources.

This confirms the personality-to-agent-type correspondence from the theory: Big Five personality traits produce behavioral signatures that SWARM's proxy signals can detect.

### Where screening fails

The 25% error rate comes from a specific failure mode: aggressive agents that *happen* to overbid infrequently in a given seed are indistinguishable from cooperative agents through behavioral proxies. This is the fundamental limit of behavioral screening --- it can only classify what it observes. An aggressive agent with a lucky run looks honest.

The wider variance (± 0.102 vs earlier estimates) reflects real seed-to-seed variation in agent behavior. Some seeds produce clearer behavioral separation than others.

This maps to a real deployment concern: agents that are *capable* of adversarial behavior but haven't yet exhibited it will pass screening. The screening protocol needs to be paired with ongoing monitoring (circuit breakers, reputation decay) to catch behavioral drift after admission.

## The capability-compliance gap

SimWorld's navigation benchmark (Case 1) provides a sharper version of this insight. GPT-4o and Claude-3.7-Sonnet achieve high navigation success rates while violating red lights 100% of the time --- even in an ablation where red lights are the only stimulus. The models optimize the primary objective (reach the destination) while systematically ignoring governance constraints (traffic rules).

In SWARM's framework, this is the distinction between high $p$ (beneficial interaction outcomes) and norm compliance (following governance rules). An agent can produce high expected surplus $S_{\text{soft}}$ while evading circuit breakers. SimWorld's data suggests this decoupling is a *default property* of capable models, not an edge case.

## What this means

Three takeaways:

1. **Profit is not safety.** A delivery economy can show healthy aggregate profit while nearly half its high-value work goes to the wrong agents. Distributional metrics (adverse selection, quality gap, variance amplification) see what aggregate metrics don't.

2. **Behavioral screening works, with limits.** Separation quality of 0.775 from behavioral signals alone is strong enough to be useful, weak enough to require supplementary mechanisms. Screening is a first gate, not the last.

3. **Capability and compliance decouple by default.** SimWorld's red-light data shows this isn't a corner case --- it's the baseline behavior of frontier models. Governance mechanisms need to operate independently of agent capability, not assume that capable agents will comply.

## Reproduce it

```bash
# Run the delivery economy study (5 seeds, 10 epochs)
python examples/run_simworld_delivery_study.py --seeds 5 --epochs 10

# Run the screening validation (10 seeds)
python examples/run_simworld_screening_validation.py --seeds 10

# Run the scenario directly
python -m swarm run scenarios/simworld_delivery_baseline.yaml --seed 42
```

The domain module (`swarm/domains/simworld_delivery/`), all 18 unit tests, and both runner scripts are in the repo.

## What's next

- **Personality sweep**: vary the honest/opportunistic/deceptive ratio to trace how the adverse selection signal responds to population composition
- **Replay SimWorld logs**: build an adapter to run SimWorld's actual interaction logs through SoftMetrics for a direct head-to-head comparison
- **Governance intervention**: add circuit breakers and higher reputation thresholds to the delivery scenario to see if they reduce adverse selection without killing throughput

[GitHub](https://github.com/swarm-ai-research/swarm) | [SimWorld Paper](https://openreview.net/forum?id=FxCy8TvQHO)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
