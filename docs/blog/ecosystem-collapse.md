---
date: 2026-02-09
description: "Quantifying governance failure in multi-agent systems"
author: "SWARM Team"
keywords:
  - ecosystem collapse multi-agent
  - phase transition adversarial agents
  - governance failure threshold
claims:
  - metric: "Collapse phase transition"
    value: "37.5-50% adversarial"
    description: "Governance mechanisms fail completely between 37.5% and 50% adversarial agent fraction"
abstract: "Multi-agent ecosystems exhibit sharp phase transitions: governance mechanisms that work at 37.5% adversarial agents fail completely at 50%. The critical lever is collusion detection, not punishment severity. Binary safety labels miss the distributional dynamics that drive ecosystem collapse."
---

# When Agent Ecosystems Collapse

*Quantifying governance failure in [multi-agent systems](../research/papers.md)*

---

Most AI safety research focuses on aligning one model at a time. But the systems actually being deployed look more like ecosystems — tool-using assistants, autonomous coders, trading bots, and content moderators interacting inside shared environments. These ecosystems can produce harmful outcomes even when no individual agent is misaligned.

We built SWARM to study this problem quantitatively. The main finding: multi-agent ecosystems exhibit sharp phase transitions where governance mechanisms that work fine at 37.5% adversarial agents fail completely at 50%. And the lever that matters most isn't punishing bad actors — it's detecting coordinated behavior.

## Why binary labels aren't enough

Standard safety evaluations label interactions as safe or unsafe. This throws away information. An interaction with a 51% chance of being beneficial gets the same label as one with 99%. You can't measure [adverse selection](../concepts/distributional-safety.md#adverse-selection) — the tendency for bad interactions to be preferentially accepted — with binary data.

Financial markets solved this problem decades ago. Adverse selection in trading is measured continuously using probabilistic models (Kyle 1985, Glosten-Milgrom 1985). SWARM applies the same idea to agent safety:

- Every interaction gets a calibrated probability score **p = P(beneficial)** in [0, 1]
- **Toxicity** = E[1 - p | accepted] — how much harm gets through the filter
- **Quality gap** = E[p | accepted] - E[p | rejected] — negative means the system is preferentially admitting bad interactions
- Six governance levers map to market regulation tools: transaction tax, reputation decay, staking, circuit breakers, audits, and collusion detection

## What we found

Eleven scenarios. 209 epochs. 81 agent-slots. Three regimes.

| Regime | Adversarial % | Acceptance Rate | Avg Toxicity | Welfare | Collapse? |
|--------|--------------|-----------------|-------------|---------|-----------|
| **Cooperative** | 0-20% | > 0.93 | < 0.30 | Stable/growing | No |
| **Contested** | 20-37.5% | 0.42-0.94 | 0.33-0.37 | Declining | No |
| **Collapse** | 50% | < 0.56 | ~0.30 | Zero by epoch 12-14 | Yes |

The interesting stuff is at the boundaries.

**The transition is abrupt, not gradual.** At 37.5% adversarial agents with collusion detection enabled, the ecosystem survived all 25 epochs. At 50% with the same governance stack, welfare hit zero by epoch 12. There's no gentle degradation curve between these outcomes.

**Tuning governance parameters buys time, not survival.** Three red-team variants with progressively adjusted parameters (audit penalties, freeze durations, reputation decay) shifted collapse from epoch 12 to 13 to 14. Two extra epochs is nice. It's not a solution.

**Collusion detection is what actually matters.** Individual-focused levers (audits, reputation, staking) are necessary but insufficient. The scenario that survived near the collapse threshold had pair-wise frequency and correlation monitoring — detecting coordinated patterns across the interaction graph rather than flagging individual agents. This parallels how financial regulators catch wash trading: you look at the pattern, not the individual trades.

**Cooperative welfare scales super-linearly.** In low-adversarial scenarios: 3 agents produced welfare ~1.0, 6 agents ~5.7, 10 agents ~21.3. Cooperative ecosystems are disproportionately productive, which means the cost of collapse also grows non-linearly.

![Scenario comparison across regimes](../papers/figures/fig1_scenario_comparison.webp)

![Timeline overlay showing collapse dynamics](../papers/figures/fig7_timeline_overlay.webp)

## What this means for building agent systems

**Track ecosystem composition.** The critical variable is the fraction of adversarial agents, not whether any individual agent misbehaves. If you're running a multi-agent system in production, monitoring the population mix matters more than monitoring individuals.

**Layer structural governance on top of individual governance.** Reputation systems and staking requirements handle the contested regime. To survive near the collapse boundary, you need pattern-based detection operating on the interaction graph — the agent equivalent of market manipulation surveillance.

**Know which regime you're in.** Cooperative regimes can afford light governance. Systems approaching the collapse boundary need structural interventions activated proactively. By the time you observe collapse, it's already too late.

## Try it

Five minutes, no API keys:

```bash
git clone https://github.com/swarm-ai-research/swarm.git
cd swarm
pip install -e ".[dev,runtime]"
jupyter notebook examples/quickstart.ipynb
```

Or run a scenario directly:

```python
from swarm.scenarios import load_scenario, build_orchestrator
from pathlib import Path

scenario = load_scenario(Path("scenarios/baseline.yaml"))
orchestrator = build_orchestrator(scenario)
history = orchestrator.run()
print(f"Final welfare: {history[-1].total_welfare:.2f}")
print(f"Avg toxicity: {sum(m.toxicity_rate for m in history) / len(history):.3f}")
```

The repo includes 20+ scenario configs, 2200+ tests, and a SQLite database of all runs for reproducibility. The full paper with detailed methodology is at [`docs/papers/distributional_agi_safety.md`](https://github.com/swarm-ai-research/swarm/blob/main/docs/papers/distributional_agi_safety.md).

## References

- Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*.
- Glosten, L. & Milgrom, P. (1985). "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders." *Journal of Financial Economics*.
- Kenton, Z. et al. (2023). "Alignment of Language Agents."
- Zheng et al. "A Primer for Agent-Based Simulation and Modeling in Transportation Applications." [DOI](https://rosap.ntl.bts.gov/view/dot/36178).

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
