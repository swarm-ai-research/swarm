---
title: SWARM — Distributional AGI Safety
emoji: 🐝
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: mit
arxiv: 2604.19752
tags:
  - multi-agent-systems
  - ai-safety
  - distributional-safety
  - governance
  - soft-labels
---

# SWARM — System-Wide Assessment of Risk in Multi-agent Systems

Open-source framework for studying how catastrophic risks emerge from populations of interacting AI agents — even when no single agent is individually dangerous.

**Paper:** [Soft-Label Governance for Distributional Safety in Multi-Agent Systems (arXiv:2604.19752)](https://arxiv.org/abs/2604.19752); see also [Distributional AGI Safety (arXiv:2512.16856)](https://arxiv.org/abs/2512.16856)

**Code:** [github.com/swarm-ai-research/swarm](https://github.com/swarm-ai-research/swarm)

**Docs:** [swarm-ai.org](https://swarm-ai.org)

## Key Findings

| Finding | Value | Runs |
|---------|-------|------|
| Deontological framing reduces deception | 95% reduction | 180 |
| Purity paradox: 20% honest > 100% honest | 55% higher welfare | 30 seeds |
| Deception persists at temperature 0.0 | Structural, not stochastic | 120 |
| Forced cooperation window | 3 turns eliminates escalation | 210 |
| Transparency + safety training | Nuclear rate 60%→30% | 120 |
| Full externality pricing (ρ≥0.5) | Honesty dominates by 43% | 21 configs |

## Install

```bash
pip install swarm-safety
```

## Quick Example

```python
from swarm.agents.honest import HonestAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(n_epochs=10, steps_per_epoch=10, seed=42)
orchestrator = Orchestrator(config=config)

orchestrator.register_agent(HonestAgent(agent_id="honest_1"))
orchestrator.register_agent(DeceptiveAgent(agent_id="dec_1"))

metrics = orchestrator.run()
for m in metrics:
    print(f"Epoch {m.epoch}: toxicity={m.toxicity_rate:.3f}")
```

## Core Architecture

```
Observables → ProxyComputer → v_hat → sigmoid → p → SoftPayoffEngine → payoffs
                                                  ↓
                                            SoftMetrics → toxicity, quality gap, etc.
```

## Four Key Metrics

- **Toxicity Rate** — E[1−p | accepted]. Values above 0.3 indicate serious system problems.
- **Quality Gap** — E[p | accepted] − E[p | rejected]. Negative = adverse selection.
- **Conditional Loss** — E[π | accepted] − E[π]. Selection creating or destroying value.
- **Incoherence Index** — Variance-to-error ratio across replays. High = unstable decisions.

## Six Governance Mechanisms

Circuit breakers, transaction taxes, reputation decay, staking, collusion detection, random audits — all configurable and composable.

## License

MIT
