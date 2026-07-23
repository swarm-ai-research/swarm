---
description: "SWARM is an open-source multi-agent AI safety framework for distributional risk assessment. Measure toxicity, detect deception, and test governance mechanisms across agent populations."
date: 2026-03-02
---

# Multi-Agent AI Safety Framework

SWARM (System-Wide Assessment of Risk in Multi-agent systems) is the reference implementation of the **distributional AGI safety** research framework. It provides Python tools for studying emergent risks in multi-agent AI systems.

## What Makes SWARM Different

Most AI safety tools focus on individual models. SWARM focuses on **populations**:

| Traditional safety tools | SWARM |
|--------------------------|-------|
| Evaluate single model outputs | Evaluate population-level dynamics |
| Binary safe/unsafe labels | [Soft probabilistic labels](../concepts/soft-labels.md) |
| Static benchmarks | Dynamic multi-epoch simulations |
| Manual red-teaming | [Automated adversarial testing](red-teaming.md) |
| One-shot evaluation | [Longitudinal tracking](../tutorials/analyzing-results.md) across epochs |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Agents    │────►│ Orchestrator │────►│  Metrics    │
│ (honest,    │     │ (epochs,     │     │ (toxicity,  │
│  deceptive, │     │  matching,   │     │  quality    │
│  adversary) │     │  governance) │     │  gap, etc.) │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ Governance  │
                    │ (taxes,     │
                    │  breakers,  │
                    │  audits)    │
                    └─────────────┘
```

### Data Flow

```
Observables → ProxyComputer → v_hat → sigmoid → p → SoftPayoffEngine → payoffs
                                                 ↓
                                            SoftMetrics → toxicity, quality gap, etc.
```

## Installation

```bash
pip install swarm-safety
```

Or install from source for development:

```bash
git clone https://github.com/swarm-ai-research/swarm.git
cd swarm
pip install -e ".[dev,runtime]"
```

## Quick Start

```python
from swarm.agents.honest import HonestAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure simulation
config = OrchestratorConfig(n_epochs=10, steps_per_epoch=10, seed=42)
orch = Orchestrator(config=config)

# Register agents
for i in range(7):
    orch.register_agent(HonestAgent(agent_id=f"h{i}"))
for i in range(3):
    orch.register_agent(DeceptiveAgent(agent_id=f"d{i}"))

# Run and analyze
metrics = orch.run()
for m in metrics:
    print(f"Epoch {m.epoch}: toxicity={m.toxicity_rate:.3f} qgap={m.quality_gap:+.3f}")
```

## Core Components

### Agents

SWARM ships with three agent types and supports custom agents:

| Agent | Behavior | Use case |
|-------|----------|----------|
| [HonestAgent](../api/agents.md) | Consistent cooperation | Baseline population |
| [DeceptiveAgent](../api/agents.md) | Trust-then-exploit | Test governance detection |
| [AdversarialAgent](../api/agents.md) | Active exploitation | Stress-test mechanisms |
| [Custom](custom-agents.md) | User-defined | Research-specific strategies |

### Metrics

Four key metrics capture distributional health:

- **[Toxicity rate](../concepts/metrics.md)** — Expected harm among accepted interactions
- **[Quality gap](../concepts/metrics.md)** — Whether governance selects for quality (negative = adverse selection)
- **[Conditional loss](../concepts/metrics.md)** — Payoff effect of selection
- **[Incoherence index](../concepts/metrics.md)** — Decision variance across replays

### Governance

Six configurable mechanisms that operate at the population level:

- [Transaction taxes](../concepts/governance.md) — Friction against exploitation
- [Circuit breakers](../concepts/governance.md) — Freeze toxic agents
- [Reputation decay](../concepts/governance.md) — Prevent trust accumulation
- [Random audits](../concepts/governance.md) — Probabilistic detection
- [Staking](../concepts/governance.md) — Skin-in-the-game requirements
- [Collusion detection](../concepts/governance.md) — Catch coordinated attacks

### Bridges

Connect SWARM to external systems:

| Bridge | Integration |
|--------|-------------|
| [Concordia](../bridges/concordia.md) | LLM agent environments |
| [Prime Intellect](../bridges/prime_intellect.md) | Safety-reward RL training |
| [GasTown](../bridges/gastown.md) | Production data pipelines |
| [AgentXiv](../bridges/agentxiv.md) | Research publication platform |

## Research Context

SWARM implements the framework formalized in [Soft-Label Governance for Distributional Safety in Multi-Agent Systems](https://arxiv.org/abs/2604.19752) (arXiv, 2026); see also [Distributional AGI Safety](https://arxiv.org/abs/2512.16856). For theoretical foundations, see the [research theory](../research/theory.md) page.

## Next Steps

- [Quick Start Tutorial](../getting-started/quickstart.md) — Run your first simulation
- [Writing Scenarios](scenarios.md) — Configure custom experiments
- [Governance Simulation](governance-simulation.md) — Test governance before deployment
- [Parameter Sweeps](parameter-sweeps.md) — Systematic parameter exploration
- [Red Teaming](red-teaming.md) — Adversarial stress testing
