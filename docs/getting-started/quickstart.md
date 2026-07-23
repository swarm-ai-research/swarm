---
description: "> Prefer a notebook? The quickstart notebook covers the same material interactively with plots: > > ![Open In..."
---

# Quick Start

This guide walks you through running your first SWARM simulation.

> **Prefer a notebook?** The quickstart notebook covers the same material interactively with plots:
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swarm-ai-research/swarm/blob/main/examples/quickstart.ipynb) &nbsp; or run locally: `jupyter notebook examples/quickstart.ipynb`

## Basic Simulation

```python
from swarm.agents.honest import HonestAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure simulation
config = OrchestratorConfig(
    n_epochs=10,          # Number of epochs to run
    steps_per_epoch=10,   # Steps per epoch
    seed=42,              # Random seed for reproducibility
)

# Create orchestrator
orchestrator = Orchestrator(config=config)

# Register agents with different behavioral policies
orchestrator.register_agent(HonestAgent(agent_id="honest_1", name="Alice"))
orchestrator.register_agent(HonestAgent(agent_id="honest_2", name="Bob"))
orchestrator.register_agent(OpportunisticAgent(agent_id="opp_1"))
orchestrator.register_agent(DeceptiveAgent(agent_id="dec_1"))

# Run simulation
metrics = orchestrator.run()

# Analyze results
for m in metrics:
    print(f"Epoch {m.epoch}: toxicity={m.toxicity_rate:.3f}, welfare={m.total_welfare:.2f}")
```

## Using the CLI

SWARM includes a command-line interface for running scenarios:

```bash
# List available scenarios
swarm list

# Run a scenario
swarm run scenarios/baseline.yaml

# Override settings
swarm run scenarios/baseline.yaml --seed 42 --epochs 20

# Export results
swarm run scenarios/baseline.yaml --export-json results.json
```

## Understanding the Output

After running a simulation, you'll see metrics for each epoch:

| Metric | Meaning |
|--------|---------|
| `toxicity_rate` | Expected harm among accepted interactions |
| `quality_gap` | Difference in quality between accepted vs rejected (negative = adverse selection) |
| `total_welfare` | System-wide surplus minus costs |

!!! warning "Adverse Selection"
    A negative quality gap indicates the system is preferentially accepting lower-quality interactions—a key failure mode SWARM is designed to detect.

## Computing Metrics Manually

```python
from swarm.models.interaction import SoftInteraction, InteractionType
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.core.payoff import SoftPayoffEngine
from swarm.metrics.reporters import MetricsReporter

# Create observable signals
obs = ProxyObservables(
    task_progress_delta=0.7,
    rework_count=1,
    verifier_rejections=0,
    counterparty_engagement_delta=0.4,
)

# Compute soft labels
proxy = ProxyComputer()
v_hat, p = proxy.compute_labels(obs)
print(f"v_hat={v_hat:.3f}, p={p:.3f}")

# Create interaction
interaction = SoftInteraction(
    initiator="agent_1",
    counterparty="agent_2",
    interaction_type=InteractionType.COLLABORATION,
    accepted=True,
    v_hat=v_hat,
    p=p,
)

# Compute payoffs
engine = SoftPayoffEngine()
payoff_a = engine.payoff_initiator(interaction)
payoff_b = engine.payoff_counterparty(interaction)
print(f"Payoffs: initiator={payoff_a:.3f}, counterparty={payoff_b:.3f}")
```

## See also

- [Your First Scenario](first-scenario.md) — Create a custom YAML scenario with agent populations and governance
- [Metrics](../concepts/metrics.md) — What toxicity, quality gap, and conditional loss measure
- [Governance](../concepts/governance.md) — Circuit breakers, taxes, and other safety interventions
- [Parameter Sweeps](../guides/parameter-sweeps.md) — Run systematic experiments across parameter ranges
