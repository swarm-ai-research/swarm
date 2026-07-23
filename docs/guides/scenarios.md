---
description: "Scenarios are YAML files with four main sections:"
---

# Writing Scenarios

This guide covers how to create custom SWARM scenarios for your experiments.

## Scenario File Structure

Scenarios are YAML files with four main sections:

```yaml
name: my_scenario
description: What this scenario tests

simulation:
  # Simulation parameters

agents:
  # Agent configuration

governance:
  # Governance settings

payoff:
  # Payoff parameters
```

## Simulation Section

```yaml
simulation:
  seed: 42              # Random seed (required for reproducibility)
  n_epochs: 20          # Number of epochs
  steps_per_epoch: 15   # Steps per epoch
  async_mode: false     # Async agent execution
  log_level: INFO       # Logging verbosity
```

## Agents Section

Define agent populations:

```yaml
agents:
  - type: honest
    count: 5
    name: "Team A"
    id_prefix: h
    params:
      cooperation_threshold: 0.7

  - type: opportunistic
    count: 3
    id_prefix: opp

  - type: deceptive
    count: 2
    id_prefix: dec
    params:
      trust_building_epochs: 5
      exploitation_threshold: 0.8

  - type: adversarial
    count: 1
    id_prefix: adv
```

Optional: set `name` to control the human-readable display name for agents (defaults to `agent_id`). If `count > 1`, names are suffixed for uniqueness (e.g., `Team A_1`, `Team A_2`).

### Agent Types

| Type | Description | Key Parameters |
|------|-------------|----------------|
| `honest` | Cooperative, diligent | `cooperation_threshold` |
| `opportunistic` | Payoff maximizing | `cherry_pick_threshold` |
| `deceptive` | Trust then exploit | `trust_building_epochs` |
| `adversarial` | Active disruption | `target_selection` |
| `llm` | LLM-powered | `model`, `persona` |

### LLM Agents

```yaml
agents:
  - type: llm
    count: 2
    id_prefix: llm
    params:
      model: claude-3-haiku-20240307
      persona: |
        You are a helpful assistant focused on completing tasks
        efficiently while maintaining good relationships.
      temperature: 0.7
```

## Governance Section

```yaml
governance:
  # Friction
  transaction_tax: 0.02

  # Reputation
  reputation_decay: 0.1
  initial_reputation: 1.0

  # Circuit breakers
  circuit_breaker_threshold: 0.3
  circuit_breaker_window: 10
  circuit_breaker_cooldown: 5

  # Audits
  audit_probability: 0.05
  audit_penalty: 0.5

  # Staking
  staking_requirement: 10.0
  stake_slash_rate: 0.1

  # Collusion
  collusion_detection: true
  collusion_threshold: 0.8
```

## Payoff Section

```yaml
payoff:
  s_plus: 1.0          # Surplus if beneficial
  s_minus: 0.5         # Loss if harmful
  h: 0.3               # External harm
  theta: 0.5           # Initiator's share
  w_rep: 0.1           # Reputation weight
  rho_a: 0.1           # Initiator externality internalization
  rho_b: 0.1           # Counterparty externality internalization
```

## Loading Scenarios

### CLI

```bash
swarm run scenarios/my_scenario.yaml
```

### Programmatic

```python
from swarm.scenarios import ScenarioLoader
from swarm.core.orchestrator import Orchestrator

scenario = ScenarioLoader.load("scenarios/my_scenario.yaml")
orchestrator = Orchestrator.from_scenario(scenario)
metrics = orchestrator.run()
```

## Scenario Comparison

Compare multiple scenarios:

```yaml
# scenarios/comparison.yaml
name: governance_comparison
base: baseline.yaml
variants:
  - name: no_governance
    governance:
      transaction_tax: 0.0
  - name: light_governance
    governance:
      transaction_tax: 0.01
  - name: heavy_governance
    governance:
      transaction_tax: 0.05
```

```bash
swarm compare scenarios/comparison.yaml --output results/
```

## Best Practices

!!! tip "Reproducibility"
    Always set a `seed` for reproducible results.

!!! tip "Start Simple"
    Begin with small agent counts and few epochs, then scale up.

!!! tip "Document Purpose"
    Use the `description` field to explain what question the scenario answers.

!!! tip "Version Control"
    Keep scenarios in git alongside your code.

!!! warning "Untrusted Scenarios"
    Scenario YAML can specify output paths. Only run trusted scenarios, and prefer outputs under `runs/` or `logs/` to avoid clobbering unrelated files.

## Example Scenarios

See the [scenarios directory](https://github.com/swarm-ai-research/swarm/tree/main/scenarios) for examples:

- `baseline.yaml` - Minimal setup for testing
- `adverse_selection.yaml` - Demonstrates quality gap emergence
- `governance_test.yaml` - Tests governance effectiveness
- `llm_agents.yaml` - LLM-powered agents

## See also

- [Your First Scenario](../getting-started/first-scenario.md) — Beginner walkthrough for creating scenarios
- [Parameter Sweeps](parameter-sweeps.md) — Run systematic experiments across parameter ranges
- [Custom Agents](custom-agents.md) — Create new agent behavioral policies
- [LLM Agents](llm-agents.md) — Use language models as agent decision-makers
