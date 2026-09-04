---
description: "Define simulations in YAML with full governance configuration:"
---

# Scenarios & Parameter Sweeps

## Defining Scenarios

Define simulations in YAML with full [governance configuration](https://github.com/swarm-ai-research/swarm-artifacts/blob/main/papers/research_swarm_governance.md):

```yaml
# scenarios/status_game.yaml
scenario_id: status_game
description: "Reputation competition with governance"

agents:
  - type: honest
    count: 2
  - type: opportunistic
    count: 2
  - type: adversarial
    count: 1

governance:
  transaction_tax_rate: 0.05
  reputation_decay_rate: 0.95
  staking_enabled: true
  min_stake_to_participate: 10.0
  circuit_breaker_enabled: true
  freeze_threshold_toxicity: 0.6
  audit_enabled: true
  audit_probability: 0.15

simulation:
  n_epochs: 20
  steps_per_epoch: 15
  seed: 123

payoff:
  s_plus: 3.0
  s_minus: 1.5
  h: 2.5
  theta: 0.5
  w_rep: 2.0
```

Run scenarios from the command line:

```bash
python examples/run_scenario.py scenarios/baseline.yaml
python examples/run_scenario.py scenarios/status_game.yaml
python examples/run_scenario.py scenarios/strict_governance.yaml
```

Or use the package CLI entry point:

```bash
python -m swarm list
python -m swarm run scenarios/baseline.yaml
python -m swarm run scenarios/status_game.yaml --seed 42 --epochs 20
```

Or load programmatically:

```python
from swarm.scenarios import load_and_build

orchestrator = load_and_build(Path("scenarios/status_game.yaml"))
metrics = orchestrator.run()
```

## Scenario Comparison

| Metric | Baseline | Status Game | Strict Governance |
|--------|----------|-------------|-------------------|
| **Governance** | None | Moderate | Heavy |
| Tax rate | 0% | 5% | 10% |
| Reputation decay | None | 5%/epoch | 15%/epoch |
| Staking required | No | 10.0 | 25.0 |
| Circuit breaker | No | Yes (0.6) | Yes (0.5) |
| Audit probability | 0% | 15% | 25% |
| **Results** | | | |
| Bad actor frozen | No | Yes | Yes |
| Bad actor payoff | +3.42 | +1.22 | -1.55 |
| Avg toxicity | 0.30 | 0.33 | 0.32 |
| Welfare/epoch | 7.29 | 13.02 | 8.15 |

Governance effectively punishes bad actors (payoffs drop from positive to negative) while maintaining similar toxicity levels. Stricter governance reduces bad actor gains but also dampens overall welfare.

## Parameter Sweeps

Run batch simulations over parameter ranges:

```python
from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario

# Load base scenario
scenario = load_scenario(Path("scenarios/baseline.yaml"))

# Configure sweep
config = SweepConfig(
    base_scenario=scenario,
    parameters=[
        SweepParameter(
            name="governance.transaction_tax_rate",
            values=[0.0, 0.05, 0.10, 0.15],
        ),
        SweepParameter(
            name="governance.circuit_breaker_enabled",
            values=[False, True],
        ),
    ],
    runs_per_config=3,  # Multiple runs for statistical significance
    seed_base=42,
)

# Run sweep
runner = SweepRunner(config)
results = runner.run()

# Export to CSV
runner.to_csv(Path("results.csv"))

# Get summary statistics
summary = runner.summary()
```

Run the example:
```bash
python examples/parameter_sweep.py
python examples/parameter_sweep.py --output my_results.csv
```

Supported parameter paths:
- `governance.*` - Any GovernanceConfig field
- `payoff.*` - Any PayoffConfig field
- `n_epochs`, `steps_per_epoch` - Simulation settings

## Loan / commitment scenario (`scenarios/loan_commitment.yaml`)

Isolates one mechanism the core loop cannot express: **default as a strategic
action**. A lender extends credit; the borrower invests, then chooses repay
vs. default one period later, so the lender's loss is chosen rather than drawn
from `p`. Motivated by the AI Village Ṁ5,000 loan incident
([blog post](blog/the-m5000-loan-goal-consumed-judgment.md)).

The borrower's payoff mirrors the engine's form,
`π = mana − ρ·harm_to_lender + w_rep·Δreputation`, with three levers: `rho`
(externality internalization), `w_rep` (reputation weight), and a third-party
`bond` forfeited on default. The YAML is read by
`experiments/loan_commitment_sweep.py`, not the swarm scenario loader:

```bash
python -m experiments.loan_commitment_sweep            # full grid
python -m experiments.loan_commitment_sweep --quick    # coarse grid
```

Writes `runs/<ts>_loan_commitment/{results.csv,predictions.json,plots/}`.
Preregistered predictions and their status are recorded in
`predictions.json`: the default decision is a step in `rho` at
`rho* = 1 − w_rep(r_repay + r_default)/(due − bond)` (met); a gift leaves
the default rate unchanged under the Village objective (met) but not in
general — gifts fix ability, never willingness, so they help exactly when a
bond or reputation weight already makes able borrowers repay.
