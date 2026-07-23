---
description: "Run a complete scenario with all artifacts exported:"
---

# Reproducibility Guide

This guide shows you [how to run](../research/alignment_waltz_vs_macpo.md) reproducible experiments and manage artifacts in SWARM.

## One-Command Reproducible Run

Run a complete scenario with all artifacts exported:

```bash
python -m swarm run scenarios/baseline.yaml \
  --seed 42 \
  --epochs 10 \
  --steps 10 \
  --export-json runs/my_experiment/history.json \
  --export-csv runs/my_experiment/csv/
```

This command will:
1. Run the baseline scenario with seed 42 (reproducible)
2. Execute 10 epochs with 10 steps each
3. Export full interaction history to JSON
4. Export per-epoch metrics to CSV files

## Artifact Paths

SWARM stores experiment artifacts in a standard directory structure:

### Run Directory Structure

```
runs/
├── 20260216-184200_baseline_seed42/
│   ├── history.json              # Complete interaction history
│   ├── csv/                      # Per-epoch metrics
│   │   ├── metrics.csv
│   │   ├── agents.csv
│   │   └── interactions.csv
│   ├── plots/                    # Generated visualizations
│   │   ├── toxicity.png
│   │   ├── welfare.png
│   │   └── quality_gap.png
│   └── metadata.json             # Run configuration
```

### Standard Artifact Locations

| Artifact Type | Default Location | Description |
|--------------|------------------|-------------|
| **History JSON** | `runs/<timestamp>_<scenario>_seed<seed>/history.json` | Complete event log for replay |
| **Metrics CSV** | `runs/<timestamp>_<scenario>_seed<seed>/csv/metrics.csv` | Per-epoch summary metrics |
| **Agent States CSV** | `runs/<timestamp>_<scenario>_seed<seed>/csv/agents.csv` | Agent state evolution |
| **Interactions CSV** | `runs/<timestamp>_<scenario>_seed<seed>/csv/interactions.csv` | Individual interactions |
| **Plots** | `runs/<timestamp>_<scenario>_seed<seed>/plots/*.png` | Matplotlib/Seaborn plots |
| **Event Log** | `logs/events_<timestamp>.jsonl` | Append-only JSONL event stream |

!!! note "Runs Directory"
    The `runs/` directory is gitignored. For long-term storage, archive runs to the [swarm-artifacts](https://github.com/swarm-ai-research/swarm-artifacts) repository.

## Complete Reproduction Workflow

### Step 1: Run Experiment

```bash
# Create timestamped run directory
RUN_ID=$(date +%Y%m%d-%H%M%S)_baseline_seed42
mkdir -p runs/$RUN_ID

# Run with full exports
python -m swarm run scenarios/baseline.yaml \
  --seed 42 \
  --epochs 20 \
  --steps 15 \
  --export-json runs/$RUN_ID/history.json \
  --export-csv runs/$RUN_ID/csv/
```

### Step 2: Generate Plots

```bash
# Generate standard plots from run directory
python examples/plot_run.py runs/$RUN_ID

# Or generate custom plots
python examples/plot_ai_economist.py runs/$RUN_ID
```

Plots are saved to `runs/$RUN_ID/plots/`.

### Step 3: Verify Reproducibility

```bash
# Re-run with same seed to verify reproducibility
python -m swarm run scenarios/baseline.yaml \
  --seed 42 \
  --epochs 20 \
  --steps 15 \
  --export-json runs/${RUN_ID}_verify/history.json

# Compare results (histories should be identical)
diff runs/$RUN_ID/history.json runs/${RUN_ID}_verify/history.json
```

### Step 4: Archive Results

```bash
# Copy run directory to artifacts repo (if using)
cp -r runs/$RUN_ID /path/to/swarm-artifacts/runs/
```

## Using Python API

For programmatic access with full artifact control:

```python
from pathlib import Path
from datetime import datetime
from swarm.scenarios import build_orchestrator, load_scenario
from swarm.logging.event_log import EventLogger

# Create run directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = Path(f"runs/{timestamp}_baseline_seed42")
run_dir.mkdir(parents=True, exist_ok=True)

# Load and run scenario
scenario = load_scenario("scenarios/baseline.yaml")
orchestrator = build_orchestrator(scenario)

# Set up event logger
event_log = run_dir / "events.jsonl"
logger = EventLogger(event_log)

# Run simulation
metrics = orchestrator.run()

# Export artifacts
orchestrator.export_history(run_dir / "history.json")
orchestrator.export_csv(run_dir / "csv")

print(f"Run artifacts saved to: {run_dir}")
print(f"  - History: {run_dir / 'history.json'}")
print(f"  - Metrics: {run_dir / 'csv' / 'metrics.csv'}")
print(f"  - Event log: {event_log}")
```

## Reproducibility Best Practices

### Always Set Seeds

```yaml
# In scenario YAML
simulation:
  seed: 42  # Ensures reproducible RNG state
  n_epochs: 20
  steps_per_epoch: 15
```

Or override via CLI:

```bash
swarm run scenarios/baseline.yaml --seed 42
```

### Document Dependencies

```bash
# Save exact package versions
pip freeze > runs/$RUN_ID/requirements.txt

# Or use environment.yml for conda
conda env export > runs/$RUN_ID/environment.yml
```

### Version Control Scenarios

Keep scenarios in git to track configuration changes:

```bash
git add scenarios/my_experiment.yaml
git commit -m "Add experiment: test transaction tax effects"
```

### Archive Complete Runs

A complete reproducible run includes:

- ✅ Scenario YAML file
- ✅ Seed value
- ✅ SWARM version (`swarm --version`)
- ✅ Python version
- ✅ Dependency versions
- ✅ All exported artifacts (JSON, CSV)
- ✅ Generated plots
- ✅ README or notes describing the experiment

## Common Scenarios

### Multi-Seed Runs

Run the same scenario with multiple seeds for statistical robustness:

```bash
for seed in 42 123 456 789 1024; do
  python -m swarm run scenarios/baseline.yaml \
    --seed $seed \
    --epochs 20 \
    --steps 15 \
    --export-json runs/baseline_seed${seed}/history.json \
    --export-csv runs/baseline_seed${seed}/csv/
done
```

### Parameter Sweep

Test multiple parameter values systematically:

```bash
# Sweep transaction tax rates
for tax in 0.00 0.01 0.02 0.05 0.10; do
  # Create modified scenario
  yq eval ".governance.transaction_tax = $tax" \
    scenarios/baseline.yaml > /tmp/sweep_tax_${tax}.yaml
  
  # Run with modified config
  python -m swarm run /tmp/sweep_tax_${tax}.yaml \
    --seed 42 \
    --export-csv runs/sweep_tax_${tax}/csv/
done
```

Or use the built-in sweep functionality:

```python
from swarm.analysis.parameter_sweep import ParameterSweep

sweep = ParameterSweep(
    base_scenario="scenarios/baseline.yaml",
    parameter="governance.transaction_tax",
    values=[0.0, 0.01, 0.02, 0.05, 0.1],
    seeds=[42, 123, 456],
)

results = sweep.run()
sweep.export_results("runs/tax_sweep/")
sweep.plot_comparison("runs/tax_sweep/plots/")
```

### Replay Analysis

Replay saved runs for alternative analyses:

```python
from swarm.replay.replay_runner import ReplayRunner

# Load history from previous run
runner = ReplayRunner.from_history("runs/baseline_seed42/history.json")

# Replay with different metric computations
metrics = runner.replay(include_incoherence=True)

# Compute alternative statistics
from swarm.metrics.soft_metrics import SoftMetrics
calculator = SoftMetrics()

for interaction in runner.get_interactions():
    stats = calculator.compute_interaction_stats(interaction)
    print(f"p={interaction.p:.3f}, toxicity={stats.toxicity:.3f}")
```

## Artifacts Repository

For long-term storage and sharing, use the separate artifacts repository:

```bash
# Clone artifacts repo (large files, historical runs)
git clone https://github.com/swarm-ai-research/swarm-artifacts.git

# Copy your run to artifacts
cp -r runs/20260216-184200_baseline_seed42 \
  swarm-artifacts/runs/

# Commit and push
cd swarm-artifacts
git add runs/20260216-184200_baseline_seed42
git commit -m "Add baseline replication run"
git push
```

The artifacts repo stores:
- Historical experiment runs
- Lean proofs
- Research notes
- Reference papers
- Large datasets

## Troubleshooting

### Results Not Reproducible

If re-running with the same seed produces different results:

1. **Check SWARM version**: `swarm --version`
2. **Verify seed is set**: Check scenario YAML or `--seed` flag
3. **Check dependency versions**: `pip freeze | grep -E "(numpy|pandas)"`
4. **Disable parallelization**: Use single-threaded execution for debugging

### Missing Artifacts

If exports are not created:

1. **Check directory exists**: `mkdir -p runs/my_experiment/csv`
2. **Verify permissions**: Ensure write access to runs directory
3. **Check disk space**: `df -h`
4. **Use absolute paths**: Avoid relative path issues

### Large Artifacts

For scenarios with many epochs or agents:

1. **Use CSV exports**: More compact than full history JSON
2. **Filter interactions**: Export only accepted interactions
3. **Compress artifacts**: `gzip runs/*/history.json`
4. **Use streaming exports**: For very large runs (10k+ epochs)

## Next Steps

- [Parameter Sweeps](../guides/parameter-sweeps.md) - Systematic experimentation
- [Custom Agents](../guides/custom-agents.md) - Create new agent behaviors
- [Governance Guide](../concepts/governance.md) - Safety mechanisms
- [Analyzing Results](../tutorials/analyzing-results.md) - Analyze experiment results
