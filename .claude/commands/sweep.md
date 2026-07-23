# /sweep

Run a parameter sweep and write results to a standardized run folder — use standalone when you need sweep data only, without statistical analysis or paper scaffolding. Distinct from `/run_scenario` (single scenario, single seed), `/analyze_experiment` (statistical analysis on existing data), and `/full_study` (end-to-end pipeline chaining sweep + analysis + plots + paper).

## Usage

`/sweep [scenario_path] [output_name]`

Examples:
- `/sweep` (defaults to `scenarios/baseline.yaml`)
- `/sweep scenarios/boundary_test.yaml boundary_sweep`

## Behavior

1) Create `runs/<YYYYMMDD-HHMMSS>_sweep/`.
2) Execute the built-in sweep example:
- `python examples/parameter_sweep.py --scenario <scenario_path> --output <run_dir>/<output_name>.csv`
3) If pandas+pyarrow are available, also write Parquet:
- `<run_dir>/<output_name>.parquet`
4) **Auto-generate `summary.json`**: After the CSV is written, read it with pandas, group by parameter columns, and compute per-config summary statistics:
```python
import pandas as pd, json
from pathlib import Path

df = pd.read_csv(csv_path)
run_dir = Path(run_dir)

# Identify parameter columns (governance.* or any column with few unique values)
param_cols = [c for c in df.columns if c.startswith("governance.")]
grouped = df.groupby(param_cols)

configs = []
for name, group in grouped:
    config = dict(zip(param_cols, name if isinstance(name, tuple) else [name]))
    config["n_runs"] = len(group)
    for metric in ["welfare", "toxicity_rate", "quality_gap", "honest_payoff", "adversarial_payoff", "opportunistic_payoff"]:
        if metric in group.columns:
            config[f"mean_{metric}"] = round(group[metric].mean(), 4)
            config[f"std_{metric}"] = round(group[metric].std(), 4)
    configs.append(config)

summary = {
    "scenario": scenario_id,
    "seed_base": seed,
    "epochs": epochs,
    "runs_per_config": runs_per_config,
    "total_runs": len(df),
    "param_combinations": len(configs),
    "configs": configs,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
```
This eliminates the manual post-processing step that was previously required before running `/council_review`.

5) Print a compact table of the best/worst configs by:
- `mean_welfare` (higher is better)
- `mean_toxicity` (lower is better)

## `--delm` mode (directed search instead of a grid)

`/sweep --delm [scenario_path] [output_name]`

This is a mode of the `/sweep` **slash command**, not a flag on
`examples/parameter_sweep.py`. When invoked, run the dedicated module CLI shown
below (`python -m swarm.analysis.delm_hillclimb`) instead of the grid-sweep
script — the rest of this section tells you how.

A grid sweep evaluates every cell of a fixed lattice. When you instead want to
*search* the governance/payoff space for a high-fitness configuration, use
`--delm`: a DeLM-style (decentralized language-model) parallel hill-climber that
shares verified state across virtual workers. It optimizes the same knobs as the
grid sweep (`governance.*`, `payoff.*` from `PARAM_RANGES`) against the composite
soft-safety fitness in `swarm.analysis.evolver.compute_fitness` (toxicity,
welfare, quality gap, payoff gap).

How it works (see `swarm/analysis/delm_hillclimb.py` for the full docstring):
- **Shared context** holds the running best, a trail of verified *gists*, and
  binding *constraints* (known dead-end neighbors) — workers read it before
  paying for an evaluation, so they skip already-seen and pruned cells.
- **Task queue** of `explore` / `mutate_dim` / `restart` / `diversify` work
  items lets many workers explore neighbors asynchronously; a verified
  improvement spawns fresh neighbor tasks around the new best.
- **Verified admission**: a candidate that beats the best is re-evaluated at an
  independent seed and only admitted if it still improves (filters noise).
- **Escape local optima** via `restart` (fresh random point) and `diversify`
  (jump to a distant basin another worker found).

Run it via the module CLI (writes a self-contained run folder with
`best_params.json`, `result.json`, `gists.jsonl`, and `plots/trajectory.png`):

```bash
python -m swarm.analysis.delm_hillclimb <scenario_path> \
    --max-evals 60 --workers 4 --eval-epochs 2 --eval-steps 4 --seed 42 \
    --output-dir runs/<YYYYMMDD-HHMMSS>_delm/
```

Useful flags: `--no-verify` (faster, noisier), `--threads` (real OS threads,
not reproducible), `--restart-prob`, `--step-frac`, `--no-plot`, and
`--weight-{toxicity,welfare,quality-gap,payoff-gap}` to reshape the fitness.
The default deterministic scheduler is reproducible from `scenario YAML + seed`.

**Objectives.** By default `--delm` searches the governance/payoff knobs. Pass
`--objective adaptive_policy` to instead optimize the 8-dim adaptive-agent
policy vector (`swarm.adaptive.PARAM_SPEC`) via `run_episode` reward — the same
target the CEM trainer searches (`--n-interactions` sets the episode length).
The landscape is pluggable: `build_governance_objective` /
`build_adaptive_policy_objective` (or a custom `Objective`) decouple the
parameter space + evaluation from the search machinery.

This is a sibling of `swarm.analysis.gepa_optimizer` (LLM-guided Pareto search)
and `swarm.analysis.evolver` (darwinian search) — three optimizers over one
landscape. Prefer `--delm` when you want a dependency-free, reproducible search;
prefer GEPA/evolver when you want LLM-reasoned mutations and have the extras
installed.

## Notes

- Keep sweeps small by default; prefer fewer epochs/runs unless explicitly requested.
- The `summary.json` file is consumed by `/council_review` — always generate it.
- `--delm` writes its own run folder (`best_params.json` + `gists.jsonl`); it does
  not produce the grid `summary.json`, so it is not a `/council_review` input.

