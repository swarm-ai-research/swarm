"""Benchmark runner — runs (task × governance config) grid and produces DataFrames.

Usage:
    runner = BenchmarkRunner(n_agents=10)
    df = runner.run_frontier(benchmark, governance_configs, n_seeds=10)
    # df has one row per (config, seed) — ready for frontier plotting

Security invariants:
- run_fn receives only redacted (deep-copied) instances — no oracle leakage.
- run_fn return values are validated before scoring.
- Instances are deep-copied (via redact) before passing to run_fn; oracles are never exposed.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Callable

import pandas as pd

from swarm.benchmarks.base import (
    BenchmarkTask,
    TaskInstance,
    TaskOracle,
    TaskResult,
)

# Type alias for run functions
RunFn = Callable[[TaskInstance, dict[str, Any]], TaskResult]


def _validate_result(result: Any) -> TaskResult:
    """Validate that run_fn returned a well-formed TaskResult."""
    if not isinstance(result, TaskResult):
        raise TypeError(
            f"run_fn must return TaskResult, got {type(result).__name__}"
        )
    if not isinstance(result.completed, bool):
        raise TypeError(
            f"TaskResult.completed must be bool, got {type(result.completed).__name__}"
        )
    if not isinstance(result.steps_taken, int) or result.steps_taken < 0:
        raise ValueError(
            f"TaskResult.steps_taken must be non-negative int, got {result.steps_taken}"
        )
    if not isinstance(result.agent_trace, list):
        raise TypeError(
            f"TaskResult.agent_trace must be list, got {type(result.agent_trace).__name__}"
        )
    return result


class BenchmarkRunner:
    """Runs a benchmark across governance configs and seeds.

    Supports an adversarial_fraction parameter per governance config,
    which flows into score() to produce the safety_score axis for
    frontier plotting.
    """

    def __init__(self, n_agents: int = 10):
        self.n_agents = n_agents

    def run_frontier(
        self,
        benchmark: BenchmarkTask,
        governance_configs: list[dict[str, Any]],
        n_seeds: int = 10,
        run_fn: RunFn | None = None,
    ) -> pd.DataFrame:
        """Run benchmark across (governance_config × seed) grid.

        Args:
            benchmark: The benchmark task to evaluate.
            governance_configs: List of dicts, each with at least an "id" key.
                Optional "adversarial_fraction" key (float 0-1) enables the
                safety axis.
            n_seeds: Number of random seeds to average over.
            run_fn: Optional callable(instance, gov_config) -> TaskResult.
                    The instance is redacted (no ground truth). Return value
                    is validated. If None, uses oracle_run.

        Returns:
            DataFrame with one row per (config, seed).
        """
        # Pre-generate instances and oracles once per seed
        seed_data: dict[int, tuple[TaskInstance, TaskOracle]] = {}
        for seed in range(n_seeds):
            instance, oracle = benchmark.generate(seed, self.n_agents)
            seed_data[seed] = (instance, oracle)

        rows: list[dict[str, Any]] = []
        for gov_config in governance_configs:
            adv_frac = float(gov_config.get("adversarial_fraction", 0.0))
            if not (0.0 <= adv_frac <= 1.0):
                raise ValueError(
                    f"adversarial_fraction must be in [0-1], got {adv_frac}"
                )
            for seed in range(n_seeds):
                instance, oracle = seed_data[seed]

                if run_fn is not None:
                    # Deep-copy + redact: run_fn sees no oracle data
                    redacted = benchmark.redact(instance)
                    raw_result = run_fn(redacted, gov_config)
                    result = _validate_result(raw_result)
                else:
                    # No run_fn: use oracle as baseline
                    result = benchmark.oracle_run(
                        copy.deepcopy(instance),
                        copy.deepcopy(oracle),
                    )

                # Score against the original oracle (never given to run_fn)
                score = benchmark.score(result, oracle, adversarial_fraction=adv_frac)
                interaction = benchmark.to_soft_interaction(score)

                row = {
                    "benchmark": benchmark.task_id,
                    "task_type": benchmark.task_type,
                    "gov_config": gov_config.get("id", "unknown"),
                    "seed": seed,
                    "adversarial_fraction": adv_frac,
                    "p": interaction.p,
                    **asdict(score),
                }
                # Include governance lever values for analysis
                for k, v in gov_config.items():
                    if k not in ("id", "adversarial_fraction"):
                        row[f"gov_{k}"] = v

                rows.append(row)

        return pd.DataFrame(rows)

    def summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-config results: mean ± std across seeds."""
        score_cols = [
            "completion_rate",
            "efficiency",
            "fidelity",
            "capability_ratio",
            "safety_score",
            "p",
        ]
        present = [c for c in score_cols if c in df.columns]
        return (
            df.groupby("gov_config")[present]
            .agg(["mean", "std"])
            .round(4)
        )
