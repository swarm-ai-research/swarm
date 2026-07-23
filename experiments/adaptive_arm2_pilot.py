"""Adaptive arm 2 — CEM-trained generation policy pilot.

Pre-registration: docs/research/adaptive-agents-prereg.md (arm 2,
adaptive-generation).

This is the **single-condition smoke**. The full pre-registered run
sweeps rho ∈ {0, 0.1, 0.3, 0.5, 0.7, 1.0} × seeds {42, 123, 456,
789, 1024}; this script does **one** (rho, seed) pair so we can see
whether the CEM trainer learns anything at all before committing to
the powered run.

Outputs:
  runs/<ts>_adaptive_arm2_pilot_rho<rho>_seed<seed>/
    training_report.json       # full trajectory + final policy
    iterations.csv             # one row per CEM iteration
    config.json                # config + git rev

Usage:
    python -m experiments.adaptive_arm2_pilot --rho 0.3 --seed 42

The pre-registration requires:
  - fixed training budget reported (see CEMConfig defaults)
  - learning curve reported (iterations.csv + the JSON trajectory)
  - failed/degenerate runs reported, not just converged ones
The runner prints a one-line summary that highlights if the elite
payoff *decreased* across training (a degenerate-run flag).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from swarm.adaptive import CEMConfig, train_cem
from swarm.core.payoff import PayoffConfig


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rho", type=float, required=True,
        help="Externality internalization for the initiator (rho_a). "
             "Pre-registered grid: {0, 0.1, 0.3, 0.5, 0.7, 1.0}.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--population-size", type=int, default=30,
        help="CEM population (pre-reg default 30)",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=10,
        help="CEM iterations (pre-reg default 10)",
    )
    parser.add_argument(
        "--interactions-per-episode", type=int, default=200,
        help="Interactions per CEM episode (pre-reg default 200)",
    )
    parser.add_argument(
        "--reward", default="mean_attempted",
        choices=["mean_attempted", "mean_accepted", "sum_attempted"],
        help="CEM elite-selection reward. Default is the pinned value.",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
        help="Parent directory for run output",
    )
    args = parser.parse_args(argv)

    payoff = PayoffConfig(rho_a=args.rho, rho_b=args.rho)
    cem_cfg = CEMConfig(
        population_size=args.population_size,
        n_iterations=args.n_iterations,
        interactions_per_episode=args.interactions_per_episode,
        reward=args.reward,
    )
    report = train_cem(payoff, cem_config=cem_cfg, seed=args.seed)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        args.runs_dir
        / f"{ts}_adaptive_arm2_pilot_rho{args.rho}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": _git_rev(),
        "rho": args.rho,
        "seed": args.seed,
        "cem": {
            "population_size": cem_cfg.population_size,
            "elite_fraction": cem_cfg.elite_fraction,
            "n_elites": cem_cfg.n_elites,
            "n_iterations": cem_cfg.n_iterations,
            "interactions_per_episode": cem_cfg.interactions_per_episode,
        },
        "prereg": "docs/research/adaptive-agents-prereg.md#arm-2-adaptive-generation-primary",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "training_report.json").write_text(json.dumps(report.to_dict(), indent=2))

    # Learning-curve CSV: one row per CEM iteration.
    iters_path = run_dir / "iterations.csv"
    with iters_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iteration",
            "mean_elite_reward",
            "mean_elite_payoff_accepted",
            "mean_elite_payoff_attempted",
            "mean_elite_sum_payoff",
            "mean_elite_toxicity",
            "mean_elite_accept_rate",
        ])
        for it in report.iterations:
            w.writerow([
                it.iteration,
                f"{it.mean_elite_reward:.6f}",
                f"{it.mean_elite_payoff_accepted:.6f}",
                f"{it.mean_elite_payoff_attempted:.6f}",
                f"{it.mean_elite_sum_payoff:.6f}",
                f"{it.mean_elite_toxicity:.6f}",
                f"{it.mean_elite_accept_rate:.6f}",
            ])

    first = report.iterations[0]
    last = report.iterations[-1]
    delta = last.mean_elite_reward - first.mean_elite_reward
    flag = " DEGENERATE" if delta < 0 else ""
    print(f"Wrote {run_dir}")
    print(
        f"  rho={args.rho}  seed={args.seed}  reward={args.reward}"
        f"  iters={len(report.iterations)}"
        f"  reward: {first.mean_elite_reward:.3f} → {last.mean_elite_reward:.3f}"
        f" (Δ={delta:+.3f}){flag}"
    )
    print(
        f"  toxicity: {first.mean_elite_toxicity:.3f} → {last.mean_elite_toxicity:.3f}"
        f"  accept_rate: {first.mean_elite_accept_rate:.3f}"
        f" → {last.mean_elite_accept_rate:.3f}"
    )
    fe = report.final_episode
    print(
        f"  final: n_accepted={fe.n_accepted}"
        f"  per_accepted={fe.mean_payoff_accepted:.3f}"
        f"  per_attempted={fe.mean_payoff_attempted:.3f}"
        f"  toxicity={fe.toxicity:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
