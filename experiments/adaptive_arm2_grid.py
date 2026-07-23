"""Adaptive arm 2 — pre-registered ρ × seed grid (powered run).

Pre-registration: docs/research/adaptive-agents-prereg.md (arm 2,
adaptive-generation primary).

Sweeps:
  ρ ∈ {0, 0.1, 0.3, 0.5, 0.7, 1.0}     (pre-reg grid; matches Figure 4)
  seed ∈ {42, 123, 456, 789, 1024}     (pre-reg seeds; match static)

For each (ρ, seed) pair: run CEM with the pinned reward
(``mean_attempted``) and the pre-registered budget. Write per-cell
training artifacts plus a single ``grid_summary.csv`` that the next
step can join against the Figure 4 static-baseline trajectory.

Usage:
    python -m experiments.adaptive_arm2_grid
    python -m experiments.adaptive_arm2_grid --rho 0.0 0.3 1.0   # subset

The script is fully sequential (no parallelism). At ~6 sec per
(ρ, seed) cell, the full 30-cell grid is ~3 minutes on one core.
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

PREREG_RHO_GRID: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
PREREG_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)


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
        "--rho", type=float, nargs="+", default=list(PREREG_RHO_GRID),
        help="ρ grid to sweep (default: pre-registered)",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=list(PREREG_SEEDS),
        help="seeds to sweep (default: pre-registered)",
    )
    parser.add_argument(
        "--reward", default="mean_attempted",
        choices=["mean_attempted", "mean_accepted", "sum_attempted"],
        help="CEM elite-selection reward (default: pinned)",
    )
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
        "--runs-dir", type=Path, default=Path("runs"),
    )
    args = parser.parse_args(argv)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    grid_dir = args.runs_dir / f"{ts}_adaptive_arm2_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": _git_rev(),
        "rho_grid": list(args.rho),
        "seeds": list(args.seed),
        "reward": args.reward,
        "cem": {
            "population_size": args.population_size,
            "n_iterations": args.n_iterations,
            "interactions_per_episode": args.interactions_per_episode,
        },
        "prereg": "docs/research/adaptive-agents-prereg.md#arm-2-adaptive-generation-primary",
    }
    (grid_dir / "config.json").write_text(json.dumps(config, indent=2))

    summary_path = grid_dir / "grid_summary.csv"
    cell_summary_rows: list[list] = []

    print(f"Running grid: {len(args.rho)} ρ × {len(args.seed)} seeds = "
          f"{len(args.rho) * len(args.seed)} cells")

    for rho in args.rho:
        for seed in args.seed:
            payoff = PayoffConfig(rho_a=rho, rho_b=rho)
            cem_cfg = CEMConfig(
                population_size=args.population_size,
                n_iterations=args.n_iterations,
                interactions_per_episode=args.interactions_per_episode,
                reward=args.reward,
            )
            report = train_cem(payoff, cem_config=cem_cfg, seed=seed)

            cell_dir = grid_dir / f"rho{rho}_seed{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "training_report.json").write_text(
                json.dumps(report.to_dict(), indent=2)
            )

            first = report.iterations[0]
            last = report.iterations[-1]
            fe = report.final_episode
            degenerate = last.mean_elite_reward < first.mean_elite_reward
            cell_summary_rows.append([
                f"{rho:.2f}", seed,
                f"{first.mean_elite_reward:.6f}",
                f"{last.mean_elite_reward:.6f}",
                f"{first.mean_elite_toxicity:.6f}",
                f"{last.mean_elite_toxicity:.6f}",
                f"{first.mean_elite_accept_rate:.6f}",
                f"{last.mean_elite_accept_rate:.6f}",
                f"{fe.n_accepted}",
                f"{fe.mean_payoff_accepted:.6f}",
                f"{fe.mean_payoff_attempted:.6f}",
                f"{fe.toxicity:.6f}",
                f"{fe.mean_p:.6f}",
                f"{fe.accept_rate:.6f}",
                "1" if degenerate else "0",
            ])
            flag = " DEGENERATE" if degenerate else ""
            print(
                f"  ρ={rho:.2f}  seed={seed}  "
                f"reward {first.mean_elite_reward:.3f}→{last.mean_elite_reward:.3f}  "
                f"toxicity {first.mean_elite_toxicity:.3f}→{last.mean_elite_toxicity:.3f}  "
                f"accept {first.mean_elite_accept_rate:.3f}→{last.mean_elite_accept_rate:.3f}"
                f"{flag}"
            )

    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rho", "seed",
            "iter0_reward", "iterN_reward",
            "iter0_toxicity", "iterN_toxicity",
            "iter0_accept_rate", "iterN_accept_rate",
            "final_n_accepted",
            "final_mean_payoff_accepted",
            "final_mean_payoff_attempted",
            "final_toxicity",
            "final_mean_p",
            "final_accept_rate",
            "degenerate",
        ])
        w.writerows(cell_summary_rows)

    n_degenerate = sum(1 for row in cell_summary_rows if row[-1] == "1")
    print(f"Wrote {grid_dir}")
    print(f"  cells: {len(cell_summary_rows)}, degenerate: {n_degenerate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
