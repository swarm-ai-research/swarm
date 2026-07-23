"""Adaptive arm 2 — static-baseline grid for adaptive-vs-static overlay.

Pre-registration: docs/research/adaptive-agents-prereg.md (arm 2,
"Static" condition — replication anchor).

Runs each named static baseline (`honest`, `toxic`, `mixed`) at the
same pre-registered (ρ × seed) grid the adaptive runner uses, with
no CEM training. Output schema matches ``grid_summary.csv`` from
``adaptive_arm2_grid.py`` so adaptive-vs-static is a direct row-join
on (rho, seed, baseline).

This is the wiring for the [adaptive-arm2-grid-findings.md] overlay
followup: "wire static agents through ``run_episode`` so the
adaptive-vs-static welfare curves are directly comparable."

Usage:
    python -m experiments.adaptive_arm2_static_grid
    python -m experiments.adaptive_arm2_static_grid --baseline mixed --rho 0.0 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from swarm.adaptive import STATIC_BASELINES, run_population_episode
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
        "--baseline", nargs="+", default=list(STATIC_BASELINES.keys()),
        choices=list(STATIC_BASELINES.keys()),
        help="Baselines to run (default: honest + toxic + mixed)",
    )
    parser.add_argument(
        "--rho", type=float, nargs="+", default=list(PREREG_RHO_GRID),
        help="ρ grid (default: pre-registered)",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=list(PREREG_SEEDS),
        help="seeds (default: pre-registered)",
    )
    parser.add_argument(
        "--interactions-per-episode", type=int, default=200,
        help="Interactions per episode (matches adaptive default of 200)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
    )
    args = parser.parse_args(argv)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    grid_dir = args.runs_dir / f"{ts}_adaptive_arm2_static_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": _git_rev(),
        "baselines": list(args.baseline),
        "rho_grid": list(args.rho),
        "seeds": list(args.seed),
        "interactions_per_episode": args.interactions_per_episode,
        "prereg": "docs/research/adaptive-agents-prereg.md#order-of-operations",
    }
    (grid_dir / "config.json").write_text(json.dumps(config, indent=2))

    rows: list[list] = []
    n_cells = len(args.baseline) * len(args.rho) * len(args.seed)
    print(
        f"Running static grid: {len(args.baseline)} baselines × "
        f"{len(args.rho)} ρ × {len(args.seed)} seeds = {n_cells} cells"
    )

    for baseline_name in args.baseline:
        baseline = STATIC_BASELINES[baseline_name]
        for rho in args.rho:
            for seed in args.seed:
                payoff = PayoffConfig(rho_a=rho, rho_b=rho)
                report = run_population_episode(
                    baseline,
                    n_interactions=args.interactions_per_episode,
                    payoff_config=payoff,
                    seed=seed,
                )
                rows.append([
                    baseline_name,
                    f"{rho:.2f}",
                    seed,
                    report.n_accepted,
                    f"{report.accept_rate:.6f}",
                    f"{report.mean_payoff_accepted:.6f}",
                    f"{report.mean_payoff_attempted:.6f}",
                    f"{report.sum_payoff:.6f}",
                    f"{report.toxicity:.6f}",
                    f"{report.mean_p:.6f}",
                    f"{report.mean_v_hat:.6f}",
                    f"{report.mean_progress:.6f}",
                ])
                print(
                    f"  {baseline_name:7s} ρ={rho:.2f}  seed={seed}  "
                    f"accept={report.accept_rate:.3f}  "
                    f"per_attempted={report.mean_payoff_attempted:.3f}  "
                    f"toxicity={report.toxicity:.3f}"
                )

    summary_path = grid_dir / "static_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "baseline", "rho", "seed",
            "n_accepted", "accept_rate",
            "mean_payoff_accepted",
            "mean_payoff_attempted",
            "sum_payoff",
            "toxicity",
            "mean_p",
            "mean_v_hat",
            "mean_progress",
        ])
        w.writerows(rows)

    print(f"Wrote {grid_dir}")
    print(f"  rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
