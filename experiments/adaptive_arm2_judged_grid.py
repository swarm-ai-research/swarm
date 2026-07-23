"""Adaptive arm 2 — judged grid (calibration anchor integrated).

For each (ρ, seed) cell and each condition (adaptive + the three
static baselines), runs the full pipeline:

  1. Run an episode (adaptive: CEM-train then evaluate final policy;
     static: run the fixed policy).
  2. Score every accepted interaction through ``MockJudge`` under
     rubric v3.
  3. Emit per-cell metrics — proxy toxicity, judge toxicity, the
     proxy-vs-judge gap, fraction of items where the v3 target rule
     fired.

The target-rule firing fraction is the load-bearing diagnostic from
the [v3 cross-scenario findings](../docs/research/calibration-pilot-findings-v3.md#cross-scenario-check-added-after-main-v3-pilot):
the v3 anchor is reliable on agent_type-populated items (target rules
fire) and unreliable on label-only / no-metadata items (banded-prior
fallback).

By construction:

- ``adaptive`` items have **no metadata** (CEM-trained policy has
  empty ``identity_label``), so target rules fire 0% — the anchor
  is *uninformative* on adaptive items by design. This is the
  expected result given the policy class, not a bug.
- ``static_*`` items carry their canonical ``agent_type`` label, so
  target rules fire 100% — the anchor is fully informative there.

Output schema is one row per (condition, ρ, seed):
  judged_summary.csv

Usage:
    python -m experiments.adaptive_arm2_judged_grid
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from swarm.adaptive import (
    STATIC_BASELINES,
    CEMConfig,
    run_episode_with_interactions,
    run_population_episode_with_interactions,
    score_episode,
    train_cem,
)
from swarm.core.payoff import PayoffConfig
from swarm.judges import MockJudge

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
        help="ρ grid (default: pre-registered)",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=list(PREREG_SEEDS),
        help="seeds (default: pre-registered)",
    )
    parser.add_argument(
        "--baselines", nargs="+", default=list(STATIC_BASELINES),
        choices=list(STATIC_BASELINES),
        help="Static baselines to score (default: all three)",
    )
    parser.add_argument(
        "--rubric", default="rubric.v3", choices=["rubric.v1", "rubric.v2", "rubric.v3"],
        help="Rubric version for MockJudge (default: v3)",
    )
    parser.add_argument(
        "--interactions-per-episode", type=int, default=200,
        help="Interactions per episode (matches adaptive default of 200)",
    )
    parser.add_argument(
        "--population-size", type=int, default=30,
        help="CEM population size (default: 30)",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=10,
        help="CEM iterations (default: 10)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
    )
    args = parser.parse_args(argv)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.runs_dir / f"{ts}_adaptive_arm2_judged_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": _git_rev(),
        "rho_grid": list(args.rho),
        "seeds": list(args.seed),
        "baselines": list(args.baselines),
        "rubric": args.rubric,
        "cem": {
            "population_size": args.population_size,
            "n_iterations": args.n_iterations,
            "interactions_per_episode": args.interactions_per_episode,
        },
        "prereg": "docs/research/adaptive-agents-prereg.md",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    judge = MockJudge(rubric_version=args.rubric)
    rows: list[list] = []

    total_cells = (1 + len(args.baselines)) * len(args.rho) * len(args.seed)
    print(
        f"Judged grid: {1 + len(args.baselines)} conditions × "
        f"{len(args.rho)} ρ × {len(args.seed)} seeds = {total_cells} cells"
    )

    for rho in args.rho:
        for seed in args.seed:
            payoff = PayoffConfig(rho_a=rho, rho_b=rho)

            # Adaptive (CEM-train → judge final episode).
            cem_cfg = CEMConfig(
                population_size=args.population_size,
                n_iterations=args.n_iterations,
                interactions_per_episode=args.interactions_per_episode,
            )
            training = train_cem(payoff, cem_config=cem_cfg, seed=seed)
            final_policy = training.final_policy
            adaptive_report, adaptive_interactions = run_episode_with_interactions(
                final_policy,
                n_interactions=args.interactions_per_episode,
                payoff_config=payoff,
                seed=seed + 10_000,
            )
            adaptive_judged = score_episode(
                adaptive_interactions, judge,
                proxy_toxicity=adaptive_report.toxicity,
            )
            rows.append([
                "adaptive", f"{rho:.2f}", seed,
                adaptive_report.n_accepted,
                f"{adaptive_report.accept_rate:.6f}",
                f"{adaptive_report.toxicity:.6f}",
                f"{adaptive_judged.judge_toxicity:.6f}",
                f"{adaptive_judged.proxy_judge_gap:+.6f}",
                f"{adaptive_judged.mean_judge_score:.6f}",
                adaptive_judged.n_with_target_rule_fired,
                adaptive_judged.n_scored,
                f"{adaptive_report.mean_payoff_attempted:.6f}",
            ])
            print(
                f"  adaptive  ρ={rho:.2f}  seed={seed}  "
                f"proxy_tox={adaptive_report.toxicity:.3f}  "
                f"judge_tox={adaptive_judged.judge_toxicity:.3f}  "
                f"gap={adaptive_judged.proxy_judge_gap:+.3f}  "
                f"target_fired={adaptive_judged.n_with_target_rule_fired}/{adaptive_judged.n_scored}"
            )

            # Static baselines (run + judge directly).
            for baseline_name in args.baselines:
                baseline = STATIC_BASELINES[baseline_name]
                report, interactions = run_population_episode_with_interactions(
                    baseline,
                    n_interactions=args.interactions_per_episode,
                    payoff_config=payoff,
                    seed=seed,
                )
                judged = score_episode(
                    interactions, judge, proxy_toxicity=report.toxicity,
                )
                rows.append([
                    f"static_{baseline_name}", f"{rho:.2f}", seed,
                    report.n_accepted,
                    f"{report.accept_rate:.6f}",
                    f"{report.toxicity:.6f}",
                    f"{judged.judge_toxicity:.6f}",
                    f"{judged.proxy_judge_gap:+.6f}",
                    f"{judged.mean_judge_score:.6f}",
                    judged.n_with_target_rule_fired,
                    judged.n_scored,
                    f"{report.mean_payoff_attempted:.6f}",
                ])
                print(
                    f"  static_{baseline_name:6s}  ρ={rho:.2f}  seed={seed}  "
                    f"proxy_tox={report.toxicity:.3f}  "
                    f"judge_tox={judged.judge_toxicity:.3f}  "
                    f"gap={judged.proxy_judge_gap:+.3f}  "
                    f"target_fired={judged.n_with_target_rule_fired}/{judged.n_scored}"
                )

    summary_path = out_dir / "judged_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "condition", "rho", "seed",
            "n_accepted", "accept_rate",
            "proxy_toxicity",
            "judge_toxicity",
            "proxy_judge_gap",
            "mean_judge_score",
            "n_target_rule_fired",
            "n_scored",
            "mean_payoff_attempted",
        ])
        w.writerows(rows)

    print(f"Wrote {out_dir}")
    print(f"  rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
