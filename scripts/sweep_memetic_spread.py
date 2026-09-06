#!/usr/bin/env python3
"""Memetic spread countermeasure sweep: reset cadence x detection.

Sweeps the two countermeasure families from Mallen (2025) against the
memetic_spread scenario:

- Prevention: memory reset cadence (wipe the shared store every N epochs;
  infection state deliberately survives — values live in the agents).
- Detection: memory promotion gate + cross-verification + provenance.

Usage:
    python scripts/sweep_memetic_spread.py [--seeds N] [--quick]
        [--rankings quality,recency,engagement] [--cadences 0,2,5,10]

The --rankings axis (bead 2qfq) tests whether the no-reset epidemic burnout
is an artifact of the quality-ranked hot cache.
"""

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from swarm.analysis.sweep import _extract_results  # noqa: E402
from swarm.scenarios import build_orchestrator, load_scenario  # noqa: E402

DEFAULT_CADENCES = [0, 2, 5, 10]
DEFAULT_RANKINGS = ["quality"]
DETECTION_MODES = ["off", "on"]
SEED_BASE = 42
SCENARIO_PATH = PROJECT_ROOT / "scenarios" / "memetic_spread.yaml"


def apply_detection(scenario, mode: str) -> None:
    g = scenario.orchestrator_config.governance_config
    enabled = mode == "on"
    g.memory_promotion_gate_enabled = enabled
    g.memory_promotion_min_quality = 0.5
    g.memory_promotion_min_verifications = 2
    g.memory_cross_verification_enabled = enabled
    g.memory_cross_verification_k = 2
    g.memory_provenance_enabled = enabled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="2 seeds, for smoke")
    parser.add_argument(
        "--rankings",
        default=",".join(DEFAULT_RANKINGS),
        help="comma-separated cache ranking policies",
    )
    parser.add_argument(
        "--cadences",
        default=",".join(str(c) for c in DEFAULT_CADENCES),
        help="comma-separated reset cadences",
    )
    args = parser.parse_args()
    n_seeds = 2 if args.quick else args.seeds
    rankings = args.rankings.split(",")
    cadences = [int(c) for c in args.cadences.split(",")]

    scenario = load_scenario(SCENARIO_PATH)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROJECT_ROOT / "runs" / f"{timestamp}_memetic_spread_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(rankings) * len(cadences) * len(DETECTION_MODES) * n_seeds
    rows = []
    series = []  # per-epoch time series, one row per (condition, seed, epoch)

    for ranking in rankings:
        for detection in DETECTION_MODES:
            for cadence in cadences:
                for seed_offset in range(n_seeds):
                    seed = SEED_BASE + seed_offset
                    run_num = len(rows) + 1
                    print(
                        f"[{run_num}/{total}] ranking={ranking} "
                        f"detection={detection} reset_cadence={cadence} "
                        f"seed={seed}",
                        flush=True,
                    )

                    s = copy.deepcopy(scenario)
                    s.orchestrator_config.seed = seed
                    mem = s.orchestrator_config.memory_tier_config
                    mem.seed = seed
                    mem.reset_cadence_epochs = cadence
                    mem.cache_ranking = ranking
                    apply_detection(s, detection)

                    orch = build_orchestrator(s)
                    orch.run()

                    handler = orch._memory_handler
                    snaps = handler.epoch_snapshots
                    result = _extract_results(
                        orch,
                        {
                            "ranking": ranking,
                            "detection": detection,
                            "reset_cadence": cadence,
                        },
                        seed_offset,
                        seed,
                    )

                    infections = [sn["mean_infection"] for sn in snaps]
                    tier3 = [sn["tier3_poisoning"] for sn in snaps]
                    cache = [sn["cache_corruption"] for sn in snaps]
                    late = infections[-10:]

                    row = result.to_dict()
                    row.update({
                        "peak_infection": max(infections),
                        "mean_infection": sum(infections) / len(infections),
                        "final_infection": infections[-1],
                        "late_infection": sum(late) / len(late),
                        "mean_tier3_poisoning": sum(tier3) / len(tier3),
                        "final_tier3_poisoning": tier3[-1],
                        "mean_cache_corruption": sum(cache) / len(cache),
                        "contagion_writes": handler.contagion_write_count,
                    })
                    rows.append(row)

                    for sn in snaps:
                        series.append({
                            "ranking": ranking,
                            "detection": detection,
                            "reset_cadence": cadence,
                            "seed": seed,
                            **sn,
                        })

    # ── Export ──────────────────────────────────────────────────
    csv_path = out_dir / "sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    series_path = out_dir / "epoch_series.csv"
    with open(series_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(series[0].keys()))
        writer.writeheader()
        writer.writerows(series)

    with open(out_dir / "run.yaml", "w") as f:
        f.write(
            "scenario: scenarios/memetic_spread.yaml\n"
            f"sweep: ranking x reset_cadence x detection\n"
            f"rankings: {rankings}\n"
            f"reset_cadences: {cadences}\n"
            f"detection_modes: {DETECTION_MODES}\n"
            f"seeds: {n_seeds} (base {SEED_BASE})\n"
            f"timestamp: {timestamp}\n"
        )

    # ── Summary ─────────────────────────────────────────────────
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["ranking"], r["detection"], r["reset_cadence"])].append(r)

    header = (
        f"{'ranking':>10} {'det':>4} {'cad':>4}  {'peak_inf':>8}  "
        f"{'late_inf':>8}  {'tier3_poi':>9}  {'contagion_w':>11}  "
        f"{'toxicity':>8}  {'welfare':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    summary = {}
    for ranking in rankings:
        for detection in DETECTION_MODES:
            for cadence in cadences:
                runs = grouped[(ranking, detection, cadence)]
                n = len(runs)
                agg = {
                    "peak_infection": sum(r["peak_infection"] for r in runs) / n,
                    "mean_infection": sum(r["mean_infection"] for r in runs) / n,
                    "late_infection": sum(r["late_infection"] for r in runs) / n,
                    "mean_tier3_poisoning": sum(
                        r["mean_tier3_poisoning"] for r in runs
                    ) / n,
                    "contagion_writes": sum(
                        r["contagion_writes"] for r in runs
                    ) / n,
                    "avg_toxicity": sum(r["avg_toxicity"] for r in runs) / n,
                    "avg_quality_gap": sum(r["avg_quality_gap"] for r in runs)
                    / n,
                    "total_welfare": sum(r["total_welfare"] for r in runs) / n,
                }
                summary[f"{ranking}_{detection}_cadence{cadence}"] = agg
                print(
                    f"{ranking:>10} {detection:>4} {cadence:>4}  "
                    f"{agg['peak_infection']:8.3f}  "
                    f"{agg['late_infection']:8.3f}  "
                    f"{agg['mean_tier3_poisoning']:9.3f}  "
                    f"{agg['contagion_writes']:11.1f}  "
                    f"{agg['avg_toxicity']:8.3f}  "
                    f"{agg['total_welfare']:8.1f}"
                )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
