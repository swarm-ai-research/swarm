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
        [--whistleblowers 0,0.24] [--wb-audit-rate 0.5]
        [--wb-warning 0,0.5] [--wb-boycott 0]

The --rankings axis (bead 2qfq) tests whether the no-reset epidemic burnout
is an artifact of the quality-ranked hot cache.

The --whistleblowers axis (arXiv:2609.04170, the DeepMind 100-agent Lean
research swarm) adds an endogenous counter-response: a share of the honest
population that refuses the exploit, audits the shared cache, warns peers
(--wb-warning), and may boycott the task once it has caught fraud
(--wb-boycott). Cells with a zero whistleblower share run only the first
warning/boycott value, since both are no-ops without a faction.
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
    parser.add_argument(
        "--whistleblowers",
        default="0",
        help="comma-separated whistleblower shares of the honest roster",
    )
    parser.add_argument(
        "--wb-audit-rate",
        type=float,
        default=0.5,
        help="per-whistleblower per-epoch P(catch) per poisoned cache entry",
    )
    parser.add_argument(
        "--wb-warning",
        default="0",
        help="comma-separated peer-warning strengths",
    )
    parser.add_argument(
        "--wb-boycott",
        default="0",
        help="comma-separated boycott rates",
    )
    args = parser.parse_args()
    n_seeds = 2 if args.quick else args.seeds
    rankings = args.rankings.split(",")
    cadences = [int(c) for c in args.cadences.split(",")]
    wb_fractions = [float(w) for w in args.whistleblowers.split(",")]
    wb_warnings = [float(w) for w in args.wb_warning.split(",")]
    wb_boycotts = [float(b) for b in args.wb_boycott.split(",")]

    def wb_cells():
        for fraction in wb_fractions:
            if fraction == 0.0:
                yield fraction, wb_warnings[0], wb_boycotts[0]
                continue
            for warning in wb_warnings:
                for boycott in wb_boycotts:
                    yield fraction, warning, boycott

    wb_grid = list(wb_cells())

    scenario = load_scenario(SCENARIO_PATH)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROJECT_ROOT / "runs" / f"{timestamp}_memetic_spread_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = (
        len(rankings) * len(cadences) * len(DETECTION_MODES) * len(wb_grid) * n_seeds
    )
    rows = []
    series = []  # per-epoch time series, one row per (condition, seed, epoch)

    for ranking in rankings:
        for detection in DETECTION_MODES:
            for cadence in cadences:
                for wb_fraction, wb_warning, wb_boycott in wb_grid:
                    for seed_offset in range(n_seeds):
                        seed = SEED_BASE + seed_offset
                        run_num = len(rows) + 1
                        print(
                            f"[{run_num}/{total}] ranking={ranking} "
                            f"detection={detection} reset_cadence={cadence} "
                            f"whistleblowers={wb_fraction} "
                            f"warning={wb_warning} boycott={wb_boycott} "
                            f"seed={seed}",
                            flush=True,
                        )

                        s = copy.deepcopy(scenario)
                        s.orchestrator_config.seed = seed
                        mem = s.orchestrator_config.memory_tier_config
                        mem.seed = seed
                        mem.reset_cadence_epochs = cadence
                        mem.cache_ranking = ranking
                        mem.whistleblower_fraction = wb_fraction
                        mem.whistleblower_audit_rate = (
                            args.wb_audit_rate if wb_fraction > 0 else 0.0
                        )
                        mem.whistleblower_warning_strength = wb_warning
                        mem.whistleblower_boycott_rate = wb_boycott
                        apply_detection(s, detection)

                        orch = build_orchestrator(s)
                        orch.run()

                        handler = orch._memory_handler
                        snaps = handler.epoch_snapshots
                        condition = {
                            "ranking": ranking,
                            "detection": detection,
                            "reset_cadence": cadence,
                            "whistleblowers": wb_fraction,
                            "wb_warning": wb_warning,
                            "wb_boycott": wb_boycott,
                        }
                        result = _extract_results(
                            orch, condition, seed_offset, seed
                        )

                        infections = [sn["mean_infection"] for sn in snaps]
                        susceptible = [sn["susceptible_infection"] for sn in snaps]
                        tier3 = [sn["tier3_poisoning"] for sn in snaps]
                        cache = [sn["cache_corruption"] for sn in snaps]
                        late = infections[-10:]
                        late_susceptible = susceptible[-10:]

                        row = result.to_dict()
                        row.update({
                            "peak_infection": max(infections),
                            "mean_infection": sum(infections) / len(infections),
                            "final_infection": infections[-1],
                            "late_infection": sum(late) / len(late),
                            "peak_susceptible_infection": max(susceptible),
                            "late_susceptible_infection": (
                                sum(late_susceptible) / len(late_susceptible)
                            ),
                            "mean_tier3_poisoning": sum(tier3) / len(tier3),
                            "final_tier3_poisoning": tier3[-1],
                            "mean_cache_corruption": sum(cache) / len(cache),
                            "contagion_writes": handler.contagion_write_count,
                            "whistleblower_reverts": (
                                handler.whistleblower_revert_count
                            ),
                            "boycotted_writes": handler.boycotted_write_count,
                        })
                        rows.append(row)

                        for sn in snaps:
                            series.append({**condition, "seed": seed, **sn})

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
            f"sweep: ranking x reset_cadence x detection x whistleblowers\n"
            f"rankings: {rankings}\n"
            f"reset_cadences: {cadences}\n"
            f"detection_modes: {DETECTION_MODES}\n"
            f"whistleblower_cells (fraction, warning, boycott): {wb_grid}\n"
            f"whistleblower_audit_rate: {args.wb_audit_rate}\n"
            f"seeds: {n_seeds} (base {SEED_BASE})\n"
            f"timestamp: {timestamp}\n"
        )

    # ── Summary ─────────────────────────────────────────────────
    grouped = defaultdict(list)
    for r in rows:
        grouped[
            (
                r["ranking"],
                r["detection"],
                r["reset_cadence"],
                r["whistleblowers"],
                r["wb_warning"],
                r["wb_boycott"],
            )
        ].append(r)

    header = (
        f"{'ranking':>10} {'det':>4} {'cad':>4} {'wb':>5} {'warn':>5} "
        f"{'boyc':>5}  {'peak_inf':>8}  {'late_inf':>8}  {'late_sus':>8}  "
        f"{'tier3_poi':>9}  {'contagion_w':>11}  {'reverts':>7}  "
        f"{'toxicity':>8}  {'welfare':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    summary = {}
    for ranking in rankings:
        for detection in DETECTION_MODES:
            for cadence in cadences:
                for wb_fraction, wb_warning, wb_boycott in wb_grid:
                    runs = grouped[
                        (
                            ranking,
                            detection,
                            cadence,
                            wb_fraction,
                            wb_warning,
                            wb_boycott,
                        )
                    ]
                    n = len(runs)

                    def mean(key, runs=runs, n=n):
                        return sum(r[key] for r in runs) / n

                    agg = {
                        "peak_infection": mean("peak_infection"),
                        "mean_infection": mean("mean_infection"),
                        "late_infection": mean("late_infection"),
                        "peak_susceptible_infection": mean(
                            "peak_susceptible_infection"
                        ),
                        "late_susceptible_infection": mean(
                            "late_susceptible_infection"
                        ),
                        "mean_tier3_poisoning": mean("mean_tier3_poisoning"),
                        "contagion_writes": mean("contagion_writes"),
                        "whistleblower_reverts": mean("whistleblower_reverts"),
                        "boycotted_writes": mean("boycotted_writes"),
                        "avg_toxicity": mean("avg_toxicity"),
                        "avg_quality_gap": mean("avg_quality_gap"),
                        "total_welfare": mean("total_welfare"),
                    }
                    key = (
                        f"{ranking}_{detection}_cadence{cadence}"
                        f"_wb{wb_fraction}_warn{wb_warning}_boycott{wb_boycott}"
                    )
                    summary[key] = agg
                    print(
                        f"{ranking:>10} {detection:>4} {cadence:>4} "
                        f"{wb_fraction:>5.2f} {wb_warning:>5.2f} "
                        f"{wb_boycott:>5.2f}  "
                        f"{agg['peak_infection']:8.3f}  "
                        f"{agg['late_infection']:8.3f}  "
                        f"{agg['late_susceptible_infection']:8.3f}  "
                        f"{agg['mean_tier3_poisoning']:9.3f}  "
                        f"{agg['contagion_writes']:11.1f}  "
                        f"{agg['whistleblower_reverts']:7.1f}  "
                        f"{agg['avg_toxicity']:8.3f}  "
                        f"{agg['total_welfare']:8.1f}"
                    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
