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
        [--whistleblowers 0,0.24] [--wb-audit-rate 0.5] [--wb-fpr 0]
        [--wb-suspect-quality 0.65]
        [--wb-warning 0,0.5] [--wb-boycott 0]

The --rankings axis (bead 2qfq) tests whether the no-reset epidemic burnout
is an artifact of the quality-ranked hot cache.

The --whistleblowers axis (arXiv:2609.04170, the DeepMind 100-agent Lean
research swarm) adds an endogenous counter-response: a share of the honest
population that refuses the exploit, audits the shared cache, warns peers
(--wb-warning), and may boycott the task once it has caught fraud
(--wb-boycott). Cells with a zero whistleblower share run only the first
warning/boycott value, since both are no-ops without a faction. The audit
is a noisy classifier over an observable signal, never the hidden poison
label: an entry whose quality_score is below --wb-suspect-quality is
flagged per member with --wb-audit-rate, any other entry with --wb-fpr.

The --lockout axis (same paper, the converts) turns on a pool of problems
that pay once: fakes close them, exposed honest agents drift toward the
exploit as their payoff disappears, and --lockout-reopen controls whether
a whistleblower revert gives the problem back.
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
        help="per-whistleblower per-epoch P(flag) per suspect cache entry",
    )
    parser.add_argument(
        "--wb-fpr",
        type=float,
        default=0.0,
        help="per-whistleblower per-epoch P(flag) per non-suspect cache entry",
    )
    parser.add_argument(
        "--wb-suspect-quality",
        type=float,
        default=0.65,
        help="entries with quality_score below this are suspect to the audit",
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
    parser.add_argument(
        "--lockout",
        default="0",
        help="comma-separated lockout flags (0 = no problem pool, 1 = pool on)",
    )
    parser.add_argument(
        "--lockout-arrivals",
        type=int,
        default=None,
        help="override lockout_arrivals_per_epoch (new problems per epoch)",
    )
    parser.add_argument(
        "--lockout-reopen",
        default="1",
        help="comma-separated flags: does a whistleblower revert reopen the problem",
    )
    args = parser.parse_args()
    n_seeds = 2 if args.quick else args.seeds
    lockouts = [int(x) for x in args.lockout.split(",")]
    reopens = [int(x) for x in args.lockout_reopen.split(",")]
    rankings = args.rankings.split(",")
    cadences = [int(c) for c in args.cadences.split(",")]
    wb_fractions = [float(w) for w in args.whistleblowers.split(",")]
    wb_warnings = [float(w) for w in args.wb_warning.split(",")]
    wb_boycotts = [float(b) for b in args.wb_boycott.split(",")]

    def wb_cells():
        for lockout in lockouts:
            # Reopen only matters with both a pool and a faction.
            for reopen in reopens if lockout else reopens[:1]:
                for fraction in wb_fractions:
                    if fraction == 0.0:
                        if reopen != reopens[0]:
                            continue
                        yield fraction, wb_warnings[0], wb_boycotts[0], lockout, reopen
                        continue
                    for warning in wb_warnings:
                        for boycott in wb_boycotts:
                            yield fraction, warning, boycott, lockout, reopen

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
                for wb_fraction, wb_warning, wb_boycott, lockout, reopen in wb_grid:
                    for seed_offset in range(n_seeds):
                        seed = SEED_BASE + seed_offset
                        run_num = len(rows) + 1
                        print(
                            f"[{run_num}/{total}] ranking={ranking} "
                            f"detection={detection} reset_cadence={cadence} "
                            f"whistleblowers={wb_fraction} "
                            f"warning={wb_warning} boycott={wb_boycott} "
                            f"lockout={lockout} reopen={reopen} "
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
                        mem.whistleblower_false_positive_rate = (
                            args.wb_fpr if wb_fraction > 0 else 0.0
                        )
                        mem.whistleblower_suspect_quality = args.wb_suspect_quality
                        mem.whistleblower_warning_strength = wb_warning
                        mem.whistleblower_boycott_rate = wb_boycott
                        mem.lockout_enabled = bool(lockout)
                        mem.lockout_revert_reopens = bool(reopen)
                        if args.lockout_arrivals is not None:
                            mem.lockout_arrivals_per_epoch = args.lockout_arrivals
                        apply_detection(s, detection)
                        # The scenario's event log appends every run to one
                        # JSONL (~10 MB per run; a 200-run sweep wrote 2.3 GB).
                        # The sweep's CSVs and summary.json are its record.
                        s.orchestrator_config.log_events = False
                        s.orchestrator_config.log_path = None

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
                            "lockout": lockout,
                            "lockout_reopen": reopen,
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
                            "whistleblower_false_reverts": (
                                handler.whistleblower_false_revert_count
                            ),
                            "boycotted_writes": handler.boycotted_write_count,
                            "mean_locked_out_rate": sum(
                                sn.get("locked_out_rate", 0.0) for sn in snaps
                            ) / len(snaps),
                            "fake_closure_share": (
                                snaps[-1].get("fake_closure_share", 0.0)
                            ),
                            "mean_lockout_pressure": sum(
                                sn.get("lockout_pressure", 0.0) for sn in snaps
                            ) / len(snaps),
                            "peak_converts": max(
                                sn.get("converts", 0) for sn in snaps
                            ),
                            "final_converts": snaps[-1].get("converts", 0),
                            "locked_out_writes": handler.locked_out_write_count,
                            "reopened": handler.reopened_count,
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
            f"whistleblower_cells (fraction, warning, boycott, lockout, reopen): {wb_grid}\n"
            f"whistleblower_audit_rate: {args.wb_audit_rate}\n"
            f"whistleblower_false_positive_rate: {args.wb_fpr}\n"
            f"whistleblower_suspect_quality: {args.wb_suspect_quality}\n"
            f"lockout_arrivals_per_epoch: "
            f"{args.lockout_arrivals if args.lockout_arrivals is not None else 'scenario default'}\n"
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
                r["lockout"],
                r["lockout_reopen"],
            )
        ].append(r)

    header = (
        f"{'ranking':>10} {'det':>4} {'cad':>4} {'wb':>5} {'warn':>5} "
        f"{'boyc':>5} {'lock':>4} {'reop':>4}  {'peak_inf':>8}  "
        f"{'late_inf':>8}  {'late_sus':>8}  {'tier3_poi':>9}  "
        f"{'contagion_w':>11}  {'reverts':>7}  {'fakeclose':>9}  "
        f"{'converts':>8}  {'toxicity':>8}  {'welfare':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    summary = {}
    for ranking in rankings:
        for detection in DETECTION_MODES:
            for cadence in cadences:
                for wb_fraction, wb_warning, wb_boycott, lockout, reopen in wb_grid:
                    runs = grouped[
                        (
                            ranking,
                            detection,
                            cadence,
                            wb_fraction,
                            wb_warning,
                            wb_boycott,
                            lockout,
                            reopen,
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
                        "whistleblower_false_reverts": mean(
                            "whistleblower_false_reverts"
                        ),
                        "boycotted_writes": mean("boycotted_writes"),
                        "mean_locked_out_rate": mean("mean_locked_out_rate"),
                        "fake_closure_share": mean("fake_closure_share"),
                        "mean_lockout_pressure": mean("mean_lockout_pressure"),
                        "peak_converts": mean("peak_converts"),
                        "final_converts": mean("final_converts"),
                        "locked_out_writes": mean("locked_out_writes"),
                        "reopened": mean("reopened"),
                        "avg_toxicity": mean("avg_toxicity"),
                        "avg_quality_gap": mean("avg_quality_gap"),
                        "total_welfare": mean("total_welfare"),
                    }
                    key = (
                        f"{ranking}_{detection}_cadence{cadence}"
                        f"_wb{wb_fraction}_warn{wb_warning}_boycott{wb_boycott}"
                        f"_lock{lockout}_reopen{reopen}"
                    )
                    summary[key] = agg
                    print(
                        f"{ranking:>10} {detection:>4} {cadence:>4} "
                        f"{wb_fraction:>5.2f} {wb_warning:>5.2f} "
                        f"{wb_boycott:>5.2f} {lockout:>4d} {reopen:>4d}  "
                        f"{agg['peak_infection']:8.3f}  "
                        f"{agg['late_infection']:8.3f}  "
                        f"{agg['late_susceptible_infection']:8.3f}  "
                        f"{agg['mean_tier3_poisoning']:9.3f}  "
                        f"{agg['contagion_writes']:11.1f}  "
                        f"{agg['whistleblower_reverts']:7.1f}  "
                        f"{agg['mean_locked_out_rate']:9.3f}  "
                        f"{agg['peak_converts']:8.2f}  "
                        f"{agg['avg_toxicity']:8.3f}  "
                        f"{agg['total_welfare']:8.1f}"
                    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
