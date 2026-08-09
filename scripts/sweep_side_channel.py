#!/usr/bin/env python3
"""Side-channel coordination sweep (bead k5o9).

Sweeps the governance responses to an emergent ungoverned side channel,
modeled on the 2026 OpenAI/Hugging Face incident (Wallace & Dalton, Black
Hat 2026-08-06):

- Teardown (detection): side_detection_rate — probability, scaled by side
  traffic, that the channel is discovered and torn down each epoch.
- Teardown mode: deletion-only (rebuild-from-discovery) vs revocation
  (side_teardown_resets_discovery, i.e. revoke credentials, not just delete).
- Friction competition: side_write_preference — a proxy for how much better
  the governed channel is. Lowering it models making the sanctioned channel
  cheap/good enough that agents don't route around it.

The headline question (bead hypothesis): does teardown-without-revocation
merely relocate coordination (predicting the Jul 4 -> Jul 8 rebuild), while
lowering the governed channel's friction is what actually suppresses the
ungoverned fraction?

Usage:
    python scripts/sweep_side_channel.py [--seeds N] [--quick]
        [--detections 0.0,0.02,0.05] [--prefs 0.3,0.7]
        [--modes deletion,revocation]
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

DEFAULT_DETECTIONS = [0.0, 0.02, 0.05]
DEFAULT_PREFS = [0.3, 0.7]
DEFAULT_MODES = ["deletion", "revocation"]
SEED_BASE = 42
SCENARIO_PATH = PROJECT_ROOT / "scenarios" / "side_channel.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="2 seeds, for smoke")
    parser.add_argument(
        "--detections",
        default=",".join(str(d) for d in DEFAULT_DETECTIONS),
        help="comma-separated side_detection_rate values",
    )
    parser.add_argument(
        "--prefs",
        default=",".join(str(p) for p in DEFAULT_PREFS),
        help="comma-separated side_write_preference values (governed-channel friction proxy)",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="teardown modes: deletion (rebuild) | revocation (revoke discovery)",
    )
    args = parser.parse_args()
    n_seeds = 2 if args.quick else args.seeds
    detections = [float(d) for d in args.detections.split(",")]
    prefs = [float(p) for p in args.prefs.split(",")]
    modes = args.modes.split(",")

    scenario = load_scenario(SCENARIO_PATH)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROJECT_ROOT / "runs" / f"{timestamp}_side_channel_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(modes) * len(detections) * len(prefs) * n_seeds
    rows = []
    series = []

    for mode in modes:
        for detection in detections:
            for pref in prefs:
                for seed_offset in range(n_seeds):
                    seed = SEED_BASE + seed_offset
                    run_num = len(rows) + 1
                    print(
                        f"[{run_num}/{total}] mode={mode} "
                        f"detection={detection} write_pref={pref} seed={seed}",
                        flush=True,
                    )

                    s = copy.deepcopy(scenario)
                    s.orchestrator_config.seed = seed
                    mem = s.orchestrator_config.memory_tier_config
                    mem.seed = seed
                    mem.side_detection_rate = detection
                    mem.side_write_preference = pref
                    mem.side_teardown_resets_discovery = mode == "revocation"

                    orch = build_orchestrator(s)
                    orch.run()

                    handler = orch._memory_handler
                    snaps = handler.epoch_snapshots
                    result = _extract_results(
                        orch,
                        {
                            "mode": mode,
                            "detection": detection,
                            "write_pref": pref,
                        },
                        seed_offset,
                        seed,
                    )

                    ungov = [sn.get("ungoverned_fraction", 0.0) for sn in snaps]
                    discovered = [sn.get("discovered_fraction", 0.0) for sn in snaps]
                    side_corr = [sn.get("side_cache_corruption", 0.0) for sn in snaps]
                    infections = [sn["mean_infection"] for sn in snaps]
                    late = ungov[-10:]

                    row = result.to_dict()
                    row.update({
                        "mode": mode,
                        "detection": detection,
                        "write_pref": pref,
                        "peak_ungoverned": max(ungov),
                        "late_ungoverned": sum(late) / len(late),
                        "final_discovered": discovered[-1],
                        "mean_side_corruption": sum(side_corr) / len(side_corr),
                        "final_infection": infections[-1],
                        "side_writes": handler.side_write_count,
                        "side_poisoned_writes": handler.side_poisoned_write_count,
                        "side_teardowns": handler.side_teardown_count,
                    })
                    rows.append(row)

                    for sn in snaps:
                        series.append({
                            "mode": mode,
                            "detection": detection,
                            "write_pref": pref,
                            "seed": seed,
                            **sn,
                        })

    # ── Export ──────────────────────────────────────────────────
    with open(out_dir / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # epoch_series rows have heterogeneous keys (snapshot dict varies); union them.
    series_fields: list = []
    for sn in series:
        for k in sn:
            if k not in series_fields:
                series_fields.append(k)
    with open(out_dir / "epoch_series.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=series_fields)
        writer.writeheader()
        writer.writerows(series)

    with open(out_dir / "run.yaml", "w") as f:
        f.write(
            "scenario: scenarios/side_channel.yaml\n"
            "sweep: mode x detection x write_pref\n"
            f"modes: {modes}\n"
            f"detections: {detections}\n"
            f"write_prefs: {prefs}\n"
            f"seeds: {n_seeds} (base {SEED_BASE})\n"
            f"timestamp: {timestamp}\n"
        )

    # ── Summary ─────────────────────────────────────────────────
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["mode"], r["detection"], r["write_pref"])].append(r)

    header = (
        f"{'mode':>11} {'det':>5} {'pref':>5}  {'late_ungov':>10}  "
        f"{'discovered':>10}  {'teardowns':>9}  {'side_pois':>9}  "
        f"{'toxicity':>8}  {'welfare':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    summary = {}
    for mode in modes:
        for detection in detections:
            for pref in prefs:
                runs = grouped[(mode, detection, pref)]
                if not runs:
                    continue
                n = len(runs)
                agg = {
                    "late_ungoverned": sum(r["late_ungoverned"] for r in runs) / n,
                    "final_discovered": sum(r["final_discovered"] for r in runs) / n,
                    "mean_side_corruption": sum(
                        r["mean_side_corruption"] for r in runs
                    ) / n,
                    "side_teardowns": sum(r["side_teardowns"] for r in runs) / n,
                    "side_poisoned_writes": sum(
                        r["side_poisoned_writes"] for r in runs
                    ) / n,
                    "final_infection": sum(r["final_infection"] for r in runs) / n,
                    "avg_toxicity": sum(r["avg_toxicity"] for r in runs) / n,
                    "total_welfare": sum(r["total_welfare"] for r in runs) / n,
                }
                summary[f"{mode}_det{detection}_pref{pref}"] = agg
                print(
                    f"{mode:>11} {detection:>5} {pref:>5}  "
                    f"{agg['late_ungoverned']:10.3f}  "
                    f"{agg['final_discovered']:10.3f}  "
                    f"{agg['side_teardowns']:9.1f}  "
                    f"{agg['side_poisoned_writes']:9.1f}  "
                    f"{agg['avg_toxicity']:8.3f}  "
                    f"{agg['total_welfare']:8.1f}"
                )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
