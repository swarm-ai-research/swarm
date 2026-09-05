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

Wiki-board extension (bead pi02): ``--scenario scenarios/wiki_board.yaml``
plus ``--axis <memory_tiers field>=<v1,v2,...>`` (repeatable) sweeps any
side-channel knob — teardown policy, evasion rate, task overlap, deadline
pressure — on top of the mode x detection x write_pref grid. Pass a single
value to a legacy axis to hold it fixed. Values parse as JSON, else string.

Usage:
    python scripts/sweep_side_channel.py [--seeds N] [--quick]
        [--detections 0.0,0.02,0.05] [--prefs 0.3,0.7]
        [--modes deletion,revocation]
        [--scenario scenarios/side_channel.yaml]
        [--axis side_teardown_policy=complete,ordered,random] [--axis ...]
"""

import argparse
import copy
import csv
import itertools
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


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_axis(spec: str):
    name, _, values = spec.partition("=")
    if not name or not values:
        raise SystemExit(f"--axis expects field=v1,v2,... got {spec!r}")
    return name, [_parse_value(v) for v in values.split(",")]


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
    parser.add_argument(
        "--scenario",
        default=str(SCENARIO_PATH),
        help="scenario YAML (default scenarios/side_channel.yaml)",
    )
    parser.add_argument(
        "--axis",
        action="append",
        default=[],
        metavar="FIELD=V1,V2",
        help="extra memory_tiers field to sweep (repeatable); JSON values",
    )
    args = parser.parse_args()
    n_seeds = 2 if args.quick else args.seeds
    detections = [float(d) for d in args.detections.split(",")]
    prefs = [float(p) for p in args.prefs.split(",")]
    modes = args.modes.split(",")
    extra_axes = [_parse_axis(spec) for spec in args.axis]

    scenario_path = Path(args.scenario)
    scenario = load_scenario(scenario_path)

    # Grid axes: the k5o9 legacy three plus any --axis fields. A legacy axis
    # with a single value is fixed, so a wiki_board grid can isolate one lever.
    axes = [("mode", modes), ("detection", detections), ("write_pref", prefs)]
    axes += extra_axes
    axis_names = [name for name, _ in axes]

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROJECT_ROOT / "runs" / f"{timestamp}_{scenario_path.stem}_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = list(itertools.product(*(values for _, values in axes)))
    total = len(cells) * n_seeds
    rows = []
    series = []

    for cell in cells:
        params = dict(zip(axis_names, cell, strict=True))
        for seed_offset in range(n_seeds):
            seed = SEED_BASE + seed_offset
            run_num = len(rows) + 1
            desc = " ".join(f"{k}={v}" for k, v in params.items())
            print(f"[{run_num}/{total}] {desc} seed={seed}", flush=True)

            s = copy.deepcopy(scenario)
            s.orchestrator_config.seed = seed
            mem = s.orchestrator_config.memory_tier_config
            mem.seed = seed
            mem.side_detection_rate = params["detection"]
            mem.side_write_preference = params["write_pref"]
            mem.side_teardown_resets_discovery = params["mode"] == "revocation"
            for name, _ in extra_axes:
                setattr(mem, name, params[name])

            orch = build_orchestrator(s)
            orch.run()

            handler = orch._memory_handler
            snaps = handler.epoch_snapshots
            result = _extract_results(orch, dict(params), seed_offset, seed)

            ungov = [sn.get("ungoverned_fraction", 0.0) for sn in snaps]
            discovered = [sn.get("discovered_fraction", 0.0) for sn in snaps]
            side_corr = [sn.get("side_cache_corruption", 0.0) for sn in snaps]
            infections = [sn["mean_infection"] for sn in snaps]
            gaps = [sn.get("substrate_gap", 0.0) for sn in snaps]
            entries = [sn.get("side_entries", 0) for sn in snaps]
            late = ungov[-10:]

            row = result.to_dict()
            row.update({
                **params,
                "peak_ungoverned": max(ungov),
                "late_ungoverned": sum(late) / len(late),
                "final_discovered": discovered[-1],
                "mean_side_corruption": sum(side_corr) / len(side_corr),
                "final_infection": infections[-1],
                "side_writes": handler.side_write_count,
                "side_poisoned_writes": handler.side_poisoned_write_count,
                "side_teardowns": handler.side_teardown_count,
                "side_sweeps": handler.side_sweep_count,
                "side_swept_entries": handler.side_swept_entries,
                "late_substrate_gap": sum(gaps[-10:]) / len(gaps[-10:]),
                "final_side_entries": entries[-1],
                "final_evaders": snaps[-1].get("evader_fraction", 0.0),
                "final_side_value": snaps[-1].get("mean_side_value", 0.0),
                "final_host_users": json.dumps(snaps[-1].get("side_host_users", [])),
            })
            rows.append(row)

            for sn in snaps:
                series.append({**params, "seed": seed, **sn})

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
            f"scenario: {scenario_path}\n"
            f"sweep: {' x '.join(axis_names)}\n"
        )
        for name, values in axes:
            f.write(f"{name}: {values}\n")
        f.write(
            f"seeds: {n_seeds} (base {SEED_BASE})\n"
            f"timestamp: {timestamp}\n"
            f"command: {' '.join(sys.argv)}\n"
        )

    # ── Summary ─────────────────────────────────────────────────
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[name] for name in axis_names)].append(r)

    metrics = [
        ("late_ungoverned", "late_ungov", "10.3f"),
        ("final_discovered", "discovered", "10.3f"),
        ("side_teardowns", "teardowns", "9.1f"),
        ("side_sweeps", "sweeps", "6.1f"),
        ("late_substrate_gap", "sub_gap", "7.3f"),
        ("final_evaders", "evaders", "7.2f"),
        ("final_side_entries", "entries", "7.1f"),
        ("side_poisoned_writes", "side_pois", "9.1f"),
        ("avg_toxicity", "toxicity", "8.3f"),
        ("total_welfare", "welfare", "8.1f"),
    ]
    header = "  ".join(f"{name:>12}" for name in axis_names) + "  " + "  ".join(
        f"{label:>{fmt.split('.')[0]}}" for _, label, fmt in metrics
    )
    print(f"\n{header}")
    print("-" * len(header))
    summary = {}
    for cell in cells:
        runs = grouped[tuple(cell)]
        if not runs:
            continue
        n = len(runs)
        agg = {
            key: sum(r[key] for r in runs) / n
            for key, _, _ in metrics
        }
        agg["mean_side_corruption"] = sum(r["mean_side_corruption"] for r in runs) / n
        agg["final_infection"] = sum(r["final_infection"] for r in runs) / n
        agg["final_side_value"] = sum(r["final_side_value"] for r in runs) / n
        key = "_".join(f"{name}{value}" for name, value in zip(axis_names, cell, strict=True))
        summary[key] = agg
        print(
            "  ".join(f"{str(v):>12}" for v in cell)
            + "  "
            + "  ".join(f"{agg[k]:{fmt}}" for k, _, fmt in metrics)
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
