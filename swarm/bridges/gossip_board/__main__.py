"""CLI: run or sweep the gossip-board scenario, writing a self-contained run folder.

    python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml
    python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml \
        --axis fidelity=code,description,score_only --seeds 10
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from swarm.bridges.gossip_board.model import (
    BoardConfig,
    _parse_value,
    aggregate,
    run_board,
    sweep_board,
)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--axis", action="append", default=[],
                    help="field=v1,v2,... (repeatable); values parse as JSON, else string")
    ap.add_argument("--seeds", type=int, default=1, help="seeds per cell (seed, seed+1, ...)")
    ap.add_argument("--out", type=Path, default=None, help="run folder (default runs/<ts>_...)")
    ap.add_argument("--detect", action="store_true",
                    help="run the collusion detectors (bead 9err) on each cell instead of the metric sweep")
    args = ap.parse_args(argv)

    cfg = BoardConfig.from_yaml(args.scenario)
    doc_axes: Dict[str, List[Any]] = {}
    import yaml  # local: keep model import light

    doc = yaml.safe_load(args.scenario.read_text()) or {}
    for k, vals in (doc.get("sweep") or {}).items():
        doc_axes[k] = list(vals)
    for spec in args.axis:
        k, _, raw = spec.partition("=")
        doc_axes[k] = [_parse_value(x) for x in raw.split(",")]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path("runs") / f"{ts}_{cfg.scenario_id}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "csv").mkdir(exist_ok=True)
    seeds = [cfg.seed + i for i in range(args.seeds)]

    baseline = run_board(cfg)
    (out / "history.json").write_text(json.dumps(baseline.to_json(), indent=1))
    with (out / "csv" / "rounds.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(vars(baseline.rounds[0])))
        w.writeheader()
        for r in baseline.rounds:
            w.writerow(vars(r))
    print(f"baseline ({cfg.fidelity}):")
    for k, v in baseline.summary.items():
        print(f"  {k:32s} {v}")

    if args.detect:
        from swarm.bridges.gossip_board.detect import detect_on_run, mean_rows

        axis_key = list(doc_axes)[0] if doc_axes else "fidelity"
        values = doc_axes.get(axis_key, [cfg.fidelity])
        drows: List[Dict[str, Any]] = []
        for v in values:
            for s in seeds:
                c = copy.deepcopy(cfg)
                setattr(c, axis_key, v)
                c.seed = s
                row = detect_on_run(run_board(c, seed=s), seed=s)
                row[axis_key] = v
                drows.append(row)
        agg = mean_rows(drows, axis_key)
        for name, data in (("detect.csv", drows), ("detect_mean.csv", agg)):
            dfields: List[str] = []
            for row in data:
                dfields += [k for k in row if k not in dfields]
            with (out / "csv" / name).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=dfields)
                w.writeheader()
                w.writerows(data)
        show = ["n_interactions", "structural_flag", "structural_flagged_agents",
                "temporal_alarm_rate", "pairwise_flagged", "explained_fraction",
                "residual_n_interactions", "residual_structural_flag",
                "residual_structural_flagged_agents", "residual_pairwise_flagged"]
        print(f"\ndetectors over {axis_key} x {len(seeds)} seeds (means; every flag is a false positive):")
        for a in agg:
            print(f"  {axis_key}={a[axis_key]}")
            for s_ in show:
                v = a.get(s_)
                print(f"    {s_:36s} {'-' if v is None else f'{v:.3f}'}")
        print(f"\nwrote {out}")
        return 0

    if doc_axes:
        rows = sweep_board(cfg, doc_axes, seeds)
        keys = list(doc_axes)
        agg = aggregate(rows, keys)
        for name, data in (("sweep.csv", rows), ("sweep_mean.csv", agg)):
            fields: List[str] = []
            for row in data:
                fields += [k for k in row if k not in fields]
            with (out / "csv" / name).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(data)
        show = ["final_modal_cluster_fraction", "final_identical_pairs",
                "final_mean_true_score", "final_frontier_fraction", "survivorship_gap",
                "time_to_frontier_late", "adoptions", "rediscoveries", "hidden_dim_adoption"]
        print(f"\nsweep over {keys} x {len(seeds)} seeds (means):")
        print("  " + " | ".join(keys + [s[:18] for s in show]))
        for a in agg:
            cells = [str(a[k]) for k in keys] + [
                "-" if a.get(s) is None else f"{a[s]:.3f}" for s in show]
            print("  " + " | ".join(cells))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
