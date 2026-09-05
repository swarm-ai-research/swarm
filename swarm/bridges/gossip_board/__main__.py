"""CLI: run or sweep the gossip-board scenario, writing a self-contained run folder.

    python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml
    python -m swarm.bridges.gossip_board scenarios/gossip_board_fidelity.yaml \
        --axis fidelity=code,description,score_only --seeds 10
"""

from __future__ import annotations

import argparse
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

    if doc_axes:
        rows = sweep_board(cfg, doc_axes, seeds)
        keys = list(doc_axes)
        agg = aggregate(rows, keys)
        for name, data in (("sweep.csv", rows), ("sweep_mean.csv", agg)):
            with (out / "csv" / name).open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
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
