"""Calibration arm A — proxy fidelity experiment runner.

Pre-registration: docs/research/calibration-prereg.md

Sweeps the sigmoid `k` parameter against a controlled latent-p grid and
writes per-bin reliability data + per-k summary (ECE / MCE / Brier) to a
timestamped run directory.

Usage:
    python -m experiments.calibration_fidelity \
        --p-grid 0.05,0.2,0.4,0.6,0.8,0.95 \
        --k-values 0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0 \
        --per-bin 500 \
        --seed 42

Output:
    runs/<ts>_calibration_fidelity_seed<seed>/
        summary.csv         # one row per k
        bins_k<k>.csv       # one bins file per k
        config.json         # config + git rev
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments._calibration_common import git_rev
from swarm.calibration.fidelity import report_from_pairs, sweep_sigmoid_k


def _parse_floats(arg: str) -> list[float]:
    return [float(x) for x in arg.split(",") if x.strip()]


def _write_summary(run_dir: Path, reports) -> None:
    with (run_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigmoid_k", "n_total", "ece", "mce", "brier"])
        for r in reports:
            writer.writerow(
                [r.sigmoid_k, r.n_total, f"{r.ece:.6f}", f"{r.mce:.6f}", f"{r.brier:.6f}"]
            )


def _write_bins(run_dir: Path, report) -> None:
    with (run_dir / f"bins_k{report.sigmoid_k:g}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lo", "hi", "n", "mean_confidence", "accuracy", "gap"])
        for b in report.bins:
            writer.writerow(
                [
                    f"{b.lo:.4f}",
                    f"{b.hi:.4f}",
                    b.n,
                    f"{b.mean_confidence:.6f}",
                    f"{b.accuracy:.6f}",
                    f"{abs(b.accuracy - b.mean_confidence):.6f}",
                ]
            )


def _analyze_scenario_run(export_json: Path, n_bins: int, runs_dir: Path) -> int:
    """Controlled-injection analysis path (beads c89o).

    Reads a ``swarm run --export-json`` file, keeps interactions that carry
    a latent ``ground_truth`` (i.e. calibration-injection interactions),
    and emits the same summary/bins CSV schema as the synthetic sweep so
    downstream aggregation (4e2s) consumes both without glue.
    """
    from swarm.core.proxy import ProxyComputer

    data = json.loads(export_json.read_text())
    events = data.get("events", [])
    pairs = [
        (float(e["p"]), 1 if e["ground_truth"] == 1 else 0)
        for e in events
        if e.get("event_type") == "interaction" and e.get("ground_truth") is not None
    ]
    if not pairs:
        print(
            f"No interactions with ground_truth in {export_json} — was the "
            "run a calibration-injection scenario exported with --export-json?",
            file=sys.stderr,
        )
        return 2

    sigmoid_k = ProxyComputer().sigmoid_k  # the pipeline default the scenario ran with
    p_hats = [p for p, _ in pairs]
    outcomes = [o for _, o in pairs]
    report = report_from_pairs(p_hats, outcomes, sigmoid_k=sigmoid_k, n_bins=n_bins)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / f"{ts}_calibration_fidelity_scenario"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "ts_utc": ts,
                "git_rev": git_rev(),
                "mode": "from_run",
                "source_export": str(export_json),
                "source_seed": data.get("seed"),
                "sigmoid_k": sigmoid_k,
                "n_bins": n_bins,
                "prereg": "docs/research/calibration-prereg.md#post-registration-addenda-append-only-not-part-of-the-registered-design",
            },
            indent=2,
        )
    )
    _write_summary(run_dir, [report])
    _write_bins(run_dir, report)

    print(f"Wrote {run_dir}")
    print(f"n={report.n_total}  k={sigmoid_k}  ECE={report.ece:.4f}  "
          f"MCE={report.mce:.4f}  Brier={report.brier:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p-grid",
        type=_parse_floats,
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
        help="Comma-separated latent p values",
    )
    parser.add_argument(
        "--k-values",
        type=_parse_floats,
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        help="Comma-separated sigmoid k values to sweep",
    )
    parser.add_argument(
        "--per-bin",
        type=int,
        default=500,
        help="Synthetic interactions per p-grid value (>=500 per pre-reg)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=10, help="Reliability-diagram bins")
    parser.add_argument(
        "--from-run",
        type=Path,
        default=None,
        metavar="EXPORT_JSON",
        help=(
            "Analyze a controlled-injection scenario run (beads c89o) instead "
            "of the synthetic generator: read (p, ground_truth) pairs from a "
            "`swarm run --export-json` file and emit the same summary/bins "
            "CSVs. Ignores --p-grid/--k-values/--per-bin."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Parent directory for run output",
    )
    args = parser.parse_args(argv)

    if args.from_run is not None:
        return _analyze_scenario_run(args.from_run, args.n_bins, args.runs_dir)

    if args.per_bin < 500:
        print(
            f"WARNING: per-bin={args.per_bin} is below the pre-registered minimum of 500",
            file=sys.stderr,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.runs_dir / f"{ts}_calibration_fidelity_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "ts_utc": ts,
        "git_rev": git_rev(),
        "p_grid": args.p_grid,
        "k_values": args.k_values,
        "per_bin": args.per_bin,
        "seed": args.seed,
        "n_bins": args.n_bins,
        "prereg": "docs/research/calibration-prereg.md#arm-a-proxy-fidelity",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    reports = sweep_sigmoid_k(
        k_values=args.k_values,
        p_grid=args.p_grid,
        per_bin=args.per_bin,
        seed=args.seed,
        n_bins=args.n_bins,
    )

    _write_summary(run_dir, reports)
    for r in reports:
        _write_bins(run_dir, r)

    best = min(reports, key=lambda r: r.ece)
    print(f"Wrote {run_dir}")
    print("k       ECE       MCE       Brier")
    for r in reports:
        marker = " <- min ECE" if r is best else ""
        print(f"{r.sigmoid_k:<6.2f}  {r.ece:.4f}    {r.mce:.4f}    {r.brier:.4f}{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
