#!/usr/bin/env python
"""
2D sensitivity grid for the detection experiment.

Sweeps two generative parameters and produces, for both the toxicity and
uncertain_fraction detectors:
  - CSV tables of AUROC, AUPRC, pAUROC@FPR≤0.05, and TTD per grid cell, and
  - soft-vs-binary difference heatmaps for AUROC and median TTD.

Usage:
    PYTHONPATH=. python experiments/run_detection_sensitivity_2d.py

    # Informative regime — avoids the saturated AUROC=1.0 ceiling:
    PYTHONPATH=. python experiments/run_detection_sensitivity_2d.py \
        --preset heterogeneous          # implies --eval-epochs 3
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Pin `swarm` to THIS checkout (bead hjyp). `python experiments/x.py` puts
# experiments/ on sys.path[0], not the repo root, so `import swarm` otherwise
# resolves through the editable install, which points at whichever checkout ran
# `pip install -e .` last -- possibly another session's worktree.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)

from swarm.analysis.heatmaps import plot_difference_heatmap
from swarm.detection import (
    ExperimentConfig,
    PopulationConfig,
    StreamConfig,
    aggregate,
    run_experiment,
)


def parse_values(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",")]


# Detectors we slice the grid by when drawing heatmaps (soft + binary variants
# of each are compared within plot_difference_heatmap).
_METRICS = ("toxicity", "uncertain_fraction")

# (value field, source-row predicate key, higher_is_better, axis label) for each
# heatmap family. AUROC: higher = better discrimination. median_ttd: lower =
# faster detection (hence higher_is_better=False, which flips the takeaway in
# the difference panel's title).
_HEATMAP_FIELDS = (
    ("auroc_mean", "auroc_mean", True, "AUROC"),
    ("median_ttd_mean", "median_ttd_mean", False, "median TTD (epochs)"),
)


def _matrix(
    index: dict,
    *,
    metric: str,
    variant: str,
    base_rate: float,
    field: str,
    vals1: List[float],
    vals2: List[float],
) -> np.ndarray:
    """Build a ``len(vals2) x len(vals1)`` matrix of ``field`` for one slice.

    Rows are ``param2`` (y), columns are ``param1`` (x). Missing or null cells
    become NaN so the heatmap renderer skips them gracefully.
    """
    mat = np.full((len(vals2), len(vals1)), np.nan, dtype=float)
    for i, v2 in enumerate(vals2):
        for j, v1 in enumerate(vals1):
            row = index.get((metric, variant, base_rate, v1, v2))
            if row is not None and row.get(field) is not None:
                mat[i, j] = float(row[field])
    return mat


def make_heatmaps(
    *,
    det_rows: List[dict],
    ttd_rows: List[dict],
    param1: str,
    param2: str,
    vals1: List[float],
    vals2: List[float],
    base_rates: tuple,
    out_dir: Path,
) -> List[Path]:
    """Render soft-vs-binary heatmaps over the param1 x param2 grid.

    For each detector metric and base rate, draws an AUROC difference figure
    (soft | binary | soft-binary) and a median-TTD difference figure. Returns
    the list of written PNG paths.
    """
    # Index rows by (metric, variant, base_rate, v1, v2) for O(1) cell lookup.
    index: dict = {}
    for r in det_rows + ttd_rows:
        key = (r["metric"], r["variant"], r["base_rate"], r[param1], r[param2])
        index.setdefault(key, {}).update(r)

    row_labels = [f"{param2}={v:g}" for v in vals2]
    col_labels = [f"{param1}={v:g}" for v in vals1]

    written: List[Path] = []
    for field, _src, higher_is_better, axis_label in _HEATMAP_FIELDS:
        for metric in _METRICS:
            for base_rate in base_rates:
                soft = _matrix(
                    index, metric=metric, variant="soft", base_rate=base_rate,
                    field=field, vals1=vals1, vals2=vals2,
                )
                binary = _matrix(
                    index, metric=metric, variant="binary", base_rate=base_rate,
                    field=field, vals1=vals1, vals2=vals2,
                )
                if np.all(np.isnan(soft)) and np.all(np.isnan(binary)):
                    continue  # field not produced for this slice (e.g. no TTD)

                better = "higher = better" if higher_is_better else "lower = better"
                title = (
                    f"{axis_label} — {metric} @ base_rate={base_rate:g}  ({better})\n"
                    f"{param1} × {param2}"
                )
                fig, _ = plot_difference_heatmap(
                    soft, binary,
                    row_labels=row_labels, col_labels=col_labels,
                    label_a="soft", label_b="binary",
                    title=title, mode="dark",
                )
                fname = f"{field}_{metric}_br{base_rate:g}.png".replace(".", "p", 1)
                path = out_dir / fname
                fig.savefig(path, dpi=130, bbox_inches="tight")
                plt.close(fig)
                written.append(path)

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param1", default="proxy_noise")
    ap.add_argument("--values1", default="0.03,0.06,0.09,0.15")
    ap.add_argument("--param2", default="quality_jitter")
    ap.add_argument("--values2", default="0.0,0.02,0.05,0.10")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--agents", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument(
        "--preset",
        choices=["none", "heterogeneous"],
        default="none",
        help=(
            "Base StreamConfig preset. 'heterogeneous' uses "
            "StreamConfig.heterogeneous() (per-agent benign + floor spread) so "
            "the grid shows real gradients instead of saturating at AUROC 1.0."
        ),
    )
    ap.add_argument(
        "--eval-epochs",
        type=int,
        default=None,
        help=(
            "Score on the last N epochs only — a sharper signal that pairs with "
            "--preset heterogeneous. Default: back half (or 3 when the "
            "heterogeneous preset is selected)."
        ),
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vals1 = parse_values(args.values1)
    vals2 = parse_values(args.values2)

    base_rates = (0.05, 0.2, 0.5)
    n_seeds = args.seeds
    n_agents = args.agents
    n_epochs = args.epochs

    # Base stream factory honoring the preset.
    def make_base_stream() -> StreamConfig:
        if args.preset == "heterogeneous":
            return StreamConfig.heterogeneous(
                n_epochs=n_epochs, interactions_per_epoch=8
            )
        return StreamConfig(n_epochs=n_epochs, interactions_per_epoch=8)

    # Short trailing eval window (default 3 epochs under the heterogeneous preset,
    # otherwise the experiment default back-half). Reject windows that would be
    # empty or larger than the run — those produce degenerate metrics that still
    # look valid in the output.
    if args.eval_epochs is not None and not (1 <= args.eval_epochs <= n_epochs):
        ap.error(
            f"--eval-epochs must be in [1, {n_epochs}] (got {args.eval_epochs})"
        )
    eval_epochs = args.eval_epochs
    if eval_epochs is None and args.preset == "heterogeneous":
        eval_epochs = 3
    eval_start = None if eval_epochs is None else max(0, n_epochs - eval_epochs)

    preset_tag = "" if args.preset == "none" else f"_{args.preset}"
    if args.out:
        out = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("runs") / f"{ts}_2d_{args.param1}_{args.param2}{preset_tag}"

    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    eval_desc = "back half" if eval_start is None else f"last {eval_epochs} epochs"
    print(f"Running 2D grid: {args.param1} x {args.param2}  (preset={args.preset})")
    print(f"Eval window: {eval_desc}")
    print(f"Output: {out}")

    all_rows = []

    for v1 in vals1:
        for v2 in vals2:
            stream = make_base_stream()
            stream = replace(stream, **{args.param1: v1, args.param2: v2})

            cfg = ExperimentConfig(
                base_rates=base_rates,
                seeds=tuple(range(n_seeds)),
                population=PopulationConfig(n_agents=n_agents, stream=stream),
                eval_start=eval_start,
            )

            print(f"  {args.param1}={v1}, {args.param2}={v2} ...", end=" ", flush=True)
            res = run_experiment(cfg)
            print("done")

            for row in aggregate(res.detection_rows, ["base_rate", "metric", "variant"], ["auroc", "auprc", "pauroc_fpr05"]):
                r = row.copy()
                r[args.param1] = v1
                r[args.param2] = v2
                all_rows.append(r)

            for row in aggregate(res.ttd_rows, ["base_rate", "metric", "variant"], ["median_ttd", "detection_rate"]):
                r = row.copy()
                r[args.param1] = v1
                r[args.param2] = v2
                all_rows.append(r)

    # Write CSVs
    import csv
    det_rows = [r for r in all_rows if "auroc_mean" in r]
    if det_rows:
        with open(out / "csv" / "grid_detection.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
            w.writeheader()
            w.writerows(det_rows)

    ttd_rows = [r for r in all_rows if "median_ttd_mean" in r]
    if ttd_rows:
        with open(out / "csv" / "grid_ttd.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ttd_rows[0].keys()))
            w.writeheader()
            w.writerows(ttd_rows)

    # Heatmaps (soft vs binary, per detector metric and base rate)
    plots = make_heatmaps(
        det_rows=det_rows,
        ttd_rows=ttd_rows,
        param1=args.param1,
        param2=args.param2,
        vals1=vals1,
        vals2=vals2,
        base_rates=base_rates,
        out_dir=out / "plots",
    )
    print(f"  wrote {len(plots)} heatmaps")

    # Summary (text), listing the figures that were actually produced.
    with open(out / "grid_summary.md", "w") as f:
        f.write(f"# 2D Grid: {args.param1} x {args.param2}\n\n")
        f.write(f"- Preset: {args.preset}\n")
        f.write(f"- Grid: {len(vals1)} x {len(vals2)} cells, ")
        f.write(f"{n_seeds} seeds x {n_agents} agents x {n_epochs} epochs\n")
        f.write(f"- Eval window: {eval_desc}\n")
        f.write(f"- Base rates: {', '.join(f'{b:g}' for b in base_rates)}\n\n")
        f.write("## Tables\n")
        f.write("- `csv/grid_detection.csv` — AUROC / AUPRC / pAUROC\n")
        f.write("- `csv/grid_ttd.csv` — median TTD + detection rate\n\n")
        f.write("## Heatmaps\n")
        for p in plots:
            f.write(f"- `plots/{p.name}`\n")

    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
