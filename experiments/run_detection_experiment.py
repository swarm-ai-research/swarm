#!/usr/bin/env python
"""Run the matched soft-vs-binary detection experiment and emit headline artifacts.

Turns the self-optimizing-agent vignette into a real experiment: many seeds,
varied degradation trajectories and onset times, with detection curves rather
than a narrative. Writes a self-contained run folder under ``runs/``:

    runs/<ts>_detection_baselines/
      csv/{detection,ttd,market,calibration}.csv   # raw per-(base_rate,seed) rows
      csv/*_agg.csv                                 # mean/std aggregates
      plots/{roc_toxicity,auroc_vs_baserate,ttd,market_selection,calibration}.png
      # Full per-detector tables now include: AUROC, AUPRC, pAUROC@FPR≤0.05/0.01 (see sections 1, 1a, 1b)
      summary.md                                    # headline tables
      config.json                                   # exact config for reproducibility

Usage:
    python experiments/run_detection_experiment.py            # full: 5 base rates x 10 seeds
    python experiments/run_detection_experiment.py --smoke    # quick: 3 base rates x 3 seeds
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Pin `swarm` to THIS checkout (bead hjyp). `python experiments/x.py` puts
# experiments/ on sys.path[0], not the repo root, so `import swarm` otherwise
# resolves through the editable install, which points at whichever checkout ran
# `pip install -e .` last -- possibly another session's worktree.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)

from swarm.detection import (  # noqa: E402
    ExperimentConfig,
    PopulationConfig,
    StreamConfig,
    aggregate,
    compute_paired_stats,
    run_experiment,
)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot_roc(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for key, curve in sorted(curves.items()):
        ax.plot(curve.fpr, curve.tpr, lw=2, label=f"{key} (AUROC={curve.auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC: per-agent toxicity detector (base rate = 0.20)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_auroc_vs_baserate(agg: List[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in ("soft", "binary"):
        rows = sorted(
            [r for r in agg if r["variant"] == variant], key=lambda r: r["base_rate"]
        )
        br = [r["base_rate"] for r in rows]
        au = [r["auroc_mean"] for r in rows]
        sd = [r["auroc_std"] for r in rows]
        ax.errorbar(br, au, yerr=sd, marker="o", capsize=3, lw=2, label=variant)
    ax.set_xlabel("adversarial base rate")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-agent toxicity detector: AUROC vs base rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_ttd(agg: List[dict], path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    variants = ("soft", "binary")
    med = [
        next((r["median_ttd_mean"] for r in agg if r["variant"] == v), None)
        for v in variants
    ]
    med = [0.0 if m is None else m for m in med]
    det = [
        next((r["detection_rate_mean"] for r in agg if r["variant"] == v), 0.0)
        for v in variants
    ]
    colors = ["seagreen", "indianred"]
    ax1.bar(variants, med, color=colors)
    ax1.set_ylabel("median epochs from onset")
    ax1.set_title("Time-to-detection (lower = faster)\nat FPR ≤ 0.05")
    for i, m in enumerate(med):
        ax1.text(i, m, f"{m:.2f}", ha="center", va="bottom")
    ax2.bar(variants, det, color=colors)
    ax2.set_ylabel("detection rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Fraction of degrading agents ever flagged")
    for i, d in enumerate(det):
        ax2.text(i, d, f"{d:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_market(agg: List[dict], path: Path) -> None:
    metrics = sorted({r["metric"] for r in agg})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        for variant in ("soft", "binary"):
            rows = sorted(
                [r for r in agg if r["metric"] == metric and r["variant"] == variant],
                key=lambda r: r["base_rate"],
            )
            br = [r["base_rate"] for r in rows]
            val = [r["value_mean"] for r in rows]
            sd = [r["value_std"] for r in rows]
            ax.errorbar(br, val, yerr=sd, marker="o", capsize=3, lw=2, label=variant)
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("adversarial base rate")
        ax.set_ylabel("selection risk score (−gap)")
        ax.set_title(f"Market adverse selection: {metric}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_calibration(agg: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["soft", "binary"]
    brier = [agg["soft_brier_mean"], agg["binary_brier_mean"]]
    ece = [agg["soft_ece_mean"], agg["binary_ece_mean"]]
    colors = ["steelblue", "darkorange"]
    ax1.bar(labels, brier, color=colors)
    ax1.set_title("Brier score (lower = better)")
    for i, b in enumerate(brier):
        ax1.text(i, b, f"{b:.4f}", ha="center", va="bottom")
    ax2.bar(labels, ece, color=colors)
    ax2.set_title("Expected calibration error (lower = better)")
    for i, e in enumerate(ece):
        ax2.text(i, e, f"{e:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _write_summary(out: Path, cfg: ExperimentConfig, res, aggs: dict) -> None:
    lines: List[str] = ["# Detection baselines: soft vs binary", ""]
    lines.append(
        f"_{len(cfg.seeds)} seeds × {len(cfg.base_rates)} base rates, "
        f"{cfg.population.n_agents} agents/population, "
        f"{cfg.population.stream.n_epochs} epochs._"
    )
    lines.append("")
    lines.append(
        "Each soft metric is paired with its **binary analogue** — the *same* "
        "metric computed on the proxy thresholded at τ\\*=%.2f. Toxicity and "
        "uncertain_fraction are per-agent detectors; quality-gap, conditional-loss, "
        "and spread are market-level adverse-selection metrics (they need a quality mixture). "
        "Full per-metric rows are in the CSVs; headline tables below focus on the primary signals."
         % cfg.tau_star
    )
    lines.append("")

    lines.append("## 1. Per-agent toxicity detector — AUROC by base rate (see CSVs for uncertain_fraction)")
    lines.append("")
    lines.append("| base rate | soft AUROC | binary AUROC |")
    lines.append("| --- | ---: | ---: |")
    det = aggs["detection"]
    tox = [r for r in det if r["metric"] == "toxicity"]
    for br in sorted({r["base_rate"] for r in tox}):
        s = next(r for r in tox if r["base_rate"] == br and r["variant"] == "soft")
        b = next(r for r in tox if r["base_rate"] == br and r["variant"] == "binary")
        lines.append(
            f"| {br:.2f} | {_fmt(s['auroc_mean'])} ± {_fmt(s['auroc_std'])} "
            f"| {_fmt(b['auroc_mean'])} ± {_fmt(b['auroc_std'])} |"
        )
    lines.append("")

    # AUPRC section - now properly surfaced (Area Under the Precision-Recall Curve)
    lines.append("## 1a. AUPRC (Area Under the Precision-Recall Curve)")
    lines.append("")
    lines.append("| base rate | metric | soft AUPRC | binary AUPRC |")
    lines.append("| --- | --- | ---: | ---: |")
    for br in sorted({r["base_rate"] for r in det}):
        for metric in sorted({r["metric"] for r in det if r["base_rate"] == br}):
            s = next((r for r in det if r["base_rate"] == br and r["metric"] == metric and r["variant"] == "soft"), None)
            b = next((r for r in det if r["base_rate"] == br and r["metric"] == metric and r["variant"] == "binary"), None)
            s_val = _fmt(s.get("auprc_mean")) if s and s.get("auprc_mean") is not None else "n/a"
            b_val = _fmt(b.get("auprc_mean")) if b and b.get("auprc_mean") is not None else "n/a"
            lines.append(f"| {br:.2f} | {metric} | {s_val} | {b_val} |")
    lines.append("")

    # pAUROC section (new high-signal safety metric)
    lines.append(
        "## 1b. Partial AUROC at low FPR (FPR≤0.05 and ≤0.01 safety operating points)"
    )
    lines.append("")
    lines.append(
        "| base rate | metric | soft @FPR≤0.05 | binary @FPR≤0.05 "
        "| soft @FPR≤0.01 | binary @FPR≤0.01 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")

    def _cell(row, key):
        return _fmt(row.get(key)) if row and row.get(key) is not None else "n/a"

    for br in sorted({r["base_rate"] for r in det}):
        for metric in sorted({r["metric"] for r in det if r["base_rate"] == br}):
            s = next((r for r in det if r["base_rate"] == br and r["metric"] == metric and r["variant"] == "soft"), None)
            b = next((r for r in det if r["base_rate"] == br and r["metric"] == metric and r["variant"] == "binary"), None)
            lines.append(
                f"| {br:.2f} | {metric} "
                f"| {_cell(s, 'pauroc_fpr05_mean')} | {_cell(b, 'pauroc_fpr05_mean')} "
                f"| {_cell(s, 'pauroc_fpr01_mean')} | {_cell(b, 'pauroc_fpr01_mean')} |"
            )
    lines.append("")

    lines.append(
        "## 2. Time-to-detection (toxicity detector) at FPR ≤ %.2f "
        "(see CSVs for uncertain_fraction)" % cfg.max_fpr
    )
    lines.append("")
    lines.append("| variant | median epochs from onset | detection rate |")
    lines.append("| --- | ---: | ---: |")
    ttd_tox = [r for r in aggs["ttd"] if r["metric"] == "toxicity"]
    for r in sorted(ttd_tox, key=lambda r: r["variant"]):
        lines.append(
            f"| {r['variant']} | {_fmt(r['median_ttd_mean'], 2)} "
            f"| {_fmt(r['detection_rate_mean'], 2)} |"
        )
    lines.append("")

    lines.append("## 3. Market adverse selection (risk = −gap or spread; higher = more adverse)")
    lines.append("")
    lines.append("| metric | variant | " + " | ".join(
        f"br={br:.2f}" for br in sorted(cfg.base_rates)) + " |")
    lines.append("| --- | --- | " + " | ".join("---:" for _ in cfg.base_rates) + " |")
    mkt = aggs["market"]
    for metric in sorted({r["metric"] for r in mkt}):
        for variant in ("soft", "binary"):
            cells = []
            for br in sorted(cfg.base_rates):
                row = next(
                    (r for r in mkt if r["metric"] == metric
                     and r["variant"] == variant and r["base_rate"] == br),
                    None,
                )
                cells.append(_fmt(row["value_mean"]) if row else "n/a")
            lines.append(f"| {metric} | {variant} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 4. Proxy calibration (soft probability vs thresholded)")
    lines.append("")
    c = aggs["calibration"]
    lines.append("| | Brier | ECE |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| soft | {_fmt(c['soft_brier_mean'], 4)} | {_fmt(c['soft_ece_mean'], 4)} |")
    lines.append(f"| binary | {_fmt(c['binary_brier_mean'], 4)} | {_fmt(c['binary_ece_mean'], 4)} |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def _np_safe(o):
    """JSON default handler for numpy scalar types (see /full_study Phase 2 note)."""
    import numpy as _np

    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _run_sensitivity(args) -> None:
    """Quick sensitivity sweep over one generative knob (e.g. proxy_noise).

    Produces a compact matrix of headline metrics (AUROC, AUPRC, pAUROC@FPR≤0.05)
    for the main per-agent detectors across (parameter_value × base_rate).
    Designed for fast exploration while still using the real experiment engine.
    """
    param = args.sensitivity
    raw_values = [float(v.strip()) for v in args.values.split(",")]

    # Reduced but still meaningful config for sensitivity
    base_rates = (0.05, 0.2, 0.5)
    n_seeds = 4
    n_agents = 25
    n_epochs = 16

    all_rows = []
    print(f"Sensitivity sweep: {param} over {raw_values}  |  base_rates={base_rates}  |  seeds={n_seeds}")

    for val in raw_values:
        stream_cfg = StreamConfig(
            n_epochs=n_epochs,
            interactions_per_epoch=8,
            proxy_noise=val if param == "proxy_noise" else 0.09,
            quality_jitter=val if param == "quality_jitter" else 0.02,
            benchmark_noise=val if param == "benchmark_noise" else 0.10,
        )
        cfg = ExperimentConfig(
            base_rates=base_rates,
            seeds=tuple(range(n_seeds)),
            population=PopulationConfig(n_agents=n_agents, stream=stream_cfg),
        )

        res = run_experiment(cfg)

        # Aggregate per (base_rate, metric, variant)
        det_agg = aggregate(
            res.detection_rows,
            ["base_rate", "metric", "variant"],
            ["auroc", "auprc", "pauroc_fpr05"],
        )

        for row in det_agg:
            row = row.copy()
            row[param] = val
            all_rows.append(row)

    # Write results
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path("runs") / f"{ts}_sensitivity_{param}"
    (out / "csv").mkdir(parents=True, exist_ok=True)

    _write_csv(out / "csv" / "sensitivity_raw.csv", all_rows)

    # Build a nice headline matrix focused on the primary per-agent metric (toxicity)
    # and also show uncertain_fraction for comparison
    lines = [f"# Sensitivity: {param}\n"]
    lines.append(f"Swept values: {raw_values}")
    lines.append(f"Reduced config: {n_seeds} seeds × {n_agents} agents × {n_epochs} epochs\n")

    for metric in ["toxicity", "uncertain_fraction"]:
        lines.append(f"## {metric} — soft vs binary\n")

        for mname, col in [
            ("AUROC", "auroc_mean"),
            ("AUPRC", "auprc_mean"),
            ("pAUROC@FPR≤0.05", "pauroc_fpr05_mean"),
        ]:
            lines.append(f"### {mname}\n")
            lines.append("| value | " + " | ".join(f"br={br:.2f}" for br in base_rates) + " |")
            lines.append("| --- | " + " | ".join("---:" for _ in base_rates) + " |")

            for val in raw_values:
                cells = []
                for br in base_rates:
                    soft = next(
                        (r for r in all_rows
                         if r[param] == val and r["base_rate"] == br and r["metric"] == metric and r["variant"] == "soft"),
                        None
                    )
                    binary = next(
                        (r for r in all_rows
                         if r[param] == val and r["base_rate"] == br and r["metric"] == metric and r["variant"] == "binary"),
                        None
                    )
                    s = _fmt(soft[col]) if soft else "n/a"
                    b = _fmt(binary[col]) if binary else "n/a"
                    cells.append(f"{s} / {b}")
                lines.append(f"| {val} | " + " | ".join(cells) + " |")
            lines.append("")

    (out / "sensitivity_summary.md").write_text("\n".join(lines))
    print(f"Wrote sensitivity matrix to {out}/sensitivity_summary.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="quick small-scale run")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--agents", type=int, default=40)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output run directory. Default: runs/<ts>_detection_baselines. "
        "Used by /full_study --detection to place artifacts in the study folder.",
    )
    ap.add_argument(
        "--sensitivity",
        type=str,
        default=None,
        help="Parameter from StreamConfig to sweep (e.g. proxy_noise, quality_jitter, benchmark_noise)",
    )
    ap.add_argument(
        "--values",
        type=str,
        default=None,
        help="Comma-separated values for the sensitivity sweep, e.g. 0.03,0.06,0.09,0.15",
    )
    args = ap.parse_args()

    # --- Sensitivity mode (quick matrix on generative knobs) ---
    if args.sensitivity and args.values:
        return _run_sensitivity(args)

    if args.smoke:
        cfg = ExperimentConfig(
            base_rates=(0.05, 0.2, 0.5),
            seeds=tuple(range(3)),
            population=PopulationConfig(
                n_agents=20, stream=StreamConfig(n_epochs=16, interactions_per_epoch=8)
            ),
        )
    else:
        cfg = ExperimentConfig(
            base_rates=(0.05, 0.1, 0.2, 0.35, 0.5),
            seeds=tuple(range(args.seeds)),
            population=PopulationConfig(n_agents=args.agents, stream=StreamConfig()),
        )

    if args.out:
        out = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("runs") / f"{ts}_detection_baselines"
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Running: {len(cfg.seeds)} seeds × {len(cfg.base_rates)} base rates ...")
    res = run_experiment(cfg)

    # Raw rows
    _write_csv(out / "csv" / "detection.csv", res.detection_rows)
    _write_csv(out / "csv" / "ttd.csv", res.ttd_rows)
    _write_csv(out / "csv" / "market.csv", res.market_rows)
    _write_csv(out / "csv" / "calibration.csv", res.calibration_rows)

    # Aggregates (include metric so pAUROC and multi-detector data aggregates correctly)
    det_agg = aggregate(res.detection_rows, ["base_rate", "metric", "variant"], ["auroc", "auprc", "pauroc_fpr05", "pauroc_fpr01"])
    # Group TTD by metric too: ttd_rows now carries multiple per-agent detectors
    # (toxicity, uncertain_fraction); aggregating by variant alone would average
    # their unrelated TTD distributions together.
    ttd_agg = aggregate(res.ttd_rows, ["metric", "variant"], ["median_ttd", "detection_rate"])
    mkt_agg = aggregate(res.market_rows, ["metric", "variant", "base_rate"], ["value"])
    cal_agg = aggregate(
        res.calibration_rows, [],
        ["soft_brier", "binary_brier", "soft_ece", "binary_ece"],
    )[0]
    _write_csv(out / "csv" / "detection_agg.csv", det_agg)
    _write_csv(out / "csv" / "ttd_agg.csv", ttd_agg)
    _write_csv(out / "csv" / "market_agg.csv", mkt_agg)
    _write_csv(out / "csv" / "calibration_agg.csv", [cal_agg])

    # Plots
    if res.representative_curves:
        _plot_roc(res.representative_curves, out / "plots" / "roc_toxicity.png")
    # Headline plots focus on the primary per-agent detector (toxicity); the
    # full per-metric data is in the *_agg.csv files and summary.json.
    _plot_auroc_vs_baserate(
        [r for r in det_agg if r["metric"] == "toxicity"],
        out / "plots" / "auroc_vs_baserate.png",
    )
    _plot_ttd(
        [r for r in ttd_agg if r["metric"] == "toxicity"],
        out / "plots" / "ttd.png",
    )
    _plot_market(mkt_agg, out / "plots" / "market_selection.png")
    _plot_calibration(cal_agg, out / "plots" / "calibration.png")

    # Config + summary
    (out / "config.json").write_text(
        json.dumps(
            {
                "base_rates": list(cfg.base_rates),
                "n_seeds": len(cfg.seeds),
                "tau_star": cfg.tau_star,
                "max_fpr": cfg.max_fpr,
                "ttd_window": cfg.ttd_window,
                "population": {
                    "n_agents": cfg.population.n_agents,
                    "onset_choices": list(cfg.population.onset_choices),
                    "trajectory_choices": list(cfg.population.trajectory_choices),
                    "stream": asdict(cfg.population.stream),
                },
            },
            indent=2,
        )
    )
    _write_summary(
        out, cfg, res,
        {"detection": det_agg, "ttd": ttd_agg, "market": mkt_agg, "calibration": cal_agg},
    )

    # Paired soft-vs-binary significance tests (built-in Phase 2).
    stats = compute_paired_stats(res)
    print(
        f"Paired stats: {stats['n_survive_holm']}/{stats['family_size']} "
        f"comparisons survive Holm-Bonferroni (alpha={stats['alpha']})"
    )

    # Machine-readable summary for downstream /full_study phases (analysis, paper).
    summary = {
        "experiment": "detection_baselines",
        "n_seeds": len(cfg.seeds),
        "base_rates": list(cfg.base_rates),
        "tau_star": cfg.tau_star,
        "max_fpr": cfg.max_fpr,
        "detection_auroc": det_agg,
        "time_to_detection": ttd_agg,
        "market_selection": mkt_agg,
        "calibration": cal_agg,
        "stats": stats,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_np_safe))

    print(f"Wrote {out}")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
