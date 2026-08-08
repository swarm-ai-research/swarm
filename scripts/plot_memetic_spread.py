#!/usr/bin/env python3
"""Plots for the memetic spread countermeasure sweep.

Usage:
    python scripts/plot_memetic_spread.py runs/<ts>_memetic_spread_sweep
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (dataviz skill reference instance), fixed
# order by reset cadence: 0, 2, 5, 10.
CADENCE_COLORS = {
    0: "#2a78d6",   # blue
    2: "#eb6834",   # orange
    5: "#1baf7a",   # aqua
    10: "#eda100",  # yellow
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def load_rows(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style_axes(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=MUTED, alpha=0.15, linewidth=0.8)


def plot_trajectories(series_rows: list, out: Path) -> None:
    rankings = sorted(
        {r.get("ranking", "quality") for r in series_rows},
        key=lambda x: ["quality", "recency", "engagement"].index(x),
    )
    cadences = sorted({int(r["reset_cadence"]) for r in series_rows})

    # (ranking, detection, cadence) -> epoch -> [values across seeds]
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in series_rows:
        key = (r.get("ranking", "quality"), r["detection"], int(r["reset_cadence"]))
        acc[key][int(r["epoch"])].append(float(r["mean_infection"]))

    fig, axes = plt.subplots(
        len(rankings),
        2,
        figsize=(11, 3.4 * len(rankings)),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    for row, ranking in enumerate(rankings):
        for col, detection in enumerate(["off", "on"]):
            ax = axes[row][col]
            for cadence in cadences:
                per_epoch = acc[(ranking, detection, cadence)]
                epochs = sorted(per_epoch)
                means = [sum(per_epoch[e]) / len(per_epoch[e]) for e in epochs]
                label = "no resets" if cadence == 0 else f"every {cadence}"
                ax.plot(
                    epochs,
                    means,
                    color=CADENCE_COLORS[cadence],
                    linewidth=2,
                    label=label,
                )
            title = f"{ranking} cache · detection {detection}"
            style_axes(ax, title)
            if row == len(rankings) - 1:
                ax.set_xlabel("epoch", color=MUTED, fontsize=9)
        axes[row][0].set_ylabel(
            "mean infection (seed avg)", color=MUTED, fontsize=9
        )
    axes[0][0].legend(
        title="reset cadence", frameon=False, fontsize=9, title_fontsize=9
    )
    fig.suptitle(
        "Memetic infection: cache ranking x reset cadence x detection",
        color=TEXT,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_outcomes(rows: list, out: Path) -> None:
    rankings = sorted(
        {r.get("ranking", "quality") for r in rows},
        key=lambda x: ["quality", "recency", "engagement"].index(x),
    )
    cadences = sorted({int(r["reset_cadence"]) for r in rows})
    grouped: dict = defaultdict(list)
    for r in rows:
        key = (r.get("ranking", "quality"), r["detection"], int(r["reset_cadence"]))
        grouped[key].append(r)

    metrics = [
        (
            "late_infection"
            if "late_infection" in rows[0]
            else "mean_infection",
            "late-run infection"
            if "late_infection" in rows[0]
            else "mean infection",
        ),
        ("mean_tier3_poisoning", "mean tier-3 poisoning"),
        ("total_welfare", "total welfare"),
    ]

    if len(rankings) > 1:
        # x = ranking; series = detection x cadence
        x = range(len(rankings))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        fig.patch.set_facecolor(SURFACE)
        for ax, (field, title) in zip(axes, metrics, strict=False):
            for detection in ["off", "on"]:
                for cadence in cadences:
                    ys = []
                    for rk in rankings:
                        vals = [
                            float(r[field])
                            for r in grouped[(rk, detection, cadence)]
                        ]
                        ys.append(sum(vals) / len(vals))
                    color = "#4a3aa7" if detection == "on" else "#e34948"
                    style = "-" if cadence == 0 else "--"
                    label = (
                        f"det {detection}, "
                        + ("no resets" if cadence == 0 else f"reset/{cadence}")
                    )
                    ax.plot(
                        x,
                        ys,
                        style,
                        marker="o",
                        markersize=5,
                        linewidth=2,
                        color=color,
                        label=label,
                    )
            ax.set_xticks(list(x), rankings)
            ax.set_xlabel("cache ranking", color=MUTED, fontsize=9)
            style_axes(ax, title)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            "Countermeasure outcomes by cache ranking (seed means)",
            color=TEXT,
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    fig.patch.set_facecolor(SURFACE)
    x = range(len(cadences))
    ranking = rankings[0]
    for ax, (field, title) in zip(axes, metrics, strict=False):
        for detection, marker in [("off", "o"), ("on", "s")]:
            ys, lo, hi = [], [], []
            for c in cadences:
                vals = sorted(
                    float(r[field]) for r in grouped[(ranking, detection, c)]
                )
                n = len(vals)
                ys.append(sum(vals) / n)
                lo.append(vals[max(0, int(0.1 * n))])
                hi.append(vals[min(n - 1, int(0.9 * n))])
            color = "#4a3aa7" if detection == "on" else "#e34948"
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(
                x,
                ys,
                marker=marker,
                markersize=6,
                linewidth=2,
                color=color,
                label=f"detection {detection}",
            )
        ax.set_xticks(
            list(x), ["never" if c == 0 else str(c) for c in cadences]
        )
        ax.set_xlabel("reset cadence (epochs)", color=MUTED, fontsize=9)
        style_axes(ax, title)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Countermeasure outcomes (band = 10th-90th pct across seeds)",
        color=TEXT,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)

    series_rows = load_rows(run_dir / "epoch_series.csv")
    rows = load_rows(run_dir / "sweep.csv")

    plot_trajectories(series_rows, plots / "infection_trajectories.png")
    plot_outcomes(rows, plots / "outcomes_by_condition.png")
    print(f"Plots written to {plots}")


if __name__ == "__main__":
    main()
