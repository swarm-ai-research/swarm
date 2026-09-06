#!/usr/bin/env python3
"""Figures for the wiki behavior Monte Carlo write-up.

Reads the run folders documented in docs/research/wiki-monte-carlo-results.md
(pilot, detection and moderation confirmation, page-level extension) and
writes PNGs. Every number plotted comes from the run artifacts; nothing is
re-simulated here.

Usage: python scripts/plot_wiki_mc.py [--runs runs] [--out docs/blog/figures/wiki-mc]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3

POLICY_LABEL = {
    "none": "no intervention",
    "ordered": "ordered host deletion",
    "random": "random host deletion",
    "lock": "host lock",
    "global_lock": "global write lock",
}


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)


def load_summary(directory: Path) -> list[dict]:
    return json.loads((directory / "summary.json").read_text())["cells"]


def load_paired(directory: Path) -> list[dict]:
    with (directory / "paired_summary.csv").open() as handle:
        return list(csv.DictReader(handle))


def agreement_scores(directory: Path, cell_id: str) -> list[float]:
    return sorted(
        json.loads(path.read_text())["treatment_metrics"]["screen_agreement"]
        for path in directory.glob(f"{cell_id}-seed-*.json")
    )


def fig_detection(runs: Path, out: Path) -> None:
    d = runs / "wiki_mc_confirm_detection"
    independent = agreement_scores(d, "detection-001")
    authorized = agreement_scores(d, "detection-003")
    pilot = runs / "wiki_mc_pilot_all"
    pilot_independent = agreement_scores(pilot, "detection-019")
    fitted = pilot_independent[int(0.95 * len(pilot_independent))]
    frozen = 0.8
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    bins = [x / 100 for x in range(60, 101, 2)]
    ax.hist(independent, bins=bins, color=BLUE, alpha=0.85, label="independent (no sharing)")
    ax.hist(authorized, bins=bins, color=ORANGE, alpha=0.65, label="authorized sharing")
    for x, name, dy in ((frozen, f"frozen threshold {frozen:.2f}", 0), (fitted, f"pilot-fit at 5% FPR {fitted:.3f}", -6)):
        ax.axvline(x, color=TEXT, linewidth=1, linestyle="--" if x == frozen else ":")
        ax.annotate(name, (x, ax.get_ylim()[1] * 0.97), xytext=(4, dy), textcoords="offset points",
                    fontsize=8, color=TEXT, va="top")
    def rate(values, t):
        return sum(v >= t for v in values) / len(values)
    ax.text(0.605, ax.get_ylim()[1] * 0.62,
            f"alarm rate at {frozen:.2f}\n  independent {rate(independent, frozen):.0%}\n  authorized {rate(authorized, frozen):.0%}\n"
            f"alarm rate at {fitted:.3f}\n  independent {rate(independent, fitted):.1%}\n  authorized {rate(authorized, fitted):.0%}",
            fontsize=8, color=TEXT, family="monospace", va="top")
    style(ax, "Output agreement per run, full coverage, 200 seeds each",
          "share of same-task submission pairs with identical answers", "runs")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out / "detection_agreement.png", facecolor=SURFACE)
    plt.close(fig)


def fig_moderation(runs: Path, out: Path) -> None:
    rows = load_paired(runs / "wiki_mc_confirm_moderation")
    cells = {c["id"]: c["overrides"] for c in json.loads((runs / "wiki_mc_confirm_moderation" / "manifest.json").read_text())["cells"]}
    order = ["ordered", "random", "lock", "global_lock"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, metric, title, xlabel in (
        (axes[0], "completion_rate", "Completion rate, paired difference", "treatment minus untreated, per run"),
        (axes[1], "total_writes", "Total board writes, paired difference", "treatment minus untreated, writes per run"),
    ):
        ys, labels = [], []
        for i, policy in enumerate(order):
            for row in rows:
                ov = cells[row["cell_id"]]
                if row["metric"] == metric and ov["moderation_policy"] == policy and ov["relocation_mode"] == "endogenous":
                    mean, lo, hi = float(row["paired_difference_mean"]), float(row["ci_low"]), float(row["ci_high"])
                    ax.plot([lo, hi], [i, i], color=BLUE, linewidth=2, solid_capstyle="round")
                    ax.plot(mean, i, "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
                    fmt = f"{mean:+.4f}" if metric == "completion_rate" else f"{mean:+.1f}"
                    ax.annotate(fmt, (hi, i), xytext=(6, 0), textcoords="offset points", fontsize=8, color=TEXT, va="center")
            ys.append(i)
            labels.append(POLICY_LABEL[policy])
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        style(ax, title, xlabel)
        ax.margins(x=0.25)
    fig.suptitle("Endogenous relocation, 200 paired seeds, 95% seed-level bootstrap intervals",
                 color=MUTED, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "moderation_paired.png", facecolor=SURFACE)
    plt.close(fig)


def fig_displacement(runs: Path, out: Path) -> None:
    cells = load_summary(runs / "wiki_mc_confirm_moderation")
    order = ["ordered", "random", "lock", "global_lock"]
    picked = {}
    for c in cells:
        ov = json.loads(c["overrides"])
        if ov["relocation_mode"] == "endogenous" and ov["moderation_policy"] in order:
            picked[ov["moderation_policy"]] = c
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    h = 0.26
    for j, (key, label, color) in enumerate((("disrupted_works", "works disrupted", BLUE),
                                              ("relocated_works", "chose to relocate", ORANGE),
                                              ("displaced_works", "traced displacement (wrote elsewhere)", AQUA))):
        for i, policy in enumerate(order):
            v = picked[policy][key] / picked[policy]["n"]
            ax.barh(i + (j - 1) * h, v, height=h - 0.03, color=color, label=label if i == 0 else None)
            ax.annotate(f"{v:.1f}", (v, i + (j - 1) * h), xytext=(4, 0), textcoords="offset points", fontsize=8, color=TEXT, va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([POLICY_LABEL[p] for p in order])
    ax.invert_yaxis()
    style(ax, "From disruption to traced displacement, per run", "events per run")
    fig.suptitle("200 paired seeds, endogenous relocation", color=MUTED, fontsize=9, x=0.01, ha="left")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, labelcolor=TEXT)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out / "displacement_funnel.png", facecolor=SURFACE)
    plt.close(fig)


def fig_emergence(runs: Path, out: Path) -> None:
    cells = [c for c in load_summary(runs / "wiki_mc_pilot_all") if c["family"] == "emergence"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, referrals in zip(axes, (False, True), strict=True):
        for deadline, color in ((6.0, BLUE), (18.0, ORANGE)):
            pts = sorted((json.loads(c["overrides"])["task_overlap"], c["shared_submission_rate_mean"])
                         for c in cells if json.loads(c["overrides"])["referrals_enabled"] == referrals
                         and json.loads(c["overrides"])["deadline"] == deadline)
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"deadline {deadline:.0f}")
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=TEXT, va="center")
        style(ax, f"Referrals {'on' if referrals else 'off'}", "task overlap", "shared-submission rate" if not referrals else "")
        ax.set_xticks([0.2, 0.8])
        ax.set_xlim(0.05, 1.0)
        ax.set_ylim(0, 0.5)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=TEXT)
    fig.suptitle("Emergence pilot: share of assignments answered with a peer's board post (30 seeds per cell)",
                 color=MUTED, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "emergence_sharing.png", facecolor=SURFACE)
    plt.close(fig)


def fig_page_level(runs: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    labels, vals, colors = [], [], []
    for tag, evasion in (("e00", "0.0"), ("e50", "0.5")):
        for c in load_summary(runs / f"wiki_mc_confirm_page_{tag}"):
            ov = json.loads(c["overrides"])
            if ov["relocation_mode"] == "endogenous" and ov["moderation_policy"] in ("ordered", "random"):
                labels.append(f"{ov['moderation_policy']} pages, evasion {evasion}")
                vals.append(c["removed_pages_mean"])
                colors.append(BLUE if ov["moderation_policy"] == "ordered" else ORANGE)
    ys = range(len(labels))
    ax.barh(ys, vals, color=colors, height=0.6)
    for y, v in zip(ys, vals, strict=True):
        ax.annotate(f"{v:.1f}", (v, y), xytext=(4, 0), textcoords="offset points", fontsize=8, color=TEXT, va="center")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    style(ax, "Pages removed per run at an equal budget of three sweeps", "pages removed per run")
    fig.suptitle("200 paired seeds, endogenous relocation, page_deletion_fraction 0.5", color=MUTED, fontsize=9, x=0.01, ha="left")
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(out / "page_level_budget.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("docs/blog/figures/wiki-mc"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_detection, fig_moderation, fig_displacement, fig_emergence, fig_page_level):
        fn(args.runs, args.out)
        print("wrote", fn.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
