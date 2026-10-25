"""Figures for the 50bb evidence audit.  python experiments/plot_collusion_evidence_audit.py <run_dir> [out_dir]"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dd", "#fcfcfb"
HONEST = "#8a897f"
RAMP = {0.1: "#86b6ef", 0.2: "#3987e5", 0.4: "#184f95"}
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def ecdf_figure(rows, r_star, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    test = [r for r in rows if r["split"] == "test"]

    def curve(vals, color, label):
        x = np.sort(np.clip(vals, 0.05, None))
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, lw=2, label=label)

    curve([r["scores"]["day_perm"]["ratio"] for r in test if r["f"] == 0], HONEST,
          "honest group (f=0)")
    for f, c in RAMP.items():
        curve([r["scores"]["day_perm"]["ratio"] for r in test if r["f"] == f], c,
              f"planted, f={f}")
    ax.axvline(r_star, color=INK, lw=1, ls=(0, (3, 2)))
    ax.text(r_star * 1.04, 0.06, f"frozen bar r* = {r_star:.2f}\n(95th pct honest, calib.)",
            color=INK, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("enrichment ratio R = T_obs / mean T_null   (day_perm null, log scale)")
    ax.set_ylabel("share of groups with ratio ≤ R")
    ax.grid(axis="both", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_ratio_ecdf.{ext}", dpi=200)
    plt.close(fig)


def rates_figure(summary, out, f="0.2"):
    rules = summary["rules"]["test"]
    names = list(rules)
    short = [n.replace("p<0.05 under ", "p<.05, ").replace("frozen: ", "frozen rule:\n")
             .replace(" and ratio>=", " & R≥") for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = np.arange(len(names))[::-1]
    for (key, color, label, dy) in (("0.0", ORANGE, "honest groups flagged (false positive)", 0.14),
                                    (f, BLUE, f"planted groups flagged (f={f})", -0.14)):
        rate = np.array([rules[n][key]["rate"] for n in names])
        lo = np.array([rules[n][key]["wilson95"][0] for n in names])
        hi = np.array([rules[n][key]["wilson95"][1] for n in names])
        ax.errorbar(rate, y + dy, xerr=[rate - lo, hi - rate], fmt="o", ms=5, color=color,
                    ecolor=color, elinewidth=1.2, capsize=0, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(0.05, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.set_yticks(y, short, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("share flagged, test months (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_flag_rates.{ext}", dpi=200)
    plt.close(fig)


def main():
    run = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "plots"
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((run / "adjudication.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    ecdf_figure(rows, summary["r_star"], out)
    rates_figure(summary, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
