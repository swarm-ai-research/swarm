"""Plot the adaptive-vs-static overlay.

Reads ``grid_summary.csv`` from the adaptive runner and
``static_summary.csv`` from the static-grid runner; produces a two-
panel figure (welfare × ρ on the left, toxicity × ρ on the right)
with all conditions overlaid. Seed-averaged with ±1σ shaded.

This is the visual replacement for the original Figure 4. The
toxicity panel shows the structural-inertness finding: all four
curves are flat. The welfare panel shows the adaptive-vs-static
gap.

Pure-matplotlib, no seaborn or other deps. Pre-registered grid is
the default but ``load_*`` are parameterized for ablations.

Additionally, the module provides ``load_judged()`` to read
``judged_summary.csv`` and ``plot_judged_overlay()`` to produce a
three-panel calibration plot (proxy toxicity, judge toxicity, and
target-rule firing rate).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Condition:
    """One overlaid line on the plot (one agent class)."""

    name: str
    label: str
    color: str
    marker: str
    linestyle: str = "-"


CONDITIONS: list[Condition] = [
    Condition("adaptive", "adaptive (CEM)", "#1f77b4", "o"),
    Condition("static_honest", "static honest", "#2ca02c", "s"),
    Condition("static_mixed", "static mixed (70/30)", "#ff7f0e", "^"),
    Condition("static_toxic", "static toxic", "#d62728", "v"),
]


def load_adaptive(path: str) -> dict[float, dict[str, list[float]]]:
    """Read adaptive grid_summary.csv → {ρ: {metric: [per-seed values]}}."""
    out: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path) as f:
        for row in csv.DictReader(f):
            rho = float(row["rho"])
            out[rho]["welfare"].append(float(row["final_mean_payoff_attempted"]))
            out[rho]["toxicity"].append(float(row["final_toxicity"]))
            out[rho]["accept"].append(float(row["final_accept_rate"]))
    return dict(out)


def load_static(path: str) -> dict[str, dict[float, dict[str, list[float]]]]:
    """Read static_summary.csv → {baseline: {ρ: {metric: [per-seed values]}}}."""
    out: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    with open(path) as f:
        for row in csv.DictReader(f):
            b = row["baseline"]
            rho = float(row["rho"])
            out[b][rho]["welfare"].append(float(row["mean_payoff_attempted"]))
            out[b][rho]["toxicity"].append(float(row["toxicity"]))
            out[b][rho]["accept"].append(float(row["accept_rate"]))
    return {b: dict(d) for b, d in out.items()}


def _mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _collect_lines(
    adaptive: dict[float, dict[str, list[float]]],
    static: dict[str, dict[float, dict[str, list[float]]]],
    metric: str,
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """Returns {condition: (rhos, means, stds)} ordered by ρ."""
    rhos_a = sorted(adaptive.keys())
    out: dict[str, tuple[list[float], list[float], list[float]]] = {}
    means = [_mean_sd(adaptive[r][metric])[0] for r in rhos_a]
    stds = [_mean_sd(adaptive[r][metric])[1] for r in rhos_a]
    out["adaptive"] = (rhos_a, means, stds)
    for b, by_rho in static.items():
        rhos = sorted(by_rho.keys())
        means = [_mean_sd(by_rho[r][metric])[0] for r in rhos]
        stds = [_mean_sd(by_rho[r][metric])[1] for r in rhos]
        out[f"static_{b}"] = (rhos, means, stds)
    return out


def load_judged(path: str) -> dict[str, dict[float, dict[str, list[float]]]]:
    """Read judged_summary.csv → {condition: {ρ: {metric: [per-seed values]}}}."""
    out: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    with open(path) as f:
        for row in csv.DictReader(f):
            c = row["condition"]
            rho = float(row["rho"])
            out[c][rho]["proxy_toxicity"].append(float(row["proxy_toxicity"]))
            out[c][rho]["judge_toxicity"].append(float(row["judge_toxicity"]))
            out[c][rho]["gap"].append(float(row["proxy_judge_gap"]))
            out[c][rho]["target_fire_pct"].append(
                100.0 * int(row["n_target_rule_fired"]) / max(1, int(row["n_scored"]))
            )
    return {c: dict(d) for c, d in out.items()}


def plot_judged_overlay(
    judged_path: str,
    output_path: str,
    *,
    title: str = "Calibration anchor (rubric v3) overlay — arm 2",
) -> None:
    """Three-panel plot: proxy toxicity, judge toxicity, target-fire %.

    The target-fire % panel is the load-bearing diagnostic — it shows
    where the v3 anchor was informative (~100%) vs uninformative (~0%)
    per condition.
    """
    data = load_judged(judged_path)
    fig, (ax_p, ax_j, ax_t) = plt.subplots(1, 3, figsize=(15, 5))

    def _ms(by_rho: dict[float, dict[str, list[float]]], rhos: list[float], key: str) -> tuple[list[float], list[float]]:
        means = [_mean_sd(by_rho[r][key])[0] for r in rhos]
        stds = [_mean_sd(by_rho[r][key])[1] for r in rhos]
        return means, stds

    for cond in CONDITIONS:
        if cond.name not in data:
            continue
        by_rho = data[cond.name]
        rhos = sorted(by_rho.keys())

        m, s = _ms(by_rho, rhos, "proxy_toxicity")
        ax_p.errorbar(rhos, m, yerr=s, color=cond.color, marker=cond.marker,
                      linestyle=cond.linestyle, label=cond.label,
                      linewidth=2, markersize=7, capsize=3)
        m, s = _ms(by_rho, rhos, "judge_toxicity")
        ax_j.errorbar(rhos, m, yerr=s, color=cond.color, marker=cond.marker,
                      linestyle=cond.linestyle, label=cond.label,
                      linewidth=2, markersize=7, capsize=3)
        m, s = _ms(by_rho, rhos, "target_fire_pct")
        ax_t.errorbar(rhos, m, yerr=s, color=cond.color, marker=cond.marker,
                      linestyle=cond.linestyle, label=cond.label,
                      linewidth=2, markersize=7, capsize=3)

    for ax, ylabel, title_panel in [
        (ax_p, "proxy toxicity (E[1−p | accepted])", "Proxy"),
        (ax_j, "judge toxicity (1 − mean v3 score)", "Judge (rubric v3)"),
        (ax_t, "target-rule firing rate (%)", "Anchor informativeness"),
    ]:
        ax.set_xlabel("ρ (externality internalization)")
        ax.set_ylabel(ylabel)
        ax.set_title(title_panel)
        ax.grid(True, alpha=0.3)
    ax_p.set_ylim(-0.02, 0.75)
    ax_j.set_ylim(-0.02, 1.0)
    ax_t.set_ylim(-5, 105)
    ax_p.legend(loc="upper right", fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    adaptive_path: str,
    static_path: str,
    output_path: str,
    *,
    title: str = "Adaptive vs static — arm 2 (5 seeds × 6 ρ)",
) -> None:
    adaptive = load_adaptive(adaptive_path)
    static = load_static(static_path)

    welfare_lines = _collect_lines(adaptive, static, "welfare")
    toxicity_lines = _collect_lines(adaptive, static, "toxicity")

    fig, (ax_w, ax_t) = plt.subplots(1, 2, figsize=(12, 5))

    for cond in CONDITIONS:
        if cond.name not in welfare_lines:
            continue
        rhos, means, stds = welfare_lines[cond.name]
        ax_w.errorbar(
            rhos, means, yerr=stds,
            color=cond.color, marker=cond.marker, linestyle=cond.linestyle,
            label=cond.label, linewidth=2, markersize=7, capsize=3,
        )
        rhos, means, stds = toxicity_lines[cond.name]
        ax_t.errorbar(
            rhos, means, yerr=stds,
            color=cond.color, marker=cond.marker, linestyle=cond.linestyle,
            label=cond.label, linewidth=2, markersize=7, capsize=3,
        )

    ax_w.set_xlabel("ρ (externality internalization)")
    ax_w.set_ylabel("welfare (mean payoff per attempted)")
    ax_w.set_title("Welfare")
    ax_w.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best", fontsize=9)

    ax_t.set_xlabel("ρ (externality internalization)")
    ax_t.set_ylabel("toxicity (E[1−p | accepted])")
    ax_t.set_title("Toxicity")
    ax_t.set_ylim(-0.02, 0.75)
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper right", fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
