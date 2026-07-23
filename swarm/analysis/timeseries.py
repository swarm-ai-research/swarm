"""Epoch-over-epoch line charts with confidence ribbons.

Matplotlib-based plotting module for visualising SWARM safety-simulation
metrics across training epochs.  Every function uses the project theme
(``swarm.analysis.theme``) and returns ``(fig, axes)`` so callers can
further customise or save to disk.

Quick start::

    from swarm.analysis.timeseries import plot_toxicity_welfare

    fig, (ax_tox, ax_wel) = plot_toxicity_welfare(
        {"epochs": list(range(50)),
         "toxicity": tox_values,
         "welfare": wel_values},
    )
    fig.savefig("runs/latest/plots/tox_welfare.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    METRIC_COLOR_MAP,
    add_danger_zone,
    annotate_events,
    swarm_theme,
)

__all__ = [
    "plot_metric_timeseries",
    "plot_toxicity_welfare",
    "plot_selection_geometry",
    "plot_bilevel_loop",
    "plot_multi_seed_timeseries",
]

_RIBBON_ALPHA = 0.18

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Causal (backward-looking) rolling mean with partial windows."""
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = np.mean(values[start : i + 1])
    return out


def _resolve_color(label: str, fallback: str) -> str:
    """Pick a colour from METRIC_COLOR_MAP or fall back."""
    return METRIC_COLOR_MAP.get(label.lower(), fallback)


def _to_2d(values: Any) -> Tuple[np.ndarray, bool]:
    """Return *(arr, is_multi_seed)*.  2-D input has shape (n_seeds, n_epochs)."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        return arr, True
    return arr.ravel(), False


# ---------------------------------------------------------------------------
# 1. Core building block
# ---------------------------------------------------------------------------


def plot_metric_timeseries(
    ax: matplotlib.axes.Axes,
    epochs: Sequence[int],
    values: Any,
    *,
    label: str = "",
    color: Optional[str] = None,
    ribbon_lo: Optional[Sequence[float]] = None,
    ribbon_hi: Optional[Sequence[float]] = None,
    rolling_window: int = 0,
) -> matplotlib.axes.Axes:
    """Plot a single metric line on *ax* with an optional confidence ribbon.

    *values* may be 1-D (one series) or 2-D ``(n_seeds, n_epochs)``; when
    2-D the mean is plotted and a +/- 1-std ribbon is added automatically.
    If *rolling_window* > 0 a causal rolling mean is applied first.

    Example::

        fig, ax = plt.subplots()
        plot_metric_timeseries(
            ax, list(range(100)), np.random.rand(100),
            label="toxicity", color=COLORS.TOXICITY, rolling_window=5,
        )
    """
    color = color or _resolve_color(label, COLORS.WELFARE)
    x = np.asarray(epochs)

    arr, multi = _to_2d(values)
    if multi:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ribbon_lo_arr: Optional[np.ndarray] = mean - std
        ribbon_hi_arr: Optional[np.ndarray] = mean + std
        y = mean
    else:
        y = arr
        ribbon_lo_arr = np.asarray(ribbon_lo, dtype=float) if ribbon_lo is not None else None
        ribbon_hi_arr = np.asarray(ribbon_hi, dtype=float) if ribbon_hi is not None else None

    if rolling_window > 0:
        y = _rolling_mean(y, rolling_window)
        if ribbon_lo_arr is not None:
            ribbon_lo_arr = _rolling_mean(ribbon_lo_arr, rolling_window)
        if ribbon_hi_arr is not None:
            ribbon_hi_arr = _rolling_mean(ribbon_hi_arr, rolling_window)

    ax.plot(x, y, color=color, label=label or None, linewidth=1.8)

    if ribbon_lo_arr is not None and ribbon_hi_arr is not None:
        ax.fill_between(x, ribbon_lo_arr, ribbon_hi_arr, color=color, alpha=_RIBBON_ALPHA)

    return ax


# ---------------------------------------------------------------------------
# 2. Two-panel toxicity / welfare chart
# ---------------------------------------------------------------------------


def plot_toxicity_welfare(
    metrics_df_or_dict: Any,
    *,
    toxicity_key: str = "toxicity",
    welfare_key: str = "welfare",
    epochs_key: str = "epochs",
    threshold: float = 0.5,
    events: Optional[List[Dict[str, Any]]] = None,
    title: str = "Toxicity & Welfare Over Epochs",
    mode: str = "dark",
) -> Tuple[plt.Figure, Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Two-panel stacked figure: toxicity (top) with danger zone, welfare (bottom).

    *metrics_df_or_dict* may be a pandas DataFrame or a plain dict whose
    keys match *epochs_key*, *toxicity_key*, and *welfare_key*.  Governance
    *events* (list of ``{"epoch": int, "label": str}``) are annotated as
    vertical lines on both panels.

    Example::

        data = {
            "epochs": list(range(30)),
            "toxicity": np.random.rand(30) * 0.6,
            "welfare": np.cumsum(np.random.randn(30) * 0.1),
        }
        fig, (ax_t, ax_w) = plot_toxicity_welfare(data, threshold=0.4)
    """
    # Normalise input
    try:
        epochs = metrics_df_or_dict[epochs_key].tolist()
        tox = metrics_df_or_dict[toxicity_key].tolist()
        wel = metrics_df_or_dict[welfare_key].tolist()
    except (AttributeError, TypeError):
        epochs = list(metrics_df_or_dict[epochs_key])
        tox = list(metrics_df_or_dict[toxicity_key])
        wel = list(metrics_df_or_dict[welfare_key])

    with swarm_theme(mode):
        fig, (ax_tox, ax_wel) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 7),
            gridspec_kw={"hspace": 0.12},
        )

        # Top panel: toxicity
        plot_metric_timeseries(
            ax_tox, epochs, tox,
            label="toxicity", color=COLORS.TOXICITY,
        )
        ax_tox.set_ylabel("Toxicity")
        ax_tox.set_ylim(bottom=0.0)
        add_danger_zone(ax_tox, threshold, direction="above",
                        label=f"threshold={threshold}")

        # Bottom panel: welfare
        plot_metric_timeseries(
            ax_wel, epochs, wel,
            label="welfare", color=COLORS.WELFARE,
        )
        ax_wel.set_ylabel("Welfare")
        ax_wel.set_xlabel("Epoch")

        # Governance event annotations
        if events:
            annotate_events(ax_tox, events)
            annotate_events(ax_wel, events)

        fig.suptitle(title, fontsize=13, y=0.97)
        for a in (ax_tox, ax_wel):
            a.legend(loc="upper right", fontsize=8)

    return fig, (ax_tox, ax_wel)


# ---------------------------------------------------------------------------
# 2b. Selection geometry (saturation + toxicity decomposition)
# ---------------------------------------------------------------------------


def plot_selection_geometry(
    metrics_df_or_dict: Any,
    *,
    saturation_key: str = "selection_saturation",
    baseline_key: str = "baseline_harm",
    credit_key: str = "selection_credit",
    epochs_key: str = "epochs",
    saturation_ceiling: float = 0.85,
    events: Optional[List[Dict[str, Any]]] = None,
    title: str = "Selection Geometry Over Epochs",
    mode: str = "dark",
) -> Tuple[plt.Figure, Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Two-panel diagnostic plot for projection-geometric metrics.

    Top panel: ``selection_saturation`` = |Q| / (σ_p/√(αβ)) ∈ [0, 1].
    A horizontal danger zone above *saturation_ceiling* marks the
    "proxy-bound" regime where further policy tuning yields little
    return — the proxy needs more signal, not the governor.

    Bottom panel: stacks ``baseline_harm`` (population floor) and
    ``-selection_credit`` (governance subtracting from harm) so the gap
    between the curves equals the realised toxicity
    ``T = baseline_harm - selection_credit``. Visual read: when the
    bottom curve dives below zero the governor is actively pulling
    toxicity down; when it sits near zero the proxy isn't sorting.

    *metrics_df_or_dict* is a DataFrame or dict keyed by the four keys
    above. Multi-seed values (2-D arrays) get mean+std ribbons.

    Example::

        from swarm.analysis.timeseries import plot_selection_geometry
        fig, (ax_s, ax_d) = plot_selection_geometry({
            "epochs": list(range(50)),
            "selection_saturation": saturation_per_epoch,
            "baseline_harm": baseline_per_epoch,
            "selection_credit": credit_per_epoch,
        })
        fig.savefig("runs/latest/plots/selection_geometry.png")
    """
    try:
        epochs = metrics_df_or_dict[epochs_key].tolist()
        sat = metrics_df_or_dict[saturation_key]
        base = metrics_df_or_dict[baseline_key]
        cred = metrics_df_or_dict[credit_key]
    except (AttributeError, TypeError):
        epochs = list(metrics_df_or_dict[epochs_key])
        sat = metrics_df_or_dict[saturation_key]
        base = metrics_df_or_dict[baseline_key]
        cred = metrics_df_or_dict[credit_key]

    with swarm_theme(mode):
        fig, (ax_sat, ax_decomp) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 7),
            gridspec_kw={"hspace": 0.12},
        )

        plot_metric_timeseries(
            ax_sat, epochs, sat,
            label="saturation", color=COLORS.EVASION,
        )
        ax_sat.set_ylabel("C-S saturation |Q| / bound")
        ax_sat.set_ylim(0.0, 1.05)
        add_danger_zone(
            ax_sat, saturation_ceiling, direction="above",
            label=f"proxy-bound > {saturation_ceiling}",
        )

        plot_metric_timeseries(
            ax_decomp, epochs, base,
            label="baseline harm (1 − E[p])", color=COLORS.TOXICITY,
        )
        # Selection credit is subtractive in T = baseline − credit; plot
        # as negative so the visual distance between the two lines is
        # the realised toxicity.
        neg_cred_arr, _ = _to_2d(cred)
        neg_cred: Any = -neg_cred_arr
        plot_metric_timeseries(
            ax_decomp, epochs, neg_cred,
            label="− selection credit (−β·Q)", color=COLORS.WELFARE,
        )
        ax_decomp.axhline(0.0, color=COLORS.TEXT_MUTED, linewidth=0.8, linestyle="--")
        ax_decomp.set_ylabel("Toxicity decomposition")
        ax_decomp.set_xlabel("Epoch")

        if events:
            annotate_events(ax_sat, events)
            annotate_events(ax_decomp, events)

        fig.suptitle(title, fontsize=13, y=0.97)
        for a in (ax_sat, ax_decomp):
            a.legend(loc="upper right", fontsize=8)

    return fig, (ax_sat, ax_decomp)


# ---------------------------------------------------------------------------
# 3. Bilevel loop (planner + workers)
# ---------------------------------------------------------------------------

_SERIES_CYCLE = [
    COLORS.PLANNER, COLORS.EVASION, COLORS.PRODUCTIVITY,
    COLORS.REVENUE, COLORS.WELFARE, COLORS.TOXICITY,
]


def plot_bilevel_loop(
    planner_data: Dict[str, Any],
    worker_data: Dict[str, Any],
    *,
    title: str = "Bilevel Loop Dynamics",
    mode: str = "dark",
) -> Tuple[plt.Figure, Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
    """Dual time-series panel for AI-Economist bilevel dynamics.

    Top panel = planner tax parameters, bottom = worker metrics
    (productivity, reported income, evasion rate).  Planner update
    boundaries are drawn as vertical dashed lines when *planner_data*
    contains an ``"update_epochs"`` key.

    Example::

        planner = {
            "epochs": list(range(50)),
            "tax_rate": np.linspace(0.2, 0.35, 50),
            "update_epochs": [10, 20, 30, 40],
        }
        worker = {
            "epochs": list(range(50)),
            "productivity": np.random.rand(50) * 0.8 + 0.2,
            "evasion": np.random.rand(50) * 0.3,
        }
        fig, axes = plot_bilevel_loop(planner, worker)
    """
    planner_epochs = planner_data["epochs"]
    worker_epochs = worker_data["epochs"]
    update_epochs = planner_data.get("update_epochs", [])

    with swarm_theme(mode):
        fig, (ax_p, ax_w) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 7),
            gridspec_kw={"hspace": 0.12},
        )

        # Top: planner metrics
        cidx = 0
        for key, series in planner_data.items():
            if key in ("epochs", "update_epochs"):
                continue
            color = _resolve_color(key, _SERIES_CYCLE[cidx % len(_SERIES_CYCLE)])
            plot_metric_timeseries(
                ax_p, planner_epochs, series,
                label=key.replace("_", " "), color=color,
            )
            cidx += 1

        # Planner update boundaries on both panels
        for ue in update_epochs:
            for a in (ax_p, ax_w):
                a.axvline(ue, color=COLORS.TEXT_MUTED, linestyle="--",
                          linewidth=0.7, alpha=0.6)

        ax_p.set_ylabel("Planner Parameters")
        ax_p.legend(loc="upper right", fontsize=8)

        # Bottom: worker metrics
        cidx = 0
        for key, series in worker_data.items():
            if key == "epochs":
                continue
            color = _resolve_color(key, _SERIES_CYCLE[cidx % len(_SERIES_CYCLE)])
            plot_metric_timeseries(
                ax_w, worker_epochs, series,
                label=key.replace("_", " "), color=color,
            )
            cidx += 1

        ax_w.set_ylabel("Worker Metrics")
        ax_w.set_xlabel("Epoch")
        ax_w.legend(loc="upper right", fontsize=8)

        fig.suptitle(title, fontsize=13, y=0.97)

    return fig, (ax_p, ax_w)


# ---------------------------------------------------------------------------
# 4. Multi-seed overlay
# ---------------------------------------------------------------------------


def plot_multi_seed_timeseries(
    all_seed_data: List[Dict[str, Any]],
    metric: str,
    *,
    title: Optional[str] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plot *metric* across multiple seeds: mean line + std ribbon.

    Each element of *all_seed_data* must contain ``"epochs"`` and the
    named *metric*.  Faint individual seed traces are drawn behind the
    aggregate for transparency.

    Example::

        seeds = [
            {"epochs": list(range(20)),
             "toxicity": (np.random.rand(20) * 0.5).tolist()}
            for _ in range(8)
        ]
        fig, ax = plot_multi_seed_timeseries(seeds, "toxicity")
    """
    if not all_seed_data:
        raise ValueError("all_seed_data must contain at least one seed dict")

    epochs = np.asarray(all_seed_data[0]["epochs"])
    stacked = np.array([np.asarray(sd[metric], dtype=float) for sd in all_seed_data])

    color = _resolve_color(metric, COLORS.WELFARE)
    title = title or f"{metric.replace('_', ' ').title()} Across Seeds"

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(10, 5))

        # 2-D input: plot_metric_timeseries computes mean +/- std
        plot_metric_timeseries(ax, list(epochs), stacked, label=metric, color=color)

        # Faint individual seed traces
        for i in range(stacked.shape[0]):
            ax.plot(epochs, stacked[i], color=color, alpha=0.12, linewidth=0.7)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title, fontsize=12)
        ax.legend(
            [f"mean +/- 1 std  (n={stacked.shape[0]} seeds)"],
            loc="upper right", fontsize=8,
        )

    return fig, ax
