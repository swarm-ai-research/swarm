"""Tradeoff scatters, Pareto frontiers, and sweep explorers for SWARM.

Every public function returns ``(fig, ax)`` or ``(fig, axes)``.

Usage::

    from swarm.analysis.scatter import plot_toxicity_welfare_scatter
    fig, ax = plot_toxicity_welfare_scatter([
        {"toxicity": 0.12, "welfare": 0.85, "persona": "honest"},
    ])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    SWARM_DIVERGING,
    agent_color,
    swarm_theme,
)

_MARKER_CYCLE: List[str] = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "p"]
_SIZE_MIN: float = 30.0
_SIZE_MAX: float = 200.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_pareto_frontier(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return indices of 2-D Pareto-optimal points (maximise both), sorted by *x*."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                is_pareto[i] = False
                break
    indices = np.where(is_pareto)[0]
    return indices[np.argsort(x[indices])]


def _unique_groups(values: Sequence[Any]) -> List[Any]:
    """Unique values preserving first-seen order."""
    seen: dict[Any, None] = {}
    for v in values:
        seen.setdefault(v, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Core scatter
# ---------------------------------------------------------------------------

def plot_tradeoff_scatter(
    ax: matplotlib.axes.Axes,
    x: Sequence[float],
    y: Sequence[float],
    *,
    colors: Optional[Sequence[str]] = None,
    sizes: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    shapes: Optional[Sequence[str]] = None,
    xlabel: str = "x",
    ylabel: str = "y",
    pareto: bool = False,
    density_underlay: bool = False,
    marginals: bool = False,
) -> matplotlib.axes.Axes:
    """Core tradeoff scatter on *ax* with optional Pareto frontier, KDE, and rugs.

    Example::

        fig, ax = plt.subplots()
        plot_tradeoff_scatter(ax, [0.1, 0.4], [0.9, 0.5], pareto=True)
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = len(xa)
    if colors is None:
        colors = [COLORS.WELFARE] * n
    if sizes is None:
        sizes = [60.0] * n

    # Density underlay
    if density_underlay and n >= 5:
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([xa, ya])
            kde = gaussian_kde(xy)
            xg = np.linspace(float(xa.min()), float(xa.max()), 80)
            yg = np.linspace(float(ya.min()), float(ya.max()), 80)
            X, Y = np.meshgrid(xg, yg)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contourf(X, Y, Z, levels=6, cmap="Blues", alpha=0.25)
        except ImportError:
            pass

    # Scatter points
    if shapes is not None:
        for marker in _unique_groups(shapes):
            idxs = np.array([i for i, s in enumerate(shapes) if s == marker])
            ax.scatter(
                xa[idxs], ya[idxs],
                c=[colors[i] for i in idxs],
                s=[sizes[i] for i in idxs],
                marker=marker, edgecolors="white", linewidths=0.4, zorder=3,
            )
    else:
        ax.scatter(
            xa, ya, c=list(colors), s=list(sizes),
            marker="o", edgecolors="white", linewidths=0.4, zorder=3,
        )

    # Pareto frontier
    if pareto and n >= 2:
        pidx = _compute_pareto_frontier(xa, ya)
        if len(pidx) >= 2:
            ax.plot(
                xa[pidx], ya[pidx],
                color=COLORS.EVASION, linewidth=1.6, linestyle="--",
                zorder=4, label="Pareto frontier",
            )
            ax.scatter(
                xa[pidx], ya[pidx],
                facecolors="none", edgecolors=COLORS.EVASION,
                s=120, linewidths=1.4, zorder=5,
            )

    # Marginal rug plots
    if marginals:
        rc = COLORS.TEXT_MUTED
        for xi in xa:
            ax.plot([xi, xi], [ax.get_ylim()[0], ax.get_ylim()[1]],
                    color=rc, alpha=0.5, linewidth=0.8, clip_on=False)
        for yi in ya:
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [yi, yi],
                    color=rc, alpha=0.5, linewidth=0.8, clip_on=False)
        ax.tick_params(axis="x", direction="inout", length=4)
        ax.tick_params(axis="y", direction="inout", length=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


# ---------------------------------------------------------------------------
# High-level figure builders
# ---------------------------------------------------------------------------

_KNOWN_AGENTS = {"honest", "deceptive", "opportunistic", "adversarial",
                 "planner", "adaptive_adversary"}


def plot_toxicity_welfare_scatter(
    data: List[Dict[str, Any]],
    *,
    color_by: Optional[str] = "persona",
    shape_by: Optional[str] = None,
    title: str = "Toxicity vs Welfare",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Toxicity (x) vs welfare (y) with Pareto frontier, coloured by *color_by*.

    Example::

        fig, ax = plot_toxicity_welfare_scatter([
            {"toxicity": 0.1, "welfare": 0.9, "persona": "honest"},
        ])
    """
    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(8, 6))
        xs = [d["toxicity"] for d in data]
        ys = [d["welfare"] for d in data]

        # Colours
        point_colors: Optional[List[str]] = None
        if color_by is not None:
            from matplotlib.lines import Line2D

            from swarm.analysis.theme import AGENT_CYCLE
            groups = [d.get(color_by, "unknown") for d in data]
            unique = _unique_groups(groups)
            palette: Dict[str, str] = {}
            for i, g in enumerate(unique):
                key = str(g).lower()
                palette[g] = agent_color(key) if key in _KNOWN_AGENTS \
                    else AGENT_CYCLE[i % len(AGENT_CYCLE)]
            point_colors = [palette[g] for g in groups]
            ax.legend(
                handles=[
                    Line2D([0], [0], marker="o", color="none",
                           markerfacecolor=palette[g], markersize=8, label=str(g))
                    for g in unique
                ],
                title=str(color_by).title(), loc="best",
            )

        # Shapes
        point_shapes: Optional[List[str]] = None
        if shape_by is not None:
            sgroups = [d.get(shape_by, "default") for d in data]
            smap = {g: _MARKER_CYCLE[i % len(_MARKER_CYCLE)]
                    for i, g in enumerate(_unique_groups(sgroups))}
            point_shapes = [smap[g] for g in sgroups]

        plot_tradeoff_scatter(
            ax, xs, ys,
            colors=point_colors, shapes=point_shapes,
            xlabel="Toxicity", ylabel="Welfare",
            pareto=True, density_underlay=len(data) >= 10, marginals=True,
        )
        ax.set_title(title, fontsize=13, pad=10)
        fig.tight_layout()
    return fig, ax


def plot_welfare_frontier(
    sweep_data: List[Dict[str, Any]],
    *,
    x_metric: str = "productivity",
    y_metric: str = "equality",
    color_metric: str = "enforcement_cost",
    title: str = "Welfare Frontier",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Pareto frontier sweep: *x_metric* vs *y_metric*, coloured by *color_metric*.

    Example::

        fig, ax = plot_welfare_frontier([
            {"productivity": 0.8, "equality": 0.6, "enforcement_cost": 0.1},
        ])
    """
    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(8, 6))
        xs = np.array([d[x_metric] for d in sweep_data], dtype=float)
        ys = np.array([d[y_metric] for d in sweep_data], dtype=float)

        # Colour mapping
        cvals = None
        if color_metric and all(color_metric in d for d in sweep_data):
            from swarm.analysis.theme import color_for_values
            cvals = np.array([d[color_metric] for d in sweep_data], dtype=float)
            rgba = color_for_values(list(cvals), cmap=SWARM_DIVERGING)
            pcols: Any = [rgba[i] for i in range(len(cvals))]
        else:
            pcols = [COLORS.WELFARE] * len(xs)

        ax.scatter(xs, ys, c=pcols, s=50, edgecolors="white",
                   linewidths=0.4, alpha=0.6, zorder=2)

        # Frontier
        pidx = _compute_pareto_frontier(xs, ys)
        if len(pidx) >= 2:
            ax.plot(xs[pidx], ys[pidx], color=COLORS.EVASION,
                    linewidth=2.0, zorder=4, label="Pareto frontier")
            ax.scatter(xs[pidx], ys[pidx], facecolors="none",
                       edgecolors=COLORS.EVASION, s=120, linewidths=1.6, zorder=5)

        # Colour bar
        if cvals is not None:
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=float(cvals.min()), vmax=float(cvals.max()))
            sm = cm.ScalarMappable(cmap=SWARM_DIVERGING, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(color_metric.replace("_", " ").title(), fontsize=10)

        ax.set_xlabel(x_metric.replace("_", " ").title())
        ax.set_ylabel(y_metric.replace("_", " ").title())
        ax.set_title(title, fontsize=13, pad=10)
        ax.legend(loc="best")
        fig.tight_layout()
    return fig, ax


def plot_sweep_scatter(
    sweep_results: List[Dict[str, Any]],
    *,
    x_param: str = "rho",
    y_metric: str = "toxicity",
    color_param: Optional[str] = None,
    facet_param: Optional[str] = None,
    title: str = "Sweep Explorer",
    mode: str = "dark",
) -> Tuple[plt.Figure, Any]:
    """Sweep explorer: *x_param* vs *y_metric*, faceted by *facet_param*.

    Example::

        fig, axes = plot_sweep_scatter(
            [{"rho": 0.0, "toxicity": 0.45, "model": "A"}],
            facet_param="model",
        )
    """
    with swarm_theme(mode):
        if facet_param is not None:
            facet_values = _unique_groups(
                [d.get(facet_param, "all") for d in sweep_results])
        else:
            facet_values = [None]

        ncols = min(len(facet_values), 3)
        nrows = (len(facet_values) + ncols - 1) // ncols
        fig, axes_arr = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
        axes_flat = axes_arr.ravel()

        for idx, fval in enumerate(facet_values):
            ax = axes_flat[idx]
            if fval is not None:
                subset = [d for d in sweep_results if d.get(facet_param or "") == fval]
                ax.set_title(f"{facet_param}={fval}", fontsize=10)
            else:
                subset = list(sweep_results)

            xs = np.array([d[x_param] for d in subset], dtype=float)
            ys = np.array([d[y_metric] for d in subset], dtype=float)

            pcols: Any = None
            if color_param and all(color_param in d for d in subset):
                from swarm.analysis.theme import color_for_values
                cvals = np.array([d[color_param] for d in subset], dtype=float)
                rgba = color_for_values(list(cvals), cmap=SWARM_DIVERGING)
                pcols = [rgba[i] for i in range(len(cvals))]

            ax.scatter(
                xs, ys,
                c=pcols if pcols is not None else COLORS.WELFARE,
                s=60, edgecolors="white", linewidths=0.4, zorder=3,
            )
            ax.set_xlabel(x_param.replace("_", " ").title())
            ax.set_ylabel(y_metric.replace("_", " ").title())

        for idx in range(len(facet_values), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(title, fontsize=14, y=1.01)
        fig.tight_layout()

        if facet_param is None:
            return fig, axes_flat[0]
        return fig, axes_arr
