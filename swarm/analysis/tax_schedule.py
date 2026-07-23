"""Piecewise tax schedule visualization for the AI Economist GTB scenario.

Renders marginal and effective tax-rate curves as step functions,
supports bracket-boundary shading for bunching analysis, schedule
comparisons (old vs new), and temporal evolution across epochs.

Usage::

    from swarm.analysis.tax_schedule import plot_tax_schedule_figure

    fig, ax = plot_tax_schedule_figure(
        brackets=[0, 20_000, 50_000, 100_000],
        rates=[0.10, 0.20, 0.30, 0.40],
        show_effective=True,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    swarm_theme,
)

# ---------------------------------------------------------------------------
# Pure computation
# ---------------------------------------------------------------------------


def build_step_function(
    brackets: Sequence[float],
    rates: Sequence[float],
) -> Tuple[List[float], List[float]]:
    """Build x/y arrays for a step-function plot of tax rates.

    Args:
        brackets: Income thresholds for bracket boundaries.
        rates: Marginal tax rate for each bracket.

    Returns:
        Tuple of (xs, ys) lists defining the step function vertices.
    """
    brackets = list(brackets)
    rates = list(rates)
    max_income = brackets[-1] * 1.5 if len(brackets) > 1 else 100_000
    xs: List[float] = []
    ys: List[float] = []
    for i, rate in enumerate(rates):
        lower = brackets[i]
        upper = brackets[i + 1] if i + 1 < len(brackets) else max_income
        xs.extend([lower, upper])
        ys.extend([rate, rate])
    return xs, ys


def compute_effective_rate(
    brackets: Sequence[float],
    rates: Sequence[float],
    income_range: np.ndarray,
) -> np.ndarray:
    """Compute the effective (average) tax rate for a range of income values.

    The effective rate at a given income is the total tax owed divided by
    that income.  Tax is computed by applying each marginal rate only to
    the portion of income within the corresponding bracket.

    Args:
        brackets: Monotonically increasing income thresholds defining
            bracket boundaries (e.g. ``[0, 20000, 50000, 100000]``).
        rates: Marginal tax rate for each bracket.  ``len(rates)`` must
            equal ``len(brackets)``.
        income_range: 1-D array of income values at which to evaluate
            the effective rate.

    Returns:
        Array of effective tax rates, same shape as *income_range*.
        Returns 0.0 for income values at or below zero.
    """
    brackets = list(brackets)
    rates = list(rates)
    income_range = np.asarray(income_range, dtype=float)
    effective = np.zeros_like(income_range)

    for idx, income in enumerate(income_range):
        if income <= 0:
            effective[idx] = 0.0
            continue
        total_tax = 0.0
        for i, rate in enumerate(rates):
            lower = brackets[i]
            upper = brackets[i + 1] if i + 1 < len(brackets) else np.inf
            if income <= lower:
                break
            taxable = min(income, upper) - lower
            total_tax += taxable * rate
        effective[idx] = total_tax / income

    return effective


# ---------------------------------------------------------------------------
# Core axes-level renderer
# ---------------------------------------------------------------------------


def plot_tax_schedule(
    ax: matplotlib.axes.Axes,
    brackets: Sequence[float],
    *,
    rates: Sequence[float],
    show_effective: bool = False,
    bunching_zones: Optional[Sequence[Tuple[float, float]]] = None,
    title: Optional[str] = None,
) -> None:
    """Draw a piecewise-linear tax schedule on the given *ax*.

    Args:
        ax: Matplotlib axes to draw on.
        brackets: Income thresholds for each bracket boundary.
        rates: Marginal rate for each bracket (same length as *brackets*).
        show_effective: If ``True``, overlay the effective (average) tax
            rate as a smooth line.
        bunching_zones: Optional list of ``(boundary, width)`` tuples.
            A shaded region of the given width is drawn centered on each
            boundary to highlight bunching incentives.
        title: Optional axes title.
    """
    brackets = list(brackets)
    rates = list(rates)

    # Build the step-function x/y arrays.
    xs, ys = build_step_function(brackets, rates)
    max_income = brackets[-1] * 1.5 if len(brackets) > 1 else 100_000

    ax.plot(xs, ys, color=COLORS.WELFARE, linewidth=2.0, label="Marginal rate")

    # Bracket boundary vertical lines.
    for boundary in brackets[1:]:
        ax.axvline(boundary, color=COLORS.TEXT_MUTED, linestyle="--",
                   linewidth=0.8, alpha=0.6)

    # Bunching zones.
    if bunching_zones:
        for boundary, width in bunching_zones:
            ax.axvspan(
                boundary - width / 2,
                boundary + width / 2,
                color=COLORS.EVASION,
                alpha=0.15,
                label="Bunching zone",
            )

    # Effective rate overlay.
    if show_effective:
        income_range = np.linspace(max(brackets[0], 1), max_income, 500)
        eff = compute_effective_rate(brackets, rates, income_range)
        ax.plot(income_range, eff, color=COLORS.REVENUE, linewidth=1.6,
                linestyle="-", alpha=0.85, label="Effective rate")

    ax.set_xlabel("Income")
    ax.set_ylabel("Tax rate")
    if title:
        ax.set_title(title)

    # De-duplicate legend labels.
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    unique_handles = []
    unique_labels = []
    for h, lbl in zip(handles, labels, strict=False):
        if lbl not in seen:
            seen[lbl] = True
            unique_handles.append(h)
            unique_labels.append(lbl)
    ax.legend(unique_handles, unique_labels, fontsize=8)


# ---------------------------------------------------------------------------
# Figure-level wrappers
# ---------------------------------------------------------------------------


def plot_tax_schedule_figure(
    brackets: Sequence[float],
    rates: Sequence[float],
    *,
    show_effective: bool = False,
    bunching_zones: Optional[Sequence[Tuple[float, float]]] = None,
    title: Optional[str] = "Tax Schedule",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Create a full figure showing a single piecewise tax schedule.

    Args:
        brackets: Income thresholds for each bracket boundary.
        rates: Marginal rate for each bracket.
        show_effective: Overlay the effective (average) rate curve.
        bunching_zones: Optional bunching-zone highlights.
        title: Figure title.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    with swarm_theme(mode):
        fig, ax = plt.subplots()
        plot_tax_schedule(
            ax,
            brackets,
            rates=rates,
            show_effective=show_effective,
            bunching_zones=bunching_zones,
            title=title,
        )
    return fig, ax


def plot_tax_schedule_comparison(
    schedule_old: Dict[str, Sequence[float]],
    schedule_new: Dict[str, Sequence[float]],
    *,
    title: Optional[str] = "Tax Schedule Comparison",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Overlay old and new tax schedules for visual comparison.

    The old schedule is rendered at reduced opacity so the new schedule
    stands out.

    Args:
        schedule_old: Dict with ``"brackets"`` and ``"rates"`` keys.
        schedule_new: Dict with ``"brackets"`` and ``"rates"`` keys.
        title: Figure title.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    with swarm_theme(mode):
        fig, ax = plt.subplots()

        # --- Old schedule (faded) ---
        old_b = list(schedule_old["brackets"])
        old_r = list(schedule_old["rates"])
        xs_old, ys_old = build_step_function(old_b, old_r)
        ax.plot(xs_old, ys_old, color=COLORS.TEXT_MUTED, linewidth=1.6,
                alpha=0.45, label="Old schedule")

        # --- New schedule (full opacity) ---
        new_b = list(schedule_new["brackets"])
        new_r = list(schedule_new["rates"])
        xs_new, ys_new = build_step_function(new_b, new_r)
        ax.plot(xs_new, ys_new, color=COLORS.WELFARE, linewidth=2.0,
                label="New schedule")

        ax.set_xlabel("Income")
        ax.set_ylabel("Tax rate")
        if title:
            ax.set_title(title)
        ax.legend(fontsize=8)

    return fig, ax


def plot_tax_schedule_evolution(
    schedules: Sequence[Dict[str, Sequence[float]]],
    *,
    epochs: Sequence[int],
    title: Optional[str] = "Tax Schedule Evolution",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Visualize how the tax schedule changes across training epochs.

    Each epoch's schedule is drawn as a step function.  A color gradient
    from light to dark conveys temporal progression (earlier epochs are
    lighter, later epochs are darker).

    Args:
        schedules: List of dicts, each with ``"brackets"`` and ``"rates"``
            keys.  One entry per epoch.
        epochs: Corresponding epoch numbers (same length as *schedules*).
        title: Figure title.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    n = len(schedules)
    if n == 0:
        with swarm_theme(mode):
            fig, ax = plt.subplots()
        return fig, ax

    # Build a light-to-dark color gradient.
    base_rgb = np.array(plt.matplotlib.colors.to_rgb(COLORS.WELFARE))
    if mode == "dark":
        fade_target = np.array([0.15, 0.15, 0.15])
    else:
        fade_target = np.array([0.85, 0.85, 0.85])

    with swarm_theme(mode):
        fig, ax = plt.subplots()

        for idx, (schedule, epoch) in enumerate(zip(schedules, epochs, strict=False)):
            frac = idx / max(n - 1, 1)
            # Interpolate: early epochs -> faded, later epochs -> full color
            color = tuple(fade_target + frac * (base_rgb - fade_target))

            b = list(schedule["brackets"])
            r = list(schedule["rates"])
            xs, ys = build_step_function(b, r)

            ax.plot(xs, ys, color=color, linewidth=1.4,
                    alpha=0.4 + 0.6 * frac, label=f"Epoch {epoch}")

        ax.set_xlabel("Income")
        ax.set_ylabel("Tax rate")
        if title:
            ax.set_title(title)
        # Only show a subset of legend entries to avoid clutter.
        if n <= 8:
            ax.legend(fontsize=7)
        else:
            handles, labels = ax.get_legend_handles_labels()
            # Show first, middle, and last.
            pick = [0, n // 2, n - 1]
            ax.legend(
                [handles[i] for i in pick],
                [labels[i] for i in pick],
                fontsize=7,
            )

    return fig, ax
