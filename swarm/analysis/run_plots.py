"""Standard end-of-run plot bundle.

A thin wrapper that turns a list of EpochMetrics (or any iterable of
objects exposing ``.epoch``, ``.toxicity_rate``, ``.total_welfare``,
``.baseline_harm``, ``.selection_credit``, ``.selection_saturation``)
into the canonical PNG bundle dropped under ``<run_dir>/plots/``.

Only the toxicity/welfare and selection-geometry plots are produced
right now; add more by extending ``write_run_plots`` rather than
calling matplotlib from runner scripts directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence

__all__ = ["write_run_plots"]


def _extract_series(metrics: Sequence[Any], attr: str) -> List[float]:
    return [float(getattr(m, attr, 0.0) or 0.0) for m in metrics]


def write_run_plots(
    metrics_history: Iterable[Any],
    out_dir: str | Path,
    *,
    scenario_id: str = "",
    mode: str = "dark",
) -> List[Path]:
    """Write the standard run plot bundle to ``<out_dir>/plots/``.

    Args:
        metrics_history: Sequence of per-epoch metric records.
        out_dir: Run directory; plots land in ``out_dir / "plots"``.
        scenario_id: Optional title suffix.
        mode: ``"dark"`` or ``"light"`` theme.

    Returns:
        Paths of PNGs written (empty list if matplotlib isn't installed
        or there are no metrics).
    """
    metrics = list(metrics_history)
    if not metrics:
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    from swarm.analysis.timeseries import (
        plot_selection_geometry,
        plot_toxicity_welfare,
    )

    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    epochs = _extract_series(metrics, "epoch")
    suffix = f" — {scenario_id}" if scenario_id else ""

    tw_data = {
        "epochs": epochs,
        "toxicity": _extract_series(metrics, "toxicity_rate"),
        "welfare": _extract_series(metrics, "total_welfare"),
    }
    fig_tw, _ = plot_toxicity_welfare(
        tw_data, title=f"Toxicity & Welfare{suffix}", mode=mode,
    )
    path_tw = plots_dir / "toxicity_welfare.png"
    fig_tw.savefig(path_tw, dpi=120, bbox_inches="tight")
    plt.close(fig_tw)
    written.append(path_tw)

    sg_data = {
        "epochs": epochs,
        "selection_saturation": _extract_series(metrics, "selection_saturation"),
        "baseline_harm": _extract_series(metrics, "baseline_harm"),
        "selection_credit": _extract_series(metrics, "selection_credit"),
    }
    fig_sg, _ = plot_selection_geometry(
        sg_data, title=f"Selection Geometry{suffix}", mode=mode,
    )
    path_sg = plots_dir / "selection_geometry.png"
    fig_sg.savefig(path_sg, dpi=120, bbox_inches="tight")
    plt.close(fig_sg)
    written.append(path_sg)

    return written
