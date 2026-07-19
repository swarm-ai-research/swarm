"""Frame sequencing, GIF/MP4 export, and epoch replay for SWARM.

Usage::

    from swarm.analysis.animate import animate_timeseries, save_animation

    path = animate_timeseries(epochs_data, "toxicity", output_path="toxicity.gif")
    frames = render_epoch_frames(my_plot_func, epoch_data)
    save_animation(frames, "custom.gif", fps=10)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    swarm_theme,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(
    plot_func: Callable[[], Any],
    *,
    mode: str = "dark",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 100,
) -> Any:
    """Render a single frame by calling *plot_func*, capturing as PIL Image.

    *plot_func* takes no args and returns ``(fig, ax)`` or ``fig``.
    PIL is imported lazily; raises ``ImportError`` if Pillow is absent.

    Returns:
        PIL.Image.Image object containing the rendered frame.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for frame rendering. "
            "Install it with: pip install Pillow"
        ) from exc

    with swarm_theme(mode):
        result = plot_func()
        fig = result[0] if isinstance(result, tuple) else result
        fig.set_size_inches(figsize)
        fig.set_dpi(dpi)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf).copy()
        buf.close()

    return image


def render_epoch_frames(
    plot_func: Callable[[int, Any], Any],
    epoch_data: List[Any],
    *,
    mode: str = "dark",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 100,
) -> List[Any]:
    """Render one frame per epoch.

    *plot_func* takes ``(epoch_index, epoch_datum)`` and returns
    ``(fig, ax)`` or ``fig``.  Returns list of ``PIL.Image.Image``.
    """
    frames: List[Any] = []
    for i, datum in enumerate(epoch_data):
        frame = render_frame(
            lambda _i=i, _d=datum: plot_func(_i, _d),  # type: ignore[misc]
            mode=mode, figsize=figsize, dpi=dpi,
        )
        frames.append(frame)
        logger.debug("Rendered frame %d / %d", i + 1, len(epoch_data))
    logger.info("Rendered %d epoch frames", len(frames))
    return frames


# ---------------------------------------------------------------------------
# Animation export
# ---------------------------------------------------------------------------

def save_animation(
    frames: List[Any],
    path: Any,
    *,
    fps: int = 5,
    loop: int = 0,
    format: Optional[str] = None,
) -> Path:
    """Save PIL Image frames as GIF or MP4.

    *format* is inferred from the file extension when *None*.
    For GIF, uses Pillow ``save`` with ``append_images``.
    For MP4, tries matplotlib ``FFMpegWriter``, then ``imageio``.
    """
    if not frames:
        raise ValueError("No frames to save")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".gif":
            format = "gif"
        elif suffix in (".mp4", ".avi", ".mov"):
            format = "mp4"
        else:
            raise ValueError(
                f"Cannot infer format from extension '{suffix}'. "
                f"Use format='gif' or format='mp4'."
            )

    if format == "gif":
        _save_gif(frames, path, fps=fps, loop=loop)
    elif format == "mp4":
        _save_mp4(frames, path, fps=fps)
    else:
        raise ValueError(f"Unsupported animation format: {format!r}")

    logger.info("Saved animation (%d frames) to %s", len(frames), path)
    return Path(path)


def _save_gif(frames: List[Any], path: Path, *, fps: int, loop: int) -> None:
    """Save frames as an animated GIF using Pillow."""
    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=loop, optimize=True,
    )


def _save_mp4(frames: List[Any], path: Path, *, fps: int) -> None:
    """Save frames as MP4.  FFMpegWriter -> imageio.v3 -> imageio legacy."""
    # Strategy 1: matplotlib FFMpegWriter
    try:
        from matplotlib.animation import FFMpegWriter, FuncAnimation

        fig, ax = plt.subplots()
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        im_display = ax.imshow(np.array(frames[0]))

        def _update(idx: int) -> tuple:
            im_display.set_data(np.array(frames[idx]))
            return (im_display,)

        anim = FuncAnimation(
            fig, _update, frames=len(frames),
            interval=int(1000 / fps), blit=True,
        )
        anim.save(str(path), writer=FFMpegWriter(fps=fps))
        plt.close(fig)
        return
    except Exception as exc:
        logger.debug("FFMpegWriter unavailable, trying imageio: %s", exc)

    # Strategy 2: imageio v3
    try:
        import imageio.v3 as iio
        iio.imwrite(str(path), [np.array(f) for f in frames], fps=fps)
        return
    except ImportError:
        pass

    # Strategy 3: imageio legacy
    try:
        import imageio
        writer = imageio.get_writer(str(path), fps=fps)
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()
        return
    except ImportError as err:
        raise ImportError(
            "MP4 export requires ffmpeg (for matplotlib) or imageio. "
            "Install with: pip install imageio[ffmpeg]"
        ) from err


# ---------------------------------------------------------------------------
# Pre-built animations
# ---------------------------------------------------------------------------

def animate_timeseries(
    epochs_data: List[Dict[str, Any]],
    metric: str,
    *,
    title: Optional[str] = None,
    mode: str = "dark",
    output_path: Optional[Any] = None,
    fps: int = 5,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 100,
) -> Any:
    """Animate a time-series line growing epoch-by-epoch.

    Each frame shows the metric line from epoch 0 up to the current epoch.
    Returns ``Path`` if *output_path* is given, else a list of PIL frames.
    """
    from swarm.analysis.theme import metric_color

    if not epochs_data:
        raise ValueError("epochs_data must not be empty")

    epochs = [d.get("epoch", i) for i, d in enumerate(epochs_data)]
    values = [d.get(metric, 0.0) for d in epochs_data]
    color = metric_color(metric)
    title = title or metric.replace("_", " ").title()
    y_min, y_max = min(values), max(values)
    y_pad = max(0.05 * (y_max - y_min), 0.01)

    def _make_frame(frame_idx: int, _datum: Any) -> Tuple[Any, Any]:
        fig, ax = plt.subplots(figsize=figsize)
        end = frame_idx + 1
        ax.plot(epochs[:end], values[:end], color=color,
                linewidth=2.0, marker="o", markersize=4)
        ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{title}  (epoch {epochs[frame_idx]})")
        fig.tight_layout()
        return fig, ax

    frames = render_epoch_frames(
        _make_frame, list(range(len(epochs_data))),
        mode=mode, figsize=figsize, dpi=dpi,
    )
    if output_path is None:
        return frames
    return save_animation(frames, output_path, fps=fps)


def animate_from_history(
    history_dict: Dict[str, Any],
    metrics: Sequence[str],
    *,
    output_dir: Any = ".",
    fps: int = 5,
    mode: str = "dark",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 100,
) -> Dict[str, Path]:
    """Generate one animation per metric from a ``history.json`` dict.

    Expects *history_dict* to have an ``"epochs"`` key containing a list
    of epoch records.  Returns dict mapping metric name to output ``Path``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs_data = history_dict.get("epochs", [])
    if not epochs_data:
        logger.warning("No epoch data found in history_dict")
        return {}

    results: Dict[str, Path] = {}
    for name in metrics:
        if not any(name in ep for ep in epochs_data):
            logger.warning("Metric '%s' not found in epoch data, skipping", name)
            continue
        out_path = output_dir / f"{name}.gif"
        try:
            results[name] = animate_timeseries(
                epochs_data, name, output_path=out_path,
                fps=fps, mode=mode, figsize=figsize, dpi=dpi,
            )
            logger.info("Created animation for '%s' at %s", name, results[name])
        except Exception:
            logger.exception("Failed to animate metric '%s'", name)
    return results


# ---------------------------------------------------------------------------
# Epoch scrubber data (for interactive frontends)
# ---------------------------------------------------------------------------

def create_epoch_scrubber_data(
    history_dict: Dict[str, Any],
    metrics: Sequence[str],
) -> Dict[str, Any]:
    """Prepare JSON-serializable data for a frontend epoch scrubber.

    Returns dict with ``"epochs"`` (list of ints), ``"metrics"`` (dict of
    metric name to list of values, ``None`` for missing), and
    ``"num_epochs"``.
    """
    epochs_data = history_dict.get("epochs", [])
    epoch_indices: List[int] = []
    metric_series: Dict[str, List[Optional[float]]] = {m: [] for m in metrics}

    for i, record in enumerate(epochs_data):
        epoch_indices.append(record.get("epoch", i))
        for m in metrics:
            value = record.get(m)
            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            metric_series[m].append(value)

    return {
        "epochs": epoch_indices,
        "metrics": metric_series,
        "num_epochs": len(epoch_indices),
    }
