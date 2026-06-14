"""Compare exported SWARM run metrics side by side."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_METRICS: tuple[str, ...] = (
    "acceptance_rate",
    "toxicity_rate",
    "quality_gap",
    "total_welfare",
    "net_social_welfare",
    "avg_payoff",
)

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "acceptance_rate": ("acceptance_rate", "accepted_rate"),
    "toxicity_rate": ("toxicity_rate", "avg_toxicity"),
    "quality_gap": ("quality_gap", "avg_quality_gap"),
    "total_welfare": ("total_welfare", "welfare_total"),
    "net_social_welfare": ("net_social_welfare", "avg_net_welfare"),
    "avg_payoff": ("avg_payoff", "average_payoff", "mean_payoff"),
}


@dataclass(frozen=True)
class MetricDelta:
    """One metric comparison row."""

    metric: str
    run_a: float | None
    run_b: float | None

    @property
    def delta(self) -> float | None:
        """Return run_b - run_a when both values are present."""
        if self.run_a is None or self.run_b is None:
            return None
        return self.run_b - self.run_a


def load_run_metrics(path: str | Path) -> dict[str, float]:
    """Load and normalize metrics from a SWARM JSON run artifact.

    Supported inputs include a raw list of epoch metric dictionaries, a dict with
    ``metrics_history``/``epochs``/``epoch_metrics``, a dict with ``final_metrics``,
    and flat summary dictionaries.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return normalize_run_metrics(payload)


def normalize_run_metrics(payload: Any) -> dict[str, float]:
    """Normalize common SWARM run artifact shapes into comparable metrics."""
    epochs = _find_epoch_metrics(payload)
    if epochs:
        return _aggregate_epochs(epochs)

    if isinstance(payload, Mapping):
        final_metrics = payload.get("final_metrics")
        if isinstance(final_metrics, Mapping):
            return _normalize_flat(final_metrics)
        summary = payload.get("summary")
        if isinstance(summary, Mapping):
            return _normalize_flat(summary)
        return _normalize_flat(payload)

    raise ValueError("run artifact must be a JSON object or list of epoch metrics")


def compare_run_metrics(
    run_a: Mapping[str, float],
    run_b: Mapping[str, float],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[MetricDelta]:
    """Compare selected metrics with a deterministic row order."""
    return [
        MetricDelta(metric=metric, run_a=run_a.get(metric), run_b=run_b.get(metric))
        for metric in metrics
    ]


def compare_run_files(
    run_a_path: str | Path,
    run_b_path: str | Path,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[MetricDelta]:
    """Load two run artifacts and compare selected metrics."""
    return compare_run_metrics(
        load_run_metrics(run_a_path),
        load_run_metrics(run_b_path),
        metrics=metrics,
    )


def format_markdown_table(rows: Iterable[MetricDelta], precision: int = 4) -> str:
    """Render comparison rows as a Markdown table."""
    lines = [
        "| Metric | Delta | Run A | Run B |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {delta} | {run_a} | {run_b} |".format(
                metric=row.metric,
                delta=_format_value(row.delta, precision=precision, show_sign=True),
                run_a=_format_value(row.run_a, precision=precision),
                run_b=_format_value(row.run_b, precision=precision),
            )
        )
    return "\n".join(lines)


def rows_to_dicts(rows: Iterable[MetricDelta]) -> list[dict[str, float | None | str]]:
    """Convert comparison rows into JSON-serializable dictionaries."""
    return [
        {
            "metric": row.metric,
            "run_a": row.run_a,
            "run_b": row.run_b,
            "delta": row.delta,
        }
        for row in rows
    ]


def _find_epoch_metrics(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []

    for key in ("metrics_history", "epoch_metrics", "epochs"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]

    history = payload.get("history")
    if isinstance(history, Mapping):
        for key in ("epochs", "epoch_metrics", "metrics_history"):
            value = history.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]

    return []


def _aggregate_epochs(epochs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not epochs:
        return {}

    totals = {
        "total_interactions": _sum_metric(epochs, "total_interactions"),
        "accepted_interactions": _sum_metric(epochs, "accepted_interactions"),
        "total_welfare": _sum_metric(epochs, "total_welfare"),
        "net_social_welfare": _sum_metric(epochs, "net_social_welfare"),
    }
    averaged = {
        "toxicity_rate": _mean_metric(epochs, "toxicity_rate"),
        "quality_gap": _mean_metric(epochs, "quality_gap"),
        "avg_payoff": _mean_metric(epochs, "avg_payoff"),
    }
    if averaged["avg_payoff"] is None:
        averaged["avg_payoff"] = _mean_metric(epochs, "average_payoff")

    result = {
        key: value for key, value in {**totals, **averaged}.items() if value is not None
    }
    if totals["total_interactions"]:
        result["acceptance_rate"] = (
            (totals["accepted_interactions"] or 0.0) / totals["total_interactions"]
        )
    return result


def _normalize_flat(payload: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for canonical, aliases in _METRIC_ALIASES.items():
        value = _first_number(payload, aliases)
        if value is not None:
            result[canonical] = value

    total_interactions = _first_number(payload, ("total_interactions",))
    accepted_interactions = _first_number(payload, ("accepted_interactions",))
    if "acceptance_rate" not in result and total_interactions:
        result["acceptance_rate"] = (accepted_interactions or 0.0) / total_interactions
    return result


def _sum_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _first_number(payload: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _as_float(payload.get(key))
        if value is not None:
            return value
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_value(
    value: float | None,
    precision: int,
    show_sign: bool = False,
) -> str:
    if value is None:
        return "n/a"
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{precision}f}"
