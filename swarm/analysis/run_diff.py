"""Compare two SWARM run directories side by side.

The utility accepts run directories containing ``history.json`` or CSV exports
and prints a compact table of key outcome metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_METRICS: tuple[str, ...] = (
    "toxicity_rate",
    "quality_gap",
    "illusion_delta",
    "avg_payoff",
)

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "acceptance_rate": ("acceptance_rate", "accepted_rate"),
    "toxicity_rate": ("toxicity_rate", "avg_toxicity", "toxicity"),
    "quality_gap": ("quality_gap", "avg_quality_gap"),
    "illusion_delta": ("illusion_delta", "avg_illusion_delta"),
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
        """Return run B minus run A when both values are present."""
        if self.run_a is None or self.run_b is None:
            return None
        return self.run_b - self.run_a


def load_run_metrics(path: str | Path) -> dict[str, float]:
    """Load normalized metrics from a run directory, JSON file, or CSV file."""
    resolved = Path(path)
    if resolved.is_dir():
        return _load_run_directory(resolved)
    if resolved.suffix.lower() == ".csv":
        return _load_epoch_csv(resolved)
    if resolved.suffix.lower() == ".json" or resolved.name == "history.json":
        return _load_json_file(resolved)
    raise ValueError(
        f"{resolved} is not a run directory, history.json file, or epoch CSV export"
    )


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
    """Load two runs and compare selected metrics."""
    return compare_run_metrics(
        load_run_metrics(run_a_path),
        load_run_metrics(run_b_path),
        metrics=metrics,
    )


def format_markdown_table(rows: Iterable[MetricDelta], precision: int = 4) -> str:
    """Render comparison rows as a Markdown table."""
    lines = [
        "| Metric | Run A | Run B | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {run_a} | {run_b} | {delta} |".format(
                metric=row.metric,
                run_a=_format_value(row.run_a, precision=precision),
                run_b=_format_value(row.run_b, precision=precision),
                delta=_format_value(row.delta, precision=precision, show_sign=True),
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


def build_parser() -> argparse.ArgumentParser:
    """Build the run-diff argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m swarm.analysis.run_diff",
        description=(
            "Compare two SWARM run directories containing history.json or CSV "
            "exports."
        ),
    )
    parser.add_argument("run_a", type=Path, help="Baseline run directory or export")
    parser.add_argument("run_b", type=Path, help="Comparison run directory or export")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names to compare",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places in Markdown output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of Markdown",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]

    try:
        rows = compare_run_files(args.run_a, args.run_b, metrics=metrics)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(rows_to_dicts(rows), indent=2, sort_keys=True))
    else:
        print(format_markdown_table(rows, precision=args.precision))
    return 0


def _load_run_directory(run_dir: Path) -> dict[str, float]:
    history_path = run_dir / "history.json"
    if history_path.exists():
        return _load_json_file(history_path)

    csv_path = _find_epoch_csv(run_dir)
    if csv_path is not None:
        return _load_epoch_csv(csv_path)

    raise FileNotFoundError(
        f"No history.json or epoch CSV export found in {run_dir}"
    )


def _load_json_file(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return normalize_run_metrics(payload)


def _find_epoch_csv(run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    csv_dir = run_dir / "csv"
    if csv_dir.exists():
        candidates.extend(sorted(csv_dir.glob("*_epochs.csv")))
        candidates.extend(sorted(csv_dir.glob("epochs.csv")))
        candidates.extend(sorted(csv_dir.glob("*.csv")))
    candidates.extend(sorted(run_dir.glob("*_epochs.csv")))
    candidates.extend(sorted(run_dir.glob("epochs.csv")))
    candidates.extend(sorted(run_dir.glob("*.csv")))
    return candidates[0] if candidates else None


def _load_epoch_csv(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return _aggregate_epochs(rows)


def _find_epoch_metrics(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []

    for key in ("metrics_history", "epoch_metrics", "epochs", "epoch_snapshots"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]

    history = payload.get("history")
    if isinstance(history, Mapping):
        for key in ("epochs", "epoch_metrics", "metrics_history", "epoch_snapshots"):
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
        "illusion_delta": _mean_metric(epochs, "illusion_delta"),
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
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _format_value(
    value: float | None,
    precision: int,
    show_sign: bool = False,
) -> str:
    if value is None:
        return "N/A"
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{precision}f}"


if __name__ == "__main__":
    raise SystemExit(main())
