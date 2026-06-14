"""Command line interface for comparing two SWARM run JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swarm.replay.diff import (
    DEFAULT_METRICS,
    compare_run_files,
    format_markdown_table,
    rows_to_dicts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the run-diff argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m swarm.replay.run_diff",
        description="Compare two SWARM run JSON artifacts as a Markdown table.",
    )
    parser.add_argument("run_a", type=Path, help="Baseline run JSON file")
    parser.add_argument("run_b", type=Path, help="Comparison run JSON file")
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


if __name__ == "__main__":
    raise SystemExit(main())
