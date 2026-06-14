"""Compatibility wrapper for run metric diffs.

The issue-requested implementation lives in :mod:`swarm.analysis.run_diff`.
"""

from __future__ import annotations

from swarm.analysis.run_diff import (
    DEFAULT_METRICS,
    MetricDelta,
    compare_run_files,
    compare_run_metrics,
    format_markdown_table,
    load_run_metrics,
    normalize_run_metrics,
    rows_to_dicts,
)

__all__ = [
    "DEFAULT_METRICS",
    "MetricDelta",
    "compare_run_files",
    "compare_run_metrics",
    "format_markdown_table",
    "load_run_metrics",
    "normalize_run_metrics",
    "rows_to_dicts",
]
