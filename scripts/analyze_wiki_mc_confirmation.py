#!/usr/bin/env python3
"""Analyze paired wiki Monte Carlo outputs with a sign-flip/Holm correction.

The unit is a seed-level treatment-minus-control difference. This is a
screening analysis for the synthetic model, not a test of historical claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def p_value(differences: list[float], draws: int = 20_000) -> float:
    """Two-sided Monte Carlo sign-flip p-value for a paired mean difference."""
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0
    observed = abs(sum(differences) / len(differences))
    token = repr([round(value, 12) for value in differences]).encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        signed = sum(value if rng.getrandbits(1) else -value for value in differences)
        if abs(signed / len(differences)) >= observed:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def holm(rows: list[dict]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value"])
    m = len(rows)
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * row["p_value"])
        running = max(running, adjusted)
        rows[index]["holm_p"] = running


def load_differences(directory: Path, cell_id: str, metric: str) -> list[float]:
    values = []
    for path in sorted(directory.glob(f"{cell_id}-seed-*.json")):
        payload = json.loads(path.read_text())
        values.append(float(payload["treatment_metrics"][metric]) -
                      float(payload["control_metrics"][metric]))
    if len(values) != 200:
        raise ValueError(f"{cell_id}/{metric}: expected 200 seeds, found {len(values)}")
    return values


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for an event proportion across independent runs."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (center - half, center + half)


def summarize(directory: Path) -> list[dict]:
    """Per-cell run-level summary of outcomes the paired CSV does not carry.

    Displacement is reported against two denominators: works disrupted by a
    moderation action (``response`` events) and works that chose to relocate.
    Alarm rates are run-level proportions with Wilson intervals; in the
    detection family the treatment run is the untreated run, so treatment and
    control are identical and only the treatment side is reported.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    cells = {cell["id"]: cell for cell in manifest["cells"]}
    rows = []
    for cell_id, cell in cells.items():
        runs = [json.loads(path.read_text())
                for path in sorted(directory.glob(f"{cell_id}-seed-*.json"))]
        if not runs:
            continue
        metrics: dict[str, list[float]] = defaultdict(list)
        counts = defaultdict(int)
        for run in runs:
            for key, value in run["treatment_metrics"].items():
                metrics[key].append(float(value))
            for key, value in run["control_metrics"].items():
                metrics["control_" + key].append(float(value))
            events = run["treatment"]["events"]
            for event in events:
                if event["type"] == "response":
                    counts["disrupted"] += 1
                    counts["relocated"] += event["action"] == "relocate"
                elif event["type"] == "displacement":
                    counts["displaced"] += 1
                elif event["type"] == "moderation":
                    counts["evasion_learned"] += event.get("evasion_learned", 0)
        n = len(runs)
        alarms = int(sum(metrics["screen_alarm"]))
        control_alarms = int(sum(metrics["control_screen_alarm"]))
        low, high = wilson(alarms, n)
        row = {"cell_id": cell_id, "family": cell["family"], "n": n,
               "overrides": json.dumps(cell["overrides"], sort_keys=True),
               "coverage": cell["observation_fraction"]}
        for key in ("completion_rate", "task_success_rate", "shared_submission_rate",
                    "total_writes", "post_intervention_writes", "displacements",
                    "removed_pages", "useful_reads", "screen_agreement",
                    "screen_comparable_pairs"):
            row[key + "_mean"] = statistics.fmean(metrics[key])
            row["control_" + key + "_mean"] = statistics.fmean(metrics["control_" + key])
        row.update(alarm_runs=alarms, alarm_rate=alarms / n, alarm_wilson_low=low,
                   alarm_wilson_high=high, control_alarm_runs=control_alarms,
                   zero_write_control_runs=int(sum(v == 0 for v in metrics["control_total_writes"])),
                   disrupted_works=counts["disrupted"], relocated_works=counts["relocated"],
                   displaced_works=counts["displaced"], evasion_learned=counts["evasion_learned"],
                   displaced_per_disrupted=(counts["displaced"] / counts["disrupted"]
                                            if counts["disrupted"] else None),
                   displaced_per_relocated=(counts["displaced"] / counts["relocated"]
                                            if counts["relocated"] else None))
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", action="store_true",
                        help="Write a per-cell run-level summary (any family, any seed "
                             "count) instead of the 200-seed moderation Holm analysis")
    args = parser.parse_args()
    if args.summary:
        rows = summarize(args.input)
        args.output.write_text(json.dumps({"input": str(args.input), "cells": rows}, indent=2) + "\n")
        with args.output.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return 0
    tests = []
    cells = sorted({path.name.split("-seed-")[0]
                    for path in args.input.glob("moderation-*-seed-*.json")})
    for cell_id in cells:
        for metric in ("completion_rate", "total_writes"):
            differences = load_differences(args.input, cell_id, metric)
            tests.append({"cell_id": cell_id, "metric": metric,
                          "mean_difference": sum(differences) / len(differences),
                          "p_value": p_value(differences), "n": len(differences),
                          "zero_fraction": sum(value == 0 for value in differences) / len(differences)})
    holm(tests)
    args.output.write_text(json.dumps({
        "method": "paired two-sided Monte Carlo sign-flip test; 20,000 draws per test",
        "family": "10 moderation cells x 2 prespecified outcomes",
        "correction": "Holm step-down across 20 tests",
        "tests": tests,
    }, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tests[0]))
        writer.writeheader()
        writer.writerows(tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
