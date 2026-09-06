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
import random
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
