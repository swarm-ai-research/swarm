#!/usr/bin/env python3
"""Paired Monte Carlo experiments for the explicit wiki agent model.

Each cell/seed has its own untreated counterfactual. Intervals resample whole
paired runs, never events. Defaults are a 30-seed pilot; confirmation uses 200
fresh seeds from 10000 and requires an explicit frozen detector threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def experiment_cells(family: str) -> list[dict[str, Any]]:
    """Small, explicit factorial grids; do not interpret them as calibration."""
    cells: list[dict[str, Any]] = []
    if family in ("emergence", "all"):
        for overlap, deadline, referrals in itertools.product(
            (0.2, 0.8), (6.0, 18.0), (False, True)
        ):
            cells.append({"family": "emergence", "observation_fraction": 1.0,
                          "overrides": {"task_overlap": overlap, "deadline": deadline,
                                        "referrals_enabled": referrals}})
    if family in ("moderation", "all"):
        for policy, relocation in itertools.product(
            ("none", "ordered", "random", "lock", "global_lock"),
            ("endogenous", "forced"),
        ):
            cells.append({"family": "moderation", "observation_fraction": 1.0,
                          "overrides": {"moderation_policy": policy,
                                        "relocation_mode": relocation}})
    if family in ("detection", "all"):
        for regime, coverage in itertools.product(
            ("independent", "authorized", "prohibited"), (0.25, 1.0)
        ):
            cells.append({"family": "detection", "observation_fraction": coverage,
                          "overrides": {"sharing_regime": regime}})
    for index, cell in enumerate(cells):
        cell["id"] = f"{cell['family']}-{index:03d}"
    return cells


def stream_seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")


def paired_interval(values: list[float], seed: int, draws: int = 2000) -> dict[str, Any]:
    """Percentile bootstrap of independent runs; n=1 has no uncertainty estimate."""
    if not values:
        raise ValueError("At least one paired run is required")
    result: dict[str, Any] = {"n": len(values), "mean": statistics.fmean(values),
                              "ci_low": None, "ci_high": None}
    if len(values) > 1:
        rng = random.Random(seed)
        means = sorted(statistics.fmean(rng.choices(values, k=len(values)))
                       for _ in range(draws))
        result.update(ci_low=means[int(0.025 * (draws - 1))],
                      ci_high=means[int(0.975 * (draws - 1))])
    return result


def screen_events(events: list[dict[str, Any]], coverage: float, seed: int,
                  threshold: float) -> dict[str, float]:
    """Illustrative output agreement screen; cannot determine authorization.

    Coverage independently thins submissions. Submission outputs, not hidden
    lineage, are used; this observer assumes access to submitted outputs.
    """
    rng = random.Random(stream_seed(seed, "observation"))
    submissions = [event for event in events
                   if event.get("type") == "submission" and "answer" in event]
    visible = [event for event in submissions if rng.random() < coverage]
    comparable = agreements = 0
    for left, right in itertools.combinations(visible, 2):
        if left.get("task_id") == right.get("task_id"):
            comparable += 1
            agreements += left.get("answer") == right.get("answer")
    score = agreements / comparable if comparable else 0.0
    return {"screen_agreement": score, "screen_alarm": float(comparable > 0 and score >= threshold),
            "screen_comparable_pairs": float(comparable),
            "screen_observed_submissions": float(len(visible))}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--family", choices=("emergence", "moderation", "detection", "all"), default="all")
    result.add_argument("--phase", choices=("pilot", "confirmation"), default="pilot")
    result.add_argument("--seeds", type=int, help="Number of independent seeds")
    result.add_argument("--seed-start", type=int)
    result.add_argument("--max-cells", type=int)
    result.add_argument("--smoke", action="store_true", help="Two seeds and one cell per family")
    result.add_argument("--config", type=Path, help="JSON object of SimulationConfig overrides")
    result.add_argument("--detector-threshold", type=float, help="Explicit frozen agreement screening threshold")
    result.add_argument("--output", type=Path, required=True, help="New artifact directory (must not exist)")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    threshold = arguments.detector_threshold
    if arguments.phase == "confirmation" and threshold is None:
        raise SystemExit("Confirmation requires --detector-threshold frozen before evaluating seeds")
    threshold = 0.8 if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise SystemExit("Detector threshold must lie in [0, 1]")
    count = arguments.seeds if arguments.seeds is not None else (30 if arguments.phase == "pilot" else 200)
    start = arguments.seed_start if arguments.seed_start is not None else (0 if arguments.phase == "pilot" else 10000)
    if arguments.smoke:
        count = arguments.seeds if arguments.seeds is not None else 2
    if count < 1 or start < 0 or (arguments.max_cells is not None and arguments.max_cells < 1):
        raise SystemExit("Seeds/max-cells must be positive; seed-start must be nonnegative")
    if (arguments.phase == "pilot" and start + count > 10000) or (arguments.phase == "confirmation" and start < 10000):
        raise SystemExit("Pilot seeds must be below 10000; confirmation seeds must be at least 10000")
    cells = experiment_cells(arguments.family)
    if arguments.smoke:
        seen: set[str] = set()
        cells = [cell for cell in cells if cell["family"] not in seen and not seen.add(cell["family"])]
    if arguments.max_cells:
        cells = cells[:arguments.max_cells]
    base = json.loads(arguments.config.read_text()) if arguments.config else {}
    if not isinstance(base, dict):
        raise SystemExit("Config must be a JSON object")

    from swarm.bridges.wiki_sim import SimulationConfig, simulate

    # Validate all configurations before creating an output directory.
    configs = [(cell, SimulationConfig(**(base | cell["overrides"]))) for cell in cells]
    arguments.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "phase": arguments.phase, "seeds": list(range(start, start + count)),
                "cells": cells, "base_config": base, "smoke": arguments.smoke,
                "detector": {"threshold": threshold, "status": "explicit_frozen" if arguments.detector_threshold is not None else "pilot_default_not_tuned",
                             "meaning": "Output agreement screening, not authorization or collusion inference",
                             "coverage": "Independent submission observation probability"},
                "interval": "95% percentile bootstrap across paired seed runs, 2000 draws; exploratory, unadjusted for multiple comparisons",
                "pairing": "Same seed and config except moderation_policy=none; model random streams provide common exogenous randomness"}
    (arguments.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    began = time.perf_counter()
    for cell, config in configs:
        paired: dict[str, list[float]] = {}
        treated_values: dict[str, list[float]] = {}
        control_values: dict[str, list[float]] = {}
        for seed in range(start, start + count):
            pair_start = time.perf_counter()
            treated = simulate(config, seed)
            control = simulate(replace(config, moderation_policy="none"), seed)
            treatment_metrics = dict(treated.metrics)
            control_metrics = dict(control.metrics)
            for result, metrics in ((treated, treatment_metrics), (control, control_metrics)):
                metrics.update(screen_events(result.events, cell["observation_fraction"], seed, threshold))
            for metric in sorted(treatment_metrics.keys() & control_metrics.keys()):
                paired.setdefault(metric, []).append(treatment_metrics[metric] - control_metrics[metric])
                treated_values.setdefault(metric, []).append(treatment_metrics[metric])
                control_values.setdefault(metric, []).append(control_metrics[metric])
            payload = {"cell_id": cell["id"], "seed": seed,
                       "treatment": treated.to_dict(), "control": control.to_dict(),
                       "treatment_metrics": treatment_metrics, "control_metrics": control_metrics,
                       "elapsed_seconds": time.perf_counter() - pair_start}
            (arguments.output / f"{cell['id']}-seed-{seed}.json").write_text(json.dumps(payload, indent=2) + "\n")
        for metric, differences in paired.items():
            interval = paired_interval(differences, stream_seed(0, f"{cell['id']}:{metric}"))
            rows.append({"cell_id": cell["id"], "family": cell["family"], "metric": metric,
                         "treatment_mean": statistics.fmean(treated_values[metric]),
                         "control_mean": statistics.fmean(control_values[metric]),
                         "paired_difference_mean": interval["mean"], "n": interval["n"],
                         "ci_low": interval["ci_low"], "ci_high": interval["ci_high"]})
    with (arguments.output / "paired_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest["elapsed_seconds"] = time.perf_counter() - began
    manifest["completed_pairs"] = len(cells) * count
    (arguments.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "pairs": manifest["completed_pairs"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
