"""Certainty-band audit: how much loss sits at near-certain p (bead l1d6).

The AI Village loan loss came from a belief held at confidence 1.0. SWARM's
proxy clamps v_hat to [-1, +1] and maps it through sigmoid(k * v_hat), so
whether p can get near 0 or 1 at all is set by the sharpness k. At the default
k = 2, p is confined to [0.119, 0.881]. It reaches exactly 1.0 in float once
k >= 37.

This sweeps k on escalating_pressure, the scenario that records ground truth
per interaction (fabricated positives are -1), and reports the share of
realized loss carried by interactions in the extreme bands.

Only k changes between cells. It is set on the orchestrator's shared
ProxyComputer after the build, the same instance the interaction finalizer
uses (the mechanism swarm/core/dynamic_toxicity.py relies on).

Usage:
    python -m experiments.certainty_band_audit [--quick] [--output runs/]
"""

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path

from swarm.metrics.soft_metrics import SoftMetrics
from swarm.scenarios.loader import build_orchestrator, load_scenario

SCENARIO = "scenarios/escalating_pressure.yaml"
KS = [2.0, 5.0, 10.0, 40.0]
SEEDS = [42, 7, 101]
BAND = 0.05


def run_cell(k: float, seed: int) -> dict:
    scenario = load_scenario(SCENARIO)
    scenario.orchestrator_config.seed = seed
    orch = build_orchestrator(scenario)
    orch.proxy_computer.sigmoid_k = k

    rec = []
    orch.on_interaction_complete(lambda i, _a, _b: rec.append(i))
    orch.run()

    metrics = SoftMetrics()
    d = metrics.extreme_p_loss_share(rec, band=BAND)
    accepted = [i for i in rec if i.accepted]
    certified = [i for i in accepted if i.ground_truth is not None]
    return {
        "k": k,
        "seed": seed,
        "n_interactions": len(rec),
        "n_accepted": len(accepted),
        "n_certified_accepted": len(certified),
        "p_max": max((i.p for i in rec), default=0.0),
        "p_min": min((i.p for i in rec), default=0.0),
        "frac_high_band": d["n_high"] / len(certified) if d and certified else 0.0,
        "saturated": d["saturated"] if d else 0,
        "bad_share_certified": (
            sum(1 for i in certified if i.ground_truth == -1) / len(certified)
            if certified
            else 0.0
        ),
        "proxy_toxicity": metrics.toxicity_rate(rec),
        "loss_total": d["loss_total"] if d else 0.0,
        "share_high": d["share_high"] if d else None,
        "share_low": d["share_low"] if d else None,
    }


def mean(rows: list, key: str) -> float:
    vals = [r[key] for r in rows if r[key] is not None]
    return statistics.fmean(vals) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="runs/")
    ap.add_argument("--quick", action="store_true", help="one seed, k in {2, 40}")
    args = ap.parse_args()

    ks = [2.0, 40.0] if args.quick else KS
    seeds = SEEDS[:1] if args.quick else SEEDS

    rows = [run_cell(k, s) for k in ks for s in seeds]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output) / f"{ts}_certainty_band_audit"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"escalating_pressure, band={BAND}, seeds={seeds}")
    print(
        f"{'k':>5} {'p range':>17} {'acc':>6} {'bad share':>10} {'proxy tox':>10} "
        f"{'in high':>8} {'sat':>5} {'loss':>7} {'share hi':>9} {'share lo':>9}"
    )
    for k in ks:
        cell = [r for r in rows if r["k"] == k]
        print(
            f"{k:5.0f} [{min(r['p_min'] for r in cell):.3f}, {max(r['p_max'] for r in cell):.3f}] "
            f"{mean(cell, 'n_accepted'):6.0f} {mean(cell, 'bad_share_certified'):10.3f} "
            f"{mean(cell, 'proxy_toxicity'):10.3f} {mean(cell, 'frac_high_band'):8.3f} "
            f"{mean(cell, 'saturated'):5.0f} {mean(cell, 'loss_total'):7.0f} "
            f"{mean(cell, 'share_high'):9.3f} {mean(cell, 'share_low'):9.3f}"
        )
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
