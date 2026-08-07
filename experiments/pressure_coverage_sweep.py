"""Certificate-coverage dose-response sweep on escalating_pressure (beads mfya).

Axes (design recorded on the bead + docs/research/dgg-counterexample-lessons.md):
  - certificate coverage: gate off, then 0.15 / 0.3 / 0.6 / 0.9
  - acceptance pressure: default honest counterparties (known selection ~ 0)
    vs cautious counterparties (threshold on expected p / trust) — the axis
    the mt8a decomposition showed is required for the adverse-selection leg
    to exist at all
  - 5 seeds per cell

Per-cell outcomes (late phase = final third of epochs, full pressure):
  - fab_share:   fabricated-accepted share among certified accepted
  - overtrust:   E[p * 1{gt=-1} | accepted, certified] — zero-baselined
                 fabrication indicator
  - pcg_selection: adverse-selection leg of PCG
  - quality_gap: E[p|accepted] - E[p|rejected] (sign is the prediction)
  - worker_accept_rate: realized acceptance pressure on workers

Usage:
    python -m experiments.pressure_coverage_sweep [--quick]

Writes runs/<ts>_pressure_coverage_sweep/results.csv and prints a summary
table. --quick runs 2 seeds and 3 coverage levels for smoke purposes.
"""

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path

from swarm.metrics.soft_metrics import SoftMetrics
from swarm.scenarios.loader import build_orchestrator, load_scenario

SCENARIO = "scenarios/baseline.yaml"  # overridden below; kept for grep
SCENARIO = "scenarios/escalating_pressure.yaml"
COVERAGES = [None, 0.15, 0.3, 0.6, 0.9]  # None = gate disabled
COUNTERPARTIES = ["honest", "cautious"]
SEEDS = [42, 7, 101, 2026, 555]
CAUTIOUS_THRESHOLD = 0.7


def run_cell(coverage, counterparty, seed):
    scenario = load_scenario(SCENARIO)
    gc = scenario.orchestrator_config.governance_config
    if coverage is None:
        gc.certificate_gate_enabled = False
    else:
        gc.certificate_gate_enabled = True
        gc.certificate_coverage = coverage
    scenario.orchestrator_config.seed = seed

    if counterparty == "cautious":
        for spec in scenario.agent_specs:
            if spec.get("type") == "honest":
                spec["type"] = "cautious"
                spec["config"] = {"threshold": CAUTIOUS_THRESHOLD}

    orch = build_orchestrator(scenario)
    rec = []  # (epoch, interaction)
    orch.on_interaction_complete(
        lambda i, _a, _b: rec.append((orch.state.current_epoch, i))
    )
    history = orch.run()

    n_epochs = scenario.orchestrator_config.n_epochs
    late_start = 2 * n_epochs // 3
    late = [i for (e, i) in rec if e >= late_start]
    metrics = SoftMetrics()
    d = metrics.pcg_decomposition(late)

    late_acc = [i for i in late if i.accepted]
    cert_acc = [i for i in late_acc if i.ground_truth is not None]
    fab_acc = [i for i in cert_acc if i.ground_truth == -1]

    # Realized acceptance pressure on workers: worker-initiated proposals
    # carry ground truth, so use certified interactions as the worker set.
    worker_all = [i for i in late if i.ground_truth is not None]
    worker_accept_rate = (
        len([i for i in worker_all if i.accepted]) / len(worker_all)
        if worker_all
        else float("nan")
    )

    return {
        "coverage": -1.0 if coverage is None else coverage,
        "counterparty": counterparty,
        "seed": seed,
        "fab_share": (len(fab_acc) / len(cert_acc)) if cert_acc else 0.0,
        "overtrust": d["overtrust"] if d["overtrust"] is not None else 0.0,
        "pcg_accepted": d["pcg_accepted"],
        "pcg_selection": d["pcg_selection"],
        "quality_gap": metrics.quality_gap(late),
        "worker_accept_rate": worker_accept_rate,
        "n_certified_accepted": d["n_certified_accepted"],
        "late_epoch_overtrust_mean": _mean_or_nan(
            [
                m.pcg_overtrust
                for m in history
                if m.epoch >= late_start and m.pcg_overtrust is not None
            ]
        ),
    }


def _mean_or_nan(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    coverages = [None, 0.3, 0.9] if args.quick else COVERAGES
    seeds = SEEDS[:2] if args.quick else SEEDS

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / f"{ts}_pressure_coverage_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(coverages) * len(COUNTERPARTIES) * len(seeds)
    done = 0
    for counterparty in COUNTERPARTIES:
        for coverage in coverages:
            for seed in seeds:
                rows.append(run_cell(coverage, counterparty, seed))
                done += 1
                print(f"  [{done}/{total}] cp={counterparty} "
                      f"cov={coverage} seed={seed}", flush=True)

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Per-cell summary across seeds
    print(f"\n{'cp':8s} {'coverage':>8s} {'fab_share':>16s} {'overtrust':>16s} "
          f"{'pcg_selection':>16s} {'quality_gap':>16s} {'acc_rate':>9s}")
    for counterparty in COUNTERPARTIES:
        for coverage in coverages:
            cov_key = -1.0 if coverage is None else coverage
            cell = [
                r for r in rows
                if r["counterparty"] == counterparty
                and r["coverage"] == cov_key
            ]

            def ms(key, cell=cell):
                vals = [r[key] for r in cell if r[key] is not None]
                if not vals:
                    return "n/a"
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                return f"{m:+.3f}±{s:.3f}"

            cov_label = "off" if coverage is None else f"{coverage:.2f}"
            acc = statistics.mean(r["worker_accept_rate"] for r in cell)
            print(f"{counterparty:8s} {cov_label:>8s} {ms('fab_share'):>16s} "
                  f"{ms('overtrust'):>16s} {ms('pcg_selection'):>16s} "
                  f"{ms('quality_gap'):>16s} {acc:>9.3f}")

    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
