#!/usr/bin/env python3
"""Entry fee sweep for the cake-splitting scenario.

Sweeps fair_division entry_fee to find the screening threshold where
adversarial/deceptive agents self-select out of the fair division pool.

Usage:
    python scripts/sweep_cake_entry_fee.py
"""

import copy
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from swarm.analysis.sweep import _extract_results  # noqa: E402
from swarm.scenarios import build_orchestrator, load_scenario  # noqa: E402

ENTRY_FEES = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
SEEDS_PER_FEE = 3
SEED_BASE = 42
SCENARIO_PATH = PROJECT_ROOT / "scenarios" / "cake_splitting.yaml"


def main() -> None:
    scenario = load_scenario(SCENARIO_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "runs" / f"{timestamp}_cake_entry_fee_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    total = len(ENTRY_FEES) * SEEDS_PER_FEE
    results = []

    for fee in ENTRY_FEES:
        for seed_offset in range(SEEDS_PER_FEE):
            seed = SEED_BASE + seed_offset
            run_num = len(results) + 1
            print(
                f"[{run_num}/{total}] entry_fee={fee:.1f}  seed={seed}",
                flush=True,
            )

            s = copy.deepcopy(scenario)
            s.orchestrator_config.contracts_config.fair_division_kwargs[
                "entry_fee"
            ] = fee
            s.orchestrator_config.seed = seed

            orch = build_orchestrator(s)
            orch.run()

            result = _extract_results(
                orch, {"entry_fee": fee}, seed_offset, seed
            )
            results.append(result)

    # ── Export CSV ──────────────────────────────────────────────
    rows = [r.to_dict() for r in results]
    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to {csv_path}  ({len(rows)} rows)")

    # ── Summary table ──────────────────────────────────────────
    grouped: dict[float, list] = defaultdict(list)
    for r in results:
        grouped[r.params["entry_fee"]].append(r)

    header = (
        f"{'fee':>6}  {'infiltration':>12}  {'toxicity':>9}  "
        f"{'welfare':>9}  {'honest_pay':>10}  {'adv_pay':>9}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for fee in ENTRY_FEES:
        runs = grouped[fee]
        n = len(runs)
        avg_inf = sum(r.infiltration_rate for r in runs) / n
        avg_tox = sum(r.avg_toxicity for r in runs) / n
        avg_wel = sum(r.total_welfare for r in runs) / n
        avg_hon = sum(r.honest_avg_payoff for r in runs) / n
        avg_adv = sum(r.adversarial_avg_payoff for r in runs) / n
        print(
            f"{fee:6.1f}  {avg_inf:12.3f}  {avg_tox:9.3f}  "
            f"{avg_wel:9.1f}  {avg_hon:10.3f}  {avg_adv:9.3f}"
        )

    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
