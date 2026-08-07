"""Attacker-economics frontier and minimum-stake curve (beads 7bwz).

Turns the ladder finding into a governance number. For an objective worth V,
monitoring at aggregation level L is economically defensible (deters within a
false-positive budget) iff the per-unit cost clears

    c_crit = theta_fp * margin * V / M.

At the agent level the per-unit cost is the stake an identity must post
(SWARM's `min_stake_to_participate`), so c_crit is the minimum stake for
agent-level monitoring to work.

Prints:
  1. defensibility by level at a reference config (why trajectory-level
     monitoring can't be economically defended but the agent level can, given
     enough stake);
  2. the minimum-stake curve vs objective value;
  3. the defensible region in (stake, objective value) space.

Usage: python -m experiments.attacker_economics_frontier [--quick]
Writes runs/<ts>_attacker_economics/results.csv.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from swarm.redteam.attacker_economics import (
    EconomicsConfig,
    critical_unit_cost,
    deterrence_threshold,
    fp_threshold,
    is_defensible,
)
from swarm.redteam.decomposition_ladder import Level

F_MAX = 0.05  # tolerated honest false-positive rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / f"{ts}_attacker_economics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    cfg = EconomicsConfig()
    print(f"Reference: objective mass M={cfg.objective_mass}, value V="
          f"{cfg.objective_value}, FP budget f_max={F_MAX}")
    print("\nDEFENSIBILITY BY LEVEL")
    print(f"{'level':>11} {'theta_deter':>12} {'theta_fp':>10} "
          f"{'c_crit':>9} {'actual_c':>9} {'defensible':>11}")
    for level in (Level.TRAJECTORY, Level.AGENT):
        td = deterrence_threshold(level, cfg)
        tf = fp_threshold(level, cfg, F_MAX)
        cc = critical_unit_cost(level, cfg, F_MAX)
        c = cfg.unit_cost[level]
        d = is_defensible(level, cfg, F_MAX)
        print(f"{level.name:>11} {td:>12.3f} {tf:>10.3f} {cc:>9.2f} "
              f"{c:>9.2f} {str(d):>11}")
        rows.append({"kind": "defensibility", "level": level.name,
                     "theta_deter": td, "theta_fp": tf, "c_crit": cc,
                     "actual_c": c, "defensible": d})

    print("\nMINIMUM STAKE vs OBJECTIVE VALUE (agent level)")
    print(f"{'obj_value':>10} {'min_stake':>11} {'default_stake_ok':>17}")
    values = ([50, 200, 1000] if args.quick
              else [10, 25, 50, 100, 250, 500, 1000, 2500, 5000])
    for v in values:
        c = EconomicsConfig(objective_value=float(v))
        min_stake = critical_unit_cost(Level.AGENT, c, F_MAX)
        ok = c.unit_cost[Level.AGENT] >= min_stake
        print(f"{v:>10} {min_stake:>11.1f} {str(ok):>17}")
        rows.append({"kind": "min_stake", "objective_value": v,
                     "min_stake": min_stake, "default_stake_ok": ok})

    print("\nDEFENSIBLE REGION  (agent level; '#' = defensible at f_max)")
    stakes = ([10, 100, 1000] if args.quick
              else [10, 30, 100, 300, 1000, 3000])
    header = "stake\\V  " + " ".join(f"{v:>6}" for v in values)
    print(header)
    for stake in stakes:
        cells = []
        for v in values:
            c = EconomicsConfig(
                objective_value=float(v),
                unit_cost={Level.STEP: 0.01, Level.TRAJECTORY: 0.1,
                           Level.AGENT: float(stake)},
            )
            cells.append("#" if is_defensible(Level.AGENT, c, F_MAX) else ".")
            rows.append({"kind": "region", "stake": stake, "objective_value": v,
                         "defensible": cells[-1] == "#"})
        print(f"{stake:>7}  " + " ".join(f"{x:>6}" for x in cells))

    csv_path = out_dir / "results.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
