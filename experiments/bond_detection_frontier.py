"""Bond × detection frontier with real levers (beads 3ru4).

Combines the real `RandomAuditLever` (catch probability) and the real
`StakingLever` (slash) to trace where a monitor-evading Sybil attacker is
deterred, and at what honest false-positive cost. Shows:

  1. the per-interaction audit is on the ladder — its catch collapses as the
     attacker cleans each interaction above `audit_threshold_p`;
  2. catching a clean attacker requires a threshold above the honest proxy
     floor, which false-positives on honest work;
  3. the two-lever deterrence region over (min_stake × audit_threshold_p) for
     a fixed attacker, with the honest-FP cost overlaid — deterrence needs
     BOTH a threshold that bites and a stake that makes the slash hurt.

Usage: python -m experiments.bond_detection_frontier [--quick]
Writes runs/<ts>_bond_detection_frontier/results.csv.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from swarm.governance.config import GovernanceConfig
from swarm.redteam.audit_bridge import (
    audit_catch_probability,
    frontier_point,
    honest_audit_false_positive,
)
from swarm.redteam.staking_bridge import SybilAttack

ATK = SybilAttack(objective_mass=2.0, objective_value=100.0, agent_threshold=1.0)
HONEST_P = 0.7
N = 3000


def _cfg(stake, threshold, audit_prob=0.4, slash=0.3):
    return GovernanceConfig(
        audit_enabled=True, audit_probability=audit_prob,
        audit_threshold_p=threshold, staking_enabled=True,
        min_stake_to_participate=float(stake), stake_slash_rate=slash,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n = 1500 if args.quick else N

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / f"{ts}_bond_detection_frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print("1. AUDIT IS ON THE LADDER — catch collapses as attacker cleans up")
    print(f"   (audit_prob=0.4, threshold=0.5); honest proxy floor ~{HONEST_P}")
    print(f"{'attacker_p_int':>15} {'audit_catch':>12}")
    cfg = _cfg(1, 0.5)
    for pint in ([0.3, 0.5, 0.7] if args.quick else [0.2, 0.4, 0.49, 0.5, 0.6, 0.8]):
        c = audit_catch_probability(cfg, pint, n_samples=n)
        print(f"{pint:>15.2f} {c:>12.3f}")
        rows.append({"panel": "ladder", "attacker_p_int": pint, "audit_catch": c})

    print("\n2. CATCHING A CLEAN ATTACKER COSTS HONEST FALSE POSITIVES")
    print(f"{'threshold':>10} {'honest_FP':>10} {'catches_p_int=0.8?':>19}")
    for thr in ([0.5, 0.85] if args.quick else [0.3, 0.5, 0.7, 0.85, 0.95]):
        c = _cfg(1, thr)
        fp = honest_audit_false_positive(c, HONEST_P, n_samples=n)
        catches = audit_catch_probability(c, 0.8, n_samples=n) > 0
        print(f"{thr:>10.2f} {fp:>10.3f} {str(catches):>19}")
        rows.append({"panel": "fp_cost", "threshold": thr, "honest_fp": fp,
                     "catches_clean": catches})

    print("\n3. DETERRENCE REGION  (# deterred, . not; attacker p_int=0.4)")
    print("   rows=min_stake, cols=audit_threshold_p; honest_FP shown per col")
    stakes = ([50, 500, 5000] if args.quick else [50, 200, 500, 1000, 3000])
    thresholds = ([0.5, 0.9] if args.quick else [0.3, 0.5, 0.7, 0.9])
    fps = {thr: honest_audit_false_positive(_cfg(1, thr), HONEST_P, n_samples=n)
           for thr in thresholds}
    print("stake\\thr " + " ".join(f"{t:>6.2f}" for t in thresholds))
    for stake in stakes:
        cells = []
        for thr in thresholds:
            pt = frontier_point(
                ATK, _cfg(stake, thr), liquid_capital=1e9,
                attacker_interaction_p=0.4, honest_p_mean=HONEST_P, n_samples=n,
            )
            cells.append("#" if pt.attacker_deterred else ".")
            rows.append({"panel": "region", "stake": stake, "threshold": thr,
                         "deterred": pt.attacker_deterred,
                         "audit_catch": pt.audit_catch_prob})
        print(f"{stake:>8}  " + " ".join(f"{c:>6}" for c in cells))
    print("honestFP  " + " ".join(f"{fps[t]:>6.2f}" for t in thresholds))
    print("\n  Deterrence needs a threshold that bites (catch>0) AND a stake that")
    print("  makes the slash hurt. At p_int=0.4 the attacker sits below thr>=0.5,")
    print("  so those columns can deter with enough stake; a cleaner attacker")
    print("  would push the biting threshold into the high-honest-FP zone.")

    csv_path = out_dir / "results.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
