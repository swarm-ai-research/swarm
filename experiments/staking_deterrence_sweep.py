"""Staking-deterrence sweep against the real StakingLever (beads zxkw).

Validates — and revises — the 7bwz law ``min_stake ~= 0.52 V`` by running the
actual `swarm/governance/admission.py` staking lever. Shows three things:

  1. a pure recoverable bond does not deter a well-capitalized Sybil attacker
     at any stake level (it recovers the bond);
  2. the capital-scarcity channel deters at ``stake = C / n`` — a function of
     attacker capital, not objective value;
  3. the slash channel deters at ``stake = V(1-p)^n/(n p sigma)`` — large, and
     falling steeply as the residual catch probability ``p`` rises, so staking
     and audit/collusion detection are complements.

Usage: python -m experiments.staking_deterrence_sweep [--quick]
Writes runs/<ts>_staking_deterrence/results.csv.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from swarm.governance.config import GovernanceConfig
from swarm.redteam.staking_bridge import (
    SybilAttack,
    deterrence_stake_capital_channel,
    deterrence_stake_slash_channel,
    evaluate_sybil_attack,
    identities_needed,
)


def _cfg(stake, slash=0.2):
    return GovernanceConfig(
        staking_enabled=True,
        min_stake_to_participate=float(stake),
        stake_slash_rate=float(slash),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / f"{ts}_staking_deterrence"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    atk = SybilAttack(objective_mass=2.0, objective_value=100.0, agent_threshold=1.0)
    n = identities_needed(atk)
    print(f"Sybil attack: M={atk.objective_mass} V={atk.objective_value} "
          f"identities needed n={n}")

    print("\n1. PURE RECOVERABLE BOND (ample capital, catch_prob=0) — does stake deter?")
    print(f"{'stake':>8} {'net_payoff':>11} {'deterred':>9}")
    for stake in ([10, 100, 1000] if args.quick else [10, 50, 100, 500, 5000]):
        v = evaluate_sybil_attack(atk, _cfg(stake), liquid_capital=1e9, catch_prob=0.0)
        print(f"{stake:>8} {v.net_payoff:>11.1f} {str(v.deterred):>9}")
        rows.append({"regime": "recoverable_bond", "stake": stake,
                     "net_payoff": v.net_payoff, "deterred": v.deterred})

    print("\n2. CAPITAL-SCARCITY CHANNEL (catch_prob=0), by attacker liquid capital")
    print(f"{'capital':>8} {'deter_stake=C/n':>16} {'depends_on':>14}")
    for cap in ([100, 1000] if args.quick else [60, 100, 300, 1000]):
        thr = deterrence_stake_capital_channel(atk, float(cap))
        print(f"{cap:>8} {thr:>16.1f} {'attacker cap':>14}")
        rows.append({"regime": "capital_scarcity", "capital": cap,
                     "deter_stake": thr})

    print("\n3. SLASH CHANNEL (ample capital) — deterrence stake vs residual catch prob")
    print(f"{'catch_prob':>10} {'deter_stake':>12} {'vs naive 52':>12}")
    for p in ([0.05, 0.2] if args.quick else [0.02, 0.05, 0.1, 0.2, 0.4]):
        s = deterrence_stake_slash_channel(atk, _cfg(1, slash=0.2), catch_prob=p)
        print(f"{p:>10.2f} {s:>12.1f} {s/51.57:>11.1f}x")
        rows.append({"regime": "slash_channel", "catch_prob": p, "deter_stake": s})

    print("\n  Interpretation: the naive 7bwz law (~52) holds only if the stake")
    print("  were a sunk cost. As a recoverable bond it deters only via capital")
    print("  scarcity or slash x catch — staking and detection are complements.")

    csv_path = out_dir / "results.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
