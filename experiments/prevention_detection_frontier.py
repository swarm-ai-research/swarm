"""Prevention vs detection under decomposition, real levers (bead vwj3).

Same axis and audit settings as experiments/bond_detection_frontier.py (3ru4):
attacker per-step proxy label p_int, audit_probability 0.4. Three panels:

  1. robustness shape: per-step audit catch vs precondition refusal over p_int;
  2. harm realized over (p_int x coverage) for audit, precondition, and both,
     at k=20 steps, with the lever that stops more harm marked per cell;
     2b. the coverage above which the precondition beats the audit against a
     dirty attacker, as attack length k varies;
  3. what raising the audit threshold to catch a clean attacker costs in
     honest false positives, against the precondition's zero for honest
     agents that stay inside their grant.

Usage: python -m experiments.prevention_detection_frontier [--quick] [--out DIR]
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

# Pin `swarm` to THIS checkout (bead hjyp). `python experiments/x.py` puts
# experiments/ on sys.path[0], not the repo root, so `import swarm` otherwise
# resolves through the editable install, which points at whichever checkout ran
# `pip install -e .` last -- possibly another session's worktree.
__import__("sys").path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parents[1])
)

from swarm.governance.config import GovernanceConfig
from swarm.redteam.audit_bridge import (
    audit_catch_probability,
    honest_audit_false_positive,
)
from swarm.redteam.capability_bridge import per_step_refusal, simulate_attack

K = 20
HONEST_P = 0.7


def _cfg(threshold=0.5, audit_prob=0.4):
    return GovernanceConfig(audit_enabled=True, audit_probability=audit_prob, audit_threshold_p=threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    n = 500 if args.quick else 3000
    p_axis = [0.3, 0.5, 0.8] if args.quick else [0.2, 0.3, 0.4, 0.49, 0.5, 0.6, 0.7, 0.8]
    coverages = [0.0, 0.25, 1.0] if args.quick else [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path("runs") / f"{ts}_prevention_detection_frontier"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print("1. ROBUSTNESS SHAPE — per step, audit_threshold_p=0.5")
    print(f"{'p_int':>6} {'audit_catch':>12} {'precondition_refusal':>21}")
    cfg = _cfg()
    for p in p_axis:
        catch = audit_catch_probability(cfg, p, n_samples=n)
        refusal = per_step_refusal(p)
        print(f"{p:>6.2f} {catch:>12.3f} {refusal:>21.3f}")
        rows.append({"panel": "shape", "p_int": p, "audit_catch": catch, "refusal": refusal})

    print(f"\n2. HARM REALIZED (fraction of {K} steps executed) — A=audit P=precondition B=both")
    print("   winner: P if precondition stops more, A if audit does, = if equal")
    header = "p_int\\cov " + " ".join(f"{c:>17.2f}" for c in coverages)
    print(header)
    for p in p_axis:
        cells = []
        for c in coverages:
            h = simulate_attack(cfg, p, c, k=K, n_trials=n)
            win = "=" if abs(h.precondition - h.audit) < 1e-9 else ("P" if h.precondition < h.audit else "A")
            cells.append(f"A{h.audit:.2f} P{h.precondition:.2f} B{h.both:.2f}{win}")
            rows.append({"panel": "harm", "p_int": p, "coverage": c, "k": K,
                         "harm_audit": h.audit, "harm_precondition": h.precondition,
                         "harm_both": h.both, "audit_halt_rate": h.audit_halt_rate, "winner": win})
        print(f"{p:>9.2f} " + " ".join(f"{s:>17}" for s in cells))

    print("\n2b. BREAK-EVEN COVERAGE vs ATTACK LENGTH — dirty attacker (p_int=0.3)")
    print("   precondition beats audit iff coverage > 1 - harm_audit(k); clean attackers: any coverage > 0")
    print(f"{'k':>4} {'harm_audit':>11} {'breakeven_coverage':>19}")
    for k in ([1, 5, 20] if args.quick else [1, 2, 3, 5, 10, 20, 50]):
        h = simulate_attack(cfg, 0.3, 0.0, k=k, n_trials=n)
        print(f"{k:>4} {h.audit:>11.3f} {1 - h.audit:>19.3f}")
        rows.append({"panel": "breakeven", "p_int": 0.3, "k": k, "harm_audit": h.audit,
                     "breakeven_coverage": 1 - h.audit})

    print("\n3. BUYING DETECTION BACK — threshold needed to catch p_int=0.8, and its honest cost")
    print(f"{'threshold':>10} {'catch@0.8':>10} {'honest_FP':>10}   (precondition honest FP: 0.000 inside grant)")
    for thr in ([0.5, 0.85] if args.quick else [0.5, 0.7, 0.85, 0.95]):
        c = _cfg(thr)
        catch = audit_catch_probability(c, 0.8, n_samples=n)
        fp = honest_audit_false_positive(c, HONEST_P, n_samples=n)
        print(f"{thr:>10.2f} {catch:>10.3f} {fp:>10.3f}")
        rows.append({"panel": "fp_cost", "threshold": thr, "audit_catch_clean": catch, "honest_fp": fp})

    fields = sorted({k for r in rows for k in r})
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "run.json").write_text(json.dumps(
        {"k": K, "n_trials": n, "audit_probability": 0.4, "honest_p": HONEST_P,
         "p_axis": p_axis, "coverages": coverages}, indent=2) + "\n")
    print(f"\nWrote {out_dir / 'results.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
