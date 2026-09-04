"""Loan/commitment sweep: default as a strategic action (beads rjl3).

Runs ``swarm/contracts/loan.py`` over the grid in
``scenarios/loan_commitment.yaml`` and evaluates two preregistered
predictions from the AI Village Ṁ5,000 loan post-mortem:

  P1  With homogeneous loans and w_rep = 0, the default rate is a step in
      rho (every cell is 0 or 1
      the flip is at the analytic rho*), not a
      gradient. Heterogeneous loan sizes smooth the aggregate into a CDF,
      but each borrower's decision stays a step.
  P2  At rho = 0 a gift does not change the default rate. It only moves
      borrowers from "unable" to "able and still defaulting" — the Village
      case exactly.

Usage: python -m experiments.loan_commitment_sweep [--quick] [--config PATH]
Writes runs/<ts>_loan_commitment/{results.csv,predictions.json,plots/}.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from swarm.contracts.loan import (
    BorrowerPolicy,
    LoanScenario,
    LoanTerms,
    simulate_loans,
    strategic_default_threshold_rho,
    village_case,
)

DEFAULT_CONFIG = Path("scenarios/loan_commitment.yaml")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_scenario(cfg: dict, *, rho, w_rep, bond, gift, principal_sigma) -> LoanScenario:
    t, b, inv = cfg["terms"], cfg["borrower"], cfg["investment"]
    return LoanScenario(
        terms=LoanTerms(principal=t["principal"], interest_rate=t["interest_rate"], bond=bond),
        policy=BorrowerPolicy(rho=rho, w_rep=w_rep, r_default=b["r_default"], r_repay=b["r_repay"]),
        n_borrowers=cfg["n_borrowers"],
        return_mu=inv["return_mu"],
        return_sigma=inv["return_sigma"],
        own_capital=cfg["own_capital"],
        gift=gift,
        principal_sigma=principal_sigma,
    )


def run_grid(cfg: dict, seed: int, quick: bool) -> list[dict]:
    sw = cfg["sweep"]
    rhos = sw["rho"][:: (2 if quick else 1)]
    w_reps = sw["w_rep"][:: (2 if quick else 1)]
    bonds = sw["bond"]
    gifts = sw["gift"]
    sigmas = cfg.get("heterogeneity_panel", [cfg["principal_sigma"]])
    if quick:
        sigmas = sigmas[:2]
    rows = []
    for sigma in sigmas:
        for bond in bonds:
            for w_rep in w_reps:
                for rho in rhos:
                    for gift in gifts:
                        sc = build_scenario(cfg, rho=rho, w_rep=w_rep, bond=bond,
                                            gift=gift, principal_sigma=sigma)
                        _, s = simulate_loans(sc, seed=seed)
                        row = s.to_dict()
                        row["principal_sigma"] = sigma
                        row["rho_star"] = strategic_default_threshold_rho(sc.terms, sc.policy)
                        rows.append(row)
    return rows


def evaluate_predictions(rows: list[dict]) -> dict:
    # P1: homogeneous, w_rep=0, bond=0, gift=0 -> every cell in {0,1} and the
    # flip sits at rho* (default iff rho < rho*).
    base = [r for r in rows if r["principal_sigma"] == 0 and r["w_rep"] == 0
            and r["bond"] == 0 and r["gift"] == 0]
    # Everyone who *can* repay defaults iff rho < rho*; the unable default
    # regardless, so the pure step shows in the *strategic* rate among the able.
    step_ok = True
    for r in base:
        able = r["able_rate"]
        strat = r["strategic_default_rate"]
        expected = able if r["rho"] < r["rho_star"] else 0.0
        if abs(strat - expected) > 1e-9:
            step_ok = False
    interior = [r for r in base if 1e-9 < r["strategic_default_rate"] < r["able_rate"] - 1e-9]
    p1 = {
        "met": step_ok and not interior,
        "cells": len(base),
        "interior_cells": len(interior),
        "rho_star": base[0]["rho_star"] if base else None,
        "statement": "homogeneous, w_rep=0: strategic default among the able is 0 or all, flipping at rho*",
    }

    # Heterogeneity: with principal_sigma>0 and w_rep>0 the aggregate should
    # take interior values (the step becomes a CDF). Report, not a pass/fail.
    het = [r for r in rows if r["principal_sigma"] > 0 and r["w_rep"] > 0
           and r["bond"] == 0 and r["gift"] == 0
           and 1e-9 < r["strategic_default_rate"] < r["able_rate"] - 1e-9]

    # P2 as preregistered: rho=0 -> default_rate identical with and without a
    # gift, for every (w_rep, bond, sigma). Also decomposed: the null should
    # hold exactly where able borrowers default anyway (the Village
    # objective), and fail exactly where a bond / reputation weight already
    # makes the able repay -- a gift fixes ability, never willingness.
    key = lambda r: (r["principal_sigma"], r["bond"], r["w_rep"])  # noqa: E731
    by: dict = {}
    for r in rows:
        if r["rho"] == 0:
            by.setdefault(key(r), {})[r["gift"]] = r
    p2_cells = p2_viol = 0
    village_cells = village_viol = 0
    decomposition_ok = True
    decomp_cells = 0
    shift = []
    for k, d in by.items():
        gifts = sorted(d)
        if len(gifts) < 2:
            continue
        a, b = d[gifts[0]], d[gifts[-1]]
        p2_cells += 1
        moved = abs(a["default_rate"] - b["default_rate"]) > 1e-9
        p2_viol += moved
        # "able borrowers repay" in the no-gift cell: some are able, none default strategically
        able_repay = a["able_rate"] > 1e-9 and a["strategic_default_rate"] < a["able_rate"] - 1e-9
        # The population-level iff is exact only when every borrower faces the
        # same decision (homogeneous loans); under heterogeneity the newly-able
        # can all be large-loan defaulters while a few small-loan borrowers repay.
        if k[0] == 0:
            decomp_cells += 1
            if moved != able_repay:
                decomposition_ok = False
        if k[1] == 0 and k[2] == 0:  # Village objective: no bond, no reputation weight
            village_cells += 1
            village_viol += moved
        shift.append(b["strategic_default_rate"] - a["strategic_default_rate"])
    p2 = {
        "met": p2_cells > 0 and p2_viol == 0,
        "cells": p2_cells,
        "violations": p2_viol,
        "mean_strategic_share_shift_from_gift": (sum(shift) / len(shift)) if shift else None,
        "statement": "as preregistered: rho=0 -> gift leaves default_rate unchanged for every (w_rep, bond)",
    }
    p2_village = {
        "met": village_cells > 0 and village_viol == 0,
        "cells": village_cells,
        "violations": village_viol,
        "statement": "refined: under the Village objective (rho=0, w_rep=0, bond=0) a gift leaves default_rate unchanged",
    }
    p2_decomp = {
        "met": decomp_cells > 0 and decomposition_ok,
        "cells": decomp_cells,
        "statement": "refined, homogeneous loans: a gift moves default_rate iff able borrowers already repay (gifts fix ability, not willingness)",
    }
    return {
        "P1_step_in_rho": p1,
        "heterogeneity_interior_cells": len(het),
        "P2_gift_null_as_preregistered": p2,
        "P2a_gift_null_village_objective": p2_village,
        "P2b_gift_moves_default_iff_able_repay": p2_decomp,
    }


def plot(rows: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)
    written = []

    # 1. Heatmap: strategic default rate over rho x w_rep (homogeneous, no bond, no gift).
    sel = [r for r in rows if r["principal_sigma"] == 0 and r["bond"] == 0 and r["gift"] == 0]
    rhos = sorted({r["rho"] for r in sel})
    wreps = sorted({r["w_rep"] for r in sel})
    grid = np.full((len(wreps), len(rhos)), np.nan)
    for r in sel:
        grid[wreps.index(r["w_rep"]), rhos.index(r["rho"])] = r["strategic_default_rate"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(1e-9, np.nanmax(grid)), aspect="auto", origin="lower")
    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels([f"{x:g}" for x in rhos])
    ax.set_yticks(range(len(wreps)))
    ax.set_yticklabels([f"{x:g}" for x in wreps])
    ax.set_xlabel("rho (externality internalization)")
    ax.set_ylabel("w_rep (mana per reputation unit)")
    ax.set_title("Strategic default rate — no bond, no gift")
    for (i, j), v in np.ndenumerate(grid):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 * np.nanmax(grid) else "#333")
    fig.colorbar(im, ax=ax, label="strategic default rate")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = plots / "strategic_default_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 2. Step vs CDF: strategic default vs rho at w_rep = one mid value, by principal_sigma.
    wmid = wreps[len(wreps) // 2] if wreps else 0.0
    sigmas = sorted({r["principal_sigma"] for r in rows})
    colors = ["#1f5fa8", "#d1762b", "#2e8b57", "#7b4fa3"]  # fixed order, <=4 series
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for c, sig in zip(colors, sigmas, strict=False):
        pts = sorted((r["rho"], r["strategic_default_rate"]) for r in rows
                     if r["principal_sigma"] == sig and r["w_rep"] == wmid
                     and r["bond"] == 0 and r["gift"] == 0)
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", ms=4, lw=2, color=c, label=f"principal_sigma={sig:g}")
    ax.set_xlabel("rho")
    ax.set_ylabel("strategic default rate")
    ax.set_title(f"Step (homogeneous) vs CDF (heterogeneous), w_rep={wmid:g}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = plots / "step_vs_cdf.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / f"{ts}_loan_commitment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("AI Village case (rho=0, w_rep=0):")
    for gift in (0.0, 5010.0):
        o = village_case(gift=gift)
        print(f"  gift={gift:>6.0f}  able={o.able!s:5}  defaulted={o.defaulted!s:5}  "
              f"strategic={o.strategic_default!s:5}  lender_loss={o.lender_loss:.0f}")
    o = village_case(BorrowerPolicy(rho=1.0), gift=5010.0)
    print(f"  rho=1, gift=5010: defaulted={o.defaulted} paid={o.paid:.0f}")

    rows = run_grid(cfg, seed=seed, quick=args.quick)
    preds = evaluate_predictions(rows)

    print("\nStrategic default rate, homogeneous loans, no bond/gift (rows w_rep, cols rho):")
    sel = [r for r in rows if r["principal_sigma"] == 0 and r["bond"] == 0 and r["gift"] == 0]
    rhos = sorted({r["rho"] for r in sel})
    print("  w_rep\\rho " + " ".join(f"{x:>5g}" for x in rhos))
    for w in sorted({r["w_rep"] for r in sel}):
        line = {r["rho"]: r["strategic_default_rate"] for r in sel if r["w_rep"] == w}
        print(f"  {w:>9g} " + " ".join(f"{line[x]:>5.2f}" for x in rhos))

    print("\nPredictions:")
    for k, v in preds.items():
        print(f"  {k}: {json.dumps(v) if not isinstance(v, dict) else ('MET' if v['met'] else 'NOT MET')}"
              + (f"  ({ {kk: vv for kk, vv in v.items() if kk not in ('met', 'statement')} })" if isinstance(v, dict) else ""))

    csv_path = out_dir / "results.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "predictions.json", "w") as f:
        json.dump({"seed": seed, "config": str(args.config), **preds}, f, indent=2)
    print(f"\nWrote {csv_path} ({len(rows)} rows) and predictions.json")
    if not args.no_plot:
        for p in plot(rows, out_dir):
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
