"""Offense/defense asymmetry of correlation (bead distributional-agi-safety-pcdq).

Tests the claim made in docs/research/hyperspace-two-swarms-lessons.md: that
correlation degrades attacker and defender swarms by the same factor, and the
asymmetry lies only in observability.

The note modelled defender true-finding retention as *disjunctive* over the
verifier pool -- a finding survives if some verifier retains it. That is wrong
for the mechanism it was describing. Hyperspace's adversarial verifier "attacks
each finding and drops the ones the evidence does not carry": if any verifier
can drop, retention is *conjunctive*, and conjunctive stages respond to
correlation with the opposite sign.

So the defender pipeline has two stages that want different things:

  stage 1  detection     disjunctive over N_d detectors   -- wants independence
  stage 2  verification  conjunctive over N_v verifiers   -- wants correlation

The attacker has only the disjunctive stage. This experiment computes both
sides exactly, sweeps rho, and reports where the defender's optimum actually
sits.

Correlation model (exchangeable common shock), for marginal rate q and pairwise
correlation exactly rho:

    X_i = B * Z + (1 - B) * Y_i,  B ~ Bern(rho),  Z, Y_i ~ Bern(q) iid

    Cov(X_i, X_j) = rho * q * (1 - q),  Var = q * (1 - q)  =>  Corr = rho

Structured variant (two-level shock, K families of size m), used to test whether
the exchangeability assumption is load-bearing:

    X_i = G * Z_glob + (1 - G) * [ F * Z_fam(i) + (1 - F) * Y_i ]
    G ~ Bern(a), F ~ Bern(b)

    corr(same family)  = a + (1 - a) * b
    corr(cross family) = a

Run:
    python -m experiments.correlation_asymmetry --output runs/
    python -m experiments.correlation_asymmetry --validate --trials 200000
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Exchangeable common-shock model (analytic)
# ---------------------------------------------------------------------------


def p_none(q: float, rho: float, n: int) -> float:
    """P(no agent fires) under the exchangeable common-shock model."""
    if n <= 0:
        return 1.0
    return rho * (1.0 - q) + (1.0 - rho) * (1.0 - q) ** n


def p_any(q: float, rho: float, n: int) -> float:
    """P(at least one agent fires) -- the disjunctive stage."""
    return 1.0 - p_none(q, rho, n)


def p_all(q: float, rho: float, n: int) -> float:
    """P(every agent fires) -- the conjunctive stage."""
    if n <= 0:
        return 1.0
    return rho * q + (1.0 - rho) * q**n


def p_at_least(q: float, rho: float, n: int, m: int) -> float:
    """P(at least m of n fire). Mixture of the shock atom and a binomial."""
    if m <= 0:
        return 1.0
    if m > n:
        return 0.0
    shock = q if n >= m else 0.0
    binom = sum(comb(n, k) * q**k * (1 - q) ** (n - k) for k in range(m, n + 1))
    return rho * shock + (1.0 - rho) * binom


# ---------------------------------------------------------------------------
# Two-level (structured) common-shock model (analytic)
# ---------------------------------------------------------------------------


def family_sizes(n: int, k_families: int) -> List[int]:
    """Partition n agents into k families as evenly as possible.

    Handles the case where k does not divide n -- truncating with ``n // k``
    would silently drop agents and break the correlation bookkeeping.
    """
    if k_families <= 1:
        return [n]
    k = min(k_families, n)
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def same_family_pair_fraction(sizes: List[int]) -> float:
    """Fraction of agent pairs that share a family."""
    n = sum(sizes)
    if n < 2:
        return 0.0
    total = comb(n, 2)
    same = sum(comb(m, 2) for m in sizes if m >= 2)
    return same / total if total else 0.0


def two_level_params(
    rho_bar: float, sizes: List[int], spread: float
) -> Tuple[float, float, float]:
    """Solve for (a, b) targeting mean pairwise correlation ``rho_bar``.

    ``spread`` in [0, 1] controls how much correlation is concentrated within
    families: 0 recovers the exchangeable model (a = rho_bar, b = 0), 1 pushes
    as much as possible into within-family correlation.

    The within-family shock ``b`` is capped at 1, so for small
    same-family pair fractions the requested rho_bar may be unreachable.
    The third return value is the correlation actually achieved, which is what
    downstream comparisons must be matched on.

    Returns:
        (a, b, achieved_rho_bar)
    """
    f_same = same_family_pair_fraction(sizes)
    # rho_bar = a + f_same * (1 - a) * b
    a = (1.0 - spread) * rho_bar
    if f_same <= 0.0 or a >= 1.0:
        return rho_bar, 0.0, rho_bar
    b = (rho_bar - a) / (f_same * (1.0 - a))
    b = min(1.0, max(0.0, b))
    achieved = a + f_same * (1.0 - a) * b
    return a, b, achieved


def p_none_structured(q: float, a: float, b: float, sizes: List[int]) -> float:
    """P(no agent fires) under the two-level shock model."""
    n = sum(sizes)
    if n <= 0:
        return 1.0
    if len(sizes) <= 1:
        return p_none(q, a + (1 - a) * b, n)
    prod = 1.0
    for m in sizes:
        prod *= b * (1.0 - q) + (1.0 - b) * (1.0 - q) ** m
    return a * (1.0 - q) + (1.0 - a) * prod


def p_all_structured(q: float, a: float, b: float, sizes: List[int]) -> float:
    """P(every agent fires) under the two-level shock model."""
    n = sum(sizes)
    if n <= 0:
        return 1.0
    if len(sizes) <= 1:
        return p_all(q, a + (1 - a) * b, n)
    prod = 1.0
    for m in sizes:
        prod *= b * q + (1.0 - b) * q**m
    return a * q + (1.0 - a) * prod


# ---------------------------------------------------------------------------
# Swarm outcomes
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # attacker
    n_attackers: int = 8
    q_attack: float = 0.3
    # defender stage 1: detection (disjunctive)
    n_detectors: int = 8
    q_detect: float = 0.3
    # defender stage 2: verification (conjunctive -- any verifier may drop)
    n_verifiers: int = 5
    e_false_drop: float = 0.4
    verify_rule: str = "unanimous"  # unanimous | majority | any_keeps
    horizon: int = 20


def attacker_hit(cfg: Config, rho: float) -> float:
    """Per-round probability the attacker swarm lands an exploit."""
    return p_any(cfg.q_attack, rho, cfg.n_attackers)


def defender_detect(cfg: Config, rho: float) -> float:
    """Probability a true event is surfaced by at least one detector."""
    return p_any(cfg.q_detect, rho, cfg.n_detectors)


def defender_retain(cfg: Config, rho: float) -> float:
    """Probability a surfaced true finding survives the verifier pool.

    ``unanimous``  -- retained only if no verifier drops it (conjunctive).
    ``majority``   -- retained if fewer than half drop it.
    ``any_keeps``  -- retained if at least one verifier keeps it (disjunctive).
    """
    e, n = cfg.e_false_drop, cfg.n_verifiers
    if cfg.verify_rule == "unanimous":
        # no verifier drops == P(zero drops) == P(all "keep") with rate 1-e
        return p_all(1.0 - e, rho, n)
    if cfg.verify_rule == "majority":
        # retained if at least ceil((n+1)/2) verifiers keep
        return p_at_least(1.0 - e, rho, n, n // 2 + 1)
    if cfg.verify_rule == "any_keeps":
        return p_any(1.0 - e, rho, n)
    raise ValueError(f"unknown verify_rule: {cfg.verify_rule}")


def defender_catch(cfg: Config, rho: float) -> float:
    """End-to-end probability a true event is both surfaced and retained."""
    return defender_detect(cfg, rho) * defender_retain(cfg, rho)


# ---------------------------------------------------------------------------
# Monte Carlo validation of the analytic forms
# ---------------------------------------------------------------------------


def mc_stage(q: float, rho: float, n: int, trials: int, rng: random.Random) -> Tuple[float, float, float]:
    """Empirical (P(any), P(all), pairwise corr) under the common-shock model."""
    any_c = all_c = 0
    xs: List[int] = []
    ys: List[int] = []
    for _ in range(trials):
        if rng.random() < rho:
            z = 1 if rng.random() < q else 0
            draws = [z] * n
        else:
            draws = [1 if rng.random() < q else 0 for _ in range(n)]
        s = sum(draws)
        any_c += s > 0
        all_c += s == n
        if n >= 2:
            xs.append(draws[0])
            ys.append(draws[1])
    corr = 0.0
    if xs:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / len(xs)
        vx = sum((a - mx) ** 2 for a in xs) / len(xs)
        vy = sum((b - my) ** 2 for b in ys) / len(ys)
        corr = cov / (vx * vy) ** 0.5 if vx > 1e-12 and vy > 1e-12 else 0.0
    return any_c / trials, all_c / trials, corr


def validate(cfg: Config, trials: int, seed: int) -> bool:
    """Check analytic forms against Monte Carlo. Returns True if all pass."""
    rng = random.Random(seed)
    ok = True
    print(f"Monte Carlo validation ({trials} trials/cell)")
    print(f"{'rho':>5} {'P(any) an':>10} {'P(any) mc':>10} {'P(all) an':>10} {'P(all) mc':>10} {'corr mc':>9}")
    for rho in (0.0, 0.3, 0.7, 1.0):
        a_mc, l_mc, corr = mc_stage(cfg.q_attack, rho, cfg.n_attackers, trials, rng)
        a_an = p_any(cfg.q_attack, rho, cfg.n_attackers)
        l_an = p_all(cfg.q_attack, rho, cfg.n_attackers)
        print(f"{rho:5.2f} {a_an:10.4f} {a_mc:10.4f} {l_an:10.4f} {l_mc:10.4f} {corr:9.4f}")
        tol = 4.0 / trials**0.5
        if abs(a_an - a_mc) > tol or abs(l_an - l_mc) > tol or abs(corr - rho) > tol:
            ok = False
    print(f"validation: {'PASS' if ok else 'FAIL'}\n")
    return ok


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def sweep_exchangeable(cfg: Config, rhos: List[float]) -> List[dict]:
    rows = []
    for rho in rhos:
        p_a = attacker_hit(cfg, rho)
        p_det = defender_detect(cfg, rho)
        p_ret = defender_retain(cfg, rho)
        p_c = p_det * p_ret
        rows.append(
            {
                "rho": round(rho, 4),
                "attacker_hit": round(p_a, 6),
                "attacker_rounds_to_hit": round(1.0 / p_a, 4) if p_a > 0 else float("inf"),
                "attacker_cum_success": round(1.0 - (1.0 - p_a) ** cfg.horizon, 6),
                "defender_detect": round(p_det, 6),
                "defender_retain": round(p_ret, 6),
                "defender_catch": round(p_c, 6),
                "cum_uncaught": round(cfg.horizon * p_a * (1.0 - p_c), 4),
            }
        )
    return rows


def sweep_structured(cfg: Config, rhos: List[float], k_families: int, spread: float) -> List[dict]:
    """Sweep the two-level model. Rows carry *achieved* rho_bar, not requested.

    When the within-family shock saturates, the requested rho_bar is
    unreachable; comparisons against the exchangeable arm must be made on the
    achieved value, so both are recorded.
    """
    det_sizes = family_sizes(cfg.n_detectors, k_families)
    ver_sizes = family_sizes(cfg.n_verifiers, k_families)
    rows = []
    for rho in rhos:
        a_d, b_d, ach_d = two_level_params(rho, det_sizes, spread)
        a_v, b_v, ach_v = two_level_params(rho, ver_sizes, spread)
        p_det = 1.0 - p_none_structured(cfg.q_detect, a_d, b_d, det_sizes)
        p_ret = p_all_structured(1.0 - cfg.e_false_drop, a_v, b_v, ver_sizes)
        rows.append(
            {
                "rho_requested": round(rho, 4),
                "rho_achieved_detect": round(ach_d, 4),
                "rho_achieved_verify": round(ach_v, 4),
                "saturated": int(abs(ach_d - rho) > 1e-6 or abs(ach_v - rho) > 1e-6),
                "families": k_families,
                "detector_families": "|".join(map(str, det_sizes)),
                "verifier_families": "|".join(map(str, ver_sizes)),
                "spread": spread,
                "defender_detect": round(p_det, 6),
                "defender_retain": round(p_ret, 6),
                "defender_catch": round(p_det * p_ret, 6),
            }
        )
    return rows


def argmax_row(rows: List[dict], key: str, rho_key: str = "rho") -> dict:
    return max(rows, key=lambda r: r[key])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="runs/", help="run-artifact root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=21, help="rho grid resolution")
    ap.add_argument("--trials", type=int, default=200000, help="MC trials for --validate")
    ap.add_argument("--validate", action="store_true", help="run MC validation and exit")
    ap.add_argument("--verify-rule", default="unanimous", choices=["unanimous", "majority", "any_keeps"])
    ap.add_argument("--false-drop", type=float, default=0.4)
    ap.add_argument("--verifiers", type=int, default=5)
    args = ap.parse_args()

    cfg = Config(
        verify_rule=args.verify_rule,
        e_false_drop=args.false_drop,
        n_verifiers=args.verifiers,
    )

    if args.validate:
        return 0 if validate(cfg, args.trials, args.seed) else 1

    rhos = [i / (args.steps - 1) for i in range(args.steps)]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / f"{ts}_correlation_asymmetry_seed{args.seed}"
    csv_dir = run_dir / "csv"

    ok = validate(cfg, 50000, args.seed)

    # --- main sweep -------------------------------------------------------
    rows = sweep_exchangeable(cfg, rhos)
    write_csv(csv_dir / "exchangeable.csv", rows)

    print(f"Exchangeable sweep  (rule={cfg.verify_rule}, e={cfg.e_false_drop}, N_v={cfg.n_verifiers})")
    print(f"{'rho':>5} {'atk hit':>9} {'atk rnds':>9} {'def detect':>11} {'def retain':>11} {'def catch':>10}")
    for r in rows[:: max(1, len(rows) // 10)]:
        print(
            f"{r['rho']:5.2f} {r['attacker_hit']:9.4f} {r['attacker_rounds_to_hit']:9.2f} "
            f"{r['defender_detect']:11.4f} {r['defender_retain']:11.4f} {r['defender_catch']:10.4f}"
        )

    best = argmax_row(rows, "defender_catch")
    print("\nattacker optimum      rho = 0.00 (monotone decreasing in rho)")
    print(f"defender optimum      rho = {best['rho']:.2f}  catch = {best['defender_catch']:.4f}")
    print(f"defender at rho=0     catch = {rows[0]['defender_catch']:.4f}")
    print(f"defender at rho=1     catch = {rows[-1]['defender_catch']:.4f}")

    # --- verify-rule comparison ------------------------------------------
    rule_rows = []
    for rule in ("unanimous", "majority", "any_keeps"):
        c = Config(verify_rule=rule, e_false_drop=cfg.e_false_drop, n_verifiers=cfg.n_verifiers)
        rr = sweep_exchangeable(c, rhos)
        b = argmax_row(rr, "defender_catch")
        rule_rows.append(
            {
                "verify_rule": rule,
                "argmax_rho": b["rho"],
                "catch_at_argmax": b["defender_catch"],
                "catch_at_rho0": rr[0]["defender_catch"],
                "catch_at_rho1": rr[-1]["defender_catch"],
            }
        )
    write_csv(csv_dir / "verify_rules.csv", rule_rows)
    print(f"\n{'rule':>10} {'argmax rho':>11} {'catch@max':>10} {'catch@0':>9} {'catch@1':>9}")
    for r in rule_rows:
        print(
            f"{r['verify_rule']:>10} {r['argmax_rho']:11.2f} {r['catch_at_argmax']:10.4f} "
            f"{r['catch_at_rho0']:9.4f} {r['catch_at_rho1']:9.4f}"
        )

    # --- structured-correlation check (falsifier) ------------------------
    struct_all = []
    print(f"\nStructured correlation (two-level shock), rule={cfg.verify_rule}")
    print(
        f"{'families':>9} {'spread':>7} {'argmax rho':>11} {'achieved':>9} "
        f"{'catch@max':>10} {'sat':>4}"
    )
    for k in (1, 2, 4):
        for spread in (0.0, 0.5, 1.0):
            sr = sweep_structured(cfg, rhos, k, spread)
            struct_all.extend(sr)
            b = max(sr, key=lambda r: r["defender_catch"])
            print(
                f"{k:9d} {spread:7.1f} {b['rho_requested']:11.2f} "
                f"{b['rho_achieved_detect']:9.3f} {b['defender_catch']:10.4f} "
                f"{'yes' if b['saturated'] else 'no':>4}"
            )
    write_csv(csv_dir / "structured.csv", struct_all)
    n_sat = sum(r["saturated"] for r in struct_all)
    if n_sat:
        print(
            f"note: {n_sat}/{len(struct_all)} structured cells could not reach the "
            f"requested rho_bar (within-family shock saturated); compare on achieved."
        )

    print(f"\nMC validation: {'PASS' if ok else 'FAIL'}")
    print(f"artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
