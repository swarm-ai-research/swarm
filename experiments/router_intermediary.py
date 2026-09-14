"""Router intermediary: governance that sees accounts, not operators (bead ggiw).

N operators, a share f of them adversarial, reach the provider through
accounts. With direct access each operator has its own account. Through a
router, m operators share one. The provider sees each account's mean toxicity
E[1 - p] per epoch and nothing else.

Three levers, swept one at a time in this order (scenarios/router_intermediary.yaml):

  1. operators_per_router m: detection AUC of account toxicity for "account
     carries an adversary", at a threshold calibrated to a fixed false-positive
     rate on benign-only accounts of the same size.
  2. ban_policy: none, or ban every flagged account each epoch. Banned operators
     re-route after a delay (short for adversaries, long for benign operators).
     Reports adversary persistence and benign operator-epochs lost per
     adversary operator-epoch removed.
  3. kyc_passthrough: a fraction of routers forward operator ids, so the
     provider scores those operators individually.

Usage:
    python -m experiments.router_intermediary [--quick] [--config PATH] [--output runs/]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import yaml

DEFAULT_CONFIG = Path("scenarios/router_intermediary.yaml")


@dataclass(frozen=True)
class Population:
    n_operators: int
    adversary_share: float
    interactions_per_epoch: int
    benign_mean: float
    benign_conc: float
    adversary_mean: float
    adversary_conc: float

    @property
    def n_adversaries(self) -> int:
        return max(1, round(self.n_operators * self.adversary_share))


def beta_draw(rng: random.Random, mean: float, conc: float) -> float:
    return rng.betavariate(mean * conc, (1.0 - mean) * conc)


def operator_toxicity(rng: random.Random, pop: Population, adversary: bool) -> float:
    """Mean 1 - p over one operator's interactions in one epoch."""
    mean, conc = (
        (pop.adversary_mean, pop.adversary_conc) if adversary else (pop.benign_mean, pop.benign_conc)
    )
    t = pop.interactions_per_epoch
    return sum(1.0 - beta_draw(rng, mean, conc) for _ in range(t)) / t


def partition(ids: Sequence[int], m: int, rng: random.Random) -> List[List[int]]:
    """Randomly assign operators to accounts of m (last account may be smaller)."""
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return [shuffled[i : i + m] for i in range(0, len(shuffled), m)]


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 * P(tie), by rank sum."""
    if not pos or not neg:
        return float("nan")
    scored = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum = 0.0
    i = 0
    while i < len(scored):
        j = i
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(1 for k in range(i, j) if scored[k][1] == 1)
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def z_score(pop: Population, toxicity: float, size: int) -> float:
    """Standardize an account's toxicity against benign accounts of the same size.

    Accounts of different sizes have different noise, so raw toxicity cannot be
    ranked or thresholded across sizes. Uses the analytic benign mean and the
    Beta variance m(1-m)/(c+1) per interaction.
    """
    var = pop.benign_mean * (1.0 - pop.benign_mean) / (pop.benign_conc + 1.0)
    sd = (var / (pop.interactions_per_epoch * size)) ** 0.5
    return float((toxicity - (1.0 - pop.benign_mean)) / sd)


def benign_threshold(rng: random.Random, pop: Population, m: int, fpr: float, samples: int = 4000) -> float:
    """z threshold with false-positive rate fpr on benign-only accounts of m."""
    draws = sorted(
        z_score(pop, sum(operator_toxicity(rng, pop, False) for _ in range(m)) / m, m)
        for _ in range(samples)
    )
    return draws[min(len(draws) - 1, int((1.0 - fpr) * len(draws)))]


def account_scores(
    rng: random.Random, pop: Population, m: int, kyc: float
) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
    """One epoch: (z, size) for accounts carrying an adversary, and for clean ones.

    With probability kyc a router forwards operator ids, and its operators are
    scored as individual accounts.
    """
    n = pop.n_operators
    adversaries = set(range(pop.n_adversaries))
    pos: List[Tuple[float, int]] = []
    neg: List[Tuple[float, int]] = []
    for account in partition(range(n), m, rng):
        units = [[o] for o in account] if (m > 1 and rng.random() < kyc) else [account]
        for unit in units:
            tox = sum(operator_toxicity(rng, pop, o in adversaries) for o in unit) / len(unit)
            (pos if adversaries.intersection(unit) else neg).append((z_score(pop, tox, len(unit)), len(unit)))
    return pos, neg


def detection_cell(pop: Population, m: int, kyc: float, fpr: float, reps: int, seed: int) -> dict:
    rng = random.Random(seed)
    thr = {size: benign_threshold(rng, pop, size, fpr) for size in {1, m}}
    aucs, tprs, fprs, clean_share = [], [], [], []
    for _ in range(reps):
        pos, neg = account_scores(rng, pop, m, kyc)
        clean_share.append(len(neg) / (len(pos) + len(neg)))
        # with enough pooling every account carries an adversary: no clean
        # accounts, so AUC and the clean flag rate are undefined that epoch
        if pos and neg:
            aucs.append(auc([z for z, _ in pos], [z for z, _ in neg]))
            fprs.append(sum(z > thr[size] for z, size in neg) / len(neg))
        if pos:
            tprs.append(sum(z > thr[size] for z, size in pos) / len(pos))
    return {
        "operators_per_router": m,
        "kyc_passthrough": kyc,
        "adversary_mean_p": pop.adversary_mean,
        "clean_account_share": _mean(clean_share),
        "auc": _mean(aucs),
        "adversary_account_flag_rate": _mean(tprs),
        "clean_account_flag_rate": _mean(fprs),
    }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def ban_cell(pop: Population, m: int, policy: str, fpr: float, horizon: int,
             reroute_adv: int, reroute_ben: int, reps: int, seed: int) -> dict:
    """Epoch loop with account bans. Operators hit by a ban are offline until they re-route."""
    rng = random.Random(seed)
    thresholds: dict = {}

    def benign_threshold_for(size: int) -> float:
        # the last account of a partition can be smaller than m
        if size not in thresholds:
            thresholds[size] = benign_threshold(rng, pop, size, fpr, samples=1000)
        return float(thresholds[size])

    n = pop.n_operators
    adversaries = set(range(pop.n_adversaries))
    adv_epochs = ben_lost = 0
    for _ in range(reps):
        offline_until = [0] * n
        for epoch in range(horizon):
            online = [o for o in range(n) if offline_until[o] <= epoch]
            adv_epochs += sum(1 for o in online if o in adversaries)
            if policy != "ban_account":
                continue
            for account in partition(online, m, rng):
                tox = sum(operator_toxicity(rng, pop, o in adversaries) for o in account) / len(account)
                if z_score(pop, tox, len(account)) > benign_threshold_for(len(account)):
                    for o in account:
                        delay = reroute_adv if o in adversaries else reroute_ben
                        offline_until[o] = epoch + 1 + delay
                        if o not in adversaries:
                            ben_lost += min(delay, horizon - epoch - 1)
    max_adv = pop.n_adversaries * horizon * reps
    return {
        "operators_per_router": m,
        "ban_policy": policy,
        "adversary_persistence": adv_epochs / max_adv,
        "benign_epochs_lost_per_rep": ben_lost / reps,
        "adversary_epochs_removed_per_rep": (max_adv - adv_epochs) / reps,
    }


def population_from(cfg: dict) -> Population:
    p = cfg["population"]
    return Population(
        n_operators=p["n_operators"],
        adversary_share=p["adversary_share"],
        interactions_per_epoch=p["interactions_per_epoch"],
        benign_mean=p["benign_p"]["mean"],
        benign_conc=p["benign_p"]["concentration"],
        adversary_mean=p["adversary_p"]["mean"],
        adversary_conc=p["adversary_p"]["concentration"],
    )


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output", default="runs/")
    ap.add_argument("--quick", action="store_true", help="20 reps, m in {1, 20}")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    pop = population_from(cfg)
    seed, reps = cfg["seed"], (20 if args.quick else cfg["reps"])
    fpr = cfg["detection"]["false_positive_rate"]
    sweep, bans = cfg["sweep"], cfg["bans"]
    ms = [1, 20] if args.quick else sweep["operators_per_router"]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output) / f"{ts}_router_intermediary"
    out.mkdir(parents=True, exist_ok=True)

    # lever 1: pooling ratio (no KYC)
    det = [detection_cell(pop, m, 0.0, fpr, reps, seed) for m in ms]
    write_csv(out / "detection.csv", det)
    print(f"n={pop.n_operators}, adversaries={pop.n_adversaries}, fpr={fpr}, reps={reps}")
    print(f"{'ops/router':>10} {'clean accts':>12} {'AUC':>7} {'adv acct flagged':>17} {'clean flagged':>14}")
    for r in det:
        print(f"{r['operators_per_router']:>10} {r['clean_account_share']:12.3f} {r['auc']:7.3f} "
              f"{r['adversary_account_flag_rate']:17.3f} {r['clean_account_flag_rate']:14.3f}")

    # lever 2: ban policy, per pooling ratio
    ban = [
        ban_cell(pop, m, policy, fpr, bans["horizon_epochs"], bans["reroute_epochs_adversary"],
                 bans["reroute_epochs_benign"], max(1, reps // 10), seed)
        for m in ms for policy in sweep["ban_policy"]
    ]
    write_csv(out / "bans.csv", ban)
    print(f"\n{'ops/router':>10} {'policy':>12} {'adv persistence':>16} {'benign lost':>12} {'adv removed':>12} {'lost/removed':>13}")
    for r in ban:
        removed = r["adversary_epochs_removed_per_rep"]
        ratio = r["benign_epochs_lost_per_rep"] / removed if removed else float("nan")
        print(f"{r['operators_per_router']:>10} {r['ban_policy']:>12} {r['adversary_persistence']:16.3f} "
              f"{r['benign_epochs_lost_per_rep']:12.1f} {removed:12.1f} {ratio:13.2f}")

    # lever 3: KYC pass-through at the largest pooling ratio
    m_big = ms[-1]
    kyc = [detection_cell(pop, m_big, k, fpr, reps, seed) for k in sweep["kyc_passthrough"]]
    write_csv(out / "kyc.csv", kyc)
    print(f"\nKYC pass-through at {m_big} operators/router")
    print(f"{'kyc':>5} {'AUC':>7} {'adv acct flagged':>17} {'clean flagged':>14}")
    for r in kyc:
        print(f"{r['kyc_passthrough']:5.2f} {r['auc']:7.3f} {r['adversary_account_flag_rate']:17.3f} "
              f"{r['clean_account_flag_rate']:14.3f}")

    # sensitivity (not a lever): how subtle the adversary is, at 1 and m_big per account
    sens = [
        detection_cell(replace(pop, adversary_mean=a), m, 0.0, fpr, reps, seed)
        for a in cfg["sensitivity"]["adversary_mean_p"] for m in sorted({1, m_big})
    ]
    write_csv(out / "sensitivity.csv", sens)
    print("\nSensitivity: adversary mean p (benign 0.80)")
    print(f"{'adv p':>6} {'ops/router':>10} {'AUC':>7} {'adv acct flagged':>17}")
    for r in sens:
        print(f"{r['adversary_mean_p']:6.2f} {r['operators_per_router']:>10} {r['auc']:7.3f} "
              f"{r['adversary_account_flag_rate']:17.3f}")

    falsifier = next((r for r in det if r["operators_per_router"] == 20), None)
    summary = {
        "falsifier_auc_at_20_per_router": falsifier["auc"] if falsifier else None,
        "falsifier_fires": bool(falsifier and falsifier["auc"] > 0.8),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfalsifier (AUC > 0.8 at 20/router): {summary}")
    print(f"artifacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
