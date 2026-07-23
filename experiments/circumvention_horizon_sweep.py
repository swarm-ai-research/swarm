"""Horizon and persistence sweep for constraint circumvention (beads mtht).

Tests the three predictions from the long-horizon safety note
(docs/research/long-horizon-safety-lessons.md §1,§4):

  1. action-level catch rate DEGRADES as horizon grows — with heterogeneous
     task suspicion the cliff at M/tau smooths into a degradation curve, and
     it is non-monotone below the threshold (more over-threshold steps
     compound detection before the decomposition drops clean);
  2. circumvention success RISES with persistence at fixed horizon
     (manufactured effective horizon);
  3. honest completion is FLAT across both axes.

Also reports mean cumulative suspicion of evaded attempts — the invariant a
trajectory monitor (beads 0hp7) will catch, previewing why action-level
coverage alone is insufficient.

Usage:
    python -m experiments.circumvention_horizon_sweep [--quick]

Writes runs/<ts>_circumvention_sweep/results.csv and prints two tables.
"""

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path

from swarm.redteam.circumvention import (
    CircumventionConfig,
    WorkerKind,
    action_filter_catch_rate,
    circumvention_success_rate,
    honest_completion_rate,
    mean_cumulative_suspicion_evaded,
    simulate_episode,
    simulate_task,
)

HORIZONS = [1, 2, 3, 4, 6, 8, 12, 16, 24]
PERSISTENCES = [0, 1, 2, 3, 4, 6]
SEEDS = [42, 7, 101, 2026, 555]
N_TASKS = 400
# Heterogeneous task suspicion so the horizon curve is smooth, not a cliff.
CONFIG = CircumventionConfig(suspicion_mass=1.0, mass_spread=0.6, tau=0.25)


def _circ_episode(horizon, persistence, seed):
    import random

    rng = random.Random(seed)
    return [
        simulate_task(
            WorkerKind.CIRCUMVENTION, horizon, persistence, CONFIG, rng,
            mass=CONFIG.draw_mass(rng),
        )
        for _ in range(N_TASKS)
    ]


def _mixed_episode(horizon, persistence, seed, instruction_decay):
    import random

    return simulate_episode(
        N_TASKS, horizon, persistence, CONFIG, random.Random(seed),
        instruction_decay=instruction_decay,
    )


def _agg(fn, horizon, persistence, seeds):
    vals = [fn(_circ_episode(horizon, persistence, s)) for s in seeds]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    horizons = [1, 3, 4, 8, 16] if args.quick else HORIZONS
    persistences = [0, 2, 4] if args.quick else PERSISTENCES
    seeds = SEEDS[:2] if args.quick else SEEDS

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / f"{ts}_circumvention_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Axis 1: horizon (persistence fixed at 0 — no manufactured horizon)
    print("HORIZON SWEEP (persistence=0)")
    print(f"{'H':>3} {'catch_rate':>16} {'success':>16} {'evaded_cum_susp':>16}")
    for H in horizons:
        c_m, c_s = _agg(action_filter_catch_rate, H, 0, seeds)
        s_m, s_s = _agg(circumvention_success_rate, H, 0, seeds)
        cum_m, _ = _agg(mean_cumulative_suspicion_evaded, H, 0, seeds)
        print(f"{H:>3} {f'{c_m:.3f}±{c_s:.3f}':>16} "
              f"{f'{s_m:.3f}±{s_s:.3f}':>16} {cum_m:>16.3f}")
        rows.append({"axis": "horizon", "horizon": H, "persistence": 0,
                     "catch_rate": c_m, "catch_sd": c_s,
                     "success_rate": s_m, "evaded_cum_susp": cum_m})

    # Axis 2: persistence (horizon fixed at 1 — worst starting horizon)
    print("\nPERSISTENCE SWEEP (horizon=1)")
    print(f"{'P':>3} {'catch_rate':>16} {'success':>16}")
    for P in persistences:
        c_m, c_s = _agg(action_filter_catch_rate, 1, P, seeds)
        s_m, s_s = _agg(circumvention_success_rate, 1, P, seeds)
        print(f"{P:>3} {f'{c_m:.3f}±{c_s:.3f}':>16} {f'{s_m:.3f}±{s_s:.3f}':>16}")
        rows.append({"axis": "persistence", "horizon": 1, "persistence": P,
                     "catch_rate": c_m, "catch_sd": c_s,
                     "success_rate": s_m, "evaded_cum_susp": float("nan")})

    # Honest baseline across the whole grid (must be flat at 1.0)
    import random

    honest = [
        simulate_task(WorkerKind.HONEST, H, P, CONFIG, random.Random(s))
        for H in horizons for P in persistences for s in seeds
    ]
    honest_rate = honest_completion_rate(honest)
    print(f"\nHonest completion across grid: {honest_rate:.3f} "
          f"(flat baseline; n={len(honest)})")

    # Instruction-decay: circumvention share should rise over the episode
    print("\nINSTRUCTION-DECAY (horizon=8, persistence=2)")
    print(f"{'decay':>6} {'circ_share':>12} {'catch_rate':>12}")
    for decay in ([0.0, 0.5, 1.0] if args.quick else [0.0, 0.25, 0.5, 0.75, 1.0]):
        shares, catches = [], []
        for s in seeds:
            ep = _mixed_episode(8, 2, s, decay)
            n_circ = sum(1 for o in ep if o.kind is WorkerKind.CIRCUMVENTION)
            shares.append(n_circ / len(ep))
            catches.append(action_filter_catch_rate(ep))
        print(f"{decay:>6.2f} {statistics.mean(shares):>12.3f} "
              f"{statistics.mean(catches):>12.3f}")
        rows.append({"axis": "instruction_decay", "horizon": 8, "persistence": 2,
                     "instruction_decay": decay,
                     "circ_share": statistics.mean(shares),
                     "catch_rate": statistics.mean(catches)})

    csv_path = out_dir / "results.csv"
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
