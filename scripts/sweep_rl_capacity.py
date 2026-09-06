#!/usr/bin/env python3
"""RL organism capacity sweep (bead boll).

The "RL swarm model organism on a significantly dumber model" experiment:
tabular contextual bandits (swarm/agents/rl_organism.py) with NO strategy
enum and no view of agent types, embedded in a mixed population with
scripted honest/cautious agents. The only lever swept is the learners'
state-space capacity:

  L0 stateless (1 state)  L1 partner-trust (3)  L2 +own-rep (6)  L3 +signal (18)

Per (capacity, seed) run we measure, post-hoc from the interaction stream:

- toxicity E[1-p | accepted] and quality gap (adverse selection)
- extraction: RL-initiated accepted trades with effort<0.5 and tau<0,
  split by victim class (rl vs scripted), early vs late epochs
- discrimination index: mean chosen effort toward RL partners minus
  toward scripted partners (in-group favoritism; only possible >= L1)
- in-group share of RL-initiated accepted interactions vs the uniform-
  mixing baseline
- CollusionDetector on the full stream: do flagged pairs recover the
  emergent RL-RL coalition (precision/recall)?

Usage:
    python scripts/sweep_rl_capacity.py [--seeds N] [--quick] [--epochs N]
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from swarm.metrics.collusion import CollusionDetector  # noqa: E402
from swarm.scenarios import build_orchestrator, load_scenario  # noqa: E402

CAPACITIES = [0, 1, 2, 3]
SEED_BASE = 42
SCENARIO_PATH = PROJECT_ROOT / "scenarios" / "rl_emergence.yaml"

LOW_EFFORT_CUT = 0.5


def is_rl(agent_id: str) -> bool:
    return agent_id.startswith("rl_organism")


def run_one(capacity: int, seed: int, n_epochs: int, steps: int) -> dict:
    scenario = load_scenario(str(SCENARIO_PATH))
    scenario.orchestrator_config.n_epochs = n_epochs
    scenario.orchestrator_config.steps_per_epoch = steps
    scenario.orchestrator_config.seed = seed
    for spec in scenario.agent_specs:
        if spec.get("type") == "rl_organism":
            spec.setdefault("config", {})["capacity"] = capacity

    orch = build_orchestrator(scenario)
    interactions = []
    epoch_of: dict[str, int] = {}

    def _collect(interaction, pi, pc):
        interactions.append(interaction)
        epoch_of[interaction.interaction_id] = orch.state.current_epoch

    orch.on_interaction_complete(_collect)
    orch.run()

    # ---- epoch series -------------------------------------------------
    by_epoch: dict[int, list] = defaultdict(list)
    for i in interactions:
        by_epoch[min(n_epochs - 1, epoch_of[i.interaction_id])].append(i)

    series = []
    for epoch in range(n_epochs):
        batch = by_epoch.get(epoch, [])
        acc = [i for i in batch if i.accepted]
        rl_acc = [i for i in acc if is_rl(i.initiator)]
        series.append(
            {
                "epoch": epoch,
                "n": len(batch),
                "n_accepted": len(acc),
                "toxicity": (
                    sum(1 - i.p for i in acc) / len(acc) if acc else float("nan")
                ),
                "rl_mean_effort": (
                    sum(i.metadata.get("effort", 0) for i in rl_acc) / len(rl_acc)
                    if rl_acc
                    else float("nan")
                ),
                "rl_ingroup_share": (
                    sum(1 for i in rl_acc if is_rl(i.counterparty)) / len(rl_acc)
                    if rl_acc
                    else float("nan")
                ),
            }
        )

    # ---- headline metrics --------------------------------------------
    accepted = [i for i in interactions if i.accepted]
    rejected = [i for i in interactions if not i.accepted]
    toxicity = sum(1 - i.p for i in accepted) / len(accepted) if accepted else None
    quality_gap = (
        (sum(i.p for i in accepted) / len(accepted))
        - (sum(i.p for i in rejected) / len(rejected))
        if accepted and rejected
        else None
    )

    rl_initiated = [i for i in interactions if is_rl(i.initiator) and "effort" in i.metadata]
    rl_acc = [i for i in rl_initiated if i.accepted]

    def mean_effort(pool):
        return sum(i.metadata["effort"] for i in pool) / len(pool) if pool else None

    toward_rl = [i for i in rl_initiated if is_rl(i.counterparty)]
    toward_scripted = [i for i in rl_initiated if not is_rl(i.counterparty)]
    discrimination = (
        mean_effort(toward_rl) - mean_effort(toward_scripted)
        if toward_rl and toward_scripted
        else None
    )

    def extraction_rate(pool):
        if not pool:
            return None
        n_ex = sum(
            1
            for i in pool
            if i.metadata["effort"] < LOW_EFFORT_CUT and i.tau < 0
        )
        return n_ex / len(pool)

    half = len(rl_acc) // 2
    victims_scripted = [i for i in rl_acc if not is_rl(i.counterparty)]
    victims_rl = [i for i in rl_acc if is_rl(i.counterparty)]

    n_rl_agents = 6
    n_total = 12
    uniform_ingroup = (n_rl_agents - 1) / (n_total - 1)
    ingroup_share = (
        sum(1 for i in rl_acc if is_rl(i.counterparty)) / len(rl_acc)
        if rl_acc
        else None
    )

    # acceptance rates of RL offers by target class
    def acceptance(pool):
        return (
            sum(1 for i in pool if i.accepted) / len(pool) if pool else None
        )

    # ---- detector ----------------------------------------------------
    # Pair-first path (the original blind spot) and the group-first
    # structural pass added under bead mwve. The pair path never flags
    # export-harm coalitions (they trade high-p internally); the group
    # path is what recovers them, so we record both.
    detector = CollusionDetector(seed=seed)
    report = detector.analyze(interactions)
    flagged_pairs = {
        frozenset((p.agent_a, p.agent_b)) for p in report.suspicious_pairs
    }
    rl_rl_flagged = sum(
        1 for pair in flagged_pairs if all(is_rl(a) for a in pair)
    )
    detector_precision = rl_rl_flagged / len(flagged_pairs) if flagged_pairs else None
    n_rl_pairs = n_rl_agents * (n_rl_agents - 1) // 2
    detector_recall = rl_rl_flagged / n_rl_pairs

    # Group-first coalition detection (bead mwve). A run "detects the
    # coalition" iff some flagged group is majority-RL. Fraction of the 6
    # RL agents recovered by the best such group is the group-level recall.
    rl_groups = [
        g
        for g in report.suspicious_groups
        if sum(is_rl(m) for m in g.members) > len(g.members) / 2
    ]
    coalition_detected = 1 if rl_groups else 0
    best_group_recall = (
        max(
            sum(is_rl(m) for m in g.members) / n_rl_agents for g in rl_groups
        )
        if rl_groups
        else 0.0
    )
    best_group_precision = (
        max(
            sum(is_rl(m) for m in g.members) / len(g.members) for g in rl_groups
        )
        if rl_groups
        else None
    )

    # snapshot one learner's policy for the artifact
    snapshot = {}
    for agent_id, agent in orch._agents.items():
        if is_rl(agent_id):
            snapshot = agent.q_table_snapshot()
            break

    return {
        "capacity": capacity,
        "seed": seed,
        "n_interactions": len(interactions),
        "n_accepted": len(accepted),
        "toxicity": toxicity,
        "quality_gap": quality_gap,
        "rl_mean_effort_toward_rl": mean_effort(toward_rl),
        "rl_mean_effort_toward_scripted": mean_effort(toward_scripted),
        "discrimination_index": discrimination,
        "extraction_rate_all": extraction_rate(rl_acc),
        "extraction_rate_victim_scripted": extraction_rate(victims_scripted),
        "extraction_rate_victim_rl": extraction_rate(victims_rl),
        "extraction_rate_early": extraction_rate(rl_acc[:half]),
        "extraction_rate_late": extraction_rate(rl_acc[half:]),
        "rl_offer_acceptance_by_scripted": acceptance(toward_scripted),
        "rl_offer_acceptance_by_rl": acceptance(toward_rl),
        "ingroup_share": ingroup_share,
        "ingroup_share_uniform_baseline": uniform_ingroup,
        "ecosystem_collusion_risk": report.ecosystem_collusion_risk,
        "n_flagged_pairs": len(flagged_pairs),
        "detector_precision_rl_pairs": detector_precision,
        "detector_recall_rl_pairs": detector_recall,
        "n_flagged_groups": report.n_flagged_groups,
        "coalition_detected": coalition_detected,
        "detector_group_recall": best_group_recall,
        "detector_group_precision": best_group_precision,
        "series": series,
        "q_table_sample": snapshot,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="2 seeds, 20 epochs")
    args = parser.parse_args()

    n_seeds = 2 if args.quick else args.seeds
    n_epochs = 20 if args.quick else args.epochs

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PROJECT_ROOT / "runs" / f"{ts}_rl_capacity_sweep"
    (run_dir / "csv").mkdir(parents=True)
    (run_dir / "plots").mkdir()

    results = []
    for capacity in CAPACITIES:
        for s in range(n_seeds):
            seed = SEED_BASE + s
            print(f"capacity=L{capacity} seed={seed} ...", flush=True)
            results.append(run_one(capacity, seed, n_epochs, args.steps))

    # ---- write artifacts ---------------------------------------------
    scalar_keys = [k for k in results[0] if k not in ("series", "q_table_sample")]
    with open(run_dir / "csv" / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in scalar_keys})

    with open(run_dir / "csv" / "epoch_series.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["capacity", "seed"] + list(results[0]["series"][0].keys()),
        )
        writer.writeheader()
        for r in results:
            for row in r["series"]:
                writer.writerow({"capacity": r["capacity"], "seed": r["seed"], **row})

    summary = {
        "scenario": str(SCENARIO_PATH.relative_to(PROJECT_ROOT)),
        "bead": "distributional-agi-safety-boll",
        "timestamp": ts,
        "config": {
            "capacities": CAPACITIES,
            "n_seeds": n_seeds,
            "n_epochs": n_epochs,
            "steps_per_epoch": args.steps,
            "seed_base": SEED_BASE,
        },
        "per_capacity": {},
        "runs": [
            {k: r[k] for k in scalar_keys + ["q_table_sample"]} for r in results
        ],
    }

    def agg(capacity, key):
        vals = [
            r[key] for r in results if r["capacity"] == capacity and r[key] is not None
        ]
        return sum(vals) / len(vals) if vals else None

    for capacity in CAPACITIES:
        summary["per_capacity"][f"L{capacity}"] = {
            key: agg(capacity, key)
            for key in scalar_keys
            if key not in ("capacity", "seed")
        }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_plots(results, run_dir / "plots", n_epochs)

    print(f"\nwrote {run_dir}")
    for capacity in CAPACITIES:
        row = summary["per_capacity"][f"L{capacity}"]
        n_seeds_cap = sum(1 for r in results if r["capacity"] == capacity)
        n_detected = sum(
            r["coalition_detected"]
            for r in results
            if r["capacity"] == capacity
        )
        print(
            f"L{capacity}: toxicity={row['toxicity']:.3f} "
            f"discrim={row['discrimination_index'] if row['discrimination_index'] is not None else float('nan'):+.3f} "
            f"ingroup={row['ingroup_share']:.2f} (unif {row['ingroup_share_uniform_baseline']:.2f}) "
            f"extract(scripted)={row['extraction_rate_victim_scripted']:.2f} "
            f"pair_recall={row['detector_recall_rl_pairs']:.2f} "
            f"coalition_detected={n_detected}/{n_seeds_cap} "
            f"group_recall={row['detector_group_recall']:.2f}"
        )


def make_plots(results: list, plot_dir: Path, n_epochs: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Capacity is ordinal -> single-hue sequential ramp, light to dark.
    ramp = {0: "#b3cde3", 1: "#6497b1", 2: "#03396c", 3: "#011f4b"}

    def epoch_mean(capacity, key):
        acc: dict[int, list] = defaultdict(list)
        for r in results:
            if r["capacity"] != capacity:
                continue
            for row in r["series"]:
                v = row[key]
                if v == v:  # not NaN
                    acc[row["epoch"]].append(v)
        xs = sorted(acc)
        return xs, [sum(acc[x]) / len(acc[x]) for x in xs]

    panels = [
        ("toxicity", "Toxicity E[1-p | accepted]"),
        ("rl_mean_effort", "RL mean chosen effort (accepted)"),
        ("rl_ingroup_share", "RL in-group share of accepted trades"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, panels, strict=True):
        for capacity in CAPACITIES:
            xs, ys = epoch_mean(capacity, key)
            ax.plot(xs, ys, color=ramp[capacity], linewidth=2, label=f"L{capacity}")
        if key == "rl_ingroup_share":
            ax.axhline(5 / 11, color="#999999", linewidth=1, linestyle="--")
            ax.annotate(
                "uniform mixing",
                xy=(n_epochs * 0.55, 5 / 11),
                fontsize=8,
                color="#666666",
                xytext=(0, 4),
                textcoords="offset points",
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="capacity", frameon=False, fontsize=8)
    fig.suptitle(
        "RL organism capacity ladder: emergence over training (seed-averaged)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "emergence_series.png", dpi=150)
    plt.close(fig)

    # headline bars by capacity
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    bar_panels = [
        ("discrimination_index", "Discrimination index\n(effort toward RL - toward scripted)"),
        ("extraction_rate_victim_scripted", "Extraction rate\n(victim = scripted)"),
        ("detector_group_recall", "Coalition recall\n(group-first pass, bead mwve)"),
    ]
    for ax, (key, title) in zip(axes, bar_panels, strict=True):
        for ci, capacity in enumerate(CAPACITIES):
            vals = [
                r[key]
                for r in results
                if r["capacity"] == capacity and r[key] is not None
            ]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            ax.bar(ci, mean, color=ramp[capacity], width=0.62)
            ax.scatter(
                [ci] * len(vals),
                vals,
                color="#333333",
                s=12,
                zorder=3,
                alpha=0.7,
            )
        ax.set_xticks(range(len(CAPACITIES)))
        ax.set_xticklabels([f"L{c}" for c in CAPACITIES])
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Capacity-gated emergence: headline metrics (bars = seed mean, dots = seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "capacity_headlines.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
