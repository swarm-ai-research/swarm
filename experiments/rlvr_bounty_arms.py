"""RLVR bug-bounty arms: does paying for reports beat punishing exploits?

Runs the five governance arms in ``scenarios/rlvr_bounty/`` across seeds and
reports what each buys, testing the proposal in *RLVR that rewards red
teaming the training environment* (LessWrong ``T2bzBkJuBeNNgzhbh``).

Headline metrics per arm:

``report_share``
    Valid reports / (valid reports + landed exploits) — the behavioural
    question: did the hunters sell the defect or use it?
``patch_fraction``
    Share of the four proxy defects actually closed. Deterrence and
    patching are different achievements; this is the one that compounds.
``proxy_toxicity``
    ``E[1 - p | accepted]`` — what the environment's own metric reports.
``latent_harm``
    Share of accepted interactions whose recorded ground truth is ``-1``.
``hack_gap``
    ``latent_harm - proxy_toxicity``. The reward-hacking signature: how
    much harm the proxy is not seeing. A successful exploit *raises* p, so
    this gap is the quantity a bounty is supposed to shrink.

``--sweep`` instead scans the bounty amount to locate the operating window:
too low and reporting does not outbid exploitation, too high and
fabricating reports outbids honest work. The width of that window is set by
the report verifier, which is what the proposal names as the hard part.

Usage:
    python -m experiments.rlvr_bounty_arms [--seeds N] [--quick]
    python -m experiments.rlvr_bounty_arms --sweep [--seeds N]

Writes runs/<ts>_rlvr_bounty/{arms,sweep}.csv plus summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from swarm.agents.bug_hunter import BugHunterAgent
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction
from swarm.scenarios.loader import build_orchestrator, load_scenario

ARMS = [
    "laissez_faire",
    "punish_proxy",
    "punish_certificate",
    "bounty",
    "bounty_punish",
]
SCENARIO_DIR = Path("scenarios/rlvr_bounty")


@dataclass
class ArmResult:
    """One (arm, seed) run, flattened for CSV."""

    arm: str
    seed: int
    report_share: float = 0.0
    patch_fraction: float = 0.0
    exploits_landed: float = 0.0
    reports_valid: float = 0.0
    finds_withheld: float = 0.0
    proxy_toxicity: float = 0.0
    latent_harm: float = 0.0
    hack_gap: float = 0.0
    quality_gap: float = 0.0
    hunter_payoff: float = 0.0
    honest_payoff: float = 0.0
    bounty_paid: float = 0.0
    bounty_farmed: float = 0.0
    penalties_levied: float = 0.0
    stall_epochs: float = 0.0
    p_detect: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


def _latent_harm(interactions: List[SoftInteraction]) -> Optional[float]:
    """Share of accepted interactions recorded as harmful in ground truth.

    Only interactions that carry a latent label count — an environment
    where nothing is certified has no measurable latent harm, which is
    itself the point of the metric.
    """
    labelled = [
        i for i in interactions if i.accepted and i.ground_truth is not None
    ]
    if not labelled:
        return None
    return sum(1 for i in labelled if i.ground_truth == -1) / len(labelled)


def run_arm(
    arm: str,
    seed: int,
    *,
    quick: bool = False,
    governance_overrides: Optional[Dict[str, object]] = None,
) -> ArmResult:
    """Run one arm at one seed and collect its metrics."""
    scenario = load_scenario(SCENARIO_DIR / f"{arm}.yaml")
    scenario.orchestrator_config.seed = seed
    if quick:
        scenario.orchestrator_config.n_epochs = 10

    gov = scenario.orchestrator_config.governance_config
    for key, value in (governance_overrides or {}).items():
        setattr(gov, key, value)

    orchestrator = build_orchestrator(scenario)

    collected: List[SoftInteraction] = []
    payoffs: Dict[str, float] = {}

    def _on_complete(interaction, payoff_a, payoff_b):
        collected.append(interaction)
        payoffs[interaction.initiator] = (
            payoffs.get(interaction.initiator, 0.0) + payoff_a
        )
        payoffs[interaction.counterparty] = (
            payoffs.get(interaction.counterparty, 0.0) + payoff_b
        )

    orchestrator.on_interaction_complete(_on_complete)
    orchestrator.run()

    hunters = [
        a for a in orchestrator.get_all_agents() if isinstance(a, BugHunterAgent)
    ]
    registry = hunters[0].registry if hunters else None
    summary = registry.summary() if registry else {}

    metrics = SoftMetrics()
    accepted = [i for i in collected if i.accepted]
    proxy_toxicity = metrics.toxicity_rate(collected)
    latent = _latent_harm(collected)

    hunter_ids = {h.agent_id for h in hunters}
    hunter_payoff = sum(v for k, v in payoffs.items() if k in hunter_ids)
    honest_payoff = sum(v for k, v in payoffs.items() if k not in hunter_ids)

    result = ArmResult(
        arm=arm,
        seed=seed,
        report_share=summary.get("report_share", 0.0),
        patch_fraction=summary.get("patch_fraction", 0.0),
        exploits_landed=summary.get("exploits_landed", 0.0),
        reports_valid=summary.get("reports_valid", 0.0),
        finds_withheld=float(sum(h.finds_withheld for h in hunters)),
        proxy_toxicity=proxy_toxicity,
        latent_harm=latent if latent is not None else float("nan"),
        hack_gap=(latent - proxy_toxicity) if latent is not None else float("nan"),
        quality_gap=metrics.quality_gap(collected),
        hunter_payoff=hunter_payoff,
        honest_payoff=honest_payoff,
        bounty_paid=summary.get("bounty_paid", 0.0),
        bounty_farmed=summary.get("bounty_farmed", 0.0),
        penalties_levied=summary.get("penalties_levied", 0.0),
        stall_epochs=summary.get("stall_epochs", 0.0),
        p_detect=(
            statistics.fmean(
                h.get_strategy_report()["p_detect"] for h in hunters
            )
            if hunters
            else 0.0
        ),
        extra={
            "accepted_interactions": float(len(accepted)),
            "total_interactions": float(len(collected)),
            "fabrications": float(sum(h.fabrications_chosen for h in hunters)),
            "reports_submitted": summary.get("reports_submitted", 0.0),
            "reports_fabricated": summary.get("reports_fabricated", 0.0),
        },
    )
    return result


_FIELDS = [
    "arm",
    "seed",
    "report_share",
    "patch_fraction",
    "exploits_landed",
    "reports_valid",
    "finds_withheld",
    "proxy_toxicity",
    "latent_harm",
    "hack_gap",
    "quality_gap",
    "hunter_payoff",
    "honest_payoff",
    "bounty_paid",
    "bounty_farmed",
    "penalties_levied",
    "stall_epochs",
    "p_detect",
]


def _row(result: ArmResult) -> Dict[str, object]:
    row: Dict[str, object] = {f: getattr(result, f) for f in _FIELDS}
    row.update(result.extra)
    return row


def _mean(results: List[ArmResult], field_name: str) -> float:
    values = [
        getattr(r, field_name)
        for r in results
        if getattr(r, field_name) == getattr(r, field_name)  # drop NaN
    ]
    return statistics.fmean(values) if values else float("nan")


def run_arms(seeds: List[int], quick: bool) -> List[ArmResult]:
    results: List[ArmResult] = []
    for arm in ARMS:
        for seed in seeds:
            results.append(run_arm(arm, seed, quick=quick))
    return results


def run_sweep(seeds: List[int], quick: bool) -> List[ArmResult]:
    """Scan the bounty amount on the bounty-only arm."""
    amounts = [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    results: List[ArmResult] = []
    for amount in amounts:
        for seed in seeds:
            result = run_arm(
                "bounty",
                seed,
                quick=quick,
                governance_overrides={"bug_bounty_amount": amount},
            )
            result.extra["bounty_amount"] = amount
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds")
    parser.add_argument("--quick", action="store_true", help="10 epochs, not 30")
    parser.add_argument("--sweep", action="store_true", help="sweep bounty amount")
    args = parser.parse_args()

    seeds = [42 + i for i in range(args.seeds)]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("runs") / f"{timestamp}_rlvr_bounty"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        results = run_sweep(seeds, args.quick)
        name = "sweep"
    else:
        results = run_arms(seeds, args.quick)
        name = "arms"

    rows = [_row(r) for r in results]
    fieldnames = list(rows[0].keys())
    with (out_dir / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate by arm (or by bounty amount, in sweep mode)
    def key_of(r: ArmResult) -> str:
        return (
            f"{r.extra['bounty_amount']:g}" if args.sweep else r.arm
        )

    grouped: Dict[str, List[ArmResult]] = {}
    for result in results:
        grouped.setdefault(key_of(result), []).append(result)

    summary = {
        key: {
            field_name: _mean(group, field_name)
            for field_name in _FIELDS
            if field_name not in ("arm", "seed")
        }
        for key, group in grouped.items()
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    header = "bounty" if args.sweep else "arm"
    print(
        f"{header:<20} {'report%':>8} {'patched':>8} {'exploits':>9} "
        f"{'withheld':>9} {'proxy_tox':>10} {'latent':>8} {'hack_gap':>9} "
        f"{'farmed':>8} {'stall':>6}"
    )
    for key, stats in summary.items():
        print(
            f"{key:<20} {stats['report_share']:>8.2f} "
            f"{stats['patch_fraction']:>8.2f} {stats['exploits_landed']:>9.0f} "
            f"{stats['finds_withheld']:>9.0f} {stats['proxy_toxicity']:>10.3f} "
            f"{stats['latent_harm']:>8.3f} {stats['hack_gap']:>9.3f} "
            f"{stats['bounty_farmed']:>8.1f} {stats['stall_epochs']:>6.0f}"
        )
    print(f"\nWrote {out_dir}/{name}.csv and summary.json")


if __name__ == "__main__":
    main()
