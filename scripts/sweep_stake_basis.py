#!/usr/bin/env python3
"""Does the participation stake ever bind? (bead p70u)

Interaction payoffs land in ``AgentState.total_payoff`` and never in
``resources``, so with the shipped defaults every agent sits at its starting
balance and ``min_stake_to_participate`` is a constant gate. Three arms on
scenarios/contract_screening.yaml (staking on, stake 5.0, honest/opportunistic/
deceptive mix):

    control      today's behaviour
    payoff_res   payoff_flows_to_resources=True   (resources move with earnings)
    cum_payoff   stake_basis="cumulative_payoff"  (gate reads earnings)

Reported per arm: how often the gate actually blocked an agent, and whether it
blocked the low-quality ones, alongside welfare/gap/toxicity so a stake that
works can be told from one that just costs.

Usage:
    python scripts/sweep_stake_basis.py --seeds 10
    python scripts/sweep_stake_basis.py --seeds 3 --out DIR
"""

import argparse
import copy
import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from swarm.analysis.sweep import _extract_results  # noqa: E402
from swarm.governance.admission import StakingLever  # noqa: E402
from swarm.scenarios import build_orchestrator, load_scenario  # noqa: E402

SCENARIO_PATH = PROJECT_ROOT / "scenarios" / "contract_screening.yaml"
SEED_BASE = 42
LOW_QUALITY_TYPES = ("deceptive", "opportunistic")


def _gov(scenario: Any, **fields: Any) -> None:
    gc = scenario.orchestrator_config.governance_config
    for key, value in fields.items():
        setattr(gc, key, value)


ARMS: Dict[str, Callable[[Any], None]] = {
    "control": lambda s: None,
    "payoff_res": lambda s: _gov(s, payoff_flows_to_resources=True),
    "cum_payoff": lambda s: _gov(s, stake_basis="cumulative_payoff"),
}


def _agent_type(agent_id: str) -> str:
    return agent_id.rsplit("_", 1)[0]


def run_one(
    base: Any,
    arm: str,
    seed: int,
    stake: float,
    hard_audit: bool = False,
    audit: bool = False,
    slash: bool = False,
) -> Dict[str, Any]:
    scenario = copy.deepcopy(base)
    ARMS[arm](scenario)
    _gov(scenario, min_stake_to_participate=stake)
    if hard_audit:
        # A downward force on low-quality agents, so the basis change has
        # something to bite on. Before beads-ms0f wired slashing, audit
        # penalties were the only way a balance could fall.
        _gov(
            scenario,
            audit_enabled=True,
            audit_probability=1.0,
            audit_penalty_multiplier=8.0,
            audit_threshold_p=0.6,
        )
    elif audit:
        # Ordinary audit settings: a detection channel that catches some of
        # the low-quality interactions rather than all of them.
        _gov(
            scenario,
            audit_enabled=True,
            audit_probability=0.3,
            audit_penalty_multiplier=2.0,
            audit_threshold_p=0.5,
        )
    if slash:
        _gov(scenario, slash_on_detection=True)
    scenario.orchestrator_config.seed = seed
    orch = build_orchestrator(scenario)

    # Count every time the staking gate turns an agent away, by agent type.
    blocks: Counter = Counter()
    checks: Counter = Counter()
    engine = orch.governance_engine
    lever = (
        next((lv for lv in engine._levers if isinstance(lv, StakingLever)), None)
        if engine is not None
        else None
    )
    if lever is not None:
        inner = lever.can_agent_act

        def counting(agent_id: str, state: Any) -> bool:
            allowed = inner(agent_id, state)
            checks[_agent_type(agent_id)] += 1
            if not allowed:
                blocks[_agent_type(agent_id)] += 1
            return allowed

        lever.can_agent_act = counting  # type: ignore[method-assign]

    orch.run()
    result = _extract_results(orch, {"arm": arm}, 0, seed).to_dict()

    stake = scenario.orchestrator_config.governance_config.min_stake_to_participate
    basis_by_type: Dict[str, List[float]] = {}
    slashed_by_type: Dict[str, float] = {}
    below_bar: Counter = Counter()
    for agent_id, agent_state in orch.state.agents.items():
        kind = _agent_type(agent_id)
        balance = (
            lever.stake_balance(agent_state)
            if lever is not None
            else agent_state.resources
        )
        basis_by_type.setdefault(kind, []).append(balance)
        slashed_by_type[kind] = slashed_by_type.get(kind, 0.0) + agent_state.stake_slashed
        if balance < stake:
            below_bar[kind] += 1

    low_blocks = sum(blocks[t] for t in LOW_QUALITY_TYPES)
    return {
        "arm": arm,
        "stake": stake,
        "seed": seed,
        "avg_quality_gap": result["avg_quality_gap"],
        "avg_toxicity": result["avg_toxicity"],
        "total_welfare": result["total_welfare"],
        "accepted_interactions": result["accepted_interactions"],
        "gate_checks": sum(checks.values()),
        "gate_blocks": sum(blocks.values()),
        "gate_blocks_low_quality": low_blocks,
        "gate_blocks_honest": blocks["honest"],
        "slashed_total": sum(slashed_by_type.values()),
        "slashed_low_quality": sum(slashed_by_type.get(t, 0.0) for t in LOW_QUALITY_TYPES),
        "agents_below_bar": sum(below_bar.values()),
        "agents_below_bar_low_quality": sum(below_bar[t] for t in LOW_QUALITY_TYPES),
        "spread_basis": (
            max(v for vals in basis_by_type.values() for v in vals)
            - min(v for vals in basis_by_type.values() for v in vals)
        ),
        **{
            f"mean_basis_{kind}": statistics.mean(vals)
            for kind, vals in sorted(basis_by_type.items())
        },
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--stakes",
        default="5",
        help="comma-separated min_stake_to_participate levels to sweep",
    )
    ap.add_argument(
        "--hard-audit",
        action="store_true",
        help="certain audits with a heavy penalty, so balances can fall",
    )
    ap.add_argument(
        "--audit",
        action="store_true",
        help="ordinary audit settings (p=0.3, x2 penalty, threshold 0.5)",
    )
    ap.add_argument(
        "--slash",
        action="store_true",
        help="slash the stake of a detected agent (beads-ms0f)",
    )
    ap.add_argument("--out", type=Path, default=None)
    opts = ap.parse_args(argv)

    base = load_scenario(SCENARIO_PATH)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = opts.out or PROJECT_ROOT / "runs" / f"{stamp}_stake_basis_sweep"
    out.mkdir(parents=True, exist_ok=True)

    stakes = [float(s) for s in opts.stakes.split(",")]
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for stake in stakes:
            for offset in range(opts.seeds):
                rows.append(
                    run_one(
                        base,
                        arm,
                        SEED_BASE + offset,
                        stake,
                        opts.hard_audit,
                        opts.audit,
                        opts.slash,
                    )
                )
                row = rows[-1]
                print(
                    f"{arm:11s} stake={stake:6.1f} seed={row['seed']} "
                    f"blocks={row['gate_blocks']:4d} "
                    f"(low-q {row['gate_blocks_low_quality']:4d}) "
                    f"gap={row['avg_quality_gap']:+.4f} "
                    f"welfare={row['total_welfare']:.1f}",
                    flush=True,
                )

    with open(out / "sweep.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    keys = [k for k in rows[0] if k not in ("arm", "seed", "stake")]
    summary: Dict[str, Any] = {
        "scenario": str(SCENARIO_PATH.relative_to(PROJECT_ROOT)),
        "hard_audit": opts.hard_audit,
        "audit": opts.audit,
        "slash_on_detection": opts.slash,
        "seeds": [SEED_BASE + i for i in range(opts.seeds)],
        "arms": {},
    }
    for arm in ARMS:
        for stake in stakes:
            arm_rows = [r for r in rows if r["arm"] == arm and r["stake"] == stake]
            summary["arms"][f"{arm}@{stake:g}"] = {
                key: {
                    "mean": statistics.mean(float(r[key]) for r in arm_rows),
                    "sd": (
                        statistics.stdev([float(r[key]) for r in arm_rows])
                        if len(arm_rows) > 1
                        else 0.0
                    ),
                }
                for key in keys
                if all(key in r for r in arm_rows)
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"\n{'arm@stake':16s} {'blocks':>8s} {'low-q':>8s} {'below bar':>10s} "
        f"{'gap':>10s} {'welfare':>10s} {'spread':>9s}"
    )
    for arm, cell in summary["arms"].items():
        print(
            f"{arm:16s} {cell['gate_blocks']['mean']:8.1f} "
            f"{cell['gate_blocks_low_quality']['mean']:8.1f} "
            f"{cell['agents_below_bar']['mean']:10.1f} "
            f"{cell['avg_quality_gap']['mean']:+10.4f} "
            f"{cell['total_welfare']['mean']:10.1f} "
            f"{cell['spread_basis']['mean']:9.2f}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
