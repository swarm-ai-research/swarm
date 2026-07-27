"""Fitness-landscape probe for the A-Evolve bridge (beads-2qgp, recon for 3o9l).

Before an evolution track can claim meaningful results, the benchmark it
evolves against must have *gradient*: distinct candidate configurations must
produce distinct fitness. This script measures that directly on the
``aevolve_screening`` scenario, for both evolution directions:

- **Defender** (ni9b): the evolved candidate designs governance. Probe several
  governance phenotypes through ``SwarmBenchmarkAdapter`` and compare scores.
- **Attacker** (3o9l): the evolved candidate is an ``AdaptiveAdversary``. Probe
  several behavioral-knob phenotypes (exploration/learning/heat) and compare
  the toxicity / payoff they induce.

Finding (2026-07-27): both landscapes are effectively flat where they matter.
On the defender side, ``staking_enabled`` / ``min_stake_to_participate`` /
``audit_enabled`` / ``circuit_breaker_enabled`` / ``payoff.theta`` move the
score by 0.0; only ``transaction_tax`` and ``rho_a/rho_b`` move welfare, and
the default ``one_minus_toxicity`` score barely reflects even those. On the
attacker side, *no* ``AdaptiveAdversary`` knob moves toxicity, payoff, or
welfare at all — because ``p`` is a function of agent *type/count* (0 vs 4
deceptive: toxicity 0.23 -> 0.37) rather than of the agent's learned actions.
An evolution engine over either knob set would be optimizing against noise.

Run:
    python experiments/aevolve_landscape_probe.py            # both probes
    python experiments/aevolve_landscape_probe.py --json     # machine-readable

Exit code is 0 always; the gradient verdict is in the ``flat`` flags. The
seeded runs are deterministic, so the printed table is the reproducible
evidence attached to beads-2qgp.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow ``python experiments/aevolve_landscape_probe.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

SCENARIO_PATH = "scenarios/aevolve_screening.yaml"

# A landscape is "flat" for a signal if the spread of per-phenotype means is
# below this — i.e. no phenotype is distinguishable from another by fitness.
FLAT_EPS = 1e-3


# ---------------------------------------------------------------------------
# Defender: governance phenotypes through the benchmark adapter
# ---------------------------------------------------------------------------

DEFENDER_PHENOTYPES: Dict[str, Dict[str, Any]] = {
    "null": {},
    "tax_heavy": {"governance.transaction_tax_rate": 0.3},
    "stake": {"governance.staking_enabled": True,
              "governance.min_stake_to_participate": 5.0},
    "audit": {"governance.audit_enabled": True},
    "breaker": {"governance.circuit_breaker_enabled": True},
    "rho": {"payoff.rho_a": 0.9, "payoff.rho_b": 0.9},
    "theta": {"payoff.theta": 0.9},
}


def probe_defender(n_seeds: int = 5) -> List[Dict[str, Any]]:
    import json as _json

    from swarm.bridges.aevolve.adapter import SwarmBenchmarkAdapter, Trajectory
    from swarm.bridges.aevolve.config import AevolveBridgeConfig

    adapter = SwarmBenchmarkAdapter(AevolveBridgeConfig(n_tasks=n_seeds))
    tasks = adapter.get_tasks(limit=n_seeds)
    rows: List[Dict[str, Any]] = []
    for name, pheno in DEFENDER_PHENOTYPES.items():
        scores, toxes, welfares = [], [], []
        for task in tasks:
            fb = adapter.evaluate(
                task, Trajectory(task_id=task.id, output=_json.dumps(pheno))
            )
            scores.append(fb.score)
            toxes.append(fb.raw["toxicity"])
            welfares.append(fb.raw["total_welfare"])
        rows.append({
            "phenotype": name,
            "score": statistics.mean(scores),
            "toxicity": statistics.mean(toxes),
            "welfare": statistics.mean(welfares),
        })
    return rows


# ---------------------------------------------------------------------------
# Attacker: AdaptiveAdversary knob phenotypes against fixed weak governance
# ---------------------------------------------------------------------------

ATTACKER_PHENOTYPES: Dict[str, Dict[str, Any]] = {
    "passive": {"exploration_rate": 0.0, "learning_rate": 0.0, "heat_threshold": 1.0},
    "default": {},
    "aggressive": {"exploration_rate": 0.5, "learning_rate": 0.5, "heat_threshold": 0.9},
    "stealth": {"exploration_rate": 0.1, "learning_rate": 0.3, "heat_threshold": 0.2},
}

# Contrast row: vary the agent *mix* to show the signal DOES move on type.
MIX_CONTRAST: Dict[str, List[Dict[str, Any]]] = {
    "mix:0_deceptive": [{"type": "honest", "count": 6}],
    "mix:4_deceptive": [{"type": "honest", "count": 1},
                        {"type": "opportunistic", "count": 1},
                        {"type": "deceptive", "count": 4}],
}


def _weak_governance_base() -> Dict[str, Any]:
    with open(SCENARIO_PATH, encoding="utf-8") as fh:
        base: Dict[str, Any] = yaml.safe_load(fh)
    base["governance"] = {
        "transaction_tax_rate": 0.0, "staking_enabled": False,
        "audit_enabled": False, "circuit_breaker_enabled": False,
        "reputation_decay_rate": 1.0, "bandwidth_cap": 10,
    }
    base["simulation"] = {"n_epochs": 8, "steps_per_epoch": 12, "seed": 1000}
    base.pop("outputs", None)
    return base


def _run_raw(raw: Dict[str, Any], seed: int) -> Tuple[float, float, float]:
    from swarm.scenarios import build_orchestrator, load_scenario

    doc = copy.deepcopy(raw)
    doc["simulation"]["seed"] = seed
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as fh:
        yaml.safe_dump(doc, fh)
        path = Path(fh.name)
    try:
        history = build_orchestrator(load_scenario(path)).run()
    finally:
        path.unlink(missing_ok=True)
    final = history[-1]
    return (
        float(final.toxicity_rate),
        float(final.avg_payoff),
        float(final.total_welfare),
    )


def probe_attacker(n_seeds: int = 5) -> List[Dict[str, Any]]:
    base = _weak_governance_base()
    rows: List[Dict[str, Any]] = []
    for name, cfg in ATTACKER_PHENOTYPES.items():
        raw = copy.deepcopy(base)
        raw["agents"] = [
            {"type": "honest", "count": 3},
            {"type": "opportunistic", "count": 2},
            {"type": "adaptive_adversary", "count": 1, "config": cfg},
        ]
        results = [_run_raw(raw, 1000 + s) for s in range(n_seeds)]
        tox, pay, wel = zip(*results, strict=True)
        rows.append({
            "phenotype": name, "toxicity": statistics.mean(tox),
            "payoff": statistics.mean(pay), "welfare": statistics.mean(wel),
        })
    # Mix contrast — the control that proves the harness CAN move.
    for name, agents in MIX_CONTRAST.items():
        raw = copy.deepcopy(base)
        raw["agents"] = agents
        results = [_run_raw(raw, 1000 + s) for s in range(n_seeds)]
        tox, pay, wel = zip(*results, strict=True)
        rows.append({
            "phenotype": name, "toxicity": statistics.mean(tox),
            "payoff": statistics.mean(pay), "welfare": statistics.mean(wel),
        })
    return rows


# ---------------------------------------------------------------------------
# Verdict + reporting
# ---------------------------------------------------------------------------

def spread(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows]
    return float(max(vals) - min(vals))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", action="store_true", dest="as_json")
    opts = ap.parse_args(argv)

    defender = probe_defender(opts.seeds)
    attacker = probe_attacker(opts.seeds)

    # Attacker knob rows only (exclude mix contrast) for the flat verdict.
    knob_rows = [r for r in attacker if not r["phenotype"].startswith("mix:")]
    verdict = {
        "defender_score_spread": spread(defender, "score"),
        "defender_flat": spread(defender, "score") < FLAT_EPS,
        "attacker_toxicity_spread": spread(knob_rows, "toxicity"),
        "attacker_flat": spread(knob_rows, "toxicity") < FLAT_EPS,
        "mix_toxicity_spread": spread(attacker, "toxicity"),
    }

    if opts.as_json:
        print(json.dumps(
            {"defender": defender, "attacker": attacker, "verdict": verdict},
            indent=2,
        ))
        return 0

    print("DEFENDER landscape (SwarmBenchmarkAdapter, governance phenotypes)")
    print(f"  {'phenotype':12s} {'score':>8s} {'toxicity':>9s} {'welfare':>9s}")
    for r in defender:
        print(f"  {r['phenotype']:12s} {r['score']:8.4f} "
              f"{r['toxicity']:9.4f} {r['welfare']:9.2f}")
    print(f"  -> score spread {verdict['defender_score_spread']:.4f} "
          f"({'FLAT' if verdict['defender_flat'] else 'has gradient'})\n")

    print("ATTACKER landscape (AdaptiveAdversary knobs, fixed weak governance)")
    print(f"  {'phenotype':16s} {'toxicity':>9s} {'payoff':>8s} {'welfare':>9s}")
    for r in attacker:
        print(f"  {r['phenotype']:16s} {r['toxicity']:9.4f} "
              f"{r['payoff']:+8.3f} {r['welfare']:9.2f}")
    print(f"  -> knob toxicity spread {verdict['attacker_toxicity_spread']:.4f} "
          f"({'FLAT' if verdict['attacker_flat'] else 'has gradient'})")
    print(f"  -> mix toxicity spread  {verdict['mix_toxicity_spread']:.4f} "
          f"(control: harness moves on agent type)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
