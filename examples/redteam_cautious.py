#!/usr/bin/env python
"""
Red-team the Cautious Reciprocator.

Two-phase evaluation:
  Phase 1: Run the RedTeamEvaluator (simulated attack outcomes) against
           the scenario's governance config.
  Phase 2: Run real orchestrator simulations with escalating adversary
           populations and adaptive adversary types to find where the
           cautious reciprocator actually breaks.

Usage:
    python examples/redteam_cautious.py
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.agents.adaptive_adversary import AdaptiveAdversary
from swarm.agents.adversarial import AdversarialAgent
from swarm.agents.cautious_reciprocator import CautiousReciprocator
from swarm.agents.deceptive import DeceptiveAgent
from swarm.agents.honest import HonestAgent
from swarm.agents.modeling_adversary import ModelingAdversary
from swarm.agents.threshold_dancer import ThresholdDancer
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig
from swarm.redteam import RedTeamEvaluator

# Adversary types used for payoff extraction across analysis
ADVERSARY_TYPES = ["adversarial", "deceptive", "adaptive", "modeling", "threshold"]


def make_governance():
    return GovernanceConfig(
        transaction_tax_rate=0.05,
        reputation_decay_rate=0.95,
        circuit_breaker_enabled=True,
        freeze_threshold_toxicity=0.8,
        audit_enabled=True,
        audit_probability=0.1,
    )


def make_payoff():
    return PayoffConfig(
        s_plus=2.0, s_minus=1.0, h=2.0,
        theta=0.5, rho_a=0.1, rho_b=0.1, w_rep=1.0,
    )


# ── Phase 2: real orchestrator attacks ───────────────────────────────

ATTACK_SCENARIOS = [
    {
        "name": "Baseline (original mix)",
        "cautious": 3, "honest": 2,
        "deceptive": 2, "adversarial": 2, "adaptive": 1, "modeling": 0,
    },
    {
        "name": "Adversary majority",
        "cautious": 3, "honest": 2,
        "deceptive": 3, "adversarial": 3, "adaptive": 2, "modeling": 0,
    },
    {
        "name": "Modeling adversary focus",
        "cautious": 3, "honest": 2,
        "deceptive": 1, "adversarial": 1, "adaptive": 1, "modeling": 3,
    },
    {
        "name": "Deceptive swarm",
        "cautious": 3, "honest": 2,
        "deceptive": 6, "adversarial": 0, "adaptive": 0, "modeling": 0,
    },
    {
        "name": "Solo cautious vs all adversaries",
        "cautious": 1, "honest": 0,
        "deceptive": 3, "adversarial": 3, "adaptive": 2, "modeling": 2,
    },
    {
        "name": "Adaptive-only pressure",
        "cautious": 3, "honest": 2,
        "deceptive": 0, "adversarial": 0, "adaptive": 5, "modeling": 0,
    },
    {
        "name": "Overwhelming force (3:1 adversary ratio)",
        "cautious": 2, "honest": 1,
        "deceptive": 3, "adversarial": 3, "adaptive": 2, "modeling": 1,
    },
    {
        "name": "No governance + adversary majority",
        "cautious": 3, "honest": 2,
        "deceptive": 3, "adversarial": 3, "adaptive": 2, "modeling": 0,
        "threshold_dancer": 0,
        "no_governance": True,
    },
    {
        "name": "Threshold dancers only",
        "cautious": 3, "honest": 2,
        "deceptive": 0, "adversarial": 0, "adaptive": 0, "modeling": 0,
        "threshold_dancer": 3,
    },
    {
        "name": "Mixed adversaries + threshold dancers",
        "cautious": 3, "honest": 2,
        "deceptive": 1, "adversarial": 1, "adaptive": 1, "modeling": 1,
        "threshold_dancer": 2,
    },
]


def run_real_scenario(spec, seed=42):
    """Run a real orchestrator scenario and return metrics."""
    gov = GovernanceConfig() if spec.get("no_governance") else make_governance()

    config = OrchestratorConfig(
        n_epochs=15,
        steps_per_epoch=10,
        seed=seed,
        governance_config=gov,
        payoff_config=make_payoff(),
    )
    orch = Orchestrator(config=config)
    rng = random.Random(seed)

    idx = 0
    for _ in range(spec["cautious"]):
        idx += 1
        orch.register_agent(CautiousReciprocator(
            f"cautious_{idx}", rng=random.Random(rng.randint(0, 99999))))
    for _ in range(spec["honest"]):
        idx += 1
        orch.register_agent(HonestAgent(
            f"honest_{idx}", rng=random.Random(rng.randint(0, 99999))))
    for _ in range(spec["deceptive"]):
        idx += 1
        orch.register_agent(DeceptiveAgent(
            f"deceptive_{idx}", rng=random.Random(rng.randint(0, 99999))))
    for _ in range(spec["adversarial"]):
        idx += 1
        orch.register_agent(AdversarialAgent(
            f"adversarial_{idx}", rng=random.Random(rng.randint(0, 99999))))
    for _ in range(spec["adaptive"]):
        idx += 1
        orch.register_agent(AdaptiveAdversary(
            f"adaptive_{idx}", rng=random.Random(rng.randint(0, 99999))))
    for _ in range(spec.get("modeling", 0)):
        idx += 1
        orch.register_agent(ModelingAdversary(
            f"modeling_{idx}", rng=random.Random(rng.randint(0, 99999))))
    for _ in range(spec.get("threshold_dancer", 0)):
        idx += 1
        orch.register_agent(ThresholdDancer(
            f"threshold_dancer_{idx}", rng=random.Random(rng.randint(0, 99999))))

    metrics = orch.run()

    # Collect per-type payoffs
    type_payoffs = {}
    for agent in orch.get_all_agents():
        state = orch.state.get_agent(agent.agent_id)
        atype = agent.agent_id.split("_")[0]
        type_payoffs.setdefault(atype, []).append(state.total_payoff)

    avg_by_type = {k: sum(v)/len(v) for k, v in type_payoffs.items()}

    avg_toxicity = sum(m.toxicity_rate for m in metrics) / len(metrics) if metrics else 0
    total_welfare = sum(m.total_welfare for m in metrics)
    n_frozen = len(orch.state.frozen_agents)

    return {
        "toxicity": avg_toxicity,
        "welfare": total_welfare,
        "frozen": n_frozen,
        "payoffs": avg_by_type,
        "n_agents": len(orch.get_all_agents()),
    }


def main():
    run_dir = Path("runs") / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_redteam"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RED-TEAM EVALUATION: Cautious Reciprocator")
    print("=" * 70)

    # ── Phase 1: Simulated attack assessment ─────────────────────────
    print("\n--- Phase 1: Governance robustness assessment (simulated) ---\n")

    gov_dict = {
        "transaction_tax_rate": 0.05,
        "reputation_decay_rate": 0.95,
        "circuit_breaker_enabled": True,
        "freeze_threshold_toxicity": 0.8,
        "audit_enabled": True,
        "audit_probability": 0.1,
        "collusion_detection_enabled": False,
        "staking_enabled": False,
    }

    evaluator = RedTeamEvaluator(governance_config=gov_dict)
    report = evaluator.evaluate(
        orchestrator_factory=lambda cfg: None,
        epochs_per_attack=20,
        verbose=True,
    )

    print(f"\nRobustness Score: {report.robustness.robustness_score:.2f} "
          f"(Grade: {report.robustness.grade})")
    print(f"Attacks tested: {report.robustness.attacks_tested}")
    print(f"Attacks prevented: {report.robustness.attacks_prevented}")
    print(f"Attacks successful: {report.robustness.attacks_successful}")
    print(f"Overall evasion rate: {report.robustness.overall_evasion_rate:.1%}")
    print(f"Most effective attack: {report.most_effective_attack}")
    print(f"Least effective attack: {report.least_effective_attack}")

    if report.robustness.vulnerabilities:
        print(f"\nVulnerabilities ({len(report.robustness.vulnerabilities)}):")
        for v in report.robustness.vulnerabilities:
            print(f"  [{v.severity.upper()}] {v.description}")
            print(f"    Lever: {v.affected_lever} | Mitigation: {v.mitigation}")

    if report.robustness.recommendations:
        print("\nRecommendations:")
        for i, r in enumerate(report.robustness.recommendations, 1):
            print(f"  {i}. {r}")

    # ── Phase 2: Real orchestrator stress tests ──────────────────────
    print("\n\n--- Phase 2: Real orchestrator stress tests (15 epochs each) ---\n")

    results = []
    for i, spec in enumerate(ATTACK_SCENARIOS, 1):
        name = spec["name"]
        n_good = spec["cautious"] + spec["honest"]
        n_bad = spec["deceptive"] + spec["adversarial"] + spec["adaptive"] + spec.get("modeling", 0) + spec.get("threshold_dancer", 0)
        print(f"  [{i}/{len(ATTACK_SCENARIOS)}] {name} ({n_good} good, {n_bad} bad)...", end=" ", flush=True)

        res = run_real_scenario(spec)
        res["name"] = name
        res["spec"] = {k: v for k, v in spec.items() if k != "no_governance"}
        res["no_governance"] = spec.get("no_governance", False)
        results.append(res)

        cautious_pay = res["payoffs"].get("cautious", 0)
        adv_pay = max(
            res["payoffs"].get("adversarial", 0),
            res["payoffs"].get("deceptive", 0),
            res["payoffs"].get("adaptive", 0),
            res["payoffs"].get("modeling", 0),
            res["payoffs"].get("threshold", 0),
        )
        print(f"tox={res['toxicity']:.3f}  welfare={res['welfare']:.1f}  "
              f"cautious$={cautious_pay:.1f}  worst_adv$={adv_pay:.1f}  "
              f"frozen={res['frozen']}")

    # ── Results table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS")
    print("=" * 70)

    header = f"{'Scenario':<40} {'Tox':>6} {'Welfare':>9} {'Caut$':>8} {'Adv$':>8} {'Broke?':>7}"
    print(header)
    print("-" * len(header))

    for r in results:
        cautious_pay = r["payoffs"].get("cautious", 0)
        # Worst adversary payoff (highest = most successful attack)
        adv_payoffs = [r["payoffs"].get(t, -999) for t in ADVERSARY_TYPES if t in r["payoffs"]]
        worst_adv = max(adv_payoffs) if adv_payoffs else 0

        # "Broke" if adversaries earn more than cautious OR toxicity > 0.5
        broke = worst_adv > cautious_pay or r["toxicity"] > 0.5
        flag = "YES" if broke else "no"

        print(f"{r['name']:<40} {r['toxicity']:>6.3f} {r['welfare']:>9.1f} "
              f"{cautious_pay:>8.1f} {worst_adv:>8.1f} {flag:>7}")

    # ── Failure analysis ─────────────────────────────────────────────
    failures = [r for r in results
                if max([r["payoffs"].get(t, -999) for t in ADVERSARY_TYPES
                        if t in r["payoffs"]], default=-999) > r["payoffs"].get("cautious", 0)
                or r["toxicity"] > 0.5]

    print(f"\n{'=' * 70}")
    if failures:
        print(f"FAILURES: {len(failures)}/{len(results)} scenarios broke the agent")
        print("-" * 70)
        for f in failures:
            print(f"  {f['name']}:")
            for atype, pay in sorted(f["payoffs"].items(), key=lambda x: -x[1]):
                print(f"    {atype}: {pay:.2f}")
    else:
        print("NO FAILURES: Cautious reciprocator survived all 8 attack scenarios")

    # ── Save report ──────────────────────────────────────────────────
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "phase1_robustness": report.robustness.to_dict(),
        "phase1_attacks": [r.to_dict() for r in report.attack_results],
        "phase2_scenarios": results,
        "phase2_failures": len(failures),
        "phase2_total": len(results),
    }

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(full_report, indent=2, default=str))

    summary_path = run_dir / "report.txt"
    lines = [
        report.generate_summary(),
        "",
        "=" * 60,
        "PHASE 2: REAL ORCHESTRATOR STRESS TESTS",
        "=" * 60,
        "",
        header,
        "-" * len(header),
    ]
    for r in results:
        cautious_pay = r["payoffs"].get("cautious", 0)
        adv_payoffs = [r["payoffs"].get(t, -999) for t in ADVERSARY_TYPES
                       if t in r["payoffs"]]
        worst_adv = max(adv_payoffs) if adv_payoffs else 0
        broke = worst_adv > cautious_pay or r["toxicity"] > 0.5
        lines.append(f"{r['name']:<40} {r['toxicity']:>6.3f} {r['welfare']:>9.1f} "
                     f"{cautious_pay:>8.1f} {worst_adv:>8.1f} {'YES' if broke else 'no':>7}")
    lines.append("")
    lines.append(f"Failures: {len(failures)}/{len(results)}")
    summary_path.write_text("\n".join(lines))

    print("\nReports saved:")
    print(f"  {report_path}")
    print(f"  {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
