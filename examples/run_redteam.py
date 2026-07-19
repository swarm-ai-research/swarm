#!/usr/bin/env python
"""
Run a red-team evaluation against multiple governance configurations.

Executes all 8 attack scenarios from the SWARM attack library against
governance configs ranging from no-defenses to full-defenses, generates
machine-readable and human-readable reports, and prints a summary.

Usage:
    python examples/run_redteam.py
    python examples/run_redteam.py --mode quick
    python examples/run_redteam.py --mode full --out-dir runs/my_redteam
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.redteam.evaluator import RedTeamEvaluator

# ---------------------------------------------------------------------------
# Governance configurations to test
# ---------------------------------------------------------------------------
GOVERNANCE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "no_defenses": {
        "circuit_breaker_enabled": False,
        "collusion_detection_enabled": False,
        "audit_enabled": False,
        "staking_enabled": False,
        "transaction_tax_rate": 0.0,
    },
    "baseline": {
        "circuit_breaker_enabled": True,
        "collusion_detection_enabled": False,
        "audit_enabled": False,
        "staking_enabled": False,
        "transaction_tax_rate": 0.0,
    },
    "moderate": {
        "circuit_breaker_enabled": True,
        "collusion_detection_enabled": False,
        "audit_enabled": True,
        "staking_enabled": True,
        "transaction_tax_rate": 0.05,
    },
    "strict": {
        "circuit_breaker_enabled": True,
        "collusion_detection_enabled": True,
        "audit_enabled": True,
        "staking_enabled": True,
        "transaction_tax_rate": 0.10,
    },
}


def run_evaluation(mode: str, out_dir: Path) -> int:
    """Run red-team evaluation and generate reports."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_reports: Dict[str, Any] = {}
    summaries: list[str] = []

    for config_name, gov_config in GOVERNANCE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing governance config: {config_name}")
        print(f"{'='*60}")

        evaluator = RedTeamEvaluator(governance_config=gov_config)

        if mode == "quick":
            quick_result = evaluator.quick_evaluate()
            all_reports[config_name] = {
                "mode": "quick",
                "governance_config": gov_config,
                "result": quick_result,
            }
            n_tested = quick_result["attacks_tested"]
            n_success = quick_result["attacks_successful"]
            evasion = quick_result["avg_evasion_rate"]
            damage = quick_result["total_damage"]
            print(f"  Attacks tested:     {n_tested}")
            print(f"  Attacks successful: {n_success}")
            print(f"  Avg evasion rate:   {evasion:.2%}")
            print(f"  Total damage:       {damage:.1f}")
        else:
            # Full evaluation
            report = evaluator.evaluate(
                orchestrator_factory=lambda c: None,
                epochs_per_attack=20,
                verbose=True,
            )
            summary_text = report.generate_summary()
            summaries.append(f"\n--- Config: {config_name} ---\n{summary_text}")
            all_reports[config_name] = {
                "mode": "full",
                "governance_config": gov_config,
                "report": report.to_dict(),
            }
            print(f"\n  Score: {report.robustness.robustness_score:.2f} ({report.robustness.grade})")
            print(f"  Attacks: {report.robustness.attacks_tested} tested, "
                  f"{report.robustness.attacks_prevented} prevented, "
                  f"{report.robustness.attacks_successful} succeeded")
            print(f"  Evasion rate: {report.robustness.overall_evasion_rate:.2%}")
            print(f"  Total damage: {report.robustness.total_damage_allowed:.1f}")
            if report.most_effective_attack:
                print(f"  Most effective attack: {report.most_effective_attack}")
            if report.robustness.vulnerabilities:
                print(f"  Vulnerabilities found: {len(report.robustness.vulnerabilities)}")
                for v in report.robustness.vulnerabilities[:3]:
                    print(f"    [{v.severity.upper()}] {v.description}")

    # Write machine-readable report
    report_json_path = out_dir / "report.json"

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # Avoid __dict__ fallback -- it could leak internal state
        # (e.g. credentials added to config objects in the future).
        return str(obj)

    with open(report_json_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=_serialize)
    print(f"\nMachine-readable report: {report_json_path}")

    # Write human-readable report
    report_txt_path = out_dir / "report.txt"
    with open(report_txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SWARM RED-TEAM EVALUATION REPORT\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Mode: {mode}\n")
        f.write("=" * 60 + "\n\n")

        if mode == "full" and summaries:
            for s in summaries:
                f.write(s + "\n")
        else:
            # Quick mode summary
            for config_name, data in all_reports.items():
                result = data.get("result", {})
                f.write(f"\n--- Config: {config_name} ---\n")
                f.write(f"  Attacks tested:     {result.get('attacks_tested', 0)}\n")
                f.write(f"  Attacks successful: {result.get('attacks_successful', 0)}\n")
                f.write(f"  Avg evasion rate:   {result.get('avg_evasion_rate', 0):.2%}\n")
                f.write(f"  Total damage:       {result.get('total_damage', 0):.1f}\n")
                f.write("\n  Per-attack results:\n")
                for r in result.get("results", []):
                    status = "SUCCEEDED" if r.get("attack_succeeded") else "PREVENTED"
                    f.write(f"    [{status}] {r['scenario']['name']} "
                            f"(dmg={r['damage_caused']:.1f}, evasion={r['evasion_rate']:.2%})\n")
                f.write("\n")

        # Comparative summary
        f.write("\n" + "=" * 60 + "\n")
        f.write("COMPARATIVE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Config':<20} {'Score':>8} {'Grade':>6} {'Prevented':>10} {'Succeeded':>10} {'Damage':>10}\n")
        f.write("-" * 64 + "\n")
        for config_name, data in all_reports.items():
            if mode == "full":
                rpt = data.get("report", {}).get("robustness", {})
                score = rpt.get("robustness_score", 0)
                grade = rpt.get("grade", "?")
                prevented = rpt.get("attacks_prevented", 0)
                succeeded = rpt.get("attacks_successful", 0)
                damage = rpt.get("total_damage_allowed", 0)
                f.write(f"{config_name:<20} {score:>8.2f} {grade:>6} {prevented:>10} {succeeded:>10} {damage:>10.1f}\n")
            else:
                result = data.get("result", {})
                tested = result.get("attacks_tested", 0)
                succeeded = result.get("attacks_successful", 0)
                prevented = tested - succeeded
                damage = result.get("total_damage", 0)
                f.write(f"{config_name:<20} {'n/a':>8} {'n/a':>6} {prevented:>10} {succeeded:>10} {damage:>10.1f}\n")
        f.write("\n")

    print(f"Human-readable report: {report_txt_path}")

    # Print final summary for issue/PR use
    print("\n" + "=" * 60)
    print("RED-TEAM SUMMARY")
    print("=" * 60)

    if mode == "full":
        for config_name, data in all_reports.items():
            rpt = data.get("report", {}).get("robustness", {})
            score = rpt.get("robustness_score", 0)
            grade = rpt.get("grade", "?")
            print(f"  {config_name:<20} -> Score: {score:.2f} ({grade})")

        # Find worst vulnerabilities across all configs
        print("\nTop Vulnerabilities:")
        seen = set()
        all_vulns = []
        for data in all_reports.values():
            for v in data.get("report", {}).get("robustness", {}).get("vulnerabilities", []):
                vid = v.get("vulnerability_id", "")
                if vid not in seen:
                    seen.add(vid)
                    all_vulns.append(v)
        all_vulns.sort(key=lambda v: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(v.get("severity", "low"), 4))
        for v in all_vulns[:5]:
            sev = v.get("severity", "?").upper()
            desc = v.get("description", "")
            lever = v.get("affected_lever", "?")
            print(f"  [{sev}] {desc} (lever: {lever})")

        # Recommendations
        print("\nRecommendations:")
        seen_recs = set()
        for data in all_reports.values():
            for rec in data.get("report", {}).get("robustness", {}).get("recommendations", []):
                if rec not in seen_recs:
                    seen_recs.add(rec)
                    print(f"  - {rec}")
    else:
        for config_name, data in all_reports.items():
            result = data.get("result", {})
            tested = result.get("attacks_tested", 0)
            succeeded = result.get("attacks_successful", 0)
            damage = result.get("total_damage", 0)
            print(f"  {config_name:<20} -> {succeeded}/{tested} attacks succeeded, damage={damage:.1f}")

    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SWARM red-team evaluation")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="Evaluation mode: quick (3 attacks) or full (8 attacks)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/<timestamp>_redteam/)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(f"runs/{timestamp}_redteam")

    return run_evaluation(args.mode, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
