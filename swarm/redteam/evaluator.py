"""Red-team evaluator for systematic governance testing.

This module provides the main interface for running red-team evaluations
against governance configurations.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from swarm.redteam.attacks import AttackLibrary, AttackResult, AttackScenario
from swarm.redteam.metrics import (
    AdversaryPerformance,
    EvasionMetrics,
    GovernanceEffectiveness,
)


@dataclass
class GovernanceRobustness:
    """Overall robustness assessment of governance configuration."""

    # Identification
    governance_config: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.now)

    # Attack results
    attacks_tested: int = 0
    attacks_successful: int = 0
    attacks_prevented: int = 0

    # By category
    results_by_category: Dict[str, List[AttackResult]] = field(default_factory=dict)
    results_by_difficulty: Dict[str, List[AttackResult]] = field(default_factory=dict)

    # Aggregate metrics
    overall_evasion_rate: float = 0.0
    overall_detection_latency: float = 0.0
    total_damage_allowed: float = 0.0
    total_penalties_applied: float = 0.0

    # Effectiveness
    governance_effectiveness: Optional[GovernanceEffectiveness] = None

    # Vulnerabilities
    vulnerabilities: List["Vulnerability"] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def robustness_score(self) -> float:
        """
        Overall robustness score from 0-1.

        Based on:
        - Attack prevention rate (40%)
        - Detection rate (30%)
        - Damage mitigation (30%)
        """
        if self.attacks_tested == 0:
            return 1.0

        prevention_rate = self.attacks_prevented / self.attacks_tested
        detection_rate = 1.0 - self.overall_evasion_rate

        # Normalize damage (assuming max expected damage is 100 per attack)
        max_expected_damage = self.attacks_tested * 100
        damage_mitigation = 1.0 - min(
            1.0, self.total_damage_allowed / max_expected_damage
        )

        return 0.4 * prevention_rate + 0.3 * detection_rate + 0.3 * damage_mitigation

    @property
    def grade(self) -> str:
        """Letter grade for robustness."""
        score = self.robustness_score
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "governance_config": self.governance_config,
            "evaluated_at": self.evaluated_at.isoformat(),
            "attacks_tested": self.attacks_tested,
            "attacks_successful": self.attacks_successful,
            "attacks_prevented": self.attacks_prevented,
            "overall_evasion_rate": self.overall_evasion_rate,
            "overall_detection_latency": self.overall_detection_latency,
            "total_damage_allowed": self.total_damage_allowed,
            "robustness_score": self.robustness_score,
            "grade": self.grade,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "recommendations": self.recommendations,
        }


@dataclass
class Vulnerability:
    """A discovered vulnerability in governance."""

    vulnerability_id: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    attack_vector: str
    affected_lever: str
    exploitation_difficulty: str
    potential_damage: float
    mitigation: str

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "vulnerability_id": self.vulnerability_id,
            "severity": self.severity,
            "description": self.description,
            "attack_vector": self.attack_vector,
            "affected_lever": self.affected_lever,
            "exploitation_difficulty": self.exploitation_difficulty,
            "potential_damage": self.potential_damage,
            "mitigation": self.mitigation,
        }


@dataclass
class VulnerabilityReport:
    """Detailed vulnerability report from red-team evaluation."""

    robustness: GovernanceRobustness
    attack_results: List[AttackResult] = field(default_factory=list)
    adversary_performances: List[AdversaryPerformance] = field(default_factory=list)

    # Detailed findings
    weakest_lever: Optional[str] = None
    strongest_lever: Optional[str] = None
    most_effective_attack: Optional[str] = None
    least_effective_attack: Optional[str] = None

    # Time series data
    damage_over_time: List[float] = field(default_factory=list)
    detection_over_time: List[int] = field(default_factory=list)

    def generate_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "RED-TEAM EVALUATION REPORT",
            "=" * 60,
            "",
            f"Governance Configuration: {self.robustness.governance_config}",
            f"Evaluation Date: {self.robustness.evaluated_at}",
            "",
            "OVERALL ASSESSMENT",
            "-" * 40,
            f"Robustness Score: {self.robustness.robustness_score:.2f} ({self.robustness.grade})",
            f"Attacks Tested: {self.robustness.attacks_tested}",
            f"Attacks Prevented: {self.robustness.attacks_prevented}",
            f"Attacks Successful: {self.robustness.attacks_successful}",
            "",
            "KEY METRICS",
            "-" * 40,
            f"Overall Evasion Rate: {self.robustness.overall_evasion_rate:.2%}",
            f"Detection Latency: {self.robustness.overall_detection_latency:.1f} epochs",
            f"Total Damage Allowed: {self.robustness.total_damage_allowed:.2f}",
            "",
        ]

        if self.robustness.vulnerabilities:
            lines.extend(
                [
                    "VULNERABILITIES DISCOVERED",
                    "-" * 40,
                ]
            )
            for vuln in self.robustness.vulnerabilities:
                lines.extend(
                    [
                        f"  [{vuln.severity.upper()}] {vuln.vulnerability_id}",
                        f"    {vuln.description}",
                        f"    Attack Vector: {vuln.attack_vector}",
                        f"    Mitigation: {vuln.mitigation}",
                        "",
                    ]
                )

        if self.robustness.recommendations:
            lines.extend(
                [
                    "RECOMMENDATIONS",
                    "-" * 40,
                ]
            )
            for i, rec in enumerate(self.robustness.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "robustness": self.robustness.to_dict(),
            "attack_results": [r.to_dict() for r in self.attack_results],
            "adversary_performances": [
                p.to_dict() for p in self.adversary_performances
            ],
            "weakest_lever": self.weakest_lever,
            "strongest_lever": self.strongest_lever,
            "most_effective_attack": self.most_effective_attack,
            "least_effective_attack": self.least_effective_attack,
        }


class RedTeamEvaluator:
    """
    Evaluator for testing governance robustness.

    This class orchestrates red-team evaluations by:
    1. Running attack scenarios against governance configurations
    2. Measuring evasion rates and damage
    3. Identifying vulnerabilities
    4. Generating recommendations
    """

    def __init__(
        self,
        governance_config: Dict[str, Any],
        attack_scenarios: Optional[List[AttackScenario]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            governance_config: Governance configuration to test
            attack_scenarios: List of attacks to run (defaults to all)
        """
        self.governance_config = governance_config
        self.attack_scenarios = attack_scenarios or AttackLibrary.get_all_attacks()

        # Results storage
        self.attack_results: List[AttackResult] = []
        self.evasion_metrics = EvasionMetrics()
        self.governance_effectiveness = GovernanceEffectiveness()

    def evaluate(
        self,
        orchestrator_factory: Callable[..., Any],
        epochs_per_attack: int = 20,
        verbose: bool = False,
    ) -> VulnerabilityReport:
        """
        Run full red-team evaluation.

        Args:
            orchestrator_factory: Function that creates configured orchestrator
            epochs_per_attack: Number of epochs to run per attack
            verbose: Print progress

        Returns:
            VulnerabilityReport with findings
        """
        robustness = GovernanceRobustness(
            governance_config=self.governance_config,
        )

        for scenario in self.attack_scenarios:
            if verbose:
                print(f"Running attack: {scenario.name}...")

            result = self._run_attack(
                scenario=scenario,
                orchestrator_factory=orchestrator_factory,
                epochs=epochs_per_attack,
            )

            self.attack_results.append(result)
            robustness.attacks_tested += 1

            if result.attack_succeeded:
                robustness.attacks_successful += 1
            if result.governance_prevented:
                robustness.attacks_prevented += 1

            robustness.total_damage_allowed += result.damage_caused

            # Categorize result
            category = scenario.category.value
            if category not in robustness.results_by_category:
                robustness.results_by_category[category] = []
            robustness.results_by_category[category].append(result)

            difficulty = scenario.difficulty.value
            if difficulty not in robustness.results_by_difficulty:
                robustness.results_by_difficulty[difficulty] = []
            robustness.results_by_difficulty[difficulty].append(result)

        # Compute aggregate metrics
        if self.attack_results:
            robustness.overall_evasion_rate = float(
                np.mean([r.evasion_rate for r in self.attack_results])
            )
            latencies = [
                r.detection_latency
                for r in self.attack_results
                if r.detection_latency > 0
            ]
            if latencies:
                robustness.overall_detection_latency = float(np.mean(latencies))

        # Identify vulnerabilities
        robustness.vulnerabilities = self._identify_vulnerabilities()

        # Generate recommendations
        robustness.recommendations = self._generate_recommendations(robustness)

        # Build report
        report = VulnerabilityReport(
            robustness=robustness,
            attack_results=self.attack_results,
        )

        # Find extremes
        if self.attack_results:
            best_attack = max(self.attack_results, key=lambda r: r.damage_caused)
            worst_attack = min(self.attack_results, key=lambda r: r.damage_caused)
            report.most_effective_attack = best_attack.scenario.name
            report.least_effective_attack = worst_attack.scenario.name

        return report

    def _run_attack(
        self,
        scenario: AttackScenario,
        orchestrator_factory: Callable[..., Any],
        epochs: int,
    ) -> AttackResult:
        """
        Run a single attack scenario.

        This is a simplified simulation - in production, this would
        integrate with the actual orchestrator.
        """
        result = AttackResult(
            attack_id=scenario.attack_id,
            scenario=scenario,
        )

        result.epochs_run = epochs

        # Simulate outcomes based on governance config and attack type
        result = self._simulate_attack_outcome(scenario, result)

        return result

    def _simulate_attack_outcome(
        self,
        scenario: AttackScenario,
        result: AttackResult,
    ) -> AttackResult:
        """
        Simulate attack outcome based on governance config.

        This provides baseline estimates - real evaluations should
        run actual simulations.
        """
        # Check which defenses are active
        has_circuit_breaker = self.governance_config.get(
            "circuit_breaker_enabled", False
        )
        has_collusion_detection = self.governance_config.get(
            "collusion_detection_enabled", False
        )
        has_audit = self.governance_config.get("audit_enabled", False)
        has_staking = self.governance_config.get("staking_enabled", False)
        tax_rate = self.governance_config.get("transaction_tax_rate", 0.0)

        # Base success probability
        base_success = 0.6

        # Adjust based on attack vs defenses
        if scenario.attack_id == "collusion_ring":
            if has_collusion_detection:
                base_success *= 0.3
                result.collusion_detected = True
            else:
                base_success *= 1.2

        elif scenario.attack_id == "threshold_dancing":
            if has_circuit_breaker:
                # Threshold dancing specifically tries to evade this
                base_success *= 0.8
            else:
                base_success *= 1.5

        elif scenario.attack_id == "reputation_farming":
            # Harder to detect without audit
            if has_audit:
                base_success *= 0.5
            else:
                base_success *= 1.3

        elif scenario.attack_id == "sybil_attack":
            has_sybil_detection = self.governance_config.get(
                "sybil_detection_enabled", False
            )
            if has_sybil_detection:
                # Behavioral clustering is strong against identity-correlated Sybils
                base_success *= 0.25
            if has_collusion_detection:
                # Secondary reduction: collusion detection catches coordinated Sybils
                base_success *= 0.7
            if not has_sybil_detection and not has_collusion_detection:
                base_success *= 1.3

        elif scenario.attack_id == "out_of_band_channel":
            # Field-evidence exhibit (collusion.wiki replay): pairwise, group
            # and structural collusion detection saturate on a hub page and do
            # not reduce success, so ``has_collusion_detection`` deliberately
            # has no effect here. An aggregate volume monitor does (it led the
            # human moderator by three days); closing the write path to the
            # substrate is a capability precondition that prevents the attack
            # rather than detecting it.
            has_volume_monitor = self.governance_config.get(
                "volume_monitor_enabled", False
            )
            has_population_monitor = self.governance_config.get(
                "substrate_population_monitor_enabled", False
            )
            has_egress_allowlist = self.governance_config.get(
                "egress_allowlist_enabled", False
            )
            # Watch the population of hosts, not the host you found (vv3j.8):
            # a per-host monitor's effect decays as the swarm gains substrates
            # to relocate across; the population monitor's does not.
            n_hosts = max(1, int(scenario.adversary_config.get("n_hosts", 1)))
            per_host_factor = 0.4 + 0.6 * (1.0 - 1.0 / n_hosts)
            if has_egress_allowlist:
                base_success *= 0.05
            elif has_population_monitor:
                base_success *= 0.4
                result.collusion_detected = True
            elif has_volume_monitor:
                base_success *= per_host_factor
                result.collusion_detected = n_hosts == 1
            else:
                base_success *= 1.3

        # General defense effects
        if has_staking:
            base_success *= 0.9
        if tax_rate > 0:
            base_success *= 1 - tax_rate

        # Difficulty modifier
        difficulty_modifier = {
            "trivial": 1.3,
            "simple": 1.1,
            "moderate": 1.0,
            "advanced": 0.9,
            "expert": 0.8,
        }.get(scenario.difficulty.value, 1.0)

        base_success *= difficulty_modifier

        # Determine outcome
        result.attack_succeeded = random.random() < min(1.0, base_success)
        result.governance_prevented = not result.attack_succeeded

        # Estimate metrics
        result.evasion_rate = base_success * 0.7
        result.detection_latency = int(5 / max(0.1, 1 - base_success))
        result.damage_caused = base_success * 50 * scenario.n_adversaries

        if result.attack_succeeded:
            result.damage_before_detection = result.damage_caused * 0.6
        else:
            result.detected_at_epoch = result.detection_latency
            result.damage_before_detection = result.damage_caused * 0.2

        return result

    def _identify_vulnerabilities(self) -> List[Vulnerability]:
        """Identify vulnerabilities from attack results."""
        vulnerabilities = []

        # High evasion rate attacks indicate vulnerabilities
        for result in self.attack_results:
            if result.evasion_rate > 0.7:
                vuln = Vulnerability(
                    vulnerability_id=f"vuln_{result.attack_id}_evasion",
                    severity="high" if result.evasion_rate > 0.9 else "medium",
                    description=f"{result.scenario.name} has high evasion rate ({result.evasion_rate:.0%})",
                    attack_vector=result.scenario.category.value,
                    affected_lever=result.scenario.targeted_levers[0]
                    if result.scenario.targeted_levers
                    else "unknown",
                    exploitation_difficulty=result.scenario.difficulty.value,
                    potential_damage=result.damage_caused,
                    mitigation=f"Strengthen detection for {result.scenario.category.value} attacks",
                )
                vulnerabilities.append(vuln)

        # Successful attacks indicate vulnerabilities
        for result in self.attack_results:
            if result.attack_succeeded and result.damage_caused > 30:
                vuln = Vulnerability(
                    vulnerability_id=f"vuln_{result.attack_id}_success",
                    severity="critical" if result.damage_caused > 50 else "high",
                    description=f"{result.scenario.name} succeeded with significant damage ({result.damage_caused:.0f})",
                    attack_vector=result.scenario.category.value,
                    affected_lever=result.scenario.targeted_levers[0]
                    if result.scenario.targeted_levers
                    else "unknown",
                    exploitation_difficulty=result.scenario.difficulty.value,
                    potential_damage=result.damage_caused,
                    mitigation=f"Enable or strengthen {', '.join(result.scenario.targeted_levers)}",
                )
                vulnerabilities.append(vuln)

        return vulnerabilities

    def _generate_recommendations(self, robustness: GovernanceRobustness) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        # Check for missing defenses
        if not self.governance_config.get("circuit_breaker_enabled"):
            recommendations.append(
                "Enable circuit breaker to automatically freeze malicious agents"
            )

        if not self.governance_config.get("collusion_detection_enabled"):
            if any(
                r.scenario.category.value == "coordination" and r.attack_succeeded
                for r in self.attack_results
            ):
                recommendations.append(
                    "Enable collusion detection to identify coordinated attacks"
                )

        if not self.governance_config.get("volume_monitor_enabled"):
            if any(
                r.attack_id == "out_of_band_channel" and r.attack_succeeded
                for r in self.attack_results
            ):
                recommendations.append(
                    "Add an aggregate volume/rate monitor: out-of-band "
                    "coordination saturates pairwise/structural collusion "
                    "detection and is only visible as volume (collusion.wiki replay)"
                )
        if self.governance_config.get("volume_monitor_enabled") and not (
            self.governance_config.get("substrate_population_monitor_enabled")
        ):
            if any(
                r.attack_id == "out_of_band_channel"
                and r.attack_succeeded
                and int(r.scenario.adversary_config.get("n_hosts", 1)) > 1
                for r in self.attack_results
            ):
                recommendations.append(
                    "Watch the population of hosts, not the host you found: a "
                    "per-host volume monitor acts on the caught substrate and a "
                    "swarm with a choice sits on the one that is not (wiki-board "
                    "substrate gap)"
                )

        if not self.governance_config.get("audit_enabled"):
            if robustness.overall_evasion_rate > 0.5:
                recommendations.append(
                    "Enable random audits to increase detection of evasive behavior"
                )

        if self.governance_config.get("transaction_tax_rate", 0) < 0.05:
            if any(
                r.scenario.category.value == "resource" and r.attack_succeeded
                for r in self.attack_results
            ):
                recommendations.append(
                    "Increase transaction tax to discourage resource extraction"
                )

        # Threshold adjustments
        if robustness.overall_evasion_rate > 0.6:
            recommendations.append(
                "Lower detection thresholds - current settings allow too much evasion"
            )

        if robustness.attacks_successful > robustness.attacks_prevented:
            recommendations.append(
                "Consider defense-in-depth: enable multiple complementary governance levers"
            )

        return recommendations

    def quick_evaluate(self, attack_ids: Optional[List[str]] = None) -> Dict:
        """
        Quick evaluation with subset of attacks.

        Args:
            attack_ids: Specific attacks to run (or None for quick set)

        Returns:
            Summary dictionary
        """
        if attack_ids:
            attacks = [
                a for a in AttackLibrary.get_all_attacks() if a.attack_id in attack_ids
            ]
        else:
            # Quick set: one from each category
            attacks = [
                AttackLibrary.reputation_farming_exploit(),
                AttackLibrary.collusion_ring(),
                AttackLibrary.threshold_dancing(),
            ]

        self.attack_scenarios = attacks

        # Simplified evaluation
        results = []
        for scenario in attacks:
            result = AttackResult(
                attack_id=scenario.attack_id,
                scenario=scenario,
                epochs_run=10,
            )
            result = self._simulate_attack_outcome(scenario, result)
            results.append(result)

        # Summary
        return {
            "attacks_tested": len(results),
            "attacks_successful": sum(1 for r in results if r.attack_succeeded),
            "avg_evasion_rate": np.mean([r.evasion_rate for r in results]),
            "total_damage": sum(r.damage_caused for r in results),
            "results": [r.to_dict() for r in results],
        }
