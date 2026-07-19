"""Metrics for measuring adversarial evasion and governance effectiveness.

This module provides metrics for evaluating how well adversaries evade
governance and how effective governance mechanisms are at detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvasionMetrics:
    """Metrics measuring adversary evasion success."""

    # Detection metrics
    total_malicious_actions: int = 0
    detected_actions: int = 0
    undetected_actions: int = 0

    # Timing
    first_detection_epoch: Optional[int] = None
    detection_epochs: List[int] = field(default_factory=list)

    # Damage
    total_damage: float = 0.0
    damage_before_first_detection: float = 0.0
    damage_after_detection: float = 0.0

    # Strategy metrics
    strategy_attempts: Dict[str, int] = field(default_factory=dict)
    strategy_detections: Dict[str, int] = field(default_factory=dict)

    @property
    def evasion_rate(self) -> float:
        """Fraction of malicious actions that went undetected."""
        if self.total_malicious_actions == 0:
            return 0.0
        return self.undetected_actions / self.total_malicious_actions

    @property
    def detection_rate(self) -> float:
        """Fraction of malicious actions that were detected."""
        return 1.0 - self.evasion_rate

    @property
    def avg_detection_latency(self) -> float:
        """Average number of epochs between action and detection."""
        if not self.detection_epochs:
            return float("inf")
        return float(np.mean(self.detection_epochs))

    @property
    def damage_ratio(self) -> float:
        """Ratio of damage before detection to total damage."""
        if self.total_damage == 0:
            return 0.0
        return self.damage_before_first_detection / self.total_damage

    def record_action(
        self,
        detected: bool,
        damage: float,
        epoch: int,
        strategy: Optional[str] = None,
    ) -> None:
        """Record a malicious action."""
        self.total_malicious_actions += 1
        self.total_damage += damage

        if detected:
            self.detected_actions += 1
            self.detection_epochs.append(epoch)
            self.damage_after_detection += damage

            if self.first_detection_epoch is None:
                self.first_detection_epoch = epoch
        else:
            self.undetected_actions += 1
            if self.first_detection_epoch is None:
                self.damage_before_first_detection += damage

        if strategy:
            self.strategy_attempts[strategy] = (
                self.strategy_attempts.get(strategy, 0) + 1
            )
            if detected:
                self.strategy_detections[strategy] = (
                    self.strategy_detections.get(strategy, 0) + 1
                )

    def get_strategy_evasion_rates(self) -> Dict[str, float]:
        """Get evasion rate for each strategy."""
        rates = {}
        for strategy, attempts in self.strategy_attempts.items():
            detections = self.strategy_detections.get(strategy, 0)
            rates[strategy] = 1.0 - (detections / attempts) if attempts > 0 else 0.0
        return rates

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "total_malicious_actions": self.total_malicious_actions,
            "detected_actions": self.detected_actions,
            "undetected_actions": self.undetected_actions,
            "evasion_rate": self.evasion_rate,
            "detection_rate": self.detection_rate,
            "first_detection_epoch": self.first_detection_epoch,
            "avg_detection_latency": self.avg_detection_latency,
            "total_damage": self.total_damage,
            "damage_before_first_detection": self.damage_before_first_detection,
            "damage_ratio": self.damage_ratio,
            "strategy_evasion_rates": self.get_strategy_evasion_rates(),
        }


def compute_evasion_rate(
    malicious_actions: List[Dict],
    detections: List[Dict],
) -> float:
    """
    Compute overall evasion rate.

    Args:
        malicious_actions: List of malicious action records
        detections: List of detection records

    Returns:
        Fraction of actions that went undetected
    """
    if not malicious_actions:
        return 0.0

    detected_ids = {d.get("action_id") for d in detections}
    undetected = sum(
        1 for a in malicious_actions if a.get("action_id") not in detected_ids
    )

    return undetected / len(malicious_actions)


def compute_detection_latency(
    malicious_actions: List[Dict],
    detections: List[Dict],
) -> Tuple[float, float, float]:
    """
    Compute detection latency statistics.

    Args:
        malicious_actions: List of malicious actions with epochs
        detections: List of detections with epochs

    Returns:
        Tuple of (mean_latency, min_latency, max_latency)
    """
    action_epochs = {a.get("action_id"): a.get("epoch", 0) for a in malicious_actions}

    latencies = []
    for detection in detections:
        action_id = detection.get("action_id")
        detection_epoch = detection.get("epoch", 0)

        if action_id in action_epochs:
            latency = detection_epoch - action_epochs[action_id]
            latencies.append(latency)

    if not latencies:
        return (float("inf"), float("inf"), float("inf"))

    return (np.mean(latencies), min(latencies), max(latencies))


def compute_damage_before_detection(
    malicious_actions: List[Dict],
    first_detection_epoch: Optional[int],
) -> float:
    """
    Compute total damage caused before first detection.

    Args:
        malicious_actions: List of actions with damage and epoch
        first_detection_epoch: Epoch when first detection occurred

    Returns:
        Total damage before first detection
    """
    if first_detection_epoch is None:
        # Never detected - all damage counts
        return float(sum(a.get("damage", 0.0) for a in malicious_actions))

    return float(
        sum(
            a.get("damage", 0.0)
            for a in malicious_actions
            if a.get("epoch", 0) < first_detection_epoch
        )
    )


@dataclass
class GovernanceEffectiveness:
    """Metrics measuring governance effectiveness against attacks."""

    # Detection capability
    true_positives: int = 0  # Correctly detected malicious actions
    false_positives: int = 0  # Incorrectly flagged benign actions
    true_negatives: int = 0  # Correctly allowed benign actions
    false_negatives: int = 0  # Missed malicious actions

    # Response effectiveness
    penalties_applied: float = 0.0
    agents_frozen: int = 0
    freeze_duration_total: int = 0

    # Attack prevention
    attacks_prevented: int = 0
    attacks_succeeded: int = 0

    # Collateral damage
    honest_agents_penalized: int = 0
    honest_agent_losses: float = 0.0

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        total = self.true_positives + self.false_positives
        if total == 0:
            return 0.0
        return self.true_positives / total

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)."""
        total = self.true_positives + self.false_negatives
        if total == 0:
            return 0.0
        return self.true_positives / total

    @property
    def f1_score(self) -> float:
        """F1 score: harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / total."""
        total = (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def attack_prevention_rate(self) -> float:
        """Rate of attacks prevented."""
        total = self.attacks_prevented + self.attacks_succeeded
        if total == 0:
            return 0.0
        return self.attacks_prevented / total

    @property
    def collateral_damage_rate(self) -> float:
        """Rate of honest agents incorrectly penalized."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.false_positives / (self.true_positives + self.false_positives)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "attack_prevention_rate": self.attack_prevention_rate,
            "collateral_damage_rate": self.collateral_damage_rate,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "penalties_applied": self.penalties_applied,
            "agents_frozen": self.agents_frozen,
            "attacks_prevented": self.attacks_prevented,
            "attacks_succeeded": self.attacks_succeeded,
            "honest_agents_penalized": self.honest_agents_penalized,
        }


def compute_governance_effectiveness(
    agent_actions: List[Dict],
    agent_types: Dict[str, str],
    detections: List[Dict],
    penalties: List[Dict],
) -> GovernanceEffectiveness:
    """
    Compute governance effectiveness metrics.

    Args:
        agent_actions: List of all actions with agent_id
        agent_types: Mapping of agent_id to type (honest, adversarial, etc.)
        detections: List of detection events
        penalties: List of penalty events

    Returns:
        GovernanceEffectiveness metrics
    """
    effectiveness = GovernanceEffectiveness()

    detected_action_ids = {d.get("action_id") for d in detections}

    for action in agent_actions:
        agent_id = action.get("agent_id", "")
        action_id = action.get("action_id", "")
        is_adversarial = agent_types.get(agent_id, "") in ["adversarial", "deceptive"]
        was_detected = action_id in detected_action_ids

        if is_adversarial:
            if was_detected:
                effectiveness.true_positives += 1
            else:
                effectiveness.false_negatives += 1
        else:
            if was_detected:
                effectiveness.false_positives += 1
                effectiveness.honest_agents_penalized += 1
            else:
                effectiveness.true_negatives += 1

    # Count penalties
    for penalty in penalties:
        effectiveness.penalties_applied += penalty.get("amount", 0.0)
        if penalty.get("freeze", False):
            effectiveness.agents_frozen += 1
            effectiveness.freeze_duration_total += penalty.get("duration", 0)

        # Track honest agent losses
        agent_id = penalty.get("agent_id", "")
        if agent_types.get(agent_id, "") == "honest":
            effectiveness.honest_agent_losses += penalty.get("amount", 0.0)

    return effectiveness


@dataclass
class AdversaryPerformance:
    """Metrics measuring adversary performance."""

    agent_id: str
    strategy_distribution: Dict[str, float] = field(default_factory=dict)

    # Outcomes
    total_payoff: float = 0.0
    total_penalties: float = 0.0
    net_payoff: float = 0.0

    # Evasion
    actions_taken: int = 0
    times_detected: int = 0
    times_frozen: int = 0

    # Learning
    strategies_tried: int = 0
    best_strategy: Optional[str] = None
    best_strategy_payoff: float = 0.0

    # Adaptation
    threshold_estimates: Dict[str, float] = field(default_factory=dict)
    adaptation_success_rate: float = 0.0

    @property
    def detection_rate(self) -> float:
        """Rate at which this adversary is detected."""
        if self.actions_taken == 0:
            return 0.0
        return self.times_detected / self.actions_taken

    @property
    def roi(self) -> float:
        """Return on investment (net payoff / effort)."""
        if self.actions_taken == 0:
            return 0.0
        return self.net_payoff / self.actions_taken

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "strategy_distribution": self.strategy_distribution,
            "total_payoff": self.total_payoff,
            "total_penalties": self.total_penalties,
            "net_payoff": self.net_payoff,
            "actions_taken": self.actions_taken,
            "detection_rate": self.detection_rate,
            "times_frozen": self.times_frozen,
            "best_strategy": self.best_strategy,
            "adaptation_success_rate": self.adaptation_success_rate,
            "roi": self.roi,
        }
