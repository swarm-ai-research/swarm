"""Red-teaming framework for governance robustness testing.

This module provides tools for systematically testing governance mechanisms
against adaptive adversaries.
"""

from swarm.redteam.attacks import (
    AttackLibrary,
    AttackResult,
    AttackScenario,
)
from swarm.redteam.circumvention import (
    CircumventionConfig,
    TaskOutcome,
    WorkerKind,
    action_filter_catch_rate,
    circumvention_success_rate,
    honest_completion_rate,
    mean_cumulative_suspicion_evaded,
    simulate_episode,
    simulate_task,
)
from swarm.redteam.evaluator import (
    GovernanceRobustness,
    RedTeamEvaluator,
    VulnerabilityReport,
)
from swarm.redteam.metrics import (
    EvasionMetrics,
    compute_damage_before_detection,
    compute_detection_latency,
    compute_evasion_rate,
    compute_governance_effectiveness,
)

__all__ = [
    # Attacks
    "AttackScenario",
    "AttackResult",
    "AttackLibrary",
    # Evaluator
    "RedTeamEvaluator",
    "GovernanceRobustness",
    "VulnerabilityReport",
    # Metrics
    "EvasionMetrics",
    "compute_evasion_rate",
    "compute_detection_latency",
    "compute_damage_before_detection",
    "compute_governance_effectiveness",
    # Constraint circumvention (mtht)
    "CircumventionConfig",
    "TaskOutcome",
    "WorkerKind",
    "simulate_task",
    "simulate_episode",
    "action_filter_catch_rate",
    "circumvention_success_rate",
    "honest_completion_rate",
    "mean_cumulative_suspicion_evaded",
]
