"""Red-teaming framework for governance robustness testing.

This module provides tools for systematically testing governance mechanisms
against adaptive adversaries.
"""

from swarm.redteam.attacker_economics import (
    EconomicsConfig,
    attacker_cost,
    critical_unit_cost,
    deterrence_threshold,
    false_positive_rate,
    fp_threshold,
    is_defensible,
    is_deterred,
)
from swarm.redteam.attacks import (
    AttackLibrary,
    AttackResult,
    AttackScenario,
)
from swarm.redteam.circumvention import (
    CircumventionConfig,
    MonitorVerdict,
    TaskOutcome,
    TrajectoryMonitorConfig,
    WorkerKind,
    action_filter_catch_rate,
    circumvention_success_rate,
    honest_completion_rate,
    mean_cumulative_suspicion_evaded,
    mean_realized_mass_on_catch,
    monitor_trajectory,
    simulate_episode,
    simulate_task,
    trajectory_monitor_catch_rate,
    trajectory_monitor_false_positive_rate,
)
from swarm.redteam.decomposition_ladder import (
    AttackPlan,
    LadderConfig,
    LadderVerdict,
    Level,
    evaluate_attack,
    honest_false_positive_rate,
    plan_attack,
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
    # Trajectory monitor (0hp7)
    "TrajectoryMonitorConfig",
    "MonitorVerdict",
    "monitor_trajectory",
    "trajectory_monitor_catch_rate",
    "trajectory_monitor_false_positive_rate",
    "mean_realized_mass_on_catch",
    # Decomposition ladder (njvs)
    "LadderConfig",
    "Level",
    "AttackPlan",
    "LadderVerdict",
    "plan_attack",
    "evaluate_attack",
    "honest_false_positive_rate",
    # Attacker economics (7bwz)
    "EconomicsConfig",
    "attacker_cost",
    "is_deterred",
    "deterrence_threshold",
    "false_positive_rate",
    "fp_threshold",
    "is_defensible",
    "critical_unit_cost",
]
