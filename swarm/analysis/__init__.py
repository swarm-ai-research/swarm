"""Analysis tools for simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from swarm.analysis.adverse_selection import AdverseSelectionDetector
from swarm.analysis.aggregation import AgentSnapshot
from swarm.analysis.dashboard import (
    DashboardConfig,
    DashboardState,
    MetricSnapshot,
    create_dashboard_file,
    extract_agent_snapshots,
    extract_metrics_from_orchestrator,
    run_dashboard,
)
from swarm.analysis.delm_hillclimb import (
    HillClimbConfig,
    HillClimbResult,
    Objective,
    build_adaptive_policy_objective,
    build_governance_objective,
    plot_trajectory,
    run_delm_hillclimb,
)
from swarm.analysis.dolt_export import export_run_summary_to_dolt, export_to_dolt
from swarm.analysis.enhanced_dashboard import (
    plot_enhanced_dashboard,
    plot_enhanced_line,
    plot_multi_scenario_dashboard,
)
from swarm.analysis.phylogeny import generate_phylogeny
from swarm.analysis.sweep import (
    SweepConfig,
    SweepParameter,
    SweepResult,
    SweepRunner,
    quick_sweep,
)
from swarm.analysis.trust_network import (
    TrustNetworkAnalyzer,
    plot_trust_network,
)

if TYPE_CHECKING:
    from swarm.analysis.theme import (
        COLORS as COLORS,
    )
    from swarm.analysis.theme import (
        SWARM_LIGHT_STYLE as SWARM_LIGHT_STYLE,
    )
    from swarm.analysis.theme import (
        SWARM_STYLE as SWARM_STYLE,
    )
    from swarm.analysis.theme import (
        agent_color as agent_color,
    )
    from swarm.analysis.theme import (
        apply_theme as apply_theme,
    )
    from swarm.analysis.theme import (
        metric_color as metric_color,
    )
    from swarm.analysis.theme import (
        swarm_theme as swarm_theme,
    )

__all__ = [
    # Adverse selection
    "AdverseSelectionDetector",
    # Sweep
    "SweepConfig",
    "SweepParameter",
    "SweepResult",
    "SweepRunner",
    "quick_sweep",
    # DeLM hill-climbing optimizer
    "HillClimbConfig",
    "HillClimbResult",
    "Objective",
    "build_governance_objective",
    "build_adaptive_policy_objective",
    "plot_trajectory",
    "run_delm_hillclimb",
    # Dashboard
    "DashboardConfig",
    "DashboardState",
    "MetricSnapshot",
    "AgentSnapshot",
    "extract_metrics_from_orchestrator",
    "extract_agent_snapshots",
    "create_dashboard_file",
    "run_dashboard",
    # Phylogeny
    "generate_phylogeny",
    # Dolt export
    "export_to_dolt",
    "export_run_summary_to_dolt",
    # Enhanced dashboard
    "plot_enhanced_dashboard",
    "plot_enhanced_line",
    "plot_multi_scenario_dashboard",
    # Trust network
    "TrustNetworkAnalyzer",
    "plot_trust_network",
    # Theme & colors
    "COLORS",
    "SWARM_STYLE",
    "SWARM_LIGHT_STYLE",
    "apply_theme",
    "swarm_theme",
    "agent_color",
    "metric_color",
]


def __getattr__(name: str) -> object:
    """Lazy-load optional symbols so heavy deps are not required at import time."""
    _theme_names = {
        "COLORS", "SWARM_STYLE", "SWARM_LIGHT_STYLE",
        "apply_theme", "swarm_theme", "agent_color", "metric_color",
    }
    if name in _theme_names:
        from swarm.analysis import theme  # noqa: F811

        return getattr(theme, name)

    _evolver_names = {
        "EvolverConfig", "EvolutionResult", "run_evolution",
        "GovernanceOrganism", "GovernanceEvaluator",
        "GovernanceEvaluationResult", "SimulationFailureCase",
        "RandomGovernanceMutator", "LLMGovernanceMutator",
        "compute_fitness",
    }
    if name in _evolver_names:
        from swarm.analysis import evolver

        return getattr(evolver, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
