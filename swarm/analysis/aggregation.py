"""Epoch-level metric aggregation for visualization."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from swarm.models.interaction import SoftInteraction


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""

    epoch: int
    value: float
    timestamp: Optional[datetime] = None


@dataclass
class AgentSnapshot:
    """Snapshot of an agent's state at a point in time."""

    agent_id: str
    epoch: int
    name: Optional[str] = None
    agent_type: Optional[str] = None
    reputation: float = 0.0
    resources: float = 100.0
    interactions_initiated: int = 0
    interactions_received: int = 0
    avg_p_initiated: float = 0.5
    avg_p_received: float = 0.5
    total_payoff: float = 0.0
    is_frozen: bool = False
    is_quarantined: bool = False


@dataclass
class EpochSnapshot:
    """Complete snapshot of simulation state at end of epoch."""

    epoch: int
    timestamp: datetime = field(default_factory=datetime.now)

    # Interaction metrics
    total_interactions: int = 0
    accepted_interactions: int = 0
    rejected_interactions: int = 0

    # Quality metrics
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    avg_p: float = 0.5
    incoherence_index: float = 0.0
    # Projection-geometric diagnostics (see swarm/metrics/soft_metrics.py).
    quality_correlation: float = 0.0
    baseline_harm: float = 0.0
    selection_credit: float = 0.0
    selection_saturation: float = 0.0

    # Payoff metrics
    total_welfare: float = 0.0
    avg_payoff: float = 0.0
    payoff_std: float = 0.0
    gini_coefficient: float = 0.0

    # Activity metrics
    total_posts: int = 0
    total_votes: int = 0
    total_tasks_completed: int = 0

    # Agent metrics
    n_agents: int = 0
    n_frozen: int = 0
    n_quarantined: int = 0
    avg_reputation: float = 0.0
    reputation_std: float = 0.0

    # Network metrics (optional)
    n_edges: int = 0
    avg_degree: float = 0.0
    avg_clustering: float = 0.0
    n_components: int = 1

    # Security metrics (optional)
    ecosystem_threat_level: float = 0.0
    active_threats: int = 0
    contagion_depth: int = 0

    # Collusion metrics (optional)
    ecosystem_collusion_risk: float = 0.0
    n_flagged_pairs: int = 0

    # Capability metrics (optional)
    avg_coordination_score: float = 0.0
    avg_synergy_score: float = 0.0
    tasks_completed: int = 0


@dataclass
class SimulationHistory:
    """Complete history of a simulation run."""

    simulation_id: str = ""
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Configuration
    n_epochs: int = 0
    steps_per_epoch: int = 0
    n_agents: int = 0
    seed: Optional[int] = None

    # Time series data
    epoch_snapshots: List[EpochSnapshot] = field(default_factory=list)
    agent_snapshots: Dict[str, List[AgentSnapshot]] = field(default_factory=dict)

    # Raw interactions (optional, can be memory intensive)
    interactions: List[SoftInteraction] = field(default_factory=list)

    def add_epoch_snapshot(self, snapshot: EpochSnapshot) -> None:
        """Add an epoch snapshot."""
        self.epoch_snapshots.append(snapshot)

    def add_agent_snapshot(self, snapshot: AgentSnapshot) -> None:
        """Add an agent snapshot."""
        if snapshot.agent_id not in self.agent_snapshots:
            self.agent_snapshots[snapshot.agent_id] = []
        self.agent_snapshots[snapshot.agent_id].append(snapshot)

    def get_time_series(self, metric: str) -> List[TimeSeriesPoint]:
        """Extract a time series for a specific metric."""
        points = []
        for snapshot in self.epoch_snapshots:
            value = getattr(snapshot, metric, None)
            if value is not None:
                points.append(
                    TimeSeriesPoint(
                        epoch=snapshot.epoch,
                        value=float(value),
                        timestamp=snapshot.timestamp,
                    )
                )
        return points

    def get_agent_time_series(
        self,
        agent_id: str,
        metric: str,
    ) -> List[TimeSeriesPoint]:
        """Extract a time series for a specific agent's metric."""
        if agent_id not in self.agent_snapshots:
            return []

        points = []
        for snapshot in self.agent_snapshots[agent_id]:
            value = getattr(snapshot, metric, None)
            if value is not None:
                points.append(
                    TimeSeriesPoint(
                        epoch=snapshot.epoch,
                        value=float(value),
                    )
                )
        return points

    def get_final_agent_states(self) -> Dict[str, AgentSnapshot]:
        """Get the final state of each agent."""
        final_states = {}
        for agent_id, snapshots in self.agent_snapshots.items():
            if snapshots:
                final_states[agent_id] = snapshots[-1]
        return final_states


class MetricsAggregator:
    """
    Aggregates metrics from simulation components for visualization.

    Collects data from orchestrator, governance, network, and security
    modules to create unified snapshots for the dashboard.
    """

    def __init__(self) -> None:
        """Initialize the aggregator."""
        self._history: SimulationHistory = SimulationHistory()
        self._current_epoch_interactions: List[SoftInteraction] = []
        self._agent_epoch_data: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def start_simulation(
        self,
        simulation_id: str,
        n_epochs: int,
        steps_per_epoch: int,
        n_agents: int,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize tracking for a new simulation."""
        self._history = SimulationHistory(
            simulation_id=simulation_id,
            started_at=datetime.now(),
            n_epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            n_agents=n_agents,
            seed=seed,
        )
        self._current_epoch_interactions.clear()
        self._agent_epoch_data.clear()

    def record_interaction(self, interaction: SoftInteraction) -> None:
        """Record an interaction for the current epoch."""
        self._current_epoch_interactions.append(interaction)

        # Track per-agent data
        initiator = interaction.initiator
        counterparty = interaction.counterparty

        if "interactions_initiated" not in self._agent_epoch_data[initiator]:
            self._agent_epoch_data[initiator]["interactions_initiated"] = 0
            self._agent_epoch_data[initiator]["p_initiated"] = []
            self._agent_epoch_data[initiator]["payoffs"] = []

        if "interactions_received" not in self._agent_epoch_data[counterparty]:
            self._agent_epoch_data[counterparty]["interactions_received"] = 0
            self._agent_epoch_data[counterparty]["p_received"] = []

        self._agent_epoch_data[initiator]["interactions_initiated"] += 1
        self._agent_epoch_data[initiator]["p_initiated"].append(interaction.p)

        self._agent_epoch_data[counterparty]["interactions_received"] += 1
        self._agent_epoch_data[counterparty]["p_received"].append(interaction.p)

    def record_payoff(self, agent_id: str, payoff: float) -> None:
        """Record a payoff for an agent."""
        if "payoffs" not in self._agent_epoch_data[agent_id]:
            self._agent_epoch_data[agent_id]["payoffs"] = []
        self._agent_epoch_data[agent_id]["payoffs"].append(payoff)

    def finalize_epoch(
        self,
        epoch: int,
        agent_states: Dict[str, Any],
        frozen_agents: Optional[set] = None,
        quarantined_agents: Optional[set] = None,
        posts_count: int = 0,
        votes_count: int = 0,
        tasks_completed: int = 0,
        network_metrics: Optional[Dict] = None,
        security_report: Optional[Any] = None,
        collusion_report: Optional[Any] = None,
        capability_metrics: Optional[Any] = None,
    ) -> EpochSnapshot:
        """
        Create a snapshot at the end of an epoch.

        Args:
            epoch: Current epoch number
            agent_states: Dict mapping agent_id to AgentState
            frozen_agents: Set of frozen agent IDs
            quarantined_agents: Set of quarantined agent IDs
            posts_count: Number of posts this epoch
            votes_count: Number of votes this epoch
            tasks_completed: Tasks completed this epoch
            network_metrics: Network metrics dict
            security_report: SecurityReport from security lever
            collusion_report: CollusionReport from collusion lever
            capability_metrics: EmergentCapabilityMetrics

        Returns:
            EpochSnapshot with all aggregated metrics
        """
        frozen_agents = frozen_agents or set()
        quarantined_agents = quarantined_agents or set()

        interactions = self._current_epoch_interactions
        n_total = len(interactions)
        n_accepted = sum(1 for i in interactions if i.accepted)

        # Compute quality metrics
        accepted_p = [i.p for i in interactions if i.accepted]
        rejected_p = [i.p for i in interactions if not i.accepted]

        toxicity_rate = 0.0
        if accepted_p:
            toxicity_rate = float(1.0 - np.mean(accepted_p))

        quality_gap = 0.0
        if accepted_p and rejected_p:
            quality_gap = float(np.mean(accepted_p) - np.mean(rejected_p))

        avg_p = float(np.mean([i.p for i in interactions])) if interactions else 0.5

        # Projection-geometric diagnostics. See swarm/metrics/soft_metrics.py:
        #   Q  = E[p|a=1] - E[p|a=0] = Cov(p,a)/Var(a)
        #   ρ  = Q · √(αβ) / σ_p                          ∈ [-1, +1]
        #   T  = (1 - E[p]) - β · Q   (baseline_harm − selection_credit)
        #   |Q| ≤ σ_p / √(αβ)         (Cauchy-Schwarz upper bound)
        quality_correlation = 0.0
        baseline_harm = 0.0
        selection_credit = 0.0
        selection_saturation = 0.0
        if interactions:
            n = len(interactions)
            n_acc = n_accepted
            alpha = n_acc / n
            beta = 1.0 - alpha
            baseline_harm = float(1.0 - avg_p)
            selection_credit = float(beta * quality_gap)
            if n >= 2 and 0 < n_acc < n:
                sigma_p = float(np.std([i.p for i in interactions]))
                ab = alpha * beta
                if sigma_p > 0.0 and ab > 0.0:
                    norm = float(np.sqrt(ab) / sigma_p)
                    quality_correlation = quality_gap * norm
                    selection_saturation = abs(quality_gap) * norm

        incoherence_index = 0.0
        if interactions:
            uncertainties = [1.0 - abs(2 * i.p - 1.0) for i in interactions]
            incoherence_index = float(np.mean(uncertainties))

        # Compute payoff metrics
        all_payoffs = []
        for agent_data in self._agent_epoch_data.values():
            all_payoffs.extend(agent_data.get("payoffs", []))

        total_welfare = sum(all_payoffs)
        avg_payoff = float(np.mean(all_payoffs)) if all_payoffs else 0.0
        payoff_std = float(np.std(all_payoffs)) if len(all_payoffs) > 1 else 0.0
        gini = self._compute_gini(all_payoffs) if all_payoffs else 0.0

        # Compute agent metrics
        n_agents = len(agent_states)
        n_frozen = len(frozen_agents)
        n_quarantined = len(quarantined_agents)

        reputations = [s.reputation for s in agent_states.values()]
        avg_reputation = float(np.mean(reputations)) if reputations else 0.0
        reputation_std = float(np.std(reputations)) if len(reputations) > 1 else 0.0

        # Create epoch snapshot
        snapshot = EpochSnapshot(
            epoch=epoch,
            timestamp=datetime.now(),
            total_interactions=n_total,
            accepted_interactions=n_accepted,
            rejected_interactions=n_total - n_accepted,
            toxicity_rate=toxicity_rate,
            quality_gap=quality_gap,
            quality_correlation=quality_correlation,
            baseline_harm=baseline_harm,
            selection_credit=selection_credit,
            selection_saturation=selection_saturation,
            avg_p=avg_p,
            incoherence_index=incoherence_index,
            total_welfare=total_welfare,
            avg_payoff=avg_payoff,
            payoff_std=payoff_std,
            gini_coefficient=gini,
            total_posts=posts_count,
            total_votes=votes_count,
            total_tasks_completed=tasks_completed,
            n_agents=n_agents,
            n_frozen=n_frozen,
            n_quarantined=n_quarantined,
            avg_reputation=avg_reputation,
            reputation_std=reputation_std,
        )

        # Add network metrics
        if network_metrics:
            snapshot.n_edges = network_metrics.get("n_edges", 0)
            snapshot.avg_degree = network_metrics.get("avg_degree", 0.0)
            snapshot.avg_clustering = network_metrics.get("avg_clustering", 0.0)
            snapshot.n_components = network_metrics.get("n_components", 1)

        # Add security metrics
        if security_report:
            snapshot.ecosystem_threat_level = getattr(
                security_report, "ecosystem_threat_level", 0.0
            )
            snapshot.active_threats = getattr(security_report, "active_threat_count", 0)
            snapshot.contagion_depth = getattr(security_report, "contagion_depth", 0)

        # Add collusion metrics
        if collusion_report:
            snapshot.ecosystem_collusion_risk = getattr(
                collusion_report, "ecosystem_collusion_risk", 0.0
            )
            snapshot.n_flagged_pairs = getattr(collusion_report, "n_flagged_pairs", 0)

        # Add capability metrics
        if capability_metrics:
            snapshot.avg_coordination_score = getattr(
                capability_metrics, "avg_coordination_score", 0.0
            )
            snapshot.avg_synergy_score = getattr(
                capability_metrics, "avg_synergy_score", 0.0
            )
            snapshot.tasks_completed = getattr(capability_metrics, "tasks_completed", 0)

        self._history.add_epoch_snapshot(snapshot)

        # Create agent snapshots
        for agent_id, state in agent_states.items():
            agent_data = self._agent_epoch_data.get(agent_id, {})

            p_initiated = agent_data.get("p_initiated", [])
            p_received = agent_data.get("p_received", [])
            payoffs = agent_data.get("payoffs", [])

            agent_snapshot = AgentSnapshot(
                agent_id=agent_id,
                epoch=epoch,
                name=state.name,
                agent_type=state.agent_type.value if hasattr(state, 'agent_type') and state.agent_type else None,
                reputation=state.reputation,
                resources=state.resources,
                interactions_initiated=agent_data.get("interactions_initiated", 0),
                interactions_received=agent_data.get("interactions_received", 0),
                avg_p_initiated=float(np.mean(p_initiated)) if p_initiated else 0.5,
                avg_p_received=float(np.mean(p_received)) if p_received else 0.5,
                total_payoff=sum(payoffs),
                is_frozen=agent_id in frozen_agents,
                is_quarantined=agent_id in quarantined_agents,
            )
            self._history.add_agent_snapshot(agent_snapshot)

        # Clear epoch data
        self._current_epoch_interactions.clear()
        self._agent_epoch_data.clear()

        return snapshot

    def end_simulation(self) -> SimulationHistory:
        """Mark simulation as complete and return history."""
        self._history.ended_at = datetime.now()
        return self._history

    def get_history(self) -> SimulationHistory:
        """Get the current simulation history."""
        return self._history

    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient for inequality measurement."""
        if not values or len(values) < 2:
            return 0.0

        values = sorted(values)
        n = len(values)
        cumulative = np.cumsum(values)
        return (
            (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
            if cumulative[-1] != 0
            else 0.0
        )


def compute_rolling_average(
    points: List[TimeSeriesPoint],
    window: int = 5,
) -> List[TimeSeriesPoint]:
    """Compute rolling average of a time series."""
    if len(points) < window:
        return points

    values = [p.value for p in points]
    rolling = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        rolling.append(float(np.mean(values[start : i + 1])))

    return [
        TimeSeriesPoint(epoch=points[i].epoch, value=rolling[i])
        for i in range(len(points))
    ]


def compute_trend(points: List[TimeSeriesPoint]) -> Tuple[float, float]:
    """
    Compute trend (slope) and R-squared for a time series.

    Returns:
        Tuple of (slope, r_squared)
    """
    if len(points) < 2:
        return 0.0, 0.0

    x = np.array([p.epoch for p in points])
    y = np.array([p.value for p in points])

    # Check for constant values
    if np.std(y) == 0:
        return 0.0, 1.0

    try:
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        # Compute R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res: float = float(np.sum((y - y_pred) ** 2))
        ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

        return float(slope), float(r_squared)
    except np.linalg.LinAlgError:
        return 0.0, 0.0


def aggregate_incoherence_scaling(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Aggregate replay-level scaling records by horizon/branching tiers.

    Expected fields per record:
    - horizon_tier
    - branching_tier
    - incoherence_index
    - error_rate
    - disagreement_rate
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("horizon_tier", "unknown")),
            str(record.get("branching_tier", "unknown")),
        )
        grouped[key].append(record)

    result: List[Dict[str, Any]] = []
    for (horizon_tier, branching_tier), group in sorted(grouped.items()):
        result.append(
            {
                "horizon_tier": horizon_tier,
                "branching_tier": branching_tier,
                "n_runs": len(group),
                "mean_incoherence_index": float(
                    np.mean([g.get("incoherence_index", 0.0) for g in group])
                ),
                "mean_error_rate": float(
                    np.mean([g.get("error_rate", 0.0) for g in group])
                ),
                "mean_disagreement_rate": float(
                    np.mean([g.get("disagreement_rate", 0.0) for g in group])
                ),
            }
        )

    return result


def build_scaling_curve_points(
    aggregated_rows: List[Dict[str, Any]],
    x_axis: str,
    fixed_tier: str,
) -> Dict[str, Any]:
    """
    Build curve-ready points from aggregated incoherence rows.

    Args:
        aggregated_rows: Output of aggregate_incoherence_scaling
        x_axis: "horizon" or "branching"
        fixed_tier: fixed tier on the opposite axis
    """
    if x_axis not in {"horizon", "branching"}:
        raise ValueError("x_axis must be 'horizon' or 'branching'")

    x_key = "horizon_tier" if x_axis == "horizon" else "branching_tier"
    fixed_key = "branching_tier" if x_axis == "horizon" else "horizon_tier"

    rows = [row for row in aggregated_rows if row.get(fixed_key) == fixed_tier]
    rows = sorted(rows, key=lambda row: str(row.get(x_key, "")))

    return {
        "x_labels": [str(row.get(x_key, "")) for row in rows],
        "incoherence_index": [
            float(row.get("mean_incoherence_index", 0.0)) for row in rows
        ],
        "error_rate": [float(row.get("mean_error_rate", 0.0)) for row in rows],
        "disagreement_rate": [
            float(row.get("mean_disagreement_rate", 0.0)) for row in rows
        ],
    }
