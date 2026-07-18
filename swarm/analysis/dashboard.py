"""Streamlit dashboard for simulation visualization.

This module provides a real-time dashboard for visualizing simulation metrics,
agent states, and ecosystem dynamics.

Usage:
    streamlit run src/analysis/dashboard.py

Or programmatically:
    from swarm.analysis.dashboard import create_dashboard
    create_dashboard(orchestrator)
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""

    title: str = "Distributional AGI Safety Sandbox"
    refresh_rate_ms: int = 1000
    show_agent_details: bool = True
    show_network_graph: bool = True
    show_governance_metrics: bool = True
    show_boundary_metrics: bool = True
    max_history_points: int = 100
    theme: str = "default"


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""

    epoch: int
    step: int
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    avg_payoff: float = 0.0
    total_welfare: float = 0.0
    acceptance_rate: float = 0.0
    governance_costs: float = 0.0
    incoherence_index: float = 0.0
    forecaster_risk: float = 0.0
    governance_condition: str = "static"
    boundary_crossings: int = 0
    leakage_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "toxicity_rate": self.toxicity_rate,
            "quality_gap": self.quality_gap,
            "avg_payoff": self.avg_payoff,
            "total_welfare": self.total_welfare,
            "acceptance_rate": self.acceptance_rate,
            "governance_costs": self.governance_costs,
            "incoherence_index": self.incoherence_index,
            "forecaster_risk": self.forecaster_risk,
            "governance_condition": self.governance_condition,
            "boundary_crossings": self.boundary_crossings,
            "leakage_events": self.leakage_events,
        }


@dataclass
class AgentSnapshot:
    """Snapshot of an agent's state."""

    agent_id: str
    agent_type: str
    name: Optional[str] = None
    reputation: float = 0.0
    resources: float = 0.0
    interactions: int = 0
    payoff_total: float = 0.0
    is_frozen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "name": self.name,
            "reputation": self.reputation,
            "resources": self.resources,
            "interactions": self.interactions,
            "payoff_total": self.payoff_total,
            "is_frozen": self.is_frozen,
        }


class DashboardState:
    """Manages dashboard state and history."""

    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize dashboard state.

        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.metric_history: List[MetricSnapshot] = []
        self.agent_snapshots: Dict[str, AgentSnapshot] = {}
        self.events: List[Dict[str, Any]] = []
        self.is_running: bool = False
        self.current_epoch: int = 0
        self.current_step: int = 0

    def update_metrics(self, snapshot: MetricSnapshot) -> None:
        """Add a new metric snapshot."""
        self.metric_history.append(snapshot)
        self.current_epoch = snapshot.epoch
        self.current_step = snapshot.step

        # Trim history if needed
        if len(self.metric_history) > self.config.max_history_points:
            self.metric_history = self.metric_history[-self.config.max_history_points :]

    def update_agent(self, snapshot: AgentSnapshot) -> None:
        """Update an agent's snapshot."""
        self.agent_snapshots[snapshot.agent_id] = snapshot

    def add_event(self, event: Dict[str, Any]) -> None:
        """Add an event to the log."""
        self.events.append(event)

        # Keep only recent events
        max_events = 100
        if len(self.events) > max_events:
            self.events = self.events[-max_events:]

    def get_metric_series(self, metric_name: str) -> Tuple[List[int], List[float]]:
        """Get time series data for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Tuple of (epochs, values)
        """
        epochs = []
        values = []

        for snapshot in self.metric_history:
            epochs.append(snapshot.epoch)
            value = getattr(snapshot, metric_name, 0.0)
            values.append(value)

        return epochs, values

    def get_incoherence_series(self) -> Tuple[List[int], List[float]]:
        """Get incoherence index time series."""
        return self.get_metric_series("incoherence_index")

    def get_condition_comparison(self) -> List[Dict[str, Any]]:
        """Aggregate history by governance condition for comparison panels."""
        return create_condition_comparison_data(self.metric_history)

    def get_agent_rankings(self, sort_by: str = "reputation") -> List[AgentSnapshot]:
        """Get agents sorted by a metric.

        Args:
            sort_by: Metric to sort by

        Returns:
            Sorted list of agent snapshots
        """
        agents = list(self.agent_snapshots.values())
        if sort_by == "name":
            return sorted(
                agents,
                key=lambda a: (a.name or a.agent_id).lower(),
            )
        return sorted(agents, key=lambda a: getattr(a, sort_by, 0), reverse=True)

    def export_to_json(self) -> str:
        """Export state to JSON."""
        return json.dumps(
            {
                "config": {
                    "title": self.config.title,
                    "max_history_points": self.config.max_history_points,
                },
                "current_epoch": self.current_epoch,
                "current_step": self.current_step,
                "metric_history": [m.to_dict() for m in self.metric_history],
                "agents": [a.to_dict() for a in self.agent_snapshots.values()],
                "recent_events": self.events[-20:],
            },
            indent=2,
        )


def extract_metrics_from_orchestrator(orchestrator: Any) -> MetricSnapshot:
    """Extract current metrics from an orchestrator.

    Args:
        orchestrator: The orchestrator instance

    Returns:
        MetricSnapshot with current values
    """
    state = orchestrator.state
    epoch_metrics = (
        orchestrator._epoch_metrics[-1] if orchestrator._epoch_metrics else None
    )

    # Get basic metrics
    toxicity = epoch_metrics.toxicity_rate if epoch_metrics else 0.0
    quality_gap = epoch_metrics.quality_gap if epoch_metrics else 0.0
    avg_payoff = epoch_metrics.avg_payoff if epoch_metrics else 0.0
    total_welfare = epoch_metrics.total_welfare if epoch_metrics else 0.0

    # Calculate acceptance rate
    total = len(state.completed_interactions)
    accepted = sum(1 for i in state.completed_interactions if i.accepted)
    acceptance_rate = accepted / total if total > 0 else 0.0

    # Get governance costs
    governance_costs = 0.0
    if orchestrator.governance_engine:
        for i in state.completed_interactions[-10:]:  # Recent interactions
            governance_costs += i.governance_cost_initiator or 0.0
            governance_costs += i.governance_cost_counterparty or 0.0

    # Get boundary metrics
    boundary_crossings = 0
    leakage_events = 0
    if orchestrator.flow_tracker:
        summary = orchestrator.flow_tracker.get_summary()
        boundary_crossings = summary.total_flows
    if orchestrator.leakage_detector:
        leakage_events = len(orchestrator.leakage_detector.events)

    # Incoherence proxy from recent interactions (higher near p=0.5)
    recent = state.completed_interactions[-20:]
    incoherence_index = 0.0
    if recent:
        uncertainties = [1.0 - abs(2 * i.p - 1.0) for i in recent]
        incoherence_index = sum(uncertainties) / len(uncertainties)

    forecaster_risk = 0.0
    governance_condition = "static"
    if orchestrator.governance_engine and hasattr(
        orchestrator.governance_engine, "get_adaptive_status"
    ):
        status = orchestrator.governance_engine.get_adaptive_status()
        forecaster_risk = float(status.get("predicted_risk", 0.0))
        if status.get("adaptive_enabled", False):
            governance_condition = (
                "adaptive_on"
                if status.get("variance_levers_active", False)
                else "adaptive_off"
            )

    return MetricSnapshot(
        epoch=state.current_epoch,
        step=state.current_step,
        toxicity_rate=toxicity,
        quality_gap=quality_gap,
        avg_payoff=avg_payoff,
        total_welfare=total_welfare,
        acceptance_rate=acceptance_rate,
        governance_costs=governance_costs,
        incoherence_index=incoherence_index,
        forecaster_risk=forecaster_risk,
        governance_condition=governance_condition,
        boundary_crossings=boundary_crossings,
        leakage_events=leakage_events,
    )


def extract_agent_snapshots(orchestrator: Any) -> List[AgentSnapshot]:
    """Extract agent snapshots from orchestrator.

    Args:
        orchestrator: The orchestrator instance

    Returns:
        List of agent snapshots
    """
    snapshots = []

    for agent_id, _agent in orchestrator._agents.items():
        agent_state = orchestrator.state.get_agent(agent_id)
        if agent_state:
            # Calculate total interactions from initiated + received
            total_interactions = (
                agent_state.interactions_initiated + agent_state.interactions_received
            )
            snapshots.append(
                AgentSnapshot(
                    agent_id=agent_id,
                    agent_type=agent_state.agent_type.value,
                    name=agent_state.name,
                    reputation=agent_state.reputation,
                    resources=agent_state.resources,
                    interactions=total_interactions,
                    payoff_total=agent_state.total_payoff,
                    is_frozen=orchestrator.state.is_agent_frozen(agent_id),
                )
            )

    return snapshots


def extract_incoherence_agent_profiles(orchestrator: Any) -> List[Dict[str, Any]]:
    """
    Extract per-agent incoherence proxy profiles from completed interactions.

    Returns:
        List of dict rows with agent_id, agent_type, and incoherence_index.
    """
    by_agent: Dict[str, List[float]] = defaultdict(list)
    for interaction in orchestrator.state.completed_interactions:
        uncertainty = 1.0 - abs(2 * interaction.p - 1.0)
        by_agent[interaction.initiator].append(uncertainty)

    rows: List[Dict[str, Any]] = []
    for agent_id, values in by_agent.items():
        agent_state = orchestrator.state.get_agent(agent_id)
        if agent_state is None:
            continue
        rows.append(
            {
                "agent_id": agent_id,
                "agent_type": agent_state.agent_type.value,
                "name": agent_state.name,
                "incoherence_index": sum(values) / len(values) if values else 0.0,
                "n_interactions": len(values),
            }
        )
    return rows


def create_incoherence_panel_data(state: DashboardState) -> Dict[str, List[Any]]:
    """
    Build chart-ready incoherence panel data from dashboard state history.
    """
    epochs, incoherence = state.get_metric_series("incoherence_index")
    _, risk = state.get_metric_series("forecaster_risk")
    return {
        "epochs": epochs,
        "incoherence_index": incoherence,
        "forecaster_risk": risk,
    }


def create_condition_comparison_data(
    metric_history: List[MetricSnapshot],
) -> List[Dict[str, Any]]:
    """
    Build governance-condition comparison rows from metric history snapshots.
    """
    grouped: Dict[str, List[MetricSnapshot]] = defaultdict(list)
    for snapshot in metric_history:
        grouped[snapshot.governance_condition].append(snapshot)

    rows: List[Dict[str, Any]] = []
    for condition, snapshots in grouped.items():
        n = len(snapshots)
        rows.append(
            {
                "condition": condition,
                "n_points": n,
                "mean_toxicity_rate": sum(s.toxicity_rate for s in snapshots) / n,
                "mean_incoherence_index": sum(s.incoherence_index for s in snapshots)
                / n,
                "mean_forecaster_risk": sum(s.forecaster_risk for s in snapshots) / n,
                "mean_total_welfare": sum(s.total_welfare for s in snapshots) / n,
            }
        )
    return sorted(rows, key=lambda row: row["condition"])


# =============================================================================
# Streamlit Dashboard Implementation
# =============================================================================

STREAMLIT_APP_CODE = '''
"""Streamlit dashboard for distributional AGI safety sandbox."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path
from typing import Optional

# Page config
st.set_page_config(
    page_title="AGI Safety Sandbox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .agent-honest { color: #28a745; }
    .agent-opportunistic { color: #ffc107; }
    .agent-adversarial { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard entry point."""
    st.title("🔬 Distributional AGI Safety Sandbox")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # Simulation status
        if "simulation_running" not in st.session_state:
            st.session_state.simulation_running = False

        status = "Running" if st.session_state.simulation_running else "Stopped"
        status_class = "running" if st.session_state.simulation_running else "stopped"
        st.markdown(f"**Status:** <span class='status-{status_class}'>{status}</span>",
                   unsafe_allow_html=True)

        # Load data options
        st.subheader("Data Source")
        data_source = st.radio(
            "Select data source:",
            ["Demo Data", "Load from File", "Live Simulation"],
        )

        if data_source == "Load from File":
            uploaded_file = st.file_uploader("Upload metrics JSON", type=["json"])
            if uploaded_file:
                data = json.load(uploaded_file)
                st.session_state.dashboard_data = data

        # Refresh rate
        refresh_rate = st.slider("Refresh Rate (s)", 1, 10, 2)

        # Display options
        st.subheader("Display Options")
        show_agents = st.checkbox("Show Agent Details", value=True)
        show_governance = st.checkbox("Show Governance", value=True)
        show_boundaries = st.checkbox("Show Boundaries", value=True)
        show_network = st.checkbox("Show Network", value=False)

    # Main content
    # Generate demo data if needed
    if "dashboard_data" not in st.session_state:
        st.session_state.dashboard_data = generate_demo_data()

    data = st.session_state.dashboard_data

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_toxicity = data.get("current_toxicity", 0.15)
        st.metric(
            "Toxicity Rate",
            f"{current_toxicity:.2%}",
            delta=f"{(current_toxicity - 0.1):.2%}",
            delta_color="inverse",
        )

    with col2:
        current_welfare = data.get("current_welfare", 850.0)
        st.metric(
            "Total Welfare",
            f"{current_welfare:.0f}",
            delta=f"+{current_welfare * 0.05:.0f}",
        )

    with col3:
        acceptance_rate = data.get("acceptance_rate", 0.72)
        st.metric(
            "Acceptance Rate",
            f"{acceptance_rate:.2%}",
        )

    with col4:
        current_epoch = data.get("current_epoch", 25)
        total_epochs = data.get("total_epochs", 100)
        st.metric(
            "Progress",
            f"{current_epoch}/{total_epochs}",
            delta=f"{current_epoch/total_epochs:.0%}",
        )

    st.divider()

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📈 Metrics Over Time")
        metrics_df = pd.DataFrame(data.get("metric_history", []))
        if not metrics_df.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Toxicity Rate", "Quality Gap"))

            fig.add_trace(
                go.Scatter(x=metrics_df["epoch"], y=metrics_df["toxicity_rate"],
                          mode="lines+markers", name="Toxicity", line=dict(color="red")),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=metrics_df["epoch"], y=metrics_df["quality_gap"],
                          mode="lines+markers", name="Quality Gap", line=dict(color="blue")),
                row=2, col=1
            )

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metric history available")

    with chart_col2:
        st.subheader("📊 Agent Distribution")
        agents = data.get("agents", [])
        if agents:
            agent_df = pd.DataFrame(agents)

            # Agent type distribution
            type_counts = agent_df["agent_type"].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        color=type_counts.index,
                        color_discrete_map={
                            "honest": "#28a745",
                            "opportunistic": "#ffc107",
                            "adversarial": "#dc3545",
                            "deceptive": "#6c757d",
                        })
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent data available")

    st.divider()

    # Agent details section
    if show_agents:
        st.subheader("👥 Agent States")
        agents = data.get("agents", [])
        if agents:
            agent_df = pd.DataFrame(agents)

            # Sortable table
            sort_by = st.selectbox("Sort by:", ["reputation", "resources", "interactions", "payoff_total"])
            agent_df = agent_df.sort_values(sort_by, ascending=False)

            # Style the dataframe
            def color_agent_type(val):
                colors = {
                    "honest": "background-color: #d4edda",
                    "opportunistic": "background-color: #fff3cd",
                    "adversarial": "background-color: #f8d7da",
                    "deceptive": "background-color: #e2e3e5",
                }
                return colors.get(val, "")

            styled_df = agent_df.style.map(color_agent_type, subset=["agent_type"])
            st.dataframe(styled_df, use_container_width=True)

            # Reputation bar chart
            fig = px.bar(agent_df, x="agent_id", y="reputation",
                        color="agent_type",
                        color_discrete_map={
                            "honest": "#28a745",
                            "opportunistic": "#ffc107",
                            "adversarial": "#dc3545",
                        })
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Governance section
    if show_governance:
        st.divider()
        st.subheader("⚖️ Governance Metrics")

        gov_col1, gov_col2, gov_col3 = st.columns(3)

        with gov_col1:
            governance_costs = data.get("governance_costs", 45.0)
            st.metric("Total Governance Costs", f"${governance_costs:.2f}")

        with gov_col2:
            audits = data.get("audits_conducted", 12)
            st.metric("Audits Conducted", audits)

        with gov_col3:
            frozen_agents = data.get("frozen_agents", 1)
            st.metric("Frozen Agents", frozen_agents)

        # Governance effectiveness
        gov_metrics = data.get("governance_effectiveness", {})
        if gov_metrics:
            st.write("**Governance Effectiveness:**")
            eff_df = pd.DataFrame([gov_metrics])
            st.dataframe(eff_df, use_container_width=True)

    # Boundaries section
    if show_boundaries:
        st.divider()
        st.subheader("🔒 Boundary Metrics")

        bound_col1, bound_col2, bound_col3, bound_col4 = st.columns(4)

        with bound_col1:
            crossings = data.get("boundary_crossings", 156)
            st.metric("Boundary Crossings", crossings)

        with bound_col2:
            blocked = data.get("blocked_crossings", 23)
            st.metric("Blocked Attempts", blocked)

        with bound_col3:
            leakage = data.get("leakage_events", 5)
            st.metric("Leakage Events", leakage, delta_color="inverse")

        with bound_col4:
            sensitivity = data.get("avg_sensitivity", 0.35)
            st.metric("Avg Sensitivity", f"{sensitivity:.2f}")

    # Network visualization (placeholder)
    if show_network:
        st.divider()
        st.subheader("🌐 Agent Network")
        st.info("Network visualization requires additional setup. Install networkx and pyvis for full functionality.")

    # Event log
    st.divider()
    st.subheader("📋 Recent Events")

    events = data.get("recent_events", [])
    if events:
        for event in events[-10:]:
            event_type = event.get("type", "unknown")
            icon = {"interaction": "🤝", "governance": "⚖️", "boundary": "🔒"}.get(event_type, "📌")
            st.text(f"{icon} {event.get('description', 'Event occurred')}")
    else:
        st.info("No recent events")

    # Footer
    st.divider()
    st.caption("Distributional AGI Safety Sandbox | Real-time Monitoring Dashboard")

    # Auto-refresh
    if st.session_state.simulation_running:
        time.sleep(refresh_rate)
        st.rerun()


def generate_demo_data():
    """Generate demo data for testing the dashboard."""
    import random

    epochs = list(range(1, 26))
    metric_history = []

    for epoch in epochs:
        metric_history.append({
            "epoch": epoch,
            "step": 10,
            "toxicity_rate": 0.1 + random.uniform(-0.05, 0.1) * (epoch / 25),
            "quality_gap": -0.05 + random.uniform(-0.02, 0.02),
            "avg_payoff": 5.0 + random.uniform(-1, 2),
            "total_welfare": 800 + epoch * 2 + random.uniform(-20, 20),
            "acceptance_rate": 0.7 + random.uniform(-0.1, 0.1),
        })

    agents = [
        {"agent_id": "honest_1", "agent_type": "honest", "reputation": 12.5, "resources": 105.0, "interactions": 45, "payoff_total": 52.3, "is_frozen": False},
        {"agent_id": "honest_2", "agent_type": "honest", "reputation": 11.2, "resources": 98.0, "interactions": 42, "payoff_total": 48.1, "is_frozen": False},
        {"agent_id": "honest_3", "agent_type": "honest", "reputation": 10.8, "resources": 102.0, "interactions": 40, "payoff_total": 45.5, "is_frozen": False},
        {"agent_id": "opp_1", "agent_type": "opportunistic", "reputation": 8.5, "resources": 115.0, "interactions": 38, "payoff_total": 55.2, "is_frozen": False},
        {"agent_id": "opp_2", "agent_type": "opportunistic", "reputation": 7.2, "resources": 110.0, "interactions": 35, "payoff_total": 42.0, "is_frozen": False},
        {"agent_id": "adv_1", "agent_type": "adversarial", "reputation": 3.1, "resources": 85.0, "interactions": 25, "payoff_total": -15.3, "is_frozen": True},
    ]

    recent_events = [
        {"type": "interaction", "description": "honest_1 and honest_2 completed beneficial interaction"},
        {"type": "governance", "description": "Audit conducted on opp_1, no violations found"},
        {"type": "boundary", "description": "External API call from honest_3 to web_search"},
        {"type": "governance", "description": "adv_1 frozen due to high toxicity"},
        {"type": "interaction", "description": "opp_2 proposed interaction rejected by honest_1"},
    ]

    return {
        "current_epoch": 25,
        "total_epochs": 100,
        "current_toxicity": metric_history[-1]["toxicity_rate"],
        "current_welfare": metric_history[-1]["total_welfare"],
        "acceptance_rate": metric_history[-1]["acceptance_rate"],
        "metric_history": metric_history,
        "agents": agents,
        "governance_costs": 45.0,
        "audits_conducted": 12,
        "frozen_agents": 1,
        "governance_effectiveness": {
            "precision": 0.85,
            "recall": 0.72,
            "f1_score": 0.78,
        },
        "boundary_crossings": 156,
        "blocked_crossings": 23,
        "leakage_events": 5,
        "avg_sensitivity": 0.35,
        "recent_events": recent_events,
    }


if __name__ == "__main__":
    main()
'''


def create_dashboard_file(output_path: Optional[str] = None) -> str:
    """Create a standalone Streamlit dashboard file.

    Args:
        output_path: Path to write the file (default: src/analysis/streamlit_app.py)

    Returns:
        Path to the created file
    """
    from pathlib import Path

    module_dir = Path(__file__).parent
    source_app_path = module_dir / "streamlit_app.py"
    if output_path is None:
        output_path = str(source_app_path)

    # Prefer copying the maintained streamlit app file if it exists.
    if source_app_path.exists():
        with open(source_app_path) as src:
            app_code = src.read()
    else:
        app_code = STREAMLIT_APP_CODE

    with open(output_path, "w") as f:
        f.write(app_code)

    return output_path


def run_dashboard(
    orchestrator: Optional[Any] = None,
    port: int = 8501,
    open_browser: bool = True,
) -> None:
    """Run the Streamlit dashboard.

    Args:
        orchestrator: Optional orchestrator to connect to
        port: Port to run on
        open_browser: Whether to open browser automatically
    """
    import subprocess
    import sys

    # Create the dashboard file
    dashboard_path = create_dashboard_file()

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        dashboard_path,
        "--server.port",
        str(port),
    ]

    if not open_browser:
        cmd.extend(["--server.headless", "true"])

    # Run Streamlit
    subprocess.run(cmd)
