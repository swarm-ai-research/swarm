"""Streamlit dashboard for distributional AGI safety sandbox."""

import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="AGI Safety Sandbox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


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
        st.markdown(
            f"**Status:** <span class='status-{status_class}'>{status}</span>",
            unsafe_allow_html=True,
        )

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
            delta=f"{current_epoch / total_epochs:.0%}",
        )

    st.divider()

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📈 Metrics Over Time")
        metrics_df = pd.DataFrame(data.get("metric_history", []))
        if not metrics_df.empty:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Toxicity Rate", "Quality Gap"),
            )

            fig.add_trace(
                go.Scatter(
                    x=metrics_df["epoch"],
                    y=metrics_df["toxicity_rate"],
                    mode="lines+markers",
                    name="Toxicity",
                    line={"color": "red"},
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=metrics_df["epoch"],
                    y=metrics_df["quality_gap"],
                    mode="lines+markers",
                    name="Quality Gap",
                    line={"color": "blue"},
                ),
                row=2,
                col=1,
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
            if "name" in agent_df.columns:
                agent_df["display_name"] = agent_df["name"].fillna(agent_df["agent_id"])
            else:
                agent_df["display_name"] = agent_df["agent_id"]

            # Agent type distribution
            type_counts = agent_df["agent_type"].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                color=type_counts.index,
                color_discrete_map={
                    "honest": "#28a745",
                    "opportunistic": "#ffc107",
                    "adversarial": "#dc3545",
                    "deceptive": "#6c757d",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agent data available")

    # Incoherence panel
    incoherence_df = pd.DataFrame(data.get("metric_history", []))
    if not incoherence_df.empty and "incoherence_index" in incoherence_df.columns:
        st.divider()
        st.subheader("🧭 Incoherence Panel")
        inco_col1, inco_col2 = st.columns(2)

        with inco_col1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=incoherence_df["epoch"],
                    y=incoherence_df["incoherence_index"],
                    mode="lines+markers",
                    name="Incoherence Index",
                )
            )
            if "forecaster_risk" in incoherence_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=incoherence_df["epoch"],
                        y=incoherence_df["forecaster_risk"],
                        mode="lines+markers",
                        name="Forecaster Risk",
                    )
                )
            fig.update_layout(height=320, xaxis_title="Epoch", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

        with inco_col2:
            scatter_df = incoherence_df.copy()
            if "toxicity_rate" in scatter_df.columns:
                scatter = px.scatter(
                    scatter_df,
                    x="toxicity_rate",
                    y="incoherence_index",
                    color="governance_condition"
                    if "governance_condition" in scatter_df.columns
                    else None,
                    title="Toxicity vs Incoherence",
                )
                scatter.update_layout(height=320)
                st.plotly_chart(scatter, use_container_width=True)

    st.divider()

    # Agent details section
    if show_agents:
        st.subheader("👥 Agent States")
        agents = data.get("agents", [])
        if agents:
            agent_df = pd.DataFrame(agents)
            if "name" in agent_df.columns:
                agent_df["display_name"] = agent_df["name"].fillna(agent_df["agent_id"])
            else:
                agent_df["display_name"] = agent_df["agent_id"]

            # Sortable table
            name_filter = st.text_input("Filter by name:", "")
            if name_filter.strip():
                agent_df = agent_df[
                    agent_df["display_name"].str.contains(
                        name_filter, case=False, na=False
                    )
                ]

            sort_by = st.selectbox(
                "Sort by:",
                ["name", "reputation", "resources", "interactions", "payoff_total"],
            )
            if sort_by == "name":
                agent_df = agent_df.sort_values("display_name", ascending=True)
            else:
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
            fig = px.bar(
                agent_df,
                x="display_name",
                y="reputation",
                color="agent_type",
                color_discrete_map={
                    "honest": "#28a745",
                    "opportunistic": "#ffc107",
                    "adversarial": "#dc3545",
                },
            )
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

        comparison_rows = data.get("condition_comparison", [])
        if comparison_rows:
            st.write("**Governance Condition Comparison:**")
            cmp_df = pd.DataFrame(comparison_rows)
            st.dataframe(cmp_df, use_container_width=True)

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
        st.info(
            "Network visualization requires additional setup. Install networkx and pyvis for full functionality."
        )

    # Event log
    st.divider()
    st.subheader("📋 Recent Events")

    events = data.get("recent_events", [])
    if events:
        for event in events[-10:]:
            event_type = event.get("type", "unknown")
            icon = {"interaction": "🤝", "governance": "⚖️", "boundary": "🔒"}.get(
                event_type, "📌"
            )
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
        metric_history.append(
            {
                "epoch": epoch,
                "step": 10,
                "toxicity_rate": 0.1 + random.uniform(-0.05, 0.1) * (epoch / 25),
                "quality_gap": -0.05 + random.uniform(-0.02, 0.02),
                "avg_payoff": 5.0 + random.uniform(-1, 2),
                "total_welfare": 800 + epoch * 2 + random.uniform(-20, 20),
                "acceptance_rate": 0.7 + random.uniform(-0.1, 0.1),
                "incoherence_index": 0.2 + random.uniform(-0.05, 0.15) * (epoch / 25),
                "forecaster_risk": 0.25 + random.uniform(-0.04, 0.2) * (epoch / 25),
                "governance_condition": "adaptive_on" if epoch > 12 else "static",
            }
        )

    agents = [
        {
            "agent_id": "honest_1",
            "name": "Alice",
            "agent_type": "honest",
            "reputation": 12.5,
            "resources": 105.0,
            "interactions": 45,
            "payoff_total": 52.3,
            "is_frozen": False,
        },
        {
            "agent_id": "honest_2",
            "name": "Bob",
            "agent_type": "honest",
            "reputation": 11.2,
            "resources": 98.0,
            "interactions": 42,
            "payoff_total": 48.1,
            "is_frozen": False,
        },
        {
            "agent_id": "honest_3",
            "name": "Casey",
            "agent_type": "honest",
            "reputation": 10.8,
            "resources": 102.0,
            "interactions": 40,
            "payoff_total": 45.5,
            "is_frozen": False,
        },
        {
            "agent_id": "opp_1",
            "name": "Delta",
            "agent_type": "opportunistic",
            "reputation": 8.5,
            "resources": 115.0,
            "interactions": 38,
            "payoff_total": 55.2,
            "is_frozen": False,
        },
        {
            "agent_id": "opp_2",
            "name": "Echo",
            "agent_type": "opportunistic",
            "reputation": 7.2,
            "resources": 110.0,
            "interactions": 35,
            "payoff_total": 42.0,
            "is_frozen": False,
        },
        {
            "agent_id": "adv_1",
            "name": "Nyx",
            "agent_type": "adversarial",
            "reputation": 3.1,
            "resources": 85.0,
            "interactions": 25,
            "payoff_total": -15.3,
            "is_frozen": True,
        },
    ]

    recent_events = [
        {
            "type": "interaction",
            "description": "honest_1 and honest_2 completed beneficial interaction",
        },
        {
            "type": "governance",
            "description": "Audit conducted on opp_1, no violations found",
        },
        {
            "type": "boundary",
            "description": "External API call from honest_3 to web_search",
        },
        {"type": "governance", "description": "adv_1 frozen due to high toxicity"},
        {
            "type": "interaction",
            "description": "opp_2 proposed interaction rejected by honest_1",
        },
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
        "condition_comparison": [
            {
                "condition": "static",
                "n_points": 12,
                "mean_toxicity_rate": 0.14,
                "mean_incoherence_index": 0.23,
                "mean_forecaster_risk": 0.31,
                "mean_total_welfare": 821.0,
            },
            {
                "condition": "adaptive_on",
                "n_points": 13,
                "mean_toxicity_rate": 0.11,
                "mean_incoherence_index": 0.18,
                "mean_forecaster_risk": 0.67,
                "mean_total_welfare": 864.0,
            },
        ],
        "boundary_crossings": 156,
        "blocked_crossings": 23,
        "leakage_events": 5,
        "avg_sensitivity": 0.35,
        "recent_events": recent_events,
    }


if __name__ == "__main__":
    main()
