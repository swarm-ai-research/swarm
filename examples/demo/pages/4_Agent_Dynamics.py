"""Page 4: Agent Dynamics — Track individual agent trajectories."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Agent Dynamics", page_icon="🤖", layout="wide")
st.title("Agent Dynamics")
st.markdown(
    "Run a simulation and explore per-agent reputation and payoff trajectories "
    "over time. Observe how different agent types fare under governance rules."
)

# ── Controls ──────────────────────────────────────────────────────────────────

st.sidebar.header("Agent Composition")
n_honest = st.sidebar.slider("Honest", 0, 10, 3, key="dyn_honest")
n_opportunistic = st.sidebar.slider("Opportunistic", 0, 10, 2, key="dyn_opp")
n_deceptive = st.sidebar.slider("Deceptive", 0, 10, 1, key="dyn_dec")
n_adversarial = st.sidebar.slider("Adversarial", 0, 10, 1, key="dyn_adv")

st.sidebar.header("Governance")
tax_rate = st.sidebar.slider("Tax rate", 0.0, 0.5, 0.05, 0.01, key="dyn_tax")
reputation_decay = st.sidebar.slider(
    "Reputation decay", 0.8, 1.0, 0.95, 0.01, key="dyn_decay"
)
circuit_breaker = st.sidebar.checkbox("Circuit breaker", key="dyn_cb")

st.sidebar.header("Simulation")
n_epochs = st.sidebar.slider("Epochs", 10, 50, 30, key="dyn_epochs")
seed = st.sidebar.number_input("Seed", value=42, min_value=0, step=1, key="dyn_seed")

run_btn = st.button("Run Simulation", type="primary")


@st.cache_data(show_spinner="Running simulation...", max_entries=16)
def _run(**kwargs):
    from demo.utils.simulation import run_custom

    return run_custom(**kwargs)


if run_btn:
    st.session_state["dyn_result"] = _run(
        n_honest=n_honest,
        n_opportunistic=n_opportunistic,
        n_deceptive=n_deceptive,
        n_adversarial=n_adversarial,
        n_epochs=n_epochs,
        steps_per_epoch=10,
        tax_rate=tax_rate,
        reputation_decay=reputation_decay,
        circuit_breaker_enabled=circuit_breaker,
        freeze_threshold=0.7,
        seed=int(seed),
    )

result = st.session_state.get("dyn_result")

if result is None:
    st.info("Configure parameters and click **Run Simulation**.")
    st.stop()

history = result.get("history")

# ── Reputation trajectories ───────────────────────────────────────────────────

st.subheader("Reputation Trajectories")

if history and hasattr(history, "agent_snapshots") and history.agent_snapshots:
    from demo.utils.charts import reputation_trajectories  # noqa: E402

    st.plotly_chart(
        reputation_trajectories(history, height=450), use_container_width=True
    )
else:
    st.warning(
        "No trajectory data available. The aggregator may not have captured snapshots."
    )

# ── Per-agent payoff over time ────────────────────────────────────────────────

st.subheader("Cumulative Payoff by Agent Type")

if history and hasattr(history, "agent_snapshots") and history.agent_snapshots:
    import plotly.graph_objects as go
    from demo.utils.charts import AGENT_COLORS

    fig = go.Figure()
    for agent_id, snapshots in sorted(history.agent_snapshots.items()):
        epochs = [s.epoch for s in snapshots]
        payoffs = [s.total_payoff for s in snapshots]
        agent_type = agent_id.split("_")[0]
        color = AGENT_COLORS.get(agent_type, "#999")
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=payoffs,
                mode="lines",
                name=agent_id,
                line={"color": color},
            )
        )

    fig.update_layout(
        title="Cumulative Payoff Over Time",
        xaxis_title="Epoch",
        yaxis_title="Total Payoff",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Agent detail selector ─────────────────────────────────────────────────────

st.subheader("Individual Agent Detail")

agent_ids = [a["agent_id"] for a in result["agent_states"]]
selected_agent = st.selectbox("Select agent", agent_ids)

if selected_agent and history and hasattr(history, "agent_snapshots"):
    snapshots = history.agent_snapshots.get(selected_agent, [])
    if snapshots:
        import pandas as pd

        rows = []
        for s in snapshots:
            rows.append(
                {
                    "Epoch": s.epoch,
                    "Reputation": round(s.reputation, 3),
                    "Resources": round(s.resources, 3),
                    "Total Payoff": round(s.total_payoff, 3),
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Mini sparkline
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Epoch"],
                y=df["Reputation"],
                mode="lines+markers",
                name="Reputation",
                line={"color": "#0d6efd"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Epoch"],
                y=df["Total Payoff"],
                mode="lines+markers",
                name="Payoff",
                line={"color": "#28a745"},
                yaxis="y2",
            )
        )
        fig.update_layout(
            height=300,
            yaxis={"title": "Reputation", "side": "left"},
            yaxis2={"title": "Payoff", "side": "right", "overlaying": "y"},
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No snapshot data for {selected_agent}.")

# ── Final standings ───────────────────────────────────────────────────────────

st.subheader("Final Standings")
from demo.utils.charts import agent_table_data  # noqa: E402

df = agent_table_data(result["agent_states"])
if not df.empty:
    st.dataframe(df, use_container_width=True, hide_index=True)
