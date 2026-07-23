"""Page 1: Overview — Run baseline scenario and inspect results."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("Overview — Baseline Simulation")
st.markdown(
    "Run the **baseline** scenario (3 honest, 1 opportunistic, 1 deceptive agent) "
    "and inspect ecosystem-level metrics."
)

# ── Controls ──────────────────────────────────────────────────────────────────

col_seed, col_run = st.columns([2, 1])
with col_seed:
    seed = st.number_input("Random seed", value=42, min_value=0, step=1)
with col_run:
    st.write("")  # spacer
    run_btn = st.button("Run Baseline", type="primary", use_container_width=True)


# ── Run simulation ────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Running baseline simulation...", max_entries=16)
def _run_baseline(seed_val: int):
    from demo.utils.simulation import run_scenario

    scenario_path = str(PROJECT_ROOT / "scenarios" / "baseline.yaml")
    return run_scenario(scenario_path, seed=seed_val)


if run_btn or "overview_result" not in st.session_state:
    st.session_state["overview_result"] = _run_baseline(int(seed))

result = st.session_state["overview_result"]

# ── KPI cards ─────────────────────────────────────────────────────────────────

from demo.utils.formatting import format_epoch_metrics_kpis, scenario_description_card  # noqa: E402, I001

st.markdown(scenario_description_card(result), unsafe_allow_html=True)
st.markdown(
    format_epoch_metrics_kpis(
        result["epoch_metrics"],
        incoherence_series=result.get("incoherence_series"),
    ),
    unsafe_allow_html=True,
)

# ── Charts ────────────────────────────────────────────────────────────────────

from demo.utils.charts import (  # noqa: E402, I001
    agent_payoff_bar,
    agent_reputation_bar,
    agent_type_pie,
    metrics_over_time,
)

st.subheader("Metrics Over Time")
st.plotly_chart(
    metrics_over_time(
        result["epoch_metrics"],
        incoherence_series=result.get("incoherence_series"),
    ),
    use_container_width=True,
)

st.subheader("Agent Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(
        agent_reputation_bar(result["agent_states"], height=280),
        use_container_width=True,
    )

with col2:
    st.plotly_chart(
        agent_payoff_bar(result["agent_states"], height=280), use_container_width=True
    )

with col3:
    st.plotly_chart(
        agent_type_pie(result["agent_states"], height=280), use_container_width=True
    )

# ── Agent table ───────────────────────────────────────────────────────────────

st.subheader("Agent Details")
from demo.utils.charts import agent_table_data  # noqa: E402

df = agent_table_data(result["agent_states"])
if not df.empty:
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No agent data available.")

# ── Raw epoch data ────────────────────────────────────────────────────────────

with st.expander("Raw epoch metrics"):
    import pandas as pd

    rows = []
    incoherence = result.get("incoherence_series") or []
    for i, m in enumerate(result["epoch_metrics"]):
        accepted = m.accepted_interactions
        total = m.total_interactions
        incoh = incoherence[i] if i < len(incoherence) else 0.0
        rows.append(
            {
                "Epoch": i,
                "Toxicity": round(m.toxicity_rate, 4),
                "Quality Gap": round(m.quality_gap, 4),
                "Incoherence": round(incoh, 4),
                "Total Welfare": round(m.total_welfare, 2),
                "Acceptance Rate": round(accepted / total, 3) if total > 0 else 0,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
