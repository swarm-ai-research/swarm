"""Page 2: Scenario Explorer — Load and compare YAML scenarios."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Scenario Explorer", page_icon="🗂️", layout="wide")
st.title("Scenario Explorer")
st.markdown(
    "Select one or more pre-built scenarios, run them, and compare results side by side."
)

# ── Load available scenarios ──────────────────────────────────────────────────

from demo.utils.simulation import list_scenarios  # noqa: E402

scenarios = list_scenarios()
scenario_map = {s["id"]: s for s in scenarios}

# ── Selection ─────────────────────────────────────────────────────────────────

selected_ids = st.multiselect(
    "Choose scenarios to run",
    options=[s["id"] for s in scenarios],
    default=["baseline"] if "baseline" in scenario_map else [],
    format_func=lambda sid: f"{sid} — {scenario_map[sid]['description'][:60]}",
)

col_seed, col_run = st.columns([2, 1])
with col_seed:
    seed = st.number_input(
        "Random seed", value=42, min_value=0, step=1, key="explorer_seed"
    )
with col_run:
    st.write("")
    run_btn = st.button("Run Selected", type="primary", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Running scenario...", max_entries=20)
def _run(path: str, seed_val: int):
    from demo.utils.simulation import run_scenario

    return run_scenario(path, seed=seed_val)


if run_btn and selected_ids:
    results = {}
    progress = st.progress(0, text="Running scenarios...")
    for i, sid in enumerate(selected_ids):
        results[sid] = _run(scenario_map[sid]["path"], int(seed))
        progress.progress((i + 1) / len(selected_ids), text=f"Completed {sid}")
    st.session_state["explorer_results"] = results
    progress.empty()

results = st.session_state.get("explorer_results", {})

if not results:
    st.info("Select scenarios and click **Run Selected** to begin.")
    st.stop()

# ── Per-scenario tabs ─────────────────────────────────────────────────────────

from demo.utils.charts import (  # noqa: E402, I001
    agent_payoff_bar,
    agent_reputation_bar,
    metrics_over_time,
)
from demo.utils.formatting import format_epoch_metrics_kpis, scenario_description_card  # noqa: E402, I001

tabs = st.tabs(list(results.keys()))

for tab, (_sid, result) in zip(tabs, results.items(), strict=True):
    with tab:
        st.markdown(scenario_description_card(result), unsafe_allow_html=True)
        st.markdown(
            format_epoch_metrics_kpis(
                result["epoch_metrics"],
                incoherence_series=result.get("incoherence_series"),
            ),
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            metrics_over_time(
                result["epoch_metrics"],
                incoherence_series=result.get("incoherence_series"),
            ),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                agent_reputation_bar(result["agent_states"], height=280),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                agent_payoff_bar(result["agent_states"], height=280),
                use_container_width=True,
            )

# ── Cross-scenario comparison ─────────────────────────────────────────────────

if len(results) > 1:
    st.markdown("---")
    st.subheader("Cross-Scenario Comparison")

    from demo.utils.charts import scenario_comparison_bar  # noqa: E402

    metric_choice = st.selectbox(
        "Metric to compare",
        ["toxicity_rate", "quality_gap", "incoherence_index", "total_welfare"],
    )
    st.plotly_chart(
        scenario_comparison_bar(results, metric_choice),
        use_container_width=True,
    )
