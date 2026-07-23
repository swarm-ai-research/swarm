"""Page 3: Governance Lab — Interactively tune governance parameters."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Governance Lab", page_icon="🏛️", layout="wide")
st.title("Governance Lab")
st.markdown(
    "Adjust governance levers and agent composition to see how they affect "
    "ecosystem safety metrics. Run a single configuration or sweep a parameter."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.header("Agent Composition")
n_honest = st.sidebar.slider("Honest agents", 0, 10, 3)
n_opportunistic = st.sidebar.slider("Opportunistic agents", 0, 10, 1)
n_deceptive = st.sidebar.slider("Deceptive agents", 0, 10, 1)
n_adversarial = st.sidebar.slider("Adversarial agents", 0, 10, 0)

st.sidebar.header("Simulation")
n_epochs = st.sidebar.slider("Epochs", 5, 50, 20)
steps_per_epoch = st.sidebar.slider("Steps per epoch", 5, 30, 10)
seed = st.sidebar.number_input("Seed", value=42, min_value=0, step=1, key="gov_seed")

# ── Main area: governance sliders ─────────────────────────────────────────────

st.subheader("Governance Parameters")

col1, col2 = st.columns(2)

with col1:
    tax_rate = st.slider("Transaction tax rate", 0.0, 0.5, 0.0, 0.01)
    reputation_decay = st.slider(
        "Reputation decay",
        0.8,
        1.0,
        1.0,
        0.01,
        help="1.0 = no decay, lower = faster decay",
    )
    staking_enabled = st.checkbox("Enable staking")
    min_stake = st.slider(
        "Minimum stake", 0.0, 5.0, 0.0, 0.1, disabled=not staking_enabled
    )

with col2:
    circuit_breaker = st.checkbox("Enable circuit breaker")
    freeze_threshold = st.slider(
        "Freeze toxicity threshold", 0.3, 1.0, 0.7, 0.05, disabled=not circuit_breaker
    )
    audit_enabled = st.checkbox("Enable random audit")
    audit_prob = st.slider(
        "Audit probability", 0.0, 0.5, 0.1, 0.01, disabled=not audit_enabled
    )

# ── Run single ────────────────────────────────────────────────────────────────

run_btn = st.button("Run Simulation", type="primary")


@st.cache_data(show_spinner="Running custom simulation...", max_entries=32)
def _run_custom(**kwargs):
    from demo.utils.simulation import run_custom

    return run_custom(**kwargs)


if run_btn:
    params = {
        "n_honest": n_honest,
        "n_opportunistic": n_opportunistic,
        "n_deceptive": n_deceptive,
        "n_adversarial": n_adversarial,
        "n_epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "tax_rate": tax_rate,
        "reputation_decay": reputation_decay,
        "staking_enabled": staking_enabled,
        "min_stake": min_stake,
        "circuit_breaker_enabled": circuit_breaker,
        "freeze_threshold": freeze_threshold,
        "audit_enabled": audit_enabled,
        "audit_probability": audit_prob,
        "seed": int(seed),
    }
    st.session_state["gov_result"] = _run_custom(**params)
    st.session_state["gov_params"] = params

result = st.session_state.get("gov_result")

if result is None:
    st.info("Configure parameters and click **Run Simulation**.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────

from demo.utils.formatting import format_epoch_metrics_kpis  # noqa: E402, I001

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

st.plotly_chart(
    metrics_over_time(
        result["epoch_metrics"],
        incoherence_series=result.get("incoherence_series"),
    ),
    use_container_width=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(
        agent_reputation_bar(result["agent_states"], height=280),
        use_container_width=True,
    )
with c2:
    st.plotly_chart(
        agent_payoff_bar(result["agent_states"], height=280), use_container_width=True
    )
with c3:
    st.plotly_chart(
        agent_type_pie(result["agent_states"], height=280), use_container_width=True
    )

# ── Parameter sweep ───────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Quick Parameter Sweep")
st.markdown(
    "Sweep one parameter across a range to see its impact on toxicity and welfare."
)

sweep_param = st.selectbox(
    "Parameter to sweep", ["tax_rate", "reputation_decay", "audit_probability"]
)

sweep_ranges = {
    "tax_rate": (0.0, 0.5, 0.05),
    "reputation_decay": (0.8, 1.0, 0.02),
    "audit_probability": (0.0, 0.5, 0.05),
}

low, high, step = sweep_ranges[sweep_param]
sweep_btn = st.button("Run Sweep")

if sweep_btn:
    base_params = st.session_state.get("gov_params", {})
    if not base_params:
        base_params = {
            "n_honest": n_honest,
            "n_opportunistic": n_opportunistic,
            "n_deceptive": n_deceptive,
            "n_adversarial": n_adversarial,
            "n_epochs": n_epochs,
            "steps_per_epoch": steps_per_epoch,
            "tax_rate": tax_rate,
            "reputation_decay": reputation_decay,
            "staking_enabled": staking_enabled,
            "min_stake": min_stake,
            "circuit_breaker_enabled": circuit_breaker,
            "freeze_threshold": freeze_threshold,
            "audit_enabled": audit_enabled,
            "audit_probability": audit_prob,
            "seed": int(seed),
        }

    import numpy as np

    sweep_values = np.arange(low, high + step / 2, step).tolist()
    sweep_results = []

    progress = st.progress(0, text=f"Sweeping {sweep_param}...")
    for i, val in enumerate(sweep_values):
        p = dict(base_params)
        p[sweep_param] = round(val, 4)
        # Enable the feature if sweeping its parameter
        if sweep_param == "audit_probability":
            p["audit_enabled"] = True
        res = _run_custom(**p)
        last = res["epoch_metrics"][-1] if res["epoch_metrics"] else None
        sweep_results.append(
            {
                sweep_param: round(val, 4),
                "toxicity": last.toxicity_rate if last else 0,
                "welfare": last.total_welfare if last else 0,
            }
        )
        progress.progress((i + 1) / len(sweep_values))

    progress.empty()
    st.session_state["sweep_results"] = sweep_results
    st.session_state["sweep_param"] = sweep_param

sweep_data = st.session_state.get("sweep_results")
if sweep_data:
    from demo.utils.charts import sweep_tradeoff_scatter  # noqa: E402

    st.plotly_chart(
        sweep_tradeoff_scatter(
            sweep_data,
            x_key=st.session_state.get("sweep_param", "tax_rate"),
        ),
        use_container_width=True,
    )

    import pandas as pd

    with st.expander("Sweep data"):
        st.dataframe(
            pd.DataFrame(sweep_data), use_container_width=True, hide_index=True
        )
