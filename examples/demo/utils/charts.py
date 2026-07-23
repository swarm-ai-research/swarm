"""Chart helpers for the demo, wrapping src/analysis/plots.py."""

from typing import Any, Dict, List

from .formatting import AGENT_TYPE_COLORS

# Use AGENT_TYPE_COLORS as the single source of truth for agent type colors
AGENT_COLORS = AGENT_TYPE_COLORS


def metrics_over_time(
    epoch_metrics: list,
    incoherence_series: List[float] | None = None,
    height: int = 400,
) -> Any:
    """Create a Plotly chart of key metrics over time from EpochMetrics list."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    epochs = list(range(len(epoch_metrics)))
    toxicity = [m.toxicity_rate for m in epoch_metrics]
    quality_gap = [m.quality_gap for m in epoch_metrics]
    welfare = [m.total_welfare for m in epoch_metrics]
    has_incoherence = bool(incoherence_series) and len(incoherence_series) == len(
        epochs
    )

    if has_incoherence:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Toxicity Rate",
                "Quality Gap",
                "Incoherence Index",
                "Total Welfare",
            ),
            vertical_spacing=0.15,
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Toxicity Rate",
                "Quality Gap",
                "Total Welfare",
                "Acceptance Rate",
            ),
            vertical_spacing=0.15,
        )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=toxicity,
            mode="lines+markers",
            line={"color": "#dc3545"},
            name="Toxicity",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=quality_gap,
            mode="lines+markers",
            line={"color": "#0d6efd"},
            name="Quality Gap",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=welfare,
            mode="lines+markers",
            line={"color": "#28a745"},
            name="Welfare",
        ),
        row=2,
        col=2 if has_incoherence else 1,
    )

    if has_incoherence:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=incoherence_series,
                mode="lines+markers",
                line={"color": "#fd7e14"},
                name="Incoherence",
            ),
            row=2,
            col=1,
        )
    else:
        # Acceptance rate
        accepted = [m.accepted_interactions for m in epoch_metrics]
        total = [m.total_interactions for m in epoch_metrics]
        acc_rate = [
            a / t if t > 0 else 0 for a, t in zip(accepted, total, strict=True)
        ]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=acc_rate,
                mode="lines+markers",
                line={"color": "#6f42c1"},
                name="Acceptance",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=height, showlegend=False)
    return fig


def agent_table_data(agent_states: List[Dict]) -> Any:
    """Convert agent states to a pandas DataFrame for display."""
    import pandas as pd

    df = pd.DataFrame(agent_states)
    if not df.empty:
        df = df.sort_values("reputation", ascending=False)
    return df


def agent_reputation_bar(agent_states: List[Dict], height: int = 300) -> Any:
    """Create a bar chart of agent reputations colored by type."""
    import plotly.graph_objects as go

    ids = [a["agent_id"] for a in agent_states]
    reps = [a["reputation"] for a in agent_states]
    colors = [AGENT_COLORS.get(a["agent_type"], "#999") for a in agent_states]

    fig = go.Figure(
        data=[
            go.Bar(
                x=ids,
                y=reps,
                marker_color=colors,
                text=[f"{r:.1f}" for r in reps],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Agent Reputation",
        xaxis_title="Agent",
        yaxis_title="Reputation",
        height=height,
    )
    return fig


def agent_payoff_bar(agent_states: List[Dict], height: int = 300) -> Any:
    """Create a bar chart of agent payoffs colored by type."""
    import plotly.graph_objects as go

    ids = [a["agent_id"] for a in agent_states]
    payoffs = [a["total_payoff"] for a in agent_states]
    colors = [AGENT_COLORS.get(a["agent_type"], "#999") for a in agent_states]

    fig = go.Figure(
        data=[
            go.Bar(
                x=ids,
                y=payoffs,
                marker_color=colors,
                text=[f"{p:.1f}" for p in payoffs],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Agent Total Payoff",
        xaxis_title="Agent",
        yaxis_title="Payoff",
        height=height,
    )
    return fig


def agent_type_pie(agent_states: List[Dict], height: int = 300) -> Any:
    """Create a pie chart of agent type distribution."""
    from collections import Counter

    import plotly.graph_objects as go

    type_counts = Counter(a["agent_type"] for a in agent_states)
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    colors = [AGENT_COLORS.get(t, "#999") for t in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker={"colors": colors},
                textinfo="label+value",
            )
        ]
    )
    fig.update_layout(title="Agent Distribution", height=height)
    return fig


def reputation_trajectories(history: Any, height: int = 400) -> Any:
    """Create multi-line chart of agent reputation over epochs."""
    import plotly.graph_objects as go

    fig = go.Figure()

    for agent_id, snapshots in sorted(history.agent_snapshots.items()):
        epochs = [s.epoch for s in snapshots]
        reps = [s.reputation for s in snapshots]

        # Determine agent type from ID prefix
        agent_type = agent_id.split("_")[0]
        color = AGENT_COLORS.get(agent_type, "#999")

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=reps,
                mode="lines",
                name=agent_id,
                line={"color": color},
            )
        )

    fig.update_layout(
        title="Reputation Trajectories",
        xaxis_title="Epoch",
        yaxis_title="Reputation",
        height=height,
        hovermode="x unified",
    )
    return fig


def scenario_comparison_bar(
    results: Dict[str, Dict],
    metric_name: str,
    height: int = 350,
) -> Any:
    """Compare a metric across scenarios as grouped bars."""
    import plotly.graph_objects as go

    scenario_ids = list(results.keys())
    values = []

    for sid in scenario_ids:
        if metric_name == "incoherence_index":
            series = results[sid].get("incoherence_series") or []
            values.append(float(series[-1]) if series else 0.0)
            continue

        epoch_metrics = results[sid]["epoch_metrics"]
        if epoch_metrics:
            last = epoch_metrics[-1]
            values.append(getattr(last, metric_name, 0.0))
        else:
            values.append(0.0)

    fig = go.Figure(
        data=[
            go.Bar(
                x=scenario_ids,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=metric_name.replace("_", " ").title(),
        xaxis_title="Scenario",
        yaxis_title=metric_name.replace("_", " ").title(),
        height=height,
    )
    return fig


def sweep_tradeoff_scatter(
    sweep_results: List[Dict],
    x_key: str = "tax_rate",
    y_toxicity: str = "toxicity",
    y_welfare: str = "welfare",
    height: int = 400,
) -> Any:
    """Create a scatter plot showing governance tradeoffs."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Human-readable label for the swept parameter
    pretty = x_key.replace("_", " ").title()

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=(f"Toxicity vs {pretty}", f"Welfare vs {pretty}")
    )

    x_vals = [r[x_key] for r in sweep_results]

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=[r[y_toxicity] for r in sweep_results],
            mode="lines+markers",
            name="Toxicity",
            line={"color": "#dc3545"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=[r[y_welfare] for r in sweep_results],
            mode="lines+markers",
            name="Welfare",
            line={"color": "#28a745"},
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=height, showlegend=False)
    fig.update_xaxes(title_text=pretty, row=1, col=1)
    fig.update_xaxes(title_text=pretty, row=1, col=2)
    return fig
