"""Plotting utilities for simulation visualization.

Provides both matplotlib (static) and plotly (interactive) chart generators
for the dashboard and analysis notebooks.
"""

from typing import Any, Dict, List, Optional, Tuple

from swarm.analysis.aggregation import (
    EpochSnapshot,
    SimulationHistory,
)

# Type alias for plot data that can be used by multiple backends
PlotData = Dict[str, Any]


def create_time_series_data(
    history: SimulationHistory,
    metrics: List[str],
    rolling_window: Optional[int] = None,
) -> PlotData:
    """
    Create time series plot data for specified metrics.

    Args:
        history: Simulation history
        metrics: List of metric names to plot
        rolling_window: Optional window size for rolling average

    Returns:
        PlotData dict with 'epochs' and metric values
    """
    data: PlotData = {
        "epochs": [s.epoch for s in history.epoch_snapshots],
    }

    for metric in metrics:
        values = []
        for snapshot in history.epoch_snapshots:
            value = getattr(snapshot, metric, None)
            values.append(float(value) if value is not None else 0.0)

        if rolling_window and len(values) >= rolling_window:
            import numpy as np

            smoothed = []
            for i in range(len(values)):
                start = max(0, i - rolling_window + 1)
                smoothed.append(float(np.mean(values[start : i + 1])))
            values = smoothed

        data[metric] = values

    return data


def create_agent_comparison_data(
    history: SimulationHistory,
    metric: str = "total_payoff",
) -> PlotData:
    """
    Create data for comparing agents on a specific metric.

    Args:
        history: Simulation history
        metric: Metric to compare

    Returns:
        PlotData with agent IDs and metric values
    """
    final_states = history.get_final_agent_states()

    agent_ids = []
    values = []

    for agent_id, snapshot in sorted(final_states.items()):
        agent_ids.append(agent_id)
        values.append(getattr(snapshot, metric, 0.0))

    return {
        "agent_ids": agent_ids,
        "values": values,
        "metric": metric,
    }


def create_agent_trajectory_data(
    history: SimulationHistory,
    agent_ids: List[str],
    metric: str = "reputation",
) -> PlotData:
    """
    Create data for agent trajectories over time.

    Args:
        history: Simulation history
        agent_ids: List of agents to include
        metric: Metric to track

    Returns:
        PlotData with epochs and per-agent time series
    """
    data: PlotData = {"epochs": []}
    max_epochs = 0

    for agent_id in agent_ids:
        if agent_id in history.agent_snapshots:
            snapshots = history.agent_snapshots[agent_id]
            epochs = [s.epoch for s in snapshots]
            values = [getattr(s, metric, 0.0) for s in snapshots]

            data[agent_id] = {"epochs": epochs, "values": values}
            max_epochs = max(max_epochs, max(epochs) if epochs else 0)

    data["epochs"] = list(range(max_epochs + 1))
    return data


def create_distribution_data(
    snapshots: List[EpochSnapshot],
    metric: str = "avg_p",
) -> PlotData:
    """
    Create histogram data for a metric distribution.

    Args:
        snapshots: List of epoch snapshots
        metric: Metric to plot distribution of

    Returns:
        PlotData with values and optional bin information
    """
    values = [getattr(s, metric, 0.0) for s in snapshots]

    return {
        "values": values,
        "metric": metric,
        "count": len(values),
    }


def create_heatmap_data(
    history: SimulationHistory,
    row_metric: str = "toxicity_rate",
    col_agents: bool = True,
) -> PlotData:
    """
    Create heatmap data for epoch x agent visualization.

    Args:
        history: Simulation history
        row_metric: Metric for intensity (if not per-agent)
        col_agents: If True, columns are agents; otherwise epochs

    Returns:
        PlotData for heatmap visualization
    """
    if col_agents:
        # Epochs as rows, agents as columns
        epochs = [s.epoch for s in history.epoch_snapshots]
        agent_ids = sorted(history.agent_snapshots.keys())

        matrix = []
        for epoch in epochs:
            row = []
            for agent_id in agent_ids:
                snapshots = history.agent_snapshots.get(agent_id, [])
                epoch_snap = next((s for s in snapshots if s.epoch == epoch), None)
                if epoch_snap:
                    row.append(getattr(epoch_snap, row_metric, 0.0))
                else:
                    row.append(0.0)
            matrix.append(row)

        return {
            "matrix": matrix,
            "row_labels": [str(e) for e in epochs],
            "col_labels": agent_ids,
            "metric": row_metric,
        }
    else:
        # Simple metric over epochs
        epochs = [s.epoch for s in history.epoch_snapshots]
        values = [getattr(s, row_metric, 0.0) for s in history.epoch_snapshots]

        return {
            "values": values,
            "labels": [str(e) for e in epochs],
            "metric": row_metric,
        }


def create_network_graph_data(
    edges: List[Tuple[str, str, float]],
    node_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> PlotData:
    """
    Create network graph data for visualization.

    Args:
        edges: List of (source, target, weight) tuples
        node_attributes: Optional dict of agent_id -> attributes

    Returns:
        PlotData for network visualization
    """
    nodes = set()
    for src, tgt, _ in edges:
        nodes.add(src)
        nodes.add(tgt)

    node_list = sorted(nodes)
    node_attrs = node_attributes or {}

    return {
        "nodes": [
            {
                "id": n,
                "label": n,
                **node_attrs.get(n, {}),
            }
            for n in node_list
        ],
        "edges": [{"source": src, "target": tgt, "weight": w} for src, tgt, w in edges],
    }


def create_scatter_data(
    history: SimulationHistory,
    x_metric: str,
    y_metric: str,
    color_metric: Optional[str] = None,
) -> PlotData:
    """
    Create scatter plot data for metric correlations.

    Args:
        history: Simulation history
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        color_metric: Optional metric for color coding

    Returns:
        PlotData for scatter visualization
    """
    x_values = []
    y_values = []
    colors = []
    labels = []

    for snapshot in history.epoch_snapshots:
        x_values.append(getattr(snapshot, x_metric, 0.0))
        y_values.append(getattr(snapshot, y_metric, 0.0))
        if color_metric:
            colors.append(getattr(snapshot, color_metric, 0.0))
        labels.append(f"Epoch {snapshot.epoch}")

    data = {
        "x": x_values,
        "y": y_values,
        "labels": labels,
        "x_metric": x_metric,
        "y_metric": y_metric,
    }

    if color_metric:
        data["colors"] = colors
        data["color_metric"] = color_metric

    return data


def create_incoherence_scaling_data(rows: List[Dict[str, Any]]) -> PlotData:
    """Create compact plot data for incoherence scaling summaries."""
    labels = [
        f"{row.get('horizon_tier', 'unknown')}|{row.get('branching_tier', 'unknown')}"
        for row in rows
    ]
    return {
        "labels": labels,
        "incoherence_index": [
            float(row.get("mean_incoherence_index", 0.0)) for row in rows
        ],
        "error_rate": [float(row.get("mean_error_rate", 0.0)) for row in rows],
        "disagreement_rate": [
            float(row.get("mean_disagreement_rate", 0.0)) for row in rows
        ],
    }


# =============================================================================
# Plotly Chart Generators
# =============================================================================


def plotly_time_series(
    data: PlotData,
    title: str = "Metrics Over Time",
    height: int = 400,
) -> Any:
    """
    Create a Plotly time series chart.

    Args:
        data: PlotData from create_time_series_data
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    fig = go.Figure()

    epochs = data.get("epochs", [])
    for key, values in data.items():
        if key == "epochs":
            continue
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines+markers",
                name=key.replace("_", " ").title(),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Value",
        height=height,
        hovermode="x unified",
    )

    return fig


def plotly_bar_chart(
    data: PlotData,
    title: str = "Agent Comparison",
    height: int = 400,
) -> Any:
    """
    Create a Plotly bar chart for agent comparison.

    Args:
        data: PlotData from create_agent_comparison_data
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=data["agent_ids"],
                y=data["values"],
                text=[f"{v:.2f}" for v in data["values"]],
                textposition="auto",
            )
        ]
    )

    metric_name = data.get("metric", "Value").replace("_", " ").title()
    fig.update_layout(
        title=title,
        xaxis_title="Agent",
        yaxis_title=metric_name,
        height=height,
    )

    return fig


def plotly_multi_line(
    data: PlotData,
    title: str = "Agent Trajectories",
    height: int = 400,
) -> Any:
    """
    Create a Plotly multi-line chart for agent trajectories.

    Args:
        data: PlotData from create_agent_trajectory_data
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    fig = go.Figure()

    for key, values in data.items():
        if key == "epochs":
            continue
        if isinstance(values, dict):
            fig.add_trace(
                go.Scatter(
                    x=values["epochs"],
                    y=values["values"],
                    mode="lines",
                    name=key,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Value",
        height=height,
        hovermode="x unified",
    )

    return fig


def plotly_heatmap(
    data: PlotData,
    title: str = "Epoch-Agent Heatmap",
    height: int = 400,
) -> Any:
    """
    Create a Plotly heatmap.

    Args:
        data: PlotData from create_heatmap_data
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    fig = go.Figure(
        data=go.Heatmap(
            z=data["matrix"],
            x=data["col_labels"],
            y=data["row_labels"],
            colorscale="RdYlGn_r",
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Agent",
        yaxis_title="Epoch",
        height=height,
    )

    return fig


def plotly_scatter(
    data: PlotData,
    title: str = "Metric Correlation",
    height: int = 400,
) -> Any:
    """
    Create a Plotly scatter plot.

    Args:
        data: PlotData from create_scatter_data
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    marker_dict: Dict[str, Any] = {"size": 10}
    if "colors" in data:
        marker_dict["color"] = data["colors"]
        marker_dict["colorscale"] = "Viridis"
        marker_dict["showscale"] = True
        marker_dict["colorbar"] = {
            "title": data.get("color_metric", "Color").replace("_", " ").title()
        }

    fig = go.Figure(
        data=go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            text=data["labels"],
            marker=marker_dict,
        )
    )

    x_name = data.get("x_metric", "X").replace("_", " ").title()
    y_name = data.get("y_metric", "Y").replace("_", " ").title()

    fig.update_layout(
        title=title,
        xaxis_title=x_name,
        yaxis_title=y_name,
        height=height,
    )

    return fig


def plotly_gauge(
    value: float,
    title: str = "Metric",
    min_val: float = 0.0,
    max_val: float = 1.0,
    thresholds: Optional[List[Tuple[float, str]]] = None,
    height: int = 200,
) -> Any:
    """
    Create a Plotly gauge chart.

    Args:
        value: Current value
        title: Gauge title
        min_val: Minimum value
        max_val: Maximum value
        thresholds: List of (value, color) tuples for threshold bars
        height: Chart height

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    # Default thresholds for 0-1 range
    if thresholds is None:
        thresholds = [
            (0.3, "green"),
            (0.6, "yellow"),
            (1.0, "red"),
        ]

    steps = []
    prev = min_val
    for threshold, color in thresholds:
        steps.append(
            {
                "range": [prev, threshold],
                "color": color,
            }
        )
        prev = threshold

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [min_val, max_val]},
                "steps": steps,
                "bar": {"color": "darkblue"},
            },
        )
    )

    fig.update_layout(height=height)
    return fig


def plotly_network(
    data: PlotData,
    title: str = "Agent Network",
    height: int = 500,
) -> Any:
    """
    Create a Plotly network graph visualization.

    Args:
        data: PlotData from create_network_graph_data
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure object
    """
    try:
        import numpy as np
        import plotly.graph_objects as go
    except ImportError:
        return None

    nodes = data["nodes"]
    edges = data["edges"]
    n_nodes = len(nodes)

    if n_nodes == 0:
        return go.Figure()

    # Simple circular layout
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pos = {
        node["id"]: (np.cos(angles[i]), np.sin(angles[i]))
        for i, node in enumerate(nodes)
    }

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = pos[edge["source"]]
        x1, y1 = pos[edge["target"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 1, "color": "#888"},
        hoverinfo="none",
        mode="lines",
    )

    # Create node traces
    node_x = [pos[node["id"]][0] for node in nodes]
    node_y = [pos[node["id"]][1] for node in nodes]
    node_text = [node["label"] for node in nodes]

    # Color by reputation if available
    node_color = [node.get("reputation", 0.5) for node in nodes]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        marker={
            "size": 20,
            "color": node_color,
            "colorscale": "RdYlGn",
            "showscale": True,
            "colorbar": {"title": "Reputation"},
            "line": {"width": 2, "color": "black"},
        },
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        height=height,
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
    )

    return fig
