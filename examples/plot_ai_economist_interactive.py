#!/usr/bin/env python
"""Generate interactive Plotly HTML figures for the AI Economist GTB blog post.

Usage:
    python examples/plot_ai_economist_interactive.py runs/20260215_095359_ai_economist_seed42

Outputs two standalone HTML files to docs/blog/figures/:
    - gtb_economy_interactive.html
    - gtb_adversarial_interactive.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from ai_economist_colors import AGENT_TYPE_COLORS as _AGENT_TYPE_COLORS
from plotly.subplots import make_subplots

from swarm.analysis.theme import COLORS

# Shared Plotly layout defaults matching SWARM dark theme
_LAYOUT_DEFAULTS = {
    "paper_bgcolor": COLORS.BG_DARK,
    "plot_bgcolor": COLORS.BG_PANEL,
    "font": {"color": COLORS.TEXT_PRIMARY, "family": "monospace", "size": 12},
    "hoverlabel": {
        "bgcolor": COLORS.BG_PANEL,
        "bordercolor": COLORS.ACCENT_BORDER,
        "font": {"color": COLORS.TEXT_PRIMARY, "family": "monospace"},
    },
}


def _agent_type(agent_id: str) -> str:
    """Extract agent type from worker ID like 'worker_honest_0_1'."""
    parts = agent_id.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"


def _axis_style(title: str = "", color: str = "") -> dict:
    """Return common axis style dict."""
    d = {
        "gridcolor": COLORS.GRID,
        "linecolor": COLORS.ACCENT_BORDER,
        "zerolinecolor": COLORS.GRID,
    }
    if title:
        d["title"] = {"text": title, "font": {"color": color or COLORS.TEXT_PRIMARY}}
    if color:
        d["tickfont"] = {"color": color}
    return d


def plot_economy_interactive(
    metrics: pd.DataFrame,
    tax_schedule: dict,
    output_dir: Path,
) -> Path:
    """Figure 1: Economy Interactive — production/revenue + gini/welfare."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=["Production & Tax Revenue", "Inequality & Welfare"],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    epochs = metrics["epoch"]

    # ── Row 1: Production + Tax Revenue ──
    fig.add_trace(
        go.Scatter(
            x=epochs, y=metrics["total_production"],
            name="Production",
            line={"color": COLORS.PRODUCTIVITY, "width": 2.5},
            mode="lines+markers",
            marker={"size": 5, "color": COLORS.PRODUCTIVITY,
                        "line": {"color": COLORS.BG_DARK, "width": 1}},
            fill="tozeroy",
            fillcolor="rgba(39, 174, 96, 0.08)",
            hovertemplate="Epoch %{x}<br>Production: %{y:.0f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=metrics["total_tax_revenue"],
            name="Tax Revenue",
            line={"color": COLORS.REVENUE, "width": 2, "dash": "dash"},
            fill="tozeroy",
            fillcolor="rgba(111, 207, 151, 0.08)",
            hovertemplate="Epoch %{x}<br>Revenue: %{y:.0f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=True,
    )

    # ── Row 2: Gini + Welfare ──
    fig.add_trace(
        go.Scatter(
            x=epochs, y=metrics["gini_coefficient"],
            name="Gini Coefficient",
            line={"color": COLORS.EVASION, "width": 2.5},
            mode="lines+markers",
            marker={"size": 5, "color": COLORS.EVASION},
            fill="tozeroy",
            fillcolor="rgba(242, 201, 76, 0.08)",
            hovertemplate="Epoch %{x}<br>Gini: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=False,
    )
    # Danger zone for Gini > 0.5
    fig.add_hrect(
        y0=0.5, y1=0.65, row=2, col=1,
        fillcolor=COLORS.TOXICITY, opacity=0.08,
        line_width=0,
    )
    fig.add_hline(
        y=0.5, row=2, col=1,
        line={"color": COLORS.TOXICITY, "width": 1, "dash": "dot"},
        annotation_text="high inequality",
        annotation_font_color=COLORS.TOXICITY,
        annotation_font_size=10,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=metrics["welfare"],
            name="Welfare",
            line={"color": COLORS.WELFARE, "width": 2, "dash": "dot"},
            mode="lines+markers",
            marker={"size": 4, "symbol": "square", "color": COLORS.WELFARE},
            hovertemplate="Epoch %{x}<br>Welfare: %{y:.1f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=True,
    )

    # ── Axis labels ──
    fig.update_yaxes(
        **_axis_style("Total Production", COLORS.PRODUCTIVITY),
        row=1, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        **_axis_style("Tax Revenue", COLORS.REVENUE),
        row=1, col=1, secondary_y=True,
    )
    fig.update_yaxes(
        **_axis_style("Gini Coefficient", COLORS.EVASION),
        range=[0, 0.65],
        row=2, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        **_axis_style("Welfare", COLORS.WELFARE),
        row=2, col=1, secondary_y=True,
    )
    fig.update_xaxes(
        **_axis_style("Epoch"),
        row=2, col=1,
    )
    fig.update_xaxes(gridcolor=COLORS.GRID, row=1, col=1)

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font.color = COLORS.TEXT_PRIMARY
        ann.font.size = 14

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title={
            "text": "AI Economist GTB — Economy Dashboard",
            "font": {"size": 16, "color": COLORS.TEXT_PRIMARY},
            "x": 0.5,
        },
        height=650,
        margin={"l": 60, "r": 60, "t": 80, "b": 40},
        legend={
            "bgcolor": COLORS.BG_PANEL,
            "bordercolor": COLORS.ACCENT_BORDER,
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hovermode="x unified",
    )

    out = output_dir / "gtb_economy_interactive.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out


def plot_adversarial_interactive(
    metrics: pd.DataFrame,
    workers: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Figure 2: Adversarial Interactive — wealth bars + collusion tracking."""
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.18,
        subplot_titles=["Agent Wealth by Policy Type", "Collusion Detection"],
        specs=[[{"type": "bar"}], [{"secondary_y": True}]],
    )

    epochs = metrics["epoch"]

    # ── Row 1: Agent wealth horizontal bars ──
    w = workers.copy()
    w["type"] = w["agent_id"].apply(_agent_type)
    w["house_value"] = w["houses_built"] * 50
    w["total_wealth"] = w["coin"] + w["house_value"]
    w = w.sort_values("total_wealth", ascending=True)

    labels = []
    for aid in w["agent_id"]:
        parts = aid.split("_")
        labels.append(f"{parts[1].capitalize()} {parts[-1]}")

    bar_colors = [_AGENT_TYPE_COLORS.get(t, COLORS.OPPORTUNISTIC) for t in w["type"]]

    fig.add_trace(
        go.Bar(
            y=labels,
            x=w["coin"].values,
            name="Coin",
            orientation="h",
            marker={
                "color": [f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.85)"
                       for c in bar_colors],
                "line": {
                    "color": bar_colors,
                    "width": 1,
                },
            },
            hovertemplate="%{y}<br>Coin: %{x:.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=w["house_value"].values,
            name="Houses",
            orientation="h",
            marker={
                "color": [f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.3)"
                       for c in bar_colors],
                "line": {
                    "color": [f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.8)"
                           for c in bar_colors],
                    "width": 1,
                },
            },
            hovertemplate="%{y}<br>Houses: %{x:.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.update_layout(barmode="stack")
    fig.update_xaxes(
        **_axis_style("Wealth (Coin + Houses)"),
        row=1, col=1,
    )
    fig.update_yaxes(
        gridcolor=COLORS.GRID,
        tickfont={"size": 10, "color": COLORS.TEXT_PRIMARY},
        row=1, col=1,
    )

    # ── Row 2: Collusion events + suspicion score ──
    fig.add_trace(
        go.Scatter(
            x=epochs, y=metrics["collusion_events_detected"],
            name="Events Detected",
            line={"color": COLORS.ADVERSARIAL, "width": 2.5},
            mode="lines+markers",
            marker={"size": 5, "color": COLORS.ADVERSARIAL},
            fill="tozeroy",
            fillcolor="rgba(235, 87, 87, 0.08)",
            hovertemplate="Epoch %{x}<br>Events: %{y:.0f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=False,
    )
    # Annotate peak
    peak_idx = metrics["collusion_events_detected"].idxmax()
    peak_epoch = metrics["epoch"].iloc[peak_idx]
    peak_val = metrics["collusion_events_detected"].iloc[peak_idx]
    fig.add_annotation(
        x=peak_epoch, y=peak_val,
        text=f"peak: {peak_val:.0f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS.ADVERSARIAL,
        font={"color": COLORS.ADVERSARIAL, "size": 11},
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=epochs, y=metrics["collusion_suspicion_mean"],
            name="Suspicion Score",
            line={"color": COLORS.PLANNER, "width": 2, "dash": "dot"},
            hovertemplate="Epoch %{x}<br>Suspicion: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=True,
    )

    fig.update_yaxes(
        **_axis_style("Events per Epoch", COLORS.ADVERSARIAL),
        row=2, col=1, secondary_y=False,
    )
    fig.update_yaxes(
        **_axis_style("Mean Suspicion", COLORS.PLANNER),
        range=[0.6, 0.95],
        row=2, col=1, secondary_y=True,
    )
    fig.update_xaxes(
        **_axis_style("Epoch"),
        row=2, col=1,
    )

    # Subplot title styling
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in (
            "Agent Wealth by Policy Type", "Collusion Detection"
        ):
            ann.font.color = COLORS.TEXT_PRIMARY
            ann.font.size = 14

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title={
            "text": "AI Economist GTB — Adversarial Dynamics",
            "font": {"size": 16, "color": COLORS.TEXT_PRIMARY},
            "x": 0.5,
        },
        height=700,
        margin={"l": 60, "r": 60, "t": 80, "b": 40},
        legend={
            "bgcolor": COLORS.BG_PANEL,
            "bordercolor": COLORS.ACCENT_BORDER,
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hovermode="x unified",
    )

    out = output_dir / "gtb_adversarial_interactive.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate interactive Plotly HTML figures for AI Economist GTB blog post"
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Run folder (e.g. runs/20260215_095359_ai_economist_seed42)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    csv_dir = run_dir / "csv"

    output_dir = Path("docs/blog/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    metrics = pd.read_csv(csv_dir / "metrics.csv")
    workers = pd.read_csv(csv_dir / "workers.csv")
    with open(csv_dir / "tax_schedule.json") as f:
        tax_schedule = json.load(f)

    print(f"Loaded {len(metrics)} epochs, {len(workers)} workers")

    out1 = plot_economy_interactive(metrics, tax_schedule, output_dir)
    print(f"  {out1}")

    out2 = plot_adversarial_interactive(metrics, workers, output_dir)
    print(f"  {out2}")

    print(f"\nInteractive figures saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
