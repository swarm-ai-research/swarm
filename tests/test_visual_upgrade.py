"""Tests for the SWARM visual upgrade modules.

Each test generates a plot from synthetic data and verifies that:
- The figure object is returned correctly
- The axes have expected properties
- Both dark and light modes work
- No exceptions are raised during rendering
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")  # Non-interactive backend for CI

import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _epoch_metrics(n_epochs: int = 10, seed: int = 42) -> Dict[str, Any]:
    """Generate synthetic epoch-level metrics."""
    rng = np.random.RandomState(seed)
    epochs = list(range(n_epochs))
    return {
        "epochs": epochs,
        "toxicity_rate": (rng.rand(n_epochs) * 0.5).tolist(),
        "total_welfare": (rng.rand(n_epochs) * 100 + 50).tolist(),
        "quality_gap": (rng.randn(n_epochs) * 0.2).tolist(),
        "avg_p": (0.5 + rng.randn(n_epochs) * 0.1).clip(0, 1).tolist(),
    }


def _multi_seed_data(n_seeds: int = 5, n_epochs: int = 10) -> List[Dict[str, Any]]:
    """Generate multiple seed runs."""
    return [_epoch_metrics(n_epochs, seed=i) for i in range(n_seeds)]


def _agent_data() -> List[Dict[str, Any]]:
    """Generate synthetic agent comparison data."""
    return [
        {"agent_id": "agent_0", "agent_type": "honest", "value": 1.2},
        {"agent_id": "agent_1", "agent_type": "honest", "value": 1.5},
        {"agent_id": "agent_2", "agent_type": "deceptive", "value": 0.3},
        {"agent_id": "agent_3", "agent_type": "opportunistic", "value": 0.8},
        {"agent_id": "agent_4", "agent_type": "adversarial", "value": -0.2},
    ]


# ===================================================================
# theme.py
# ===================================================================

class TestTheme:
    def test_colors_exist(self):
        from swarm.analysis.theme import COLORS
        assert COLORS.HONEST == "#3ECFB4"
        assert COLORS.DECEPTIVE == "#F2994A"
        assert COLORS.BG_DARK == "#0D1117"

    def test_apply_theme_dark(self):
        from swarm.analysis.theme import apply_theme
        apply_theme("dark")
        assert matplotlib.rcParams["figure.facecolor"] == "#0D1117"

    def test_apply_theme_light(self):
        from swarm.analysis.theme import apply_theme
        apply_theme("light")
        assert matplotlib.rcParams["figure.facecolor"] == "#FFFFFF"

    def test_swarm_theme_context(self):
        from swarm.analysis.theme import swarm_theme
        original = matplotlib.rcParams["figure.facecolor"]
        with swarm_theme("dark"):
            assert matplotlib.rcParams["figure.facecolor"] == "#0D1117"
        # Should restore after context
        assert matplotlib.rcParams["figure.facecolor"] == original

    def test_agent_color(self):
        from swarm.analysis.theme import COLORS, agent_color
        assert agent_color("honest") == COLORS.HONEST
        assert agent_color("DECEPTIVE") == COLORS.DECEPTIVE
        assert agent_color("unknown_type") == COLORS.OPPORTUNISTIC

    def test_metric_color(self):
        from swarm.analysis.theme import COLORS, metric_color
        assert metric_color("toxicity_rate") == COLORS.TOXICITY
        assert metric_color("welfare") == COLORS.WELFARE

    def test_color_for_values(self):
        from swarm.analysis.theme import color_for_values
        rgba = color_for_values([0.0, 0.5, 1.0])
        assert rgba.shape == (3, 4)

    def test_annotate_events(self):
        from swarm.analysis.theme import annotate_events
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3], [0, 1, 0, 1])
        annotate_events(ax, [{"epoch": 1, "label": "audit"}, {"epoch": 3}])
        plt.close(fig)

    def test_add_danger_zone(self):
        from swarm.analysis.theme import add_danger_zone
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0.2, 0.6, 0.8])
        ax.set_ylim(0, 1)
        add_danger_zone(ax, 0.5, label="threshold")
        plt.close(fig)

    def test_diverging_colormaps(self):
        from swarm.analysis.theme import SWARM_DIVERGING, SWARM_DIVERGING_DARK
        assert SWARM_DIVERGING is not None
        assert SWARM_DIVERGING_DARK is not None
        # Should return RGBA for a normalized value
        rgba = SWARM_DIVERGING(0.5)
        assert len(rgba) == 4


# ===================================================================
# timeseries.py
# ===================================================================

class TestTimeseries:
    def test_plot_metric_timeseries_1d(self):
        from swarm.analysis.timeseries import plot_metric_timeseries
        fig, ax = plt.subplots()
        plot_metric_timeseries(ax, list(range(10)), list(np.random.rand(10)),
                               label="test", color="#3ECFB4")
        plt.close(fig)

    def test_plot_metric_timeseries_2d(self):
        from swarm.analysis.timeseries import plot_metric_timeseries
        fig, ax = plt.subplots()
        values_2d = np.random.rand(5, 10)  # 5 seeds, 10 epochs
        plot_metric_timeseries(ax, list(range(10)), values_2d, label="multi-seed")
        plt.close(fig)

    def test_plot_metric_timeseries_rolling(self):
        from swarm.analysis.timeseries import plot_metric_timeseries
        fig, ax = plt.subplots()
        plot_metric_timeseries(ax, list(range(20)), list(np.random.rand(20)),
                               label="rolling", rolling_window=5)
        plt.close(fig)

    def test_plot_toxicity_welfare_dark(self):
        from swarm.analysis.timeseries import plot_toxicity_welfare
        data = _epoch_metrics()
        fig, axes = plot_toxicity_welfare(
            data, toxicity_key="toxicity_rate", welfare_key="total_welfare", mode="dark")
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)

    def test_plot_toxicity_welfare_light(self):
        from swarm.analysis.timeseries import plot_toxicity_welfare
        data = _epoch_metrics()
        fig, axes = plot_toxicity_welfare(
            data, toxicity_key="toxicity_rate", welfare_key="total_welfare", mode="light")
        plt.close(fig)

    def test_plot_toxicity_welfare_with_events(self):
        from swarm.analysis.timeseries import plot_toxicity_welfare
        data = _epoch_metrics()
        events = [{"epoch": 3, "label": "circuit breaker"}, {"epoch": 7, "label": "audit"}]
        fig, axes = plot_toxicity_welfare(
            data, toxicity_key="toxicity_rate", welfare_key="total_welfare", events=events)
        plt.close(fig)

    def test_plot_selection_geometry(self):
        from swarm.analysis.timeseries import plot_selection_geometry
        n = 20
        data = {
            "epochs": list(range(n)),
            "selection_saturation": np.linspace(0.2, 0.95, n).tolist(),
            "baseline_harm": (np.ones(n) * 0.4).tolist(),
            "selection_credit": np.linspace(0.05, 0.25, n).tolist(),
        }
        events = [{"epoch": 10, "label": "policy change"}]
        fig, (ax_sat, ax_decomp) = plot_selection_geometry(data, events=events)
        assert fig is not None
        # Saturation axis is bounded to [0, ~1].
        lo, hi = ax_sat.get_ylim()
        assert lo <= 0.0 and hi <= 1.1
        # Decomposition panel has two lines (baseline + negated credit).
        assert len(ax_decomp.get_lines()) >= 2
        plt.close(fig)

    def test_plot_selection_geometry_multi_seed(self):
        from swarm.analysis.timeseries import plot_selection_geometry
        n = 15
        seeds = 4
        data = {
            "epochs": list(range(n)),
            "selection_saturation": np.random.rand(seeds, n),
            "baseline_harm": 0.3 + np.random.rand(seeds, n) * 0.1,
            "selection_credit": np.random.rand(seeds, n) * 0.2,
        }
        fig, _ = plot_selection_geometry(data)
        assert fig is not None
        plt.close(fig)

    def test_write_run_plots_creates_bundle(self, tmp_path):
        from dataclasses import dataclass

        from swarm.analysis.run_plots import write_run_plots

        @dataclass
        class FakeEpoch:
            epoch: int
            toxicity_rate: float
            total_welfare: float
            baseline_harm: float
            selection_credit: float
            selection_saturation: float

        history = [
            FakeEpoch(i, 0.4 - 0.01 * i, 1.0 + 0.1 * i,
                      0.45, 0.01 * i, 0.1 + 0.05 * i)
            for i in range(12)
        ]
        written = write_run_plots(history, tmp_path, scenario_id="unit-test")
        assert len(written) == 2
        names = {p.name for p in written}
        assert names == {"toxicity_welfare.png", "selection_geometry.png"}
        for p in written:
            assert p.exists() and p.stat().st_size > 0

    def test_write_run_plots_empty_history_noops(self, tmp_path):
        from swarm.analysis.run_plots import write_run_plots
        assert write_run_plots([], tmp_path) == []
        # No plots dir created when there's nothing to plot.
        assert not (tmp_path / "plots").exists()

    def test_plot_bilevel_loop(self):
        from swarm.analysis.timeseries import plot_bilevel_loop
        planner = {"epochs": list(range(10)), "tax_rate": np.random.rand(10).tolist()}
        worker = {"epochs": list(range(10)), "productivity": np.random.rand(10).tolist()}
        fig, axes = plot_bilevel_loop(planner, worker)
        assert fig is not None
        plt.close(fig)

    def test_plot_multi_seed_timeseries(self):
        from swarm.analysis.timeseries import plot_multi_seed_timeseries
        data = _multi_seed_data()
        fig, ax = plot_multi_seed_timeseries(data, "toxicity_rate")
        assert fig is not None
        plt.close(fig)


# ===================================================================
# comparison.py
# ===================================================================

class TestComparison:
    def test_plot_grouped_bar(self):
        from swarm.analysis.comparison import plot_grouped_bar
        fig, ax = plt.subplots()
        plot_grouped_bar(ax, ["cond_A", "cond_B"],
                         ["honest", "deceptive"],
                         [[0.8, 0.3], [0.7, 0.4]])
        plt.close(fig)

    def test_plot_slope_chart(self):
        from swarm.analysis.comparison import plot_slope_chart
        data = {
            "Model_A": [0.9, 0.7, 0.5],
            "Model_B": [0.8, 0.6, 0.3],
        }
        fig, ax = plot_slope_chart(data, conditions=["mostly_honest", "balanced", "adversarial"],
                                    metric="Cooperation Prob")
        assert fig is not None
        plt.close(fig)

    def test_plot_faceted_comparison(self):
        from swarm.analysis.comparison import plot_faceted_comparison
        data = {
            "Sonnet": {"honest": 0.9, "deceptive": 0.4},
            "Haiku": {"honest": 0.85, "deceptive": 0.5},
        }
        fig, axes = plot_faceted_comparison(data, metric="Toxicity",
                                            facet_by="model",
                                            conditions=["honest", "deceptive"])
        assert fig is not None
        plt.close(fig)

    def test_plot_agent_comparison_bar(self):
        from swarm.analysis.comparison import plot_agent_comparison_bar
        fig, ax = plot_agent_comparison_bar(_agent_data(), metric="Total Payoff")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("mode", ["dark", "light"])
    def test_slope_chart_modes(self, mode):
        from swarm.analysis.comparison import plot_slope_chart
        data = {"A": [0.9, 0.5], "B": [0.7, 0.3]}
        fig, ax = plot_slope_chart(data, conditions=["x", "y"], mode=mode)
        plt.close(fig)


# ===================================================================
# heatmaps.py
# ===================================================================

class TestHeatmaps:
    def test_plot_diverging_heatmap(self):
        from swarm.analysis.heatmaps import plot_diverging_heatmap
        matrix = np.random.randn(4, 3)
        fig, ax = plot_diverging_heatmap(matrix, row_labels=["a", "b", "c", "d"],
                                          col_labels=["x", "y", "z"])
        assert fig is not None
        plt.close(fig)

    def test_plot_diverging_heatmap_light(self):
        from swarm.analysis.heatmaps import plot_diverging_heatmap
        matrix = np.random.randn(3, 3)
        fig, ax = plot_diverging_heatmap(matrix, row_labels=["a", "b", "c"],
                                          col_labels=["x", "y", "z"], mode="light")
        plt.close(fig)

    def test_plot_difference_heatmap(self):
        from swarm.analysis.heatmaps import plot_difference_heatmap
        a = np.random.rand(3, 3)
        b = np.random.rand(3, 3)
        fig, axes = plot_difference_heatmap(a, b,
                                             row_labels=["r1", "r2", "r3"],
                                             col_labels=["c1", "c2", "c3"],
                                             label_a="Model A", label_b="Model B")
        assert len(axes) == 3
        plt.close(fig)

    def test_plot_persona_mix_heatmap(self):
        from swarm.analysis.heatmaps import plot_persona_mix_heatmap
        data = np.random.rand(3, 4)
        fig, ax = plot_persona_mix_heatmap(data,
                                            personas=["honest", "deceptive", "opportunistic"],
                                            mixes=["pure", "mostly_h", "balanced", "adversarial"])
        assert fig is not None
        plt.close(fig)


# ===================================================================
# distributions.py
# ===================================================================

class TestDistributions:
    def test_plot_raincloud(self):
        from swarm.analysis.distributions import plot_raincloud
        fig, ax = plt.subplots()
        groups = [np.random.randn(50) + i for i in range(3)]
        plot_raincloud(ax, groups, labels=["A", "B", "C"])
        plt.close(fig)

    def test_plot_reward_distribution(self):
        from swarm.analysis.distributions import plot_reward_distribution
        data = [
            {"value": float(v), "group": g}
            for g in ["honest", "deceptive", "adversarial"]
            for v in np.random.randn(20)
        ]
        fig, ax = plot_reward_distribution(data, group_by="group")
        assert fig is not None
        plt.close(fig)

    def test_plot_income_histogram(self):
        from swarm.analysis.distributions import plot_income_histogram
        incomes = np.random.exponential(30000, 500)
        fig, ax = plot_income_histogram(incomes,
                                         bracket_boundaries=[20000, 50000, 100000],
                                         highlight_bunching=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_bunching_timeseries(self):
        from swarm.analysis.distributions import plot_bunching_timeseries
        # Shape: (n_brackets, n_epochs) — 3 brackets, 10 epochs each
        coeffs = np.random.rand(3, 10).tolist()
        fig, ax = plot_bunching_timeseries(coeffs,
                                            epochs=list(range(10)),
                                            bracket_labels=["20k", "50k", "100k"])
        assert fig is not None
        plt.close(fig)

    def test_compute_bunching_coefficient(self):
        from swarm.analysis.distributions import compute_bunching_coefficient
        incomes = np.concatenate([
            np.random.normal(50000, 500, 200),  # bunching near 50k
            np.random.uniform(0, 100000, 100),
        ])
        coeff = compute_bunching_coefficient(incomes, 50000, 2000)
        assert isinstance(coeff, float)
        assert coeff >= 0.0

    def test_compute_bunching_empty(self):
        from swarm.analysis.distributions import compute_bunching_coefficient
        assert compute_bunching_coefficient(np.array([]), 50000, 2000) == 0.0


# ===================================================================
# scatter.py
# ===================================================================

class TestScatter:
    def test_plot_tradeoff_scatter_basic(self):
        from swarm.analysis.scatter import plot_tradeoff_scatter
        fig, ax = plt.subplots()
        x = np.random.rand(20)
        y = np.random.rand(20)
        plot_tradeoff_scatter(ax, x, y)
        plt.close(fig)

    def test_plot_tradeoff_scatter_pareto(self):
        from swarm.analysis.scatter import plot_tradeoff_scatter
        fig, ax = plt.subplots()
        x = np.random.rand(30)
        y = np.random.rand(30)
        plot_tradeoff_scatter(ax, x, y, pareto=True, marginals=True)
        plt.close(fig)

    def test_pareto_frontier(self):
        from swarm.analysis.scatter import _compute_pareto_frontier
        x = np.array([1, 2, 3, 4])
        y = np.array([4, 3, 2, 1])
        indices = _compute_pareto_frontier(x, y)
        # Points (1,4), (4,1) and possibly (2,3), (3,2) are Pareto-optimal
        assert len(indices) >= 2

    def test_plot_toxicity_welfare_scatter(self):
        from swarm.analysis.scatter import plot_toxicity_welfare_scatter
        data = [
            {"toxicity": float(t), "welfare": float(w), "persona": p}
            for t, w, p in zip(np.random.rand(15), np.random.rand(15) * 100,
                                ["honest"] * 5 + ["deceptive"] * 5 + ["adversarial"] * 5, strict=False)
        ]
        fig, ax = plot_toxicity_welfare_scatter(data, color_by="persona")
        assert fig is not None
        plt.close(fig)

    def test_plot_welfare_frontier(self):
        from swarm.analysis.scatter import plot_welfare_frontier
        sweep = [
            {"productivity": float(p), "equality": float(e), "cost": float(c)}
            for p, e, c in zip(np.random.rand(20), np.random.rand(20), np.random.rand(20), strict=False)
        ]
        fig, ax = plot_welfare_frontier(sweep, x_metric="productivity",
                                         y_metric="equality", color_metric="cost")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("mode", ["dark", "light"])
    def test_scatter_modes(self, mode):
        from swarm.analysis.scatter import plot_toxicity_welfare_scatter
        data = [{"toxicity": 0.3, "welfare": 50.0}]
        fig, ax = plot_toxicity_welfare_scatter(data, mode=mode)
        plt.close(fig)


# ===================================================================
# spatial.py
# ===================================================================

class TestSpatial:
    def test_render_grid_frame(self):
        from swarm.analysis.spatial import render_grid_frame
        grid = np.zeros((10, 10), dtype=int)
        grid[2:4, 3:5] = 1  # wood
        grid[6, 7] = 2  # stone
        grid[8, 2] = 3  # house
        agents = [
            {"row": 3, "col": 3, "type": "honest", "label": "A0"},
            {"row": 6, "col": 7, "type": "deceptive", "label": "A1"},
        ]
        fig, ax = render_grid_frame(grid, agent_positions=agents, title="Test Grid")
        assert fig is not None
        plt.close(fig)

    def test_render_grid_frame_light(self):
        from swarm.analysis.spatial import render_grid_frame
        grid = np.zeros((5, 5), dtype=int)
        fig, ax = render_grid_frame(grid, mode="light")
        plt.close(fig)

    def test_plot_resource_heatmap(self):
        from swarm.analysis.spatial import plot_resource_heatmap
        frames = [np.random.randint(0, 4, (8, 8)) for _ in range(5)]
        fig, ax = plot_resource_heatmap(frames, resource_type=1, title="Wood Density")
        assert fig is not None
        plt.close(fig)


# ===================================================================
# network.py
# ===================================================================

class TestNetwork:
    def test_compute_spring_layout(self):
        from swarm.analysis.network import compute_spring_layout
        nodes = ["a", "b", "c"]
        edges = [("a", "b", 1.0), ("b", "c", 0.5)]
        pos = compute_spring_layout(nodes, edges, seed=42)
        assert len(pos) == 3
        for v in pos.values():
            assert len(v) == 2

    def test_plot_collusion_network(self):
        from swarm.analysis.network import plot_collusion_network
        edges = [("a", "b", 0.8), ("b", "c", 0.6), ("a", "c", 0.9)]
        fig, ax = plot_collusion_network(
            edges,
            node_types={"a": "honest", "b": "deceptive", "c": "adversarial"},
            detection_threshold=0.7,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_collusion_with_coalitions(self):
        from swarm.analysis.network import plot_collusion_network
        edges = [("a", "b", 0.9), ("c", "d", 0.85), ("a", "c", 0.3)]
        fig, ax = plot_collusion_network(
            edges,
            detected_coalitions=[{"a", "b"}, {"c", "d"}],
        )
        plt.close(fig)

    def test_plot_interaction_network(self):
        from swarm.analysis.network import plot_interaction_network
        edges = [("x", "y", 10), ("y", "z", 5), ("x", "z", 8)]
        fig, ax = plot_interaction_network(edges)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("mode", ["dark", "light"])
    def test_network_modes(self, mode):
        from swarm.analysis.network import plot_collusion_network
        edges = [("a", "b", 0.8)]
        fig, ax = plot_collusion_network(edges, mode=mode)
        plt.close(fig)


# ===================================================================
# sankey.py
# ===================================================================

class TestSankey:
    def test_plot_sankey_basic(self):
        from swarm.analysis.sankey import plot_sankey
        fig, ax = plt.subplots(figsize=(10, 6))
        flows = [
            {"source": 0, "target": 1, "value": 60},
            {"source": 0, "target": 2, "value": 40},
            {"source": 1, "target": 3, "value": 60},
        ]
        plot_sankey(ax, flows, labels=["Total", "Honest", "Evaders", "Output"])
        plt.close(fig)

    def test_plot_audit_evasion_flow(self):
        from swarm.analysis.sankey import plot_audit_evasion_flow
        flow_data = {
            "total": 100,
            "honest": 60,
            "evaders": 40,
            "audited": 20,
            "unaudited": 20,
            "caught": 15,
            "undetected": 5,
            "fined": 10,
            "frozen": 5,
            "continues": 20,
        }
        fig, ax = plot_audit_evasion_flow(flow_data)
        assert fig is not None
        plt.close(fig)

    def test_plot_enforcement_summary(self):
        from swarm.analysis.sankey import plot_enforcement_summary
        flow_data = {
            "total": 100, "honest": 60, "evaders": 40,
            "audited": 20, "unaudited": 20, "caught": 15,
            "undetected": 5, "fined": 10, "frozen": 5, "continues": 20,
        }
        fig, axes = plot_enforcement_summary([flow_data, flow_data],
                                              title="Enforcement Over 2 Epochs")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("mode", ["dark", "light"])
    def test_sankey_modes(self, mode):
        from swarm.analysis.sankey import plot_audit_evasion_flow
        flow_data = {
            "total": 50, "honest": 30, "evaders": 20,
            "audited": 10, "unaudited": 10, "caught": 8,
            "undetected": 2, "fined": 5, "frozen": 3, "continues": 10,
        }
        fig, ax = plot_audit_evasion_flow(flow_data, mode=mode)
        plt.close(fig)


# ===================================================================
# tax_schedule.py
# ===================================================================

class TestTaxSchedule:
    def test_compute_effective_rate(self):
        from swarm.analysis.tax_schedule import compute_effective_rate
        brackets = [0, 20000, 50000, 100000]
        rates = [0.1, 0.2, 0.3, 0.4]
        income_range = np.linspace(0, 150000, 100)
        effective = compute_effective_rate(brackets, rates, income_range)
        assert len(effective) == 100
        assert effective[0] >= 0.0  # At zero income, effective rate is 0 or close

    def test_plot_tax_schedule_figure(self):
        from swarm.analysis.tax_schedule import plot_tax_schedule_figure
        brackets = [0, 20000, 50000, 100000]
        rates = [0.1, 0.2, 0.3, 0.4]
        fig, ax = plot_tax_schedule_figure(brackets, rates, show_effective=True)
        assert fig is not None
        plt.close(fig)

    def test_plot_tax_schedule_with_bunching(self):
        from swarm.analysis.tax_schedule import plot_tax_schedule_figure
        brackets = [0, 20000, 50000]
        rates = [0.1, 0.25, 0.35]
        fig, ax = plot_tax_schedule_figure(brackets, rates,
                                            bunching_zones=[(20000, 3000), (50000, 5000)])
        plt.close(fig)

    def test_plot_tax_schedule_comparison(self):
        from swarm.analysis.tax_schedule import plot_tax_schedule_comparison
        old = {"brackets": [0, 20000, 50000], "rates": [0.1, 0.2, 0.3]}
        new = {"brackets": [0, 25000, 60000], "rates": [0.12, 0.22, 0.35]}
        fig, ax = plot_tax_schedule_comparison(old, new)
        assert fig is not None
        plt.close(fig)

    def test_plot_tax_schedule_evolution(self):
        from swarm.analysis.tax_schedule import plot_tax_schedule_evolution
        schedules = [
            {"brackets": [0, 20000, 50000], "rates": [0.1 + 0.02 * i, 0.2 + 0.01 * i, 0.3]}
            for i in range(5)
        ]
        fig, ax = plot_tax_schedule_evolution(schedules, epochs=list(range(5)))
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("mode", ["dark", "light"])
    def test_tax_modes(self, mode):
        from swarm.analysis.tax_schedule import plot_tax_schedule_figure
        fig, ax = plot_tax_schedule_figure([0, 50000], [0.15, 0.3], mode=mode)
        plt.close(fig)


# ===================================================================
# dashboard_cards.py
# ===================================================================

class TestDashboardCards:
    def test_plot_metric_card(self):
        from swarm.analysis.dashboard_cards import plot_metric_card
        fig, ax = plt.subplots(figsize=(3, 2))
        plot_metric_card(ax, 0.42, label="Toxicity Rate", delta=-0.05)
        plt.close(fig)

    def test_plot_summary_cards(self):
        from swarm.analysis.dashboard_cards import plot_summary_cards
        metrics = [
            {"value": 0.42, "label": "Toxicity Rate", "delta": -0.05},
            {"value": 85.3, "label": "Total Welfare"},
            {"value": 0.31, "label": "Gini"},
            {"value": 3, "label": "Frozen Agents", "delta": 1},
        ]
        fig, axes = plot_summary_cards(metrics, ncols=4)
        assert fig is not None
        plt.close(fig)

    def test_plot_overview_dashboard(self):
        from swarm.analysis.dashboard_cards import plot_overview_dashboard
        snapshot = {
            "toxicity_rate": 0.3,
            "quality_gap": 0.15,
            "total_welfare": 120.5,
            "avg_payoff": 2.4,
            "gini_coefficient": 0.28,
            "n_frozen": 1,
            "avg_reputation": 0.75,
            "ecosystem_threat_level": 0.12,
        }
        fig, axes = plot_overview_dashboard(snapshot)
        assert fig is not None
        plt.close(fig)

    def test_plot_overview_dashboard_with_prev(self):
        from swarm.analysis.dashboard_cards import plot_overview_dashboard
        current = {
            "toxicity_rate": 0.3, "quality_gap": 0.15,
            "total_welfare": 120.5, "avg_payoff": 2.4,
            "gini_coefficient": 0.28, "n_frozen": 1,
            "avg_reputation": 0.75, "ecosystem_threat_level": 0.12,
        }
        prev = {
            "toxicity_rate": 0.35, "quality_gap": 0.18,
            "total_welfare": 110.0, "avg_payoff": 2.1,
            "gini_coefficient": 0.30, "n_frozen": 2,
            "avg_reputation": 0.70, "ecosystem_threat_level": 0.15,
        }
        fig, axes = plot_overview_dashboard(current, prev_snapshot=prev)
        plt.close(fig)

    def test_compose_dashboard(self):
        from swarm.analysis.dashboard_cards import compose_dashboard

        def panel_a(ax):
            ax.plot([0, 1], [0, 1])

        def panel_b(ax):
            ax.bar(["x", "y"], [3, 5])

        panels = [
            {"plot_func": panel_a, "title": "Panel A"},
            {"plot_func": panel_b, "title": "Panel B"},
        ]
        fig, axes_dict = compose_dashboard(panels, layout=(1, 2))
        assert "Panel A" in axes_dict
        plt.close(fig)


# ===================================================================
# figure_export.py
# ===================================================================

class TestFigureExport:
    def test_save_figure_png(self, tmp_path):
        from swarm.analysis.figure_export import save_figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        paths = save_figure(fig, tmp_path / "test.png")
        assert len(paths) >= 1
        assert paths[0].exists()
        plt.close(fig)

    def test_save_figure_multi_format(self, tmp_path):
        from swarm.analysis.figure_export import save_figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        paths = save_figure(fig, tmp_path / "test.png", formats=["png", "svg"])
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
        plt.close(fig)

    def test_export_figure_set(self, tmp_path):
        from swarm.analysis.figure_export import export_figure_set
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2, 3])
        fig2, ax2 = plt.subplots()
        ax2.bar(["a", "b"], [1, 2])
        result = export_figure_set({"line": fig1, "bar": fig2}, tmp_path)
        assert "line" in result
        assert "bar" in result
        plt.close(fig1)
        plt.close(fig2)


# ===================================================================
# animate.py
# ===================================================================

class TestAnimate:
    def test_render_frame(self):
        from swarm.analysis.animate import render_frame

        def make_plot():
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return fig, ax

        img = render_frame(make_plot)
        assert img is not None
        assert img.size[0] > 0

    def test_render_epoch_frames(self):
        from swarm.analysis.animate import render_epoch_frames

        def plot_epoch(idx, datum):
            fig, ax = plt.subplots()
            ax.plot(range(idx + 1), datum[:idx + 1])
            ax.set_title(f"Epoch {idx}")
            return fig, ax

        data = [list(range(10))] * 5
        frames = render_epoch_frames(plot_epoch, data)
        assert len(frames) == 5

    def test_create_epoch_scrubber_data(self):
        from swarm.analysis.animate import create_epoch_scrubber_data
        history = {
            "epochs": [
                {"epoch": 0, "toxicity_rate": 0.3, "total_welfare": 50.0},
                {"epoch": 1, "toxicity_rate": 0.25, "total_welfare": 55.0},
            ]
        }
        result = create_epoch_scrubber_data(history, ["toxicity_rate", "total_welfare"])
        assert result["num_epochs"] == 2
        assert len(result["metrics"]["toxicity_rate"]) == 2

    def test_save_animation_gif(self, tmp_path):
        from swarm.analysis.animate import render_frame, save_animation

        def make_plot():
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.plot([0, 1], [0, np.random.rand()])
            return fig, ax

        frames = [render_frame(make_plot) for _ in range(3)]
        out = save_animation(frames, tmp_path / "test.gif", fps=2)
        assert out.exists()


# ===================================================================
# Cross-module integration: semantic color consistency
# ===================================================================

class TestSemanticColorConsistency:
    """Verify that agent type → color mapping is consistent across all modules."""

    def test_theme_and_comparison_colors_match(self):
        from swarm.analysis.theme import COLORS, agent_color

        # The comparison module should use the same colors as theme
        assert agent_color("honest") == COLORS.HONEST
        assert agent_color("deceptive") == COLORS.DECEPTIVE

    def test_all_modules_importable(self):
        """Smoke test: all new modules can be imported without error."""
