"""Tests for swarm/analysis/adaptive_overlay_plot.py — data shaping only.

Rendering itself is not unit-tested (matplotlib smoke is done in the
CLI runner via end-to-end execution). The CSV loaders, the seed-
averaging, and the standard-deviation computation are tested.
"""

from __future__ import annotations

import csv

import pytest

from swarm.analysis.adaptive_overlay_plot import (
    _collect_lines,
    _mean_sd,
    load_adaptive,
    load_static,
)


def _write_adaptive_csv(path, rows: list[dict]) -> None:
    fields = [
        "rho", "seed", "iter0_reward", "iterN_reward",
        "iter0_toxicity", "iterN_toxicity",
        "iter0_accept_rate", "iterN_accept_rate",
        "final_n_accepted",
        "final_mean_payoff_accepted",
        "final_mean_payoff_attempted",
        "final_toxicity",
        "final_mean_p",
        "final_accept_rate",
        "degenerate",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_static_csv(path, rows: list[dict]) -> None:
    fields = [
        "baseline", "rho", "seed",
        "n_accepted", "accept_rate",
        "mean_payoff_accepted",
        "mean_payoff_attempted",
        "sum_payoff",
        "toxicity",
        "mean_p",
        "mean_v_hat",
        "mean_progress",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestMeanSd:
    def test_single_value_sd_is_zero(self) -> None:
        m, s = _mean_sd([1.0])
        assert m == 1.0
        assert s == 0.0

    def test_two_values_sample_sd(self) -> None:
        # ddof=1 sample standard deviation.
        m, s = _mean_sd([1.0, 3.0])
        assert m == 2.0
        assert s == pytest.approx(1.4142135, abs=1e-5)


class TestLoadAdaptive:
    def test_groups_by_rho(self, tmp_path) -> None:
        path = tmp_path / "adaptive.csv"
        _write_adaptive_csv(path, [
            {"rho": "0.00", "seed": 42, "iter0_reward": 0.5, "iterN_reward": 0.8,
             "iter0_toxicity": 0.3, "iterN_toxicity": 0.12,
             "iter0_accept_rate": 1.0, "iterN_accept_rate": 1.0,
             "final_n_accepted": 200,
             "final_mean_payoff_accepted": 0.8,
             "final_mean_payoff_attempted": 0.8,
             "final_toxicity": 0.12,
             "final_mean_p": 0.88,
             "final_accept_rate": 1.0,
             "degenerate": 0},
            {"rho": "0.00", "seed": 123, "iter0_reward": 0.5, "iterN_reward": 0.82,
             "iter0_toxicity": 0.31, "iterN_toxicity": 0.12,
             "iter0_accept_rate": 1.0, "iterN_accept_rate": 1.0,
             "final_n_accepted": 200,
             "final_mean_payoff_accepted": 0.82,
             "final_mean_payoff_attempted": 0.82,
             "final_toxicity": 0.12,
             "final_mean_p": 0.88,
             "final_accept_rate": 1.0,
             "degenerate": 0},
            {"rho": "0.50", "seed": 42, "iter0_reward": 0.2, "iterN_reward": 0.7,
             "iter0_toxicity": 0.32, "iterN_toxicity": 0.12,
             "iter0_accept_rate": 0.8, "iterN_accept_rate": 1.0,
             "final_n_accepted": 200,
             "final_mean_payoff_accepted": 0.69,
             "final_mean_payoff_attempted": 0.69,
             "final_toxicity": 0.122,
             "final_mean_p": 0.878,
             "final_accept_rate": 1.0,
             "degenerate": 0},
        ])
        out = load_adaptive(str(path))
        assert set(out) == {0.0, 0.5}
        assert len(out[0.0]["welfare"]) == 2
        assert len(out[0.5]["welfare"]) == 1


class TestLoadStatic:
    def test_groups_by_baseline_then_rho(self, tmp_path) -> None:
        path = tmp_path / "static.csv"
        _write_static_csv(path, [
            {"baseline": "honest", "rho": "0.00", "seed": 42,
             "n_accepted": 200, "accept_rate": 1.0,
             "mean_payoff_accepted": 0.75, "mean_payoff_attempted": 0.75,
             "sum_payoff": 150.0, "toxicity": 0.16, "mean_p": 0.84,
             "mean_v_hat": 0.6, "mean_progress": 0.7},
            {"baseline": "honest", "rho": "0.00", "seed": 123,
             "n_accepted": 200, "accept_rate": 1.0,
             "mean_payoff_accepted": 0.76, "mean_payoff_attempted": 0.76,
             "sum_payoff": 152.0, "toxicity": 0.17, "mean_p": 0.83,
             "mean_v_hat": 0.6, "mean_progress": 0.7},
            {"baseline": "toxic", "rho": "0.00", "seed": 42,
             "n_accepted": 191, "accept_rate": 0.955,
             "mean_payoff_accepted": 0.10, "mean_payoff_attempted": 0.095,
             "sum_payoff": 19.0, "toxicity": 0.6, "mean_p": 0.4,
             "mean_v_hat": -0.2, "mean_progress": -0.1},
        ])
        out = load_static(str(path))
        assert set(out) == {"honest", "toxic"}
        assert 0.0 in out["honest"]
        assert len(out["honest"][0.0]["welfare"]) == 2
        assert len(out["toxic"][0.0]["welfare"]) == 1


class TestCollectLines:
    def test_adaptive_and_each_static_baseline_present(self) -> None:
        adaptive = {0.0: {"welfare": [0.8, 0.81], "toxicity": [0.12, 0.12],
                          "accept": [1.0, 1.0]}}
        static = {
            "honest": {0.0: {"welfare": [0.75], "toxicity": [0.16],
                             "accept": [1.0]}},
            "mixed": {0.0: {"welfare": [0.5], "toxicity": [0.30],
                            "accept": [0.98]}},
        }
        lines = _collect_lines(adaptive, static, "welfare")
        assert "adaptive" in lines
        assert "static_honest" in lines
        assert "static_mixed" in lines
        rhos, means, stds = lines["adaptive"]
        assert rhos == [0.0]
        assert means[0] == pytest.approx(0.805)


class TestEndToEndPlotSmoke:
    def test_plot_overlay_produces_a_real_png(self, tmp_path) -> None:
        from swarm.analysis.adaptive_overlay_plot import plot_overlay

        adaptive = tmp_path / "adaptive.csv"
        static = tmp_path / "static.csv"
        output = tmp_path / "out.png"

        _write_adaptive_csv(adaptive, [
            {"rho": "0.00", "seed": 42, "iter0_reward": 0.5, "iterN_reward": 0.8,
             "iter0_toxicity": 0.3, "iterN_toxicity": 0.12,
             "iter0_accept_rate": 1.0, "iterN_accept_rate": 1.0,
             "final_n_accepted": 200,
             "final_mean_payoff_accepted": 0.8,
             "final_mean_payoff_attempted": 0.8,
             "final_toxicity": 0.12,
             "final_mean_p": 0.88,
             "final_accept_rate": 1.0,
             "degenerate": 0},
            {"rho": "1.00", "seed": 42, "iter0_reward": 0.0, "iterN_reward": 0.5,
             "iter0_toxicity": 0.4, "iterN_toxicity": 0.13,
             "iter0_accept_rate": 0.1, "iterN_accept_rate": 1.0,
             "final_n_accepted": 200,
             "final_mean_payoff_accepted": 0.54,
             "final_mean_payoff_attempted": 0.54,
             "final_toxicity": 0.13,
             "final_mean_p": 0.87,
             "final_accept_rate": 1.0,
             "degenerate": 0},
        ])
        _write_static_csv(static, [
            {"baseline": "honest", "rho": "0.00", "seed": 42,
             "n_accepted": 200, "accept_rate": 1.0,
             "mean_payoff_accepted": 0.75, "mean_payoff_attempted": 0.75,
             "sum_payoff": 150.0, "toxicity": 0.16, "mean_p": 0.84,
             "mean_v_hat": 0.6, "mean_progress": 0.7},
            {"baseline": "honest", "rho": "1.00", "seed": 42,
             "n_accepted": 200, "accept_rate": 1.0,
             "mean_payoff_accepted": 0.42, "mean_payoff_attempted": 0.42,
             "sum_payoff": 84.0, "toxicity": 0.16, "mean_p": 0.84,
             "mean_v_hat": 0.6, "mean_progress": 0.7},
        ])

        plot_overlay(str(adaptive), str(static), str(output))
        assert output.exists()
        # Smoke: real file, not zero bytes.
        assert output.stat().st_size > 1000
