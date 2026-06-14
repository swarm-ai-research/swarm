"""Tests for replay run-diff utilities."""

import json
import subprocess
import sys

import pytest

from swarm.replay.diff import (
    compare_run_metrics,
    format_markdown_table,
    normalize_run_metrics,
)


def test_normalize_epoch_history_aggregates_core_metrics():
    payload = {
        "metrics_history": [
            {
                "total_interactions": 10,
                "accepted_interactions": 6,
                "toxicity_rate": 0.2,
                "quality_gap": -0.1,
                "total_welfare": 12.0,
                "net_social_welfare": 10.0,
                "avg_payoff": 1.2,
            },
            {
                "total_interactions": 20,
                "accepted_interactions": 10,
                "toxicity_rate": 0.4,
                "quality_gap": -0.3,
                "total_welfare": 18.0,
                "net_social_welfare": 13.0,
                "avg_payoff": 0.9,
            },
        ]
    }

    metrics = normalize_run_metrics(payload)

    assert metrics["acceptance_rate"] == pytest.approx(16 / 30)
    assert metrics["toxicity_rate"] == pytest.approx(0.3)
    assert metrics["quality_gap"] == pytest.approx(-0.2)
    assert metrics["total_welfare"] == pytest.approx(30.0)
    assert metrics["net_social_welfare"] == pytest.approx(23.0)
    assert metrics["avg_payoff"] == pytest.approx(1.05)


def test_normalize_flat_final_metrics_computes_acceptance_rate():
    payload = {
        "final_metrics": {
            "total_interactions": 8,
            "accepted_interactions": 2,
            "avg_toxicity": 0.125,
            "avg_quality_gap": 0.25,
            "welfare_total": 5.0,
        }
    }

    metrics = normalize_run_metrics(payload)

    assert metrics["acceptance_rate"] == pytest.approx(0.25)
    assert metrics["toxicity_rate"] == pytest.approx(0.125)
    assert metrics["quality_gap"] == pytest.approx(0.25)
    assert metrics["total_welfare"] == pytest.approx(5.0)


def test_compare_and_markdown_table_include_delta_first():
    rows = compare_run_metrics(
        {"acceptance_rate": 0.5, "toxicity_rate": 0.2},
        {"acceptance_rate": 0.75, "toxicity_rate": 0.1},
        metrics=["acceptance_rate", "toxicity_rate", "missing_metric"],
    )

    table = format_markdown_table(rows, precision=3)

    assert "| Metric | Delta | Run A | Run B |" in table
    assert "| acceptance_rate | +0.250 | 0.500 | 0.750 |" in table
    assert "| toxicity_rate | -0.100 | 0.200 | 0.100 |" in table
    assert "| missing_metric | n/a | n/a | n/a |" in table


def test_run_diff_cli_outputs_markdown(tmp_path):
    run_a = tmp_path / "run_a.json"
    run_b = tmp_path / "run_b.json"
    run_a.write_text(
        json.dumps({"metrics_history": [{"total_interactions": 10, "accepted_interactions": 5, "toxicity_rate": 0.2}]}),
        encoding="utf-8",
    )
    run_b.write_text(
        json.dumps({"metrics_history": [{"total_interactions": 10, "accepted_interactions": 7, "toxicity_rate": 0.1}]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "swarm.replay.run_diff",
            str(run_a),
            str(run_b),
            "--metrics",
            "acceptance_rate,toxicity_rate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "| acceptance_rate | +0.2000 | 0.5000 | 0.7000 |" in result.stdout
    assert "| toxicity_rate | -0.1000 | 0.2000 | 0.1000 |" in result.stdout
