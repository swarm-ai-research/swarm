"""Tests for run-diff utilities."""

import csv
import json
import subprocess
import sys

import pytest

from swarm.analysis.run_diff import (
    compare_run_files,
    compare_run_metrics,
    format_markdown_table,
    load_run_metrics,
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
                "illusion_delta": 0.05,
                "total_welfare": 12.0,
                "net_social_welfare": 10.0,
                "avg_payoff": 1.2,
            },
            {
                "total_interactions": 20,
                "accepted_interactions": 10,
                "toxicity_rate": 0.4,
                "quality_gap": -0.3,
                "illusion_delta": 0.15,
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
    assert metrics["illusion_delta"] == pytest.approx(0.1)
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


def test_compare_and_markdown_table_use_issue_column_order():
    rows = compare_run_metrics(
        {"acceptance_rate": 0.5, "toxicity_rate": 0.2},
        {"acceptance_rate": 0.75, "toxicity_rate": 0.1},
        metrics=["acceptance_rate", "toxicity_rate", "missing_metric"],
    )

    table = format_markdown_table(rows, precision=3)

    assert "| Metric | Run A | Run B | Delta |" in table
    assert "| acceptance_rate | 0.500 | 0.750 | +0.250 |" in table
    assert "| toxicity_rate | 0.200 | 0.100 | -0.100 |" in table
    assert "| missing_metric | N/A | N/A | N/A |" in table


def test_loads_run_directories_with_history_json(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "history.json").write_text(
        json.dumps(
            {
                "epoch_snapshots": [
                    {
                        "toxicity_rate": 0.2,
                        "quality_gap": 0.1,
                        "illusion_delta": 0.02,
                        "avg_payoff": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_b / "history.json").write_text(
        json.dumps(
            {
                "epoch_snapshots": [
                    {
                        "toxicity_rate": 0.3,
                        "quality_gap": 0.4,
                        "illusion_delta": 0.04,
                        "avg_payoff": 1.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = compare_run_files(run_a, run_b)
    table = format_markdown_table(rows, precision=2)

    assert "| toxicity_rate | 0.20 | 0.30 | +0.10 |" in table
    assert "| quality_gap | 0.10 | 0.40 | +0.30 |" in table
    assert "| illusion_delta | 0.02 | 0.04 | +0.02 |" in table
    assert "| avg_payoff | 1.00 | 1.50 | +0.50 |" in table


def test_loads_run_directory_with_csv_export(tmp_path):
    run_dir = tmp_path / "run_csv"
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "simulation_epochs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "toxicity_rate",
                "quality_gap",
                "illusion_delta",
                "avg_payoff",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "toxicity_rate": "0.10",
                "quality_gap": "0.20",
                "illusion_delta": "0.30",
                "avg_payoff": "1.00",
            }
        )
        writer.writerow(
            {
                "epoch": 1,
                "toxicity_rate": "0.30",
                "quality_gap": "0.40",
                "illusion_delta": "0.50",
                "avg_payoff": "3.00",
            }
        )

    metrics = load_run_metrics(run_dir)

    assert metrics["toxicity_rate"] == pytest.approx(0.2)
    assert metrics["quality_gap"] == pytest.approx(0.3)
    assert metrics["illusion_delta"] == pytest.approx(0.4)
    assert metrics["avg_payoff"] == pytest.approx(2.0)


def test_missing_metric_is_rendered_as_na():
    rows = compare_run_metrics(
        {"toxicity_rate": 0.2},
        {"toxicity_rate": 0.1, "quality_gap": 0.4},
        metrics=["toxicity_rate", "quality_gap", "illusion_delta"],
    )

    table = format_markdown_table(rows, precision=2)

    assert "| toxicity_rate | 0.20 | 0.10 | -0.10 |" in table
    assert "| quality_gap | N/A | 0.40 | N/A |" in table
    assert "| illusion_delta | N/A | N/A | N/A |" in table


def test_missing_run_files_raise_clear_error(tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No history.json or epoch CSV"):
        load_run_metrics(run_dir)


def test_run_diff_cli_outputs_markdown_for_directories(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "history.json").write_text(
        json.dumps(
            {
                "epoch_snapshots": [
                    {
                        "total_interactions": 10,
                        "accepted_interactions": 5,
                        "toxicity_rate": 0.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_b / "history.json").write_text(
        json.dumps(
            {
                "epoch_snapshots": [
                    {
                        "total_interactions": 10,
                        "accepted_interactions": 7,
                        "toxicity_rate": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "swarm.analysis.run_diff",
            str(run_a),
            str(run_b),
            "--metrics",
            "acceptance_rate,toxicity_rate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "| acceptance_rate | 0.5000 | 0.7000 | +0.2000 |" in result.stdout
    assert "| toxicity_rate | 0.2000 | 0.1000 | -0.1000 |" in result.stdout
