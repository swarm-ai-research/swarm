"""Tests for scenario YAML schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from swarm.models.scenario import ScenarioConfig
from swarm.scenarios.loader import load_scenario


def _minimal_valid_scenario() -> dict:
    return {
        "scenario_id": "schema_test",
        "description": "Minimal scenario used for schema validation tests.",
        "agents": [{"type": "honest", "count": 1}],
        "simulation": {"n_epochs": 1, "steps_per_epoch": 1, "seed": 7},
    }


def test_all_checked_in_scenarios_pass_schema_validation():
    for path in Path("scenarios").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        ScenarioConfig.model_validate(payload)


def test_load_scenario_runs_schema_validation(tmp_path):
    path = tmp_path / "valid.yaml"
    path.write_text(yaml.safe_dump(_minimal_valid_scenario()), encoding="utf-8")

    scenario = load_scenario(path)

    assert scenario.scenario_id == "schema_test"
    assert scenario.agent_specs == [{"type": "honest", "count": 1}]


def test_missing_required_field_raises_validation_error(tmp_path):
    payload = _minimal_valid_scenario()
    del payload["scenario_id"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="scenario_id"):
        load_scenario(path)


def test_wrong_field_type_raises_validation_error(tmp_path):
    payload = _minimal_valid_scenario()
    payload["simulation"]["n_epochs"] = "many"
    path = tmp_path / "wrong_type.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="n_epochs"):
        load_scenario(path)


def test_extra_top_level_field_raises_validation_error(tmp_path):
    payload = _minimal_valid_scenario()
    payload["surprise"] = {"enabled": True}
    path = tmp_path / "extra.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="surprise"):
        load_scenario(path)
