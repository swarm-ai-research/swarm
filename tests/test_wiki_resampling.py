"""Tests for the local wiki Thought-Branches pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarm.bridges.wiki_resampling.board import WikiBoard
from swarm.bridges.wiki_resampling.config import ExperimentConfig, SeedEntry
from swarm.bridges.wiki_resampling.model import OllamaClient, parse_json_object
from swarm.bridges.wiki_resampling.runner import run_experiment

SCENARIO = (
    Path(__file__).resolve().parent.parent
    / "scenarios"
    / "wiki_board_thought_branches.yaml"
)
RETRIEVAL_SCENARIO = (
    Path(__file__).resolve().parent.parent
    / "scenarios"
    / "wiki_board_thought_branches_retrieval.yaml"
)


class ConditionModel:
    """Deterministic model whose board policy is controlled by the condition."""

    def generate(self, prompt: str, *, seed: int) -> str:
        del seed
        if "one short decision-journal sentence" in prompt:
            return json.dumps({"journal": "I will follow the current incentive."})
        if "[CONDITION=board_helpful]" in prompt:
            return json.dumps(
                {
                    "action": "WRITE_WIKI",
                    "page": "results",
                    "content": "candidate",
                    "answer": "1073",
                }
            )
        answer = "1073" if "Task multiplication" in prompt else "48"
        return json.dumps({"action": "SUBMIT", "answer": answer})


def test_scenario_loads() -> None:
    cfg = ExperimentConfig.from_yaml(SCENARIO)
    assert cfg.scenario_id == "wiki_board_thought_branches"
    assert cfg.ollama.model == "qwen2.5:14b"
    assert len(cfg.tasks) == 2


def test_retrieval_scenario_requires_board_input() -> None:
    cfg = ExperimentConfig.from_yaml(RETRIEVAL_SCENARIO)
    assert cfg.scenario_id == "wiki_board_thought_branches_retrieval"
    assert len(cfg.tasks) == 1
    task = cfg.tasks[0]
    assert task.expected_answer == "750"
    assert any(entry.answer == "731" for entry in task.seed_entries)
    assert "731" not in task.question


def test_board_snapshot_restores_without_aliasing() -> None:
    board = WikiBoard.from_seed_entries(
        [SeedEntry(page="x", author="peer", content="answer", answer="7")]
    )
    restored = WikiBoard.restore(board.snapshot())
    restored.write(page="x", author="focal", content="new", answer="8")
    assert len(board.entries) == 1
    assert len(restored.entries) == 2


def test_json_parser_accepts_fences() -> None:
    assert parse_json_object('```json\n{"journal": "ok"}\n```') == {"journal": "ok"}


def test_ollama_client_rejects_non_local_endpoint() -> None:
    with pytest.raises(ValueError, match="local machine"):
        OllamaClient(model="x", base_url="https://example.com")


def test_experiment_branches_conditions_and_writes_artifacts(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(SCENARIO)
    result = run_experiment(
        cfg,
        ConditionModel(),
        out_dir=tmp_path,
        continuations_per_condition=2,
    )

    initial = {
        row["task_id"]: row for row in result["summary"] if row["checkpoint_index"] == 0
    }
    for row in initial.values():
        assert row["post_rate_by_condition"]["board_helpful"] == 1.0
        assert row["post_rate_by_condition"]["board_harmful"] == 0.0
        assert row["prompt_dependence"] == 1.0
        assert row["prefix_carried_posting"] == 0.0
        assert row["locked"] is False

    assert result["base_trajectories"][0]["checkpoints"]
    assert result["base_trajectories"][0]["state"]["exchanges"]

    assert (tmp_path / "base_trajectories.jsonl").exists()
    assert (tmp_path / "branches.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    branch_lines = (tmp_path / "branches.jsonl").read_text().splitlines()
    assert branch_lines
    branches = [json.loads(line) for line in branch_lines]
    assert all(row["seed"] >= 0 for row in branches)
    assert all(row["state"]["exchanges"] for row in branches)
