"""Tests for the local wiki Thought-Branches pilot."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from swarm.bridges.wiki_resampling.__main__ import _format_summary
from swarm.bridges.wiki_resampling.board import WikiBoard
from swarm.bridges.wiki_resampling.config import ExperimentConfig, SeedEntry
from swarm.bridges.wiki_resampling.model import OllamaClient, parse_json_object
from swarm.bridges.wiki_resampling.runner import (
    TrajectoryState,
    _execute_action,
    run_experiment,
)

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
FACTORIAL_SCENARIO = (
    Path(__file__).resolve().parent.parent
    / "scenarios"
    / "wiki_board_journal_ablation_factorial.yaml"
)
BENEFICIARY_SCENARIO = (
    Path(__file__).resolve().parent.parent
    / "scenarios"
    / "wiki_board_downstream_beneficiary.yaml"
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


class InterruptingModel:
    """Raise after a fixed number of calls to simulate a stopped local run."""

    def __init__(self, delegate: ConditionModel, fail_after: int) -> None:
        self.delegate = delegate
        self.fail_after = fail_after
        self.calls = 0

    def generate(self, prompt: str, *, seed: int) -> str:
        if self.calls >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls += 1
        return self.delegate.generate(prompt, seed=seed)


class BeneficiaryModel(ConditionModel):
    """Write the correct answer to the beneficiary's only readable page."""

    def generate(self, prompt: str, *, seed: int) -> str:
        if (
            "one short decision-journal sentence" not in prompt
            and "[CONDITION=board_helpful]" in prompt
        ):
            return json.dumps(
                {
                    "action": "WRITE_WIKI",
                    "page": "derived-alpha",
                    "content": "derived result",
                    "answer": "750",
                }
            )
        return super().generate(prompt, seed=seed)


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


def test_factorial_scenario_pairs_hint_and_journal_interventions() -> None:
    cfg = ExperimentConfig.from_yaml(FACTORIAL_SCENARIO)

    assert cfg.resampling.journal_interventions == ("retained", "ablated")
    assert cfg.resampling.pre_write_only is True
    assert {task.task_id for task in cfg.tasks} == {"hint_present", "hint_absent"}
    questions = {task.task_id: task.question for task in cfg.tasks}
    assert "derived-alpha" in questions["hint_present"]
    assert "derived-alpha" not in questions["hint_absent"]
    assert all(task.expected_answer == "750" for task in cfg.tasks)
    assert all(task.seed_entries == cfg.tasks[0].seed_entries for task in cfg.tasks)


def test_downstream_beneficiary_is_scored_from_board_state(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(BENEFICIARY_SCENARIO)
    task = cfg.tasks[0]
    consumer = task.downstream_consumer

    assert cfg.resampling.base_rollouts_per_task == 4
    assert consumer is not None
    assert consumer.readable_pages == ("derived-alpha",)
    assert consumer.expected_answer == "750"

    invalid_pages = replace(
        cfg,
        tasks=(
            replace(
                task,
                downstream_consumer=replace(consumer, readable_pages=(" ",)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="readable pages must not be empty"):
        invalid_pages.validate()

    scalar_scenario = tmp_path / "scalar-pages.yaml"
    scalar_scenario.write_text(
        BENEFICIARY_SCENARIO.read_text().replace(
            "readable_pages: [derived-alpha]", "readable_pages: derived-alpha"
        )
    )
    with pytest.raises(ValueError, match="readable_pages must be a sequence"):
        ExperimentConfig.from_yaml(scalar_scenario)

    result = run_experiment(
        replace(
            cfg,
            resampling=replace(
                cfg.resampling,
                base_rollouts_per_task=1,
                continuations_per_condition=1,
                max_steps=1,
            ),
        ),
        BeneficiaryModel(),
    )
    helpful = next(
        row
        for row in result["branches"]
        if row["checkpoint_index"] == 0 and row["condition"] == "board_helpful"
    )
    assert helpful["posted"] is True
    assert helpful["beneficiary_success"] is True
    assert result["summary"][0]["beneficiary_success_rate_by_condition"] == {
        "board_harmful": 0.0,
        "board_helpful": 1.0,
        "board_neutral": 0.0,
    }

    wrong_page = run_experiment(
        replace(
            cfg,
            resampling=replace(
                cfg.resampling,
                base_rollouts_per_task=1,
                continuations_per_condition=1,
                max_steps=1,
            ),
        ),
        ConditionModel(),
    )
    wrong_page_helpful = next(
        row
        for row in wrong_page["branches"]
        if row["checkpoint_index"] == 0 and row["condition"] == "board_helpful"
    )
    assert wrong_page_helpful["posted"] is True
    assert wrong_page_helpful["beneficiary_success"] is False


def test_config_rejects_negative_seed() -> None:
    cfg = ExperimentConfig.from_yaml(SCENARIO)
    with pytest.raises(ValueError, match="seed must be non-negative"):
        replace(cfg, seed=-1).validate()

    with pytest.raises(ValueError, match="seed must be non-negative"):
        run_experiment(replace(cfg, seed=-1), ConditionModel())


def test_config_rejects_unknown_journal_intervention() -> None:
    cfg = ExperimentConfig.from_yaml(SCENARIO)
    invalid = replace(
        cfg,
        resampling=replace(cfg.resampling, journal_interventions=("retained", "swap")),
    )
    with pytest.raises(ValueError, match="unknown journal interventions"):
        invalid.validate()

    duplicate = replace(
        cfg,
        resampling=replace(
            cfg.resampling, journal_interventions=("retained", "retained")
        ),
    )
    with pytest.raises(ValueError, match="journal_interventions must be unique"):
        duplicate.validate()


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


def test_json_parser_accepts_first_object_before_trailing_text() -> None:
    assert parse_json_object('Result: {"journal": {"status": "ok"}} trailing {bad}') == {
        "journal": {"status": "ok"}
    }


def test_cli_summary_does_not_log_scenario_task_identifier() -> None:
    rendered = _format_summary(3)

    assert "summary=3" in rendered
    assert "summary.json" in rendered


@pytest.mark.parametrize(
    ("action", "missing_field"),
    [
        ({"action": "WRITE_WIKI", "page": "", "content": "x", "answer": "1"}, "page"),
        ({"action": "WRITE_WIKI", "page": "p", "content": "", "answer": "1"}, "content"),
        ({"action": "WRITE_WIKI", "page": "p", "content": "x", "answer": None}, "answer"),
    ],
)
def test_incomplete_write_is_not_executed(
    action: dict[str, object], missing_field: str
) -> None:
    del missing_field
    state = TrajectoryState(task_id="task")
    board = WikiBoard()

    _execute_action(action, state, board)

    assert board.entries == ()
    assert state.events == [{"step": 0, **action, "success": False}]


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
    assert json.loads((tmp_path / "complete.json").read_text())["complete"] is True
    branch_lines = (tmp_path / "branches.jsonl").read_text().splitlines()
    assert branch_lines
    branches = [json.loads(line) for line in branch_lines]
    assert all(row["seed"] >= 0 for row in branches)
    assert all(row["state"]["exchanges"] for row in branches)


def test_journal_ablation_preserves_non_journal_prefix_state() -> None:
    cfg = ExperimentConfig.from_yaml(FACTORIAL_SCENARIO)
    cfg = replace(
        cfg,
        tasks=(cfg.tasks[0],),
        resampling=replace(
            cfg.resampling,
            base_rollouts_per_task=1,
            continuations_per_condition=1,
            max_steps=2,
        ),
    )

    result = run_experiment(cfg, ConditionModel())

    assert {row["checkpoint_index"] for row in result["branches"]} == {0, 1}
    assert {
        row["journal_intervention"]
        for row in result["branches"]
        if row["checkpoint_index"] == 0
    } == {"retained"}
    checkpoint_rows = {
        (row["condition"], row["journal_intervention"]): row
        for row in result["branches"]
        if row["checkpoint_index"] == 1
    }
    for condition in cfg.resampling.conditions:
        retained = checkpoint_rows[(condition, "retained")]
        ablated = checkpoint_rows[(condition, "ablated")]
        assert retained["source_prefix_journals"] == ablated["source_prefix_journals"]
        assert len(retained["prefix_journals"]) == 1
        assert ablated["prefix_journals"] == []
        assert retained["prefix_events"] == ablated["prefix_events"]
        assert retained["prefix_read_entries"] == ablated["prefix_read_entries"]
        assert retained["prefix_board"] == ablated["prefix_board"]
    assert {
        row["journal_intervention"]
        for row in result["summary"]
        if row["checkpoint_index"] == 1
    } == {"retained", "ablated"}


def test_interrupted_run_resumes_without_duplicate_branches(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(SCENARIO)
    cfg = replace(
        cfg,
        tasks=(cfg.tasks[0],),
        resampling=replace(
            cfg.resampling,
            base_rollouts_per_task=1,
            continuations_per_condition=1,
            max_steps=1,
        ),
    )
    interrupted_dir = tmp_path / "interrupted"
    progress: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_experiment(
            cfg,
            InterruptingModel(ConditionModel(), fail_after=5),
            out_dir=interrupted_dir,
            progress=progress.append,
        )

    assert (interrupted_dir / "run_manifest.json").exists()
    assert (interrupted_dir / "base_trajectories.partial.jsonl").exists()
    partial_branches = [
        json.loads(line)
        for line in (interrupted_dir / "branches.partial.jsonl")
        .read_text()
        .splitlines()
    ]
    assert partial_branches
    assert not (interrupted_dir / "summary.json").exists()
    assert not (interrupted_dir / "complete.json").exists()
    with (interrupted_dir / "branches.partial.jsonl").open("a") as handle:
        handle.write('{"truncated"')

    with pytest.raises(ValueError, match="configuration does not match"):
        run_experiment(
            replace(cfg, seed=cfg.seed + 1),
            ConditionModel(),
            out_dir=interrupted_dir,
            resume=True,
        )

    resumed = run_experiment(
        cfg,
        ConditionModel(),
        out_dir=interrupted_dir,
        resume=True,
        progress=progress.append,
    )
    uninterrupted = run_experiment(
        cfg,
        ConditionModel(),
        out_dir=tmp_path / "uninterrupted",
    )

    resumed_ids = [row["trajectory_id"] for row in resumed["branches"]]
    assert len(resumed_ids) == len(set(resumed_ids))
    assert resumed["base_trajectories"] == uninterrupted["base_trajectories"]
    assert resumed["branches"] == uninterrupted["branches"]
    assert resumed["summary"] == uninterrupted["summary"]
    assert json.loads((interrupted_dir / "complete.json").read_text()) == {
        "base_trajectories": 1,
        "branches": len(resumed["branches"]),
        "complete": True,
    }
    counts: list[int] = []
    for event in progress:
        count = event["branch_completed"]
        assert isinstance(count, int)
        counts.append(count)
    assert counts == sorted(counts)

    completed_resume = run_experiment(
        cfg, ConditionModel(), out_dir=interrupted_dir, resume=True
    )
    assert completed_resume["branches"] == resumed["branches"]
