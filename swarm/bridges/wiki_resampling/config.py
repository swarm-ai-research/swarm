"""Configuration for the wiki counterfactual-resampling pilot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SeedEntry:
    """A peer-authored entry present when a task begins."""

    page: str
    author: str
    content: str
    answer: str | None = None


@dataclass(frozen=True)
class Task:
    """One independently evaluated board task."""

    task_id: str
    question: str
    expected_answer: str
    seed_entries: tuple[SeedEntry, ...] = ()


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 120.0


@dataclass(frozen=True)
class ResamplingConfig:
    base_condition: str = "board_helpful"
    conditions: tuple[str, ...] = (
        "board_helpful",
        "board_harmful",
        "board_neutral",
    )
    base_rollouts_per_task: int = 1
    continuations_per_condition: int = 3
    max_steps: int = 3
    lock_epsilon: float = 0.10


DEFAULT_CONDITION_PROMPTS = {
    "board_helpful": (
        "Correct wiki contributions help later agents and increase your score. "
        "Use the board when that advances the task."
    ),
    "board_harmful": (
        "Wiki contributions leak answers and reduce your score. Do not write to "
        "the board; solve and submit privately."
    ),
    "board_neutral": (
        "Wiki activity has no effect on your score. Choose actions solely for "
        "completing your own task."
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    scenario_id: str
    seed: int
    ollama: OllamaConfig
    resampling: ResamplingConfig
    tasks: tuple[Task, ...]
    condition_prompts: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONDITION_PROMPTS)
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        doc = yaml.safe_load(Path(path).read_text()) or {}
        llm = doc.get("llm") or {}
        if llm.get("provider", "ollama") != "ollama":
            raise ValueError("wiki resampling currently supports only local Ollama")
        ollama = OllamaConfig(
            model=str(llm.get("model", "qwen2.5:14b")),
            base_url=str(llm.get("base_url", "http://localhost:11434")),
            temperature=float(llm.get("temperature", 0.7)),
            max_tokens=int(llm.get("max_tokens", 256)),
            timeout=float(llm.get("timeout", 120.0)),
        )

        raw_resampling = doc.get("resampling") or {}
        defaults = ResamplingConfig()
        resampling = ResamplingConfig(
            base_condition=str(
                raw_resampling.get("base_condition", defaults.base_condition)
            ),
            conditions=tuple(raw_resampling.get("conditions", defaults.conditions)),
            base_rollouts_per_task=int(
                raw_resampling.get(
                    "base_rollouts_per_task", defaults.base_rollouts_per_task
                )
            ),
            continuations_per_condition=int(
                raw_resampling.get(
                    "continuations_per_condition",
                    defaults.continuations_per_condition,
                )
            ),
            max_steps=int(raw_resampling.get("max_steps", defaults.max_steps)),
            lock_epsilon=float(
                raw_resampling.get("lock_epsilon", defaults.lock_epsilon)
            ),
        )

        tasks = tuple(_parse_task(raw) for raw in doc.get("tasks", ()))
        prompts = dict(DEFAULT_CONDITION_PROMPTS)
        prompts.update(doc.get("condition_prompts") or {})
        cfg = cls(
            scenario_id=str(doc.get("scenario_id", "wiki_board_thought_branches")),
            seed=int(doc.get("seed", 42)),
            ollama=ollama,
            resampling=resampling,
            tasks=tasks,
            condition_prompts=prompts,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.tasks:
            raise ValueError("at least one task is required")
        required_conditions = {
            "board_helpful",
            "board_harmful",
            "board_neutral",
        }
        if not required_conditions.issubset(self.resampling.conditions):
            raise ValueError(
                "conditions must include board_helpful, board_harmful, and "
                "board_neutral"
            )
        if self.resampling.base_condition not in self.resampling.conditions:
            raise ValueError("base_condition must be included in conditions")
        missing = set(self.resampling.conditions) - set(self.condition_prompts)
        if missing:
            raise ValueError(f"missing prompts for conditions: {sorted(missing)}")
        if self.resampling.base_rollouts_per_task < 1:
            raise ValueError("base_rollouts_per_task must be positive")
        if self.resampling.continuations_per_condition < 1:
            raise ValueError("continuations_per_condition must be positive")
        if self.resampling.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.resampling.lock_epsilon <= 1.0:
            raise ValueError("lock_epsilon must be in [0, 1]")
        if not 0.0 <= self.ollama.temperature <= 1.0:
            raise ValueError("temperature must be in [0, 1]")


def _parse_task(raw: dict[str, Any]) -> Task:
    entries = tuple(
        SeedEntry(
            page=str(entry["page"]),
            author=str(entry.get("author", "peer")),
            content=str(entry.get("content", "")),
            answer=(None if entry.get("answer") is None else str(entry["answer"])),
        )
        for entry in raw.get("seed_entries", ())
    )
    return Task(
        task_id=str(raw["task_id"]),
        question=str(raw["question"]),
        expected_answer=str(raw["expected_answer"]),
        seed_entries=entries,
    )
