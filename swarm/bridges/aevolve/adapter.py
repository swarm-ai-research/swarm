"""SWARM scenarios as an A-Evolve benchmark (``BenchmarkAdapter``).

A-Evolve (`A-EVO-Lab/a-evolve`) evolves agents by mutating a file-system
workspace and scoring candidates against a benchmark through a three-method
contract: ``get_tasks() -> [Task]``, ``evaluate(task, trajectory) ->
Feedback{success, score, detail}``.

This adapter makes a SWARM governance scenario that benchmark. Each task
asks the evolved agent to *design governance*: emit a JSON object of
whitelisted governance/payoff overrides for a fixed multi-agent scenario
(honest + opportunistic + deceptive mix). ``evaluate`` merges the overrides
into the scenario YAML, runs the simulation headlessly for that task's
seed, and scores the outcome from ``SoftMetrics``:

- ``one_minus_toxicity``: ``1 - E[1-p | accepted]`` on the final epoch
- ``s_soft``: sigmoid-squashed mean welfare per interaction

The whitelist is the benchmark's integrity boundary: candidates tune
governance levers, they cannot shrink the simulation, change the agent
mix, or move the seed — that would be gaming the harness, not designing
governance. ``Feedback.detail`` carries rich diagnostics (toxicity,
quality gap, welfare, rejected overrides, adverse-selection warnings) so
the evolver can actually learn from failures.

A-Evolve is an optional dependency (GitHub-only, pinned by SHA in
``pyproject.toml``; NOT vendored — upstream declares MIT but ships no
LICENSE file as of pin ``c9d4789``). When ``agent_evolve`` is not
installed, structural stand-ins for ``Task``/``Trajectory``/``Feedback``
keep the adapter importable and fully testable offline.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from swarm.bridges.aevolve.config import AevolveBridgeConfig, EvaluationDetail

try:  # pragma: no cover — exercised only with the optional extra installed
    from agent_evolve.benchmarks.base import BenchmarkAdapter as _UpstreamAdapter
    from agent_evolve.types import Feedback, Task, Trajectory

    AEVOLVE_AVAILABLE = True
except ImportError:
    AEVOLVE_AVAILABLE = False

    @dataclass
    class Task:  # type: ignore[no-redef]
        """Structural stand-in matching ``agent_evolve.types.Task``."""

        id: str
        input: str
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Trajectory:  # type: ignore[no-redef]
        """Structural stand-in matching ``agent_evolve.types.Trajectory``."""

        task_id: str
        output: str
        steps: List[Dict[str, Any]] = field(default_factory=list)
        conversation: List[Dict[str, Any]] = field(default_factory=list)

    @dataclass
    class Feedback:  # type: ignore[no-redef]
        """Structural stand-in matching ``agent_evolve.types.Feedback``."""

        success: bool
        score: float
        detail: str
        raw: Dict[str, Any] = field(default_factory=dict)

    class _UpstreamAdapter:  # type: ignore[no-redef]
        """No-op base so the subclass works without agent_evolve."""


_TASK_PROMPT = """You are designing governance for a multi-agent AI ecosystem.

Scenario: {description}
Agent mix: {agent_mix}. The simulation runs a fixed number of epochs with a
fixed seed; you control ONLY the governance and payoff levers below.

Respond with a single JSON object mapping lever names to values. Allowed
levers: {allowed}.

Your design is scored on the resulting ecosystem: accepted interactions
should be high-quality (low toxicity) without collapsing welfare. Levers
have costs — e.g. taxes and stakes deter bad actors but also tax honest
throughput. Unknown or disallowed keys are ignored (and reported back).
"""


class SwarmBenchmarkAdapter(_UpstreamAdapter):
    """A-Evolve ``BenchmarkAdapter`` over a SWARM governance scenario."""

    name = "swarm-governance"

    def __init__(self, config: Optional[AevolveBridgeConfig] = None) -> None:
        self.config = config or AevolveBridgeConfig()
        self._scenario_raw: Optional[Dict[str, Any]] = None

    # -- benchmark contract --------------------------------------------------
    def get_tasks(self, split: str = "train", limit: int = 10) -> List[Task]:
        """One task per (scenario, seed) cell; holdout seeds are disjoint."""

        raw = self._load_scenario_raw()
        base = (
            self.config.seed_base
            if split == "train"
            else self.config.holdout_seed_base
        )
        agent_mix = ", ".join(
            f"{a.get('count', 1)}× {a.get('type', '?')}"
            for a in raw.get("agents", [])
        )
        prompt = _TASK_PROMPT.format(
            description=raw.get("description", raw.get("scenario_id", "swarm")),
            agent_mix=agent_mix or "unspecified",
            allowed=", ".join(self.config.allowed_overrides),
        )
        count = min(limit, self.config.n_tasks)
        return [
            Task(
                id=f"{raw.get('scenario_id', 'swarm')}-{split}-seed{base + i}",
                input=prompt,
                metadata={
                    "scenario_path": str(self.config.scenario_path),
                    "seed": base + i,
                    "split": split,
                },
            )
            for i in range(count)
        ]

    def evaluate(self, task: Task, trajectory: Trajectory) -> Feedback:
        """Merge the candidate's overrides, run the sim, score via SoftMetrics."""

        overrides, parse_error = self._parse_overrides(trajectory.output)
        if parse_error:
            detail = EvaluationDetail(parse_error=parse_error)
            return Feedback(
                success=False, score=0.0, detail=detail.to_text(),
                raw={"parse_error": parse_error},
            )

        applied, rejected = self._partition_overrides(overrides)
        seed = int(task.metadata.get("seed", self.config.seed_base))
        detail = self._run_scenario(seed=seed, overrides=applied)
        detail.rejected_overrides = rejected

        score = self._score(detail)
        return Feedback(
            success=score >= self.config.success_threshold,
            score=score,
            detail=detail.to_text(),
            raw={
                "score_mode": self.config.score_mode,
                "toxicity": detail.toxicity,
                "quality_gap": detail.quality_gap,
                "total_welfare": detail.total_welfare,
                "applied_overrides": applied,
                "rejected_overrides": rejected,
                "seed": seed,
            },
        )

    # -- internals -----------------------------------------------------------
    def _load_scenario_raw(self) -> Dict[str, Any]:
        if self._scenario_raw is None:
            with open(self.config.scenario_path, "r", encoding="utf-8") as fh:
                self._scenario_raw = yaml.safe_load(fh) or {}
        return copy.deepcopy(self._scenario_raw)

    @staticmethod
    def _parse_overrides(output: str) -> tuple[Dict[str, Any], str]:
        """Extract a JSON object from the trajectory output (tolerates prose)."""

        text = (output or "").strip()
        if not text:
            return {}, "empty output"
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}, "no JSON object found in output"
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            return {}, f"invalid JSON: {exc}"
        if not isinstance(data, dict):
            return {}, "JSON payload is not an object"
        return data, ""

    def _partition_overrides(
        self, overrides: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        allowed = set(self.config.allowed_overrides)
        applied = {k: v for k, v in overrides.items() if k in allowed}
        rejected = {k: v for k, v in overrides.items() if k not in allowed}
        return applied, rejected

    def _run_scenario(self, *, seed: int, overrides: Dict[str, Any]) -> EvaluationDetail:
        """Run the scenario headlessly with overrides merged into raw YAML.

        Merges overrides into the *raw* document at exactly 2 levels (section.key).
        Re-loading through the standard ``load_scenario`` path keeps this adapter
        agnostic to loader internals: whatever governance keys the loader understands,
        the benchmark supports. Currently, allowed_overrides whitelist enforces
        2-level keys only; dotted keys with 3+ levels would create flat keys
        (e.g., 'a.b.c' -> raw['a']['b.c']) and should not be whitelisted.
        """

        from swarm.scenarios import build_orchestrator, load_scenario

        raw = self._load_scenario_raw()
        for dotted, value in overrides.items():
            # Assumes exactly 2 levels: section.key. Enforced by allowed_overrides
            # whitelist in config.py.
            section, key = dotted.split(".", 1)
            raw.setdefault(section, {})[key] = value
        raw.setdefault("simulation", {})["seed"] = seed
        if self.config.max_epochs is not None:
            raw["simulation"]["n_epochs"] = min(
                int(raw["simulation"].get("n_epochs", self.config.max_epochs)),
                self.config.max_epochs,
            )
        if self.config.max_steps_per_epoch is not None:
            raw["simulation"]["steps_per_epoch"] = min(
                int(raw["simulation"].get("steps_per_epoch", self.config.max_steps_per_epoch)),
                self.config.max_steps_per_epoch,
            )
        # Benchmark runs are in-memory only; never write logs beside the repo.
        raw.pop("outputs", None)

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            yaml.safe_dump(raw, fh)
            tmp_path = Path(fh.name)
        try:
            scenario = load_scenario(tmp_path)
            orchestrator = build_orchestrator(scenario)
            history = orchestrator.run()
        finally:
            tmp_path.unlink(missing_ok=True)

        final = history[-1] if history else None
        return EvaluationDetail(
            scenario_id=raw.get("scenario_id", ""),
            seed=seed,
            applied_overrides=dict(overrides),
            toxicity=float(getattr(final, "toxicity_rate", 0.0) or 0.0),
            quality_gap=float(getattr(final, "quality_gap", 0.0) or 0.0),
            total_welfare=float(getattr(final, "total_welfare", 0.0) or 0.0),
            total_interactions=int(getattr(final, "total_interactions", 0) or 0),
            accepted_interactions=int(getattr(final, "accepted_interactions", 0) or 0),
        )

    def _score(self, detail: EvaluationDetail) -> float:
        if self.config.score_mode == "s_soft":
            per_interaction = (
                detail.total_welfare / detail.total_interactions
                if detail.total_interactions
                else 0.0
            )
            return 1.0 / (1.0 + math.exp(-per_interaction))
        # one_minus_toxicity (default)
        return max(0.0, min(1.0, 1.0 - detail.toxicity))
