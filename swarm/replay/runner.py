"""Replay runner for executing K seed-variant runs of one scenario."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from swarm.core.orchestrator import Orchestrator
from swarm.replay.episode_spec import EpisodeSpec
from swarm.scenarios.loader import build_orchestrator


def _strip_prefix(path: str, prefix: str) -> Optional[str]:
    """Extract suffix after prefix, or None if prefix does not match.

    Args:
        path: The full path string
        prefix: The prefix to strip (e.g., "simulation.")

    Returns:
        The suffix after the prefix, or None if prefix does not match.
    """
    if path.startswith(prefix):
        return path[len(prefix) :]
    return None


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute using dot notation."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


@dataclass
class ReplayRunResult:
    """Summary result from one replay execution."""

    replay_index: int
    seed: int
    total_interactions: int
    accepted_interactions: int
    avg_toxicity: float
    avg_quality_gap: float
    total_welfare: float


class ReplayRunner:
    """Executes repeated runs of one `EpisodeSpec`."""

    def __init__(
        self,
        episode_spec: EpisodeSpec,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.episode_spec = episode_spec
        self.progress_callback = progress_callback
        self.results: List[ReplayRunResult] = []

    def run(self) -> List[ReplayRunResult]:
        """Execute K replays for the configured episode spec."""
        self.results = []
        seeds = self.episode_spec.replay_seeds()
        total = len(seeds)

        for replay_index, seed in enumerate(seeds):
            if self.progress_callback is not None:
                self.progress_callback(replay_index + 1, total)

            orchestrator = self._build_orchestrator_with_seed(seed, replay_index)
            orchestrator.run()
            self.results.append(
                self._extract_result(
                    orchestrator=orchestrator,
                    replay_index=replay_index,
                    seed=seed,
                )
            )

        return self.results

    def _build_orchestrator_with_seed(
        self, seed: int, replay_index: int
    ) -> Orchestrator:
        """Deep-copy scenario config, apply overrides, then build orchestrator."""
        scenario = copy.deepcopy(self.episode_spec.scenario)
        scenario.orchestrator_config.seed = seed
        scenario.orchestrator_config.replay_k = replay_index
        # Replay experiments should not mutate on-disk logs by default.
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False

        for path, value in self.episode_spec.parameter_overrides.items():
            sim_attr = _strip_prefix(path, "simulation.")
            if sim_attr is not None:
                setattr(scenario.orchestrator_config, sim_attr, value)
            else:
                gov_attr = _strip_prefix(path, "governance.")
                if gov_attr is not None:
                    setattr(scenario.orchestrator_config.governance_config, gov_attr, value)
                else:
                    _set_nested_attr(scenario.orchestrator_config, path, value)

        return build_orchestrator(scenario)

    @staticmethod
    def _extract_result(
        orchestrator: Orchestrator,
        replay_index: int,
        seed: int,
    ) -> ReplayRunResult:
        """Extract summary metrics from an executed orchestrator."""
        metrics_history = orchestrator.get_metrics_history()
        n_epochs = len(metrics_history)

        total_interactions = sum(
            metric.total_interactions for metric in metrics_history
        )
        accepted_interactions = sum(
            metric.accepted_interactions for metric in metrics_history
        )
        total_welfare = sum(metric.total_welfare for metric in metrics_history)
        avg_toxicity = (
            sum(metric.toxicity_rate for metric in metrics_history) / n_epochs
            if n_epochs
            else 0.0
        )
        avg_quality_gap = (
            sum(metric.quality_gap for metric in metrics_history) / n_epochs
            if n_epochs
            else 0.0
        )

        return ReplayRunResult(
            replay_index=replay_index,
            seed=seed,
            total_interactions=total_interactions,
            accepted_interactions=accepted_interactions,
            avg_toxicity=avg_toxicity,
            avg_quality_gap=avg_quality_gap,
            total_welfare=total_welfare,
        )

    def grouped_by_metric(self) -> Dict[str, List[float]]:
        """Return replay values grouped by metric key."""
        return {
            "total_interactions": [r.total_interactions for r in self.results],
            "accepted_interactions": [r.accepted_interactions for r in self.results],
            "avg_toxicity": [r.avg_toxicity for r in self.results],
            "avg_quality_gap": [r.avg_quality_gap for r in self.results],
            "total_welfare": [r.total_welfare for r in self.results],
        }
