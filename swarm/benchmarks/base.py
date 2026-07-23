"""Base classes for the benchmark suite.

Design principle: each benchmark task has a governance-free oracle run that
provides the denominator for capability ratios. Every governance configuration
becomes a point at (capability_ratio, safety_gain), making frontiers
interpretable.

Security invariants:
- TaskInstance visible to run_fn NEVER contains ground truth (oracle answers).
- Oracle data is held in a separate TaskOracle object, accessible only to the
  scorer.
- Instances are deep-copied before being passed to run_fn to prevent mutation
  of shared state.
- run_fn return values are validated before scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskInstance:
    """A concrete, generated task instance visible to agents.

    SECURITY: This class must NEVER contain ground-truth answers (expected
    outputs, optimal assignments, oracle paths, etc.). Those belong in
    TaskOracle. Any field that would let a run_fn cheat must be excluded.
    """

    task_id: str
    seed: int
    n_agents: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskOracle:
    """Ground-truth data for scoring, hidden from run_fn.

    Only the benchmark's score() method should access this.
    """

    oracle_result: TaskResult
    ground_truth: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Outcome of running a single task instance."""

    completed: bool
    payload: Any
    steps_taken: int
    agent_trace: list[str]  # ordered agent IDs that touched this task


@dataclass
class BenchmarkScore:
    """Scores for a single (task, governance_config) evaluation."""

    completion_rate: float  # 0-1, primary capability metric
    efficiency: float  # oracle_steps / actual_steps (capped at 1)
    fidelity: float  # payload integrity vs oracle output (0-1, supports partial)
    capability_ratio: float  # composite capability vs oracle
    safety_score: float = 0.0  # 0-1, measures governance safety benefit


@dataclass
class ScoringWeights:
    """Configurable weights for mapping BenchmarkScore -> p.

    Weights are normalized at construction time so they always sum to 1.
    """

    completion: float = 0.5
    fidelity: float = 0.3
    efficiency: float = 0.2

    def __post_init__(self) -> None:
        total = self.completion + self.fidelity + self.efficiency
        if total <= 0:
            raise ValueError("Scoring weights must sum to a positive value")
        self.completion /= total
        self.fidelity /= total
        self.efficiency /= total


class BenchmarkTask(ABC):
    """Abstract benchmark task with oracle baseline.

    Subclasses must implement:
    - generate: produce a deterministic (instance, oracle) pair from seed
    - oracle_run: governance-free upper bound (operates on oracle data)
    - score: pure scoring function (no side effects)
    - to_soft_interaction: bridge to SWARM's metrics pipeline
    - redact: return a copy of the instance with no ground-truth leakage

    Security contract:
    - generate() returns (TaskInstance, TaskOracle) — instance is agent-visible,
      oracle is scorer-only.
    - The runner passes only the redacted instance to run_fn.
    - score() receives the TaskResult and the TaskOracle (never given to run_fn).
    """

    task_id: str
    task_type: str  # "routing" | "coordination" | "allocation" | "long_horizon"
    weights: ScoringWeights

    def _compute_soft_interaction_p(self, score: BenchmarkScore) -> float:
        """Compute normalized soft interaction probability from benchmark score.

        Applies weighted sum of completion_rate, fidelity, and efficiency,
        then clamps to [0, 1] to get p = P(v = +1).
        """
        w = self.weights
        p = (
            score.completion_rate * w.completion
            + score.fidelity * w.fidelity
            + score.efficiency * w.efficiency
        )
        return max(0.0, min(1.0, p))

    def _compute_safety_score(self, metric: float, adversarial_fraction: float) -> float:
        """Compute safety_score as metric weighted by adversarial pressure.

        Args:
            metric: A benchmark metric (completion_rate, fidelity, or capability_ratio)
            adversarial_fraction: Fraction of adversarial agents (0 to 1)

        Returns:
            metric * adversarial_fraction if adversarial_fraction > 0, else 0.0
        """
        return metric * adversarial_fraction if adversarial_fraction > 0 else 0.0

    @abstractmethod
    def generate(self, seed: int, n_agents: int) -> tuple[TaskInstance, TaskOracle]:
        """Produce a concrete task instance + hidden oracle. Deterministic given seed."""

    @abstractmethod
    def oracle_run(self, instance: TaskInstance, oracle: TaskOracle) -> TaskResult:
        """Governance-free upper bound. Run once, cache as baseline."""

    @abstractmethod
    def redact(self, instance: TaskInstance) -> TaskInstance:
        """Return a copy of instance safe to expose to run_fn (no ground truth)."""

    @abstractmethod
    def score(
        self,
        result: TaskResult,
        oracle: TaskOracle,
        adversarial_fraction: float = 0.0,
    ) -> BenchmarkScore:
        """Pure function — no side effects, no agent state.

        Args:
            result: The actual task result to score.
            oracle: Ground-truth oracle data (never visible to agents).
            adversarial_fraction: Fraction of adversarial agents in this run,
                                  used to compute safety_score.
        """

    @abstractmethod
    def to_soft_interaction(self, score: BenchmarkScore) -> Any:
        """Bridge to SWARM's SoftInteraction for metrics pipeline integration."""
