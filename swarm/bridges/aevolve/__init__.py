"""SWARM ↔ A-Evolve bridge: SWARM scenarios as an evolution benchmark."""

from swarm.bridges.aevolve.adapter import (
    AEVOLVE_AVAILABLE,
    Feedback,
    SwarmBenchmarkAdapter,
    Task,
    Trajectory,
)
from swarm.bridges.aevolve.config import (
    DEFAULT_ALLOWED_OVERRIDES,
    AevolveBridgeConfig,
    EvaluationDetail,
)

__all__ = [
    "AEVOLVE_AVAILABLE",
    "AevolveBridgeConfig",
    "DEFAULT_ALLOWED_OVERRIDES",
    "EvaluationDetail",
    "Feedback",
    "SwarmBenchmarkAdapter",
    "Task",
    "Trajectory",
]
