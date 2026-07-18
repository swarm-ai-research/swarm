"""Entities for the A-Evolve evolution-loop domain.

The domain's unit of analysis is the *evolution loop itself*: candidates
proposed by the engine, evaluated against the benchmark, and gated
(accepted → ``evo-N`` tag, or rolled back). Mapping that loop into SWARM's
soft-label frame lets the standard metrics ask a question A-Evolve's own
tooling doesn't: is the gate adversely selecting — accepting mutations that
score well on the benchmark while degrading true quality?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class GateDecision(str, Enum):
    ACCEPTED = "accepted"
    ROLLED_BACK = "rolled_back"
    PENDING = "pending"


@dataclass
class CandidateOutcome:
    """One evolution candidate's trip through solve → evaluate → gate."""

    candidate_id: str
    trial: int = 0
    benchmark_score: float = 0.0  # the score the gate saw, in [0, 1]
    holdout_score: float | None = None  # disjoint-split score, if measured
    gate: GateDecision = GateDecision.PENDING
    mutation_kind: str = ""  # e.g. "prompt", "skill", "tool", "memory"
    details: Dict[str, Any] = field(default_factory=dict)
