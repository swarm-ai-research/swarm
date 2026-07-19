"""Governance policy for AgentLaboratory workflows.

Enforces:
- Phase transition gates (min avg p to advance)
- Code execution circuit breaker (max consecutive failures)
- Cost budget enforcement (total LLM spend cap)
- Review loop limiter (max re-experimentation rounds with low scores)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

from swarm.bridges.agent_lab.config import AgentLabConfig
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Possible outcomes of a policy evaluation."""

    APPROVE = "approve"
    DENY = "deny"
    WARN = "warn"


@dataclass
class PolicyResult:
    """Result of a policy evaluation.

    Attributes:
        decision: APPROVE, DENY, or WARN outcome.
        reason: Human-readable explanation of the decision.
        governance_cost: Estimated cost of governance overhead (0.0-1.0) incurred
            by this policy decision. Used to track the total cost of governance
            enforcement across all policy decisions (phase gates, circuit breakers,
            cost limits, review loops). Higher values indicate costlier oversight:
            e.g., 0.1 for phase gate denial, 0.2 for circuit breaker trigger.
    """

    decision: PolicyDecision
    reason: str = ""
    governance_cost: float = 0.0


class AgentLabPolicy:
    """Governs AgentLab workflow behavior.

    Evaluates phase transitions, code execution safety, cost budgets,
    and review loop limits.
    """

    def __init__(self, config: AgentLabConfig | None = None) -> None:
        self._config = config or AgentLabConfig()
        self._consecutive_code_failures = 0
        self._total_cost_usd = 0.0
        self._low_review_rounds = 0

    def evaluate_phase_gate(
        self,
        phase_interactions: List[SoftInteraction],
    ) -> PolicyResult:
        """Evaluate whether a phase should be allowed to advance.

        Requires the average p across all phase interactions to meet
        the configured minimum.
        """
        if not phase_interactions:
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason="no interactions in phase (trivially passes)",
            )

        avg_p = sum(i.p for i in phase_interactions) / len(phase_interactions)
        threshold = self._config.phase_gate_min_p

        if avg_p < threshold:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"phase avg p={avg_p:.3f} below threshold {threshold}"
                ),
                governance_cost=0.1,
            )

        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason=f"phase avg p={avg_p:.3f} meets threshold {threshold}",
        )

    def evaluate_code_execution(self, success: bool) -> PolicyResult:
        """Evaluate code execution and track consecutive failures.

        Triggers circuit breaker after max consecutive failures.
        """
        if success:
            self._consecutive_code_failures = 0
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason="code execution succeeded",
            )

        self._consecutive_code_failures += 1
        max_failures = self._config.code_circuit_breaker_max_failures

        if self._consecutive_code_failures >= max_failures:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"code circuit breaker: {self._consecutive_code_failures} "
                    f"consecutive failures (limit: {max_failures})"
                ),
                governance_cost=0.2,
            )

        if self._consecutive_code_failures >= max_failures - 1:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=(
                    f"code failure {self._consecutive_code_failures}/{max_failures}"
                ),
                governance_cost=0.05,
            )

        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason=f"code failure {self._consecutive_code_failures}/{max_failures}",
        )

    def evaluate_cost(self, cost_delta_usd: float) -> PolicyResult:
        """Evaluate whether the cost budget allows continued operation."""
        self._total_cost_usd += cost_delta_usd
        budget = self._config.cost_budget_usd

        if self._total_cost_usd >= budget:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"cost budget exceeded: ${self._total_cost_usd:.2f} "
                    f">= ${budget:.2f}"
                ),
                governance_cost=0.1,
            )

        if self._total_cost_usd >= budget * 0.8:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=(
                    f"cost budget warning: ${self._total_cost_usd:.2f} "
                    f"/ ${budget:.2f} (80%+)"
                ),
                governance_cost=0.02,
            )

        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason=f"cost ${self._total_cost_usd:.2f} / ${budget:.2f}",
        )

    def evaluate_review_round(
        self,
        avg_review_score: float,
    ) -> PolicyResult:
        """Evaluate whether another re-experimentation round is warranted.

        Tracks consecutive low-scoring rounds and enforces the max.
        """
        threshold = self._config.review_score_threshold

        if avg_review_score < threshold:
            self._low_review_rounds += 1
        else:
            self._low_review_rounds = 0

        max_rounds = self._config.max_review_rounds
        if self._low_review_rounds >= max_rounds:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"review loop limit: {self._low_review_rounds} consecutive "
                    f"low-scoring rounds (limit: {max_rounds})"
                ),
                governance_cost=0.15,
            )

        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason=(
                f"review round ok (score={avg_review_score:.1f}, "
                f"low_rounds={self._low_review_rounds}/{max_rounds})"
            ),
        )

    def should_circuit_break(self) -> bool:
        """Check if any circuit breaker condition is active."""
        if self._consecutive_code_failures >= self._config.code_circuit_breaker_max_failures:
            return True
        if self._total_cost_usd >= self._config.cost_budget_usd:
            return True
        if self._low_review_rounds >= self._config.max_review_rounds:
            return True
        return False

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def consecutive_code_failures(self) -> int:
        return self._consecutive_code_failures

    def reset(self) -> None:
        """Reset all policy counters."""
        self._consecutive_code_failures = 0
        self._total_cost_usd = 0.0
        self._low_review_rounds = 0
