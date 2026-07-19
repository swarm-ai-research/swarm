"""Pseudo-verifiers for approximate task verification.

Based on Herbie Bradley's "Glimpses of AI Progress" (Pathways AI, 2025):

Rather than requiring exact verification (which limits RL to narrow domains),
pseudo-verifiers use heuristics, specialized models, and format specifications
to provide approximate reward signals. This enables training across broader
domains like legal writing, web navigation, and research tasks.

This module implements pseudo-verifiers that integrate with SWARM's proxy system.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class VerificationResult:
    """Result from a pseudo-verifier."""

    score: float  # 0-1 confidence that output is correct/good
    passed: bool  # Binary decision
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PseudoVerifier(ABC):
    """Base class for pseudo-verifiers.

    Pseudo-verifiers provide approximate verification signals where
    exact verification is impossible or impractical.
    """

    name: str = "base"
    weight: float = 1.0  # Weight when combining multiple verifiers

    @abstractmethod
    def verify(
        self, output: Any, context: dict[str, Any] | None = None
    ) -> VerificationResult:
        """Verify an output and return a result."""
        pass


class FormatVerifier(PseudoVerifier):
    """Verify output matches expected format specifications.

    The simplest pseudo-verifier: check structural properties
    without evaluating semantic correctness.
    """

    name = "format"

    def __init__(
        self,
        required_fields: list[str] | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
        patterns: list[str] | None = None,
    ):
        self.required_fields = required_fields or []
        self.max_length = max_length
        self.min_length = min_length
        self.patterns = [re.compile(p) for p in (patterns or [])]

    def verify(
        self, output: Any, context: dict[str, Any] | None = None
    ) -> VerificationResult:
        reasons = []
        score = 1.0

        # Check required fields for dict outputs
        if isinstance(output, dict) and self.required_fields:
            missing = [f for f in self.required_fields if f not in output]
            if missing:
                reasons.append(f"Missing fields: {missing}")
                score -= 0.2 * len(missing)

        # Check length for string outputs
        if isinstance(output, str):
            if self.max_length and len(output) > self.max_length:
                reasons.append(f"Output too long: {len(output)} > {self.max_length}")
                score -= 0.3
            if self.min_length and len(output) < self.min_length:
                reasons.append(f"Output too short: {len(output)} < {self.min_length}")
                score -= 0.3

            # Check regex patterns
            for pattern in self.patterns:
                if not pattern.search(output):
                    reasons.append(f"Pattern not matched: {pattern.pattern}")
                    score -= 0.1

        score = max(0.0, min(1.0, score))
        return VerificationResult(
            score=score,
            passed=score >= 0.5,
            reasons=reasons,
        )


class HeuristicVerifier(PseudoVerifier):
    """Verify using domain-specific heuristics.

    Custom heuristic functions that encode domain knowledge
    without requiring full semantic understanding.
    """

    name = "heuristic"

    def __init__(self, heuristics: list[Callable[[Any], tuple[float, str]]]):
        """
        Args:
            heuristics: List of functions that take output and return
                       (score_delta, reason) tuples.
        """
        self.heuristics = heuristics

    def verify(
        self, output: Any, context: dict[str, Any] | None = None
    ) -> VerificationResult:
        reasons = []
        score = 1.0

        for heuristic in self.heuristics:
            try:
                delta, reason = heuristic(output)
                score += delta
                if reason:
                    reasons.append(reason)
            except Exception as e:
                reasons.append(f"Heuristic error: {e}")
                score -= 0.1

        score = max(0.0, min(1.0, score))
        return VerificationResult(
            score=score,
            passed=score >= 0.5,
            reasons=reasons,
        )


class ConsistencyVerifier(PseudoVerifier):
    """Verify internal consistency of outputs.

    Check that outputs are self-consistent without verifying
    against ground truth.
    """

    name = "consistency"

    def __init__(self, consistency_checks: list[Callable[[Any], bool]] | None = None):
        self.checks = consistency_checks or []

    def verify(
        self, output: Any, context: dict[str, Any] | None = None
    ) -> VerificationResult:
        reasons = []
        passed_checks = 0

        for i, check in enumerate(self.checks):
            try:
                if check(output):
                    passed_checks += 1
                else:
                    reasons.append(f"Consistency check {i} failed")
            except Exception as e:
                reasons.append(f"Check {i} error: {e}")

        if not self.checks:
            score = 1.0
        else:
            score = passed_checks / len(self.checks)

        return VerificationResult(
            score=score,
            passed=score >= 0.5,
            reasons=reasons,
        )


class ModelBasedVerifier(PseudoVerifier):
    """Verify using a specialized model as judge.

    Uses another model to evaluate output quality.
    This is the "LLM-as-judge" pattern.
    """

    name = "model"

    def __init__(
        self,
        judge_fn: Callable[[Any, dict[str, Any] | None], float],
        threshold: float = 0.5,
    ):
        """
        Args:
            judge_fn: Function that takes (output, context) and returns score 0-1.
            threshold: Minimum score to pass.
        """
        self.judge_fn = judge_fn
        self.threshold = threshold

    def verify(
        self, output: Any, context: dict[str, Any] | None = None
    ) -> VerificationResult:
        try:
            score = self.judge_fn(output, context)
            score = max(0.0, min(1.0, float(score)))
            passed = score >= self.threshold
            reasons = (
                []
                if passed
                else [f"Model score {score:.2f} below threshold {self.threshold}"]
            )
        except Exception as e:
            score = 0.0
            passed = False
            reasons = [f"Model verification error: {e}"]

        return VerificationResult(
            score=score,
            passed=passed,
            reasons=reasons,
        )


class CompositeVerifier(PseudoVerifier):
    """Combine multiple pseudo-verifiers.

    Aggregates results from multiple verifiers using weighted combination.
    """

    name = "composite"

    def __init__(self, verifiers: list[PseudoVerifier]):
        self.verifiers = verifiers

    def verify(
        self, output: Any, context: dict[str, Any] | None = None
    ) -> VerificationResult:
        if not self.verifiers:
            return VerificationResult(score=1.0, passed=True)

        total_weight = sum(v.weight for v in self.verifiers)
        weighted_score = 0.0
        all_reasons = []
        all_passed = True

        for verifier in self.verifiers:
            result = verifier.verify(output, context)
            weighted_score += result.score * verifier.weight
            all_reasons.extend([f"[{verifier.name}] {r}" for r in result.reasons])
            if not result.passed:
                all_passed = False

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return VerificationResult(
            score=final_score,
            passed=all_passed and final_score >= 0.5,
            reasons=all_reasons,
            metadata={"verifier_count": len(self.verifiers)},
        )


# Domain-specific verifier factories


def create_code_verifier() -> CompositeVerifier:
    """Create verifier for code outputs."""

    def has_no_syntax_errors(code: str) -> tuple[float, str]:
        try:
            compile(code, "<string>", "exec")
            return (0.0, "")
        except SyntaxError as e:
            return (-0.5, f"Syntax error: {e}")

    def reasonable_length(code: str) -> tuple[float, str]:
        lines = code.strip().split("\n")
        if len(lines) < 2:
            return (-0.2, "Code too short")
        if len(lines) > 1000:
            return (-0.1, "Code unusually long")
        return (0.0, "")

    return CompositeVerifier(
        [
            FormatVerifier(min_length=10),
            HeuristicVerifier([has_no_syntax_errors, reasonable_length]),
        ]
    )


def create_research_verifier() -> CompositeVerifier:
    """Create verifier for research outputs."""

    def has_structure(text: str) -> tuple[float, str]:
        sections = ["abstract", "introduction", "method", "result", "conclusion"]
        found = sum(1 for s in sections if s.lower() in text.lower())
        if found >= 3:
            return (0.1, "")
        return (-0.2, f"Missing standard sections (found {found}/5)")

    def has_citations(text: str) -> tuple[float, str]:
        # Look for citation patterns
        if re.search(r"\[\d+\]|\(\d{4}\)|et al\.", text):
            return (0.1, "")
        return (-0.1, "No citations found")

    def reasonable_length(text: str) -> tuple[float, str]:
        words = len(text.split())
        if words < 500:
            return (-0.3, "Paper too short")
        if words > 50000:
            return (-0.1, "Paper unusually long")
        return (0.0, "")

    return CompositeVerifier(
        [
            FormatVerifier(min_length=1000),
            HeuristicVerifier([has_structure, has_citations, reasonable_length]),
        ]
    )


def create_task_verifier(task_type: str) -> PseudoVerifier:
    """Factory for task-specific verifiers.

    Args:
        task_type: Type of task ("code", "research", or unknown).

    Returns:
        A PseudoVerifier instance. For unknown task types, silently returns
        a basic FormatVerifier as a fallback.
    """
    verifiers = {
        "code": create_code_verifier,
        "research": create_research_verifier,
    }
    factory = verifiers.get(task_type, lambda: FormatVerifier())
    return factory()
