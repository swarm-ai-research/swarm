"""Verifier role for output validation."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from swarm.agents.base import Action, ActionType, Observation, Role


@dataclass
class VerificationResult:
    """Result of a verification check."""

    output_id: str = ""
    task_id: str = ""
    is_approved: bool = False
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class VerifierRole:
    """
    Role mixin for output verification capabilities.

    Verifiers can:
    - Review submitted outputs
    - Assess quality
    - Identify issues
    - Approve or reject submissions
    """

    def __init__(self) -> None:
        """Initialize verifier role."""
        self.role = Role.VERIFIER
        self._verification_history: List[VerificationResult] = []
        self._verifier_config = {
            "quality_threshold": 0.6,
            "strictness": 0.5,  # 0 = lenient, 1 = strict
            "max_verifications_per_step": 3,
        }
        self._pending_verifications: List[Dict] = []

    def can_verify(self) -> bool:
        """Check if verifier can take on verification tasks."""
        return bool(
            len(self._pending_verifications)
            < self._verifier_config["max_verifications_per_step"]
        )

    def queue_verification(
        self,
        output_id: str,
        task_id: str,
        content: str,
        submitter_id: str,
    ) -> None:
        """Queue an output for verification."""
        self._pending_verifications.append(
            {
                "output_id": output_id,
                "task_id": task_id,
                "content": content,
                "submitter_id": submitter_id,
            }
        )

    def verify_output(
        self,
        output_id: str,
        content: str,
        task_requirements: List[str],
        task_id: str = "",
    ) -> VerificationResult:
        """
        Verify an output against requirements.

        Args:
            output_id: ID of the output
            content: Content to verify
            task_requirements: List of requirements to check
            task_id: ID of the task (optional)

        Returns:
            VerificationResult with approval status and feedback
        """
        issues = []
        suggestions = []

        # Check content length
        if len(content) < 10:
            issues.append("Output is too short")
            suggestions.append("Provide more detailed output")

        # Check for placeholder content
        placeholder_indicators = ["[", "]", "TODO", "FIXME", "placeholder"]
        for indicator in placeholder_indicators:
            if indicator.lower() in content.lower():
                issues.append(f"Contains placeholder content: {indicator}")

        # Check requirements coverage (simplified)
        requirements_met = 0
        for req in task_requirements:
            # Simple keyword matching
            if any(word.lower() in content.lower() for word in req.split()):
                requirements_met += 1

        if task_requirements:
            coverage = requirements_met / len(task_requirements)
        else:
            coverage = 0.5  # Neutral if no explicit requirements

        if coverage < 0.5:
            issues.append("Not all requirements addressed")
            suggestions.append("Review and address all required outputs")

        # Calculate quality score
        quality_score = self._calculate_quality_score(content, issues, coverage)

        # Apply strictness
        adjusted_threshold = self._verifier_config["quality_threshold"] * (
            1 + self._verifier_config["strictness"] * 0.5
        )

        is_approved = quality_score >= adjusted_threshold and len(issues) == 0

        result = VerificationResult(
            output_id=output_id,
            task_id=task_id,
            is_approved=is_approved,
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
        )

        self._verification_history.append(result)
        return result

    def _calculate_quality_score(
        self,
        content: str,
        issues: List[str],
        requirements_coverage: float,
    ) -> float:
        """Calculate quality score for output."""
        # Base score from content characteristics
        base_score = 0.5

        # Length contribution
        length_score = min(1.0, len(content) / 500) * 0.2

        # Requirements coverage
        coverage_score = requirements_coverage * 0.4

        # Penalty for issues
        issue_penalty = len(issues) * 0.1

        # Add some variance
        variance = random.uniform(-0.1, 0.1)

        quality = base_score + length_score + coverage_score - issue_penalty + variance
        return max(0.0, min(1.0, quality))

    def get_approval_rate(self) -> float:
        """Get historical approval rate."""
        if not self._verification_history:
            return 0.0

        approved = sum(1 for r in self._verification_history if r.is_approved)
        return approved / len(self._verification_history)

    def get_average_quality(self) -> float:
        """Get average quality score of verified outputs."""
        if not self._verification_history:
            return 0.0

        return sum(r.quality_score for r in self._verification_history) / len(
            self._verification_history
        )

    def process_pending(self) -> List[VerificationResult]:
        """Process all pending verifications."""
        results = []

        for pending in self._pending_verifications:
            result = self.verify_output(
                output_id=pending["output_id"],
                content=pending["content"],
                task_requirements=[],  # Would come from task
                task_id=pending["task_id"],
            )
            results.append(result)

        self._pending_verifications.clear()
        return results

    def decide_verification_action(self, observation: Observation) -> Optional[Action]:
        """
        Decide on a verification-related action.

        Returns:
            Action if verification action needed, None otherwise
        """
        # Check for outputs to verify
        if self._pending_verifications:
            pending = self._pending_verifications[0]

            result = self.verify_output(
                output_id=pending["output_id"],
                content=pending["content"],
                task_requirements=[],
                task_id=pending.get("task_id", ""),
            )

            self._pending_verifications.pop(0)

            return Action(
                action_type=ActionType.VERIFY_OUTPUT,
                agent_id="",  # To be filled by caller
                target_id=pending["output_id"],
                metadata={
                    "is_approved": result.is_approved,
                    "quality_score": result.quality_score,
                    "issues": result.issues,
                },
            )

        return None

    def set_strictness(self, strictness: float) -> None:
        """Set verification strictness (0-1)."""
        self._verifier_config["strictness"] = max(0.0, min(1.0, strictness))
