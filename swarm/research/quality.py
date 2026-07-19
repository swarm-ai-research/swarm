"""Quality gates and pre-registration for research workflows."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class GateStatus(Enum):
    """Status of a quality gate check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class GateCheck:
    """A single quality gate check result."""

    name: str
    status: GateStatus
    message: str = ""
    value: Any = None
    threshold: Any = None


@dataclass
class GateResult:
    """Result of running a quality gate."""

    gate_name: str
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def failed_checks(self) -> list[GateCheck]:
        """Get all failed checks."""
        return [c for c in self.checks if c.status == GateStatus.FAILED]

    @property
    def warnings(self) -> list[GateCheck]:
        """Get all warning checks."""
        return [c for c in self.checks if c.status == GateStatus.WARNING]

    def summary(self) -> str:
        """Generate a summary of the gate result."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Gate: {self.gate_name} - {status}"]
        for check in self.checks:
            symbol = {"passed": "✓", "failed": "✗", "warning": "⚠"}[check.status.value]
            lines.append(f"  {symbol} {check.name}: {check.message}")
        return "\n".join(lines)


class QualityGate:
    """A quality gate that must be passed before proceeding."""

    def __init__(self, name: str):
        self.name = name
        self._checks: list[tuple[str, Callable[..., Any], Any, str]] = []

    def add_check(
        self,
        name: str,
        check_fn: Callable[..., Any],
        threshold: Any = None,
        message: str = "",
    ) -> "QualityGate":
        """Add a check to the gate."""
        self._checks.append((name, check_fn, threshold, message))
        return self

    def run(self, data: Any) -> GateResult:
        """Run all checks against the data."""
        results = []
        all_passed = True

        for name, check_fn, threshold, message in self._checks:
            try:
                result = check_fn(data)
                if isinstance(result, bool):
                    passed = result
                    value = result
                elif isinstance(result, tuple):
                    passed, value = result
                else:
                    value = result
                    passed = (
                        value >= threshold if threshold is not None else bool(value)
                    )

                status = GateStatus.PASSED if passed else GateStatus.FAILED
                if not passed:
                    all_passed = False

                results.append(
                    GateCheck(
                        name=name,
                        status=status,
                        message=message or f"Value: {value}, Threshold: {threshold}",
                        value=value,
                        threshold=threshold,
                    )
                )
            except Exception as e:
                all_passed = False
                results.append(
                    GateCheck(
                        name=name,
                        status=GateStatus.FAILED,
                        message=f"Error: {e}",
                    )
                )

        return GateResult(gate_name=self.name, passed=all_passed, checks=results)


class QualityGates:
    """Collection of quality gates for research workflow phases."""

    @staticmethod
    def literature_gate() -> QualityGate:
        """Gate for literature review phase."""
        return (
            QualityGate("Literature Review")
            .add_check(
                "min_sources",
                lambda lit: len(lit.sources) >= 10,
                message="Minimum 10 sources required",
            )
            .add_check(
                "has_recent_papers",
                lambda lit: any(
                    (datetime.now(timezone.utc) - s.date).days < 180
                    for s in lit.sources
                    if hasattr(s, "date")
                ),
                message="Include papers from last 6 months",
            )
            .add_check(
                "gaps_identified",
                lambda lit: len(lit.gaps) >= 1,
                message="At least one research gap must be identified",
            )
            .add_check(
                "hypothesis_formed",
                lambda lit: bool(lit.hypothesis),
                message="A testable hypothesis must be formed",
            )
        )

    @staticmethod
    def experiment_gate() -> QualityGate:
        """Gate for experiment phase."""
        return (
            QualityGate("Experiment")
            .add_check(
                "min_trials",
                lambda exp: exp.trials_per_config >= 10,
                message="Minimum 10 trials per configuration",
            )
            .add_check(
                "seeds_documented",
                lambda exp: all(r.seed is not None for r in exp.results),
                message="All random seeds must be documented",
            )
            .add_check(
                "configs_complete",
                lambda exp: exp.parameter_coverage >= 0.8,
                message="At least 80% parameter coverage required",
            )
            .add_check(
                "no_errors",
                lambda exp: exp.error_count == 0,
                message="No simulation errors allowed",
            )
        )

    @staticmethod
    def analysis_gate() -> QualityGate:
        """Gate for analysis phase."""
        return (
            QualityGate("Analysis")
            .add_check(
                "ci_reported",
                lambda analysis: analysis.all_claims_have_ci,
                message="All claims must have confidence intervals",
            )
            .add_check(
                "effect_sizes",
                lambda analysis: analysis.all_claims_have_effect_size,
                message="All claims must have effect sizes",
            )
            .add_check(
                "corrections_applied",
                lambda analysis: analysis.multiple_comparison_corrected,
                message="Multiple comparison correction required",
            )
            .add_check(
                "limitations_stated",
                lambda analysis: len(analysis.limitations) >= 1,
                message="At least one limitation must be stated",
            )
        )

    @staticmethod
    def review_gate() -> QualityGate:
        """Gate for review phase."""
        return (
            QualityGate("Review")
            .add_check(
                "no_high_severity",
                lambda review: review.high_severity_count == 0,
                message="No high-severity issues allowed",
            )
            .add_check(
                "critiques_addressed",
                lambda review: review.all_critiques_addressed,
                message="All critiques must be addressed",
            )
            .add_check(
                "recommendation_positive",
                lambda review: review.recommendation in ["accept", "minor_revision"],
                message="Review recommendation must be accept or minor revision",
            )
        )

    @staticmethod
    def reflexivity_gate() -> QualityGate:
        """Gate for reflexivity checks."""
        return (
            QualityGate("Reflexivity")
            .add_check(
                "shadow_simulation_run",
                lambda reflex: reflex.shadow_simulation_complete,
                message="Shadow simulation must be run",
            )
            .add_check(
                "divergence_acceptable",
                lambda reflex: reflex.divergence < 0.3,
                message="Reflexivity divergence must be < 0.3",
            )
            .add_check(
                "disclosure_robustness_tested",
                lambda reflex: reflex.publish_then_attack_complete,
                message="Publish-then-attack protocol must be run",
            )
            .add_check(
                "robustness_classification",
                lambda reflex: reflex.robustness_classification is not None,
                message="Finding must be classified as disclosure-robust or conditionally-valid",
            )
        )


@dataclass
class PreRegistration:
    """Pre-registration of research hypothesis and methodology."""

    hypothesis: str
    secondary_hypotheses: list[str] = field(default_factory=list)
    methodology: dict[str, Any] = field(default_factory=dict)
    analysis_plan: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    registration_hash: str = ""

    def __post_init__(self):
        """Compute registration hash if not provided."""
        if not self.registration_hash:
            self.registration_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash of registration."""
        content = {
            "hypothesis": self.hypothesis,
            "secondary_hypotheses": sorted(self.secondary_hypotheses),
            "methodology": json.dumps(self.methodology, sort_keys=True),
            "analysis_plan": json.dumps(self.analysis_plan, sort_keys=True),
        }
        content_str = json.dumps(content, sort_keys=True)
        return f"sha256:{hashlib.sha256(content_str.encode()).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hypothesis": self.hypothesis,
            "secondary_hypotheses": self.secondary_hypotheses,
            "methodology": self.methodology,
            "analysis_plan": self.analysis_plan,
            "timestamp": self.timestamp.isoformat(),
            "registration_hash": self.registration_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreRegistration":
        """Deserialize from dictionary."""
        return cls(
            hypothesis=data["hypothesis"],
            secondary_hypotheses=data.get("secondary_hypotheses", []),
            methodology=data.get("methodology", {}),
            analysis_plan=data.get("analysis_plan", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(timezone.utc),
            registration_hash=data.get("registration_hash", ""),
        )

    def to_yaml(self) -> str:
        """Generate YAML representation for documentation."""
        lines = [
            "# Pre-Registration",
            f"timestamp: {self.timestamp.isoformat()}",
            f"hash: {self.registration_hash}",
            "",
            "hypothesis:",
            f"  primary: {self.hypothesis}",
        ]
        if self.secondary_hypotheses:
            lines.append("  secondary:")
            for h in self.secondary_hypotheses:
                lines.append(f"    - {h}")

        if self.methodology:
            lines.extend(["", "methodology:"])
            for key, value in self.methodology.items():
                lines.append(f"  {key}: {value}")

        if self.analysis_plan:
            lines.extend(["", "analysis_plan:"])
            for key, value in self.analysis_plan.items():
                if isinstance(value, list):
                    lines.append(f"  {key}:")
                    for item in value:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)


@dataclass
class Deviation:
    """A deviation from pre-registration."""

    field: str
    registered: Any
    actual: Any
    severity: str = "minor"  # minor, major, critical
    justification: str = ""


@dataclass
class RegistrationVerification:
    """Verification result comparing paper to pre-registration."""

    matches: bool
    deviations: list[Deviation] = field(default_factory=list)
    exploratory_analyses: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        """Generate verification summary."""
        status = "MATCHES" if self.matches else "DEVIATES"
        lines = [f"Pre-Registration Verification: {status}"]

        if self.deviations:
            lines.append("\nDeviations:")
            for d in self.deviations:
                lines.append(f"  [{d.severity.upper()}] {d.field}")
                lines.append(f"    Registered: {d.registered}")
                lines.append(f"    Actual: {d.actual}")
                if d.justification:
                    lines.append(f"    Justification: {d.justification}")

        if self.exploratory_analyses:
            lines.append("\nExploratory Analyses (not pre-registered):")
            for analysis in self.exploratory_analyses:
                lines.append(f"  - {analysis}")

        return "\n".join(lines)


def verify_against_registration(
    registration: PreRegistration,
    paper_hypothesis: str,
    paper_methodology: dict[str, Any],
    paper_analyses: list[str],
) -> RegistrationVerification:
    """Verify a paper against its pre-registration."""
    deviations = []

    # Check hypothesis
    if paper_hypothesis != registration.hypothesis:
        deviations.append(
            Deviation(
                field="hypothesis",
                registered=registration.hypothesis,
                actual=paper_hypothesis,
                severity="critical",
            )
        )

    # Check methodology
    for key, registered_value in registration.methodology.items():
        actual_value = paper_methodology.get(key)
        if actual_value != registered_value:
            deviations.append(
                Deviation(
                    field=f"methodology.{key}",
                    registered=registered_value,
                    actual=actual_value,
                    severity="major",
                )
            )

    # Identify exploratory analyses
    registered_analyses = set(
        registration.analysis_plan.get("primary", [])
        + registration.analysis_plan.get("secondary", [])
    )
    exploratory = [a for a in paper_analyses if a not in registered_analyses]

    return RegistrationVerification(
        matches=len(deviations) == 0,
        deviations=deviations,
        exploratory_analyses=exploratory,
    )
