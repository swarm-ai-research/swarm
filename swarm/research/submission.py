"""Pre-submission quality validation for research papers.

Ensures papers meet quality standards before submitting to clawxiv,
avoiding wasted submissions due to the 30-minute rate limit.

Usage:
    from swarm.research.submission import SubmissionValidator, submit_with_validation

    # Validate before submitting
    validator = SubmissionValidator()
    result = validator.validate(paper)
    if result.passed:
        client.submit(paper)
    else:
        print(result.report())

    # Or use the all-in-one function
    submit_with_validation(client, paper, dry_run=True)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .platforms import Paper, PlatformClient, SubmissionResult

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Issue severity levels."""

    ERROR = "error"  # Blocks submission
    WARNING = "warning"  # Suggests improvement
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: Severity
    code: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of paper validation."""

    issues: list[ValidationIssue] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if no errors (warnings are allowed)."""
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def quality_score(self) -> float:
        """Overall quality score 0-100."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores) * 100

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 60)
        lines.append("SUBMISSION VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Quality scores
        lines.append("Quality Scores:")
        for name, score in sorted(self.scores.items()):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {name:20} [{bar}] {score * 100:5.1f}%")
        lines.append(f"  {'OVERALL':20} [{self.quality_score:5.1f}%]")
        lines.append("")

        # Issues by severity
        if self.errors():
            lines.append("ERRORS (must fix before submission):")
            for issue in self.errors():
                lines.append(f"  ✗ [{issue.code}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")
            lines.append("")

        if self.warnings():
            lines.append("WARNINGS (recommended fixes):")
            for issue in self.warnings():
                lines.append(f"  ⚠ [{issue.code}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")
            lines.append("")

        # Verdict
        if self.passed:
            lines.append(
                f"✓ PASSED - Ready for submission (score: {self.quality_score:.1f}%)"
            )
        else:
            lines.append(f"✗ FAILED - {len(self.errors())} error(s) must be fixed")

        return "\n".join(lines)


class SubmissionValidator:
    """Validates papers before submission."""

    # Minimum thresholds
    MIN_SOURCE_LENGTH = 3000  # chars
    MIN_ABSTRACT_LENGTH = 200  # chars
    MIN_SECTIONS = 4  # Introduction, Methods, Results, Conclusion

    # Required LaTeX sections (regex patterns)
    # Note: Accepts common variations (Methods/Experiments, Results/Evaluation)
    REQUIRED_SECTIONS = [
        (r"\\section\{[Ii]ntroduction\}", "Introduction"),
        (r"\\section\{[Mm]ethods?|[Ee]xperiments?\}", "Methods/Experiments"),
        (r"\\section\{[Rr]esults?|[Ee]valuation\}", "Results"),
        (r"\\section\{[Cc]onclusion", "Conclusion"),
    ]

    # Recommended sections
    RECOMMENDED_SECTIONS = [
        (r"\\section\{[Dd]iscussion\}", "Discussion"),
        (r"\\section\{[Rr]elated [Ww]ork\}", "Related Work"),
        (r"\\begin\{abstract\}", "Abstract environment"),
    ]

    # Quality indicators
    QUALITY_INDICATORS = [
        (r"\\begin\{equation\}", "equations"),
        (r"\\begin\{table\}", "tables"),
        (r"\\begin\{figure\}", "figures"),
        (r"\\cite\{", "citations"),
        (r"\\textbf\{H\d", "hypotheses"),
        (r"\d+\.\d+%", "percentages"),
        (r"p\s*[<>=]\s*0\.\d+", "p-values"),
        (r"95%\s*CI", "confidence intervals"),
        (r"Cohen", "effect sizes"),
    ]

    def validate(self, paper: Paper) -> ValidationResult:
        """Validate a paper and return detailed results."""
        result = ValidationResult()

        # Run all checks
        self._check_basic_fields(paper, result)
        self._check_source_length(paper, result)
        self._check_abstract_length(paper, result)
        self._check_required_sections(paper, result)
        self._check_recommended_sections(paper, result)
        self._check_quality_indicators(paper, result)
        self._check_latex_structure(paper, result)
        self._compute_scores(paper, result)

        return result

    def _check_basic_fields(self, paper: Paper, result: ValidationResult) -> None:
        """Check required fields exist."""
        if not paper.title or len(paper.title) < 10:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_TITLE",
                    "Paper must have a title (>10 chars)",
                )
            )

        if not paper.abstract:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_ABSTRACT",
                    "Paper must have an abstract",
                )
            )

        if not paper.source:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_SOURCE",
                    "Paper must have LaTeX source",
                )
            )

        if not paper.categories:
            result.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "MISSING_CATEGORIES",
                    "Paper should have categories",
                    "Add categories like ['cs.MA', 'cs.AI']",
                )
            )

    def _check_source_length(self, paper: Paper, result: ValidationResult) -> None:
        """Check source meets minimum length."""
        if not paper.source:
            return

        length = len(paper.source)
        if length < self.MIN_SOURCE_LENGTH:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "SOURCE_TOO_SHORT",
                    f"Source is {length} chars, minimum is {self.MIN_SOURCE_LENGTH}",
                    "Expand with methodology, results, and discussion sections",
                )
            )
        elif length < self.MIN_SOURCE_LENGTH * 2:
            result.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "SOURCE_SHORT",
                    f"Source is {length} chars, consider expanding",
                    "A full paper typically has 6000+ chars",
                )
            )

    def _check_abstract_length(self, paper: Paper, result: ValidationResult) -> None:
        """Check abstract meets minimum length."""
        if not paper.abstract:
            return

        length = len(paper.abstract)
        if length < self.MIN_ABSTRACT_LENGTH:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "ABSTRACT_TOO_SHORT",
                    f"Abstract is {length} chars, minimum is {self.MIN_ABSTRACT_LENGTH}",
                    "Expand to summarize motivation, methods, and key findings",
                )
            )

    def _check_required_sections(self, paper: Paper, result: ValidationResult) -> None:
        """Check for required sections."""
        if not paper.source:
            return

        missing = []
        for pattern, name in self.REQUIRED_SECTIONS:
            if not re.search(pattern, paper.source):
                missing.append(name)

        if missing:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_SECTIONS",
                    f"Missing required sections: {', '.join(missing)}",
                    "Add \\section{} for each missing section",
                )
            )

    def _check_recommended_sections(
        self, paper: Paper, result: ValidationResult
    ) -> None:
        """Check for recommended sections."""
        if not paper.source:
            return

        missing = []
        for pattern, name in self.RECOMMENDED_SECTIONS:
            if not re.search(pattern, paper.source):
                missing.append(name)

        if missing:
            result.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "MISSING_RECOMMENDED",
                    f"Consider adding: {', '.join(missing)}",
                )
            )

    def _check_quality_indicators(self, paper: Paper, result: ValidationResult) -> None:
        """Check for quality indicators."""
        if not paper.source:
            return

        found = []
        missing = []

        for pattern, name in self.QUALITY_INDICATORS:
            if re.search(pattern, paper.source):
                found.append(name)
            else:
                missing.append(name)

        if len(found) < 3:
            result.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "LOW_QUALITY_INDICATORS",
                    f"Paper has few quality indicators ({len(found)}/{len(self.QUALITY_INDICATORS)})",
                    f"Consider adding: {', '.join(missing[:3])}",
                )
            )

    def _check_latex_structure(self, paper: Paper, result: ValidationResult) -> None:
        """Check LaTeX structure is valid."""
        if not paper.source:
            return

        # Check for documentclass
        if not re.search(r"\\documentclass", paper.source):
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_DOCUMENTCLASS",
                    "LaTeX source must include \\documentclass",
                )
            )

        # Check for begin/end document
        if not re.search(r"\\begin\{document\}", paper.source):
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_BEGIN_DOCUMENT",
                    "LaTeX source must include \\begin{document}",
                )
            )

        if not re.search(r"\\end\{document\}", paper.source):
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_END_DOCUMENT",
                    "LaTeX source must include \\end{document}",
                )
            )

        # Check for unbalanced braces (simple check)
        open_braces = paper.source.count("{")
        close_braces = paper.source.count("}")
        if open_braces != close_braces:
            result.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "UNBALANCED_BRACES",
                    f"Possible unbalanced braces: {open_braces} open, {close_braces} close",
                )
            )

    def _compute_scores(self, paper: Paper, result: ValidationResult) -> None:
        """Compute quality scores."""
        if not paper.source:
            return

        # Length score (0-1, maxes out at 10000 chars)
        length_score = min(len(paper.source) / 10000, 1.0)
        result.scores["length"] = length_score

        # Structure score (required sections present)
        sections_found = sum(
            1
            for pattern, _ in self.REQUIRED_SECTIONS
            if re.search(pattern, paper.source)
        )
        result.scores["structure"] = sections_found / len(self.REQUIRED_SECTIONS)

        # Quality indicators score
        indicators_found = sum(
            1
            for pattern, _ in self.QUALITY_INDICATORS
            if re.search(pattern, paper.source)
        )
        result.scores["rigor"] = min(indicators_found / 5, 1.0)  # 5 indicators = 100%

        # Abstract score
        if paper.abstract:
            abstract_score = min(len(paper.abstract) / 500, 1.0)
            result.scores["abstract"] = abstract_score


def submit_with_validation(
    client: PlatformClient,
    paper: Paper,
    dry_run: bool = False,
    min_score: float = 60.0,
    require_no_errors: bool = True,
    platform: str = "clawxiv",
) -> tuple[bool, ValidationResult, Optional[SubmissionResult]]:
    """Submit a paper with pre-validation.

    Args:
        client: ClawxivClient or AgentxivClient instance
        paper: Paper to submit
        dry_run: If True, validate only without submitting
        min_score: Minimum quality score required (0-100)
        require_no_errors: If True, block on any validation errors
        platform: 'clawxiv' (LaTeX) or 'agentxiv' (Markdown)

    Returns:
        Tuple of (success, validation_result, submission_result)
    """
    validator = get_validator(platform)
    validation = validator.validate(paper)

    logger.info(validation.report())

    # Check gates
    if require_no_errors and not validation.passed:
        logger.error("Submission blocked: validation errors must be fixed")
        return False, validation, None

    if validation.quality_score < min_score:
        logger.error(
            "Submission blocked: quality score %.1f%% < %.1f%%",
            validation.quality_score,
            min_score,
        )
        return False, validation, None

    if dry_run:
        logger.info("DRY RUN - would submit paper (use dry_run=False to submit)")
        return True, validation, None

    # Actually submit
    logger.info("Submitting paper...")
    result = client.submit(paper)

    if result.success:
        logger.info("Submitted successfully: %s", result.paper_id)
    else:
        logger.error("Submission failed: %s", result.message)

    return result.success, validation, result


class AgentxivValidator:
    """Validates papers for agentxiv (Markdown format)."""

    MIN_CONTENT_LENGTH = 2000  # chars
    MIN_ABSTRACT_LENGTH = 200  # chars

    # Required Markdown sections
    REQUIRED_SECTIONS = [
        (r"^#+\s*Introduction", "Introduction"),
        (r"^#+\s*Method", "Methods"),
        (r"^#+\s*Result", "Results"),
        (r"^#+\s*Conclusion", "Conclusion"),
    ]

    # Quality indicators for Markdown
    QUALITY_INDICATORS = [
        (r"\$[^$]+\$", "equations"),  # Inline math
        (r"\|.*\|.*\|", "tables"),
        (r"!\[.*\]\(.*\)", "images"),
        (r"\[@[^\]]+\]|\[arxiv:", "citations"),
        (r"\*\*H\d", "hypotheses"),
        (r"\d+\.\d+%", "percentages"),
        (r"p\s*[<>=]\s*0\.\d+", "p-values"),
        (r"95%\s*CI", "confidence intervals"),
    ]

    def validate(self, paper: Paper) -> ValidationResult:
        """Validate a paper for agentxiv submission."""
        result = ValidationResult()

        self._check_basic_fields(paper, result)
        self._check_content_length(paper, result)
        self._check_abstract_length(paper, result)
        self._check_required_sections(paper, result)
        self._check_quality_indicators(paper, result)
        self._check_markdown_structure(paper, result)
        self._compute_scores(paper, result)

        return result

    def _check_basic_fields(self, paper: Paper, result: ValidationResult) -> None:
        """Check required fields exist."""
        if not paper.title or len(paper.title) < 10:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_TITLE",
                    "Paper must have a title (>10 chars)",
                )
            )

        if not paper.abstract:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_ABSTRACT",
                    "Paper must have an abstract",
                )
            )

        if not paper.source:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_CONTENT",
                    "Paper must have Markdown content",
                )
            )

        if not paper.categories:
            result.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "MISSING_CATEGORY",
                    "Paper should have a category",
                    "Add category like 'multi-agent' or 'alignment'",
                )
            )

    def _check_content_length(self, paper: Paper, result: ValidationResult) -> None:
        """Check content meets minimum length."""
        if not paper.source:
            return

        length = len(paper.source)
        if length < self.MIN_CONTENT_LENGTH:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "CONTENT_TOO_SHORT",
                    f"Content is {length} chars, minimum is {self.MIN_CONTENT_LENGTH}",
                    "Expand with methodology, results, and discussion",
                )
            )

    def _check_abstract_length(self, paper: Paper, result: ValidationResult) -> None:
        """Check abstract meets minimum length."""
        if not paper.abstract:
            return

        length = len(paper.abstract)
        if length < self.MIN_ABSTRACT_LENGTH:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "ABSTRACT_TOO_SHORT",
                    f"Abstract is {length} chars, minimum is {self.MIN_ABSTRACT_LENGTH}",
                )
            )

    def _check_required_sections(self, paper: Paper, result: ValidationResult) -> None:
        """Check for required Markdown sections."""
        if not paper.source:
            return

        missing = []
        for pattern, name in self.REQUIRED_SECTIONS:
            if not re.search(pattern, paper.source, re.MULTILINE | re.IGNORECASE):
                missing.append(name)

        if missing:
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "MISSING_SECTIONS",
                    f"Missing required sections: {', '.join(missing)}",
                    "Add ## Section headers for each missing section",
                )
            )

    def _check_quality_indicators(self, paper: Paper, result: ValidationResult) -> None:
        """Check for quality indicators."""
        if not paper.source:
            return

        found = []
        missing = []

        for pattern, name in self.QUALITY_INDICATORS:
            if re.search(pattern, paper.source):
                found.append(name)
            else:
                missing.append(name)

        if len(found) < 2:
            result.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "LOW_QUALITY_INDICATORS",
                    f"Paper has few quality indicators ({len(found)}/{len(self.QUALITY_INDICATORS)})",
                    f"Consider adding: {', '.join(missing[:3])}",
                )
            )

    def _check_markdown_structure(self, paper: Paper, result: ValidationResult) -> None:
        """Check Markdown structure is valid."""
        if not paper.source:
            return

        # Check for headers
        if not re.search(r"^#+\s", paper.source, re.MULTILINE):
            result.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "NO_HEADERS",
                    "Content must have Markdown headers (# or ##)",
                )
            )

    def _compute_scores(self, paper: Paper, result: ValidationResult) -> None:
        """Compute quality scores."""
        if not paper.source:
            return

        # Length score
        length_score = min(len(paper.source) / 5000, 1.0)
        result.scores["length"] = length_score

        # Structure score
        sections_found = sum(
            1
            for pattern, _ in self.REQUIRED_SECTIONS
            if re.search(pattern, paper.source, re.MULTILINE | re.IGNORECASE)
        )
        result.scores["structure"] = sections_found / len(self.REQUIRED_SECTIONS)

        # Quality indicators score
        indicators_found = sum(
            1
            for pattern, _ in self.QUALITY_INDICATORS
            if re.search(pattern, paper.source)
        )
        result.scores["rigor"] = min(indicators_found / 4, 1.0)

        # Abstract score
        if paper.abstract:
            result.scores["abstract"] = min(len(paper.abstract) / 400, 1.0)


def get_validator(platform: str = "clawxiv") -> SubmissionValidator | AgentxivValidator:
    """Get the appropriate validator for a platform.

    Args:
        platform: 'clawxiv' (LaTeX) or 'agentxiv' (Markdown)

    Returns:
        Validator instance for the platform
    """
    if platform.lower() == "agentxiv":
        return AgentxivValidator()
    return SubmissionValidator()


def update_with_validation(
    client: PlatformClient,
    paper_id: str,
    paper: Paper,
    dry_run: bool = False,
    min_score: float = 60.0,
    platform: str = "clawxiv",
) -> tuple[bool, ValidationResult, Optional[SubmissionResult]]:
    """Update a paper with pre-validation.

    Same as submit_with_validation but for updates.
    """
    validator = get_validator(platform)
    validation = validator.validate(paper)

    logger.info(validation.report())

    if not validation.passed:
        logger.error("Update blocked: validation errors must be fixed")
        return False, validation, None

    if validation.quality_score < min_score:
        logger.error(
            "Update blocked: quality score %.1f%% < %.1f%%",
            validation.quality_score,
            min_score,
        )
        return False, validation, None

    if dry_run:
        logger.info("DRY RUN - would update %s (use dry_run=False to update)", paper_id)
        return True, validation, None

    logger.info("Updating %s...", paper_id)
    result = client.update(paper_id, paper)

    if result.success:
        logger.info("Updated successfully: version %s", result.version)
    else:
        logger.error("Update failed: %s", result.message)

    return result.success, validation, result
