"""Multi-agent review layer for agent-authored git changes.

Instead of one end-of-task review, a panel of specialized reviewers each
examines the same diff from its own angle and emits structured findings. The
panel output is then *synthesized*: humans see where reviewers disagree,
which findings are policy-blocking, and which spots are high-risk or
low-confidence — not every line every reviewer looked at.

The v1 panel ships the boring-valuable reviewers:

- :class:`TestCoverageReviewer` — source changes without test changes,
  deleted tests.
- :class:`DependencyReviewer` — manifest/lockfile drift, lockfile-only edits.
- :class:`SecurityReviewer` — sensitive paths, secret-looking added lines,
  risky calls introduced by the diff.

Reviewers are deterministic (pure functions of the diff), so the same change
always yields the same panel verdicts — reviews recorded into a provenance
bundle are reproducible facts, not one-off opinions.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from swarm.agentgit.git import GitSnapshot, collect_snapshot
from swarm.agentgit.policy import DEPENDENCY_FILENAMES
from swarm.agentgit.store import _git

# Verdict ordinals: a reviewer's overall verdict is the worst of its findings.
VERDICT_APPROVE = "approve"
VERDICT_WARN = "warn"
VERDICT_BLOCK = "block"
_SEVERITY_TO_VERDICT = {"info": VERDICT_APPROVE, "warning": VERDICT_WARN, "blocking": VERDICT_BLOCK}
_VERDICT_RANK = {VERDICT_APPROVE: 0, VERDICT_WARN: 1, VERDICT_BLOCK: 2}

# Findings below this confidence are surfaced in the attention list even when
# non-blocking: a reviewer that is unsure is exactly where a human should look.
LOW_CONFIDENCE = 0.5


@dataclass(frozen=True)
class ReviewFinding:
    """One structured finding from one reviewer."""

    reviewer: str
    rule: str
    severity: str  # info | warning | blocking
    confidence: float  # [0, 1]
    paths: Tuple[str, ...]
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITY_TO_VERDICT:
            raise ValueError(f"invalid severity {self.severity!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        object.__setattr__(self, "paths", tuple(self.paths))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "rule": self.rule,
            "severity": self.severity,
            "confidence": self.confidence,
            "paths": list(self.paths),
            "summary": self.summary,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ReviewContext:
    """Everything a reviewer may inspect: the snapshot plus added-line text.

    ``added_lines`` maps changed path -> lines added by the diff (full content
    for untracked files), so content-based reviewers see exactly what the
    change introduces — never pre-existing code.
    """

    snapshot: GitSnapshot
    added_lines: Dict[str, List[str]]


class Reviewer(ABC):
    """A specialized review agent producing structured findings for one diff."""

    name: str = "reviewer"

    @abstractmethod
    def review(self, ctx: ReviewContext) -> List[ReviewFinding]:
        """Return findings for the change (empty list = clean approve)."""

    def _finding(self, **kwargs: Any) -> ReviewFinding:
        return ReviewFinding(reviewer=self.name, **kwargs)


@dataclass(frozen=True)
class ReviewerReport:
    """One reviewer's verdict plus the findings that produced it."""

    reviewer: str
    verdict: str
    findings: Tuple[ReviewFinding, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "decision": self.verdict,
            "findings": [f.to_dict() for f in self.findings],
        }


def verdict_for(findings: Sequence[ReviewFinding]) -> str:
    """A reviewer's overall verdict is the worst severity among its findings."""

    worst = VERDICT_APPROVE
    for finding in findings:
        candidate = _SEVERITY_TO_VERDICT[finding.severity]
        if _VERDICT_RANK[candidate] > _VERDICT_RANK[worst]:
            worst = candidate
    return worst


# ---------------------------------------------------------------------------
# built-in reviewers
# ---------------------------------------------------------------------------
class TestCoverageReviewer(Reviewer):
    """Flags source changes that arrive without any test changes."""

    __test__ = False  # keep pytest from collecting this as a test class
    name = "test-coverage"

    def __init__(
        self,
        source_suffixes: Sequence[str] = (".py",),
        test_markers: Sequence[str] = ("tests/", "test_"),
    ) -> None:
        self.source_suffixes = tuple(source_suffixes)
        self.test_markers = tuple(test_markers)

    def _is_test_path(self, path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        return any(path.startswith(m) or f"/{m}" in path for m in self.test_markers) or (
            name.startswith("test_")
        )

    def review(self, ctx: ReviewContext) -> List[ReviewFinding]:
        findings: List[ReviewFinding] = []
        source_changes = [
            f
            for f in ctx.snapshot.changed_files
            if f.path.endswith(self.source_suffixes) and not self._is_test_path(f.path)
        ]
        test_changes = [
            f for f in ctx.snapshot.changed_files if self._is_test_path(f.path)
        ]

        deleted_tests = [f.path for f in test_changes if f.status.startswith("D")]
        if deleted_tests:
            findings.append(
                self._finding(
                    rule="test-deletion",
                    severity="blocking",
                    confidence=0.9,
                    paths=tuple(deleted_tests),
                    summary=f"{len(deleted_tests)} test file(s) deleted",
                    details={"deleted": deleted_tests},
                )
            )

        if source_changes and not test_changes:
            added = sum(f.additions for f in source_changes)
            findings.append(
                self._finding(
                    rule="untested-change",
                    severity="warning",
                    # Larger untested changes are more confidently a problem; a
                    # 1-line tweak may legitimately need no new test.
                    confidence=min(0.9, 0.3 + added / 200.0),
                    paths=tuple(f.path for f in source_changes),
                    summary=(
                        f"{len(source_changes)} source file(s) changed "
                        f"(+{added} lines) with no test changes"
                    ),
                    details={"added_lines": added},
                )
            )
        return findings


class DependencyReviewer(Reviewer):
    """Flags dependency manifest/lockfile changes and drift between them."""

    name = "dependency"

    _LOCKFILES = {
        "poetry.lock",
        "Pipfile.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Cargo.lock",
        "go.sum",
        "Gemfile.lock",
    }

    def review(self, ctx: ReviewContext) -> List[ReviewFinding]:
        findings: List[ReviewFinding] = []
        dep_paths = [
            f.path
            for f in ctx.snapshot.changed_files
            if f.path.rsplit("/", 1)[-1] in DEPENDENCY_FILENAMES
        ]
        if not dep_paths:
            return findings

        lock_paths = [p for p in dep_paths if p.rsplit("/", 1)[-1] in self._LOCKFILES]
        manifest_paths = [p for p in dep_paths if p not in lock_paths]

        findings.append(
            self._finding(
                rule="dependency-change",
                severity="warning",
                confidence=0.95,
                paths=tuple(dep_paths),
                summary=f"{len(dep_paths)} dependency file(s) changed",
                details={"manifests": manifest_paths, "lockfiles": lock_paths},
            )
        )
        if lock_paths and not manifest_paths:
            # A lockfile that moves with no manifest change is the classic
            # supply-chain smell: the dependency *graph* changed while the
            # declared intent did not.
            findings.append(
                self._finding(
                    rule="lockfile-only-change",
                    severity="blocking",
                    confidence=0.7,
                    paths=tuple(lock_paths),
                    summary="Lockfile changed without a manifest change",
                    details={"lockfiles": lock_paths},
                )
            )
        return findings


class SecurityReviewer(Reviewer):
    """Flags sensitive paths and secret-looking or risky added lines."""

    name = "security"

    _SENSITIVE_PATTERNS = (
        ".env",
        "*.pem",
        "*.key",
        "*_rsa",
        "id_rsa*",
        ".github/workflows/*",
        "*credentials*",
        "*secret*",
    )
    _SECRET_RES: Tuple[Tuple[str, re.Pattern[str], float], ...] = (
        ("private-key-material", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), 0.95),
        ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0.9),
        (
            "hardcoded-credential",
            re.compile(
                r"(?i)\b(api_?key|secret|token|password|passwd)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
            ),
            0.5,
        ),
    )
    _RISKY_RES: Tuple[Tuple[str, re.Pattern[str], float], ...] = (
        ("shell-true", re.compile(r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True"), 0.6),
        ("eval-exec", re.compile(r"\b(eval|exec)\s*\("), 0.4),
        ("pickle-load", re.compile(r"\bpickle\.loads?\s*\("), 0.5),
        ("os-system", re.compile(r"\bos\.system\s*\("), 0.6),
    )

    def review(self, ctx: ReviewContext) -> List[ReviewFinding]:
        from fnmatch import fnmatch

        findings: List[ReviewFinding] = []
        sensitive = [
            f.path
            for f in ctx.snapshot.changed_files
            if any(
                fnmatch(f.path, pat) or fnmatch(f.path.rsplit("/", 1)[-1], pat)
                for pat in self._SENSITIVE_PATTERNS
            )
        ]
        if sensitive:
            findings.append(
                self._finding(
                    rule="sensitive-path",
                    severity="blocking",
                    confidence=0.8,
                    paths=tuple(sensitive),
                    summary=f"{len(sensitive)} sensitive path(s) touched",
                    details={"paths": sensitive},
                )
            )

        for rules, severity in ((self._SECRET_RES, "blocking"), (self._RISKY_RES, "warning")):
            for rule, pattern, confidence in rules:
                hits: List[Dict[str, Any]] = []
                for path, lines in sorted(ctx.added_lines.items()):
                    for line in lines:
                        if pattern.search(line):
                            hits.append({"path": path, "line": line.strip()[:120]})
                if hits:
                    findings.append(
                        self._finding(
                            rule=rule,
                            severity=severity,
                            confidence=confidence,
                            paths=tuple(sorted({h["path"] for h in hits})),
                            summary=f"{len(hits)} added line(s) match {rule}",
                            details={"hits": hits[:20]},
                        )
                    )
        return findings


DEFAULT_REVIEWERS: Tuple[Reviewer, ...] = (
    TestCoverageReviewer(),
    DependencyReviewer(),
    SecurityReviewer(),
)


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReviewSynthesis:
    """What a human actually reads: disagreements, blockers, attention spots."""

    verdict: str  # worst verdict across the panel
    reports: Tuple[ReviewerReport, ...]
    disagreements: Tuple[Dict[str, Any], ...]
    blocking: Tuple[ReviewFinding, ...]
    attention: Tuple[ReviewFinding, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reviewers": {r.reviewer: r.verdict for r in self.reports},
            "disagreements": [dict(d) for d in self.disagreements],
            "blocking": [f.to_dict() for f in self.blocking],
            "attention": [f.to_dict() for f in self.attention],
        }

    def to_review_dicts(self) -> List[Dict[str, Any]]:
        """Per-reviewer dicts for the provenance bundle's ``reviews`` field.

        Recorded reviews are folded into the signed payload by
        :func:`swarm.agentgit.bundle.build_bundle`, making the panel outcome
        tamper-evident alongside the rest of the provenance.
        """

        reviews = [report.to_dict() for report in self.reports]
        reviews.append(
            {
                "reviewer": "panel-synthesis",
                "decision": self.verdict,
                "disagreements": [dict(d) for d in self.disagreements],
            }
        )
        return reviews

    def format_text(self) -> str:
        lines = [f"panel verdict: {self.verdict.upper()}"]
        for report in self.reports:
            lines.append(
                f"- {report.reviewer}: {report.verdict}"
                + (f" ({len(report.findings)} finding(s))" if report.findings else "")
            )
        if self.disagreements:
            lines.append("disagreements:")
            for d in self.disagreements:
                sides = "; ".join(
                    f"{verdict}: {', '.join(names)}" for verdict, names in d["sides"].items()
                )
                lines.append(f"- {sides}")
        if self.blocking:
            lines.append("blocking findings:")
            for f in self.blocking:
                lines.append(f"- [{f.reviewer}/{f.rule}] {f.summary}")
        if self.attention:
            lines.append("needs human attention (high-risk / low-confidence):")
            for f in self.attention:
                lines.append(
                    f"- [{f.reviewer}/{f.rule}] {f.summary} "
                    f"(severity={f.severity}, confidence={f.confidence:.2f})"
                )
        return "\n".join(lines)


def synthesize(reports: Sequence[ReviewerReport]) -> ReviewSynthesis:
    """Merge per-reviewer reports into the human-facing synthesis.

    - **Disagreements**: the panel splits across distinct verdicts (e.g. two
      reviewers approve, one blocks). Each disagreement lists who is on which
      side and the findings driving the harsher verdicts.
    - **Blocking**: every blocking finding (policy-violation grade).
    - **Attention**: blocking findings plus any finding a reviewer was unsure
      about (confidence < ``LOW_CONFIDENCE``) — high-risk or low-confidence
      spots, surfaced even when non-blocking.
    """

    all_findings = [f for report in reports for f in report.findings]
    blocking = tuple(f for f in all_findings if f.severity == "blocking")
    attention = tuple(
        f
        for f in all_findings
        if f.severity == "blocking" or f.confidence < LOW_CONFIDENCE
    )

    by_verdict: Dict[str, List[str]] = {}
    for report in reports:
        by_verdict.setdefault(report.verdict, []).append(report.reviewer)
    disagreements: List[Dict[str, Any]] = []
    if len(by_verdict) > 1:
        harsher = [v for v in by_verdict if v != VERDICT_APPROVE]
        disagreements.append(
            {
                "sides": {v: sorted(names) for v, names in by_verdict.items()},
                "driving_findings": [
                    f.to_dict()
                    for report in reports
                    if report.verdict in harsher
                    for f in report.findings
                ],
            }
        )

    verdict = VERDICT_APPROVE
    for report in reports:
        if _VERDICT_RANK[report.verdict] > _VERDICT_RANK[verdict]:
            verdict = report.verdict

    return ReviewSynthesis(
        verdict=verdict,
        reports=tuple(reports),
        disagreements=tuple(disagreements),
        blocking=blocking,
        attention=attention,
    )


# ---------------------------------------------------------------------------
# running a panel
# ---------------------------------------------------------------------------
def collect_added_lines(repo: Path, base_ref: str = "HEAD") -> Dict[str, List[str]]:
    """Map changed path -> lines the diff adds (full content when untracked)."""

    repo = repo.resolve()
    added: Dict[str, List[str]] = {}
    diff = _git(repo, ["diff", "--unified=0", base_ref, "--"])
    current: Optional[str] = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+++"):
            current = None  # /dev/null (deletion)
        elif current and line.startswith("+") and not line.startswith("+++"):
            added.setdefault(current, []).append(line[1:])
    untracked = _git(repo, ["ls-files", "--others", "--exclude-standard"])
    for path in untracked.splitlines():
        full = repo / path
        if full.is_file():
            try:
                added[path] = full.read_text(errors="replace").splitlines()
            except OSError:
                added[path] = []
    return added


def run_review_panel(
    repo: Path,
    *,
    base_ref: str = "HEAD",
    reviewers: Sequence[Reviewer] | None = None,
) -> ReviewSynthesis:
    """Run every reviewer over the current diff and synthesize the outcome."""

    snapshot = collect_snapshot(repo, base_ref=base_ref)
    ctx = ReviewContext(
        snapshot=snapshot, added_lines=collect_added_lines(repo, base_ref)
    )
    reports = [
        ReviewerReport(
            reviewer=reviewer.name,
            verdict=verdict_for(findings),
            findings=tuple(findings),
        )
        for reviewer in (reviewers if reviewers is not None else DEFAULT_REVIEWERS)
        for findings in (reviewer.review(ctx),)
    ]
    return synthesize(reports)


