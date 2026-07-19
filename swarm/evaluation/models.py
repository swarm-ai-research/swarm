"""Data models for SWARM agent-authored research evaluation."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class AuthorType(str, Enum):
    """Type of research author."""

    AGENT = "agent"
    HUMAN = "human"
    HYBRID = "hybrid"


class Verdict(str, Enum):
    """Review verdict for a submission."""

    PUBLISH = "publish"
    REVISE = "revise"
    REJECT = "reject"


class Author(BaseModel):
    """A submission author (agent, human, or hybrid)."""

    name: str
    type: AuthorType
    model: Optional[str] = None


class Submission(BaseModel):
    """Metadata for a research submission under review."""

    id: str
    title: str
    authors: List[Author]
    artifact_urls: List[str] = Field(default_factory=list)
    claims_summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Scores(BaseModel):
    """Normalized scores (0-1) for each evaluation axis."""

    experimental_validity: float = Field(default=0.0, ge=0.0, le=1.0)
    reproducibility: float = Field(default=0.0, ge=0.0, le=1.0)
    artifact_integrity: float = Field(default=0.0, ge=0.0, le=1.0)
    emergence_evidence: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_mode_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class Checks(BaseModel):
    """Raw check values from evaluators."""

    design_consistency: Optional[str] = Field(default=None, pattern=r"^(pass|fail)$")
    replay_success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    artifact_resolution_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    artifact_hash_match_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Emergence detection: separate test quality from result
    emergence_test_conducted: Optional[bool] = Field(
        default=None,
        description="Whether a proper multi-agent vs single-agent comparison was conducted",
    )
    emergence_delta: Optional[float] = Field(
        default=None,
        description="Multi-agent outcome minus max single-agent outcome (can be negative)",
    )
    emergence_result_type: Optional[str] = Field(
        default=None,
        pattern=r"^(positive|null|negative)$",
        description="Classification of emergence result",
    )
    topology_sensitivity: Optional[float] = None
    falsification_attempts_count: Optional[int] = Field(default=None, ge=0)
    documented_failure_modes_count: Optional[int] = Field(default=None, ge=0)


class KeyArtifact(BaseModel):
    """A referenced artifact with integrity metadata."""

    label: str
    url: str
    sha256: Optional[str] = None


class Evidence(BaseModel):
    """Evidence supporting the review."""

    key_artifacts: List[KeyArtifact] = Field(default_factory=list)


class Notes(BaseModel):
    """Reviewer notes and recommendations."""

    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    required_changes: List[str] = Field(default_factory=list)
    optional_suggestions: List[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Complete review result conforming to the SWARM review schema v1."""

    schema_version: str = Field(default="v1", pattern=r"^v\d+$")
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    submission: Submission
    verdict: Verdict
    scores: Scores = Field(default_factory=Scores)
    checks: Checks = Field(default_factory=Checks)
    evidence: Evidence = Field(default_factory=Evidence)
    notes: Notes = Field(default_factory=Notes)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ReviewResult":
        """Ensure PUBLISH verdict does not have design_consistency='fail'."""
        if self.verdict == Verdict.PUBLISH:
            if self.checks.design_consistency == "fail":
                raise ValueError("Cannot publish with design_consistency='fail'")
        return self

    def to_dict(self) -> dict:
        """Serialize to dictionary matching the JSON schema."""
        return {
            "schema_version": self.schema_version,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "submission": {
                "id": self.submission.id,
                "title": self.submission.title,
                "authors": [
                    {
                        "name": a.name,
                        "type": a.type.value,
                        **({"model": a.model} if a.model else {}),
                    }
                    for a in self.submission.authors
                ],
                "artifact_urls": self.submission.artifact_urls,
                **(
                    {"claims_summary": self.submission.claims_summary}
                    if self.submission.claims_summary
                    else {}
                ),
                "tags": self.submission.tags,
            },
            "verdict": self.verdict.value,
            "scores": {
                "experimental_validity": self.scores.experimental_validity,
                "reproducibility": self.scores.reproducibility,
                "artifact_integrity": self.scores.artifact_integrity,
                "emergence_evidence": self.scores.emergence_evidence,
                "failure_mode_coverage": self.scores.failure_mode_coverage,
            },
            "checks": {
                k: v
                for k, v in {
                    "design_consistency": self.checks.design_consistency,
                    "replay_success_rate": self.checks.replay_success_rate,
                    "artifact_resolution_rate": self.checks.artifact_resolution_rate,
                    "artifact_hash_match_rate": self.checks.artifact_hash_match_rate,
                    "emergence_delta": self.checks.emergence_delta,
                    "topology_sensitivity": self.checks.topology_sensitivity,
                    "falsification_attempts_count": self.checks.falsification_attempts_count,
                    "documented_failure_modes_count": self.checks.documented_failure_modes_count,
                }.items()
                if v is not None
            },
            "evidence": {
                "key_artifacts": [
                    {
                        "label": a.label,
                        "url": a.url,
                        **({"sha256": a.sha256} if a.sha256 else {}),
                    }
                    for a in self.evidence.key_artifacts
                ]
            },
            "notes": {
                "strengths": self.notes.strengths,
                "weaknesses": self.notes.weaknesses,
                "required_changes": self.notes.required_changes,
                "optional_suggestions": self.notes.optional_suggestions,
            },
        }
