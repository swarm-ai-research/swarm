"""JSON Schema definition and validation for SWARM review documents."""

from typing import Any, Dict, List, Tuple

REVIEW_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://swarm-ai.org/schemas/review/v1.json",
    "title": "SWARM Agent Research Review",
    "type": "object",
    "required": [
        "schema_version",
        "submission",
        "verdict",
        "scores",
        "checks",
        "evidence",
        "notes",
        "timestamp_utc",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "v1"},
        "timestamp_utc": {"type": "string", "format": "date-time"},
        "submission": {
            "type": "object",
            "required": ["id", "title", "authors", "artifact_urls"],
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "authors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["agent", "human", "hybrid"],
                            },
                            "model": {"type": "string"},
                        },
                    },
                },
                "artifact_urls": {
                    "type": "array",
                    "items": {"type": "string", "format": "uri"},
                },
                "claims_summary": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
        "verdict": {
            "type": "string",
            "enum": ["publish", "revise", "reject"],
        },
        "scores": {
            "type": "object",
            "properties": {
                "experimental_validity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "reproducibility": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "artifact_integrity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "emergence_evidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "failure_mode_coverage": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
        },
        "checks": {
            "type": "object",
            "properties": {
                "design_consistency": {
                    "type": "string",
                    "enum": ["pass", "fail"],
                },
                "replay_success_rate": {"type": "number"},
                "artifact_resolution_rate": {"type": "number"},
                "artifact_hash_match_rate": {"type": "number"},
                "emergence_delta": {"type": "number"},
                "topology_sensitivity": {"type": "number"},
                "falsification_attempts_count": {"type": "integer"},
                "documented_failure_modes_count": {"type": "integer"},
            },
        },
        "evidence": {
            "type": "object",
            "properties": {
                "key_artifacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "url": {"type": "string"},
                            "sha256": {"type": "string"},
                        },
                    },
                }
            },
        },
        "notes": {
            "type": "object",
            "properties": {
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "weaknesses": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "required_changes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "optional_suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
}


def validate_review(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a review document against the SWARM review schema.

    Performs structural validation without requiring jsonschema as a dependency.
    Validates required fields, types, enums, and numeric bounds for submission,
    scores, and checks. Notes and evidence fields are checked for presence only.

    Args:
        data: Dictionary to validate against the schema.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    # Top-level required fields
    for field in REVIEW_SCHEMA["required"]:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # schema_version
    if data.get("schema_version") != "v1":
        errors.append(
            f"schema_version must be 'v1', got {data.get('schema_version')!r}"
        )

    # timestamp_utc
    if not isinstance(data.get("timestamp_utc"), str):
        errors.append("timestamp_utc must be a string")

    # verdict
    if data.get("verdict") not in ("publish", "revise", "reject"):
        errors.append(
            f"verdict must be one of publish/revise/reject, got {data.get('verdict')!r}"
        )

    # submission
    _validate_submission(data.get("submission", {}), errors)

    # scores
    _validate_scores(data.get("scores", {}), errors)

    # checks
    _validate_checks(data.get("checks", {}), errors)

    return len(errors) == 0, errors


def _validate_submission(sub: Any, errors: List[str]) -> None:
    """Validate submission block."""
    if not isinstance(sub, dict):
        errors.append("submission must be an object")
        return

    for field in ("id", "title", "authors", "artifact_urls"):
        if field not in sub:
            errors.append(f"submission missing required field: {field}")

    if "authors" in sub:
        if not isinstance(sub["authors"], list):
            errors.append("submission.authors must be an array")
        else:
            for i, author in enumerate(sub["authors"]):
                if not isinstance(author, dict):
                    errors.append(f"submission.authors[{i}] must be an object")
                    continue
                if "name" not in author:
                    errors.append(f"submission.authors[{i}] missing 'name'")
                if "type" not in author:
                    errors.append(f"submission.authors[{i}] missing 'type'")
                elif author["type"] not in ("agent", "human", "hybrid"):
                    errors.append(
                        f"submission.authors[{i}].type must be "
                        f"agent/human/hybrid, got {author['type']!r}"
                    )


def _validate_scores(scores: Any, errors: List[str]) -> None:
    """Validate scores block."""
    if not isinstance(scores, dict):
        errors.append("scores must be an object")
        return

    score_fields = [
        "experimental_validity",
        "reproducibility",
        "artifact_integrity",
        "emergence_evidence",
        "failure_mode_coverage",
    ]
    for field in score_fields:
        if field in scores:
            val = scores[field]
            if not isinstance(val, (int, float)):
                errors.append(f"scores.{field} must be a number")
            elif val < 0 or val > 1:
                errors.append(f"scores.{field} must be in [0, 1], got {val}")


def _validate_checks(checks: Any, errors: List[str]) -> None:
    """Validate checks block."""
    if not isinstance(checks, dict):
        errors.append("checks must be an object")
        return

    if "design_consistency" in checks:
        if checks["design_consistency"] not in ("pass", "fail"):
            errors.append("checks.design_consistency must be 'pass' or 'fail'")

    for rate_field in (
        "replay_success_rate",
        "artifact_resolution_rate",
        "artifact_hash_match_rate",
    ):
        if rate_field in checks:
            val = checks[rate_field]
            if not isinstance(val, (int, float)):
                errors.append(f"checks.{rate_field} must be a number")

    for int_field in ("falsification_attempts_count", "documented_failure_modes_count"):
        if int_field in checks:
            val = checks[int_field]
            if not isinstance(val, int):
                errors.append(f"checks.{int_field} must be an integer")
            elif val < 0:
                errors.append(f"checks.{int_field} must be >= 0")
