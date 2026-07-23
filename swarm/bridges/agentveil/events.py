"""AgentVeil event types and typed event payloads.

Every interaction the bridge has with AVP (trust check, attestation
submission, reputation fetch, circuit-breaker trip) is recorded as an
``AgentVeilEvent`` and appended to SWARM's event log. This is the
mitigation for failure-mode D4 (no hidden attestation side-effects):
all AVP calls go through the bridge and emit a corresponding event, so
the JSONL log fully replays the AVP-visible portion of a run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentVeilEventType(Enum):
    # Trust / admission
    TRUST_CHECK_REQUESTED = "trust:check_requested"
    TRUST_ALLOWED = "trust:allowed"
    TRUST_DENIED = "trust:denied"

    # Reputation
    REPUTATION_FETCHED = "reputation:fetched"

    # Write-back attestations (mitigation D4 — never hidden)
    ATTESTATION_SUBMITTED = "attestation:submitted"
    ATTESTATION_SUPPRESSED = "attestation:suppressed"  # uncertain band / rate-limited

    # Operational
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker:tripped"
    REGISTRY_ERROR = "registry:error"

    GENERIC = "generic"
    ERROR = "error"


@dataclass
class AgentVeilEvent:
    """Base envelope for any AVP-related event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AgentVeilEventType = AgentVeilEventType.GENERIC
    timestamp: datetime = field(default_factory=_utcnow)
    subject_did: str = ""  # DID the event is about (counterparty, attestee, etc.)
    actor_did: str = ""  # DID performing the action (usually the orchestrator)
    step: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "subject_did": self.subject_did,
            "actor_did": self.actor_did,
            "step": self.step,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentVeilEvent:
        ts = data.get("timestamp", _utcnow())
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=AgentVeilEventType(data["event_type"]),
            timestamp=ts,
            subject_did=data.get("subject_did", ""),
            actor_did=data.get("actor_did", ""),
            step=int(data.get("step", 0)),
            payload=data.get("payload", {}),
        )


@dataclass
class TrustDecisionEvent:
    """Typed payload for a TRUST_ALLOWED / TRUST_DENIED event."""

    allowed: bool = False
    tier: str = ""  # newcomer | basic | trusted | elite
    risk_level: str = ""  # low | medium | high
    confidence: float = 0.0  # AVP Bayesian confidence in [0, 1]
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "tier": self.tier,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrustDecisionEvent:
        return cls(
            allowed=bool(data.get("allowed", False)),
            tier=data.get("tier", ""),
            risk_level=data.get("risk_level", ""),
            confidence=float(data.get("confidence", 0.0)),
            reason=data.get("reason", ""),
        )


@dataclass
class AttestationEvent:
    """Typed payload for an ATTESTATION_SUBMITTED / SUPPRESSED event.

    Per failure-mode E3, the bridge never sends the raw ``p`` to the
    registry — only the sign and the canonical evidence hash from the
    plan doc (``SHA-256(interaction_id || outcome_sign)``).
    """

    interaction_id: str = ""
    outcome_sign: int = 0  # +1 (positive), -1 (negative); 0 = suppressed
    evidence_hash: str = ""  # hex-encoded SHA-256
    suppression_reason: Optional[str] = None  # set when SUPPRESSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "outcome_sign": self.outcome_sign,
            "evidence_hash": self.evidence_hash,
            "suppression_reason": self.suppression_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AttestationEvent:
        return cls(
            interaction_id=data.get("interaction_id", ""),
            outcome_sign=int(data.get("outcome_sign", 0)),
            evidence_hash=data.get("evidence_hash", ""),
            suppression_reason=data.get("suppression_reason"),
        )


@dataclass
class ReputationSnapshotEvent:
    """Typed payload for a REPUTATION_FETCHED event."""

    score: float = 0.0  # AVP Bayesian score in [0, 1]
    confidence: float = 0.0  # in [0, 1]
    tier: str = ""
    attestation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "tier": self.tier,
            "attestation_count": self.attestation_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReputationSnapshotEvent:
        return cls(
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            tier=data.get("tier", ""),
            attestation_count=int(data.get("attestation_count", 0)),
        )
