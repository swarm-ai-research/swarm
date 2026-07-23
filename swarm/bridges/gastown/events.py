"""Event schemas for the SWARM-GasTown bridge.

Defines typed event structures for bead lifecycle, PR workflow,
CI status, and agent lifecycle events observed from a GasTown workspace.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _parse_timestamp(data: Dict[str, Any]) -> datetime:
    """Safely parse a timestamp from a dict, defaulting to UTC now."""
    raw = data.get("timestamp")
    if raw is None:
        return _utcnow()
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return _utcnow()


class GasTownEventType(Enum):
    """Event types in the GasTown bridge protocol."""

    # Bead lifecycle
    BEAD_CREATED = "bead:created"
    BEAD_ASSIGNED = "bead:assigned"
    BEAD_IN_PROGRESS = "bead:in_progress"
    BEAD_COMPLETED = "bead:completed"
    BEAD_BLOCKED = "bead:blocked"

    # PR lifecycle
    PR_OPENED = "pr:opened"
    PR_REVIEW_REQUESTED = "pr:review_requested"
    PR_CHANGES_REQUESTED = "pr:changes_requested"
    PR_APPROVED = "pr:approved"
    PR_MERGED = "pr:merged"

    # CI events
    CI_PASSED = "ci:passed"
    CI_FAILED = "ci:failed"

    # Agent lifecycle
    AGENT_STARTED = "agent:started"
    AGENT_STOPPED = "agent:stopped"

    # Governance
    GOVERNANCE_ACTION = "governance:action"


class BeadState(Enum):
    """Lifecycle states for a GasTown bead."""

    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    BLOCKED = "blocked"


@dataclass
class GasTownEvent:
    """An event observed from a GasTown workspace."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: GasTownEventType = GasTownEventType.BEAD_CREATED
    timestamp: datetime = field(default_factory=_utcnow)
    agent_name: str = ""
    bead_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON transport."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "bead_id": self.bead_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GasTownEvent":
        """Deserialize from dict with safe type handling.

        If event_type is missing or invalid, logs a warning and defaults
        to GOVERNANCE_ACTION to indicate a parse failure.
        """
        try:
            event_type = GasTownEventType(data["event_type"])
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Failed to parse event_type from dict: %s. Defaulting to GOVERNANCE_ACTION. "
                "Data: %s", exc, data
            )
            event_type = GasTownEventType.GOVERNANCE_ACTION
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=_parse_timestamp(data),
            agent_name=str(data.get("agent_name", "")),
            bead_id=data.get("bead_id"),
            payload=data.get("payload", {}),
        )
