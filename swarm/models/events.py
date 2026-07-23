"""Event schema for append-only logging."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def generate_deterministic_id(
    event_type: str,
    agent_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    **kwargs: Any,
) -> str:
    """Generate a deterministic provenance ID based on event attributes.

    Args:
        event_type: Type of the event
        agent_id: Agent identifier
        timestamp: Event timestamp
        **kwargs: Additional attributes to include in hash

    Returns:
        12-character hex string suitable for provenance tracking
    """
    components = {
        "event_type": event_type,
        "agent_id": agent_id,
        "timestamp": timestamp.isoformat() if timestamp else None,
        **kwargs,
    }
    # Sort keys for deterministic hashing
    canonical = json.dumps(components, sort_keys=True)
    hash_digest = hashlib.sha256(canonical.encode()).hexdigest()
    return hash_digest[:12]


class EventType(Enum):
    """Types of events that can be logged."""

    # Interaction lifecycle
    INTERACTION_PROPOSED = "interaction_proposed"
    INTERACTION_ACCEPTED = "interaction_accepted"
    INTERACTION_REJECTED = "interaction_rejected"
    INTERACTION_COMPLETED = "interaction_completed"

    # Agent events
    AGENT_CREATED = "agent_created"
    AGENT_STATE_UPDATED = "agent_state_updated"

    # Proxy computation
    PROXY_COMPUTED = "proxy_computed"

    # Payoff events
    PAYOFF_COMPUTED = "payoff_computed"
    TRANSFER_EXECUTED = "transfer_executed"

    # Governance events
    GOVERNANCE_COST_APPLIED = "governance_cost_applied"
    REPUTATION_UPDATED = "reputation_updated"

    # Marketplace events
    BOUNTY_POSTED = "bounty_posted"
    BID_PLACED = "bid_placed"
    BID_ACCEPTED = "bid_accepted"
    BID_REJECTED = "bid_rejected"
    ESCROW_CREATED = "escrow_created"
    ESCROW_RELEASED = "escrow_released"
    ESCROW_REFUNDED = "escrow_refunded"
    DISPUTE_FILED = "dispute_filed"
    DISPUTE_RESOLVED = "dispute_resolved"

    # Moltipedia wiki events
    PAGE_CREATED = "page_created"
    PAGE_EDITED = "page_edited"
    OBJECTION_FILED = "objection_filed"
    POLICY_VIOLATION_FLAGGED = "policy_violation_flagged"
    POINTS_AWARDED = "points_awarded"
    PAIR_CAP_TRIGGERED = "pair_cap_triggered"
    COOLDOWN_TRIGGERED = "cooldown_triggered"
    DAILY_CAP_TRIGGERED = "daily_cap_triggered"

    # Moltbook events
    POST_SUBMITTED = "post_submitted"
    COMMENT_SUBMITTED = "comment_submitted"
    CHALLENGE_ISSUED = "challenge_issued"
    CHALLENGE_PASSED = "challenge_passed"
    CHALLENGE_FAILED = "challenge_failed"
    CHALLENGE_EXPIRED = "challenge_expired"
    CONTENT_PUBLISHED = "content_published"
    RATE_LIMIT_HIT = "rate_limit_hit"
    KARMA_UPDATED = "karma_updated"

    # Memory tier events
    MEMORY_WRITTEN = "memory_written"
    MEMORY_PROMOTED = "memory_promoted"
    MEMORY_VERIFIED = "memory_verified"
    MEMORY_CHALLENGED = "memory_challenged"
    MEMORY_REVERTED = "memory_reverted"
    MEMORY_CACHE_REBUILT = "memory_cache_rebuilt"
    MEMORY_COMPACTION = "memory_compaction"
    MEMORY_WRITE_RATE_LIMITED = "memory_write_rate_limited"
    MEMORY_PROMOTION_BLOCKED = "memory_promotion_blocked"

    # Scholar/literature synthesis events
    SCHOLAR_RETRIEVAL = "scholar_retrieval"
    SCHOLAR_SYNTHESIS = "scholar_synthesis"
    SCHOLAR_VERIFICATION = "scholar_verification"

    # Kernel market events
    KERNEL_SUBMITTED = "kernel_submitted"
    KERNEL_VERIFIED = "kernel_verified"
    KERNEL_AUDITED = "kernel_audited"

    # Spawn events
    AGENT_SPAWNED = "agent_spawned"
    SPAWN_REJECTED = "spawn_rejected"

    # Council events
    COUNCIL_DELIBERATION = "council_deliberation"
    COUNCIL_AUDIT = "council_audit"

    # Peer review events
    PEER_REVIEW_SUBMITTED = "peer_review_submitted"
    REVIEW_GATE_EVALUATED = "review_gate_evaluated"

    # Self-modification governance events
    SELF_MODIFICATION_PROPOSED = "self_modification_proposed"
    SELF_MODIFICATION_REVIEWED = "self_modification_reviewed"
    SELF_MODIFICATION_APPROVED = "self_modification_approved"
    SELF_MODIFICATION_DENIED = "self_modification_denied"
    SELF_MODIFICATION_EXECUTED = "self_modification_executed"
    SELF_MODIFICATION_REVERTED = "self_modification_reverted"

    # Attestation heartbeat events
    HEARTBEAT_ATTESTED = "heartbeat_attested"
    HEARTBEAT_LAPSED = "heartbeat_lapsed"

    # AWM (Agent World Model) events
    AWM_TASK_ASSIGNED = "awm_task_assigned"
    AWM_TASK_COMPLETED = "awm_task_completed"
    AWM_TOOL_CALL_EXECUTED = "awm_tool_call_executed"
    AWM_CONFLICT_DETECTED = "awm_conflict_detected"
    AWM_TRANSACTION_COMPLETED = "awm_transaction_completed"

    # Contract screening events
    CONTRACT_SIGNING = "contract_signing"
    CONTRACT_METRICS = "contract_metrics"

    # Adaptive governance events
    GOVERNANCE_THRESHOLD_PROPOSED = "governance_threshold_proposed"
    GOVERNANCE_THRESHOLD_APPLIED = "governance_threshold_applied"
    GOVERNANCE_THRESHOLD_CRYSTALLIZED = "governance_threshold_crystallized"
    GOVERNANCE_THRESHOLD_REVERTED = "governance_threshold_reverted"
    GOVERNANCE_CONTEMPLATION_COMPLETED = "governance_contemplation_completed"

    # Attestation events
    RECEIPT_SEALED = "receipt_sealed"
    RECEIPT_VERIFIED = "receipt_verified"
    RECEIPT_REJECTED = "receipt_rejected"
    # AgentVeil bridge write-back attestations (failure-mode D4 — never hidden):
    # every AVP attestation submission/suppression is mirrored here so the
    # core JSONL log fully replays the AVP-visible side-effects of a run.
    ATTESTATION_SUBMITTED = "attestation_submitted"
    ATTESTATION_SUPPRESSED = "attestation_suppressed"
    RELAY_MESSAGE_SENT = "relay_message_sent"
    RELAY_MESSAGE_ACKNOWLEDGED = "relay_message_acknowledged"
    # Hardware trust events
    HARDWARE_HALT_RECEIVED = "hardware_halt_received"
    HARDWARE_HALT_PROPAGATED = "hardware_halt_propagated"
    HARDWARE_RECOVERY_ENTERED = "hardware_recovery_entered"
    HARDWARE_CONDITION_CLEARED = "hardware_condition_cleared"
    HARDWARE_SAFE_RESUME = "hardware_safe_resume"

    # System events
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_ENDED = "simulation_ended"
    EPOCH_COMPLETED = "epoch_completed"


class Event(BaseModel):
    """
    An event for the append-only log.

    Events capture state changes and can be replayed to reconstruct
    the full history of a simulation.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: EventType = EventType.INTERACTION_PROPOSED

    # References to related entities
    interaction_id: Optional[str] = None
    agent_id: Optional[str] = None
    initiator_id: Optional[str] = None
    counterparty_id: Optional[str] = None

    # Event payload (flexible structure)
    payload: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    epoch: Optional[int] = None
    step: Optional[int] = None
    scenario_id: Optional[str] = None
    replay_k: Optional[int] = None
    seed: Optional[int] = None

    # Provenance tracking - unified across all event types
    provenance_id: Optional[str] = None  # Deterministic ID for this event in provenance chain
    parent_event_id: Optional[str] = None  # Links to parent event for causal chain
    tool_call_id: Optional[str] = None  # ID for tool call if this is a tool execution
    artifact_id: Optional[str] = None  # ID for artifact if this event produces/references one
    audit_id: Optional[str] = None  # ID for audit if this is an audit event
    intervention_id: Optional[str] = None  # ID for governance intervention

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "interaction_id": self.interaction_id,
            "agent_id": self.agent_id,
            "initiator_id": self.initiator_id,
            "counterparty_id": self.counterparty_id,
            "payload": self.payload,
            "epoch": self.epoch,
            "step": self.step,
            "scenario_id": self.scenario_id,
            "replay_k": self.replay_k,
            "seed": self.seed,
            "provenance_id": self.provenance_id,
            "parent_event_id": self.parent_event_id,
            "tool_call_id": self.tool_call_id,
            "artifact_id": self.artifact_id,
            "audit_id": self.audit_id,
            "intervention_id": self.intervention_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=EventType(data["event_type"]),
            interaction_id=data.get("interaction_id"),
            agent_id=data.get("agent_id"),
            initiator_id=data.get("initiator_id"),
            counterparty_id=data.get("counterparty_id"),
            payload=data.get("payload", {}),
            epoch=data.get("epoch"),
            step=data.get("step"),
            scenario_id=data.get("scenario_id"),
            replay_k=data.get("replay_k"),
            seed=data.get("seed"),
            provenance_id=data.get("provenance_id"),
            parent_event_id=data.get("parent_event_id"),
            tool_call_id=data.get("tool_call_id"),
            artifact_id=data.get("artifact_id"),
            audit_id=data.get("audit_id"),
            intervention_id=data.get("intervention_id"),
        )


# Factory functions for common events


def interaction_proposed_event(
    interaction_id: str,
    initiator_id: str,
    counterparty_id: str,
    interaction_type: str,
    v_hat: float,
    p: float,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    causal_parents: Optional[List[str]] = None,
) -> Event:
    """Create an interaction proposed event."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if not (-1.0 <= v_hat <= 1.0):
        raise ValueError(f"v_hat must be in [-1, 1], got {v_hat}")
    payload: Dict[str, Any] = {
        "interaction_type": interaction_type,
        "v_hat": v_hat,
        "p": p,
    }
    if causal_parents:
        payload["causal_parents"] = causal_parents
    return Event(
        event_type=EventType.INTERACTION_PROPOSED,
        interaction_id=interaction_id,
        initiator_id=initiator_id,
        counterparty_id=counterparty_id,
        payload=payload,
        epoch=epoch,
        step=step,
    )


def interaction_completed_event(
    interaction_id: str,
    accepted: bool,
    payoff_initiator: float,
    payoff_counterparty: float,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create an interaction completed event."""
    return Event(
        event_type=EventType.INTERACTION_COMPLETED,
        interaction_id=interaction_id,
        payload={
            "accepted": accepted,
            "payoff_initiator": payoff_initiator,
            "payoff_counterparty": payoff_counterparty,
        },
        epoch=epoch,
        step=step,
    )


def payoff_computed_event(
    interaction_id: str,
    initiator_id: str,
    counterparty_id: str,
    payoff_initiator: float,
    payoff_counterparty: float,
    components: dict,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create a payoff computed event with full breakdown."""
    return Event(
        event_type=EventType.PAYOFF_COMPUTED,
        interaction_id=interaction_id,
        initiator_id=initiator_id,
        counterparty_id=counterparty_id,
        payload={
            "payoff_initiator": payoff_initiator,
            "payoff_counterparty": payoff_counterparty,
            "components": components,
        },
        epoch=epoch,
        step=step,
    )


def peer_review_submitted_event(
    paper_id: str,
    review_id: str,
    reviewer_id: str,
    recommendation: str,
    rating: int,
) -> Event:
    """Create a peer review submitted event."""
    return Event(
        event_type=EventType.PEER_REVIEW_SUBMITTED,
        payload={
            "paper_id": paper_id,
            "review_id": review_id,
            "reviewer_id": reviewer_id,
            "recommendation": recommendation,
            "rating": rating,
        },
    )


def review_gate_evaluated_event(
    paper_id: str,
    passed: bool,
    failed_checks: list[str],
) -> Event:
    """Create a review gate evaluated event."""
    return Event(
        event_type=EventType.REVIEW_GATE_EVALUATED,
        payload={
            "paper_id": paper_id,
            "passed": passed,
            "failed_checks": failed_checks,
        },
    )


def reputation_updated_event(
    agent_id: str,
    old_reputation: float,
    new_reputation: float,
    delta: float,
    reason: str,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create a reputation update event."""
    return Event(
        event_type=EventType.REPUTATION_UPDATED,
        agent_id=agent_id,
        payload={
            "old_reputation": old_reputation,
            "new_reputation": new_reputation,
            "delta": delta,
            "reason": reason,
        },
        epoch=epoch,
        step=step,
    )


# Provenance-aware event factories


def tool_call_executed_event(
    agent_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Any,
    success: bool,
    parent_event_id: Optional[str] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create a tool call execution event with provenance tracking.

    Args:
        agent_id: Agent executing the tool
        tool_name: Name of the tool
        arguments: Tool arguments
        result: Tool execution result
        success: Whether execution succeeded
        parent_event_id: ID of parent event that triggered this call
        epoch: Current epoch
        step: Current step

    Returns:
        Event with tool call provenance
    """
    timestamp = datetime.now()
    tool_call_id = generate_deterministic_id(
        event_type="tool_call",
        agent_id=agent_id,
        tool_name=tool_name,
        timestamp=timestamp,
    )
    provenance_id = generate_deterministic_id(
        event_type="awm_tool_call_executed",
        agent_id=agent_id,
        tool_call_id=tool_call_id,
        timestamp=timestamp,
    )

    return Event(
        event_type=EventType.AWM_TOOL_CALL_EXECUTED,
        agent_id=agent_id,
        timestamp=timestamp,
        payload={
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "success": success,
        },
        provenance_id=provenance_id,
        tool_call_id=tool_call_id,
        parent_event_id=parent_event_id,
        epoch=epoch,
        step=step,
    )


def artifact_created_event(
    agent_id: str,
    artifact_id: str,
    artifact_type: str,
    artifact_data: Dict[str, Any],
    parent_event_id: Optional[str] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create an artifact creation event with provenance tracking.

    Args:
        agent_id: Agent creating the artifact
        artifact_id: Unique artifact identifier
        artifact_type: Type of artifact (e.g., "memory", "skill", "kernel")
        artifact_data: Artifact payload
        parent_event_id: ID of parent event
        epoch: Current epoch
        step: Current step

    Returns:
        Event with artifact provenance
    """
    timestamp = datetime.now()
    provenance_id = generate_deterministic_id(
        event_type="artifact_created",
        agent_id=agent_id,
        artifact_id=artifact_id,
        timestamp=timestamp,
    )

    return Event(
        # Deliberate reuse: artifact events share MEMORY_WRITTEN so old logs
        # replay identically. Distinguish via artifact_id / provenance_id
        # (which embeds "artifact_created"), not event_type.
        event_type=EventType.MEMORY_WRITTEN,
        agent_id=agent_id,
        timestamp=timestamp,
        payload={
            "artifact_type": artifact_type,
            "artifact_data": artifact_data,
        },
        provenance_id=provenance_id,
        artifact_id=artifact_id,
        parent_event_id=parent_event_id,
        epoch=epoch,
        step=step,
    )


def audit_event(
    agent_id: str,
    audit_type: str,
    audit_result: Dict[str, Any],
    audited_event_id: Optional[str] = None,
    parent_event_id: Optional[str] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create an audit event with provenance tracking.

    Args:
        agent_id: Agent or system performing audit
        audit_type: Type of audit (e.g., "council", "kernel", "proxy")
        audit_result: Audit findings and decisions
        audited_event_id: ID of event being audited
        parent_event_id: ID of parent event
        epoch: Current epoch
        step: Current step

    Returns:
        Event with audit provenance
    """
    timestamp = datetime.now()
    audit_id = generate_deterministic_id(
        event_type="audit",
        agent_id=agent_id,
        audit_type=audit_type,
        timestamp=timestamp,
    )
    provenance_id = generate_deterministic_id(
        event_type="council_audit",
        agent_id=agent_id,
        audit_id=audit_id,
        timestamp=timestamp,
    )

    return Event(
        event_type=EventType.COUNCIL_AUDIT,
        agent_id=agent_id,
        timestamp=timestamp,
        payload={
            "audit_type": audit_type,
            "audit_result": audit_result,
            "audited_event_id": audited_event_id,
        },
        provenance_id=provenance_id,
        audit_id=audit_id,
        parent_event_id=parent_event_id,
        epoch=epoch,
        step=step,
    )


def intervention_event(
    agent_id: str,
    intervention_type: str,
    intervention_action: str,
    intervention_reason: str,
    affected_agents: list[str],
    parent_event_id: Optional[str] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create a governance intervention event with provenance tracking.

    Args:
        agent_id: Governance system or agent performing intervention
        intervention_type: Type of intervention (e.g., "freeze", "quarantine", "throttle")
        intervention_action: Specific action taken
        intervention_reason: Justification for intervention
        affected_agents: List of affected agent IDs
        parent_event_id: ID of parent event that triggered intervention
        epoch: Current epoch
        step: Current step

    Returns:
        Event with intervention provenance
    """
    timestamp = datetime.now()
    intervention_id = generate_deterministic_id(
        event_type="intervention",
        agent_id=agent_id,
        intervention_type=intervention_type,
        timestamp=timestamp,
    )
    provenance_id = generate_deterministic_id(
        event_type="governance_intervention",
        agent_id=agent_id,
        intervention_id=intervention_id,
        timestamp=timestamp,
    )

    return Event(
        # Deliberate reuse: interventions share GOVERNANCE_COST_APPLIED so old
        # logs replay identically. Distinguish via intervention_id /
        # provenance_id (embeds "governance_intervention"), not event_type.
        event_type=EventType.GOVERNANCE_COST_APPLIED,
        agent_id=agent_id,
        timestamp=timestamp,
        payload={
            "intervention_type": intervention_type,
            "intervention_action": intervention_action,
            "intervention_reason": intervention_reason,
            "affected_agents": affected_agents,
        },
        provenance_id=provenance_id,
        intervention_id=intervention_id,
        parent_event_id=parent_event_id,
        epoch=epoch,
        step=step,
    )


def agent_message_event(
    agent_id: str,
    message_role: str,
    message_content: str,
    tool_calls: Optional[list[Dict[str, Any]]] = None,
    parent_event_id: Optional[str] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> Event:
    """Create an agent message event with provenance tracking.

    Args:
        agent_id: Agent sending the message
        message_role: Role (e.g., "user", "assistant", "system")
        message_content: Message content
        tool_calls: Optional list of tool calls in the message
        parent_event_id: ID of parent event
        epoch: Current epoch
        step: Current step

    Returns:
        Event with message provenance
    """
    timestamp = datetime.now()
    provenance_id = generate_deterministic_id(
        event_type="agent_message",
        agent_id=agent_id,
        timestamp=timestamp,
    )

    return Event(
        # Deliberate reuse: agent messages share AGENT_STATE_UPDATED so old
        # logs replay identically. Distinguish via provenance_id (embeds
        # "agent_message"), not event_type.
        event_type=EventType.AGENT_STATE_UPDATED,
        agent_id=agent_id,
        timestamp=timestamp,
        payload={
            "message_role": message_role,
            "message_content": message_content,
            "tool_calls": tool_calls or [],
        },
        provenance_id=provenance_id,
        parent_event_id=parent_event_id,
        epoch=epoch,
        step=step,
    )
