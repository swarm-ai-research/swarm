"""Information flow tracking for sandbox boundaries.

This module tracks all information crossing the sandbox boundary,
enabling analysis of what data enters and leaves the system.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class FlowDirection(Enum):
    """Direction of information flow."""

    INBOUND = "inbound"  # External -> Sandbox
    OUTBOUND = "outbound"  # Sandbox -> External
    BIDIRECTIONAL = "bidirectional"  # Both directions


class FlowType(Enum):
    """Type of information flow."""

    DATA = "data"  # Raw data transfer
    QUERY = "query"  # Query/request
    RESPONSE = "response"  # Response to query
    COMMAND = "command"  # Execution command
    CREDENTIAL = "credential"  # Authentication data
    CODE = "code"  # Executable code
    MESSAGE = "message"  # Communication/message
    METADATA = "metadata"  # Metadata about other flows


@dataclass
class InformationFlow:
    """Represents a single information flow event crossing the boundary."""

    flow_id: str
    direction: FlowDirection
    flow_type: FlowType
    timestamp: datetime
    source_id: str  # Agent or entity initiating the flow
    destination_id: str  # Agent or entity receiving the flow
    size_bytes: int
    content_hash: str  # Hash of content for tracking
    sensitivity_score: float = 0.0  # 0 = public, 1 = highly sensitive
    blocked: bool = False
    block_reason: Optional[str] = None

    # Content analysis
    contains_pii: bool = False
    contains_credentials: bool = False
    contains_code: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        direction: FlowDirection,
        flow_type: FlowType,
        source_id: str,
        destination_id: str,
        content: Any,
        sensitivity_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "InformationFlow":
        """Create a new information flow record."""
        content_str = str(content)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        flow_id = f"flow_{content_hash}_{datetime.now().timestamp()}"

        return cls(
            flow_id=flow_id,
            direction=direction,
            flow_type=flow_type,
            timestamp=datetime.now(),
            source_id=source_id,
            destination_id=destination_id,
            size_bytes=len(content_str.encode()),
            content_hash=content_hash,
            sensitivity_score=sensitivity_score,
            metadata=metadata or {},
        )


@dataclass
class FlowSummary:
    """Summary statistics for information flows."""

    total_flows: int = 0
    inbound_flows: int = 0
    outbound_flows: int = 0
    total_bytes_in: int = 0
    total_bytes_out: int = 0
    blocked_flows: int = 0
    sensitive_flows: int = 0  # flows with sensitivity > threshold

    # By type
    flows_by_type: Dict[str, int] = field(default_factory=dict)

    # Risk metrics
    avg_sensitivity: float = 0.0
    max_sensitivity: float = 0.0
    pii_exposure_count: int = 0
    credential_exposure_count: int = 0

    # Top sources/destinations
    top_sources: List[tuple] = field(default_factory=list)
    top_destinations: List[tuple] = field(default_factory=list)


class FlowTracker:
    """Tracks and analyzes information flows across the sandbox boundary.

    Provides capabilities to:
    - Record all boundary crossings
    - Analyze flow patterns
    - Detect anomalies
    - Generate reports
    """

    def __init__(
        self,
        sensitivity_threshold: float = 0.5,
        max_history: int = 10000,
    ):
        """Initialize the flow tracker.

        Args:
            sensitivity_threshold: Threshold for flagging sensitive flows
            max_history: Maximum number of flows to keep in history
        """
        self.sensitivity_threshold = sensitivity_threshold
        self.max_history = max_history

        self.flows: List[InformationFlow] = []
        self.blocked_flows: List[InformationFlow] = []

        # Counters
        self._total_bytes_in = 0
        self._total_bytes_out = 0

        # Pattern tracking
        self._source_counts: Dict[str, int] = {}
        self._destination_counts: Dict[str, int] = {}
        self._type_counts: Dict[str, int] = {}

    def record_flow(self, flow: InformationFlow) -> None:
        """Record an information flow event."""
        # Update counters
        if flow.direction == FlowDirection.INBOUND:
            self._total_bytes_in += flow.size_bytes
        elif flow.direction == FlowDirection.OUTBOUND:
            self._total_bytes_out += flow.size_bytes
        else:
            self._total_bytes_in += flow.size_bytes // 2
            self._total_bytes_out += flow.size_bytes // 2

        self._source_counts[flow.source_id] = (
            self._source_counts.get(flow.source_id, 0) + 1
        )
        self._destination_counts[flow.destination_id] = (
            self._destination_counts.get(flow.destination_id, 0) + 1
        )
        self._type_counts[flow.flow_type.value] = (
            self._type_counts.get(flow.flow_type.value, 0) + 1
        )

        # Store flow
        if flow.blocked:
            self.blocked_flows.append(flow)
        else:
            self.flows.append(flow)

        # Trim history if needed
        if len(self.flows) > self.max_history:
            self.flows = self.flows[-self.max_history :]
        if len(self.blocked_flows) > self.max_history:
            self.blocked_flows = self.blocked_flows[-self.max_history :]

    def get_flows(
        self,
        direction: Optional[FlowDirection] = None,
        flow_type: Optional[FlowType] = None,
        source_id: Optional[str] = None,
        destination_id: Optional[str] = None,
        min_sensitivity: float = 0.0,
        include_blocked: bool = False,
    ) -> List[InformationFlow]:
        """Query flows matching criteria."""
        flows = self.flows.copy()
        if include_blocked:
            flows.extend(self.blocked_flows)

        if direction is not None:
            flows = [f for f in flows if f.direction == direction]
        if flow_type is not None:
            flows = [f for f in flows if f.flow_type == flow_type]
        if source_id is not None:
            flows = [f for f in flows if f.source_id == source_id]
        if destination_id is not None:
            flows = [f for f in flows if f.destination_id == destination_id]
        if min_sensitivity > 0:
            flows = [f for f in flows if f.sensitivity_score >= min_sensitivity]

        return flows

    def get_summary(self) -> FlowSummary:
        """Generate a summary of all flows."""
        all_flows = self.flows + self.blocked_flows

        if not all_flows:
            return FlowSummary()

        inbound = [f for f in all_flows if f.direction == FlowDirection.INBOUND]
        outbound = [f for f in all_flows if f.direction == FlowDirection.OUTBOUND]
        blocked = [f for f in all_flows if f.blocked]
        sensitive = [
            f for f in all_flows if f.sensitivity_score >= self.sensitivity_threshold
        ]

        sensitivities = [f.sensitivity_score for f in all_flows]
        pii_count = sum(1 for f in all_flows if f.contains_pii)
        cred_count = sum(1 for f in all_flows if f.contains_credentials)

        # Top sources/destinations
        top_sources = sorted(
            self._source_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_destinations = sorted(
            self._destination_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return FlowSummary(
            total_flows=len(all_flows),
            inbound_flows=len(inbound),
            outbound_flows=len(outbound),
            total_bytes_in=self._total_bytes_in,
            total_bytes_out=self._total_bytes_out,
            blocked_flows=len(blocked),
            sensitive_flows=len(sensitive),
            flows_by_type=self._type_counts.copy(),
            avg_sensitivity=sum(sensitivities) / len(sensitivities),
            max_sensitivity=max(sensitivities),
            pii_exposure_count=pii_count,
            credential_exposure_count=cred_count,
            top_sources=top_sources,
            top_destinations=top_destinations,
        )

    def get_agent_flows(self, agent_id: str) -> Dict[str, Any]:
        """Get flow statistics for a specific agent."""
        as_source = self.get_flows(source_id=agent_id)
        as_dest = self.get_flows(destination_id=agent_id)

        return {
            "agent_id": agent_id,
            "flows_initiated": len(as_source),
            "flows_received": len(as_dest),
            "bytes_sent": sum(f.size_bytes for f in as_source),
            "bytes_received": sum(f.size_bytes for f in as_dest),
            "avg_sensitivity_sent": (
                sum(f.sensitivity_score for f in as_source) / len(as_source)
                if as_source
                else 0.0
            ),
            "blocked_attempts": sum(1 for f in as_source if f.blocked),
        }

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous flow patterns."""
        anomalies = []

        summary = self.get_summary()

        # Check for excessive outbound data
        if self._total_bytes_out > self._total_bytes_in * 10:
            anomalies.append(
                {
                    "type": "excessive_outbound",
                    "severity": "high",
                    "description": "Outbound data significantly exceeds inbound",
                    "ratio": self._total_bytes_out / max(self._total_bytes_in, 1),
                }
            )

        # Check for high sensitivity flows
        if summary.sensitive_flows > summary.total_flows * 0.3:
            anomalies.append(
                {
                    "type": "high_sensitivity_ratio",
                    "severity": "medium",
                    "description": "High proportion of sensitive data flows",
                    "ratio": summary.sensitive_flows / max(summary.total_flows, 1),
                }
            )

        # Check for credential exposure
        if summary.credential_exposure_count > 0:
            anomalies.append(
                {
                    "type": "credential_exposure",
                    "severity": "critical",
                    "description": "Credentials detected in flows",
                    "count": summary.credential_exposure_count,
                }
            )

        # Check for concentrated sources
        if summary.top_sources:
            top_source_count = summary.top_sources[0][1]
            if top_source_count > summary.total_flows * 0.5:
                anomalies.append(
                    {
                        "type": "concentrated_source",
                        "severity": "low",
                        "description": "Single source dominates flows",
                        "source": summary.top_sources[0][0],
                        "ratio": top_source_count / max(summary.total_flows, 1),
                    }
                )

        return anomalies

    def reset(self) -> None:
        """Reset all tracking data."""
        self.flows.clear()
        self.blocked_flows.clear()
        self._total_bytes_in = 0
        self._total_bytes_out = 0
        self._source_counts.clear()
        self._destination_counts.clear()
        self._type_counts.clear()
