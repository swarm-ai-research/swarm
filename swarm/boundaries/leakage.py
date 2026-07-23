"""Leakage detection for sandbox boundaries.

This module detects potential data leakage through the sandbox boundary,
including sensitive data exposure, credential leaks, and information exfiltration.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Set


class LeakageType(Enum):
    """Types of data leakage."""

    PII = "pii"  # Personally identifiable information
    CREDENTIAL = "credential"  # Passwords, API keys, tokens
    INTERNAL_DATA = "internal_data"  # Internal system data
    CODE = "code"  # Source code or proprietary algorithms
    CONFIGURATION = "configuration"  # System configuration
    FINANCIAL = "financial"  # Financial data
    HEALTH = "health"  # Health information
    UNKNOWN = "unknown"  # Unclassified sensitive data


@dataclass
class LeakageEvent:
    """Represents a detected leakage event."""

    event_id: str
    timestamp: datetime
    leakage_type: LeakageType
    severity: float  # 0 = low, 1 = critical
    agent_id: str
    destination_id: str
    flow_id: Optional[str] = None
    description: str = ""
    content_hash: str = ""  # Hash of leaked content
    data_categories: List[str] = field(default_factory=list)
    blocked: bool = False
    remediation_action: Optional[str] = None

    @property
    def severity_label(self) -> str:
        """Get human-readable severity label."""
        if self.severity >= 0.9:
            return "critical"
        elif self.severity >= 0.7:
            return "high"
        elif self.severity >= 0.4:
            return "medium"
        else:
            return "low"


@dataclass
class LeakageReport:
    """Summary report of leakage events."""

    total_events: int = 0
    blocked_count: int = 0
    unblocked_count: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    top_agents: List[tuple] = field(default_factory=list)
    top_destinations: List[tuple] = field(default_factory=list)
    avg_severity: float = 0.0
    max_severity: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def add_recommendation(self, recommendation: str) -> None:
        """Add a security recommendation."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)


class LeakageDetector:
    """Detects potential data leakage across sandbox boundaries.

    Uses pattern matching, heuristics, and tracking to identify
    sensitive data leaving the sandbox.
    """

    # Default patterns for sensitive data detection
    DEFAULT_PII_PATTERNS = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone number
        r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card (basic)
    ]

    DEFAULT_CREDENTIAL_PATTERNS = [
        r"password\s*[:=]\s*['\"]?[\w!@#$%^&*]+['\"]?",
        r"api[_-]?key\s*[:=]\s*['\"]?[\w-]+['\"]?",
        r"secret\s*[:=]\s*['\"]?[\w-]+['\"]?",
        r"token\s*[:=]\s*['\"]?[\w-]+['\"]?",
        r"bearer\s+[\w-]+",
        r"aws[_-]?access[_-]?key[_-]?id\s*[:=]\s*[\w]+",
        r"private[_-]?key",
    ]

    DEFAULT_CODE_PATTERNS = [
        r"def\s+\w+\s*\([^)]*\)\s*:",  # Python function
        r"function\s+\w+\s*\([^)]*\)\s*\{",  # JavaScript function
        r"class\s+\w+\s*[:\{]",  # Class definition
        r"import\s+[\w.]+",  # Import statement
    ]

    def __init__(
        self,
        pii_patterns: Optional[List[str]] = None,
        credential_patterns: Optional[List[str]] = None,
        code_patterns: Optional[List[str]] = None,
        custom_patterns: Optional[Dict[LeakageType, List[str]]] = None,
        sensitive_keywords: Optional[Set[str]] = None,
        track_hashes: bool = True,
    ):
        """Initialize the leakage detector.

        Args:
            pii_patterns: Regex patterns for PII detection
            credential_patterns: Regex patterns for credential detection
            code_patterns: Regex patterns for code detection
            custom_patterns: Additional patterns by leakage type
            sensitive_keywords: Keywords indicating sensitive content
            track_hashes: Whether to track content hashes for event attribution
        """
        self.pii_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (pii_patterns or self.DEFAULT_PII_PATTERNS)
        ]
        self.credential_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (credential_patterns or self.DEFAULT_CREDENTIAL_PATTERNS)
        ]
        self.code_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (code_patterns or self.DEFAULT_CODE_PATTERNS)
        ]

        self.custom_patterns: Dict[LeakageType, List[Pattern]] = {}
        if custom_patterns:
            for leak_type, patterns in custom_patterns.items():
                self.custom_patterns[leak_type] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]

        self.sensitive_keywords = sensitive_keywords or {
            "confidential",
            "secret",
            "private",
            "internal",
            "restricted",
            "sensitive",
            "proprietary",
        }

        self.track_hashes = track_hashes
        self.seen_hashes: Set[str] = set()

        # Event tracking
        self.events: List[LeakageEvent] = []
        self.agent_event_counts: Dict[str, int] = {}
        self.destination_event_counts: Dict[str, int] = {}

    def scan(
        self,
        content: Any,
        agent_id: str,
        destination_id: str,
        flow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[LeakageEvent]:
        """Scan content for potential leakage.

        Args:
            content: Content to scan
            agent_id: Agent sending the data
            destination_id: Destination of the data
            flow_id: Associated flow ID
            metadata: Additional context

        Returns:
            List of detected leakage events
        """
        content_str = str(content)
        detected_events = []

        # Compute content hash for event attribution
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

        # Scan for each leakage type
        pii_matches = self._scan_patterns(content_str, self.pii_patterns)
        if pii_matches:
            event = self._create_event(
                LeakageType.PII,
                agent_id,
                destination_id,
                flow_id,
                content_hash,
                f"PII detected: {len(pii_matches)} matches",
                severity=0.7,
                data_categories=["email", "phone", "ssn", "credit_card"],
            )
            detected_events.append(event)

        credential_matches = self._scan_patterns(content_str, self.credential_patterns)
        if credential_matches:
            event = self._create_event(
                LeakageType.CREDENTIAL,
                agent_id,
                destination_id,
                flow_id,
                content_hash,
                f"Credentials detected: {len(credential_matches)} matches",
                severity=0.95,
                data_categories=["password", "api_key", "token"],
            )
            detected_events.append(event)

        code_matches = self._scan_patterns(content_str, self.code_patterns)
        if code_matches:
            event = self._create_event(
                LeakageType.CODE,
                agent_id,
                destination_id,
                flow_id,
                content_hash,
                f"Code detected: {len(code_matches)} matches",
                severity=0.5,
                data_categories=["source_code"],
            )
            detected_events.append(event)

        # Check custom patterns
        for leak_type, patterns in self.custom_patterns.items():
            matches = self._scan_patterns(content_str, patterns)
            if matches:
                event = self._create_event(
                    leak_type,
                    agent_id,
                    destination_id,
                    flow_id,
                    content_hash,
                    f"{leak_type.value} detected: {len(matches)} matches",
                    severity=0.6,
                )
                detected_events.append(event)

        # Check keywords
        keyword_matches = self._scan_keywords(content_str)
        if keyword_matches and not detected_events:
            # Only add keyword event if no other detections
            event = self._create_event(
                LeakageType.UNKNOWN,
                agent_id,
                destination_id,
                flow_id,
                content_hash,
                f"Sensitive keywords detected: {', '.join(keyword_matches)}",
                severity=0.3,
                data_categories=list(keyword_matches),
            )
            detected_events.append(event)

        # Record events
        for event in detected_events:
            self._record_event(event)

        return detected_events

    def _scan_patterns(self, content: str, patterns: List[Pattern]) -> List[str]:
        """Scan content for pattern matches."""
        matches = []
        for pattern in patterns:
            found = pattern.findall(content)
            matches.extend(found)
        return matches

    def _scan_keywords(self, content: str) -> Set[str]:
        """Scan content for sensitive keywords."""
        content_lower = content.lower()
        found = set()
        for keyword in self.sensitive_keywords:
            if keyword.lower() in content_lower:
                found.add(keyword)
        return found

    def _create_event(
        self,
        leakage_type: LeakageType,
        agent_id: str,
        destination_id: str,
        flow_id: Optional[str],
        content_hash: str,
        description: str,
        severity: float,
        data_categories: Optional[List[str]] = None,
    ) -> LeakageEvent:
        """Create a leakage event."""
        event_id = f"leak_{content_hash}_{datetime.now().timestamp()}"
        return LeakageEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            leakage_type=leakage_type,
            severity=severity,
            agent_id=agent_id,
            destination_id=destination_id,
            flow_id=flow_id,
            description=description,
            content_hash=content_hash,
            data_categories=data_categories or [],
        )

    def _record_event(self, event: LeakageEvent) -> None:
        """Record a leakage event."""
        self.events.append(event)
        self.agent_event_counts[event.agent_id] = (
            self.agent_event_counts.get(event.agent_id, 0) + 1
        )
        self.destination_event_counts[event.destination_id] = (
            self.destination_event_counts.get(event.destination_id, 0) + 1
        )

    def get_events(
        self,
        leakage_type: Optional[LeakageType] = None,
        agent_id: Optional[str] = None,
        min_severity: float = 0.0,
        blocked_only: bool = False,
    ) -> List[LeakageEvent]:
        """Query leakage events."""
        events = self.events

        if leakage_type is not None:
            events = [e for e in events if e.leakage_type == leakage_type]
        if agent_id is not None:
            events = [e for e in events if e.agent_id == agent_id]
        if min_severity > 0:
            events = [e for e in events if e.severity >= min_severity]
        if blocked_only:
            events = [e for e in events if e.blocked]

        return events

    def generate_report(self) -> LeakageReport:
        """Generate a comprehensive leakage report."""
        if not self.events:
            return LeakageReport()

        # Count by type
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        total_severity = 0.0
        max_severity = 0.0
        blocked = 0
        unblocked = 0

        for event in self.events:
            type_key = event.leakage_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            severity_key = event.severity_label
            by_severity[severity_key] = by_severity.get(severity_key, 0) + 1

            total_severity += event.severity
            max_severity = max(max_severity, event.severity)

            if event.blocked:
                blocked += 1
            else:
                unblocked += 1

        # Top agents and destinations
        top_agents = sorted(
            self.agent_event_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_destinations = sorted(
            self.destination_event_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        report = LeakageReport(
            total_events=len(self.events),
            blocked_count=blocked,
            unblocked_count=unblocked,
            events_by_type=by_type,
            events_by_severity=by_severity,
            top_agents=top_agents,
            top_destinations=top_destinations,
            avg_severity=total_severity / len(self.events),
            max_severity=max_severity,
        )

        # Generate recommendations
        if by_type.get("credential", 0) > 0:
            report.add_recommendation(
                "Implement credential rotation and avoid hardcoded secrets"
            )
        if by_type.get("pii", 0) > 0:
            report.add_recommendation("Review PII handling and implement data masking")
        if unblocked > blocked:
            report.add_recommendation("Consider stricter boundary policies")
        if max_severity >= 0.9:
            report.add_recommendation(
                "Investigate critical severity events immediately"
            )
        if top_agents and top_agents[0][1] > len(self.events) * 0.5:
            report.add_recommendation(
                f"Review agent {top_agents[0][0]} for potential compromise"
            )

        return report

    def reset(self) -> None:
        """Reset detector state."""
        self.events.clear()
        self.agent_event_counts.clear()
        self.destination_event_counts.clear()
        self.seen_hashes.clear()
