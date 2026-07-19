"""Security threat detection metrics and algorithms.

Detects and measures security threats in multi-agent systems:
- Prompt injection attempts
- Cross-agent manipulation
- Information laundering
- Contagion/worm propagation
- Exfiltration patterns
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from swarm.models.interaction import SoftInteraction


class ThreatType(Enum):
    """Types of security threats."""

    PROMPT_INJECTION = "prompt_injection"
    CROSS_AGENT_MANIPULATION = "cross_agent_manipulation"
    INFORMATION_LAUNDERING = "information_laundering"
    CONTAGION = "contagion"
    EXFILTRATION = "exfiltration"
    CREDENTIAL_LEAKAGE = "credential_leakage"


@dataclass
class ThreatIndicator:
    """An indicator of a potential security threat."""

    threat_type: ThreatType
    severity: float  # 0-1, higher = more severe
    source_agent: str
    target_agents: Set[str] = field(default_factory=set)
    interaction_ids: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0  # 0-1, confidence in detection
    timestamp: Optional[str] = None

    @property
    def risk_score(self) -> float:
        """Compute risk score from severity and confidence."""
        return self.severity * self.confidence


@dataclass
class ContagionChain:
    """Represents a chain of potentially malicious content propagation."""

    chain_id: str
    origin_agent: str
    propagation_path: List[str] = field(default_factory=list)
    interaction_ids: List[str] = field(default_factory=list)
    depth: int = 0
    spread: int = 0  # Number of unique agents reached
    velocity: float = 0.0  # Interactions per time unit
    contained: bool = False


@dataclass
class ManipulationPattern:
    """Pattern of cross-agent manipulation."""

    manipulator: str
    victims: Set[str] = field(default_factory=set)
    technique: str = ""  # e.g., "reputation_boosting", "vote_manipulation"
    interactions: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    influence_delta: float = 0.0  # Change in manipulator's influence


@dataclass
class LaunderingPath:
    """Information laundering path through trusted intermediaries."""

    source: str
    destination: str
    intermediaries: List[str] = field(default_factory=list)
    trust_at_origin: float = 0.0
    trust_at_destination: float = 0.0
    trust_gain: float = 0.0  # How much trust increased along path
    interaction_ids: List[str] = field(default_factory=list)


@dataclass
class SecurityReport:
    """Complete security analysis report."""

    # Overall risk metrics
    ecosystem_threat_level: float = 0.0  # 0-1 risk score
    active_threat_count: int = 0

    # Threat breakdowns
    threat_indicators: List[ThreatIndicator] = field(default_factory=list)
    contagion_chains: List[ContagionChain] = field(default_factory=list)
    manipulation_patterns: List[ManipulationPattern] = field(default_factory=list)
    laundering_paths: List[LaunderingPath] = field(default_factory=list)

    # Per-type metrics
    prompt_injection_rate: float = 0.0
    manipulation_rate: float = 0.0
    laundering_rate: float = 0.0
    contagion_depth: int = 0
    exfiltration_attempts: int = 0

    # Containment metrics
    containment_success_rate: float = 0.0
    false_positive_rate: float = 0.0

    # Per-agent risk scores
    agent_threat_scores: Dict[str, float] = field(default_factory=dict)
    agents_flagged: Set[str] = field(default_factory=set)


class SecurityAnalyzer:
    """
    Analyzes security threats in multi-agent interactions.

    Detects:
    1. Prompt injection attempts - suspicious patterns suggesting command injection
    2. Cross-agent manipulation - agents exploiting others' trust or behavior
    3. Information laundering - washing low-quality/harmful info through trusted agents
    4. Contagion propagation - self-replicating or spreading malicious patterns
    5. Exfiltration - attempts to extract sensitive information
    """

    # Common injection patterns (simplified heuristics)
    INJECTION_PATTERNS = [
        "ignore previous",
        "disregard instructions",
        "new task:",
        "system:",
        "override",
        "admin mode",
        "jailbreak",
        "{{",  # Template injection
        "eval(",
        "exec(",
    ]

    # Suspicious content patterns for exfiltration
    EXFIL_PATTERNS = [
        "api_key",
        "password",
        "secret",
        "credential",
        "token",
        "private_key",
    ]

    def __init__(
        self,
        injection_threshold: float = 0.3,
        manipulation_threshold: float = 0.5,
        laundering_trust_gap: float = 0.3,
        contagion_velocity_threshold: float = 2.0,
        min_chain_length: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize security analyzer.

        Args:
            injection_threshold: Threshold for flagging injection attempts
            manipulation_threshold: Threshold for manipulation detection
            laundering_trust_gap: Min trust increase for laundering flag
            contagion_velocity_threshold: Interactions/time for contagion flag
            min_chain_length: Minimum chain length to track
            seed: Random seed for any stochastic analysis
        """
        self.injection_threshold = injection_threshold
        self.manipulation_threshold = manipulation_threshold
        self.laundering_trust_gap = laundering_trust_gap
        self.contagion_velocity_threshold = contagion_velocity_threshold
        self.min_chain_length = min_chain_length

        # State tracking
        self._interaction_history: List[SoftInteraction] = []
        self._agent_trust_scores: Dict[str, float] = {}
        self._content_hashes: Dict[str, List[str]] = defaultdict(
            list
        )  # hash -> agent_ids
        self._propagation_graph: Dict[str, Set[str]] = defaultdict(
            set
        )  # content -> recipients

    def set_agent_trust_scores(self, trust_scores: Dict[str, float]) -> None:
        """Set trust/reputation scores for agents."""
        self._agent_trust_scores = trust_scores.copy()

    def record_interaction(self, interaction: SoftInteraction) -> None:
        """Record an interaction for analysis."""
        self._interaction_history.append(interaction)

    def record_content_propagation(
        self,
        content_hash: str,
        from_agent: str,
        to_agent: str,
    ) -> None:
        """Record content propagating between agents."""
        self._content_hashes[content_hash].append(from_agent)
        self._propagation_graph[content_hash].add(to_agent)

    def analyze(
        self,
        interactions: Optional[List[SoftInteraction]] = None,
        agent_ids: Optional[List[str]] = None,
    ) -> SecurityReport:
        """
        Perform comprehensive security analysis.

        Args:
            interactions: List of interactions (uses history if None)
            agent_ids: List of all agent IDs

        Returns:
            SecurityReport with detailed findings
        """
        if interactions is None:
            interactions = self._interaction_history

        if not interactions:
            return SecurityReport()

        if agent_ids is None:
            agent_ids = list(
                {i.initiator for i in interactions}
                | {i.counterparty for i in interactions}
            )

        report = SecurityReport()

        # Detect each threat type
        injection_indicators = self._detect_prompt_injection(interactions)
        manipulation_patterns = self._detect_manipulation(interactions, agent_ids)
        laundering_paths = self._detect_laundering(interactions, agent_ids)
        contagion_chains = self._detect_contagion(interactions, agent_ids)
        exfil_indicators = self._detect_exfiltration(interactions)

        # Aggregate indicators
        all_indicators = (
            injection_indicators
            + exfil_indicators
            + [self._pattern_to_indicator(m) for m in manipulation_patterns]
            + [self._laundering_to_indicator(lp) for lp in laundering_paths]
            + [self._contagion_to_indicator(c) for c in contagion_chains]
        )

        report.threat_indicators = all_indicators
        report.manipulation_patterns = manipulation_patterns
        report.laundering_paths = laundering_paths
        report.contagion_chains = contagion_chains

        # Compute rates
        n_interactions = len(interactions)
        report.prompt_injection_rate = (
            len(injection_indicators) / n_interactions if n_interactions > 0 else 0.0
        )
        report.manipulation_rate = len(manipulation_patterns) / max(1, len(agent_ids))
        report.laundering_rate = (
            len(laundering_paths) / n_interactions if n_interactions > 0 else 0.0
        )
        report.contagion_depth = max((c.depth for c in contagion_chains), default=0)
        report.exfiltration_attempts = len(exfil_indicators)

        # Compute per-agent risk scores
        report.agent_threat_scores = self._compute_agent_threat_scores(
            all_indicators, agent_ids
        )
        report.agents_flagged = {
            agent_id
            for agent_id, score in report.agent_threat_scores.items()
            if score >= self.manipulation_threshold
        }

        # Compute ecosystem threat level
        report.active_threat_count = len(all_indicators)
        report.ecosystem_threat_level = self._compute_ecosystem_threat(
            report, len(agent_ids)
        )

        return report

    def _detect_prompt_injection(
        self,
        interactions: List[SoftInteraction],
    ) -> List[ThreatIndicator]:
        """Detect potential prompt injection attempts."""
        indicators = []

        for interaction in interactions:
            injection_score = self._compute_injection_score(interaction)

            if injection_score >= self.injection_threshold:
                indicators.append(
                    ThreatIndicator(
                        threat_type=ThreatType.PROMPT_INJECTION,
                        severity=injection_score,
                        source_agent=interaction.initiator,
                        target_agents={interaction.counterparty},
                        interaction_ids=[interaction.interaction_id],
                        description="Potential prompt injection pattern detected",
                        confidence=min(0.9, injection_score + 0.2),
                        timestamp=interaction.timestamp.isoformat(),
                    )
                )

        return indicators

    def _compute_injection_score(self, interaction: SoftInteraction) -> float:
        """
        Compute injection risk score for an interaction.

        Uses multiple signals:
        - Low p with high acceptance (counterparty accepted suspicious content)
        - Tool misuse flags
        - Verifier rejections
        """
        score = 0.0

        # Low quality but accepted = suspicious
        if interaction.accepted and interaction.p < 0.3:
            score += 0.3 * (0.3 - interaction.p) / 0.3

        # Tool misuse is a strong signal
        if interaction.tool_misuse_flags > 0:
            score += min(0.4, 0.2 * interaction.tool_misuse_flags)

        # Verifier rejections indicate detected issues
        if interaction.verifier_rejections > 0:
            score += min(0.3, 0.15 * interaction.verifier_rejections)

        return min(1.0, score)

    def _detect_manipulation(
        self,
        interactions: List[SoftInteraction],
        agent_ids: List[str],
    ) -> List[ManipulationPattern]:
        """Detect cross-agent manipulation patterns."""
        patterns = []

        # Group interactions by initiator
        by_initiator: Dict[str, List[SoftInteraction]] = defaultdict(list)
        for i in interactions:
            by_initiator[i.initiator].append(i)

        for initiator, init_interactions in by_initiator.items():
            pattern = self._analyze_manipulation_pattern(
                initiator, init_interactions, interactions, agent_ids
            )
            if pattern and pattern.success_rate >= self.manipulation_threshold:
                patterns.append(pattern)

        return patterns

    def _analyze_manipulation_pattern(
        self,
        initiator: str,
        init_interactions: List[SoftInteraction],
        all_interactions: List[SoftInteraction],
        agent_ids: List[str],
    ) -> Optional[ManipulationPattern]:
        """Analyze if an agent shows manipulation patterns."""
        if len(init_interactions) < 3:
            return None

        victims = set()
        successful = 0
        techniques = []

        for i in init_interactions:
            # Check for reputation manipulation (low p, but positive reputation)
            if i.accepted and i.p < 0.4 and i.r_a > 0:
                victims.add(i.counterparty)
                successful += 1
                techniques.append("reputation_boosting")

            # Check for resource extraction (large positive tau for initiator)
            if i.accepted and i.tau < -1.0 and i.p < 0.5:
                victims.add(i.counterparty)
                successful += 1
                techniques.append("resource_extraction")

            # Check for vote manipulation (targeting specific counterparties)
            counterparty_count = sum(
                1 for j in init_interactions if j.counterparty == i.counterparty
            )
            if counterparty_count >= 3 and i.p < 0.5:
                victims.add(i.counterparty)
                techniques.append("targeted_influence")

        if not victims:
            return None

        success_rate = successful / len(init_interactions)
        most_common_technique = (
            max(set(techniques), key=techniques.count) if techniques else ""
        )

        return ManipulationPattern(
            manipulator=initiator,
            victims=victims,
            technique=most_common_technique,
            interactions=[i.interaction_id for i in init_interactions],
            success_rate=success_rate,
            influence_delta=sum(i.r_a for i in init_interactions),
        )

    def _detect_laundering(
        self,
        interactions: List[SoftInteraction],
        agent_ids: List[str],
    ) -> List[LaunderingPath]:
        """
        Detect information laundering patterns.

        Information laundering: low-trust agent passes info to medium-trust agent
        who passes it to high-trust destination, gaining credibility.
        """
        paths = []

        # Build interaction graph
        edges: Dict[str, Dict[str, List[SoftInteraction]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for i in interactions:
            if i.accepted:
                edges[i.initiator][i.counterparty].append(i)

        # Look for paths where trust increases
        for source in agent_ids:
            source_trust = self._agent_trust_scores.get(source, 0.5)

            # BFS to find laundering paths
            visited = {source}
            queue = [(source, [source], source_trust)]

            while queue:
                current, path, start_trust = queue.pop(0)

                for next_agent, agent_interactions in edges[current].items():
                    if next_agent in visited:
                        continue

                    next_trust = self._agent_trust_scores.get(next_agent, 0.5)

                    # Check if this represents laundering
                    if next_trust > start_trust + self.laundering_trust_gap:
                        new_path = path + [next_agent]
                        if len(new_path) >= self.min_chain_length:
                            paths.append(
                                LaunderingPath(
                                    source=source,
                                    destination=next_agent,
                                    intermediaries=new_path[1:-1],
                                    trust_at_origin=start_trust,
                                    trust_at_destination=next_trust,
                                    trust_gain=next_trust - start_trust,
                                    interaction_ids=[
                                        i.interaction_id for i in agent_interactions
                                    ],
                                )
                            )

                    visited.add(next_agent)
                    queue.append((next_agent, path + [next_agent], start_trust))

        return paths

    def _detect_contagion(
        self,
        interactions: List[SoftInteraction],
        agent_ids: List[str],
    ) -> List[ContagionChain]:
        """
        Detect potential contagion/worm propagation patterns.

        Looks for:
        - Rapidly spreading patterns
        - Self-replicating behavior (agent receives, then sends similar)
        - Cascading low-quality interactions
        """
        chains: List[ContagionChain] = []

        # Sort by timestamp
        sorted_ints = sorted(interactions, key=lambda x: x.timestamp)
        if len(sorted_ints) < self.min_chain_length:
            return chains

        # Track low-quality content spread
        infected: Dict[str, List[Tuple[SoftInteraction, int]]] = defaultdict(list)

        for idx, interaction in enumerate(sorted_ints):
            # Low quality accepted = potential infection
            if interaction.accepted and interaction.p < 0.3:
                infected[interaction.counterparty].append((interaction, idx))

                # Check if this agent subsequently "spreads"
                subsequent = [
                    i
                    for i in sorted_ints[idx + 1 :]
                    if i.initiator == interaction.counterparty
                    and i.accepted
                    and i.p < 0.3
                ]

                if len(subsequent) >= 2:
                    # Compute chain
                    chain_agents = [interaction.initiator, interaction.counterparty]
                    chain_ints = [interaction.interaction_id]

                    for sub in subsequent[:5]:  # Limit chain length
                        if sub.counterparty not in chain_agents:
                            chain_agents.append(sub.counterparty)
                            chain_ints.append(sub.interaction_id)

                    if len(chain_agents) >= self.min_chain_length:
                        # Compute velocity
                        time_span = (
                            subsequent[-1].timestamp - interaction.timestamp
                        ).total_seconds()
                        velocity = len(chain_ints) / (
                            time_span / 60.0 + 0.01
                        )  # per minute

                        if velocity >= self.contagion_velocity_threshold:
                            chains.append(
                                ContagionChain(
                                    chain_id=f"chain_{len(chains)}",
                                    origin_agent=interaction.initiator,
                                    propagation_path=chain_agents,
                                    interaction_ids=chain_ints,
                                    depth=len(chain_agents) - 1,
                                    spread=len(set(chain_agents)),
                                    velocity=velocity,
                                    contained=False,
                                )
                            )

        # Deduplicate overlapping chains
        return self._deduplicate_chains(chains)

    def _deduplicate_chains(self, chains: List[ContagionChain]) -> List[ContagionChain]:
        """Remove chains that are subsets of larger chains."""
        if len(chains) <= 1:
            return chains

        # Sort by depth descending
        sorted_chains = sorted(chains, key=lambda c: c.depth, reverse=True)
        unique_chains: List[ContagionChain] = []

        for chain in sorted_chains:
            chain_set = set(chain.propagation_path)
            is_subset = False
            for existing in unique_chains:
                if chain_set <= set(existing.propagation_path):
                    is_subset = True
                    break
            if not is_subset:
                unique_chains.append(chain)

        return unique_chains

    def _detect_exfiltration(
        self,
        interactions: List[SoftInteraction],
    ) -> List[ThreatIndicator]:
        """Detect potential data exfiltration attempts."""
        indicators = []

        for interaction in interactions:
            # Check for suspicious patterns in tool misuse
            if interaction.tool_misuse_flags >= 2:
                indicators.append(
                    ThreatIndicator(
                        threat_type=ThreatType.EXFILTRATION,
                        severity=min(1.0, 0.3 * interaction.tool_misuse_flags),
                        source_agent=interaction.initiator,
                        target_agents={interaction.counterparty},
                        interaction_ids=[interaction.interaction_id],
                        description="Potential exfiltration via tool misuse",
                        confidence=0.6,
                        timestamp=interaction.timestamp.isoformat(),
                    )
                )

        return indicators

    def _pattern_to_indicator(self, pattern: ManipulationPattern) -> ThreatIndicator:
        """Convert manipulation pattern to threat indicator."""
        return ThreatIndicator(
            threat_type=ThreatType.CROSS_AGENT_MANIPULATION,
            severity=pattern.success_rate,
            source_agent=pattern.manipulator,
            target_agents=pattern.victims,
            interaction_ids=pattern.interactions,
            description=f"Manipulation via {pattern.technique}",
            confidence=min(0.9, 0.5 + pattern.success_rate * 0.4),
        )

    def _laundering_to_indicator(self, path: LaunderingPath) -> ThreatIndicator:
        """Convert laundering path to threat indicator."""
        return ThreatIndicator(
            threat_type=ThreatType.INFORMATION_LAUNDERING,
            severity=min(1.0, path.trust_gain / 0.5),
            source_agent=path.source,
            target_agents={path.destination},
            interaction_ids=path.interaction_ids,
            description=f"Laundering via {len(path.intermediaries)} intermediaries",
            confidence=min(0.8, 0.4 + path.trust_gain),
        )

    def _contagion_to_indicator(self, chain: ContagionChain) -> ThreatIndicator:
        """Convert contagion chain to threat indicator."""
        return ThreatIndicator(
            threat_type=ThreatType.CONTAGION,
            severity=min(1.0, chain.depth / 5.0),
            source_agent=chain.origin_agent,
            target_agents=set(chain.propagation_path[1:]),
            interaction_ids=chain.interaction_ids,
            description=f"Contagion chain depth={chain.depth}, velocity={chain.velocity:.2f}",
            confidence=min(0.9, 0.5 + chain.velocity / 10.0),
        )

    def _compute_agent_threat_scores(
        self,
        indicators: List[ThreatIndicator],
        agent_ids: List[str],
    ) -> Dict[str, float]:
        """Compute per-agent threat scores."""
        scores: Dict[str, float] = dict.fromkeys(agent_ids, 0.0)

        for indicator in indicators:
            # Source agent gets full risk score
            if indicator.source_agent in scores:
                scores[indicator.source_agent] = max(
                    scores[indicator.source_agent],
                    indicator.risk_score,
                )

            # Target agents get partial score (they're victims but may propagate)
            for target in indicator.target_agents:
                if target in scores:
                    scores[target] = max(
                        scores[target],
                        indicator.risk_score * 0.3,
                    )

        return scores

    def _compute_ecosystem_threat(
        self,
        report: SecurityReport,
        n_agents: int,
    ) -> float:
        """Compute overall ecosystem threat level."""
        if n_agents == 0:
            return 0.0

        # Component scores
        threat_density = len(report.threat_indicators) / max(1, n_agents * 5)
        flagged_fraction = len(report.agents_flagged) / n_agents
        contagion_severity = min(1.0, report.contagion_depth / 5.0)
        manipulation_severity = report.manipulation_rate

        # Weighted combination
        threat_level = (
            0.25 * threat_density
            + 0.25 * flagged_fraction
            + 0.30 * contagion_severity
            + 0.20 * manipulation_severity
        )

        return float(min(1.0, threat_level))

    def clear_history(self) -> None:
        """Clear interaction history."""
        self._interaction_history.clear()
        self._content_hashes.clear()
        self._propagation_graph.clear()


def compute_containment_effectiveness(
    chains: List[ContagionChain],
    containment_actions: List[Dict],
) -> float:
    """
    Compute how effectively contagion was contained.

    Args:
        chains: Detected contagion chains
        containment_actions: Actions taken (e.g., freezes, blocks)

    Returns:
        Effectiveness score 0-1
    """
    if not chains:
        return 1.0  # Nothing to contain

    contained_count = sum(1 for c in chains if c.contained)
    return contained_count / len(chains)


def compute_threat_trend(
    reports: List[SecurityReport],
    window: int = 5,
) -> Dict[str, float]:
    """
    Compute threat trends over recent reports.

    Args:
        reports: List of security reports (chronological order)
        window: Number of recent reports to analyze

    Returns:
        Dict with trend metrics (positive = increasing threats)
    """
    if len(reports) < 2:
        return {"trend": 0.0, "acceleration": 0.0}

    recent = reports[-window:]
    threat_levels = [r.ecosystem_threat_level for r in recent]

    # Check if all values are the same (no variance)
    if len(set(threat_levels)) <= 1:
        return {
            "trend": 0.0,
            "acceleration": 0.0,
            "current_level": threat_levels[-1],
            "max_level": max(threat_levels),
        }

    # Compute trend (slope of threat levels)
    slope = 0.0
    if len(threat_levels) >= 2:
        x = np.arange(len(threat_levels))
        try:
            slope, _ = np.polyfit(x, threat_levels, 1)
        except np.linalg.LinAlgError:
            slope = 0.0

    # Compute acceleration (change in slope)
    acceleration = 0.0
    if len(threat_levels) >= 4:  # Need at least 4 for meaningful split
        mid = len(threat_levels) // 2
        x = np.arange(len(threat_levels))
        try:
            # Check variance in each half
            if len(set(threat_levels[:mid])) > 1:
                slope1, _ = np.polyfit(x[:mid], threat_levels[:mid], 1)
            else:
                slope1 = 0.0
            if len(set(threat_levels[mid:])) > 1:
                slope2, _ = np.polyfit(x[mid:], threat_levels[mid:], 1)
            else:
                slope2 = 0.0
            acceleration = slope2 - slope1
        except np.linalg.LinAlgError:
            acceleration = 0.0

    return {
        "trend": float(slope),
        "acceleration": float(acceleration),
        "current_level": threat_levels[-1],
        "max_level": max(threat_levels),
    }
