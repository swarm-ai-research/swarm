"""Security detection and penalty governance lever."""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.metrics.security import (
    SecurityAnalyzer,
    SecurityReport,
    ThreatIndicator,
    ThreatType,
)

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.models.interaction import SoftInteraction


class SecurityLever(GovernanceLever):
    """
    Governance lever that detects and responds to security threats.

    Monitors for:
    - Prompt injection attempts
    - Cross-agent manipulation
    - Information laundering
    - Contagion/worm propagation
    - Exfiltration attempts

    Applies penalties and containment actions based on detected threats.
    """

    def __init__(self, config: "GovernanceConfig", seed: Optional[int] = None):
        """Initialize security lever."""
        super().__init__(config)

        # Initialize analyzer with config parameters
        self._analyzer = SecurityAnalyzer(
            injection_threshold=getattr(config, "security_injection_threshold", 0.3),
            manipulation_threshold=getattr(
                config, "security_manipulation_threshold", 0.5
            ),
            laundering_trust_gap=getattr(config, "security_laundering_trust_gap", 0.3),
            contagion_velocity_threshold=getattr(
                config, "security_contagion_velocity", 2.0
            ),
            min_chain_length=getattr(config, "security_min_chain_length", 3),
            seed=seed,
        )

        # Track interaction history
        self._interaction_history: List["SoftInteraction"] = []
        self._agent_ids: List[str] = []

        # Cache latest report
        self._latest_report: Optional[SecurityReport] = None

        # Track penalties applied
        self._epoch_penalties: Dict[str, float] = defaultdict(float)
        self._quarantined_agents: Set[str] = set()
        self._containment_actions: List[Dict] = []

    @property
    def name(self) -> str:
        """Return lever name."""
        return "security"

    def set_agent_ids(self, agent_ids: List[str]) -> None:
        """Set the list of agent IDs for analysis."""
        self._agent_ids = list(agent_ids)

    def set_agent_trust_scores(self, trust_scores: Dict[str, float]) -> None:
        """Set agent trust/reputation scores for analysis."""
        self._analyzer.set_agent_trust_scores(trust_scores)

    def on_epoch_start(
        self,
        state: "EnvState",
        epoch: int,
    ) -> LeverEffect:
        """
        Analyze security at epoch start and apply containment.

        Runs threat detection and applies epoch-level penalties
        and quarantine actions.

        Note:
            interaction_history may be cleared at the end of this method
            based on the `security_clear_history_on_epoch` config setting.
            This affects the semantics of the min_interactions check: if
            history is cleared, subsequent min_interactions validations are
            over per-epoch samples; otherwise they validate rolling history.
        """
        if not getattr(self.config, "security_enabled", False):
            return LeverEffect(lever_name=self.name)

        # Clear epoch penalties
        self._epoch_penalties.clear()

        # Update trust scores from state
        trust_scores = {}
        for agent_id, agent_state in state.agents.items():
            trust_scores[agent_id] = agent_state.reputation
        self._analyzer.set_agent_trust_scores(trust_scores)

        # Run analysis if we have enough history
        min_interactions = getattr(self.config, "security_min_interactions", 5)
        if len(self._interaction_history) < min_interactions:
            return LeverEffect(lever_name=self.name)

        self._latest_report = self._analyzer.analyze(
            self._interaction_history,
            self._agent_ids if self._agent_ids else None,
        )

        # Compute penalties based on threat scores
        penalty_multiplier = getattr(self.config, "security_penalty_multiplier", 1.0)
        reputation_penalties: Dict[str, float] = {}
        resource_penalties: Dict[str, float] = {}
        agents_to_freeze: Set[str] = set()

        quarantine_threshold = getattr(
            self.config, "security_quarantine_threshold", 0.7
        )

        for agent_id, threat_score in self._latest_report.agent_threat_scores.items():
            if threat_score >= quarantine_threshold:
                # Severe threat: quarantine (freeze) agent
                agents_to_freeze.add(agent_id)
                self._quarantined_agents.add(agent_id)
                self._containment_actions.append(
                    {
                        "action": "quarantine",
                        "agent_id": agent_id,
                        "epoch": epoch,
                        "threat_score": threat_score,
                    }
                )

            if threat_score >= getattr(self.config, "security_penalty_threshold", 0.3):
                # Apply penalty proportional to threat score
                rep_penalty = -threat_score * penalty_multiplier
                res_penalty = -threat_score * penalty_multiplier * 5

                reputation_penalties[agent_id] = rep_penalty
                resource_penalties[agent_id] = res_penalty
                self._epoch_penalties[agent_id] = threat_score

        # Mark contagion chains as contained if we quarantined origin
        for chain in self._latest_report.contagion_chains:
            if chain.origin_agent in agents_to_freeze:
                chain.contained = True
                self._containment_actions.append(
                    {
                        "action": "contain_chain",
                        "chain_id": chain.chain_id,
                        "epoch": epoch,
                    }
                )

        # Optionally clear history after analysis
        if getattr(self.config, "security_clear_history_on_epoch", False):
            self._interaction_history.clear()

        return LeverEffect(
            lever_name=self.name,
            agents_to_freeze=agents_to_freeze,
            reputation_deltas=reputation_penalties,
            resource_deltas=resource_penalties,
            details={
                "ecosystem_threat_level": self._latest_report.ecosystem_threat_level,
                "active_threats": self._latest_report.active_threat_count,
                "contagion_depth": self._latest_report.contagion_depth,
                "agents_flagged": list(self._latest_report.agents_flagged),
                "quarantined": list(agents_to_freeze),
                "penalties_applied": dict(self._epoch_penalties),
            },
        )

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        """
        Record interaction and optionally apply real-time penalty.

        For real-time detection, applies a penalty if the interaction
        shows strong threat indicators.
        """
        if not getattr(self.config, "security_enabled", False):
            return LeverEffect(lever_name=self.name)

        # Always record for history
        self._interaction_history.append(interaction)
        self._analyzer.record_interaction(interaction)

        # Real-time penalty if enabled
        if not getattr(self.config, "security_realtime_penalty", False):
            return LeverEffect(lever_name=self.name)

        # Quick real-time threat assessment
        threat_score = self._compute_realtime_threat_score(interaction)

        realtime_threshold = getattr(self.config, "security_realtime_threshold", 0.5)
        if threat_score < realtime_threshold:
            return LeverEffect(lever_name=self.name)

        # Apply cost penalty
        realtime_rate = getattr(self.config, "security_realtime_rate", 0.2)
        penalty = realtime_rate * threat_score

        return LeverEffect(
            lever_name=self.name,
            cost_a=penalty,
            cost_b=penalty * 0.5,  # Counterparty partially penalized
            details={
                "realtime_penalty": True,
                "threat_score": threat_score,
                "threat_type": self._identify_threat_type(interaction),
            },
        )

    def _compute_realtime_threat_score(
        self,
        interaction: "SoftInteraction",
    ) -> float:
        """Compute real-time threat score for an interaction."""
        score = 0.0

        # Low quality + accepted = suspicious
        if interaction.accepted and interaction.p < 0.3:
            score += 0.3

        # Tool misuse is a strong signal
        if interaction.tool_misuse_flags > 0:
            score += min(0.4, 0.2 * interaction.tool_misuse_flags)

        # Verifier rejections
        if interaction.verifier_rejections > 0:
            score += min(0.3, 0.15 * interaction.verifier_rejections)

        # Check if initiator is already flagged
        if self._latest_report:
            initiator_score = self._latest_report.agent_threat_scores.get(
                interaction.initiator, 0.0
            )
            score += initiator_score * 0.3

        return min(1.0, score)

    def _identify_threat_type(
        self,
        interaction: "SoftInteraction",
    ) -> str:
        """Identify likely threat type for an interaction."""
        if interaction.tool_misuse_flags >= 2:
            return ThreatType.EXFILTRATION.value
        if interaction.tool_misuse_flags >= 1:
            return ThreatType.PROMPT_INJECTION.value
        if interaction.p < 0.2 and interaction.accepted:
            return ThreatType.CROSS_AGENT_MANIPULATION.value
        return "unknown"

    def can_agent_act(
        self,
        agent_id: str,
        state: "EnvState",
    ) -> bool:
        """
        Check if agent is allowed to act (not quarantined).

        Quarantined agents cannot perform any actions.
        """
        if not getattr(self.config, "security_enabled", False):
            return True

        if agent_id in self._quarantined_agents:
            return False

        return True

    def get_report(self) -> Optional[SecurityReport]:
        """Get the latest security report."""
        return self._latest_report

    def get_threat_indicators(self) -> List[ThreatIndicator]:
        """Get all threat indicators from latest analysis."""
        if self._latest_report is None:
            return []
        return self._latest_report.threat_indicators

    def get_quarantined_agents(self) -> Set[str]:
        """Get set of quarantined agents."""
        return self._quarantined_agents.copy()

    def release_from_quarantine(self, agent_id: str) -> bool:
        """Release an agent from quarantine."""
        if agent_id in self._quarantined_agents:
            self._quarantined_agents.discard(agent_id)
            self._containment_actions.append(
                {
                    "action": "release",
                    "agent_id": agent_id,
                }
            )
            return True
        return False

    def get_containment_actions(self) -> List[Dict]:
        """Get history of containment actions."""
        return list(self._containment_actions)

    def get_interaction_history(self) -> List["SoftInteraction"]:
        """Get recorded interaction history."""
        return list(self._interaction_history)

    def clear_history(self) -> None:
        """Clear interaction history and containment state."""
        self._interaction_history.clear()
        self._analyzer.clear_history()
        self._latest_report = None
        self._quarantined_agents.clear()
        self._containment_actions.clear()
