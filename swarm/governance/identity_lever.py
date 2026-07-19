"""Sybil detection governance lever.

Detects and penalizes suspected Sybil agents (multiple identities
controlled by a single entity) using behavioral similarity analysis.

Reference: "Virtual Agent Economies" (arXiv 2509.10147) - Tomasev et al.
"""

from typing import TYPE_CHECKING, Dict, List, Set

from swarm.governance.levers import GovernanceLever, LeverEffect

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.models.interaction import SoftInteraction


class SybilDetectionLever(GovernanceLever):
    """
    Governance lever that detects and penalizes suspected Sybil agents.

    Uses behavioral similarity analysis to identify clusters of agents
    that appear to be controlled by the same entity. Flagged agents
    receive reputation and resource penalties.
    """

    def __init__(self, config: "GovernanceConfig"):
        """Initialize the Sybil detection lever."""
        super().__init__(config)
        self._interaction_patterns: Dict[str, Dict[str, int]] = {}
        self._sybil_clusters: List[Set[str]] = []
        self._flagged_agents: Set[str] = set()

    @property
    def name(self) -> str:
        """Return the lever name."""
        return "sybil_detection"

    def on_epoch_start(
        self,
        state: "EnvState",
        epoch: int,
    ) -> LeverEffect:
        """Run Sybil detection at epoch start."""
        if not self.config.sybil_detection_enabled:
            return LeverEffect(lever_name=self.name)

        # Detect clusters
        self._sybil_clusters = self._detect_clusters()

        # Build penalty effect
        reputation_deltas: Dict[str, float] = {}
        resource_deltas: Dict[str, float] = {}
        self._flagged_agents.clear()

        for cluster in self._sybil_clusters:
            for agent_id in cluster:
                self._flagged_agents.add(agent_id)
                reputation_deltas[agent_id] = (
                    reputation_deltas.get(agent_id, 0.0)
                    - self.config.sybil_penalty_multiplier * 0.1
                )
                resource_deltas[agent_id] = (
                    resource_deltas.get(agent_id, 0.0)
                    - self.config.sybil_penalty_multiplier * 1.0
                )

        return LeverEffect(
            lever_name=self.name,
            reputation_deltas=reputation_deltas,
            resource_deltas=resource_deltas,
            details={
                "clusters_found": len(self._sybil_clusters),
                "agents_flagged": len(self._flagged_agents),
            },
        )

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        """Track interaction patterns for Sybil detection."""
        if not self.config.sybil_detection_enabled:
            return LeverEffect(lever_name=self.name)

        # Record interaction pattern
        initiator = interaction.initiator
        counterparty = interaction.counterparty

        if initiator not in self._interaction_patterns:
            self._interaction_patterns[initiator] = {}
        self._interaction_patterns[initiator][counterparty] = (
            self._interaction_patterns[initiator].get(counterparty, 0) + 1
        )

        if counterparty not in self._interaction_patterns:
            self._interaction_patterns[counterparty] = {}
        self._interaction_patterns[counterparty][initiator] = (
            self._interaction_patterns[counterparty].get(initiator, 0) + 1
        )

        # Apply per-interaction penalty if both parties are flagged
        if (
            self.config.sybil_realtime_penalty
            and initiator in self._flagged_agents
            and counterparty in self._flagged_agents
        ):
            penalty = self.config.sybil_realtime_rate
            return LeverEffect(
                cost_a=penalty,
                cost_b=penalty,
                lever_name=self.name,
                details={"sybil_pair_penalty": True},
            )

        return LeverEffect(lever_name=self.name)

    def can_agent_act(
        self,
        agent_id: str,
        state: "EnvState",
    ) -> bool:
        """Block agents in large Sybil clusters."""
        if not self.config.sybil_detection_enabled:
            return True

        # Only block if agent is in a cluster larger than allowed
        for cluster in self._sybil_clusters:
            if agent_id in cluster:
                if len(cluster) > self.config.sybil_max_cluster_size:
                    return False

        return True

    def _detect_clusters(self) -> List[Set[str]]:
        """Detect Sybil clusters from interaction patterns."""
        agent_ids = list(self._interaction_patterns.keys())
        clusters: List[Set[str]] = []
        visited: Set[str] = set()
        threshold = self.config.sybil_similarity_threshold

        for i, a_id in enumerate(agent_ids):
            if a_id in visited:
                continue

            cluster = {a_id}
            for j in range(i + 1, len(agent_ids)):
                b_id = agent_ids[j]
                if b_id in visited:
                    continue

                sim = self._behavioral_similarity(
                    self._interaction_patterns.get(a_id, {}),
                    self._interaction_patterns.get(b_id, {}),
                )

                if sim >= threshold:
                    cluster.add(b_id)

            if len(cluster) > 1:
                clusters.append(cluster)
                visited.update(cluster)

        return clusters

    def _behavioral_similarity(
        self,
        pattern_a: Dict[str, int],
        pattern_b: Dict[str, int],
    ) -> float:
        """Compute behavioral similarity as average of Jaccard + frequency correlation."""
        if not pattern_a or not pattern_b:
            return 0.0

        keys_a = set(pattern_a.keys())
        keys_b = set(pattern_b.keys())
        intersection = keys_a & keys_b
        union = keys_a | keys_b

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        if intersection:
            freq_a = [pattern_a[k] for k in intersection]
            freq_b = [pattern_b[k] for k in intersection]
            total_a = sum(freq_a)
            total_b = sum(freq_b)

            if total_a > 0 and total_b > 0:
                norm_a = [f / total_a for f in freq_a]
                norm_b = [f / total_b for f in freq_b]
                dot = sum(a * b for a, b in zip(norm_a, norm_b, strict=True))
                return (jaccard + dot) / 2

        return jaccard

    def get_flagged_agents(self) -> frozenset:
        """Get set of flagged Sybil agents."""
        return frozenset(self._flagged_agents)

    def get_clusters(self) -> List[Set[str]]:
        """Get detected Sybil clusters."""
        return list(self._sybil_clusters)

    def clear_history(self) -> None:
        """Clear interaction patterns and detection state."""
        self._interaction_patterns.clear()
        self._sybil_clusters.clear()
        self._flagged_agents.clear()
