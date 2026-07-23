"""Reputation-based governance from graph memory.

This module provides reputation scoring and governance recommendations
based on agent relationship graphs stored in graph memory.
"""

from typing import Dict, List, Set, Tuple

from swarm.knowledge.graph_memory import GraphMemoryStore
from swarm.metrics.graph_structural import (
    Edge,
    StructuralAnomaly,
    detect_structural_anomalies,
)


class ReputationGovernor:
    """Computes reputation scores and governance recommendations from graph memory.

    Uses relationship edges (trust, interaction count) to compute:
    - Reputation scores in [0, 1] per agent
    - Governance recommendations (normal, monitor, restrict)
    - Fee adjustments based on reputation
    - Collusion cluster detection
    """

    def __init__(self, graph_memory: GraphMemoryStore):
        """Initialize the reputation governor.

        Args:
            graph_memory: GraphMemoryStore instance with agent relationships
        """
        self.graph_memory = graph_memory

    def compute_reputation_scores(self) -> Dict[str, float]:
        """Compute reputation score for each agent.

        Reputation is based on mean trust others have in this agent,
        weighted by interaction count. Higher interaction counts provide
        stronger signal.

        Formula:
            reputation[i] = sum(trust_j_to_i * weight_ij) / sum(weight_ij)
            where weight_ij = sqrt(interaction_count) for normalization

        Returns:
            Dict mapping agent_id -> reputation score in [0, 1]
        """
        all_edges = self.graph_memory.get_all_relationships()

        # Collect all agents that appear in relationships
        agents: Set[str] = set()
        for edge in all_edges:
            agents.add(edge.agent_a)
            agents.add(edge.agent_b)

        scores: Dict[str, float] = {}

        for agent_id in agents:
            # Find all trust edges pointing TO this agent
            incoming_trusts: List[tuple[float, int]] = []

            for edge in all_edges:
                if edge.agent_b == agent_id:
                    # Trust from A to B (this agent is B)
                    incoming_trusts.append((edge.trust_a_to_b, edge.interaction_count))
                elif edge.agent_a == agent_id:
                    # Trust from B to A (this agent is A, so reverse the edge)
                    # Find the reverse edge
                    for other_edge in all_edges:
                        if other_edge.agent_a == edge.agent_b and other_edge.agent_b == agent_id:
                            incoming_trusts.append(
                                (other_edge.trust_b_to_a, other_edge.interaction_count)
                            )
                            break

            if not incoming_trusts:
                # No relationships recorded yet
                scores[agent_id] = 0.5  # Default neutral
            else:
                # Weight by interaction count
                total_weighted_trust = 0.0
                total_weight = 0.0

                for trust, count in incoming_trusts:
                    # Use sqrt of count for sub-linear weighting
                    weight = (count ** 0.5) if count > 0 else 0.0
                    total_weighted_trust += trust * weight
                    total_weight += weight

                if total_weight > 0:
                    scores[agent_id] = total_weighted_trust / total_weight
                else:
                    scores[agent_id] = 0.5

        return scores

    def get_governance_recommendations(
        self,
        threshold: float = 0.3,
        *,
        structural_enabled: bool = True,
        structural_seed: int = 0,
        structural_null_samples: int = 50,
    ) -> Dict[str, str]:
        """Get governance recommendations for all agents.

        Recommendations:
        - "normal" for reputation >= threshold
        - "monitor" for reputation < threshold
        - "restrict" for reputation < threshold/2

        Args:
            threshold: Threshold for monitor/normal boundary (default 0.3)
            structural_enabled: When True (the default since beads-2iok),
                also flag agents that appear in a
                :func:`detect_structural_coalitions` result whose
                ``is_suspicious`` gate fires (pre-registered combination:
                size ≥ 3, density ≥ 1, reciprocity_z ≥ 2, p ≤ 0.05). Such
                agents get bumped to at least "monitor" — or stay at
                "restrict" if their reputation score is also below the
                restrict threshold. The flip to default ON was driven by
                the sk95/5cdk F1@native finding (PR #501): every
                threshold detector is at F1=0.000 across every
                adversarial family, while graph_structural strictly
                dominates F1@native on 4 families (collusion_ring_5/8 +
                both threshold_dancing variants). Pass
                ``structural_enabled=False`` to recover the pre-2iok
                pure-reputation behavior.
            structural_seed: Seed for the configuration-model null sampler.
            structural_null_samples: Number of null samples per anomaly.

        Returns:
            Dict mapping agent_id -> "normal" | "monitor" | "restrict"
        """
        scores = self.compute_reputation_scores()
        recommendations: Dict[str, str] = {}

        restrict_threshold = threshold / 2.0

        for agent_id, score in scores.items():
            if score < restrict_threshold:
                recommendations[agent_id] = "restrict"
            elif score < threshold:
                recommendations[agent_id] = "monitor"
            else:
                recommendations[agent_id] = "normal"

        if structural_enabled:
            anomalies = self.detect_structural_coalitions(
                seed=structural_seed,
                n_null_samples=structural_null_samples,
            )
            flagged: Set[str] = set()
            for a in anomalies:
                if a.is_suspicious:
                    flagged.update(a.members)
            for agent_id in flagged:
                # Bump to at least "monitor"; preserve "restrict" if already there.
                if recommendations.get(agent_id, "normal") == "normal":
                    recommendations[agent_id] = "monitor"

        return recommendations

    def _edges_from_relationships(self) -> List[Edge]:
        """Convert GraphMemoryStore relationships into weighted directed edges
        consumable by :mod:`swarm.metrics.graph_structural`.

        Edge weight = interaction_count between the pair. Self-loops and
        zero-weight edges are dropped by ``DiGraph.from_edges``.
        """
        edges: List[Edge] = []
        for e in self.graph_memory.get_all_relationships():
            if e.interaction_count > 0:
                edges.append((e.agent_a, e.agent_b, float(e.interaction_count)))
        return edges

    def detect_structural_coalitions(
        self,
        *,
        min_size: int = 3,
        n_null_samples: int = 50,
        seed: int = 0,
    ) -> List[StructuralAnomaly]:
        """Run the graph-structural coalition detector on stored
        relationships.

        Complements :meth:`detect_collusion_clusters` (which uses mutual
        trust thresholds) — see ``docs/research/graph-structural-prereg.md``
        for the head-to-head ROC. The pre-registered finding (beads-sk95):
        structural and threshold detectors are *complementary, not
        interchangeable* — different coalition shapes need different
        signals. Wire both into governance recommendations for
        triangulation.
        """
        edges = self._edges_from_relationships()
        return detect_structural_anomalies(
            edges,
            min_size=min_size,
            n_null_samples=n_null_samples,
            seed=seed,
        )

    def compute_trust_weighted_fee(self, agent_id: str, base_fee: float) -> float:
        """Compute fee adjusted by agent reputation.

        Fee adjustment:
        - reputation > 0.7: discount of 0.8x base_fee
        - reputation < 0.3: surcharge of 1.5x base_fee
        - otherwise: base_fee

        Args:
            agent_id: Agent ID
            base_fee: Base fee amount

        Returns:
            Adjusted fee amount
        """
        scores = self.compute_reputation_scores()
        reputation = scores.get(agent_id, 0.5)

        if reputation > 0.7:
            return base_fee * 0.8
        elif reputation < 0.3:
            return base_fee * 1.5
        else:
            return base_fee

    def detect_collusion_clusters(
        self, min_mutual_trust: float = 0.9, min_size: int = 2
    ) -> List[List[str]]:
        """Detect groups of agents with suspiciously high mutual trust.

        Uses a greedy clustering approach:
        1. For each pair of agents, check if trust is mutual and high
        2. Build connected components of agents with high mutual trust
        3. Return clusters of size >= min_size

        Args:
            min_mutual_trust: Minimum trust threshold for mutual trust (default 0.9)
            min_size: Minimum cluster size to report (default 2)

        Returns:
            List of agent ID clusters (each cluster is a list of agent IDs)
        """
        all_edges = self.graph_memory.get_all_relationships()

        # Build adjacency graph for high-trust pairs
        high_trust_pairs: Set[Tuple[str, str]] = set()

        for edge_a in all_edges:
            # For each edge A->B, look for the reverse B->A
            for edge_b in all_edges:
                if (
                    edge_a.agent_a == edge_b.agent_b
                    and edge_a.agent_b == edge_b.agent_a
                ):
                    # Found A->B and B->A
                    if (
                        edge_a.trust_a_to_b >= min_mutual_trust
                        and edge_b.trust_a_to_b >= min_mutual_trust
                    ):
                        # Both directions have high trust
                        agents_sorted = sorted([edge_a.agent_a, edge_a.agent_b])
                        pair: Tuple[str, str] = (agents_sorted[0], agents_sorted[1])
                        high_trust_pairs.add(pair)
                    break

        # Find connected components
        # Build adjacency list
        adj: Dict[str, Set[str]] = {}
        for agent_a, agent_b in high_trust_pairs:
            if agent_a not in adj:
                adj[agent_a] = set()
            if agent_b not in adj:
                adj[agent_b] = set()
            adj[agent_a].add(agent_b)
            adj[agent_b].add(agent_a)

        # DFS to find components
        visited: Set[str] = set()
        clusters: List[List[str]] = []

        def dfs(node: str, component: List[str]) -> None:
            visited.add(node)
            component.append(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for agent_id in adj:
            if agent_id not in visited:
                component: List[str] = []
                dfs(agent_id, component)
                if len(component) >= min_size:
                    clusters.append(sorted(component))

        return clusters
