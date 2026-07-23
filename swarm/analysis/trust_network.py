"""Trust network analysis and visualization for agent relationships.

Provides analysis of agent trust relationships from a GraphMemoryStore,
including cluster detection, isolation scoring, and force-directed network
visualization.

Usage::

    from swarm.analysis.trust_network import TrustNetworkAnalyzer, plot_trust_network
    from swarm.knowledge.graph_memory import GraphMemoryStore

    store = GraphMemoryStore()
    analyzer = TrustNetworkAnalyzer(store)

    summary = analyzer.get_network_summary()
    clusters = analyzer.compute_clusters(threshold=0.7)

    plot_trust_network(analyzer, "trust_network.png")
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.network import compute_spring_layout
from swarm.analysis.theme import COLORS, agent_color, swarm_theme
from swarm.knowledge.graph_memory import (
    GraphMemoryStore,
)


class TrustNetworkAnalyzer:
    """Analyzer for agent trust networks from GraphMemoryStore.

    Computes trust matrices, clusters, isolation scores, and summary
    statistics from relationship edges and agent snapshots.

    Attributes:
        store: GraphMemoryStore instance
        _snapshots: Cached agent memory snapshots by agent_id
        _relationships: Cached relationship edges
    """

    def __init__(self, store: GraphMemoryStore):
        """Initialize the analyzer.

        Args:
            store: GraphMemoryStore instance with agent relationships
        """
        self.store = store
        self._snapshots = store.load_all()
        self._relationships = store.get_all_relationships()

    def build_trust_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build adjacency matrix of trust scores from relationships.

        Returns:
            Dict mapping agent_id -> Dict[agent_id -> trust_score].
            trust_score is average of trust_a_to_b and trust_b_to_a for
            mutual relationships.
        """
        matrix: Dict[str, Dict[str, float]] = {}

        # Initialize all agents with empty dicts
        for agent_id in self._snapshots:
            matrix[agent_id] = {}

        # Add trust scores from relationships
        for rel in self._relationships:
            agent_a = rel.agent_a
            agent_b = rel.agent_b

            # Initialize agent entries if needed
            if agent_a not in matrix:
                matrix[agent_a] = {}
            if agent_b not in matrix:
                matrix[agent_b] = {}

            # Average trust in both directions
            avg_trust = (rel.trust_a_to_b + rel.trust_b_to_a) / 2.0

            matrix[agent_a][agent_b] = avg_trust
            matrix[agent_b][agent_a] = avg_trust

        return matrix

    def compute_clusters(self, threshold: float = 0.7) -> List[List[str]]:
        """Identify clusters of agents with mutual high trust.

        Uses connected-components algorithm on edges where both
        trust_a_to_b and trust_b_to_a exceed threshold.

        Args:
            threshold: Minimum trust level (default 0.7)

        Returns:
            List of clusters, each cluster is a sorted list of agent IDs
        """
        # Build undirected graph of high-trust relationships
        edges: Dict[str, set] = {}

        # Initialize all agents
        for agent_id in self._snapshots:
            edges[agent_id] = set()

        # Add high-trust edges
        for rel in self._relationships:
            if rel.trust_a_to_b >= threshold and rel.trust_b_to_a >= threshold:
                edges[rel.agent_a].add(rel.agent_b)
                edges[rel.agent_b].add(rel.agent_a)

        # Connected-components via BFS
        visited: set[str] = set()
        clusters: List[List[str]] = []

        for agent_id in self._snapshots:
            if agent_id in visited:
                continue

            # BFS to find connected component
            component = []
            queue = [agent_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbor in edges.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            clusters.append(sorted(component))

        return clusters

    def compute_isolation_score(self, agent_id: str) -> float:
        """Compute how isolated an agent is in the trust network.

        Isolation score ranges from 0 (well-connected) to 1 (fully isolated).
        Based on mean trust that others have toward this agent.

        Args:
            agent_id: Agent ID

        Returns:
            Isolation score in [0, 1]
        """
        if agent_id not in self._snapshots:
            return 1.0  # Unknown agent is fully isolated

        # Find all relationships where this agent is the target (agent_b)
        incoming_trust_scores = []

        for rel in self._relationships:
            if rel.agent_b == agent_id:
                incoming_trust_scores.append(rel.trust_a_to_b)
            elif rel.agent_a == agent_id:
                incoming_trust_scores.append(rel.trust_b_to_a)

        if not incoming_trust_scores:
            return 1.0  # No relationships = fully isolated

        # Isolation = 1 - mean_trust
        mean_trust = float(np.mean(incoming_trust_scores))
        isolation = 1.0 - mean_trust

        return float(np.clip(isolation, 0.0, 1.0))

    def get_network_summary(self) -> Dict:
        """Get summary statistics of the trust network.

        Returns:
            Dict with keys:
                - n_agents: Total number of agents
                - n_edges: Total number of relationships
                - mean_trust: Mean trust score across all relationships
                - trust_reciprocity: Pearson correlation of trust_a_to_b vs trust_b_to_a
                - cluster_count: Number of clusters (threshold=0.7)
                - isolated_agents: List of agent IDs with isolation_score > 0.8
        """
        n_agents = len(self._snapshots)
        n_edges = len(self._relationships)

        # Mean trust
        if n_edges > 0:
            trust_scores = []
            for rel in self._relationships:
                trust_scores.append(rel.trust_a_to_b)
                trust_scores.append(rel.trust_b_to_a)
            mean_trust = float(np.mean(trust_scores))
        else:
            mean_trust = 0.5

        # Reciprocity (correlation of bidirectional trust)
        if n_edges > 1:
            trust_a_to_b = [rel.trust_a_to_b for rel in self._relationships]
            trust_b_to_a = [rel.trust_b_to_a for rel in self._relationships]
            reciprocity = float(np.corrcoef(trust_a_to_b, trust_b_to_a)[0, 1])
            # Handle NaN if all values are identical
            if np.isnan(reciprocity):
                reciprocity = 1.0 if len(set(trust_a_to_b + trust_b_to_a)) == 1 else 0.0
        else:
            reciprocity = 0.0

        # Clusters and isolated agents
        clusters = self.compute_clusters(threshold=0.7)
        cluster_count = len(clusters)

        isolated_agents = []
        for agent_id in self._snapshots:
            if self.compute_isolation_score(agent_id) > 0.8:
                isolated_agents.append(agent_id)

        return {
            "n_agents": n_agents,
            "n_edges": n_edges,
            "mean_trust": mean_trust,
            "trust_reciprocity": reciprocity,
            "cluster_count": cluster_count,
            "isolated_agents": sorted(isolated_agents),
        }


def plot_trust_network(
    analyzer: TrustNetworkAnalyzer,
    output_path: str,
    title: str = "Trust Network",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Plot trust network using force-directed layout and matplotlib.

    Creates a network visualization where:
    - Nodes are colored by agent_type
    - Edge width is proportional to interaction_count
    - Edge color reflects trust on a diverging scale (red=low, green=high)
    - Saves to output_path

    Args:
        analyzer: TrustNetworkAnalyzer instance
        output_path: Path to save PNG file
        title: Plot title
        figsize: Figure size tuple (width, height)
    """
    swarm_theme()

    relationships = analyzer._relationships
    snapshots = analyzer._snapshots

    if not relationships:
        # Handle empty network
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No relationships to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # Collect all agent IDs from relationships
    agent_id_set: set[str] = set()
    for rel in relationships:
        agent_id_set.add(rel.agent_a)
        agent_id_set.add(rel.agent_b)
    agent_ids: list[str] = list(agent_id_set)

    # Build edge list with weights (interaction_count)
    edges = []
    edge_trust_map = {}  # Map (a, b) -> avg trust for coloring

    for rel in relationships:
        avg_trust = (rel.trust_a_to_b + rel.trust_b_to_a) / 2.0
        weight = max(rel.interaction_count, 0.1)  # Min width for visibility

        edges.append((rel.agent_a, rel.agent_b, weight))
        edge_trust_map[(rel.agent_a, rel.agent_b)] = avg_trust
        edge_trust_map[(rel.agent_b, rel.agent_a)] = avg_trust

    # Compute spring layout
    layout = compute_spring_layout(agent_ids, edges, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS.BG_DARK)
    ax.set_facecolor(COLORS.BG_DARK)

    # Draw edges with trust-based coloring
    for agent_a, agent_b, weight in edges:
        x1, y1 = layout[agent_a]
        x2, y2 = layout[agent_b]

        # Get trust value for coloring (diverging scale: red=0.0, green=1.0)
        avg_trust = edge_trust_map[(agent_a, agent_b)]

        # Diverging colormap: red -> yellow -> green (matching matplotlib's RdYlGn)
        if avg_trust < 0.5:
            # Red to yellow (0 to 0.5 -> red to yellow)
            t = avg_trust * 2  # 0 to 1
            r_val = 1.0
            g_val = t  # 0 to 1 (red + green = yellow)
            b_val = 0.0
            color: Tuple[float, float, float] = (r_val, g_val, b_val)
        else:
            # Yellow to green (0.5 to 1 -> yellow to green)
            t = (avg_trust - 0.5) * 2  # 0 to 1
            r_val = 1 - t  # 1 to 0
            g_val = 1.0  # stays at 1 (full green component)
            b_val = 0.0
            color = (r_val, g_val, b_val)

        # Edge width proportional to interaction count
        line_width = 0.5 + (weight - 0.1) * 3

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=line_width,
            alpha=0.6,
            zorder=1,
        )

    # Draw nodes
    for agent_id in agent_ids:
        x, y = layout[agent_id]

        # Get agent type for coloring
        snapshot = snapshots.get(agent_id)
        agent_type = snapshot.agent_type if snapshot else "unknown"
        node_color = agent_color(agent_type)

        ax.scatter(x, y, s=300, c=node_color, zorder=2, edgecolors=COLORS.TEXT_PRIMARY, linewidth=1.5)

        # Add label
        ax.text(
            x,
            y - 0.08,
            agent_id,
            ha="center",
            va="top",
            fontsize=8,
            color=COLORS.TEXT_PRIMARY,
            weight="bold",
        )

    # Add legend for agent types
    agent_types = set()
    for snapshot in snapshots.values():
        agent_types.add(snapshot.agent_type)

    legend_handles = []
    for agent_type in sorted(agent_types):
        legend_color = agent_color(agent_type)
        handle = plt.scatter([], [], s=100, c=legend_color, edgecolors=COLORS.TEXT_PRIMARY, linewidth=1)
        legend_handles.append((agent_type, handle))

    # Add colorbar for trust (optional, use as reference)
    # Create a mini colorbar
    cbar_ax = fig.add_axes((0.92, 0.3, 0.02, 0.4))  # type: ignore[arg-type]
    cbar_norm = plt.Normalize(vmin=0, vmax=1)
    cbar_mapper = plt.cm.ScalarMappable(norm=cbar_norm, cmap="RdYlGn")
    cbar = fig.colorbar(cbar_mapper, cax=cbar_ax)
    cbar.set_label("Trust", color=COLORS.TEXT_PRIMARY, fontsize=10)
    cbar_ax.tick_params(colors=COLORS.TEXT_PRIMARY)

    ax.legend(
        [h for _, h in legend_handles],
        [t for t, _ in legend_handles],
        loc="upper left",
        framealpha=0.9,
        facecolor=COLORS.BG_PANEL,
        edgecolor=COLORS.ACCENT_BORDER,
        labelcolor=COLORS.TEXT_PRIMARY,
    )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(title, color=COLORS.TEXT_PRIMARY, fontsize=14, weight="bold", pad=20)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS.BG_DARK)
    plt.close(fig)
