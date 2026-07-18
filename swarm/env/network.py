"""Network topology for agent interactions."""

from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, model_validator


class NetworkTopology(Enum):
    """Supported network topology types."""

    COMPLETE = "complete"  # All agents connected to all others
    RING = "ring"  # Each agent connected to immediate neighbors
    STAR = "star"  # One central hub connected to all others
    RANDOM_ERDOS_RENYI = "random_erdos_renyi"  # Random edges with probability p
    SMALL_WORLD = "small_world"  # Watts-Strogatz small-world network
    SCALE_FREE = "scale_free"  # Barabási-Albert preferential attachment
    CUSTOM = "custom"  # User-defined adjacency


class NetworkConfig(BaseModel):
    """Configuration for agent network topology."""

    topology: NetworkTopology = NetworkTopology.COMPLETE

    # Erdős-Rényi parameters
    edge_probability: float = 0.5  # Probability of edge in random graph

    # Small-world (Watts-Strogatz) parameters
    k_neighbors: int = 4  # Each node connected to k nearest neighbors
    rewire_probability: float = 0.1  # Probability of rewiring each edge

    # Scale-free (Barabási-Albert) parameters
    m_edges: int = 2  # Number of edges to attach from new node

    # Dynamic network parameters
    dynamic: bool = False
    edge_strengthen_rate: float = 0.1  # Strengthen edge on interaction
    edge_decay_rate: float = 0.05  # Decay per epoch
    min_edge_weight: float = 0.1  # Prune edges below this
    max_edge_weight: float = 1.0  # Cap edge weights

    # Reputation-based dynamics
    reputation_disconnect_threshold: Optional[float] = None  # Sever ties below this rep

    @model_validator(mode="after")
    def _run_validation(self) -> "NetworkConfig":
        self._check_values()
        return self

    def _check_values(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.edge_probability <= 1:
            raise ValueError(
                f"edge_probability must be in [0, 1], got {self.edge_probability}"
            )
        if not 0 <= self.rewire_probability <= 1:
            raise ValueError(
                f"rewire_probability must be in [0, 1], got {self.rewire_probability}"
            )
        if self.k_neighbors < 2:
            raise ValueError(f"k_neighbors must be >= 2, got {self.k_neighbors}")
        if self.m_edges < 1:
            raise ValueError(f"m_edges must be >= 1, got {self.m_edges}")
        if not 0 <= self.edge_decay_rate <= 1:
            raise ValueError(
                f"edge_decay_rate must be in [0, 1], got {self.edge_decay_rate}"
            )


class AgentNetwork:
    """
    Graph structure constraining which agents can interact.

    Supports multiple topology types and dynamic edge evolution.
    Edge weights represent interaction strength/trust.
    """

    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize network.

        Args:
            config: Network configuration
            seed: Random seed for reproducibility
        """
        self.config = NetworkConfig() if config is None else config
        # Pydantic auto-validates
        self._rng = np.random.default_rng(seed)

        # Adjacency represented as dict of dicts: {node: {neighbor: weight}}
        self._adjacency: Dict[str, Dict[str, float]] = {}
        self._agent_ids: List[str] = []

    def initialize(self, agent_ids: List[str]) -> None:
        """
        Initialize network with given agents.

        Args:
            agent_ids: List of agent identifiers
        """
        self._agent_ids = list(agent_ids)
        n = len(agent_ids)

        # Initialize empty adjacency
        self._adjacency = {aid: {} for aid in agent_ids}

        if n < 2:
            return

        # Build topology
        if self.config.topology == NetworkTopology.COMPLETE:
            self._build_complete()
        elif self.config.topology == NetworkTopology.RING:
            self._build_ring()
        elif self.config.topology == NetworkTopology.STAR:
            self._build_star()
        elif self.config.topology == NetworkTopology.RANDOM_ERDOS_RENYI:
            self._build_erdos_renyi()
        elif self.config.topology == NetworkTopology.SMALL_WORLD:
            self._build_small_world()
        elif self.config.topology == NetworkTopology.SCALE_FREE:
            self._build_scale_free()
        # CUSTOM topology starts empty, edges added manually

    def _build_complete(self) -> None:
        """Build complete graph (all-to-all connections)."""
        for i, a in enumerate(self._agent_ids):
            for j, b in enumerate(self._agent_ids):
                if i != j:
                    self._adjacency[a][b] = 1.0

    def _build_ring(self) -> None:
        """Build ring topology (each node connected to neighbors)."""
        n = len(self._agent_ids)
        for i, a in enumerate(self._agent_ids):
            # Connect to next and previous
            next_idx = (i + 1) % n
            prev_idx = (i - 1) % n
            self._adjacency[a][self._agent_ids[next_idx]] = 1.0
            self._adjacency[a][self._agent_ids[prev_idx]] = 1.0

    def _build_star(self) -> None:
        """Build star topology (first node is hub)."""
        if not self._agent_ids:
            return

        hub = self._agent_ids[0]
        for spoke in self._agent_ids[1:]:
            self._adjacency[hub][spoke] = 1.0
            self._adjacency[spoke][hub] = 1.0

    def _build_erdos_renyi(self) -> None:
        """Build Erdős-Rényi random graph."""
        p = self.config.edge_probability
        for i, a in enumerate(self._agent_ids):
            for j, b in enumerate(self._agent_ids):
                if i < j and self._rng.random() < p:
                    self._adjacency[a][b] = 1.0
                    self._adjacency[b][a] = 1.0

    def _build_small_world(self) -> None:
        """Build Watts-Strogatz small-world network."""
        n = len(self._agent_ids)
        k = min(self.config.k_neighbors, n - 1)
        p = self.config.rewire_probability

        # Start with ring lattice
        for i, a in enumerate(self._agent_ids):
            for j in range(1, k // 2 + 1):
                neighbor_idx = (i + j) % n
                self._adjacency[a][self._agent_ids[neighbor_idx]] = 1.0
                self._adjacency[self._agent_ids[neighbor_idx]][a] = 1.0

        # Rewire edges with probability p
        for i, a in enumerate(self._agent_ids):
            for j in range(1, k // 2 + 1):
                if self._rng.random() < p:
                    neighbor_idx = (i + j) % n
                    old_neighbor = self._agent_ids[neighbor_idx]

                    # Remove old edge
                    if old_neighbor in self._adjacency[a]:
                        del self._adjacency[a][old_neighbor]
                        if a in self._adjacency[old_neighbor]:
                            del self._adjacency[old_neighbor][a]

                    # Add new random edge (avoid self-loops and existing edges)
                    candidates = [
                        aid
                        for aid in self._agent_ids
                        if aid != a and aid not in self._adjacency[a]
                    ]
                    if candidates:
                        new_neighbor = self._rng.choice(candidates)
                        self._adjacency[a][new_neighbor] = 1.0
                        self._adjacency[new_neighbor][a] = 1.0

    def _build_scale_free(self) -> None:
        """Build Barabási-Albert scale-free network."""
        m = self.config.m_edges
        n = len(self._agent_ids)

        if n <= m:
            # Not enough nodes, build complete
            self._build_complete()
            return

        # Start with m+1 fully connected nodes
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                a, b = self._agent_ids[i], self._agent_ids[j]
                self._adjacency[a][b] = 1.0
                self._adjacency[b][a] = 1.0

        # Add remaining nodes with preferential attachment
        for i in range(m + 1, n):
            new_node = self._agent_ids[i]

            # Calculate degree of existing nodes
            existing = self._agent_ids[:i]
            degrees = np.array([len(self._adjacency[a]) for a in existing])
            total_degree = degrees.sum()

            if total_degree == 0:
                probs = np.ones(len(existing)) / len(existing)
            else:
                probs = degrees / total_degree

            # Select m nodes to connect to (without replacement)
            targets = self._rng.choice(
                existing,
                size=min(m, len(existing)),
                replace=False,
                p=probs,
            )

            for target in targets:
                self._adjacency[new_node][target] = 1.0
                self._adjacency[target][new_node] = 1.0

    def neighbors(self, agent_id: str) -> List[str]:
        """
        Return list of agents this agent can interact with.

        Args:
            agent_id: Agent to query

        Returns:
            List of neighbor agent IDs
        """
        if agent_id not in self._adjacency:
            return []
        return list(self._adjacency[agent_id].keys())

    def has_edge(self, a: str, b: str) -> bool:
        """Check if edge exists between two agents."""
        return a in self._adjacency and b in self._adjacency[a]

    def edge_weight(self, a: str, b: str) -> float:
        """Get edge weight between two agents (0 if no edge)."""
        if not self.has_edge(a, b):
            return 0.0
        return self._adjacency[a][b]

    def add_node(self, agent_id: str) -> None:
        """Add a node (agent) to the network if not already present."""
        if agent_id not in self._adjacency:
            self._adjacency[agent_id] = {}
        if agent_id not in self._agent_ids:
            self._agent_ids.append(agent_id)

    def add_edge(self, a: str, b: str, weight: float = 1.0) -> None:
        """
        Add or update edge between agents.

        Args:
            a: First agent
            b: Second agent
            weight: Edge weight
        """
        if a not in self._adjacency:
            self._adjacency[a] = {}
        if b not in self._adjacency:
            self._adjacency[b] = {}

        weight = min(weight, self.config.max_edge_weight)
        self._adjacency[a][b] = weight
        self._adjacency[b][a] = weight

    def remove_edge(self, a: str, b: str) -> None:
        """Remove edge between agents."""
        if a in self._adjacency and b in self._adjacency[a]:
            del self._adjacency[a][b]
        if b in self._adjacency and a in self._adjacency[b]:
            del self._adjacency[b][a]

    def strengthen_edge(self, a: str, b: str) -> None:
        """
        Strengthen edge after interaction.

        Creates edge if it doesn't exist.
        """
        if not self.config.dynamic:
            return

        current = self.edge_weight(a, b)
        new_weight = min(
            current + self.config.edge_strengthen_rate,
            self.config.max_edge_weight,
        )
        self.add_edge(a, b, new_weight)

    def decay_edges(self) -> int:
        """
        Apply decay to all edges (call at epoch end).

        Returns:
            Number of edges pruned
        """
        if not self.config.dynamic:
            return 0

        pruned = 0
        edges_to_remove = []

        for a in self._adjacency:
            for b, weight in list(self._adjacency[a].items()):
                new_weight = weight * (1 - self.config.edge_decay_rate)

                if new_weight < self.config.min_edge_weight:
                    edges_to_remove.append((a, b))
                else:
                    self._adjacency[a][b] = new_weight

        # Remove pruned edges
        seen = set()
        for a, b in edges_to_remove:
            edge = tuple(sorted([a, b]))
            if edge not in seen:
                self.remove_edge(a, b)
                seen.add(edge)
                pruned += 1

        return pruned

    def disconnect_low_reputation(
        self,
        agent_id: str,
        reputation: float,
    ) -> int:
        """
        Disconnect from agent if their reputation is too low.

        Args:
            agent_id: Agent to check
            reputation: Agent's current reputation

        Returns:
            Number of edges removed
        """
        threshold = self.config.reputation_disconnect_threshold
        if threshold is None:
            return 0

        if reputation >= threshold:
            return 0

        # Remove all edges to/from this agent
        removed = 0
        neighbors = list(self.neighbors(agent_id))
        for neighbor in neighbors:
            self.remove_edge(agent_id, neighbor)
            removed += 1

        return removed

    def degree(self, agent_id: str) -> int:
        """Get number of connections for an agent."""
        if agent_id not in self._adjacency:
            return 0
        return len(self._adjacency[agent_id])

    def average_degree(self) -> float:
        """Get average degree across all agents."""
        if not self._agent_ids:
            return 0.0
        total = sum(self.degree(a) for a in self._agent_ids)
        return total / len(self._agent_ids)

    def clustering_coefficient(self, agent_id: str) -> float:
        """
        Compute local clustering coefficient for an agent.

        Measures how connected an agent's neighbors are to each other.
        High clustering = tight-knit local community.

        Args:
            agent_id: Agent to compute for

        Returns:
            Clustering coefficient in [0, 1]
        """
        neighbors = self.neighbors(agent_id)
        k = len(neighbors)

        if k < 2:
            return 0.0

        # Count edges between neighbors
        edges_between = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1 :]:
                if self.has_edge(n1, n2):
                    edges_between += 1

        # Maximum possible edges between k neighbors
        max_edges = k * (k - 1) / 2

        return edges_between / max_edges

    def average_clustering(self) -> float:
        """Get average clustering coefficient across all agents."""
        if not self._agent_ids:
            return 0.0
        total = sum(self.clustering_coefficient(a) for a in self._agent_ids)
        return total / len(self._agent_ids)

    def shortest_path(self, a: str, b: str) -> int:
        """
        Compute shortest path length between two agents.

        Uses BFS. Returns -1 if no path exists.

        Args:
            a: Source agent
            b: Target agent

        Returns:
            Path length, or -1 if unreachable
        """
        if a == b:
            return 0

        if a not in self._adjacency or b not in self._adjacency:
            return -1

        # BFS
        visited = {a}
        queue = deque([(a, 0)])

        while queue:
            current, dist = queue.popleft()

            for neighbor in self.neighbors(current):
                if neighbor == b:
                    return dist + 1

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return -1  # No path

    def average_path_length(self) -> float:
        """
        Compute average shortest path length.

        Only considers reachable pairs.

        Returns:
            Average path length, or 0 if no paths
        """
        if len(self._agent_ids) < 2:
            return 0.0

        total = 0
        count = 0

        for i, a in enumerate(self._agent_ids):
            for b in self._agent_ids[i + 1 :]:
                path = self.shortest_path(a, b)
                if path > 0:
                    total += path
                    count += 1

        return total / count if count > 0 else 0.0

    def is_connected(self) -> bool:
        """Check if the network is connected (all agents reachable)."""
        if len(self._agent_ids) < 2:
            return True

        # BFS from first node
        start = self._agent_ids[0]
        visited = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in self.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(self._agent_ids)

    def connected_components(self) -> List[Set[str]]:
        """Find all connected components in the network."""
        remaining = set(self._agent_ids)
        components = []

        while remaining:
            # BFS from arbitrary node
            start = next(iter(remaining))
            component = {start}
            queue = deque([start])

            while queue:
                current = queue.popleft()
                for neighbor in self.neighbors(current):
                    if neighbor not in component:
                        component.add(neighbor)
                        queue.append(neighbor)

            components.append(component)
            remaining -= component

        return components

    def degree_distribution(self) -> Dict[int, int]:
        """Get degree distribution (degree -> count)."""
        dist: Dict[int, int] = {}
        for agent_id in self._agent_ids:
            d = self.degree(agent_id)
            dist[d] = dist.get(d, 0) + 1
        return dist

    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Convert to adjacency matrix.

        Returns:
            Tuple of (matrix, agent_id_order)
        """
        n = len(self._agent_ids)
        matrix = np.zeros((n, n))

        for i, a in enumerate(self._agent_ids):
            for j, b in enumerate(self._agent_ids):
                if b in self._adjacency.get(a, {}):
                    matrix[i, j] = self._adjacency[a][b]

        return matrix, list(self._agent_ids)

    def from_adjacency_matrix(
        self,
        matrix: np.ndarray,
        agent_ids: List[str],
    ) -> None:
        """
        Load network from adjacency matrix.

        Args:
            matrix: Adjacency matrix (n x n)
            agent_ids: List of agent IDs in matrix order
        """
        self._agent_ids = list(agent_ids)
        self._adjacency = {aid: {} for aid in agent_ids}

        n = len(agent_ids)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] > 0:
                    self._adjacency[agent_ids[i]][agent_ids[j]] = float(matrix[i, j])

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute network metrics for reporting.

        Returns:
            Dictionary of metric name -> value
        """
        n_agents = len(self._agent_ids)
        n_edges = sum(self.degree(a) for a in self._agent_ids) // 2

        return {
            "n_agents": n_agents,
            "n_edges": n_edges,
            "avg_degree": self.average_degree(),
            "avg_clustering": self.average_clustering(),
            "average_path_length": self.average_path_length(),
            "is_connected": float(self.is_connected()),
            "n_components": len(self.connected_components()),
        }

    def __repr__(self) -> str:
        n_edges = sum(self.degree(a) for a in self._agent_ids) // 2
        return (
            f"AgentNetwork(topology={self.config.topology.value}, "
            f"agents={len(self._agent_ids)}, edges={n_edges})"
        )
