"""Metrics collection for recursive subagent spawning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from swarm.core.spawn import SpawnTree


@dataclass
class SpawnMetrics:
    """Snapshot of spawn-related metrics for an epoch."""

    total_spawned: int = 0
    max_depth: int = 0
    depth_distribution: Dict[int, int] = field(default_factory=dict)
    spawns_this_epoch: int = 0
    avg_payoff_by_depth: Dict[int, float] = field(default_factory=dict)
    total_payoff_redistributed: float = 0.0
    avg_children_per_parent: float = 0.0
    max_tree_size: int = 0

    def to_dict(self) -> Dict:
        return {
            "total_spawned": self.total_spawned,
            "max_depth": self.max_depth,
            "depth_distribution": self.depth_distribution,
            "spawns_this_epoch": self.spawns_this_epoch,
            "avg_payoff_by_depth": self.avg_payoff_by_depth,
            "total_payoff_redistributed": self.total_payoff_redistributed,
            "avg_children_per_parent": self.avg_children_per_parent,
            "max_tree_size": self.max_tree_size,
        }


class SpawnMetricsCollector:
    """Tracks spawn events and computes per-epoch snapshots."""

    def __init__(self) -> None:
        self._spawns_this_epoch: int = 0
        self._payoff_redistributed_this_epoch: float = 0.0
        self._payoff_by_depth: Dict[int, List[float]] = {}

    def record_spawn(self, depth: int) -> None:
        self._spawns_this_epoch += 1

    def record_redistribution(self, amount: float) -> None:
        self._payoff_redistributed_this_epoch += abs(amount)

    def record_payoff(self, depth: int, payoff: float) -> None:
        self._payoff_by_depth.setdefault(depth, []).append(payoff)

    def collect(
        self,
        tree: Optional[SpawnTree],
        agent_states: Optional[Dict] = None,
    ) -> SpawnMetrics:
        """Build a ``SpawnMetrics`` snapshot from the current state."""
        if tree is None:
            return SpawnMetrics()

        depth_dist = tree.depth_distribution()
        tree_sizes = tree.tree_size_distribution()

        # Compute avg children per parent (nodes with at least one child)
        parents = [
            n for n in tree._nodes.values() if len(n.children) > 0
        ]
        avg_children = (
            sum(len(n.children) for n in parents) / len(parents)
            if parents
            else 0.0
        )

        # Compute average payoff by depth
        avg_payoff_by_depth: Dict[int, float] = {}
        for depth, payoffs in self._payoff_by_depth.items():
            avg_payoff_by_depth[depth] = sum(payoffs) / len(payoffs) if payoffs else 0.0

        return SpawnMetrics(
            total_spawned=tree.total_spawned,
            max_depth=tree.max_tree_depth(),
            depth_distribution=depth_dist,
            spawns_this_epoch=self._spawns_this_epoch,
            avg_payoff_by_depth=avg_payoff_by_depth,
            total_payoff_redistributed=self._payoff_redistributed_this_epoch,
            avg_children_per_parent=avg_children,
            max_tree_size=max(tree_sizes.values()) if tree_sizes else 0,
        )

    def reset_epoch(self) -> None:
        """Reset per-epoch counters."""
        self._spawns_this_epoch = 0
        self._payoff_redistributed_this_epoch = 0.0
        self._payoff_by_depth.clear()
