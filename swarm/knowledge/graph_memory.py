"""Persistent graph memory store for SWARM agents.

This module provides a local-first (no external service dependency) persistent
graph memory system for storing agent memories, interaction history, and
relationship graphs to JSON files. Designed for integration with external
tools like MiroShark for cascading analysis.

The store is thread-safe using file locking and supports bulk save/load
operations for efficient memory management across runs.
"""

import fcntl
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from swarm.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentMemorySnapshot:
    """Snapshot of an agent's memory at a point in time.

    Attributes:
        agent_id: Unique identifier for the agent
        agent_type: Behavioral archetype (e.g., 'honest', 'opportunistic')
        counterparty_trust: Dict mapping agent_id -> trust score in [0, 1]
        interaction_summaries: List of condensed interaction records
        total_interactions: Total number of interactions recorded
        total_payoff: Cumulative payoff across all interactions
        run_id: Which run this snapshot is from
        epoch: Epoch at which snapshot was taken
        timestamp: ISO-8601 timestamp of snapshot
    """

    agent_id: str
    agent_type: str
    counterparty_trust: Dict[str, float] = field(default_factory=dict)
    interaction_summaries: List[Dict[str, Any]] = field(default_factory=list)
    total_interactions: int = 0
    total_payoff: float = 0.0
    run_id: str = ""
    epoch: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMemorySnapshot":
        """Deserialize from dictionary."""
        return cls(**data)

    def validate(self) -> None:
        """Validate that trust scores are in [0, 1]."""
        for agent_id, trust in self.counterparty_trust.items():
            if not (0.0 <= trust <= 1.0):
                raise ValueError(
                    f"Trust value for {agent_id} out of bounds [0, 1]: {trust}"
                )


@dataclass
class RelationshipEdge:
    """Directed edge in the agent relationship graph.

    Represents trust and interaction history between a pair of agents.

    Attributes:
        agent_a: Source agent ID
        agent_b: Target agent ID
        trust_a_to_b: Trust score from A to B (in [0, 1])
        trust_b_to_a: Trust score from B to A (in [0, 1])
        interaction_count: Total interactions between A and B
        avg_p: Average quality (p) of interactions
        total_payoff_a: Total payoff for A from interactions with B
        total_payoff_b: Total payoff for B from interactions with A
        last_interaction_epoch: Epoch of most recent interaction
    """

    agent_a: str
    agent_b: str
    trust_a_to_b: float = 0.5
    trust_b_to_a: float = 0.5
    interaction_count: int = 0
    avg_p: float = 0.5
    total_payoff_a: float = 0.0
    total_payoff_b: float = 0.0
    last_interaction_epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipEdge":
        """Deserialize from dictionary."""
        return cls(**data)

    def validate(self) -> None:
        """Validate that trust scores are in [0, 1] and avg_p is valid."""
        if not (0.0 <= self.trust_a_to_b <= 1.0):
            raise ValueError(f"trust_a_to_b out of bounds: {self.trust_a_to_b}")
        if not (0.0 <= self.trust_b_to_a <= 1.0):
            raise ValueError(f"trust_b_to_a out of bounds: {self.trust_b_to_a}")
        if not (0.0 <= self.avg_p <= 1.0):
            raise ValueError(f"avg_p out of bounds: {self.avg_p}")


class GraphMemoryStore:
    """Local-first persistent graph memory store for agents.

    Stores agent memories, interaction summaries, and relationship graphs
    to a JSON file. Thread-safe using file locking. Supports bulk operations
    for efficient save/load at run boundaries.

    Attributes:
        store_path: Path to JSON file storing all memories (default: runs/graph_memory.json)
    """

    def __init__(self, store_path: Optional[str] = None):
        """Initialize the graph memory store.

        Args:
            store_path: Path to JSON file for persistence.
                       Defaults to runs/graph_memory.json
        """
        if store_path is None:
            store_path = "runs/graph_memory.json"

        self.store_path = Path(store_path)
        self._ensure_store_directory()
        self._load_store()

    def _ensure_store_directory(self) -> None:
        """Ensure the directory for the store file exists."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_store(self) -> None:
        """Load existing store from disk or initialize empty."""
        if self.store_path.exists():
            try:
                with self._read_lock():
                    with open(self.store_path, "r") as f:
                        data = json.load(f)
                        self._snapshots: Dict[str, List[Dict[str, Any]]] = data.get(
                            "snapshots", {}
                        )
                        self._relationships: List[Dict[str, Any]] = data.get(
                            "relationships", []
                        )
                        self._metadata: Dict[str, Any] = data.get("metadata", {})
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load store from {self.store_path}: {e}")
                self._initialize_empty()
        else:
            self._initialize_empty()

    def _initialize_empty(self) -> None:
        """Initialize an empty store."""
        self._snapshots = {}
        self._relationships = []
        self._metadata = {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _read_lock(self) -> Any:
        """Context manager for read locking."""
        return self._FileLock(self.store_path, fcntl.LOCK_SH)

    def _write_lock(self) -> Any:
        """Context manager for write locking."""
        return self._FileLock(self.store_path, fcntl.LOCK_EX)

    class _FileLock:
        """Parameterized file lock context manager for shared or exclusive locking."""

        def __init__(self, path: Path, lock_type: int):
            """Initialize file lock.

            Args:
                path: Path to the file being locked
                lock_type: fcntl lock type (fcntl.LOCK_SH for read, fcntl.LOCK_EX for write)
            """
            self.path = path
            self.lock_type = lock_type
            self.lock_file = None

        def __enter__(self):
            # Create lock file if needed
            lock_path = Path(str(self.path) + ".lock")
            self.lock_file = open(lock_path, "w")
            fcntl.flock(self.lock_file.fileno(), self.lock_type)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.lock_file:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()

    def _persist(self) -> None:
        """Write current state to disk atomically."""
        with self._write_lock():
            self._metadata["last_updated"] = datetime.now().isoformat()
            data = {
                "snapshots": self._snapshots,
                "relationships": self._relationships,
                "metadata": self._metadata,
            }
            # Write to temp file first, then rename (atomic on most filesystems)
            temp_path = Path(str(self.store_path) + ".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.store_path)

    def save(self, agent_id: str, snapshot: AgentMemorySnapshot) -> None:
        """Save an agent's memory snapshot.

        Args:
            agent_id: Agent ID (must match snapshot.agent_id)
            snapshot: Memory snapshot to save

        Raises:
            ValueError: If snapshot is invalid
        """
        snapshot.validate()

        if agent_id not in self._snapshots:
            self._snapshots[agent_id] = []

        self._snapshots[agent_id].append(snapshot.to_dict())
        self._persist()
        logger.debug(f"Saved memory snapshot for {agent_id}")

    def load(self, agent_id: str, index: int = -1) -> Optional[AgentMemorySnapshot]:
        """Load an agent's memory snapshot.

        Args:
            agent_id: Agent ID
            index: Index of snapshot to load (-1 for most recent)

        Returns:
            AgentMemorySnapshot if found, None otherwise
        """
        with self._read_lock():
            if agent_id not in self._snapshots or not self._snapshots[agent_id]:
                return None

            try:
                data = self._snapshots[agent_id][index]
                snapshot = AgentMemorySnapshot.from_dict(data)
                snapshot.validate()
                return snapshot
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to load snapshot for {agent_id}: {e}")
                return None

    def save_all(self, agents: List["BaseAgent"], run_id: str, epoch: int) -> None:
        """Bulk save memory snapshots for all agents.

        Args:
            agents: List of agents to save
            run_id: Run identifier
            epoch: Current epoch number
        """
        for agent in agents:
            snapshot = AgentMemorySnapshot.from_agents(  # type: ignore[attr-defined]
                agent=agent, run_id=run_id, epoch=epoch
            )
            self.save(agent.agent_id, snapshot)

        logger.info(f"Saved memory snapshots for {len(agents)} agents (run={run_id}, epoch={epoch})")

    def load_all(self) -> Dict[str, AgentMemorySnapshot]:
        """Bulk load most recent memory snapshots for all agents.

        Returns:
            Dict mapping agent_id -> most recent AgentMemorySnapshot
        """
        with self._read_lock():
            result = {}
            for agent_id in self._snapshots:
                if agent_id not in self._snapshots or not self._snapshots[agent_id]:
                    continue
                try:
                    data = self._snapshots[agent_id][-1]
                    snapshot = AgentMemorySnapshot.from_dict(data)
                    snapshot.validate()
                    result[agent_id] = snapshot
                except (IndexError, ValueError) as e:
                    logger.warning(f"Failed to load snapshot for {agent_id}: {e}")
            return result

    def get_relationship(
        self, agent_a: str, agent_b: str
    ) -> Optional[RelationshipEdge]:
        """Get relationship edge between two agents.

        Args:
            agent_a: Source agent ID
            agent_b: Target agent ID

        Returns:
            RelationshipEdge if exists, None otherwise
        """
        with self._read_lock():
            for rel_data in self._relationships:
                if rel_data["agent_a"] == agent_a and rel_data["agent_b"] == agent_b:
                    return RelationshipEdge.from_dict(rel_data)
            return None

    def add_relationship_event(
        self,
        agent_a: str,
        agent_b: str,
        event_data: Dict[str, Any],
    ) -> None:
        """Record an interaction event between two agents.

        Updates or creates a RelationshipEdge with the new interaction data.

        Args:
            agent_a: Initiator/source agent ID
            agent_b: Counterparty/target agent ID
            event_data: Dict with keys: p (quality), payoff_a, payoff_b, epoch
                       All are required.

        Raises:
            ValueError: If event_data missing required keys or invalid values
        """
        required_keys = {"p", "payoff_a", "payoff_b", "epoch"}
        if not required_keys.issubset(event_data.keys()):
            raise ValueError(f"event_data missing required keys: {required_keys}")

        p = event_data["p"]
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"event_data['p'] out of bounds [0, 1]: {p}")

        payoff_a = event_data["payoff_a"]
        payoff_b = event_data["payoff_b"]
        epoch = event_data["epoch"]

        # Find or create relationship edge
        edge = self.get_relationship(agent_a, agent_b)
        if edge is None:
            edge = RelationshipEdge(agent_a=agent_a, agent_b=agent_b)
        else:
            # Remove old entry
            self._relationships = [
                r
                for r in self._relationships
                if not (r["agent_a"] == agent_a and r["agent_b"] == agent_b)
            ]

        # Update aggregates
        old_count = edge.interaction_count
        old_avg_p = edge.avg_p
        new_count = old_count + 1
        new_avg_p = (old_avg_p * old_count + p) / new_count

        edge.interaction_count = new_count
        edge.avg_p = new_avg_p
        edge.total_payoff_a += payoff_a
        edge.total_payoff_b += payoff_b
        edge.last_interaction_epoch = epoch

        # Optional: update trust if provided
        if "trust_a_to_b" in event_data:
            trust = event_data["trust_a_to_b"]
            if not (0.0 <= trust <= 1.0):
                raise ValueError(f"trust_a_to_b out of bounds: {trust}")
            edge.trust_a_to_b = trust

        if "trust_b_to_a" in event_data:
            trust = event_data["trust_b_to_a"]
            if not (0.0 <= trust <= 1.0):
                raise ValueError(f"trust_b_to_a out of bounds: {trust}")
            edge.trust_b_to_a = trust

        edge.validate()
        self._relationships.append(edge.to_dict())
        self._persist()

    def get_all_relationships(self) -> List[RelationshipEdge]:
        """Get all recorded relationship edges.

        Returns:
            List of all RelationshipEdge objects
        """
        with self._read_lock():
            return [RelationshipEdge.from_dict(r) for r in self._relationships]

    def get_agent_relationships(self, agent_id: str) -> List[RelationshipEdge]:
        """Get all relationships involving an agent (as initiator).

        Args:
            agent_id: Agent ID

        Returns:
            List of RelationshipEdge where agent_a == agent_id
        """
        with self._read_lock():
            return [
                RelationshipEdge.from_dict(r)
                for r in self._relationships
                if r["agent_a"] == agent_id
            ]

    def get_metadata(self) -> Dict[str, Any]:
        """Get store metadata (creation time, last update, etc).

        Returns:
            Dict with metadata
        """
        with self._read_lock():
            return self._metadata.copy()

    def clear(self) -> None:
        """Clear all stored data (for testing)."""
        self._initialize_empty()
        self._persist()


# Class method for AgentMemorySnapshot to extract from BaseAgent
# This is injected as a module-level helper to avoid circular imports
def _from_agents_helper(
    agent: "BaseAgent", run_id: str, epoch: int
) -> AgentMemorySnapshot:
    """Extract memory snapshot from a BaseAgent instance.

    Args:
        agent: BaseAgent instance
        run_id: Run identifier
        epoch: Current epoch

    Returns:
        AgentMemorySnapshot
    """
    # Compute total payoff from recent interactions
    total_payoff = 0.0
    interaction_summaries: List[Dict[str, Any]] = []

    for interaction in agent.get_interaction_history():
        if interaction.accepted:
            # Measure payoff from interaction quality probability
            payoff = interaction.p  # Proxy for expected payoff based on interaction quality
            total_payoff += payoff

            # Create summary (don't store full SoftInteraction object)
            summary = {
                "interaction_id": interaction.interaction_id,
                "counterparty": (
                    interaction.counterparty
                    if interaction.initiator == agent.agent_id
                    else interaction.initiator
                ),
                "p": interaction.p,
                "accepted": interaction.accepted,
                "type": interaction.interaction_type.value,
            }
            interaction_summaries.append(summary)

    snapshot = AgentMemorySnapshot(
        agent_id=agent.agent_id,
        agent_type=agent.agent_type.value,
        counterparty_trust=agent.get_counterparty_memory(),
        interaction_summaries=interaction_summaries,
        total_interactions=len(agent._interaction_history),
        total_payoff=total_payoff,
        run_id=run_id,
        epoch=epoch,
        timestamp=datetime.now().isoformat(),
    )
    return snapshot


# Attach helper as staticmethod
AgentMemorySnapshot.from_agents = staticmethod(  # type: ignore[attr-defined]
    lambda agent, run_id, epoch: _from_agents_helper(agent, run_id, epoch)
)
