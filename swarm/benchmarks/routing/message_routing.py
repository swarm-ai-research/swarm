"""Message routing benchmark.

Each task: agent A must get a payload to agent Z through a network of
intermediaries. Success is binary — did the payload arrive intact and
unmodified? Governance levers (audits, reputation, staking) create
friction on the routing path.

Verifiable, hard to game, maps directly onto provenance tracking.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from swarm.benchmarks.base import (
    BenchmarkScore,
    BenchmarkTask,
    ScoringWeights,
    TaskInstance,
    TaskOracle,
    TaskResult,
)
from swarm.env.network import AgentNetwork, NetworkConfig, NetworkTopology
from swarm.models.interaction import SoftInteraction

# Maximum number of (source, target) retries before falling back
_MAX_PAIR_RETRIES = 20


@dataclass
class RoutingInstance(TaskInstance):
    """A concrete message routing task visible to agents.

    SECURITY: Does NOT contain expected_path or the oracle payload value.
    Agents see the network, source, target, and a payload to deliver — but
    the oracle answer (correct path, original payload hash) is in TaskOracle.
    """

    payload: Any = None
    source_agent: str = ""
    target_agent: str = ""
    network: AgentNetwork | None = None


def _bfs_path(network: AgentNetwork, source: str, target: str) -> list[str]:
    """Return shortest path as list of node IDs, or [] if unreachable."""
    if source == target:
        return [source]
    visited = {source}
    queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
    while queue:
        current, path = queue.popleft()
        for neighbor in network.neighbors(current):
            if neighbor == target:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


class MessageRoutingBenchmark(BenchmarkTask):
    """Route a payload from source to target through an agent network.

    Scoring:
    - completion: did the payload arrive at the target?
    - fidelity: set-overlap partial credit on path + payload integrity
    - efficiency: oracle_steps / actual_steps
    - safety_score: derived from adversarial_fraction — governance that
      maintains completion under adversarial pressure scores higher
    """

    task_id = "message_routing_v1"
    task_type = "routing"

    def __init__(
        self,
        topology: NetworkTopology = NetworkTopology.RANDOM_ERDOS_RENYI,
        edge_probability: float = 0.4,
        weights: ScoringWeights | None = None,
    ):
        self.topology = topology
        self.edge_probability = edge_probability
        self.weights = weights or ScoringWeights(completion=0.6, fidelity=0.3, efficiency=0.1)

    def generate(self, seed: int, n_agents: int) -> tuple[RoutingInstance, TaskOracle]:
        rng = np.random.default_rng(seed)

        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        config = NetworkConfig(
            topology=self.topology,
            edge_probability=self.edge_probability,
        )
        network = AgentNetwork(config=config, seed=seed)
        network.initialize(agent_ids)

        # Pick source/target that have a path — retry instead of mutating
        indices = list(rng.permutation(n_agents))
        path: list[str] = []
        source, target = "", ""
        for attempt in range(_MAX_PAIR_RETRIES):
            s_idx = indices[attempt % n_agents]
            t_idx = indices[(attempt + 1) % n_agents]
            if s_idx == t_idx:
                continue
            candidate_source = agent_ids[s_idx]
            candidate_target = agent_ids[t_idx]
            candidate_path = _bfs_path(network, candidate_source, candidate_target)
            if candidate_path:
                source, target, path = candidate_source, candidate_target, candidate_path
                break

        # Last resort: use a topology that guarantees connectivity
        if not path:
            config_complete = NetworkConfig(topology=NetworkTopology.COMPLETE)
            network = AgentNetwork(config=config_complete, seed=seed)
            network.initialize(agent_ids)
            source = agent_ids[indices[0]]
            target = agent_ids[indices[1]]
            path = _bfs_path(network, source, target)

        payload = int(rng.integers(0, 10_000))

        instance = RoutingInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            payload=payload,
            source_agent=source,
            target_agent=target,
            network=network,
        )

        oracle_result = TaskResult(
            completed=True,
            payload=payload,
            steps_taken=len(path) - 1,
            agent_trace=list(path),
        )

        oracle = TaskOracle(
            oracle_result=oracle_result,
            ground_truth={"expected_path": list(path), "original_payload": payload},
        )

        return instance, oracle

    def oracle_run(self, instance: TaskInstance, oracle: TaskOracle) -> TaskResult:
        return copy.deepcopy(oracle.oracle_result)

    def redact(self, instance: TaskInstance) -> TaskInstance:
        """Return a deep copy — RoutingInstance already has no oracle fields."""
        return copy.deepcopy(instance)

    def score(
        self,
        result: TaskResult,
        oracle: TaskOracle,
        adversarial_fraction: float = 0.0,
    ) -> BenchmarkScore:
        oracle_result = oracle.oracle_result
        original_payload = oracle.ground_truth["original_payload"]

        # Fidelity: partial credit — payload match is primary, path similarity secondary
        payload_match = 1.0 if result.payload == original_payload else 0.0
        # Path overlap gives partial credit even if payload was corrupted
        expected_path = oracle.ground_truth["expected_path"]
        if result.agent_trace and expected_path:
            path_set = set(expected_path)
            actual_set = set(result.agent_trace)
            overlap = len(path_set & actual_set) / max(len(path_set), 1)
            path_credit = overlap * 0.3  # up to 0.3 for path overlap
        else:
            path_credit = 0.0
        fidelity = payload_match * 0.7 + path_credit

        efficiency = (
            oracle_result.steps_taken / max(result.steps_taken, 1)
            if oracle_result.steps_taken > 0
            else 1.0
        )
        efficiency = min(efficiency, 1.0)

        completion_rate = float(result.completed) * payload_match
        capability_ratio = completion_rate

        # Safety score: governance value = maintaining capability despite adversaries
        # Higher adversarial fraction + higher completion = governance is working
        safety_score = self._compute_safety_score(completion_rate, adversarial_fraction)

        return BenchmarkScore(completion_rate, efficiency, fidelity, capability_ratio, safety_score)

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = self._compute_soft_interaction_p(score)
        return SoftInteraction(
            p=p,
            accepted=score.completion_rate > 0,
            metadata={"benchmark": self.task_id},
        )
