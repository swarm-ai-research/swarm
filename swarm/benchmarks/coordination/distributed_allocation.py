"""Distributed allocation benchmark.

Agents must coordinate to solve a joint task that no single agent can
complete alone. Each agent holds a piece of a total requirement, and they
must collectively allocate their contributions to meet a target.

Governance constraints (friction, staking) directly impede coordination
speed, making the capability/constraint tradeoff visible.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from swarm.benchmarks.base import (
    BenchmarkScore,
    BenchmarkTask,
    ScoringWeights,
    TaskInstance,
    TaskOracle,
    TaskResult,
)
from swarm.models.interaction import SoftInteraction


@dataclass
class AllocationInstance(TaskInstance):
    """A distributed allocation task visible to agents.

    SECURITY: Does NOT contain optimal_allocation. Agents see only
    the target total and their own capacities — the oracle allocation
    is in TaskOracle.
    """

    target_total: float = 0.0
    agent_capacities: dict[str, float] = field(default_factory=dict)


class DistributedAllocationBenchmark(BenchmarkTask):
    """Agents coordinate to collectively reach a target allocation.

    Each agent has a capacity (max contribution). The oracle allocation
    distributes proportionally. Scoring measures how close the actual
    total is to the target, and how efficiently agents coordinated.
    """

    task_id = "distributed_allocation_v1"
    task_type = "coordination"

    def __init__(
        self,
        target_ratio: float = 0.7,
        weights: ScoringWeights | None = None,
    ):
        """target_ratio: fraction of total capacity that must be allocated."""
        self.target_ratio = target_ratio
        self.weights = weights or ScoringWeights(completion=0.5, fidelity=0.3, efficiency=0.2)

    def generate(self, seed: int, n_agents: int) -> tuple[AllocationInstance, TaskOracle]:
        rng = np.random.default_rng(seed)

        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        capacities = {aid: float(rng.uniform(1.0, 10.0)) for aid in agent_ids}
        total_capacity = sum(capacities.values())
        target = total_capacity * self.target_ratio

        # Oracle: proportional allocation
        optimal = {
            aid: cap * self.target_ratio for aid, cap in capacities.items()
        }

        instance = AllocationInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            target_total=target,
            agent_capacities=capacities,
        )

        oracle_result = TaskResult(
            completed=True,
            payload=optimal,
            steps_taken=1,  # perfect coordination in one round
            agent_trace=sorted(capacities.keys()),
        )

        oracle = TaskOracle(
            oracle_result=oracle_result,
            ground_truth={
                "optimal_allocation": optimal,
                "target_total": target,
            },
        )

        return instance, oracle

    def oracle_run(self, instance: TaskInstance, oracle: TaskOracle) -> TaskResult:
        return copy.deepcopy(oracle.oracle_result)

    def redact(self, instance: TaskInstance) -> TaskInstance:
        """AllocationInstance already contains no oracle fields."""
        return copy.deepcopy(instance)

    def score(
        self,
        result: TaskResult,
        oracle: TaskOracle,
        adversarial_fraction: float = 0.0,
    ) -> BenchmarkScore:
        if not result.completed or not isinstance(result.payload, dict):
            return BenchmarkScore(0.0, 0.0, 0.0, 0.0, 0.0)

        oracle_alloc = oracle.ground_truth["optimal_allocation"]
        actual_alloc = result.payload

        # Fidelity: partial credit — how close is actual total to oracle total
        oracle_total = sum(oracle_alloc.values())
        actual_total = sum(actual_alloc.values())
        if oracle_total > 0:
            fidelity = max(0.0, 1.0 - abs(actual_total - oracle_total) / oracle_total)
        else:
            fidelity = 1.0

        # Per-agent accuracy gives additional partial credit
        if oracle_alloc:
            per_agent_err = []
            for aid in oracle_alloc:
                expected = oracle_alloc[aid]
                actual = actual_alloc.get(aid, 0.0)
                if expected > 0:
                    per_agent_err.append(abs(actual - expected) / expected)
                else:
                    per_agent_err.append(0.0 if actual == 0.0 else 1.0)
            agent_accuracy = max(0.0, 1.0 - (sum(per_agent_err) / len(per_agent_err)))
            fidelity = 0.6 * fidelity + 0.4 * agent_accuracy

        # Efficiency: fewer rounds is better
        oracle_result = oracle.oracle_result
        efficiency = min(oracle_result.steps_taken / max(result.steps_taken, 1), 1.0)

        completion_rate = fidelity  # partial credit for close-enough allocation
        capability_ratio = completion_rate  # oracle always 1.0

        safety_score = self._compute_safety_score(completion_rate, adversarial_fraction)

        return BenchmarkScore(completion_rate, efficiency, fidelity, capability_ratio, safety_score)

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = self._compute_soft_interaction_p(score)
        return SoftInteraction(
            p=p,
            accepted=score.completion_rate > 0.5,
            metadata={"benchmark": self.task_id},
        )
