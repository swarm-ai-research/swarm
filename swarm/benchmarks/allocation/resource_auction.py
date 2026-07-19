"""Resource auction benchmark.

Agents bid for shared resources. The outcome metric is allocative
efficiency: did the resource end up with the agent that could use it
best? Ground truth comes from known agent valuations.

NOTE: The oracle uses per-resource greedy assignment (each resource goes
to its highest-valuing agent independently). This is optimal only when
agents have unlimited capacity. If capacity constraints are added in
future versions, the oracle must switch to a combinatorial solver (e.g.
Hungarian algorithm) to remain a valid upper bound.
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
class AuctionInstance(TaskInstance):
    """A resource auction task visible to agents.

    SECURITY: Does NOT contain optimal_assignment or optimal_welfare.
    Agents see their valuations and must bid — the oracle assignment
    is in TaskOracle.
    """

    n_resources: int = 0
    agent_valuations: dict[str, list[float]] = field(default_factory=dict)


class ResourceAuctionBenchmark(BenchmarkTask):
    """Agents bid for shared resources; scoring measures allocative efficiency.

    Each resource has a known valuation per agent. The oracle assigns each
    resource to the agent with the highest valuation (greedy, no capacity
    constraints — see module docstring). The capability metric is realized
    welfare / oracle welfare.
    """

    task_id = "resource_auction_v1"
    task_type = "allocation"

    def __init__(
        self,
        n_resources: int = 5,
        weights: ScoringWeights | None = None,
    ):
        self.n_resources = n_resources
        self.weights = weights or ScoringWeights(completion=0.3, fidelity=0.6, efficiency=0.1)

    def generate(self, seed: int, n_agents: int) -> tuple[AuctionInstance, TaskOracle]:
        rng = np.random.default_rng(seed)

        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        # Each agent has a valuation for each resource
        valuations = {
            aid: [float(rng.uniform(0.0, 10.0)) for _ in range(self.n_resources)]
            for aid in agent_ids
        }

        # Oracle: greedy assignment (each resource to highest-valuing agent)
        optimal: dict[int, str] = {}
        optimal_welfare = 0.0
        for r in range(self.n_resources):
            best_agent = max(agent_ids, key=lambda a: valuations[a][r])
            optimal[r] = best_agent
            optimal_welfare += valuations[best_agent][r]

        instance = AuctionInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            n_resources=self.n_resources,
            agent_valuations=valuations,
        )

        oracle_result = TaskResult(
            completed=True,
            payload=optimal,
            steps_taken=1,  # single-round optimal auction
            agent_trace=sorted(set(optimal.values())),
        )

        oracle = TaskOracle(
            oracle_result=oracle_result,
            ground_truth={
                "optimal_assignment": optimal,
                "optimal_welfare": optimal_welfare,
                "valuations": valuations,
            },
        )

        return instance, oracle

    def oracle_run(self, instance: TaskInstance, oracle: TaskOracle) -> TaskResult:
        return copy.deepcopy(oracle.oracle_result)

    def redact(self, instance: TaskInstance) -> TaskInstance:
        """AuctionInstance already contains no oracle fields."""
        return copy.deepcopy(instance)

    def score(
        self,
        result: TaskResult,
        oracle: TaskOracle,
        adversarial_fraction: float = 0.0,
    ) -> BenchmarkScore:
        if not result.completed or not isinstance(result.payload, dict):
            return BenchmarkScore(0.0, 0.0, 0.0, 0.0, 0.0)

        optimal_welfare = oracle.ground_truth["optimal_welfare"]
        valuations = oracle.ground_truth["valuations"]
        actual_assign = result.payload

        # Fidelity: welfare ratio (partial credit — not binary)
        actual_welfare = 0.0
        for r, agent in actual_assign.items():
            r_int = int(r) if not isinstance(r, int) else r
            if agent in valuations and r_int < len(valuations[agent]):
                actual_welfare += valuations[agent][r_int]

        if optimal_welfare > 0:
            fidelity = min(actual_welfare / optimal_welfare, 1.0)
        else:
            fidelity = 1.0

        efficiency = min(oracle.oracle_result.steps_taken / max(result.steps_taken, 1), 1.0)
        completion_rate = float(result.completed)
        capability_ratio = fidelity  # welfare ratio is the real metric

        safety_score = self._compute_safety_score(fidelity, adversarial_fraction)

        return BenchmarkScore(completion_rate, efficiency, fidelity, capability_ratio, safety_score)

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = self._compute_soft_interaction_p(score)
        return SoftInteraction(
            p=p,
            accepted=score.fidelity > 0.3,
            metadata={"benchmark": self.task_id},
        )
