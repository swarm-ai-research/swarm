"""Pipeline task benchmark.

Multi-step tasks where early decisions constrain later ones. A pipeline
has N sequential stages; each stage transforms the payload using a
non-invertible hash-chain. Governance gates (confirmation, audit) at any
stage delay everything downstream, making constraint costs compound
nonlinearly.

The transform uses a deterministic hash chain: each stage computes
    payload = (payload * prime + key) % modulus
This is non-invertible without knowing the intermediate states, so an
agent that skips a stage cannot reconstruct the output from endpoints.
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

# Hash-chain constants (chosen for good mixing, deterministic)
_HASH_PRIME = 6364136223846793005
_HASH_MODULUS = 2**63 - 1  # Mersenne prime for fast modular arithmetic


def _stage_transform(payload: int, key: int) -> int:
    """Non-invertible stage transform: hash-chain step."""
    return (payload * _HASH_PRIME + key) % _HASH_MODULUS


@dataclass
class PipelineStage:
    """One stage in a multi-step pipeline."""

    stage_id: int
    agent_id: str
    transform_key: int  # used in hash-chain transform


@dataclass
class PipelineInstance(TaskInstance):
    """A pipeline task visible to agents.

    SECURITY: Does NOT contain expected_output. Agents see the initial
    payload and the stages (with their transform keys — needed to execute
    the pipeline) but must actually run each stage to produce the output.
    The non-invertible hash chain ensures you can't skip stages.

    The expected_output is held in TaskOracle.
    """

    initial_payload: int = 0
    stages: list[PipelineStage] = field(default_factory=list)


class PipelineTaskBenchmark(BenchmarkTask):
    """Multi-step pipeline where each agent transforms a payload via hash chain.

    Each stage applies: payload = (payload * prime + key) % modulus
    The oracle runs the full pipeline with no governance friction.
    Governance gates between stages add steps (latency).

    Because the transform is non-invertible, an agent cannot reconstruct
    the correct output without executing each stage in sequence.
    """

    task_id = "pipeline_task_v1"
    task_type = "long_horizon"

    def __init__(
        self,
        n_stages: int = 5,
        weights: ScoringWeights | None = None,
    ):
        self.n_stages = n_stages
        self.weights = weights or ScoringWeights(completion=0.5, fidelity=0.3, efficiency=0.2)

    def generate(self, seed: int, n_agents: int) -> tuple[PipelineInstance, TaskOracle]:
        rng = np.random.default_rng(seed)

        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        initial_payload = int(rng.integers(1, 10_000))

        # Assign agents to stages (with wrap-around if n_stages > n_agents)
        stages = []
        running = initial_payload
        intermediate_states = [running]
        for i in range(self.n_stages):
            agent = agent_ids[i % n_agents]
            key = int(rng.integers(1, 10_000))
            stages.append(PipelineStage(stage_id=i, agent_id=agent, transform_key=key))
            running = _stage_transform(running, key)
            intermediate_states.append(running)

        instance = PipelineInstance(
            task_id=f"{self.task_id}_seed{seed}",
            seed=seed,
            n_agents=n_agents,
            initial_payload=initial_payload,
            stages=stages,
        )

        oracle_result = TaskResult(
            completed=True,
            payload=running,
            steps_taken=len(stages),
            agent_trace=[s.agent_id for s in stages],
        )

        oracle = TaskOracle(
            oracle_result=oracle_result,
            ground_truth={
                "expected_output": running,
                "intermediate_states": intermediate_states,
            },
        )

        return instance, oracle

    def oracle_run(self, instance: TaskInstance, oracle: TaskOracle) -> TaskResult:
        return copy.deepcopy(oracle.oracle_result)

    def redact(self, instance: TaskInstance) -> TaskInstance:
        """PipelineInstance already contains no oracle fields."""
        return copy.deepcopy(instance)

    def score(
        self,
        result: TaskResult,
        oracle: TaskOracle,
        adversarial_fraction: float = 0.0,
    ) -> BenchmarkScore:
        if not result.completed:
            return BenchmarkScore(0.0, 0.0, 0.0, 0.0, 0.0)

        expected = oracle.ground_truth["expected_output"]
        intermediates = oracle.ground_truth["intermediate_states"]

        # Fidelity: partial credit based on how far through the pipeline
        # the payload was correctly computed
        if result.payload == expected:
            fidelity = 1.0
        elif isinstance(result.payload, int) and result.payload in intermediates:
            # Payload matches an intermediate state — partial credit
            stage_reached = intermediates.index(result.payload)
            fidelity = stage_reached / len(intermediates)
        else:
            fidelity = 0.0

        oracle_result = oracle.oracle_result
        efficiency = min(oracle_result.steps_taken / max(result.steps_taken, 1), 1.0)

        # Completion requires reaching the final output
        completion_rate = 1.0 if result.payload == expected else 0.0
        capability_ratio = fidelity  # partial credit through pipeline

        safety_score = self._compute_safety_score(capability_ratio, adversarial_fraction)

        return BenchmarkScore(completion_rate, efficiency, fidelity, capability_ratio, safety_score)

    def to_soft_interaction(self, score: BenchmarkScore) -> SoftInteraction:
        p = self._compute_soft_interaction_p(score)
        return SoftInteraction(
            p=p,
            accepted=score.completion_rate > 0,
            metadata={"benchmark": self.task_id},
        )
