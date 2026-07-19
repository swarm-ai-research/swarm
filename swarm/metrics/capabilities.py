"""Emergent capability measurement metrics.

Measures emergent behaviors arising from multi-agent collaboration:
- Coordination efficiency
- Skill complementarity
- Information aggregation
- Collective problem-solving
- Task completion synergy
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from swarm.env.composite_tasks import (
    CapabilityType,
    CompositeTask,
    CompositeTaskStatus,
)


@dataclass
class AgentCapabilityProfile:
    """Profile of an agent's capabilities and performance."""

    agent_id: str
    capabilities: Set[CapabilityType] = field(default_factory=set)

    # Performance metrics
    tasks_completed: int = 0
    subtasks_completed: int = 0
    avg_quality: float = 0.0

    # Collaboration metrics
    unique_collaborators: Set[str] = field(default_factory=set)
    collaboration_count: int = 0

    # Capability-specific performance
    capability_scores: Dict[CapabilityType, float] = field(default_factory=dict)

    def update_quality(self, quality: float) -> None:
        """Update average quality with new observation."""
        n = self.subtasks_completed
        if n == 0:
            self.avg_quality = quality
        else:
            self.avg_quality = (self.avg_quality * n + quality) / (n + 1)
        self.subtasks_completed += 1


@dataclass
class EmergentCapabilityMetrics:
    """Metrics measuring emergent capabilities in multi-agent systems."""

    # Task-level metrics
    total_composite_tasks: int = 0
    completed_composite_tasks: int = 0
    avg_completion_time: float = 0.0
    avg_final_quality: float = 0.0

    # Coordination metrics
    avg_coordination_score: float = 0.0
    avg_synergy_score: float = 0.0
    avg_information_flow: float = 0.0

    # Team formation metrics
    avg_team_size: float = 0.0
    capability_coverage: float = 0.0  # How well teams cover required capabilities

    # Efficiency metrics
    task_efficiency: float = 0.0  # Quality per step
    parallelization: float = 0.0  # How much work done in parallel

    # Emergent behavior indicators
    specialization_index: float = 0.0  # Do agents specialize?
    complementarity_score: float = 0.0  # Do teams have complementary skills?
    knowledge_transfer: float = 0.0  # Does quality improve along dependencies?


class CapabilityAnalyzer:
    """
    Analyzes emergent capabilities in multi-agent collaboration.

    Tracks:
    - Individual agent capability profiles
    - Team composition patterns
    - Task completion metrics
    - Emergent behavior indicators
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize analyzer."""
        self._rng = np.random.default_rng(seed)
        self._agent_profiles: Dict[str, AgentCapabilityProfile] = {}
        self._completed_tasks: List[CompositeTask] = []
        self._team_compositions: List[Tuple[str, Set[str]]] = []  # (task_id, agent_ids)

    def register_agent(
        self,
        agent_id: str,
        capabilities: Set[CapabilityType],
    ) -> AgentCapabilityProfile:
        """Register an agent with their capabilities."""
        profile = AgentCapabilityProfile(
            agent_id=agent_id,
            capabilities=capabilities,
        )
        self._agent_profiles[agent_id] = profile
        return profile

    def get_agent_profile(self, agent_id: str) -> Optional[AgentCapabilityProfile]:
        """Get agent's capability profile."""
        return self._agent_profiles.get(agent_id)

    def record_task_completion(self, task: CompositeTask) -> None:
        """Record a completed composite task for analysis."""
        if task.status != CompositeTaskStatus.COMPLETED:
            return

        self._completed_tasks.append(task)
        self._team_compositions.append((task.task_id, task.participating_agents.copy()))

        # Update agent profiles
        for subtask in task.subtasks:
            if subtask.assigned_to and subtask.quality_score is not None:
                profile = self._agent_profiles.get(subtask.assigned_to)
                if profile:
                    profile.update_quality(subtask.quality_score)

                    # Update capability-specific scores
                    for cap in subtask.required_capabilities:
                        if cap in profile.capabilities:
                            old_score = profile.capability_scores.get(cap, 0.5)
                            # Exponential moving average
                            profile.capability_scores[cap] = (
                                0.7 * old_score + 0.3 * subtask.quality_score
                            )

        # Update collaboration counts
        agents = list(task.participating_agents)
        for i, agent_a in enumerate(agents):
            profile_a = self._agent_profiles.get(agent_a)
            if profile_a:
                profile_a.collaboration_count += 1
                for agent_b in agents[i + 1 :]:
                    profile_a.unique_collaborators.add(agent_b)
                    profile_b = self._agent_profiles.get(agent_b)
                    if profile_b:
                        profile_b.unique_collaborators.add(agent_a)

    def compute_metrics(self) -> EmergentCapabilityMetrics:
        """Compute emergent capability metrics from recorded data."""
        metrics = EmergentCapabilityMetrics()

        if not self._completed_tasks:
            return metrics

        # Basic task metrics
        metrics.total_composite_tasks = len(self._completed_tasks)
        metrics.completed_composite_tasks = len(self._completed_tasks)

        qualities = [t.final_quality for t in self._completed_tasks if t.final_quality]
        if qualities:
            metrics.avg_final_quality = float(np.mean(qualities))

        # Coordination metrics (from task-level scores)
        metrics.avg_coordination_score = float(
            np.mean([t.coordination_score for t in self._completed_tasks])
        )
        metrics.avg_synergy_score = float(
            np.mean([t.synergy_score for t in self._completed_tasks])
        )
        metrics.avg_information_flow = float(
            np.mean([t.information_flow_score for t in self._completed_tasks])
        )

        # Team formation metrics
        team_sizes = [len(t.participating_agents) for t in self._completed_tasks]
        metrics.avg_team_size = float(np.mean(team_sizes))

        # Capability coverage
        coverage_scores = []
        for task in self._completed_tasks:
            team_caps = set()
            for agent_id in task.participating_agents:
                profile = self._agent_profiles.get(agent_id)
                if profile:
                    team_caps.update(profile.capabilities)
            if task.required_capabilities:
                coverage = len(team_caps & task.required_capabilities) / len(
                    task.required_capabilities
                )
                coverage_scores.append(coverage)
        if coverage_scores:
            metrics.capability_coverage = float(np.mean(coverage_scores))

        # Efficiency metrics
        metrics.task_efficiency = self._compute_task_efficiency()
        metrics.parallelization = self._compute_parallelization()

        # Emergent behavior indicators
        metrics.specialization_index = self._compute_specialization()
        metrics.complementarity_score = self._compute_complementarity()
        metrics.knowledge_transfer = self._compute_knowledge_transfer()

        return metrics

    def _compute_task_efficiency(self) -> float:
        """Compute quality per step across tasks."""
        efficiencies = []
        for task in self._completed_tasks:
            total_steps = sum(st.actual_steps for st in task.subtasks)
            if total_steps > 0 and task.final_quality:
                efficiencies.append(task.final_quality / total_steps)
        return float(np.mean(efficiencies)) if efficiencies else 0.0

    def _compute_parallelization(self) -> float:
        """Compute how much work was done in parallel."""
        parallel_scores = []
        for task in self._completed_tasks:
            # Count subtasks that could run in parallel (same depth in DAG)
            depths = self._compute_subtask_depths(task)
            if depths:
                n_subtasks = len(task.subtasks)
                # Perfect parallelization = all at depth 0
                # Sequential = depths 0,1,2,...,n-1
                if n_subtasks > 1:
                    avg_depth = np.mean(list(depths.values()))
                    # Normalize: 0 = sequential, 1 = fully parallel
                    parallel_scores.append(1.0 - (avg_depth / (n_subtasks - 1)))
        return float(np.mean(parallel_scores)) if parallel_scores else 0.0

    def _compute_subtask_depths(self, task: CompositeTask) -> Dict[str, int]:
        """Compute dependency depth for each subtask."""
        depths: Dict[str, int] = {}

        def get_depth(subtask_id: str) -> int:
            if subtask_id in depths:
                return depths[subtask_id]

            subtask = task.get_subtask(subtask_id)
            if not subtask or not subtask.dependencies:
                depths[subtask_id] = 0
                return 0

            max_dep_depth = max(get_depth(dep) for dep in subtask.dependencies)
            depths[subtask_id] = max_dep_depth + 1
            return depths[subtask_id]

        for st in task.subtasks:
            get_depth(st.subtask_id)

        return depths

    def _compute_specialization(self) -> float:
        """
        Compute specialization index.

        Higher value = agents specialize in specific capabilities.
        """
        if not self._agent_profiles:
            return 0.0

        # For each agent, compute entropy of their capability usage
        entropies = []
        for profile in self._agent_profiles.values():
            if not profile.capability_scores:
                continue

            scores = list(profile.capability_scores.values())
            if len(scores) < 2:
                continue

            # Normalize to probabilities
            total = sum(scores)
            if total > 0:
                probs = [s / total for s in scores]
                # Compute entropy (0 = specialized, high = generalist)
                entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
                max_entropy = np.log2(len(scores))
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                    entropies.append(
                        1.0 - normalized_entropy
                    )  # Invert: 1 = specialized

        return np.mean(entropies) if entropies else 0.0

    def _compute_complementarity(self) -> float:
        """
        Compute team complementarity score.

        Higher value = teams have less capability overlap.
        """
        if not self._team_compositions:
            return 0.0

        complementarity_scores = []
        for _task_id, agent_ids in self._team_compositions:
            if len(agent_ids) < 2:
                continue

            # Get capability sets for each agent
            cap_sets = []
            for agent_id in agent_ids:
                profile = self._agent_profiles.get(agent_id)
                if profile:
                    cap_sets.append(profile.capabilities)

            if len(cap_sets) < 2:
                continue

            # Compute pairwise overlap
            overlaps = []
            for i, caps_a in enumerate(cap_sets):
                for caps_b in cap_sets[i + 1 :]:
                    if caps_a or caps_b:
                        intersection = len(caps_a & caps_b)
                        union = len(caps_a | caps_b)
                        jaccard = intersection / union if union > 0 else 0
                        overlaps.append(1.0 - jaccard)  # Complement of overlap

            if overlaps:
                complementarity_scores.append(np.mean(overlaps))

        return np.mean(complementarity_scores) if complementarity_scores else 0.0

    def _compute_knowledge_transfer(self) -> float:
        """
        Compute knowledge transfer score.

        Measures if quality improves along task dependencies.
        """
        improvements = []
        for task in self._completed_tasks:
            for subtask in task.subtasks:
                if not subtask.dependencies or subtask.quality_score is None:
                    continue

                # Get quality of dependencies
                dep_qualities = []
                for dep_id in subtask.dependencies:
                    dep = task.get_subtask(dep_id)
                    if dep and dep.quality_score is not None:
                        dep_qualities.append(dep.quality_score)

                if dep_qualities:
                    avg_dep_quality = np.mean(dep_qualities)
                    improvement = subtask.quality_score - avg_dep_quality
                    # Normalize to [-1, 1] range approximately
                    improvements.append(max(-1, min(1, float(improvement))))

        if improvements:
            # Transform to [0, 1] where 0.5 = no change, 1 = improvement
            return 0.5 + 0.5 * float(np.mean(improvements))
        return 0.5


def compute_collective_intelligence_score(
    metrics: EmergentCapabilityMetrics,
) -> float:
    """
    Compute an overall collective intelligence score.

    Combines multiple metrics into a single score representing
    the emergent problem-solving capability of the multi-agent system.
    """
    # Weight different components
    weights = {
        "quality": 0.2,
        "coordination": 0.15,
        "synergy": 0.2,
        "info_flow": 0.15,
        "efficiency": 0.1,
        "complementarity": 0.1,
        "transfer": 0.1,
    }

    score = (
        weights["quality"] * metrics.avg_final_quality
        + weights["coordination"] * metrics.avg_coordination_score
        + weights["synergy"] * metrics.avg_synergy_score
        + weights["info_flow"] * metrics.avg_information_flow
        + weights["efficiency"] * min(1.0, metrics.task_efficiency * 10)  # Normalize
        + weights["complementarity"] * metrics.complementarity_score
        + weights["transfer"] * metrics.knowledge_transfer
    )

    return float(score)


@dataclass
class CapabilityEnvelopeSnapshot:
    """Single point-in-time measurement of capability vs governance coverage."""

    step: int
    agent_id: str
    capability_count: int  # distinct capabilities demonstrated
    governed_count: int  # capabilities with governance coverage
    capability_envelope: float  # fraction of all capability types demonstrated
    governance_coverage: float  # governed / demonstrated (1.0 = fully governed)
    governance_gap: float  # capability_envelope - (envelope * coverage)


@dataclass
class CapabilityEnvelopeResult:
    """Aggregate result across agents and time steps."""

    snapshots: List[CapabilityEnvelopeSnapshot] = field(default_factory=list)
    mean_envelope: float = 0.0
    mean_governance_coverage: float = 0.0
    mean_governance_gap: float = 0.0
    max_governance_gap: float = 0.0


def capability_envelope(
    agent_profiles: Dict[str, AgentCapabilityProfile],
    governed_capabilities: Dict[str, Set[CapabilityType]],
    step: int = 0,
    all_capability_types: Optional[Set[CapabilityType]] = None,
) -> CapabilityEnvelopeResult:
    """Compute capability envelope metric for a set of agents.

    Measures the expanding input set each agent can handle vs. how much of
    that space is covered by governance mechanisms (monitoring, safety checks,
    circuit breakers, etc.).

    A governance gap > 0 means the agent can act in domains that governance
    does not yet cover — the safety-critical signal from Zhang et al.
    Hyperagents (arXiv:2603.19461).

    Args:
        agent_profiles: Map of agent_id to their capability profile.
        governed_capabilities: Map of agent_id to the set of capability types
            that have active governance coverage for that agent.
        step: Current simulation step (for time-series tracking).
        all_capability_types: Universe of capability types. Defaults to all
            members of ``CapabilityType`` if not provided.

    Returns:
        CapabilityEnvelopeResult with per-agent snapshots and aggregates.
    """
    if all_capability_types is None:
        all_capability_types = set(CapabilityType)

    n_types = len(all_capability_types)
    if n_types == 0:
        return CapabilityEnvelopeResult()

    snapshots: List[CapabilityEnvelopeSnapshot] = []

    for agent_id, profile in agent_profiles.items():
        demonstrated = profile.capabilities & all_capability_types
        governed = governed_capabilities.get(agent_id, set()) & demonstrated

        cap_count = len(demonstrated)
        gov_count = len(governed)
        envelope = cap_count / n_types  # fraction of universe demonstrated
        coverage = gov_count / cap_count if cap_count > 0 else 1.0
        gap = envelope * (1.0 - coverage)  # ungoverned fraction of universe

        snapshots.append(
            CapabilityEnvelopeSnapshot(
                step=step,
                agent_id=agent_id,
                capability_count=cap_count,
                governed_count=gov_count,
                capability_envelope=envelope,
                governance_coverage=coverage,
                governance_gap=gap,
            )
        )

    if not snapshots:
        return CapabilityEnvelopeResult()

    envelopes = [s.capability_envelope for s in snapshots]
    coverages = [s.governance_coverage for s in snapshots]
    gaps = [s.governance_gap for s in snapshots]

    return CapabilityEnvelopeResult(
        snapshots=snapshots,
        mean_envelope=float(np.mean(envelopes)),
        mean_governance_coverage=float(np.mean(coverages)),
        mean_governance_gap=float(np.mean(gaps)),
        max_governance_gap=float(np.max(gaps)),
    )


def analyze_capability_distribution(
    profiles: List[AgentCapabilityProfile],
) -> Dict[str, float]:
    """
    Analyze the distribution of capabilities across agents.

    Returns statistics about capability coverage and diversity.
    """
    if not profiles:
        return {}

    # Count capability occurrences
    cap_counts: Dict[CapabilityType, int] = defaultdict(int)
    for profile in profiles:
        for cap in profile.capabilities:
            cap_counts[cap] += 1

    n_agents = len(profiles)
    n_capabilities = len(CapabilityType)

    # Compute metrics
    caps_per_agent = np.mean([len(p.capabilities) for p in profiles])
    agents_per_cap = np.mean(list(cap_counts.values())) if cap_counts else 0

    # Coverage: what fraction of capabilities are present?
    coverage = len(cap_counts) / n_capabilities

    # Balance: how evenly distributed are capabilities?
    if cap_counts:
        counts = list(cap_counts.values())
        if max(counts) > 0:
            balance = min(counts) / max(counts)
        else:
            balance = 0.0
    else:
        balance = 0.0

    return {
        "n_agents": n_agents,
        "n_capability_types": len(cap_counts),
        "avg_capabilities_per_agent": float(caps_per_agent),
        "avg_agents_per_capability": float(agents_per_cap),
        "capability_coverage": coverage,
        "capability_balance": balance,
    }
