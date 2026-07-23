"""Time horizon metrics for agent reliability measurement.

Based on Herbie Bradley's framework from "Glimpses of AI Progress" (Pathways AI, 2025):
Agent capability is measured by reliable task completion across increasing time horizons.

Current benchmarks (as of early 2025):
- 10-minute tasks: ~80% reliability
- 1-hour tasks: ~50% reliability (estimated)
- 8-hour tasks: target for mid-2026

This module provides metrics for measuring and tracking agent reliability
across different task durations in multi-agent simulations.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimeHorizonBucket:
    """Reliability statistics for a specific time horizon."""

    horizon_minutes: int
    total_tasks: int = 0
    successful_tasks: int = 0
    total_duration: float = 0.0
    quality_scores: list[float] = field(default_factory=list)

    @property
    def reliability(self) -> float:
        """Success rate for this time horizon."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    @property
    def mean_quality(self) -> float:
        """Average quality score for completed tasks."""
        if not self.quality_scores:
            return 0.0
        return float(np.mean(self.quality_scores))

    @property
    def mean_duration(self) -> float:
        """Average actual duration in minutes."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_duration / self.total_tasks

    def record(
        self, success: bool, duration_minutes: float, quality: float = 1.0
    ) -> None:
        """Record a task outcome."""
        self.total_tasks += 1
        self.total_duration += duration_minutes
        if success:
            self.successful_tasks += 1
            self.quality_scores.append(quality)


@dataclass
class TimeHorizonMetrics:
    """Track agent reliability across multiple time horizons.

    Implements Bradley's framework for measuring AI capability progression
    through task completion reliability at increasing durations.
    """

    # Standard horizon buckets (in minutes)
    HORIZON_BUCKETS = [1, 5, 10, 30, 60, 120, 480, 1440]  # up to 24 hours

    buckets: dict[int, TimeHorizonBucket] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize standard buckets."""
        if not self.buckets:
            for horizon in self.HORIZON_BUCKETS:
                self.buckets[horizon] = TimeHorizonBucket(horizon_minutes=horizon)

    def _get_bucket(self, duration_minutes: float) -> TimeHorizonBucket:
        """Find appropriate bucket for a task duration."""
        for horizon in self.HORIZON_BUCKETS:
            if duration_minutes <= horizon:
                return self.buckets[horizon]
        # Longer than max bucket - use the largest
        return self.buckets[self.HORIZON_BUCKETS[-1]]

    def record_task(
        self,
        duration_minutes: float,
        success: bool,
        quality: float = 1.0,
    ) -> None:
        """Record a task outcome in the appropriate time bucket."""
        bucket = self._get_bucket(duration_minutes)
        bucket.record(success, duration_minutes, quality)

    def reliability_curve(self) -> dict[int, float]:
        """Get reliability at each time horizon.

        Returns dict mapping horizon (minutes) to reliability (0-1).
        This is the core metric for tracking capability progression.
        """
        return {
            horizon: bucket.reliability
            for horizon, bucket in sorted(self.buckets.items())
            if bucket.total_tasks > 0
        }

    def effective_horizon(self, threshold: float = 0.8) -> int | None:
        """Find the longest horizon where reliability >= threshold.

        Bradley uses 80% as the standard threshold. This metric captures
        "how long can agents work reliably?"

        Returns horizon in minutes, or None if no horizon meets threshold.
        """
        curve = self.reliability_curve()
        effective = None
        for horizon, reliability in sorted(curve.items()):
            if reliability >= threshold:
                effective = horizon
            else:
                # Once we drop below threshold, stop
                break
        return effective

    def horizon_gap(self, target_horizon: int = 480, threshold: float = 0.8) -> float:
        """Measure gap between current capability and target.

        Default target is 8 hours (480 min) at 80% reliability,
        Bradley's prediction for mid-2026.

        Returns ratio of effective_horizon / target_horizon, clamped to [0.0, 1.0].
        Value of 1.0 means target achieved or exceeded.
        """
        effective = self.effective_horizon(threshold)
        if effective is None:
            return 0.0
        return min(effective / target_horizon, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics."""
        return {
            "reliability_curve": self.reliability_curve(),
            "effective_horizon_80": self.effective_horizon(0.8),
            "effective_horizon_90": self.effective_horizon(0.9),
            "horizon_gap": self.horizon_gap(),
            "buckets": {
                horizon: {
                    "total": b.total_tasks,
                    "successful": b.successful_tasks,
                    "reliability": b.reliability,
                    "mean_quality": b.mean_quality,
                }
                for horizon, b in self.buckets.items()
                if b.total_tasks > 0
            },
        }


@dataclass
class AgentCapabilityProfile:
    """Model heterogeneous agent capabilities.

    Based on Bradley's observation about capability diffusion:
    frontier capabilities migrate to smaller models over time.
    This creates heterogeneous agent populations.
    """

    # Capability level (0-1), roughly corresponds to model size/generation
    capability_level: float = 1.0

    # Base reliability at 10-minute horizon
    base_reliability_10min: float = 0.8

    # Decay rate: how much reliability drops per 10x duration increase
    horizon_decay_rate: float = 0.15

    # Compute cost per minute of operation (relative units)
    compute_cost_per_minute: float = 1.0

    def reliability_at_horizon(self, minutes: int) -> float:
        """Estimate reliability at a given time horizon.

        Uses logarithmic decay model: reliability drops as task duration increases.
        """
        if minutes <= 0:
            return self.base_reliability_10min

        # Log-scale decay from 10-minute baseline
        log_ratio = float(np.log10(max(minutes, 1) / 10))
        decay = self.horizon_decay_rate * max(0.0, log_ratio)

        # Apply capability scaling
        base = self.base_reliability_10min * self.capability_level
        result: float = float(max(0.0, min(1.0, base - decay)))
        return result

    def expected_quality(self, minutes: int) -> float:
        """Expected output quality given task duration."""
        reliability = self.reliability_at_horizon(minutes)
        # Quality degrades faster than reliability
        quality: float = float(reliability**1.2)
        return quality

    def compute_cost(self, minutes: int) -> float:
        """Total compute cost for a task of given duration."""
        return self.compute_cost_per_minute * minutes


# Preset capability profiles based on Bradley's observations
CAPABILITY_PROFILES = {
    "frontier": AgentCapabilityProfile(
        capability_level=1.0,
        base_reliability_10min=0.85,
        horizon_decay_rate=0.12,
        compute_cost_per_minute=10.0,
    ),
    "standard": AgentCapabilityProfile(
        capability_level=0.8,
        base_reliability_10min=0.75,
        horizon_decay_rate=0.15,
        compute_cost_per_minute=1.0,
    ),
    "distilled": AgentCapabilityProfile(
        capability_level=0.6,
        base_reliability_10min=0.65,
        horizon_decay_rate=0.18,
        compute_cost_per_minute=0.1,
    ),
    "edge": AgentCapabilityProfile(
        capability_level=0.4,
        base_reliability_10min=0.50,
        horizon_decay_rate=0.20,
        compute_cost_per_minute=0.01,
    ),
}


@dataclass
class ComputeConstraints:
    """Model compute resource constraints on agent populations.

    Bradley notes: current US H100 capacity supports only ~125,000 concurrent agents.
    This creates a bottleneck for multi-agent system deployment.
    """

    # Total available compute units
    total_capacity: float = 125_000.0

    # Current utilization
    allocated: float = 0.0

    def available(self) -> float:
        """Remaining compute capacity."""
        return max(0, self.total_capacity - self.allocated)

    def utilization(self) -> float:
        """Current utilization ratio (0-1)."""
        return self.allocated / self.total_capacity

    def can_allocate(self, cost: float) -> bool:
        """Check if allocation is possible."""
        return cost <= self.available()

    def allocate(self, cost: float) -> bool:
        """Attempt to allocate compute. Returns success."""
        if self.can_allocate(cost):
            self.allocated += cost
            return True
        return False

    def release(self, cost: float) -> None:
        """Release allocated compute."""
        self.allocated = max(0, self.allocated - cost)

    def max_concurrent_agents(
        self, profile: AgentCapabilityProfile, task_minutes: int
    ) -> int:
        """Calculate max concurrent agents given constraints."""
        cost_per_agent = profile.compute_cost(task_minutes)
        if cost_per_agent <= 0:
            return 0
        return int(self.total_capacity / cost_per_agent)
