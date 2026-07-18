"""Adaptive governance: contemplation + crystallization for threshold tuning.

This module implements a three-phase adaptive governance loop:
1. Evidence accumulation (sliding window of epoch metrics)
2. Contemplation (3-pass evaluation: signal, trend, propose)
3. Crystallization (3-gate conjunction: time, alignment, human review)
"""

from __future__ import annotations

import math
import statistics
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProposalStatus(Enum):
    """Status of a threshold adjustment proposal."""

    PENDING = "pending"
    ACTIVE = "active"
    CRYSTALLIZED = "crystallized"
    REJECTED = "rejected"
    REVERTED = "reverted"


@dataclass
class AdaptableParameter:
    """Declares a config field as adaptable with bounds and constraints."""

    field_name: str
    min_value: float
    max_value: float
    max_delta_per_proposal: float
    requires_enabled: Optional[str] = None  # config field that must be True
    direction_metric: str = "toxicity_rate"  # metric to optimize
    tighten_on_high: bool = True  # if True, tighten (decrease) when metric is high


@dataclass
class EpochEvidence:
    """Per-epoch metrics snapshot for evidence accumulation."""

    epoch: int
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    total_welfare: float = 0.0
    avg_payoff: float = 0.0
    total_interactions: int = 0
    accepted_interactions: int = 0


@dataclass
class ThresholdProposal:
    """A concrete proposal to adjust a governance threshold."""

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parameter: str = ""
    current_value: float = 0.0
    proposed_value: float = 0.0
    delta: float = 0.0
    confidence: float = 0.0
    status: ProposalStatus = ProposalStatus.PENDING
    created_epoch: int = 0
    epochs_active: int = 0
    evidence_summary: str = ""
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_during: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContemplationResult:
    """Output of the 3-pass contemplation evaluation."""

    signals: Dict[str, str] = field(default_factory=dict)  # param -> direction
    trends: Dict[str, float] = field(default_factory=dict)  # param -> slope
    proposals: List[ThresholdProposal] = field(default_factory=list)
    epoch: int = 0


class EvidenceAccumulator:
    """Sliding-window accumulator for epoch evidence."""

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._evidence: deque[EpochEvidence] = deque(maxlen=window_size)

    def record_epoch(self, evidence: EpochEvidence) -> None:
        """Append an epoch snapshot."""
        self._evidence.append(evidence)

    def get_window(self, n: Optional[int] = None) -> List[EpochEvidence]:
        """Retrieve the last N epochs (or all if n is None)."""
        data = list(self._evidence)
        if n is not None:
            return data[-n:]
        return data

    def get_metric_series(self, metric_name: str) -> List[float]:
        """Extract a time series for a single metric."""
        return [getattr(e, metric_name, 0.0) for e in self._evidence]

    def __len__(self) -> int:
        return len(self._evidence)


class ContemplationEvaluator:
    """Three-pass evaluator for governance threshold adjustments.

    Pass 1 (Signal): detect which parameters need adjustment
    Pass 2 (Trend): validate with linear trend analysis
    Pass 3 (Propose): generate bounded proposals
    """

    def __init__(
        self,
        adaptable_params: List[AdaptableParameter],
        config: Any,
        min_evidence_epochs: int = 10,
        confidence_threshold: float = 0.8,
    ) -> None:
        self._params = {p.field_name: p for p in adaptable_params}
        self._config = config
        self._min_evidence_epochs = min_evidence_epochs
        self._confidence_threshold = confidence_threshold

    def evaluate(
        self, accumulator: EvidenceAccumulator, current_epoch: int
    ) -> ContemplationResult:
        """Run 3-pass contemplation evaluation."""
        result = ContemplationResult(epoch=current_epoch)

        if len(accumulator) < self._min_evidence_epochs:
            return result

        window = accumulator.get_window()

        # Pass 1: Signal detection
        signals = self._pass_signal(window)
        result.signals = signals

        # Pass 2: Trend validation
        trends = self._pass_trend(window, signals)
        result.trends = trends

        # Pass 3: Proposal generation
        proposals = self._pass_propose(window, signals, trends, current_epoch)
        result.proposals = proposals

        return result

    def _pass_signal(
        self, window: List[EpochEvidence]
    ) -> Dict[str, str]:
        """Pass 1: Detect direction signals for each adaptable parameter."""
        signals: Dict[str, str] = {}
        mid = len(window) // 2
        recent = window[mid:]
        older = window[:mid]

        if not recent or not older:
            return signals

        for name, param in self._params.items():
            # Check requires_enabled
            if param.requires_enabled and not getattr(
                self._config, param.requires_enabled, False
            ):
                continue

            metric = param.direction_metric
            recent_vals = [getattr(e, metric, 0.0) for e in recent]
            older_vals = [getattr(e, metric, 0.0) for e in older]

            recent_mean = statistics.mean(recent_vals) if recent_vals else 0.0
            older_mean = statistics.mean(older_vals) if older_vals else 0.0

            diff = recent_mean - older_mean

            if param.tighten_on_high:
                # High and rising metric -> tighten; low and falling -> loosen
                if recent_mean > 0.5 and diff > 0.01:
                    signals[name] = "tighten"
                elif recent_mean < 0.3 and diff < -0.01:
                    signals[name] = "loosen"
            else:
                # For metrics where higher is better (e.g., welfare)
                if recent_mean < older_mean * 0.9:
                    signals[name] = "tighten"
                elif recent_mean > older_mean * 1.1:
                    signals[name] = "loosen"

        return signals

    def _pass_trend(
        self,
        window: List[EpochEvidence],
        signals: Dict[str, str],
    ) -> Dict[str, float]:
        """Pass 2: Compute linear slope and check significance."""
        trends: Dict[str, float] = {}
        n = len(window)
        if n < 3:
            return trends

        for name in signals:
            param = self._params[name]
            metric = param.direction_metric
            values = [getattr(e, metric, 0.0) for e in window]

            slope, t_stat = self._linear_trend(values)

            # Use |t_stat| > 2.0 as rough significance threshold
            # (approx p < 0.05 for n > 10)
            if abs(t_stat) > 2.0:
                trends[name] = slope

        return trends

    def _pass_propose(
        self,
        window: List[EpochEvidence],
        signals: Dict[str, str],
        trends: Dict[str, float],
        current_epoch: int,
    ) -> List[ThresholdProposal]:
        """Pass 3: Generate bounded threshold proposals."""
        proposals: List[ThresholdProposal] = []

        # Snapshot current metrics for metrics_before
        recent = window[-3:] if len(window) >= 3 else window
        metrics_before = {
            "toxicity_rate": statistics.mean([e.toxicity_rate for e in recent]),
            "quality_gap": statistics.mean([e.quality_gap for e in recent]),
            "total_welfare": statistics.mean([e.total_welfare for e in recent]),
        }

        for name, slope in trends.items():
            param = self._params[name]
            signal = signals[name]
            current_value = getattr(self._config, name, 0.0)

            # Compute delta proportional to signal strength
            # Normalize slope to [0, 1] range roughly
            strength = min(abs(slope) * 10, 1.0)
            raw_delta = strength * param.max_delta_per_proposal

            if signal == "tighten":
                # For most params, tightening means decreasing the threshold
                delta = -raw_delta
            else:
                delta = raw_delta

            # Clamp delta
            delta = max(-param.max_delta_per_proposal, min(param.max_delta_per_proposal, delta))

            proposed = current_value + delta
            proposed = max(param.min_value, min(param.max_value, proposed))
            actual_delta = proposed - current_value

            if abs(actual_delta) < 1e-6:
                continue

            confidence = strength * (abs(slope) / max(abs(slope), 0.01))

            proposal = ThresholdProposal(
                parameter=name,
                current_value=current_value,
                proposed_value=proposed,
                delta=actual_delta,
                confidence=min(confidence, 1.0),
                status=ProposalStatus.PENDING,
                created_epoch=current_epoch,
                evidence_summary=f"signal={signal}, slope={slope:.4f}, strength={strength:.2f}",
                metrics_before=dict(metrics_before),
            )
            proposals.append(proposal)

        return proposals

    @staticmethod
    def _linear_trend(values: List[float]) -> tuple[float, float]:
        """Compute linear slope and t-statistic for a time series.

        Returns (slope, t_statistic). Uses simple OLS regression.
        """
        n = len(values)
        if n < 3:
            return 0.0, 0.0

        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(values)

        ss_xy = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        ss_xx = sum((i - x_mean) ** 2 for i in range(n))

        if ss_xx < 1e-12:
            return 0.0, 0.0

        slope = ss_xy / ss_xx

        # Residuals and standard error
        residuals = [v - (y_mean + slope * (i - x_mean)) for i, v in enumerate(values)]
        ss_res = sum(r * r for r in residuals)

        if n <= 2:
            return slope, 0.0

        mse = ss_res / (n - 2)
        se_slope = math.sqrt(mse / ss_xx) if mse > 0 and ss_xx > 0 else 1e-12

        if se_slope < 1e-12:
            # Perfect or near-perfect linear fit — trend is highly significant
            t_stat = float("inf") if abs(slope) > 1e-12 else 0.0
        else:
            t_stat = slope / se_slope
        return slope, t_stat


class CrystallizationGate:
    """Three-gate conjunction for permanent threshold adoption.

    Gate 1 (Time): Proposal must be active for min_sustained_epochs
    Gate 2 (Alignment): No metric degradation beyond tolerance
    Gate 3 (Human): Optional human review approval
    """

    def __init__(
        self,
        min_sustained_epochs: int = 5,
        alignment_metrics: Optional[List[str]] = None,
        require_human_review: bool = True,
        max_degradation_tolerance: float = 0.05,
    ) -> None:
        self._min_sustained_epochs = min_sustained_epochs
        self._alignment_metrics = alignment_metrics or [
            "toxicity_rate",
            "quality_gap",
            "total_welfare",
        ]
        self._require_human_review = require_human_review
        self._max_degradation_tolerance = max_degradation_tolerance
        self._human_approvals: Dict[str, bool] = {}  # proposal_id -> approved

    def check_gates(
        self, proposal: ThresholdProposal, accumulator: EvidenceAccumulator
    ) -> tuple[bool, Dict[str, Any]]:
        """Check all three crystallization gates.

        Returns (passed, gate_results) where gate_results details each gate.
        """
        gate_results: Dict[str, Any] = {}

        # Gate 1: Time
        time_passed = proposal.epochs_active >= self._min_sustained_epochs
        gate_results["time"] = {
            "passed": time_passed,
            "epochs_active": proposal.epochs_active,
            "required": self._min_sustained_epochs,
        }

        # Gate 2: Alignment
        alignment_passed, alignment_detail = self._check_alignment(proposal)
        gate_results["alignment"] = {
            "passed": alignment_passed,
            "detail": alignment_detail,
        }

        # Gate 3: Human review
        if self._require_human_review:
            human_passed = self._human_approvals.get(proposal.proposal_id, False)
        else:
            human_passed = True
        gate_results["human_review"] = {
            "passed": human_passed,
            "required": self._require_human_review,
        }

        all_passed = time_passed and alignment_passed and human_passed
        return all_passed, gate_results

    def _check_alignment(
        self, proposal: ThresholdProposal
    ) -> tuple[bool, Dict[str, Any]]:
        """Check that no alignment metric has degraded beyond tolerance."""
        detail: Dict[str, Any] = {}
        any_improved = False
        all_ok = True

        for metric in self._alignment_metrics:
            before = proposal.metrics_before.get(metric, 0.0)
            during = proposal.metrics_during.get(metric, 0.0)

            if metric == "total_welfare":
                # Higher is better for welfare
                degradation = (before - during) / max(abs(before), 1e-6)
                improved = during > before
            else:
                # Lower is better for toxicity_rate, quality_gap
                degradation = (during - before) / max(abs(before), 1e-6)
                improved = during < before

            if degradation > self._max_degradation_tolerance:
                all_ok = False

            if improved:
                any_improved = True

            detail[metric] = {
                "before": before,
                "during": during,
                "degradation": degradation,
                "improved": improved,
            }

        # At least one metric must improve
        passed = all_ok and any_improved
        return passed, detail

    def approve_proposal(self, proposal_id: str) -> None:
        """Approve a proposal for human review gate."""
        self._human_approvals[proposal_id] = True

    def reject_proposal(self, proposal_id: str) -> None:
        """Reject a proposal at human review gate."""
        self._human_approvals[proposal_id] = False
