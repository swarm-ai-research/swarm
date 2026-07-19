"""Adaptive governance controller: orchestrates contemplation + crystallization.

Wires the EvidenceAccumulator, ContemplationEvaluator, and CrystallizationGate
into an on_epoch_end callback that the Orchestrator invokes.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

from swarm.governance.adaptive import (
    AdaptableParameter,
    ContemplationEvaluator,
    CrystallizationGate,
    EpochEvidence,
    EvidenceAccumulator,
    ProposalStatus,
    ThresholdProposal,
)
from swarm.governance.config import GovernanceConfig
from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType

logger = logging.getLogger(__name__)

# Default adaptable parameters
DEFAULT_ADAPTABLE_PARAMS: List[AdaptableParameter] = [
    AdaptableParameter(
        field_name="freeze_threshold_toxicity",
        min_value=0.2,
        max_value=0.9,
        max_delta_per_proposal=0.1,
        requires_enabled="circuit_breaker_enabled",
        direction_metric="toxicity_rate",
        tighten_on_high=True,
    ),
    AdaptableParameter(
        field_name="audit_probability",
        min_value=0.01,
        max_value=0.5,
        max_delta_per_proposal=0.05,
        requires_enabled="audit_enabled",
        direction_metric="toxicity_rate",
        tighten_on_high=True,
    ),
    AdaptableParameter(
        field_name="audit_threshold_p",
        min_value=0.1,
        max_value=0.9,
        max_delta_per_proposal=0.1,
        requires_enabled="audit_enabled",
        direction_metric="toxicity_rate",
        tighten_on_high=True,
    ),
    AdaptableParameter(
        field_name="collusion_frequency_threshold",
        min_value=1.0,
        max_value=5.0,
        max_delta_per_proposal=0.5,
        requires_enabled="collusion_detection_enabled",
        direction_metric="toxicity_rate",
        tighten_on_high=True,
    ),
    AdaptableParameter(
        field_name="collusion_correlation_threshold",
        min_value=0.3,
        max_value=0.9,
        max_delta_per_proposal=0.1,
        requires_enabled="collusion_detection_enabled",
        direction_metric="toxicity_rate",
        tighten_on_high=True,
    ),
]


class AdaptiveGovernanceController:
    """Orchestrates the adaptive governance loop.

    Accumulates evidence, runs periodic contemplation, applies proposals
    as temporary threshold changes, and crystallizes or reverts them
    based on sustained outcomes.
    """

    def __init__(
        self,
        governance_engine: Any,
        event_bus: EventBus,
        config: GovernanceConfig,
        adaptable_params: Optional[List[AdaptableParameter]] = None,
    ) -> None:
        self._engine = governance_engine
        self._event_bus = event_bus
        self._config = config

        params = adaptable_params or DEFAULT_ADAPTABLE_PARAMS

        # Core components
        self._accumulator = EvidenceAccumulator(
            window_size=config.adaptive_controller_evidence_window,
        )
        self._evaluator = ContemplationEvaluator(
            adaptable_params=params,
            config=config,
            min_evidence_epochs=config.adaptive_controller_min_evidence_epochs,
            confidence_threshold=config.adaptive_controller_confidence_threshold,
        )
        self._crystallizer = CrystallizationGate(
            min_sustained_epochs=config.adaptive_controller_crystallization_min_epochs,
            require_human_review=config.adaptive_controller_require_human_review,
            max_degradation_tolerance=config.adaptive_controller_max_degradation_tolerance,
        )

        # State
        self._active_proposals: List[ThresholdProposal] = []
        self._proposal_history: List[ThresholdProposal] = []
        self._original_values: Dict[str, float] = {}  # parameter -> original value
        self._last_contemplation_epoch: int = -999
        self._contemplation_interval = config.adaptive_controller_contemplation_interval
        self._max_active_proposals = config.adaptive_controller_max_active_proposals

    def on_epoch_end(self, epoch_metrics: Any) -> None:
        """Main callback invoked at end of each epoch.

        Args:
            epoch_metrics: EpochMetrics from the orchestrator
        """
        # 1. Extract evidence from epoch metrics
        evidence = EpochEvidence(
            epoch=epoch_metrics.epoch,
            toxicity_rate=epoch_metrics.toxicity_rate,
            quality_gap=epoch_metrics.quality_gap,
            total_welfare=epoch_metrics.total_welfare,
            avg_payoff=epoch_metrics.avg_payoff,
            total_interactions=epoch_metrics.total_interactions,
            accepted_interactions=epoch_metrics.accepted_interactions,
        )

        # 2. Record to accumulator
        self._accumulator.record_epoch(evidence)

        # 3. Tick active proposals
        self._tick_active_proposals()

        # 4. Check crystallization gates for active proposals
        self._check_crystallization()

        # 5. Run contemplation if due
        current_epoch = epoch_metrics.epoch
        if (
            current_epoch - self._last_contemplation_epoch
            >= self._contemplation_interval
            and len(self._active_proposals) < self._max_active_proposals
        ):
            self._run_contemplation(current_epoch)
            self._last_contemplation_epoch = current_epoch

    def _tick_active_proposals(self) -> None:
        """Increment epochs_active and update metrics_during for each active proposal."""
        if not self._active_proposals:
            return

        window = self._accumulator.get_window(3)
        if not window:
            return

        current_metrics = {
            "toxicity_rate": statistics.mean([e.toxicity_rate for e in window]),
            "quality_gap": statistics.mean([e.quality_gap for e in window]),
            "total_welfare": statistics.mean([e.total_welfare for e in window]),
        }

        for proposal in self._active_proposals:
            proposal.epochs_active += 1
            proposal.metrics_during = dict(current_metrics)

    def _check_crystallization(self) -> None:
        """Check crystallization gates and crystallize or revert proposals."""
        to_remove: List[ThresholdProposal] = []

        for proposal in self._active_proposals:
            # Check for degradation that warrants immediate revert
            if self._should_revert(proposal):
                self._revert(proposal)
                to_remove.append(proposal)
                continue

            # Check crystallization gates
            passed, gate_results = self._crystallizer.check_gates(
                proposal, self._accumulator
            )

            if passed:
                self._crystallize(proposal)
                to_remove.append(proposal)
            elif (
                proposal.epochs_active >= self._crystallizer._min_sustained_epochs
                and not passed
            ):
                # Time gate passed but alignment gate failed -> revert.
                # (Human-review gate failure alone does not revert; the
                # proposal stays active pending review.)
                if gate_results.get("time", {}).get("passed", False):
                    alignment_passed = gate_results.get("alignment", {}).get(
                        "passed", True
                    )
                    if not alignment_passed:
                        self._revert(proposal)
                        to_remove.append(proposal)

        for p in to_remove:
            if p in self._active_proposals:
                self._active_proposals.remove(p)

    def _run_contemplation(self, current_epoch: int) -> None:
        """Run contemplation evaluation and apply resulting proposals."""
        result = self._evaluator.evaluate(self._accumulator, current_epoch)

        self._event_bus.emit(
            Event(
                event_type=EventType.GOVERNANCE_CONTEMPLATION_COMPLETED,
                payload={
                    "epoch": current_epoch,
                    "n_signals": len(result.signals),
                    "n_trends": len(result.trends),
                    "n_proposals": len(result.proposals),
                    "signals": result.signals,
                },
                epoch=current_epoch,
            )
        )

        # Apply proposals up to max_active
        slots = self._max_active_proposals - len(self._active_proposals)
        for proposal in result.proposals[:slots]:
            # Don't propose for params that already have active proposals
            if any(
                p.parameter == proposal.parameter for p in self._active_proposals
            ):
                continue
            self._apply_proposal(proposal)

    def _apply_proposal(self, proposal: ThresholdProposal) -> None:
        """Apply a proposal by mutating the governance config."""
        # Store original value if not already stored
        if proposal.parameter not in self._original_values:
            self._original_values[proposal.parameter] = getattr(
                self._config, proposal.parameter
            )

        # Mutate config
        setattr(self._config, proposal.parameter, proposal.proposed_value)
        proposal.status = ProposalStatus.ACTIVE
        self._active_proposals.append(proposal)
        self._proposal_history.append(proposal)

        logger.info(
            "Adaptive governance: applied proposal %s for %s: %.4f -> %.4f",
            proposal.proposal_id,
            proposal.parameter,
            proposal.current_value,
            proposal.proposed_value,
        )

        self._event_bus.emit(
            Event(
                event_type=EventType.GOVERNANCE_THRESHOLD_APPLIED,
                payload={
                    "proposal_id": proposal.proposal_id,
                    "parameter": proposal.parameter,
                    "old_value": proposal.current_value,
                    "new_value": proposal.proposed_value,
                    "confidence": proposal.confidence,
                    "evidence": proposal.evidence_summary,
                },
                epoch=proposal.created_epoch,
            )
        )

    def _crystallize(self, proposal: ThresholdProposal) -> None:
        """Permanently adopt a proposal."""
        proposal.status = ProposalStatus.CRYSTALLIZED
        # Remove from original values (change is permanent)
        self._original_values.pop(proposal.parameter, None)

        logger.info(
            "Adaptive governance: crystallized %s for %s at %.4f",
            proposal.proposal_id,
            proposal.parameter,
            proposal.proposed_value,
        )

        self._event_bus.emit(
            Event(
                event_type=EventType.GOVERNANCE_THRESHOLD_CRYSTALLIZED,
                payload={
                    "proposal_id": proposal.proposal_id,
                    "parameter": proposal.parameter,
                    "final_value": proposal.proposed_value,
                    "epochs_active": proposal.epochs_active,
                    "metrics_before": proposal.metrics_before,
                    "metrics_during": proposal.metrics_during,
                },
                epoch=proposal.created_epoch + proposal.epochs_active,
            )
        )

    def _revert(self, proposal: ThresholdProposal) -> None:
        """Revert a proposal to the original value."""
        original = self._original_values.pop(proposal.parameter, proposal.current_value)
        setattr(self._config, proposal.parameter, original)
        proposal.status = ProposalStatus.REVERTED

        logger.info(
            "Adaptive governance: reverted %s for %s: %.4f -> %.4f",
            proposal.proposal_id,
            proposal.parameter,
            proposal.proposed_value,
            original,
        )

        self._event_bus.emit(
            Event(
                event_type=EventType.GOVERNANCE_THRESHOLD_REVERTED,
                payload={
                    "proposal_id": proposal.proposal_id,
                    "parameter": proposal.parameter,
                    "reverted_to": original,
                    "was_proposed": proposal.proposed_value,
                    "epochs_active": proposal.epochs_active,
                    "metrics_during": proposal.metrics_during,
                },
                epoch=proposal.created_epoch + proposal.epochs_active,
            )
        )

    def _should_revert(self, proposal: ThresholdProposal) -> bool:
        """Check if a proposal should be immediately reverted due to degradation."""
        if proposal.epochs_active < 2:
            return False

        tol = self._crystallizer._max_degradation_tolerance
        for metric in ["toxicity_rate", "quality_gap"]:
            before = proposal.metrics_before.get(metric, 0.0)
            during = proposal.metrics_during.get(metric, 0.0)
            # For these metrics, lower is better
            if before > 0:
                degradation = (during - before) / max(abs(before), 1e-6)
                if degradation > tol * 3:  # 3x tolerance = emergency revert
                    return True

        # Check welfare (higher is better)
        welfare_before = proposal.metrics_before.get("total_welfare", 0.0)
        welfare_during = proposal.metrics_during.get("total_welfare", 0.0)
        if welfare_before > 0:
            welfare_drop = (welfare_before - welfare_during) / max(
                abs(welfare_before), 1e-6
            )
            if welfare_drop > tol * 3:
                return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get current controller state for inspection."""
        return {
            "enabled": True,
            "evidence_count": len(self._accumulator),
            "active_proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "parameter": p.parameter,
                    "current_value": p.current_value,
                    "original_value": p.proposed_value,
                    "epochs_active": p.epochs_active,
                    "status": p.status.value,
                }
                for p in self._active_proposals
            ],
            "total_proposals": len(self._proposal_history),
            "crystallized": sum(
                1
                for p in self._proposal_history
                if p.status == ProposalStatus.CRYSTALLIZED
            ),
            "reverted": sum(
                1
                for p in self._proposal_history
                if p.status == ProposalStatus.REVERTED
            ),
        }

    def get_proposal_history(self) -> List[Dict[str, Any]]:
        """Get serializable proposal history."""
        return [
            {
                "proposal_id": p.proposal_id,
                "parameter": p.parameter,
                "current_value": p.current_value,
                "proposed_value": p.proposed_value,
                "delta": p.delta,
                "confidence": p.confidence,
                "status": p.status.value,
                "created_epoch": p.created_epoch,
                "epochs_active": p.epochs_active,
                "evidence_summary": p.evidence_summary,
                "metrics_before": p.metrics_before,
                "metrics_during": p.metrics_during,
            }
            for p in self._proposal_history
        ]

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve a proposal for human review gate."""
        self._crystallizer.approve_proposal(proposal_id)
        return True

    def reject_proposal(self, proposal_id: str) -> bool:
        """Reject a proposal at human review gate."""
        self._crystallizer.reject_proposal(proposal_id)
        # Find and revert if active
        for p in self._active_proposals:
            if p.proposal_id == proposal_id:
                self._revert(p)
                self._active_proposals.remove(p)
                return True
        return False
