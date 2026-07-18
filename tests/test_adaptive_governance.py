"""Tests for adaptive governance: contemplation + crystallization."""

import pytest

from swarm.governance.adaptive import (
    ContemplationEvaluator,
    CrystallizationGate,
    EpochEvidence,
    EvidenceAccumulator,
    ProposalStatus,
    ThresholdProposal,
)
from swarm.governance.adaptive_controller import (
    DEFAULT_ADAPTABLE_PARAMS,
    AdaptiveGovernanceController,
)
from swarm.governance.config import GovernanceConfig
from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType

# =========================================================================
# Helpers
# =========================================================================


def make_evidence(epoch: int, toxicity: float = 0.1, quality_gap: float = 0.0,
                  welfare: float = 10.0, avg_payoff: float = 1.0,
                  total: int = 50, accepted: int = 40) -> EpochEvidence:
    return EpochEvidence(
        epoch=epoch,
        toxicity_rate=toxicity,
        quality_gap=quality_gap,
        total_welfare=welfare,
        avg_payoff=avg_payoff,
        total_interactions=total,
        accepted_interactions=accepted,
    )


def make_rising_toxicity_evidence(n: int = 20) -> list[EpochEvidence]:
    """Generate evidence with steadily rising toxicity."""
    return [
        make_evidence(i, toxicity=0.3 + 0.03 * i, quality_gap=0.01 * i)
        for i in range(n)
    ]


def make_stable_evidence(n: int = 20) -> list[EpochEvidence]:
    """Generate stable evidence with low toxicity."""
    return [make_evidence(i, toxicity=0.1) for i in range(n)]


def make_config(**overrides) -> GovernanceConfig:
    defaults = {
        "circuit_breaker_enabled": True,
        "freeze_threshold_toxicity": 0.7,
        "audit_enabled": True,
        "audit_probability": 0.1,
        "audit_threshold_p": 0.5,
        "collusion_detection_enabled": True,
        "collusion_frequency_threshold": 2.0,
        "collusion_correlation_threshold": 0.7,
        "adaptive_controller_enabled": True,
        "adaptive_controller_evidence_window": 20,
        "adaptive_controller_contemplation_interval": 5,
        "adaptive_controller_min_evidence_epochs": 10,
        "adaptive_controller_confidence_threshold": 0.8,
        "adaptive_controller_crystallization_min_epochs": 5,
        "adaptive_controller_require_human_review": False,
        "adaptive_controller_max_degradation_tolerance": 0.05,
        "adaptive_controller_max_active_proposals": 3,
    }
    defaults.update(overrides)
    return GovernanceConfig(**defaults)


class FakeEpochMetrics:
    """Minimal stand-in for EpochMetrics."""
    def __init__(self, epoch=0, toxicity_rate=0.1, quality_gap=0.0,
                 total_welfare=10.0, avg_payoff=1.0,
                 total_interactions=50, accepted_interactions=40):
        self.epoch = epoch
        self.toxicity_rate = toxicity_rate
        self.quality_gap = quality_gap
        self.total_welfare = total_welfare
        self.avg_payoff = avg_payoff
        self.total_interactions = total_interactions
        self.accepted_interactions = accepted_interactions


class FakeGovernanceEngine:
    """Minimal stand-in for GovernanceEngine."""
    def __init__(self, config):
        self.config = config


# =========================================================================
# EvidenceAccumulator Tests
# =========================================================================


class TestEvidenceAccumulator:
    def test_window_size_respected(self):
        acc = EvidenceAccumulator(window_size=5)
        for i in range(10):
            acc.record_epoch(make_evidence(i))
        assert len(acc) == 5
        window = acc.get_window()
        assert window[0].epoch == 5
        assert window[-1].epoch == 9

    def test_metric_series_extraction(self):
        acc = EvidenceAccumulator(window_size=10)
        for i in range(5):
            acc.record_epoch(make_evidence(i, toxicity=0.1 * (i + 1)))
        series = acc.get_metric_series("toxicity_rate")
        assert len(series) == 5
        assert series[0] == pytest.approx(0.1)
        assert series[-1] == pytest.approx(0.5)

    def test_get_window_n(self):
        acc = EvidenceAccumulator(window_size=10)
        for i in range(8):
            acc.record_epoch(make_evidence(i))
        last3 = acc.get_window(3)
        assert len(last3) == 3
        assert last3[0].epoch == 5


# =========================================================================
# ContemplationEvaluator Tests
# =========================================================================


class TestContemplationEvaluator:
    def test_no_proposals_with_insufficient_data(self):
        config = make_config()
        evaluator = ContemplationEvaluator(
            adaptable_params=DEFAULT_ADAPTABLE_PARAMS,
            config=config,
            min_evidence_epochs=10,
        )
        acc = EvidenceAccumulator()
        for i in range(5):  # Only 5, need 10
            acc.record_epoch(make_evidence(i, toxicity=0.9))
        result = evaluator.evaluate(acc, current_epoch=5)
        assert len(result.proposals) == 0

    def test_proposals_on_rising_toxicity(self):
        config = make_config()
        evaluator = ContemplationEvaluator(
            adaptable_params=DEFAULT_ADAPTABLE_PARAMS,
            config=config,
            min_evidence_epochs=10,
        )
        acc = EvidenceAccumulator()
        for ev in make_rising_toxicity_evidence(20):
            acc.record_epoch(ev)
        result = evaluator.evaluate(acc, current_epoch=20)
        # Should generate at least one proposal for toxicity-related params
        assert len(result.proposals) > 0
        for p in result.proposals:
            assert p.status == ProposalStatus.PENDING

    def test_proposals_bounded(self):
        config = make_config()
        evaluator = ContemplationEvaluator(
            adaptable_params=DEFAULT_ADAPTABLE_PARAMS,
            config=config,
            min_evidence_epochs=10,
        )
        acc = EvidenceAccumulator()
        for ev in make_rising_toxicity_evidence(20):
            acc.record_epoch(ev)
        result = evaluator.evaluate(acc, current_epoch=20)
        for proposal in result.proposals:
            param = next(
                p for p in DEFAULT_ADAPTABLE_PARAMS
                if p.field_name == proposal.parameter
            )
            assert param.min_value <= proposal.proposed_value <= param.max_value
            assert abs(proposal.delta) <= param.max_delta_per_proposal + 1e-6

    def test_evaluation_is_deterministic(self):
        config = make_config()
        evidence = make_rising_toxicity_evidence(20)

        results = []
        for _ in range(2):
            evaluator = ContemplationEvaluator(
                adaptable_params=DEFAULT_ADAPTABLE_PARAMS,
                config=config,
                min_evidence_epochs=10,
            )
            acc = EvidenceAccumulator()
            for ev in evidence:
                acc.record_epoch(ev)
            results.append(evaluator.evaluate(acc, current_epoch=20))

        assert len(results[0].proposals) == len(results[1].proposals)
        for p1, p2 in zip(results[0].proposals, results[1].proposals, strict=True):
            assert p1.parameter == p2.parameter
            assert p1.proposed_value == pytest.approx(p2.proposed_value)

    def test_no_proposals_for_disabled_levers(self):
        config = make_config(
            circuit_breaker_enabled=False,
            audit_enabled=False,
            collusion_detection_enabled=False,
        )
        evaluator = ContemplationEvaluator(
            adaptable_params=DEFAULT_ADAPTABLE_PARAMS,
            config=config,
            min_evidence_epochs=10,
        )
        acc = EvidenceAccumulator()
        for ev in make_rising_toxicity_evidence(20):
            acc.record_epoch(ev)
        result = evaluator.evaluate(acc, current_epoch=20)
        assert len(result.proposals) == 0


# =========================================================================
# CrystallizationGate Tests
# =========================================================================


class TestCrystallizationGate:
    def _make_proposal(self, **kwargs) -> ThresholdProposal:
        defaults = {
            "parameter": "freeze_threshold_toxicity",
            "current_value": 0.7,
            "proposed_value": 0.6,
            "delta": -0.1,
            "confidence": 0.9,
            "status": ProposalStatus.ACTIVE,
            "created_epoch": 0,
            "epochs_active": 0,
            "metrics_before": {"toxicity_rate": 0.5, "quality_gap": 0.1, "total_welfare": 10.0},
            "metrics_during": {"toxicity_rate": 0.4, "quality_gap": 0.08, "total_welfare": 11.0},
        }
        defaults.update(kwargs)
        return ThresholdProposal(**defaults)

    def test_time_gate_blocks_early(self):
        gate = CrystallizationGate(min_sustained_epochs=5, require_human_review=False)
        acc = EvidenceAccumulator()
        proposal = self._make_proposal(epochs_active=3)
        passed, results = gate.check_gates(proposal, acc)
        assert not passed
        assert not results["time"]["passed"]

    def test_time_gate_passes_after_min(self):
        gate = CrystallizationGate(min_sustained_epochs=5, require_human_review=False)
        acc = EvidenceAccumulator()
        proposal = self._make_proposal(epochs_active=5)
        passed, results = gate.check_gates(proposal, acc)
        assert results["time"]["passed"]

    def test_alignment_gate_blocks_degradation(self):
        gate = CrystallizationGate(
            min_sustained_epochs=1,
            require_human_review=False,
            max_degradation_tolerance=0.05,
        )
        acc = EvidenceAccumulator()
        proposal = self._make_proposal(
            epochs_active=5,
            metrics_before={"toxicity_rate": 0.3, "quality_gap": 0.1, "total_welfare": 10.0},
            metrics_during={"toxicity_rate": 0.5, "quality_gap": 0.2, "total_welfare": 8.0},
        )
        passed, results = gate.check_gates(proposal, acc)
        assert not passed
        assert not results["alignment"]["passed"]

    def test_alignment_gate_passes_on_improvement(self):
        gate = CrystallizationGate(
            min_sustained_epochs=1,
            require_human_review=False,
        )
        acc = EvidenceAccumulator()
        proposal = self._make_proposal(
            epochs_active=5,
            metrics_before={"toxicity_rate": 0.5, "quality_gap": 0.1, "total_welfare": 10.0},
            metrics_during={"toxicity_rate": 0.3, "quality_gap": 0.08, "total_welfare": 12.0},
        )
        passed, results = gate.check_gates(proposal, acc)
        assert results["alignment"]["passed"]

    def test_human_review_blocks_unapproved(self):
        gate = CrystallizationGate(
            min_sustained_epochs=1,
            require_human_review=True,
        )
        acc = EvidenceAccumulator()
        proposal = self._make_proposal(epochs_active=5)
        passed, results = gate.check_gates(proposal, acc)
        assert not passed
        assert not results["human_review"]["passed"]

    def test_human_review_passes_after_approval(self):
        gate = CrystallizationGate(
            min_sustained_epochs=1,
            require_human_review=True,
        )
        acc = EvidenceAccumulator()
        proposal = self._make_proposal(epochs_active=5)
        gate.approve_proposal(proposal.proposal_id)
        passed, results = gate.check_gates(proposal, acc)
        assert results["human_review"]["passed"]

    def test_auto_approve_when_review_disabled(self):
        gate = CrystallizationGate(
            min_sustained_epochs=1,
            require_human_review=False,
        )
        acc = EvidenceAccumulator()
        proposal = self._make_proposal(epochs_active=5)
        passed, results = gate.check_gates(proposal, acc)
        assert results["human_review"]["passed"]


# =========================================================================
# AdaptiveGovernanceController Integration Tests
# =========================================================================


class TestAdaptiveGovernanceController:
    def _make_controller(self, **config_overrides):
        config = make_config(**config_overrides)
        engine = FakeGovernanceEngine(config)
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(events.append)
        controller = AdaptiveGovernanceController(
            governance_engine=engine,
            event_bus=bus,
            config=config,
        )
        return controller, config, events

    def test_full_lifecycle_crystallize(self):
        """Accumulate -> contemplate -> apply -> crystallize."""
        controller, config, events = self._make_controller(
            adaptive_controller_contemplation_interval=5,
            adaptive_controller_min_evidence_epochs=10,
            adaptive_controller_crystallization_min_epochs=3,
            adaptive_controller_require_human_review=False,
        )
        # Feed rising toxicity for 15 epochs
        for i in range(15):
            metrics = FakeEpochMetrics(
                epoch=i,
                toxicity_rate=0.3 + 0.03 * i,
                quality_gap=0.01 * i,
                total_welfare=10.0,
            )
            controller.on_epoch_end(metrics)

        # After 15 epochs with contemplation_interval=5, contemplation should
        # have run at least once (around epoch 10+ after min_evidence)
        history = controller.get_proposal_history()

        # If proposals were generated and conditions met, we should see
        # threshold changes or at least active proposals
        if history:
            # Feed improving metrics to crystallize
            for i in range(15, 25):
                metrics = FakeEpochMetrics(
                    epoch=i,
                    toxicity_rate=max(0.1, 0.5 - 0.02 * (i - 15)),
                    quality_gap=0.0,
                    total_welfare=12.0,
                )
                controller.on_epoch_end(metrics)

            # Check that at least one event was emitted
            event_types = [e.event_type for e in events]
            assert EventType.GOVERNANCE_CONTEMPLATION_COMPLETED in event_types

    def test_revert_on_degradation(self):
        """Proposals should revert when metrics degrade significantly."""
        controller, config, events = self._make_controller(
            adaptive_controller_contemplation_interval=5,
            adaptive_controller_min_evidence_epochs=10,
            adaptive_controller_crystallization_min_epochs=5,
            adaptive_controller_require_human_review=False,
            adaptive_controller_max_degradation_tolerance=0.05,
        )

        # Build up evidence with rising toxicity
        for i in range(15):
            metrics = FakeEpochMetrics(
                epoch=i,
                toxicity_rate=0.3 + 0.03 * i,
                quality_gap=0.01 * i,
            )
            controller.on_epoch_end(metrics)

        # If proposals were applied, now degrade metrics severely
        if controller._active_proposals:
            for i in range(15, 20):
                metrics = FakeEpochMetrics(
                    epoch=i,
                    toxicity_rate=0.9,  # Severe degradation
                    quality_gap=0.5,
                    total_welfare=2.0,
                )
                controller.on_epoch_end(metrics)

            # Verify reverted proposals exist if degradation was detected
            assert any(
                p["status"] == "reverted"
                for p in controller.get_proposal_history()
            ) or not controller.get_proposal_history()

    def test_max_active_proposals_respected(self):
        """Controller should not exceed max_active_proposals."""
        controller, config, events = self._make_controller(
            adaptive_controller_max_active_proposals=2,
            adaptive_controller_contemplation_interval=1,
            adaptive_controller_min_evidence_epochs=5,
        )

        for i in range(20):
            metrics = FakeEpochMetrics(
                epoch=i,
                toxicity_rate=0.3 + 0.03 * i,
                quality_gap=0.01 * i,
            )
            controller.on_epoch_end(metrics)

        assert len(controller._active_proposals) <= 2

    def test_events_emitted(self):
        """Events should be emitted at each lifecycle stage."""
        controller, config, events = self._make_controller(
            adaptive_controller_contemplation_interval=5,
            adaptive_controller_min_evidence_epochs=10,
        )

        for i in range(15):
            metrics = FakeEpochMetrics(
                epoch=i,
                toxicity_rate=0.3 + 0.03 * i,
                quality_gap=0.01 * i,
            )
            controller.on_epoch_end(metrics)

        # Should have contemplation events
        contemplation_events = [
            e for e in events
            if e.event_type == EventType.GOVERNANCE_CONTEMPLATION_COMPLETED
        ]
        assert len(contemplation_events) > 0

    def test_config_mutation_bounded(self):
        """Config changes must never violate parameter bounds."""
        controller, config, events = self._make_controller(
            adaptive_controller_contemplation_interval=1,
            adaptive_controller_min_evidence_epochs=5,
            adaptive_controller_crystallization_min_epochs=100,  # Never crystallize
        )

        for i in range(30):
            metrics = FakeEpochMetrics(
                epoch=i,
                toxicity_rate=0.3 + 0.02 * i,
                quality_gap=0.01 * i,
            )
            controller.on_epoch_end(metrics)

        # Check all adaptable params are within bounds
        for param in DEFAULT_ADAPTABLE_PARAMS:
            value = getattr(config, param.field_name)
            assert param.min_value <= value <= param.max_value, (
                f"{param.field_name}={value} outside [{param.min_value}, {param.max_value}]"
            )


# =========================================================================
# Config Validation Tests
# =========================================================================


class TestAdaptiveControllerConfig:
    def test_valid_config(self):
        config = make_config()
        assert config.adaptive_controller_enabled is True

    def test_invalid_evidence_window(self):
        with pytest.raises(ValueError, match="evidence_window"):
            make_config(adaptive_controller_evidence_window=0)

    def test_invalid_contemplation_interval(self):
        with pytest.raises(ValueError, match="contemplation_interval"):
            make_config(adaptive_controller_contemplation_interval=0)

    def test_invalid_confidence_threshold(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            make_config(adaptive_controller_confidence_threshold=1.5)

    def test_invalid_max_active_proposals(self):
        with pytest.raises(ValueError, match="max_active_proposals"):
            make_config(adaptive_controller_max_active_proposals=0)
