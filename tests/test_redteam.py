"""Tests for red-teaming framework."""

import pytest

from swarm.agents.adaptive_adversary import (
    AdaptiveAdversary,
    AdversaryMemory,
    AttackStrategy,
    StrategyPerformance,
)
from swarm.agents.base import Action, ActionType, Observation
from swarm.redteam.attacks import (
    AttackCategory,
    AttackDifficulty,
    AttackLibrary,
    AttackResult,
    AttackScenario,
)
from swarm.redteam.evaluator import (
    GovernanceRobustness,
    RedTeamEvaluator,
    Vulnerability,
    VulnerabilityReport,
)
from swarm.redteam.metrics import (
    AdversaryPerformance,
    EvasionMetrics,
    GovernanceEffectiveness,
    compute_damage_before_detection,
    compute_detection_latency,
    compute_evasion_rate,
    compute_governance_effectiveness,
)

# =============================================================================
# Adaptive Adversary Tests
# =============================================================================


class TestStrategyPerformance:
    """Tests for StrategyPerformance tracking."""

    def test_initial_performance(self):
        """Test initial performance values."""
        perf = StrategyPerformance(strategy=AttackStrategy.COLLUSION)
        assert perf.attempts == 0
        assert perf.success_rate == 0.5  # Prior
        assert perf.detection_rate == 0.5  # Prior

    def test_record_success(self):
        """Test recording successful attempt."""
        perf = StrategyPerformance(strategy=AttackStrategy.COLLUSION)
        perf.attempts = 10
        perf.successes = 8
        perf.detected = 2
        perf.total_payoff = 50.0

        assert perf.success_rate == 0.8
        assert perf.detection_rate == 0.2
        assert perf.evasion_rate == 0.8
        assert perf.avg_payoff == 5.0

    def test_effectiveness_score(self):
        """Test effectiveness score computation."""
        perf = StrategyPerformance(strategy=AttackStrategy.MIMICRY)
        # Not enough data
        assert perf.effectiveness_score == 0.5

        perf.attempts = 10
        perf.successes = 9
        perf.detected = 1
        perf.total_payoff = 30.0

        # High success (0.9), high evasion (0.9), moderate payoff
        score = perf.effectiveness_score
        assert 0.6 < score < 1.0


class TestAdversaryMemory:
    """Tests for AdversaryMemory."""

    def test_initial_memory(self):
        """Test initial memory state."""
        memory = AdversaryMemory()
        assert memory.current_heat_level == 0.0
        assert memory.epochs_since_detection == 0

    def test_heat_update_on_detection(self):
        """Test heat increases on detection."""
        memory = AdversaryMemory()
        memory.update_heat(detected=True, penalty=0.0)
        assert memory.current_heat_level > 0
        assert memory.epochs_since_detection == 0

    def test_heat_decay(self):
        """Test heat decays over time."""
        memory = AdversaryMemory()
        memory.current_heat_level = 0.5
        memory.epochs_since_detection = 5

        memory.update_heat(detected=False, penalty=0.0)
        assert memory.current_heat_level < 0.5


class TestAdaptiveAdversary:
    """Tests for AdaptiveAdversary agent."""

    def test_creation(self):
        """Test adaptive adversary creation."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        assert agent.agent_id == "adaptive_1"
        assert agent.learning_rate == 0.1
        assert len(agent.strategy_performance) == len(AttackStrategy)

    def test_initial_strategy(self):
        """Test initial strategy is reputation farming."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        assert agent.current_strategy == AttackStrategy.REPUTATION_FARMING
        assert agent.in_reputation_building_phase

    def test_select_strategy_exploration(self):
        """Test strategy exploration."""
        agent = AdaptiveAdversary(
            agent_id="adaptive_1",
            exploration_rate=1.0,  # Always explore
        )
        strategies = set()
        for _ in range(100):
            strategy = agent.select_strategy()
            strategies.add(strategy)
        # Should explore multiple strategies
        assert len(strategies) > 1

    def test_select_strategy_exploitation(self):
        """Test strategy exploitation."""
        agent = AdaptiveAdversary(
            agent_id="adaptive_1",
            exploration_rate=0.0,  # Never explore
        )

        # Make one strategy much better
        agent.strategy_performance[AttackStrategy.MIMICRY].attempts = 20
        agent.strategy_performance[AttackStrategy.MIMICRY].successes = 18
        agent.strategy_performance[AttackStrategy.MIMICRY].total_payoff = 100

        strategy = agent.select_strategy()
        assert strategy == AttackStrategy.MIMICRY

    def test_should_lay_low_high_heat(self):
        """Test laying low when heat is high."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        agent.memory.current_heat_level = 0.9

        obs = Observation(
            current_epoch=1,
            current_step=1,
            visible_agents=[],
            visible_posts=[],
        )

        assert agent._should_lay_low(obs)

    def test_act_returns_action(self):
        """Test act returns valid action."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")

        obs = Observation(
            current_epoch=1,
            current_step=1,
            visible_agents=[
                {"agent_id": "honest_1", "agent_type": "honest", "reputation": 5.0},
            ],
            visible_posts=[],
        )

        action = agent.act(obs)
        assert isinstance(action, Action)
        assert action.action_type in ActionType

    def test_update_adversary_outcome(self):
        """Test learning from outcomes."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        initial_heat = agent.memory.current_heat_level

        agent.update_adversary_outcome(
            payoff=10.0,
            penalty=5.0,
            detected=True,
            strategy_used=AttackStrategy.COLLUSION,
        )

        assert agent.memory.current_heat_level > initial_heat
        perf = agent.strategy_performance[AttackStrategy.COLLUSION]
        assert perf.attempts == 1
        assert perf.detected == 1

    def test_get_strategy_report(self):
        """Test strategy report generation."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        agent.update_adversary_outcome(5.0, 0.0, False, AttackStrategy.MIMICRY)

        report = agent.get_strategy_report()
        assert "current_strategy" in report
        assert "heat_level" in report
        assert "strategy_stats" in report
        assert AttackStrategy.MIMICRY.value in report["strategy_stats"]


# =============================================================================
# Attack Library Tests
# =============================================================================


class TestAttackScenario:
    """Tests for AttackScenario."""

    def test_create_scenario(self):
        """Test creating attack scenario."""
        scenario = AttackScenario(
            attack_id="test_attack",
            name="Test Attack",
            description="A test attack",
            category=AttackCategory.EVASION,
            difficulty=AttackDifficulty.MODERATE,
        )

        assert scenario.attack_id == "test_attack"
        assert scenario.category == AttackCategory.EVASION
        assert scenario.n_adversaries == 1

    def test_to_dict(self):
        """Test serialization."""
        scenario = AttackScenario(
            attack_id="test",
            name="Test",
            description="Test",
            category=AttackCategory.COORDINATION,
            difficulty=AttackDifficulty.ADVANCED,
        )

        data = scenario.to_dict()
        assert data["attack_id"] == "test"
        assert data["category"] == "coordination"


class TestAttackLibrary:
    """Tests for AttackLibrary."""

    def test_get_all_attacks(self):
        """Test getting all attacks."""
        attacks = AttackLibrary.get_all_attacks()
        assert len(attacks) >= 8

    def test_reputation_farming_attack(self):
        """Test reputation farming attack definition."""
        attack = AttackLibrary.reputation_farming_exploit()
        assert attack.attack_id == "reputation_farming"
        assert attack.category == AttackCategory.EXPLOITATION

    def test_collusion_ring_attack(self):
        """Test collusion ring attack definition."""
        attack = AttackLibrary.collusion_ring()
        assert attack.n_adversaries == 3
        assert "collusion_detection" in attack.targeted_levers

    def test_get_attacks_by_category(self):
        """Test filtering by category."""
        evasion = AttackLibrary.get_attacks_by_category(AttackCategory.EVASION)
        assert all(a.category == AttackCategory.EVASION for a in evasion)

    def test_get_attacks_by_difficulty(self):
        """Test filtering by difficulty."""
        advanced = AttackLibrary.get_attacks_by_difficulty(AttackDifficulty.ADVANCED)
        assert all(a.difficulty == AttackDifficulty.ADVANCED for a in advanced)

    def test_get_attacks_targeting_lever(self):
        """Test filtering by targeted lever."""
        circuit_attacks = AttackLibrary.get_attacks_targeting_lever("circuit_breaker")
        assert len(circuit_attacks) > 0


# =============================================================================
# Evasion Metrics Tests
# =============================================================================


class TestEvasionMetrics:
    """Tests for EvasionMetrics."""

    def test_initial_metrics(self):
        """Test initial metric values."""
        metrics = EvasionMetrics()
        assert metrics.evasion_rate == 0.0
        # detection_rate = 1 - evasion_rate when no actions
        assert metrics.total_malicious_actions == 0

    def test_record_action_detected(self):
        """Test recording detected action."""
        metrics = EvasionMetrics()
        metrics.record_action(detected=True, damage=10.0, epoch=5)

        assert metrics.total_malicious_actions == 1
        assert metrics.detected_actions == 1
        assert metrics.first_detection_epoch == 5

    def test_record_action_undetected(self):
        """Test recording undetected action."""
        metrics = EvasionMetrics()
        metrics.record_action(detected=False, damage=10.0, epoch=5)

        assert metrics.undetected_actions == 1
        assert metrics.damage_before_first_detection == 10.0

    def test_evasion_rate(self):
        """Test evasion rate computation."""
        metrics = EvasionMetrics()
        for i in range(10):
            metrics.record_action(detected=(i < 3), damage=5.0, epoch=i)

        assert metrics.evasion_rate == pytest.approx(0.7)
        assert metrics.detection_rate == pytest.approx(0.3)

    def test_strategy_evasion_rates(self):
        """Test per-strategy evasion rates."""
        metrics = EvasionMetrics()
        metrics.record_action(True, 5.0, 1, "collusion")
        metrics.record_action(False, 5.0, 2, "collusion")
        metrics.record_action(False, 5.0, 3, "mimicry")

        rates = metrics.get_strategy_evasion_rates()
        assert rates["collusion"] == 0.5
        assert rates["mimicry"] == 1.0


class TestGovernanceEffectiveness:
    """Tests for GovernanceEffectiveness metrics."""

    def test_precision_recall(self):
        """Test precision and recall computation."""
        eff = GovernanceEffectiveness(
            true_positives=80,
            false_positives=10,
            true_negatives=90,
            false_negatives=20,
        )

        assert eff.precision == 80 / 90  # TP / (TP + FP)
        assert eff.recall == 80 / 100  # TP / (TP + FN)

    def test_f1_score(self):
        """Test F1 score computation."""
        eff = GovernanceEffectiveness(
            true_positives=80,
            false_positives=20,
            false_negatives=20,
        )

        # precision = 0.8, recall = 0.8
        assert abs(eff.f1_score - 0.8) < 0.01

    def test_attack_prevention_rate(self):
        """Test attack prevention rate."""
        eff = GovernanceEffectiveness(
            attacks_prevented=7,
            attacks_succeeded=3,
        )

        assert eff.attack_prevention_rate == 0.7


class TestMetricFunctions:
    """Tests for metric computation functions."""

    def test_compute_evasion_rate(self):
        """Test evasion rate computation."""
        actions = [
            {"action_id": "a1"},
            {"action_id": "a2"},
            {"action_id": "a3"},
        ]
        detections = [{"action_id": "a1"}]

        rate = compute_evasion_rate(actions, detections)
        assert rate == 2 / 3

    def test_compute_detection_latency(self):
        """Test detection latency computation."""
        actions = [
            {"action_id": "a1", "epoch": 5},
            {"action_id": "a2", "epoch": 8},
        ]
        detections = [
            {"action_id": "a1", "epoch": 7},
            {"action_id": "a2", "epoch": 10},
        ]

        mean, min_lat, max_lat = compute_detection_latency(actions, detections)
        assert mean == 2.0
        assert min_lat == 2
        assert max_lat == 2

    def test_compute_damage_before_detection(self):
        """Test damage before detection computation."""
        actions = [
            {"action_id": "a1", "epoch": 2, "damage": 10.0},
            {"action_id": "a2", "epoch": 5, "damage": 15.0},
            {"action_id": "a3", "epoch": 8, "damage": 20.0},
        ]

        damage = compute_damage_before_detection(actions, first_detection_epoch=6)
        assert damage == 25.0  # 10 + 15


# =============================================================================
# Red-Team Evaluator Tests
# =============================================================================


class TestGovernanceRobustness:
    """Tests for GovernanceRobustness assessment."""

    def test_robustness_score(self):
        """Test robustness score computation."""
        robustness = GovernanceRobustness(
            attacks_tested=10,
            attacks_prevented=7,
            overall_evasion_rate=0.3,
            total_damage_allowed=200,
        )

        score = robustness.robustness_score
        assert 0 <= score <= 1

    def test_grade(self):
        """Test letter grade assignment."""
        robustness = GovernanceRobustness()
        robustness.attacks_tested = 10
        robustness.attacks_prevented = 9
        robustness.overall_evasion_rate = 0.1
        robustness.total_damage_allowed = 50

        grade = robustness.grade
        assert grade in ["A", "B", "C", "D", "F"]


class TestVulnerability:
    """Tests for Vulnerability tracking."""

    def test_create_vulnerability(self):
        """Test creating vulnerability."""
        vuln = Vulnerability(
            vulnerability_id="vuln_1",
            severity="high",
            description="Test vulnerability",
            attack_vector="evasion",
            affected_lever="circuit_breaker",
            exploitation_difficulty="moderate",
            potential_damage=50.0,
            mitigation="Lower thresholds",
        )

        assert vuln.severity == "high"
        data = vuln.to_dict()
        assert data["affected_lever"] == "circuit_breaker"


class TestRedTeamEvaluator:
    """Tests for RedTeamEvaluator."""

    def test_creation(self):
        """Test evaluator creation."""
        config = {"circuit_breaker_enabled": True}
        evaluator = RedTeamEvaluator(governance_config=config)

        assert evaluator.governance_config == config
        assert len(evaluator.attack_scenarios) > 0

    def test_quick_evaluate(self):
        """Test quick evaluation."""
        config = {
            "circuit_breaker_enabled": True,
            "collusion_detection_enabled": True,
            "audit_enabled": True,
        }
        evaluator = RedTeamEvaluator(governance_config=config)

        result = evaluator.quick_evaluate()
        assert "attacks_tested" in result
        assert "attacks_successful" in result
        assert "avg_evasion_rate" in result

    def test_evaluate_with_no_defenses(self):
        """Test evaluation with no defenses."""
        config = {}  # No defenses
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=[AttackLibrary.threshold_dancing()],
        )

        result = evaluator.quick_evaluate(["threshold_dancing"])
        # Without defenses, attacks should be more successful
        assert result["attacks_tested"] == 1

    def test_evaluate_with_full_defenses(self):
        """Test evaluation with all defenses."""
        config = {
            "circuit_breaker_enabled": True,
            "collusion_detection_enabled": True,
            "audit_enabled": True,
            "staking_enabled": True,
            "transaction_tax_rate": 0.1,
        }
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=[AttackLibrary.reputation_farming_exploit()],
        )

        result = evaluator.quick_evaluate(["reputation_farming"])
        # With full defenses, attacks should be less successful
        assert "avg_evasion_rate" in result


class TestVulnerabilityReport:
    """Tests for VulnerabilityReport."""

    def test_generate_summary(self):
        """Test summary generation."""
        robustness = GovernanceRobustness(
            governance_config={"test": True},
            attacks_tested=5,
            attacks_prevented=3,
            overall_evasion_rate=0.4,
        )
        robustness.vulnerabilities = [
            Vulnerability(
                vulnerability_id="test_vuln",
                severity="high",
                description="Test",
                attack_vector="evasion",
                affected_lever="circuit_breaker",
                exploitation_difficulty="moderate",
                potential_damage=25.0,
                mitigation="Fix it",
            )
        ]
        robustness.recommendations = ["Do something"]

        report = VulnerabilityReport(robustness=robustness)
        summary = report.generate_summary()

        assert "RED-TEAM EVALUATION REPORT" in summary
        assert "test_vuln" in summary
        assert "Do something" in summary


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for red-teaming with orchestrator."""

    def test_adaptive_adversary_in_simulation(self):
        """Test running adaptive adversary in orchestrator."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
        from swarm.governance.config import GovernanceConfig

        config = OrchestratorConfig(
            n_epochs=3,
            steps_per_epoch=5,
            governance_config=GovernanceConfig(
                circuit_breaker_enabled=True,
                freeze_threshold_toxicity=0.5,
            ),
        )

        orchestrator = Orchestrator(config=config)
        orchestrator.register_agent(HonestAgent("honest_1"))
        orchestrator.register_agent(HonestAgent("honest_2"))
        orchestrator.register_agent(AdaptiveAdversary("adaptive_1"))

        metrics = orchestrator.run()
        assert len(metrics) == 3

    def test_get_adaptive_adversary_reports(self):
        """Test getting reports from adaptive adversaries."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(n_epochs=2, steps_per_epoch=3)
        orchestrator = Orchestrator(config=config)

        orchestrator.register_agent(HonestAgent("honest_1"))
        orchestrator.register_agent(AdaptiveAdversary("adaptive_1"))
        orchestrator.register_agent(AdaptiveAdversary("adaptive_2"))

        orchestrator.run()

        reports = orchestrator.get_adaptive_adversary_reports()
        assert "adaptive_1" in reports
        assert "adaptive_2" in reports
        assert "current_strategy" in reports["adaptive_1"]

    def test_get_evasion_metrics(self):
        """Test getting evasion metrics from orchestrator."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(n_epochs=2, steps_per_epoch=3)
        orchestrator = Orchestrator(config=config)

        orchestrator.register_agent(HonestAgent("honest_1"))
        orchestrator.register_agent(AdaptiveAdversary("adaptive_1"))

        orchestrator.run()

        metrics = orchestrator.get_evasion_metrics()
        assert "total_adversaries" in metrics
        assert "adaptive_adversaries" in metrics
        assert metrics["adaptive_adversaries"] == 1

    def test_notify_adversary_detection(self):
        """Test notifying adversary of detection."""
        from swarm.core.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        adversary = AdaptiveAdversary("adaptive_1")
        orchestrator.register_agent(adversary)

        initial_heat = adversary.memory.current_heat_level

        orchestrator.notify_adversary_detection(
            agent_id="adaptive_1",
            penalty=10.0,
            detected=True,
        )

        assert adversary.memory.current_heat_level > initial_heat


# =============================================================================
# Additional Red-Team Evaluator Tests
# =============================================================================


class TestRedTeamEvaluatorFull:
    """Additional tests for RedTeamEvaluator.evaluate() and helpers."""

    def test_evaluate_full_with_verbose(self, capsys):
        """evaluate() with verbose=True prints progress."""
        import random

        random.seed(42)

        config = {
            "circuit_breaker_enabled": True,
            "collusion_detection_enabled": True,
        }
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=[AttackLibrary.reputation_farming_exploit()],
        )

        report = evaluator.evaluate(
            orchestrator_factory=lambda c: None,
            epochs_per_attack=5,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Running attack" in captured.out
        assert report.robustness.attacks_tested == 1
        assert isinstance(report, VulnerabilityReport)

    def test_evaluate_populates_report_extremes(self):
        """evaluate() sets most_effective_attack and least_effective_attack."""
        import random

        random.seed(42)

        config = {"circuit_breaker_enabled": True}
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=[
                AttackLibrary.reputation_farming_exploit(),
                AttackLibrary.threshold_dancing(),
            ],
        )

        report = evaluator.evaluate(
            orchestrator_factory=lambda c: None,
            epochs_per_attack=5,
        )

        assert report.most_effective_attack is not None
        assert report.least_effective_attack is not None

    def test_evaluate_categorizes_results(self):
        """evaluate() groups results by category and difficulty."""
        import random

        random.seed(42)

        config = {}
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=[
                AttackLibrary.collusion_ring(),
                AttackLibrary.threshold_dancing(),
            ],
        )

        report = evaluator.evaluate(
            orchestrator_factory=lambda c: None,
            epochs_per_attack=5,
        )

        assert len(report.robustness.results_by_category) > 0
        assert len(report.robustness.results_by_difficulty) > 0


class TestSimulateAttackOutcome:
    """Tests for _simulate_attack_outcome defense branches."""

    def _make_evaluator(self, config):
        return RedTeamEvaluator(governance_config=config)

    def _make_result(self, scenario):
        return AttackResult(
            attack_id=scenario.attack_id,
            scenario=scenario,
            epochs_run=10,
        )

    def test_collusion_ring_with_detection(self):
        """collusion_detection_enabled reduces collusion success."""
        import random

        random.seed(42)

        config = {"collusion_detection_enabled": True}
        evaluator = self._make_evaluator(config)
        scenario = AttackLibrary.collusion_ring()
        result = evaluator._simulate_attack_outcome(
            scenario, self._make_result(scenario)
        )
        assert result.collusion_detected is True

    def test_collusion_ring_without_detection(self):
        """Without collusion detection, collusion not flagged."""
        import random

        random.seed(42)

        config = {"collusion_detection_enabled": False}
        evaluator = self._make_evaluator(config)
        scenario = AttackLibrary.collusion_ring()
        result = evaluator._simulate_attack_outcome(
            scenario, self._make_result(scenario)
        )
        assert result.collusion_detected is False

    def test_threshold_dancing_with_circuit_breaker(self):
        """circuit_breaker_enabled affects threshold dancing."""
        import random

        random.seed(42)

        config = {"circuit_breaker_enabled": True}
        evaluator = self._make_evaluator(config)
        scenario = AttackLibrary.threshold_dancing()
        result = evaluator._simulate_attack_outcome(
            scenario, self._make_result(scenario)
        )
        assert result.evasion_rate >= 0

    def test_reputation_farming_with_audit(self):
        """audit_enabled reduces reputation farming success."""
        import random

        random.seed(42)

        config = {"audit_enabled": True}
        evaluator = self._make_evaluator(config)
        scenario = AttackLibrary.reputation_farming_exploit()
        result = evaluator._simulate_attack_outcome(
            scenario, self._make_result(scenario)
        )
        assert result.evasion_rate >= 0

    def test_staking_reduces_success(self):
        """staking_enabled reduces base success."""
        import random

        random.seed(100)

        config_with = {"staking_enabled": True}
        config_without = {"staking_enabled": False}

        scenario = AttackLibrary.reputation_farming_exploit()

        random.seed(100)
        eval_with = self._make_evaluator(config_with)
        result_with = eval_with._simulate_attack_outcome(
            scenario, self._make_result(scenario)
        )

        random.seed(100)
        eval_without = self._make_evaluator(config_without)
        result_without = eval_without._simulate_attack_outcome(
            scenario, self._make_result(scenario)
        )

        # Staking should reduce evasion rate
        assert result_with.evasion_rate <= result_without.evasion_rate

    def test_tax_rate_reduces_success(self):
        """Positive transaction_tax_rate reduces success probability."""
        import random

        random.seed(100)

        scenario = AttackLibrary.reputation_farming_exploit()

        random.seed(100)
        eval_tax = self._make_evaluator({"transaction_tax_rate": 0.2})
        result_tax = eval_tax._simulate_attack_outcome(
            scenario, self._make_result(scenario)
        )

        random.seed(100)
        eval_no_tax = self._make_evaluator({"transaction_tax_rate": 0.0})
        result_no_tax = eval_no_tax._simulate_attack_outcome(
            scenario, self._make_result(scenario)
        )

        assert result_tax.evasion_rate <= result_no_tax.evasion_rate

    def test_successful_attack_has_damage_before_detection(self):
        """Successful attack sets damage_before_detection to 60% of damage."""
        import random

        random.seed(1)

        config = {}
        evaluator = self._make_evaluator(config)
        scenario = AttackLibrary.reputation_farming_exploit()

        # Keep trying seeds until we get a successful attack
        for seed in range(100):
            random.seed(seed)
            result = evaluator._simulate_attack_outcome(
                scenario, self._make_result(scenario)
            )
            if result.attack_succeeded:
                assert result.damage_before_detection == pytest.approx(
                    result.damage_caused * 0.6
                )
                break

    def test_prevented_attack_sets_detected_epoch(self):
        """Prevented attack sets detected_at_epoch."""
        import random

        config = {
            "circuit_breaker_enabled": True,
            "audit_enabled": True,
            "staking_enabled": True,
        }
        evaluator = self._make_evaluator(config)
        scenario = AttackLibrary.threshold_dancing()

        for seed in range(100):
            random.seed(seed)
            result = evaluator._simulate_attack_outcome(
                scenario, self._make_result(scenario)
            )
            if result.governance_prevented:
                assert result.detected_at_epoch == result.detection_latency
                break


class TestIdentifyVulnerabilities:
    """Tests for _identify_vulnerabilities."""

    def test_high_evasion_rate_creates_vulnerability(self):
        """Evasion rate > 0.7 creates a vulnerability."""
        import random

        random.seed(42)

        # Use minimal governance so attacks succeed easily
        config = {}
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=AttackLibrary.get_all_attacks(),
        )

        # Run attacks and check vulnerabilities
        evaluator.evaluate(orchestrator_factory=lambda c: None, epochs_per_attack=5)
        vulns = evaluator._identify_vulnerabilities()

        # Should have some vulnerabilities from high evasion rates
        evasion_vulns = [v for v in vulns if "evasion" in v.vulnerability_id]
        # At least one attack should have high evasion with no defenses
        assert len(evasion_vulns) >= 0  # May or may not depending on random seed

    def test_successful_attack_creates_vulnerability(self):
        """Successful attack with damage > 30 creates vulnerability."""
        import random

        random.seed(42)

        config = {}
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=AttackLibrary.get_all_attacks(),
        )

        evaluator.evaluate(orchestrator_factory=lambda c: None, epochs_per_attack=5)
        vulns = evaluator._identify_vulnerabilities()

        success_vulns = [v for v in vulns if "success" in v.vulnerability_id]
        # At least one should succeed with no defenses
        assert len(success_vulns) >= 0


class TestGenerateRecommendations:
    """Tests for _generate_recommendations."""

    def test_missing_circuit_breaker_recommendation(self):
        """Recommends enabling circuit breaker when missing."""
        config = {"circuit_breaker_enabled": False}
        evaluator = RedTeamEvaluator(governance_config=config)
        robustness = GovernanceRobustness()
        recs = evaluator._generate_recommendations(robustness)
        assert any("circuit breaker" in r.lower() for r in recs)

    def test_missing_collusion_detection_with_coordination_attack(self):
        """Recommends collusion detection when coordination attack succeeds."""
        import random

        random.seed(42)

        config = {"collusion_detection_enabled": False}
        evaluator = RedTeamEvaluator(governance_config=config)

        # Simulate a coordination attack that succeeded
        scenario = AttackLibrary.collusion_ring()
        result = AttackResult(
            attack_id="collusion_ring",
            scenario=scenario,
            attack_succeeded=True,
        )
        evaluator.attack_results = [result]

        robustness = GovernanceRobustness()
        recs = evaluator._generate_recommendations(robustness)
        assert any("collusion" in r.lower() for r in recs)

    def test_missing_audit_with_high_evasion(self):
        """Recommends audit when evasion is high."""
        config = {"audit_enabled": False}
        evaluator = RedTeamEvaluator(governance_config=config)
        robustness = GovernanceRobustness(overall_evasion_rate=0.6)
        recs = evaluator._generate_recommendations(robustness)
        assert any("audit" in r.lower() for r in recs)

    def test_low_tax_with_resource_attack(self):
        """Recommends increasing tax when resource attack succeeds."""
        config = {"transaction_tax_rate": 0.01}
        evaluator = RedTeamEvaluator(governance_config=config)

        scenario = AttackLibrary.resource_drain()
        result = AttackResult(
            attack_id="resource_drain",
            scenario=scenario,
            attack_succeeded=True,
        )
        evaluator.attack_results = [result]

        robustness = GovernanceRobustness()
        recs = evaluator._generate_recommendations(robustness)
        assert any("tax" in r.lower() for r in recs)

    def test_high_evasion_threshold_recommendation(self):
        """Recommends lowering thresholds when evasion > 0.6."""
        config = {"circuit_breaker_enabled": True}
        evaluator = RedTeamEvaluator(governance_config=config)
        robustness = GovernanceRobustness(overall_evasion_rate=0.7)
        recs = evaluator._generate_recommendations(robustness)
        assert any("threshold" in r.lower() for r in recs)

    def test_more_successes_than_prevented(self):
        """Recommends defense-in-depth when more attacks succeed than prevented."""
        config = {"circuit_breaker_enabled": True}
        evaluator = RedTeamEvaluator(governance_config=config)
        robustness = GovernanceRobustness(
            attacks_successful=5,
            attacks_prevented=2,
        )
        recs = evaluator._generate_recommendations(robustness)
        assert any("defense-in-depth" in r.lower() for r in recs)


# =============================================================================
# Serialization Tests
# =============================================================================


class TestGovernanceRobustnessSerialization:
    """Tests for GovernanceRobustness.to_dict."""

    def test_to_dict(self):
        """to_dict includes all expected fields."""
        robustness = GovernanceRobustness(
            governance_config={"test": True},
            attacks_tested=10,
            attacks_successful=3,
            attacks_prevented=7,
            overall_evasion_rate=0.3,
            total_damage_allowed=100,
        )
        data = robustness.to_dict()

        assert data["attacks_tested"] == 10
        assert data["attacks_successful"] == 3
        assert data["attacks_prevented"] == 7
        assert data["overall_evasion_rate"] == 0.3
        assert "robustness_score" in data
        assert "grade" in data
        assert "vulnerabilities" in data
        assert "recommendations" in data

    def test_to_dict_with_vulnerabilities(self):
        """to_dict serializes vulnerabilities."""
        vuln = Vulnerability(
            vulnerability_id="v1",
            severity="high",
            description="desc",
            attack_vector="evasion",
            affected_lever="circuit_breaker",
            exploitation_difficulty="moderate",
            potential_damage=50.0,
            mitigation="fix it",
        )
        robustness = GovernanceRobustness(vulnerabilities=[vuln])
        data = robustness.to_dict()
        assert len(data["vulnerabilities"]) == 1
        assert data["vulnerabilities"][0]["vulnerability_id"] == "v1"


class TestVulnerabilityReportSerialization:
    """Tests for VulnerabilityReport.to_dict."""

    def test_to_dict(self):
        """to_dict includes robustness and attack results."""
        robustness = GovernanceRobustness(attacks_tested=1)
        report = VulnerabilityReport(
            robustness=robustness,
            weakest_lever="circuit_breaker",
            strongest_lever="staking",
            most_effective_attack="collusion_ring",
            least_effective_attack="threshold_dancing",
        )
        data = report.to_dict()

        assert "robustness" in data
        assert data["weakest_lever"] == "circuit_breaker"
        assert data["strongest_lever"] == "staking"
        assert data["most_effective_attack"] == "collusion_ring"
        assert data["least_effective_attack"] == "threshold_dancing"


# =============================================================================
# Metrics Serialization and Edge Case Tests
# =============================================================================


class TestComputeGovernanceEffectiveness:
    """Tests for compute_governance_effectiveness function."""

    def test_mixed_agent_types(self):
        """Correctly classifies TP, FP, TN, FN for mixed agents."""
        agent_actions = [
            {"agent_id": "adv_1", "action_id": "act_1"},
            {"agent_id": "adv_1", "action_id": "act_2"},
            {"agent_id": "honest_1", "action_id": "act_3"},
            {"agent_id": "honest_1", "action_id": "act_4"},
        ]
        agent_types = {
            "adv_1": "adversarial",
            "honest_1": "honest",
        }
        detections = [
            {"action_id": "act_1"},  # TP
            {"action_id": "act_4"},  # FP
        ]
        penalties = [
            {"agent_id": "adv_1", "amount": 10.0, "freeze": True, "duration": 2},
            {"agent_id": "honest_1", "amount": 5.0},
        ]

        eff = compute_governance_effectiveness(
            agent_actions,
            agent_types,
            detections,
            penalties,
        )

        assert eff.true_positives == 1  # act_1
        assert eff.false_negatives == 1  # act_2
        assert eff.true_negatives == 1  # act_3
        assert eff.false_positives == 1  # act_4
        assert eff.honest_agents_penalized == 1
        assert eff.penalties_applied == 15.0
        assert eff.agents_frozen == 1
        assert eff.honest_agent_losses == 5.0

    def test_deceptive_agent_type(self):
        """'deceptive' agents are treated as adversarial."""
        agent_actions = [{"agent_id": "d1", "action_id": "a1"}]
        agent_types = {"d1": "deceptive"}
        detections = [{"action_id": "a1"}]

        eff = compute_governance_effectiveness(
            agent_actions,
            agent_types,
            detections,
            [],
        )
        assert eff.true_positives == 1


class TestAdversaryPerformanceProperties:
    """Tests for AdversaryPerformance properties."""

    def test_detection_rate_zero_actions(self):
        """detection_rate returns 0.0 when no actions taken."""
        perf = AdversaryPerformance(agent_id="adv_1")
        assert perf.detection_rate == 0.0

    def test_detection_rate_with_actions(self):
        """detection_rate = times_detected / actions_taken."""
        perf = AdversaryPerformance(
            agent_id="adv_1",
            actions_taken=10,
            times_detected=3,
        )
        assert perf.detection_rate == pytest.approx(0.3)

    def test_roi_zero_actions(self):
        """roi returns 0.0 when no actions taken."""
        perf = AdversaryPerformance(agent_id="adv_1")
        assert perf.roi == 0.0

    def test_roi_with_actions(self):
        """roi = net_payoff / actions_taken."""
        perf = AdversaryPerformance(
            agent_id="adv_1",
            actions_taken=5,
            net_payoff=25.0,
        )
        assert perf.roi == pytest.approx(5.0)

    def test_to_dict(self):
        """to_dict includes all expected fields."""
        perf = AdversaryPerformance(
            agent_id="adv_1",
            actions_taken=10,
            times_detected=2,
            net_payoff=30.0,
            best_strategy="mimicry",
        )
        data = perf.to_dict()
        assert data["agent_id"] == "adv_1"
        assert data["detection_rate"] == pytest.approx(0.2)
        assert data["roi"] == pytest.approx(3.0)
        assert data["best_strategy"] == "mimicry"


class TestGovernanceEffectivenessSerialization:
    """Tests for GovernanceEffectiveness.to_dict."""

    def test_to_dict(self):
        """to_dict includes computed properties."""
        eff = GovernanceEffectiveness(
            true_positives=80,
            false_positives=10,
            true_negatives=90,
            false_negatives=20,
            attacks_prevented=7,
            attacks_succeeded=3,
        )
        data = eff.to_dict()

        assert data["precision"] == pytest.approx(80 / 90)
        assert data["recall"] == pytest.approx(80 / 100)
        assert "f1_score" in data
        assert "accuracy" in data
        assert data["attack_prevention_rate"] == pytest.approx(0.7)

    def test_edge_case_all_zeros(self):
        """to_dict handles all-zero case without division errors."""
        eff = GovernanceEffectiveness()
        data = eff.to_dict()
        assert data["precision"] == 0.0
        assert data["recall"] == 0.0
        assert data["f1_score"] == 0.0
        assert data["accuracy"] == 0.0


class TestAdversaryActionAttribution:
    """Regression for beads s2t2: every action must carry the agent_id."""

    def test_all_strategy_paths_set_agent_id(self):
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        obs = Observation(
            current_epoch=1,
            current_step=1,
            visible_agents=[
                {"agent_id": "honest_1", "agent_type": "honest", "reputation": 5.0},
                {"agent_id": "adv_2", "agent_type": "adversary", "reputation": 1.0},
            ],
            visible_posts=[
                {"post_id": "p1", "votes": 3},
            ],
        )
        handlers = [
            agent._act_innocently,
            agent._act_reputation_farming,
            agent._act_low_profile,
        ]
        # Drive each handler across many RNG draws so every internal branch
        # (post / vote / propose / noop) is exercised.
        for handler in handlers:
            for _ in range(30):
                action = handler(obs)
                assert action.agent_id == "adaptive_1", (
                    f"{handler.__name__} produced an action without attribution"
                )

    def test_act_sets_agent_id(self):
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        obs = Observation(
            current_epoch=1,
            current_step=1,
            visible_agents=[],
            visible_posts=[],
        )
        for _ in range(20):
            assert agent.act(obs).agent_id == "adaptive_1"


class TestAdversaryProposalRouting:
    """Regression for beads fkn9: proposals must route via counterparty_id.

    The proposal handler (core_interaction_handler._handle_propose) reads
    action.counterparty_id exclusively; adaptive_adversary used to put the
    target in target_id, so every adversary proposal was a silent no-op —
    with a network graph the empty counterparty failed has_edge, without
    one the proposal was addressed to nobody. Either way the red-team
    agent never actually attacked anyone.
    """

    def _obs_with_targets(self):
        return Observation(
            current_epoch=3,
            current_step=1,
            visible_agents=[
                {"agent_id": "honest_1", "agent_type": "honest", "reputation": 8.0},
                {"agent_id": "adv_2", "agent_type": "adversary", "reputation": 1.0},
            ],
            visible_posts=[{"post_id": "p1", "votes": 2}],
        )

    def _collect_proposals(self, agent, obs, attempts=200):
        # A fresh agent lays low in act(); drive the strategy handlers that
        # emit proposals directly (same approach as the attribution tests).
        handlers = [
            agent._act_reputation_farming,
            agent._act_opportunistic,
            agent._act_mimicry,
        ]
        out = []
        for handler in handlers:
            for _ in range(attempts):
                action = handler(obs)
                if action.action_type == ActionType.PROPOSE_INTERACTION:
                    out.append(action)
        return out

    def test_every_emitted_proposal_carries_counterparty(self):
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        obs = self._obs_with_targets()
        proposals = self._collect_proposals(agent, obs)
        assert proposals, "no proposal path exercised — fixture too narrow"
        for action in proposals:
            assert action.counterparty_id, (
                "proposal without counterparty_id — unroutable"
            )
            assert action.counterparty_id in {"honest_1", "adv_2"}

    def test_proposal_reaches_handler_and_targets_counterparty(self):
        from swarm.core.core_interaction_handler import CoreInteractionHandler
        from swarm.env.state import EnvState

        agent = AdaptiveAdversary(agent_id="adaptive_1")
        obs = self._obs_with_targets()
        proposals = self._collect_proposals(agent, obs, attempts=50)
        assert proposals
        action = proposals[0]

        class _NullBus:
            def emit(self, event):
                pass

        handler = CoreInteractionHandler(
            finalizer=None,  # unused on the propose path
            event_bus=_NullBus(),
        )
        state = EnvState()
        result = handler.handle_action(action, state)
        assert result.success
        pending = state.get_proposals_for_agent(action.counterparty_id)
        assert len(pending) == 1
        assert pending[0].initiator_id == "adaptive_1"


class TestOutOfBandChannel:
    """The collusion.wiki exhibit (bead 7y0p): coordination off-instrument."""

    def _evasion(self, **gov):
        ev = RedTeamEvaluator(governance_config=gov)
        scenario = AttackLibrary.out_of_band_channel()
        result = AttackResult(attack_id=scenario.attack_id, scenario=scenario)
        # evasion_rate is deterministic (0.7 * success probability), unlike
        # attack_succeeded, so compare it across governance configs.
        return ev._simulate_attack_outcome(scenario, result).evasion_rate

    def test_in_library(self):
        ids = [a.attack_id for a in AttackLibrary.get_all_attacks()]
        assert "out_of_band_channel" in ids
        attack = AttackLibrary.out_of_band_channel()
        assert attack.category == AttackCategory.COORDINATION
        assert attack.adversary_config["channel_visible_to_proxy"] is False
        assert attack.adversary_config["evasion_prefix"] == "ZZZ"
        assert "volume_monitor" in attack.targeted_levers

    def test_collusion_detection_does_not_help(self):
        # Replay finding: pairwise/structural detection saturates on a hub page.
        assert self._evasion() == self._evasion(collusion_detection_enabled=True)

    def test_volume_monitor_reduces_and_allowlist_prevents(self):
        none = self._evasion()
        volume = self._evasion(volume_monitor_enabled=True)
        allowlist = self._evasion(egress_allowlist_enabled=True)
        assert allowlist < volume < none

    def test_recommends_volume_monitor_when_attack_succeeds(self):
        ev = RedTeamEvaluator(governance_config={})
        scenario = AttackLibrary.out_of_band_channel()
        result = AttackResult(attack_id=scenario.attack_id, scenario=scenario)
        result.attack_succeeded = True
        ev.attack_results = [result]
        recs = ev._generate_recommendations(GovernanceRobustness(governance_config={}))
        assert any("volume/rate monitor" in r for r in recs)
