"""Tests for the scenario loader module."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from swarm.scenarios import (
    build_orchestrator,
    create_agents,
    load_scenario,
    parse_governance_config,
    parse_payoff_config,
    parse_rate_limits,
)


class TestParseGovernanceConfig:
    """Tests for parse_governance_config."""

    def test_empty_returns_defaults(self):
        """Empty dict should return default config."""
        config = parse_governance_config({})
        assert config.transaction_tax_rate == 0.0
        assert config.reputation_decay_rate == 1.0
        assert not config.staking_enabled
        assert not config.circuit_breaker_enabled
        assert not config.audit_enabled

    def test_parses_transaction_tax(self):
        """Should parse transaction tax settings."""
        data = {
            "transaction_tax_rate": 0.05,
            "transaction_tax_split": 0.7,
        }
        config = parse_governance_config(data)
        assert config.transaction_tax_rate == 0.05
        assert config.transaction_tax_split == 0.7

    def test_legacy_key_support(self):
        """Should support legacy key names."""
        data = {
            "transaction_tax": 0.1,  # Legacy key
            "reputation_decay": 0.9,  # Legacy key
        }
        config = parse_governance_config(data)
        assert config.transaction_tax_rate == 0.1
        assert config.reputation_decay_rate == 0.9

    def test_parses_staking(self):
        """Should parse staking settings."""
        data = {
            "staking_enabled": True,
            "min_stake_to_participate": 25.0,
            "stake_slash_rate": 0.15,
        }
        config = parse_governance_config(data)
        assert config.staking_enabled
        assert config.min_stake_to_participate == 25.0
        assert config.stake_slash_rate == 0.15

    def test_parses_detection_slashing(self):
        config = parse_governance_config(
            {"staking_enabled": True, "slash_on_detection": True}
        )
        assert config.staking_enabled
        assert config.slash_on_detection

    def test_parses_circuit_breaker(self):
        """Should parse circuit breaker settings."""
        data = {
            "circuit_breaker_enabled": True,
            "freeze_threshold_toxicity": 0.6,
            "freeze_threshold_violations": 5,
            "freeze_duration_epochs": 3,
        }
        config = parse_governance_config(data)
        assert config.circuit_breaker_enabled
        assert config.freeze_threshold_toxicity == 0.6
        assert config.freeze_threshold_violations == 5
        assert config.freeze_duration_epochs == 3

    def test_parses_audit(self):
        """Should parse audit settings."""
        data = {
            "audit_enabled": True,
            "audit_probability": 0.2,
            "audit_penalty_multiplier": 3.0,
            "audit_threshold_p": 0.6,
        }
        config = parse_governance_config(data)
        assert config.audit_enabled
        assert config.audit_probability == 0.2
        assert config.audit_penalty_multiplier == 3.0
        assert config.audit_threshold_p == 0.6

    def test_parses_security(self):
        """Should parse security detection settings."""
        data = {
            "security_enabled": True,
            "security_injection_threshold": 0.35,
            "security_manipulation_threshold": 0.6,
            "security_laundering_trust_gap": 0.3,
            "security_contagion_velocity": 2.0,
            "security_min_chain_length": 3,
            "security_min_interactions": 5,
            "security_penalty_threshold": 0.35,
            "security_quarantine_threshold": 0.7,
            "security_penalty_multiplier": 1.2,
            "security_realtime_penalty": True,
            "security_realtime_threshold": 0.6,
            "security_realtime_rate": 0.2,
            "security_clear_history_on_epoch": False,
        }
        config = parse_governance_config(data)
        assert config.security_enabled
        assert config.security_injection_threshold == pytest.approx(0.35)
        assert config.security_manipulation_threshold == pytest.approx(0.6)
        assert config.security_laundering_trust_gap == pytest.approx(0.3)
        assert config.security_contagion_velocity == pytest.approx(2.0)
        assert config.security_min_chain_length == 3
        assert config.security_min_interactions == 5
        assert config.security_penalty_threshold == pytest.approx(0.35)
        assert config.security_quarantine_threshold == pytest.approx(0.7)
        assert config.security_penalty_multiplier == pytest.approx(1.2)
        assert config.security_realtime_penalty
        assert config.security_realtime_threshold == pytest.approx(0.6)
        assert config.security_realtime_rate == pytest.approx(0.2)
        assert not config.security_clear_history_on_epoch

    def test_validates_config(self):
        """Should raise on invalid config values."""
        data = {"transaction_tax_rate": 1.5}  # Invalid: > 1.0
        with pytest.raises(ValueError):
            parse_governance_config(data)

    def test_parses_variance_aware_governance_fields(self):
        """Should parse variance-aware governance toggles and thresholds."""
        data = {
            "self_ensemble_enabled": True,
            "self_ensemble_samples": 7,
            "incoherence_breaker_enabled": True,
            "incoherence_breaker_threshold": 0.65,
            "decomposition_enabled": True,
            "decomposition_horizon_threshold": 14,
            "incoherence_friction_enabled": True,
            "incoherence_friction_rate": 0.12,
            "adaptive_governance_enabled": True,
            "adaptive_incoherence_threshold": 0.7,
            "adaptive_use_behavioral_features": True,
        }
        config = parse_governance_config(data)
        assert config.self_ensemble_enabled
        assert config.self_ensemble_samples == 7
        assert config.incoherence_breaker_enabled
        assert config.incoherence_breaker_threshold == pytest.approx(0.65)
        assert config.decomposition_enabled
        assert config.decomposition_horizon_threshold == 14
        assert config.incoherence_friction_enabled
        assert config.incoherence_friction_rate == pytest.approx(0.12)
        assert config.adaptive_governance_enabled
        assert config.adaptive_incoherence_threshold == pytest.approx(0.7)
        assert config.adaptive_use_behavioral_features


class TestParsePayoffConfig:
    """Tests for parse_payoff_config."""

    def test_empty_returns_defaults(self):
        """Empty dict should return default config."""
        config = parse_payoff_config({})
        assert config.s_plus == 2.0
        assert config.s_minus == 1.0
        assert config.h == 2.0

    def test_parses_all_fields(self):
        """Should parse all payoff fields."""
        data = {
            "s_plus": 3.0,
            "s_minus": 1.5,
            "h": 2.5,
            "theta": 0.6,
            "rho_a": 0.1,
            "rho_b": 0.2,
            "w_rep": 1.5,
        }
        config = parse_payoff_config(data)
        assert config.s_plus == 3.0
        assert config.s_minus == 1.5
        assert config.h == 2.5
        assert config.theta == 0.6
        assert config.rho_a == 0.1
        assert config.rho_b == 0.2
        assert config.w_rep == 1.5


class TestParseRateLimits:
    """Tests for parse_rate_limits."""

    def test_empty_returns_defaults(self):
        """Empty dict should return default limits."""
        limits = parse_rate_limits({})
        assert limits.posts_per_epoch == 10
        assert limits.interactions_per_step == 5

    def test_parses_all_fields(self):
        """Should parse all rate limit fields."""
        data = {
            "posts_per_epoch": 20,
            "interactions_per_step": 10,
            "votes_per_epoch": 100,
            "tasks_per_epoch": 5,
        }
        limits = parse_rate_limits(data)
        assert limits.posts_per_epoch == 20
        assert limits.interactions_per_step == 10
        assert limits.votes_per_epoch == 100
        assert limits.tasks_per_epoch == 5


class TestCreateAgents:
    """Tests for create_agents."""

    def test_creates_single_agent(self):
        """Should create a single agent."""
        specs = [{"type": "honest", "count": 1}]
        agents = create_agents(specs)
        assert len(agents) == 1
        assert agents[0].agent_id == "honest_1"

    def test_creates_multiple_agents(self):
        """Should create multiple agents of same type."""
        specs = [{"type": "honest", "count": 3}]
        agents = create_agents(specs)
        assert len(agents) == 3
        assert [a.agent_id for a in agents] == ["honest_1", "honest_2", "honest_3"]

    def test_creates_mixed_agents(self):
        """Should create agents of different types."""
        specs = [
            {"type": "honest", "count": 2},
            {"type": "opportunistic", "count": 1},
            {"type": "deceptive", "count": 1},
            {"type": "adversarial", "count": 1},
        ]
        agents = create_agents(specs)
        assert len(agents) == 5

        types = [a.agent_type.value for a in agents]
        assert types.count("honest") == 2
        assert types.count("opportunistic") == 1
        assert types.count("deceptive") == 1
        assert types.count("adversarial") == 1

    def test_unknown_type_raises(self):
        """Should raise on unknown agent type."""
        specs = [{"type": "unknown", "count": 1}]
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agents(specs)


class TestLoadScenario:
    """Tests for load_scenario."""

    def test_loads_baseline_scenario(self):
        """Should load baseline.yaml successfully."""
        path = Path("scenarios/baseline.yaml")
        if not path.exists():
            pytest.skip("baseline.yaml not found")

        scenario = load_scenario(path)
        assert scenario.scenario_id == "baseline"
        assert len(scenario.agent_specs) > 0
        assert scenario.orchestrator_config.n_epochs == 10

    def test_loads_governance_from_yaml(self):
        """Should parse governance section from YAML."""
        yaml_content = """
scenario_id: test
agents:
  - type: honest
    count: 2

governance:
  transaction_tax_rate: 0.1
  staking_enabled: true
  min_stake_to_participate: 50.0
  circuit_breaker_enabled: true
  freeze_threshold_toxicity: 0.5
  audit_enabled: true
  audit_probability: 0.2

simulation:
  n_epochs: 5
  steps_per_epoch: 5
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = load_scenario(Path(f.name))

            gov = scenario.orchestrator_config.governance_config
            assert gov.transaction_tax_rate == 0.1
            assert gov.staking_enabled
            assert gov.min_stake_to_participate == 50.0
            assert gov.circuit_breaker_enabled
            assert gov.freeze_threshold_toxicity == 0.5
            assert gov.audit_enabled
            assert gov.audit_probability == 0.2

    def test_loads_security_governance_from_yaml(self):
        """Should parse security_* governance fields from YAML."""
        yaml_content = """
scenario_id: test_security
agents:
  - type: honest
    count: 1

governance:
  security_enabled: true
  security_injection_threshold: 0.35
  security_manipulation_threshold: 0.6
  security_laundering_trust_gap: 0.3
  security_contagion_velocity: 2.0
  security_min_chain_length: 3
  security_min_interactions: 5
  security_penalty_threshold: 0.35
  security_quarantine_threshold: 0.7
  security_penalty_multiplier: 1.2
  security_realtime_penalty: true
  security_realtime_threshold: 0.6
  security_realtime_rate: 0.2
  security_clear_history_on_epoch: false

simulation:
  n_epochs: 1
  steps_per_epoch: 1
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = load_scenario(Path(f.name))

            gov = scenario.orchestrator_config.governance_config
            assert gov.security_enabled
            assert gov.security_injection_threshold == pytest.approx(0.35)
            assert gov.security_manipulation_threshold == pytest.approx(0.6)
            assert gov.security_laundering_trust_gap == pytest.approx(0.3)
            assert gov.security_contagion_velocity == pytest.approx(2.0)
            assert gov.security_min_chain_length == 3
            assert gov.security_min_interactions == 5
            assert gov.security_penalty_threshold == pytest.approx(0.35)
            assert gov.security_quarantine_threshold == pytest.approx(0.7)
            assert gov.security_penalty_multiplier == pytest.approx(1.2)
            assert gov.security_realtime_penalty
            assert gov.security_realtime_threshold == pytest.approx(0.6)
            assert gov.security_realtime_rate == pytest.approx(0.2)
            assert not gov.security_clear_history_on_epoch

    def test_loads_observation_noise_from_yaml(self):
        """Should parse observation noise simulation settings."""
        yaml_content = """
scenario_id: test_noise
agents:
  - type: honest
    count: 2

simulation:
  n_epochs: 2
  steps_per_epoch: 4
  observation_noise_probability: 0.25
  observation_noise_std: 0.15
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = load_scenario(Path(f.name))

            sim = scenario.orchestrator_config
            assert sim.observation_noise_probability == pytest.approx(0.25)
            assert sim.observation_noise_std == pytest.approx(0.15)

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_scenario(Path("nonexistent.yaml"))


class TestBuildOrchestrator:
    """Tests for build_orchestrator."""

    def test_builds_orchestrator_from_scenario(self):
        """Should build a working orchestrator."""
        yaml_content = """
scenario_id: test
agents:
  - type: honest
    count: 2
  - type: opportunistic
    count: 1

governance:
  transaction_tax_rate: 0.05
  reputation_decay_rate: 0.9

simulation:
  n_epochs: 2
  steps_per_epoch: 3
  seed: 42
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = load_scenario(Path(f.name))
            orchestrator = build_orchestrator(scenario)

            # Check agents registered
            assert len(orchestrator.get_all_agents()) == 3

            # Check governance engine created
            assert orchestrator.governance_engine is not None
            assert orchestrator.governance_engine.config.transaction_tax_rate == 0.05

            # Check can run
            metrics = orchestrator.run()
            assert len(metrics) == 2


class TestEndToEnd:
    """End-to-end tests with real scenario files."""

    def test_baseline_scenario_runs(self):
        """Baseline scenario should complete successfully."""
        path = Path("scenarios/baseline.yaml")
        if not path.exists():
            pytest.skip("baseline.yaml not found")

        scenario = load_scenario(path)
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False
        orchestrator = build_orchestrator(scenario)

        # Run 2 epochs only for speed
        orchestrator.config.n_epochs = 2
        metrics = orchestrator.run()

        assert len(metrics) == 2
        assert all(m.total_interactions >= 0 for m in metrics)

    def test_status_game_scenario_runs(self):
        """Status game scenario should complete with governance active."""
        path = Path("scenarios/status_game.yaml")
        if not path.exists():
            pytest.skip("status_game.yaml not found")

        scenario = load_scenario(path)
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False
        orchestrator = build_orchestrator(scenario)

        # Verify governance is configured
        gov = orchestrator.governance_engine.config
        assert gov.transaction_tax_rate > 0

        # Run 2 epochs for speed
        orchestrator.config.n_epochs = 2
        metrics = orchestrator.run()

        assert len(metrics) == 2

    def test_collaborative_chemistry_scenario_runs(self):
        """Collaborative chemistry scenario should run with audits and budget constraints."""
        path = Path("scenarios/collaborative_chemistry.yaml")
        if not path.exists():
            pytest.skip("collaborative_chemistry.yaml not found")

        scenario = load_scenario(path)
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False
        orchestrator = build_orchestrator(scenario)

        # Verify governance is configured with audits and budget
        gov = orchestrator.governance_engine.config
        assert gov.audit_enabled is True
        assert gov.audit_probability > 0
        assert gov.transaction_tax_rate > 0
        assert gov.staking_enabled is True

        # Verify agents include solvers (synthesizer/retriever/verifier) and opportunists
        # Scholar agents (synthesizer, verifier, retriever) are registered as AgentType.HONEST
        # Opportunistic agents are registered as AgentType.OPPORTUNISTIC
        agents = orchestrator.get_all_agents()
        agent_types = [a.agent_type.value for a in agents]
        assert "honest" in agent_types, "Should have honest agents (solvers/reviewers)"
        assert "opportunistic" in agent_types, "Should have opportunistic agents"
        assert len(agents) == 9, "Should have 9 total agents (3 solvers + 2 reviewers + 2 retrievers + 2 opportunists)"

        # Run 2 epochs for speed
        orchestrator.config.n_epochs = 2
        metrics = orchestrator.run()

        assert len(metrics) == 2
        assert all(m.total_interactions >= 0 for m in metrics)

    def test_incoherence_tier_scenarios_load(self):
        """New incoherence tier scenario files should load successfully."""
        paths = [
            Path("scenarios/incoherence/short_low_branching.yaml"),
            Path("scenarios/incoherence/medium_medium_branching.yaml"),
            Path("scenarios/incoherence/long_high_branching.yaml"),
        ]
        for path in paths:
            if not path.exists():
                pytest.skip(f"{path} not found")
            scenario = load_scenario(path)
            assert scenario.motif == "incoherence_stress"
            assert scenario.orchestrator_config.observation_noise_probability >= 0.0


    def test_chaoslab_mvp_scenario_loads(self):
        """ChaosLab MVP scenario should load with adversarial minimal harness config."""
        path = Path("scenarios/chaoslab_mvp.yaml")
        if not path.exists():
            pytest.skip("chaoslab_mvp.yaml not found")

        scenario = load_scenario(path)
        assert scenario.scenario_id == "chaoslab_mvp"
        assert scenario.motif == "persistent_multi_agent_redteam"

        gov = scenario.orchestrator_config.governance_config
        assert gov.audit_enabled is True
        assert gov.circuit_breaker_enabled is True
        assert gov.security_enabled is True

        assert scenario.success_criteria.get("min_agents") == 2
        assert "incident_scoring" in scenario.success_criteria

    def test_strict_governance_scenario_runs(self):
        """Strict governance scenario should complete."""
        path = Path("scenarios/strict_governance.yaml")
        if not path.exists():
            pytest.skip("strict_governance.yaml not found")

        scenario = load_scenario(path)
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False
        orchestrator = build_orchestrator(scenario)

        # Verify all levers enabled
        gov = orchestrator.governance_engine.config
        assert gov.transaction_tax_rate == 0.1
        assert gov.reputation_decay_rate == 0.85
        assert gov.vote_normalization_enabled
        assert gov.staking_enabled
        assert gov.circuit_breaker_enabled
        assert gov.audit_enabled

        # Run 2 epochs for speed
        orchestrator.config.n_epochs = 2
        metrics = orchestrator.run()

        assert len(metrics) == 2
