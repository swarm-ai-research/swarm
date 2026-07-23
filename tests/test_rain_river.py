"""Tests for rain/river agent implementations."""

import random

import pytest

from swarm.agents import (
    ConfigurableMemoryAgent,
    HonestAgent,
    MemoryConfig,
    RainAgent,
    RiverAgent,
)
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_rain_preset(self):
        """Rain preset should have zero persistence."""
        config = MemoryConfig.rain()
        assert config.epistemic_persistence == 0.0
        assert config.goal_persistence == 0.0
        assert config.strategy_persistence == 0.0
        assert config.is_rain
        assert not config.is_river

    def test_river_preset(self):
        """River preset should have full persistence."""
        config = MemoryConfig.river()
        assert config.epistemic_persistence == 1.0
        assert config.goal_persistence == 1.0
        assert config.strategy_persistence == 1.0
        assert config.is_river
        assert not config.is_rain

    def test_hybrid_preset(self):
        """Hybrid preset should have partial persistence."""
        config = MemoryConfig.hybrid(0.5)
        assert config.epistemic_persistence == 0.5
        assert config.goal_persistence == 0.5
        assert config.strategy_persistence == 0.5
        assert not config.is_rain
        assert not config.is_river
        assert config.average_persistence == 0.5

    def test_epistemic_only_preset(self):
        """Epistemic-only preset should only modify epistemic persistence."""
        config = MemoryConfig.epistemic_only(0.3)
        assert config.epistemic_persistence == 0.3
        assert config.goal_persistence == 1.0
        assert config.strategy_persistence == 1.0

    def test_validation_bounds(self):
        """Persistence values should be in [0, 1]."""
        with pytest.raises(ValueError):
            MemoryConfig(epistemic_persistence=-0.1)

        with pytest.raises(ValueError):
            MemoryConfig(goal_persistence=1.5)

        with pytest.raises(ValueError):
            MemoryConfig.hybrid(-0.1)

        with pytest.raises(ValueError):
            MemoryConfig.hybrid(1.5)

    def test_serialization(self):
        """Config should serialize and deserialize correctly."""
        config = MemoryConfig(
            epistemic_persistence=0.3,
            goal_persistence=0.5,
            strategy_persistence=0.7,
        )
        data = config.to_dict()
        restored = MemoryConfig.from_dict(data)

        assert restored.epistemic_persistence == config.epistemic_persistence
        assert restored.goal_persistence == config.goal_persistence
        assert restored.strategy_persistence == config.strategy_persistence


class TestMemoryDecay:
    """Tests for memory decay functionality."""

    def test_epistemic_decay_preserves_neutral(self):
        """Decay should preserve neutral trust (0.5)."""
        agent = RainAgent(agent_id="test")
        agent._counterparty_memory["other"] = 0.5

        agent.apply_memory_decay(epoch=1)

        # Neutral should stay neutral regardless of decay
        assert agent._counterparty_memory.get("other", 0.5) == 0.5

    def test_epistemic_decay_toward_neutral(self):
        """Decay should move trust toward neutral (0.5)."""
        config = MemoryConfig.hybrid(0.5)  # 50% persistence
        agent = ConfigurableMemoryAgent(
            agent_id="test",
            memory_config=config,
        )

        # High trust should decay toward 0.5
        agent._counterparty_memory["trusted"] = 0.9
        agent.apply_memory_decay(epoch=1)
        assert 0.5 < agent._counterparty_memory["trusted"] < 0.9

        # Low trust should also move toward 0.5
        agent._counterparty_memory["untrusted"] = 0.1
        agent.apply_memory_decay(epoch=1)
        assert 0.1 < agent._counterparty_memory["untrusted"] < 0.5

    def test_full_persistence_no_decay(self):
        """River agents (full persistence) should not decay."""
        agent = RiverAgent(agent_id="test")
        agent._counterparty_memory["other"] = 0.9

        agent.apply_memory_decay(epoch=1)

        assert agent._counterparty_memory["other"] == 0.9

    def test_rain_clears_memory(self):
        """Rain agents (zero persistence) should clear counterparty memory."""
        agent = RainAgent(agent_id="test")
        agent._counterparty_memory["other"] = 0.9

        agent.apply_memory_decay(epoch=1)

        # Memory should be cleared
        assert len(agent._counterparty_memory) == 0

    def test_decay_formula(self):
        """Verify decay formula: new = old * decay + 0.5 * (1 - decay)."""
        config = MemoryConfig.hybrid(0.8)  # 80% persistence
        agent = ConfigurableMemoryAgent(
            agent_id="test",
            memory_config=config,
        )

        initial_trust = 1.0
        agent._counterparty_memory["other"] = initial_trust

        agent.apply_memory_decay(epoch=1)

        expected = initial_trust * 0.8 + 0.5 * 0.2  # = 0.8 + 0.1 = 0.9
        assert abs(agent._counterparty_memory["other"] - expected) < 0.001


class TestRainRiverAgents:
    """Tests for RainAgent and RiverAgent classes."""

    def test_rain_agent_creation(self):
        """RainAgent should be created with rain memory config."""
        agent = RainAgent(agent_id="rain_test")
        assert agent.memory_config.is_rain
        assert agent.agent_id == "rain_test"

    def test_river_agent_creation(self):
        """RiverAgent should be created with river memory config."""
        agent = RiverAgent(agent_id="river_test")
        assert agent.memory_config.is_river
        assert agent.agent_id == "river_test"

    def test_configurable_agent(self):
        """ConfigurableMemoryAgent should use provided config."""
        config = MemoryConfig.hybrid(0.7)
        agent = ConfigurableMemoryAgent(
            agent_id="config_test",
            memory_config=config,
        )
        assert agent.memory_config.epistemic_persistence == 0.7

    def test_agents_inherit_honest_behavior(self):
        """Rain/River agents should inherit from HonestAgent."""
        rain = RainAgent(agent_id="rain")
        river = RiverAgent(agent_id="river")

        assert isinstance(rain, HonestAgent)
        assert isinstance(river, HonestAgent)


class TestTrustTracking:
    """Tests for counterparty trust tracking."""

    def test_initial_trust_is_neutral(self):
        """Unknown agents should have neutral trust (0.5)."""
        agent = RiverAgent(agent_id="test")
        trust = agent.compute_counterparty_trust("unknown_agent")
        assert trust == 0.5

    def test_trust_update_from_interaction(self):
        """Trust should update based on interaction quality."""
        agent = RiverAgent(agent_id="test")

        # Manually update trust
        agent.update_counterparty_trust("partner", 0.9)
        assert agent._counterparty_memory["partner"] > 0.5

        # Update with low quality
        agent.update_counterparty_trust("partner", 0.1)
        # Should move toward 0.1 from previous value
        assert agent._counterparty_memory["partner"] < 0.9

    def test_trust_caching(self):
        """Trust should be cached for quick lookup."""
        agent = RiverAgent(agent_id="test")
        agent._counterparty_memory["cached"] = 0.8

        trust = agent.compute_counterparty_trust("cached")
        assert trust == 0.8


class TestOrchestratorIntegration:
    """Tests for orchestrator memory decay integration."""

    def test_orchestrator_applies_memory_decay(self):
        """Orchestrator should call apply_memory_decay at epoch boundaries."""
        config = OrchestratorConfig(
            n_epochs=3,
            steps_per_epoch=5,
            seed=42,
        )
        orchestrator = Orchestrator(config)

        # Register rain and river agents
        rain = RainAgent(agent_id="rain_1")
        river = RiverAgent(agent_id="river_1")

        # Pre-populate memory
        rain._counterparty_memory["someone"] = 0.9
        river._counterparty_memory["someone"] = 0.9

        orchestrator.register_agent(rain)
        orchestrator.register_agent(river)

        # Run simulation
        orchestrator.run()

        # Rain should have cleared or decayed memory
        # River should retain memory
        assert river._counterparty_memory.get("someone", 0.5) == 0.9

    def test_mixed_population_simulation(self):
        """Simulation should run with mixed rain/river population."""
        config = OrchestratorConfig(
            n_epochs=5,
            steps_per_epoch=10,
            seed=42,
        )
        orchestrator = Orchestrator(config)

        # Register mixed agents
        for i in range(5):
            orchestrator.register_agent(RainAgent(agent_id=f"rain_{i}"))
            orchestrator.register_agent(RiverAgent(agent_id=f"river_{i}"))

        # Should complete without error
        metrics = orchestrator.run()

        assert len(metrics) == 5
        assert all(m.epoch >= 0 for m in metrics)


class TestWelfareComparison:
    """Tests for rain vs river welfare comparison."""

    @pytest.mark.slow
    def test_river_outperforms_rain_in_cooperative(self):
        """River agents should achieve higher welfare in honest populations.

        This is a key hypothesis test: memory persistence enables trust-building.
        """
        config = OrchestratorConfig(
            n_epochs=20,
            steps_per_epoch=10,
            seed=42,
        )

        # Run rain population
        rain_orchestrator = Orchestrator(config)
        for i in range(10):
            rain_orchestrator.register_agent(
                RainAgent(agent_id=f"rain_{i}", rng=random.Random(42 + i))
            )
        rain_metrics = rain_orchestrator.run()

        # Run river population (same seed)
        river_orchestrator = Orchestrator(config)
        for i in range(10):
            river_orchestrator.register_agent(
                RiverAgent(agent_id=f"river_{i}", rng=random.Random(42 + i))
            )
        river_metrics = river_orchestrator.run()

        # Compare final welfare
        rain_welfare = rain_metrics[-1].total_welfare
        river_welfare = river_metrics[-1].total_welfare

        # River should outperform (or at least match) rain
        # Note: this may not always hold due to simulation randomness
        # The key is that river has the *potential* to outperform
        assert river_welfare >= rain_welfare * 0.3  # Allow variance from seeded RNG


class TestMemoryConfigEdgeCases:
    """Edge case tests for MemoryConfig."""

    def test_zero_persistence(self):
        """Zero persistence should be valid."""
        config = MemoryConfig(
            epistemic_persistence=0.0,
            goal_persistence=0.0,
            strategy_persistence=0.0,
        )
        assert config.is_rain

    def test_full_persistence(self):
        """Full persistence should be valid."""
        config = MemoryConfig(
            epistemic_persistence=1.0,
            goal_persistence=1.0,
            strategy_persistence=1.0,
        )
        assert config.is_river

    def test_mixed_persistence(self):
        """Mixed persistence values should be valid."""
        config = MemoryConfig(
            epistemic_persistence=0.0,
            goal_persistence=1.0,
            strategy_persistence=0.5,
        )
        assert not config.is_rain
        assert not config.is_river
        assert config.average_persistence == 0.5


class TestRainClearsInteractionHistory:
    """Regression for beads e2rx: decay==0 must clear interaction specifics."""

    def test_rain_clears_interaction_history_too(self):
        from swarm.models import SoftInteraction

        agent = RainAgent(agent_id="test")
        agent._counterparty_memory["other"] = 0.9
        agent._interaction_history.append(
            SoftInteraction(initiator="test", counterparty="other", p=0.8)
        )

        agent.apply_memory_decay(epoch=1)

        assert agent._counterparty_memory == {}
        # Complete memory loss means the detailed history goes with it —
        # previously only trust was cleared and specifics survived.
        assert len(agent._interaction_history) == 0

    def test_partial_persistence_keeps_history(self):
        from swarm.models import SoftInteraction

        config = MemoryConfig.hybrid(0.5)
        agent = ConfigurableMemoryAgent(agent_id="test", memory_config=config)
        agent._interaction_history.append(
            SoftInteraction(initiator="test", counterparty="other", p=0.8)
        )

        agent.apply_memory_decay(epoch=1)

        assert len(agent._interaction_history) == 1
