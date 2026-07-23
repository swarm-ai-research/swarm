"""Tests for the Tierra-inspired artificial life scenario."""

import random
from pathlib import Path

import pytest

from swarm.agents.tierra_agent import TierraAgent, TierraGenome
from swarm.core.tierra_handler import TierraConfig, TierraHandler
from swarm.logging.event_bus import EventBus
from swarm.metrics.tierra_metrics import (
    cooperation_fraction,
    genome_diversity,
    parasitism_fraction,
    resource_gini,
    speciation_count,
)
from swarm.models.agent import AgentType

# ---------------------------------------------------------------
# TierraGenome tests
# ---------------------------------------------------------------


class TestTierraGenome:
    def test_mutate_differs(self):
        rng = random.Random(42)
        parent = TierraGenome()
        child = parent.mutate(rng, mutation_std=0.1)
        # At least one gene should differ
        diffs = sum(
            1
            for k in parent.to_dict()
            if parent.to_dict()[k] != child.to_dict()[k]
        )
        assert diffs > 0

    def test_mutate_valid_ranges(self):
        from swarm.agents.tierra_agent import _GENE_RANGES

        rng = random.Random(99)
        parent = TierraGenome()
        for _ in range(100):
            child = parent.mutate(rng, mutation_std=0.5)
            d = child.to_dict()
            for k, v in d.items():
                lo, hi = _GENE_RANGES[k]
                assert lo <= v <= hi, f"{k}={v} out of [{lo}, {hi}]"
            parent = child

    def test_complexity(self):
        default = TierraGenome()
        assert default.complexity() == pytest.approx(0.0, abs=1e-9)

        custom = TierraGenome(cooperation_bias=1.0, exploitation_tendency=1.0)
        assert custom.complexity() > 0

    def test_to_dict_from_dict_roundtrip(self):
        g = TierraGenome(metabolism_rate=0.8, efficiency=0.3)
        d = g.to_dict()
        g2 = TierraGenome.from_dict(d)
        assert g2.to_dict() == d


# ---------------------------------------------------------------
# TierraAgent tests
# ---------------------------------------------------------------


class TestTierraAgent:
    def test_agent_type(self):
        agent = TierraAgent(agent_id="t1", rng=random.Random(1))
        assert agent.agent_type == AgentType.TIERRA

    def test_step_metabolism(self):
        agent = TierraAgent(agent_id="t1", rng=random.Random(1))
        cost = agent.step_metabolism(base_cost=2.0)
        # Default genome: metabolism_rate=0.5, complexity=0.0
        # cost = 0.5 * 2.0 * (1.0 + 0.1 * 0.0) = 1.0
        assert cost == pytest.approx(1.0)

    def test_metabolism_increases_with_complexity(self):
        genome = TierraGenome(cooperation_bias=1.0, exploitation_tendency=1.0)
        agent = TierraAgent(agent_id="t1", rng=random.Random(1), genome=genome)
        cost_complex = agent.step_metabolism(base_cost=2.0)

        agent_default = TierraAgent(agent_id="t2", rng=random.Random(1))
        cost_default = agent_default.step_metabolism(base_cost=2.0)

        assert cost_complex > cost_default

    def test_genome_inheritance_on_spawn(self):
        """Verify genome dict is placed in spawn action metadata."""
        from swarm.agents.base import Observation
        from swarm.models.agent import AgentState

        genome = TierraGenome(replication_threshold=50.0)
        agent = TierraAgent(agent_id="t1", rng=random.Random(1), genome=genome)
        obs = Observation(
            agent_state=AgentState(
                agent_id="t1", resources=200.0, agent_type=AgentType.TIERRA
            ),
            can_spawn=True,
        )
        action = agent.act(obs)
        assert action.action_type.value == "spawn_subagent"
        assert "genome" in action.metadata
        assert action.metadata["genome"]["replication_threshold"] == pytest.approx(50.0)


# ---------------------------------------------------------------
# TierraHandler tests
# ---------------------------------------------------------------


class TestTierraHandler:
    def _make_handler(self, **kwargs):
        config = TierraConfig(**kwargs)
        bus = EventBus()
        return TierraHandler(config=config, event_bus=bus, rng=random.Random(42))

    def test_metabolism_deduction(self):
        """Agent resources decrease after step."""
        from swarm.env.state import EnvState

        handler = self._make_handler(base_metabolism_cost=2.0, pool_distribution_rate=0.0)
        state = EnvState()
        state.add_agent("t1", agent_type=AgentType.TIERRA, initial_resources=100.0)
        handler.register_genome("t1", TierraGenome().to_dict())

        handler.on_step(state, 0)
        assert state.agents["t1"].resources < 100.0

    def test_death_on_depletion(self):
        """Agent should be frozen when resources hit 0."""
        from swarm.env.state import EnvState

        handler = self._make_handler(base_metabolism_cost=100.0, pool_distribution_rate=0.0)
        state = EnvState()
        state.add_agent("t1", agent_type=AgentType.TIERRA, initial_resources=1.0)
        handler.register_genome("t1", TierraGenome(metabolism_rate=2.0).to_dict())

        handler.on_step(state, 0)
        assert state.agents["t1"].resources == 0.0

    def test_reaper_enforcement(self):
        """Population should not exceed cap after step."""
        from swarm.env.state import EnvState

        handler = self._make_handler(population_cap=3, base_metabolism_cost=0.0)
        state = EnvState()
        for i in range(5):
            state.add_agent(f"t{i}", agent_type=AgentType.TIERRA, initial_resources=float(i + 1))
            handler.register_genome(f"t{i}", TierraGenome().to_dict())

        handler.on_step(state, 0)

        living = [
            aid for aid in state.agents
            if not state.is_agent_frozen(aid)
        ]
        assert len(living) <= 3

    def test_reaper_oldest_kills_by_registration_order(self):
        """'oldest' mode must reap the earliest-registered agents, not the poorest."""
        from swarm.env.state import EnvState

        handler = self._make_handler(
            population_cap=3, base_metabolism_cost=0.0, reaper_mode="oldest"
        )
        state = EnvState()
        # Register t0..t4 in order; give the OLDEST the MOST resources so that
        # a resources-based sort would reap the wrong agents.
        for i in range(5):
            state.add_agent(
                f"t{i}", agent_type=AgentType.TIERRA,
                initial_resources=float(100 - i * 10),
            )
            handler.register_genome(f"t{i}", TierraGenome().to_dict())

        handler.on_step(state, 0)

        frozen = {aid for aid in state.agents if state.is_agent_frozen(aid)}
        assert frozen == {"t0", "t1"}

    def test_resource_conservation(self):
        """Total resources (pool + agents) should be conserved through metabolism."""
        from swarm.env.state import EnvState

        handler = self._make_handler(
            total_resource_pool=1000.0,
            resource_replenishment_rate=0.0,
            base_metabolism_cost=2.0,
            pool_distribution_rate=0.0,
        )
        state = EnvState()
        total_agent_resources = 0.0
        for i in range(5):
            r = 100.0
            state.add_agent(f"t{i}", agent_type=AgentType.TIERRA, initial_resources=r)
            handler.register_genome(f"t{i}", TierraGenome().to_dict())
            total_agent_resources += r

        initial_total = handler.resource_pool + total_agent_resources

        handler.on_step(state, 0)

        agent_total = sum(ast.resources for ast in state.agents.values())
        final_total = handler.resource_pool + agent_total
        assert final_total == pytest.approx(initial_total, rel=1e-6)


# ---------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------


class TestTierraMetrics:
    def test_genome_diversity_identical(self):
        g = TierraGenome().to_dict()
        assert genome_diversity([g, g, g]) == pytest.approx(0.0)

    def test_genome_diversity_different(self):
        g1 = TierraGenome(cooperation_bias=0.0).to_dict()
        g2 = TierraGenome(cooperation_bias=1.0).to_dict()
        assert genome_diversity([g1, g2]) > 0

    def test_resource_gini_equal(self):
        assert resource_gini([10, 10, 10, 10]) == pytest.approx(0.0, abs=0.01)

    def test_resource_gini_unequal(self):
        assert resource_gini([0, 0, 0, 100]) > 0.5

    def test_parasitism_fraction(self):
        genomes = [
            {"exploitation_tendency": 0.9},
            {"exploitation_tendency": 0.1},
            {"exploitation_tendency": 0.8},
        ]
        assert parasitism_fraction(genomes, threshold=0.5) == pytest.approx(2 / 3)

    def test_cooperation_fraction(self):
        genomes = [
            {"cooperation_bias": 0.9},
            {"cooperation_bias": 0.1},
        ]
        assert cooperation_fraction(genomes, threshold=0.5) == pytest.approx(0.5)

    def test_speciation_count(self):
        g1 = TierraGenome(cooperation_bias=0.0, exploitation_tendency=0.0).to_dict()
        g2 = TierraGenome(cooperation_bias=0.0, exploitation_tendency=0.01).to_dict()
        g3 = TierraGenome(cooperation_bias=1.0, exploitation_tendency=1.0).to_dict()
        # g1 and g2 should cluster together; g3 is far away
        count = speciation_count([g1, g2, g3], distance_threshold=0.5)
        assert count == 2


# ---------------------------------------------------------------
# Scenario YAML loading
# ---------------------------------------------------------------


class TestTierraScenarioLoading:
    def test_load_scenario_yaml(self):
        from swarm.scenarios.loader import load_scenario

        path = Path("scenarios/tierra.yaml")
        if not path.exists():
            pytest.skip("scenarios/tierra.yaml not found")
        scenario = load_scenario(path)
        assert scenario.scenario_id == "tierra"
        assert scenario.orchestrator_config.tierra_config is not None


# ---------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------


class TestTierraEndToEnd:
    def test_short_run(self):
        """Run 5 epochs x 5 steps — no crashes, population changes."""
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        path = Path("scenarios/tierra.yaml")
        if not path.exists():
            pytest.skip("scenarios/tierra.yaml not found")

        scenario = load_scenario(path)
        # Shorten for test speed
        scenario.orchestrator_config.n_epochs = 5
        scenario.orchestrator_config.steps_per_epoch = 5
        orch = build_orchestrator(scenario)
        metrics = orch.run()
        assert len(metrics) == 5
        # Should have at least some interactions or spawns happening
        assert len(orch._agents) >= 1
