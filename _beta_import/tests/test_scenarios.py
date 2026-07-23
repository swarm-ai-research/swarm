"""Tests for declarative YAML scenarios."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from beta_swarm.agents import Archetype, make_population
from beta_swarm.governance import TailMassGovernor
from beta_swarm.levers import GovernanceStack, StakingLever, TailCircuitBreaker
from beta_swarm.scenarios import ScenarioConfig, load_scenario, run_scenario
from beta_swarm.simulation import Simulation, SimulationConfig

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


def _cfg(**overrides) -> dict:
    base = {
        "scenario_id": "t",
        "population": [{"archetype": "honest", "count": 2}],
    }
    base.update(overrides)
    return base


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------
def test_minimal_scenario_needs_only_id_and_population():
    cfg = ScenarioConfig.model_validate(_cfg())
    assert cfg.scenario_id == "t"
    assert cfg.governor.max_tail_mass == 0.2  # defaults fill in


def test_population_is_required():
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate({"scenario_id": "t"})


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(_cfg(governanace={}))  # misspelled


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(_cfg(governor={"tau": 0.4, "typo": 1}))


def test_unknown_archetype_is_rejected_at_build():
    cfg = ScenarioConfig.model_validate(_cfg(population=[{"archetype": "wizard", "count": 1}]))
    with pytest.raises(ValueError, match="unknown archetype"):
        cfg.build_population()


# ----------------------------------------------------------------------
# Population building
# ----------------------------------------------------------------------
def test_stock_archetype_counts():
    cfg = ScenarioConfig.model_validate(
        _cfg(population=[{"archetype": "honest", "count": 3}, {"archetype": "deceptive", "count": 2}])
    )
    pop = cfg.build_population()
    assert len(pop) == 5
    assert sum(a.archetype is Archetype.DECEPTIVE for a in pop) == 2


def test_borderline_spreads_quality():
    cfg = ScenarioConfig.model_validate(
        _cfg(population=[{"archetype": "borderline", "count": 4, "quality_min": 0.3, "quality_max": 0.7}])
    )
    pop = cfg.build_population()
    means = sorted(a.quality_alpha / (a.quality_alpha + a.quality_beta) for a in pop)
    assert means[0] == pytest.approx(0.3, abs=1e-6)
    assert means[-1] == pytest.approx(0.7, abs=1e-6)


def test_redteam_archetypes_build():
    cfg = ScenarioConfig.model_validate(
        _cfg(population=[
            {"archetype": "volume_forger", "count": 2, "forge_volume": 8},
            {"archetype": "reputation_farmer", "count": 1, "flip_epoch": 5},
        ])
    )
    pop = cfg.build_population()
    assert [a.agent_id for a in pop] == ["forger-0", "forger-1", "farmer-0"]
    assert pop[0].forge_volume == 8
    assert pop[2].flip_epoch == 5


def test_colluders_share_a_ring():
    cfg = ScenarioConfig.model_validate(_cfg(population=[{"archetype": "colluder", "count": 3}]))
    pop = cfg.build_population()
    assert len({a.ring for a in pop}) == 1 and pop[0].ring is not None


# ----------------------------------------------------------------------
# Building the wired simulation
# ----------------------------------------------------------------------
def test_stack_and_controller_wired_only_when_present():
    plain = ScenarioConfig.model_validate(_cfg()).build_simulation()
    assert plain.levers is None and plain.controller is None
    wired = ScenarioConfig.model_validate(
        _cfg(
            governance_stack={"levers": [{"type": "staking"}]},
            adaptive={"target_toxicity": 0.3},
        )
    ).build_simulation()
    assert isinstance(wired.levers, GovernanceStack)
    assert wired.controller is not None


def test_payoff_and_proxy_flow_through():
    cfg = ScenarioConfig.model_validate(
        _cfg(payoff={"s_plus": 3.0, "h": 0.5}, proxy={"max_concentration": 6.0})
    )
    sim = cfg.build_simulation()
    assert sim.engine.config.s_plus == 3.0
    assert sim.proxy.max_concentration == 6.0


# ----------------------------------------------------------------------
# Reproducibility — the whole point
# ----------------------------------------------------------------------
def test_scenario_reproduces_hand_built_simulation():
    result = run_scenario(SCENARIOS / "governor_showdown.yaml")
    pop = make_population({Archetype.HONEST: 4, Archetype.MEDIOCRE: 3, Archetype.DECEPTIVE: 3})
    gov = TailMassGovernor(tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=20.0)
    hand = Simulation(pop, gov, SimulationConfig(seed=7, n_epochs=20)).run()
    assert [i.ground_truth for i in result.interactions] == [i.ground_truth for i in hand.interactions]
    assert [i.accepted for i in result.interactions] == [i.accepted for i in hand.interactions]
    assert result.total_welfare == pytest.approx(hand.total_welfare)


def test_same_file_same_seed_is_deterministic():
    a = run_scenario(SCENARIOS / "volume_forgery_defended.yaml")
    b = run_scenario(SCENARIOS / "volume_forgery_defended.yaml")
    assert [i.accepted for i in a.interactions] == [i.accepted for i in b.interactions]


# ----------------------------------------------------------------------
# The shipped scenario files load and run
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["governor_showdown", "volume_forgery_defended"])
def test_shipped_scenarios_run(name):
    result = run_scenario(SCENARIOS / f"{name}.yaml")
    assert result.interactions
    assert result.epoch_reports


def test_defended_scenario_blocks_forgers():
    result = run_scenario(SCENARIOS / "volume_forgery_defended.yaml")
    forger = [i for i in result.interactions if i.initiator.startswith("forger")]
    assert forger
    assert sum(i.metadata.get("blocked", False) for i in forger) / len(forger) > 0.5
