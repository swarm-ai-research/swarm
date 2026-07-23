"""Tests for composable governance levers."""

import pytest

from beta_swarm.agents import Archetype, make_population
from beta_swarm.belief import BetaBelief
from beta_swarm.governance import TailMassGovernor
from beta_swarm.levers import (
    GovernanceStack,
    LeverContext,
    LeverEffect,
    RiskTaxLever,
    StakingLever,
    TailCircuitBreaker,
)
from beta_swarm.redteam import AttackLibrary, run_attack
from beta_swarm.simulation import Simulation, SimulationConfig


def _ctx(agent_id="a", accepted=True, ground_truth=0.1, belief=None, resources=5.0, epoch=0):
    return LeverContext(
        epoch=epoch,
        agent_id=agent_id,
        belief=belief or BetaBelief.from_mean_concentration(0.6, 8.0),
        accepted=accepted,
        ground_truth=ground_truth,
        resources=resources,
    )


def _tail_gov():
    return TailMassGovernor(
        tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=20.0
    )


# ----------------------------------------------------------------------
# Effect algebra
# ----------------------------------------------------------------------
def test_effect_merge_is_additive():
    a = LeverEffect(cost=1.0, freeze={"x"}, resource_deltas={"a": -2.0})
    b = LeverEffect(cost=0.5, freeze={"y"}, resource_deltas={"a": -1.0, "b": 3.0})
    m = a.merge(b)
    assert m.cost == pytest.approx(1.5)
    assert m.freeze == {"x", "y"}
    assert m.resource_deltas == {"a": pytest.approx(-3.0), "b": pytest.approx(3.0)}


# ----------------------------------------------------------------------
# Individual levers
# ----------------------------------------------------------------------
def test_staking_slashes_on_realized_tail_outcome():
    lever = StakingLever(min_stake=1.0, slash_rate=0.5, tau=0.4)
    bad = lever.on_interaction(_ctx(ground_truth=0.1, resources=4.0))
    assert bad.resource_deltas["a"] == pytest.approx(-2.0)
    # A good realized outcome, or a rejected interaction, is not slashed.
    assert lever.on_interaction(_ctx(ground_truth=0.9)).resource_deltas == {}
    assert lever.on_interaction(_ctx(accepted=False)).resource_deltas == {}


def test_staking_bars_agents_below_min_stake():
    lever = StakingLever(min_stake=1.0)
    assert lever.can_act("a", resources=2.0)
    assert not lever.can_act("a", resources=0.5)


def test_circuit_breaker_freezes_on_sustained_realized_exposure():
    cb = TailCircuitBreaker(max_tail_exposure=0.5, window=6, min_samples=4, freeze_epochs=3, tau=0.4)
    cb.reset()
    effects = [cb.on_interaction(_ctx(ground_truth=0.1)) for _ in range(4)]  # all bad
    assert any(e.freeze == {"a"} for e in effects)
    assert not cb.can_act("a", 5.0)
    # Unfreezes once the freeze window elapses.
    unfreeze = cb.on_epoch_start(epoch=3)
    assert "a" in unfreeze.unfreeze
    assert cb.can_act("a", 5.0)


def test_circuit_breaker_ignores_unobserved_and_good_outcomes():
    cb = TailCircuitBreaker(max_tail_exposure=0.5, min_samples=4, tau=0.4)
    cb.reset()
    for _ in range(10):
        cb.on_interaction(_ctx(accepted=False, ground_truth=None))  # unobserved
        cb.on_interaction(_ctx(ground_truth=0.9))                   # good
    assert cb.can_act("a", 5.0)


def test_circuit_breaker_realized_vs_exante():
    """Realized mode judges ground truth; ex-ante mode judges the belief."""
    good_belief_bad_outcome = _ctx(
        belief=BetaBelief.from_mean_concentration(0.8, 40.0), ground_truth=0.1
    )
    realized = TailCircuitBreaker(min_samples=1, max_tail_exposure=0.5, use_realized=True)
    realized.reset()
    assert realized.on_interaction(good_belief_bad_outcome).freeze == {"a"}  # bad outcome caught
    exante = TailCircuitBreaker(min_samples=1, max_tail_exposure=0.5, use_realized=False)
    exante.reset()
    assert exante.on_interaction(good_belief_bad_outcome).freeze == set()  # belief looked fine


def test_risk_tax_scales_with_tail_mass():
    lever = RiskTaxLever(rate=2.0, tau=0.4)
    diffuse = _ctx(belief=BetaBelief.from_mean_concentration(0.5, 2.0))
    sharp = _ctx(belief=BetaBelief.from_mean_concentration(0.5, 100.0))
    assert lever.on_interaction(diffuse).cost > lever.on_interaction(sharp).cost
    assert lever.on_interaction(_ctx(accepted=False)).cost == 0.0


# ----------------------------------------------------------------------
# The stack + simulation integration
# ----------------------------------------------------------------------
def test_stack_requires_a_lever():
    with pytest.raises(ValueError):
        GovernanceStack([])


def test_stack_applies_slashes_and_freezes():
    stack = GovernanceStack([StakingLever(slash_rate=0.5, tau=0.4)], initial_resources=4.0)
    stack.reset(["a"])
    stack.on_interaction(_ctx(agent_id="a", ground_truth=0.1, resources=4.0))
    assert stack.resources["a"] == pytest.approx(2.0)


def test_no_stack_is_unchanged():
    """A simulation without levers behaves identically to before."""
    pop = lambda: make_population({Archetype.HONEST: 3, Archetype.DECEPTIVE: 2})
    cfg = SimulationConfig(seed=1, n_epochs=5)
    a = Simulation(pop(), _tail_gov(), cfg).run()
    b = Simulation(pop(), _tail_gov(), cfg, levers=None).run()
    assert [i.ground_truth for i in a.interactions] == [i.ground_truth for i in b.interactions]


def test_blocked_agents_produce_rejected_placeholder_interactions():
    """A frozen/broke agent still yields a rejected record, keeping counts stable."""
    stack = GovernanceStack([StakingLever(min_stake=100.0)], initial_resources=5.0)
    result = Simulation(
        make_population({Archetype.HONEST: 2}), _tail_gov(),
        SimulationConfig(seed=0, n_epochs=2), levers=stack,
    ).run()
    assert result.interactions  # everyone is barred (min_stake unreachable)...
    assert all(i.metadata.get("blocked") for i in result.interactions)
    assert all(not i.accepted and i.ground_truth is None for i in result.interactions)


def test_staking_contains_volume_forgery_without_belief_fix():
    """The headline: a realized-outcome lever catches what the belief gate misses."""
    base = run_attack(AttackLibrary.volume_forgery(), _tail_gov(), SimulationConfig(seed=7))
    stack = GovernanceStack(
        [StakingLever(min_stake=1.0, slash_rate=0.5, tau=0.4), TailCircuitBreaker(tau=0.4)],
        initial_resources=5.0,
    )
    defended = run_attack(
        AttackLibrary.volume_forgery(), _tail_gov(), SimulationConfig(seed=7), levers=stack
    )
    assert base.evasion_rate > 0.4
    assert defended.evasion_rate < 0.15
    assert defended.blocked_rate > 0.5
    assert defended.realized_damage < base.realized_damage / 5
