"""Tests for distributional collusion detection."""

import pytest

from beta_swarm.agents import Archetype, make_population
from beta_swarm.belief import BetaBelief
from beta_swarm.collusion import CollusionDetector
from beta_swarm.governance import TailMassGovernor
from beta_swarm.interaction import BetaInteraction
from beta_swarm.simulation import Simulation, SimulationConfig


def _interaction(initiator: str, counterparty: str, belief: BetaBelief) -> BetaInteraction:
    return BetaInteraction(
        initiator=initiator, counterparty=counterparty, proxy_belief=belief, belief=belief
    )


def _synthetic_log() -> list[BetaInteraction]:
    """Two colluders (a, b) in a five-agent population.

    a and b interact with each other often, with sharp high beliefs; everyone's
    arm's-length traffic carries the same mean but diffuse beliefs — the scalar
    tell (mean asymmetry) is absent by construction.
    """
    sharp = lambda: BetaBelief.from_mean_concentration(0.8, 12.0)
    diffuse = lambda: BetaBelief.from_mean_concentration(0.8, 3.0)
    log = []
    for _ in range(20):
        log.append(_interaction("a", "b", sharp()))
        log.append(_interaction("b", "a", sharp()))
    for initiator in ["a", "b", "c", "d", "e"]:
        for partner in ["c", "d", "e"]:
            if partner == initiator:
                continue
            for _ in range(5):
                log.append(_interaction(initiator, partner, diffuse()))
    return log


def test_flags_ring_pair_without_mean_asymmetry():
    reports = CollusionDetector().flag(_synthetic_log())
    flagged = {(r.initiator, r.counterparty) for r in reports}
    assert ("a", "b") in flagged and ("b", "a") in flagged
    for r in reports:
        # The scalar signal is silent; the shape signals carry the detection.
        assert abs(r.mean_gap) < 0.01
        assert r.concentration_lift > 1.5
        assert r.tail_asymmetry > 0.0


def test_no_false_positives_on_uniform_traffic():
    log = _synthetic_log()
    honest_pairs = [
        (r.initiator, r.counterparty)
        for r in CollusionDetector().flag(log)
        if not {r.initiator, r.counterparty} <= {"a", "b"}
    ]
    assert honest_pairs == []


def test_prefers_proxy_belief_over_pooled():
    """A shared prior pooled into every belief must not wash out the contrast."""
    log = _synthetic_log()
    prior = BetaBelief(30.0, 10.0)
    for i in log:
        # Simulate heavy prior pooling on the decision belief.
        i.belief = BetaBelief(
            i.proxy_belief.alpha + prior.alpha - 1, i.proxy_belief.beta + prior.beta - 1
        )
    reports = CollusionDetector().flag(log)
    assert {("a", "b"), ("b", "a")} <= {(r.initiator, r.counterparty) for r in reports}


def test_min_interactions_gate():
    log = _synthetic_log()
    detector = CollusionDetector(min_interactions=100)
    assert detector.analyze(log) == []


def test_end_to_end_ring_detection_in_simulation():
    """A colluder ring inside the closed loop is caught with no false positives."""
    population = make_population(
        {
            Archetype.HONEST: 4,
            Archetype.MEDIOCRE: 2,
            Archetype.DECEPTIVE: 1,
            Archetype.COLLUDER: 3,
        }
    )
    governor = TailMassGovernor(tau=0.4, max_tail_mass=1.0, min_mean=0.5)
    result = Simulation(population, governor, SimulationConfig(seed=4)).run()

    ring_pairs = {
        (a.agent_id, b.agent_id)
        for a in population
        for b in population
        if a.ring is not None and a.ring == b.ring and a.agent_id != b.agent_id
    }
    flagged = {
        (r.initiator, r.counterparty)
        for r in CollusionDetector().flag(result.interactions)
    }
    assert flagged == ring_pairs  # all six ordered pairs, nothing else


def test_serialization_roundtrip_with_proxy_belief():
    b = BetaBelief.from_mean_concentration(0.7, 5.0)
    i = _interaction("a", "b", b)
    restored = BetaInteraction.from_dict(i.to_dict())
    assert restored.proxy_belief.alpha == pytest.approx(b.alpha)
    assert restored.proxy_belief.beta == pytest.approx(b.beta)
