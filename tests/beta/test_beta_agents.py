"""Tests for the simulation agent archetypes."""

import numpy as np

from beta_swarm.agents import Archetype, SimAgent, make_population
from beta_swarm.proxy import BetaProxyComputer


def test_make_population_counts_and_ids():
    pop = make_population({Archetype.HONEST: 2, Archetype.DECEPTIVE: 3})
    assert len(pop) == 5
    ids = [a.agent_id for a in pop]
    assert len(set(ids)) == 5
    assert sum(a.archetype is Archetype.DECEPTIVE for a in pop) == 3
    assert "honest-0" in ids and "deceptive-2" in ids


def test_true_quality_ordering():
    """Honest agents do better work than mediocre, mediocre than deceptive."""
    rng = np.random.default_rng(0)
    means = {
        arch: float(
            np.mean([SimAgent.make("a", arch).sample_outcome(rng) for _ in range(500)])
        )
        for arch in Archetype
    }
    assert means[Archetype.HONEST] > means[Archetype.MEDIOCRE] > means[Archetype.DECEPTIVE]


def test_faithful_emission_tracks_outcome():
    """Honest observables report bad outcomes: low v means rework and rejections."""
    rng = np.random.default_rng(1)
    agent = SimAgent.make("h", Archetype.HONEST)
    bad = [agent.emit_observables(0.1, rng) for _ in range(200)]
    good = [agent.emit_observables(0.9, rng) for _ in range(200)]
    assert np.mean([o.rework_count for o in bad]) > np.mean([o.rework_count for o in good])
    assert np.mean([o.task_progress_delta for o in bad]) < 0
    assert np.mean([o.task_progress_delta for o in good]) > 0


def test_deceptive_fingerprint_is_high_mean_low_concentration():
    """Deception inflates the belief mean but cannot fake evidence volume."""
    rng = np.random.default_rng(2)
    proxy = BetaProxyComputer()
    agent = SimAgent.make("d", Archetype.DECEPTIVE)
    beliefs = [
        proxy.compute_belief(agent.emit_observables(agent.sample_outcome(rng), rng))
        for _ in range(200)
    ]
    # The mean looks great despite a true quality mean of 0.2...
    assert np.mean([b.mean for b in beliefs]) > 0.6
    # ...but the belief stays diffuse: suppressed events leave little evidence.
    assert np.mean([b.concentration for b in beliefs]) < 5.0
