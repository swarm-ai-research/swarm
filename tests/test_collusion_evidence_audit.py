"""Invariants and controls for experiments/collusion_evidence_audit.py (bead 50bb)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import collusion_evidence_audit as cea  # noqa: E402


def _records(n_agents=12, n_days=10, per_day=60, seed=0):
    """Honest chat: each day a random subset is present and addresses each
    other uniformly."""
    rng = np.random.default_rng(seed)
    agents = [f"a{i}" for i in range(n_agents)]
    out = []
    for d in range(n_days):
        present = rng.choice(n_agents, size=6, replace=False)
        for m in range(per_day):
            s, t = rng.choice(present, size=2, replace=False)
            out.append((f"2026-01-{d + 1:02d}T00:{m // 60:02d}:{m % 60:02d}",
                        agents[s], agents[t]))
    return out


@pytest.fixture
def w():
    return cea.Window(_records())


def _per_day_received(w, dst):
    return sorted(zip(w.day.tolist(), dst.tolist(), strict=True))


def test_day_perm_preserves_per_day_received_counts(w):
    """Sent counts are preserved trivially (src is never touched)."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        dst = cea.null_day_perm(w, rng)
        assert _per_day_received(w, dst) == _per_day_received(w, w.dst)
        assert not np.any(dst == w.src), "no self-addressing"


def test_copresent_only_addresses_agents_present_that_day(w):
    dst = cea.null_copresent(w, np.random.default_rng(2))
    for d in range(w.n_days):
        lo, hi = w.bounds[d], w.bounds[d + 1]
        assert set(dst[lo:hi].tolist()) <= set(w.present[d].tolist())
    assert not np.any(dst == w.src)


def test_uniform_never_self_addresses(w):
    assert not np.any(cea.null_uniform(w, np.random.default_rng(3)) == w.src)


def test_positive_control_clears_day_perm(w):
    """A planted coalition must clear the schedule-preserving null --
    a null that reports clean negatives without this is not evidence."""
    grp = [0, 1, 2, 3]
    wp = cea.plant(w, grp, 0.6, np.random.default_rng(4))
    member = np.zeros(w.n_agents, bool)
    member[grp] = True
    _, ratio, _, p = cea.enrichment(wp, member, cea.null_samples(wp, "day_perm", 200, 0))
    assert p < 0.05 and ratio > 1.3


def test_honest_group_is_not_enriched_under_day_perm(w):
    member = np.zeros(w.n_agents, bool)
    member[[0, 1, 2, 3]] = True
    _, ratio, _, p = cea.enrichment(w, member, cea.null_samples(w, "day_perm", 200, 0))
    assert p > 0.05 and 0.7 < ratio < 1.3


def test_uniform_null_mistakes_co_presence_for_coordination():
    """The confound the schedule-preserving null exists to remove: agents
    present on the same days look enriched against a null that ignores
    who was present."""
    rng = np.random.default_rng(5)
    recs = []
    for d in range(10):
        # era A (a0-a5) for days 0-4, era B (a6-a11) for days 5-9
        era = range(0, 6) if d < 5 else range(6, 12)
        for m in range(60):
            s, t = rng.choice(list(era), size=2, replace=False)
            recs.append((f"2026-01-{d + 1:02d}T00:{m // 60:02d}:{m % 60:02d}",
                         f"a{s}", f"a{t}"))
    w = cea.Window(recs)
    member = np.zeros(w.n_agents, bool)
    member[[w.index[f"a{i}"] for i in range(6)]] = True
    _, _, _, p_uniform = cea.enrichment(w, member, cea.null_samples(w, "uniform", 200, 0))
    _, ratio, _, p_day = cea.enrichment(w, member, cea.null_samples(w, "day_perm", 200, 0))
    assert p_uniform < 0.05
    assert p_day > 0.05 and ratio == pytest.approx(1.0, abs=0.05)


def test_plant_keeps_speakers_and_volume(w):
    wp = cea.plant(w, [0, 1, 2], 0.5, np.random.default_rng(6))
    assert np.array_equal(wp.src, w.src) and len(wp.dst) == len(w.dst)
    assert cea.plant(w, [0, 1, 2], 0.0, np.random.default_rng(6)) is w
