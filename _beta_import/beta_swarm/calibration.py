"""Calibration: is the belief distribution an honest forecast of the outcome?

The main SWARM framework's calibration arm checks proxy *fidelity*: sample a
latent outcome from a known generative model, run the proxy, and compare
predicted probability to realized frequency in a reliability diagram (ECE /
MCE / Brier). A distribution over a continuous outcome generalizes each piece:

- **Reliability diagram → PIT histogram.** For a calibrated belief ``F``, the
  probability integral transform ``F(v_true)`` is uniform on ``[0, 1]``. The
  PIT histogram is the whole-distribution reliability diagram, and its shape
  diagnoses the failure: a U marks overconfidence (truth keeps landing in the
  tails), a hump marks underconfidence, a slope marks mean bias.
- **Reliability of the lever → tail-mass bins.** Governance triggers on
  ``P(v < tau)``, so that number must mean what it says: bin interactions by
  predicted tail mass and compare to the empirical bad-outcome rate. ECE and
  MCE carry over verbatim as size-weighted and worst-bin gaps.
- **Brier → CRPS.** The proper score for a full predictive distribution.
- **Sharpness.** Calibration alone rewards vagueness; the goal is maximal
  sharpness (concentration) *subject to* calibration, so reports carry both.

The fidelity harness reuses the simulation archetypes as the conditional
generative model — an agent's emission model plays the role the main swarm's
``sample_observables`` played. The tunable it sweeps is ``evidence_scale``,
the proxy's concentration knob: the parameter the scalar formalism did not
have, and therefore never had to calibrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from beta_swarm.agents import Archetype, SimAgent
from beta_swarm.belief import BetaBelief
from beta_swarm.interaction import BetaInteraction
from beta_swarm.proxy import BetaProxyComputer


@dataclass
class BinStats:
    """One tail-mass reliability bin: predicted risk vs. realized bad rate."""

    lo: float
    hi: float
    n: int
    mean_predicted: float  # mean predicted P(v < tau) in the bin
    observed_rate: float   # empirical frequency of v_true < tau


@dataclass
class CalibrationReport:
    """Distributional calibration of a set of beliefs against outcomes."""

    n_total: int
    crps: float
    pit_deviation: float          # total-variation distance of PIT histogram from uniform
    pit_freqs: list[float]        # PIT histogram (fractions per equal-width bin)
    tail_ece: float               # size-weighted |observed - predicted| over tail bins
    tail_mce: float               # worst-bin gap
    tail_bins: list[BinStats] = field(default_factory=list)
    sharpness: float = 0.0        # mean concentration — maximize subject to calibration
    tau: float = 0.4


# ----------------------------------------------------------------------
# Ingredients
# ----------------------------------------------------------------------
def pit_values(beliefs: Sequence[BetaBelief], outcomes: Sequence[float]) -> np.ndarray:
    """``F_i(v_i)`` per interaction — uniform on [0, 1] iff beliefs are calibrated."""
    if len(beliefs) != len(outcomes):
        raise ValueError("beliefs and outcomes must have the same length")
    return np.array([float(b.cdf(v)) for b, v in zip(beliefs, outcomes)])


def crps(beliefs: Sequence[BetaBelief], outcomes: Sequence[float], n_grid: int = 512) -> float:
    """Mean Continuous Ranked Probability Score (lower is better)."""
    if len(beliefs) != len(outcomes):
        raise ValueError("beliefs and outcomes must have the same length")
    if not beliefs:
        return 0.0
    grid = np.linspace(0.0, 1.0, n_grid)
    total = 0.0
    for b, v in zip(beliefs, outcomes):
        total += float(np.trapezoid((b.cdf(grid) - (grid >= v)) ** 2, grid))
    return total / len(beliefs)


def tail_reliability_bins(
    beliefs: Sequence[BetaBelief],
    outcomes: Sequence[float],
    tau: float = 0.4,
    n_bins: int = 10,
) -> list[BinStats]:
    """Bin by predicted tail mass, compare to the empirical bad-outcome rate.

    Empty bins are omitted, mirroring the scalar reliability diagram.
    """
    predicted = np.array([b.tail_mass(tau) for b in beliefs])
    bad = np.array([v < tau for v in outcomes], dtype=float)
    width = 1.0 / n_bins
    bins: list[BinStats] = []
    for i in range(n_bins):
        lo = i * width
        hi = lo + width if i < n_bins - 1 else 1.0 + 1e-9
        mask = (predicted >= lo) & (predicted < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append(
            BinStats(
                lo=lo,
                hi=min(hi, 1.0),
                n=n,
                mean_predicted=float(predicted[mask].mean()),
                observed_rate=float(bad[mask].mean()),
            )
        )
    return bins


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------
def calibrate(
    beliefs: Sequence[BetaBelief],
    outcomes: Sequence[float],
    tau: float = 0.4,
    n_bins: int = 10,
) -> CalibrationReport:
    """Full distributional calibration report for beliefs vs. realized outcomes."""
    if not beliefs:
        raise ValueError("need at least one belief/outcome pair")
    pit = pit_values(beliefs, outcomes)
    freqs, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    pit_freqs = (freqs / len(pit)).tolist()
    pit_deviation = 0.5 * float(np.abs(np.array(pit_freqs) - 1.0 / n_bins).sum())

    bins = tail_reliability_bins(beliefs, outcomes, tau=tau, n_bins=n_bins)
    n_total = len(beliefs)
    tail_ece = sum((b.n / n_total) * abs(b.observed_rate - b.mean_predicted) for b in bins)
    tail_mce = max(
        (abs(b.observed_rate - b.mean_predicted) for b in bins), default=0.0
    )

    return CalibrationReport(
        n_total=n_total,
        crps=crps(beliefs, outcomes),
        pit_deviation=pit_deviation,
        pit_freqs=pit_freqs,
        tail_ece=tail_ece,
        tail_mce=tail_mce,
        tail_bins=bins,
        sharpness=float(np.mean([b.concentration for b in beliefs])),
        tau=tau,
    )


def calibrate_interactions(
    interactions: Sequence[BetaInteraction],
    tau: float = 0.4,
    n_bins: int = 10,
    use_proxy_belief: bool = False,
) -> CalibrationReport | None:
    """Calibrate a simulation log against its ground truth.

    Scores the decision belief by default (that is what governance acted on);
    pass ``use_proxy_belief=True`` to score the raw per-interaction evidence.
    Returns ``None`` if no interaction carries ground truth.
    """
    pairs = [
        ((i.proxy_belief if use_proxy_belief and i.proxy_belief is not None else i.belief),
         i.ground_truth)
        for i in interactions
        if i.ground_truth is not None
    ]
    if not pairs:
        return None
    beliefs, outcomes = zip(*pairs)
    return calibrate(list(beliefs), list(outcomes), tau=tau, n_bins=n_bins)


# ----------------------------------------------------------------------
# Fidelity harness (arm A): known generative model -> proxy -> report
# ----------------------------------------------------------------------
def fidelity_run(
    n: int,
    archetypes: Sequence[Archetype] = (Archetype.HONEST, Archetype.MEDIOCRE),
    proxy: BetaProxyComputer | None = None,
    seed: int = 0,
) -> tuple[list[BetaBelief], list[float]]:
    """Draw ``n`` synthetic interactions and run the proxy on each.

    The archetype emission models are the conditional generative model: each
    draw picks an archetype uniformly, samples its true outcome, emits
    observables, and computes the proxy belief. Faithful archetypes measure
    proxy fidelity; including ``DECEPTIVE`` measures how calibration degrades
    under adversarial emission.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    proxy = proxy or BetaProxyComputer()
    agents = [SimAgent.make(a.value, a) for a in archetypes]
    beliefs: list[BetaBelief] = []
    outcomes: list[float] = []
    for _ in range(n):
        agent = agents[rng.integers(len(agents))]
        v_true = agent.sample_outcome(rng)
        beliefs.append(proxy.compute_belief(agent.emit_observables(v_true, rng)))
        outcomes.append(v_true)
    return beliefs, outcomes


def sweep_evidence_scale(
    scales: Sequence[float],
    n: int = 2000,
    archetypes: Sequence[Archetype] = (Archetype.HONEST, Archetype.MEDIOCRE),
    seed: int = 0,
    tau: float = 0.4,
    n_bins: int = 10,
) -> list[tuple[float, CalibrationReport]]:
    """Fidelity report per ``evidence_scale`` — the concentration knob.

    Every scale reuses the same seed, so the sampled (archetype, outcome,
    observables) draws are identical across reports and the comparison is
    paired: only the proxy's belief-sharpening differs. Pick the sharpest
    scale that stays calibrated (low PIT deviation / tail ECE).
    """
    reports = []
    for scale in scales:
        proxy = BetaProxyComputer(evidence_scale=scale)
        beliefs, outcomes = fidelity_run(n, archetypes=archetypes, proxy=proxy, seed=seed)
        reports.append((scale, calibrate(beliefs, outcomes, tau=tau, n_bins=n_bins)))
    return reports
