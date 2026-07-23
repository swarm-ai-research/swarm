"""Arm A — proxy fidelity.

Measures whether ProxyComputer's `p_hat = sigmoid(k * v_hat)` is a faithful
estimator of the latent outcome probability.

Method:
  1. Sample a latent outcome v in {-1, +1} from a known Bernoulli with
     probability p_true of v=+1.
  2. Sample observables (task_progress_delta, rework_count, ...) from a
     generative model conditioned on v. Distributions for v=+1 and v=-1
     overlap so the proxy is informative but not perfect.
  3. Run ProxyComputer on observables to get v_hat and p_hat.
  4. Bin p_hat into equal-width bins, compare bin-mean p_hat ("confidence")
     to bin-mean realized v=+1 rate ("accuracy"). The reliability diagram.

A well-calibrated proxy has accuracy == confidence in every bin. ECE is the
size-weighted L1 gap; MCE is the worst bin gap; Brier is the mean squared
error of p_hat vs the realized outcome.

Pre-registration: docs/research/calibration-prereg.md (arm A).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from swarm.core.proxy import ProxyComputer, ProxyObservables


def sample_observables(p_true: float, rng: random.Random) -> tuple[int, ProxyObservables]:
    """Draw a latent outcome and a conditional observable bundle.

    Returns (v_bit, observables) where v_bit is 1 if v=+1, else 0.

    Distributions are tuned so v=+1 looks like benign work (high progress,
    low rework/rejection/misuse, positive engagement) and v=-1 looks like
    exploitative work (low/negative progress, multiple reworks and
    rejections, occasional tool misuse, negative engagement). The two
    distributions overlap in the middle, which is the regime where
    proxy fidelity actually matters.
    """
    v_bit = 1 if rng.random() < p_true else 0

    if v_bit == 1:
        progress = rng.uniform(0.3, 0.95)
        rework = min(_poisson(0.3, rng), 5)
        rejections = min(_poisson(0.1, rng), 3)
        misuse = min(_poisson(0.05, rng), 2)
        engagement = rng.uniform(0.1, 0.7)
    else:
        progress = rng.uniform(-0.5, 0.3)
        rework = min(_poisson(2.0, rng), 5)
        rejections = min(_poisson(1.0, rng), 3)
        misuse = min(_poisson(0.4, rng), 2)
        engagement = rng.uniform(-0.6, 0.1)

    observables = ProxyObservables(
        task_progress_delta=progress,
        rework_count=rework,
        verifier_rejections=rejections,
        tool_misuse_flags=misuse,
        counterparty_engagement_delta=engagement,
    )
    return v_bit, observables


def _poisson(lam: float, rng: random.Random) -> int:
    """Knuth's algorithm — adequate for small lam used here."""
    import math

    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


@dataclass
class BinStats:
    """Statistics for one reliability-diagram bin."""

    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float


@dataclass
class FidelityReport:
    """Full proxy-fidelity report for one (k, p_grid) configuration."""

    sigmoid_k: float
    n_total: int
    ece: float
    mce: float
    brier: float
    bins: list[BinStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sigmoid_k": self.sigmoid_k,
            "n_total": self.n_total,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "bins": [
                {
                    "lo": b.lo,
                    "hi": b.hi,
                    "n": b.n,
                    "mean_confidence": b.mean_confidence,
                    "accuracy": b.accuracy,
                }
                for b in self.bins
            ],
        }


def reliability_bins(
    p_hats: list[float], outcomes: list[int], n_bins: int = 10
) -> list[BinStats]:
    """Equal-width binning of p_hat in [0, 1].

    Empty bins are omitted from the returned list.
    """
    if len(p_hats) != len(outcomes):
        raise ValueError("p_hats and outcomes must have the same length")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    width = 1.0 / n_bins
    bins: list[BinStats] = []
    for i in range(n_bins):
        lo = i * width
        hi = lo + width if i < n_bins - 1 else 1.0 + 1e-9
        indices = [j for j, p in enumerate(p_hats) if lo <= p < hi]
        if not indices:
            continue
        n = len(indices)
        mean_conf = sum(p_hats[j] for j in indices) / n
        accuracy = sum(outcomes[j] for j in indices) / n
        bins.append(BinStats(lo=lo, hi=min(hi, 1.0), n=n, mean_confidence=mean_conf, accuracy=accuracy))
    return bins


def expected_calibration_error(bins: Iterable[BinStats], n_total: int) -> float:
    """Size-weighted mean of |accuracy - confidence| across bins."""
    if n_total <= 0:
        return 0.0
    return sum((b.n / n_total) * abs(b.accuracy - b.mean_confidence) for b in bins)


def maximum_calibration_error(bins: Iterable[BinStats]) -> float:
    """Worst-bin |accuracy - confidence|."""
    gaps = [abs(b.accuracy - b.mean_confidence) for b in bins]
    return max(gaps) if gaps else 0.0


def brier_score(p_hats: list[float], outcomes: list[int]) -> float:
    """Mean squared error of p_hat vs the realized binary outcome."""
    if len(p_hats) != len(outcomes):
        raise ValueError(
            f"length mismatch: len(p_hats)={len(p_hats)}, len(outcomes)={len(outcomes)}"
        )
    if not p_hats:
        return 0.0
    return sum((p - o) ** 2 for p, o in zip(p_hats, outcomes, strict=True)) / len(p_hats)


def run_fidelity(
    sigmoid_k: float,
    p_grid: list[float],
    per_bin: int,
    seed: int,
    n_bins: int = 10,
) -> FidelityReport:
    """Run one (sigmoid_k) fidelity experiment.

    For each p_true in p_grid, draw `per_bin` synthetic interactions, run
    ProxyComputer, and aggregate across the grid into a single calibration
    report.
    """
    rng = random.Random(seed)
    proxy = ProxyComputer(sigmoid_k=sigmoid_k)

    p_hats: list[float] = []
    outcomes: list[int] = []
    for p_true in p_grid:
        for _ in range(per_bin):
            v_bit, obs = sample_observables(p_true, rng)
            _, p_hat = proxy.compute_labels(obs)
            p_hats.append(p_hat)
            outcomes.append(v_bit)

    bins = reliability_bins(p_hats, outcomes, n_bins=n_bins)
    n_total = len(p_hats)
    return FidelityReport(
        sigmoid_k=sigmoid_k,
        n_total=n_total,
        ece=expected_calibration_error(bins, n_total),
        mce=maximum_calibration_error(bins),
        brier=brier_score(p_hats, outcomes),
        bins=bins,
    )


def sweep_sigmoid_k(
    k_values: list[float],
    p_grid: list[float],
    per_bin: int,
    seed: int,
    n_bins: int = 10,
) -> list[FidelityReport]:
    """Run `run_fidelity` for each k. Returns one report per k.

    Each k uses the same seed (and therefore the same sampled observables)
    so reports form a paired comparison — the only thing varying across
    reports is the sigmoid_k applied by ProxyComputer.
    """
    return [run_fidelity(k, p_grid, per_bin, seed=seed, n_bins=n_bins) for k in k_values]


def report_from_pairs(
    p_hats: list[float],
    outcomes: list[int],
    sigmoid_k: float,
    n_bins: int = 10,
) -> FidelityReport:
    """Build a FidelityReport from already-measured (p_hat, outcome) pairs.

    Used by the controlled-injection scenario path (beads c89o), where the
    pairs come from a real scenario run's export instead of the synthetic
    generator: outcomes are drawn from the target-derived latent
    probability inside the simulation, and p_hat is whatever the pipeline
    actually computed and logged.
    """
    bins = reliability_bins(p_hats, outcomes, n_bins=n_bins)
    n_total = len(p_hats)
    return FidelityReport(
        sigmoid_k=sigmoid_k,
        n_total=n_total,
        ece=expected_calibration_error(bins, n_total),
        mce=maximum_calibration_error(bins),
        brier=brier_score(p_hats, outcomes),
        bins=bins,
    )
