"""Configuration for the A-Evolve evolution-loop domain adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AevolveDomainConfig:
    """Knobs for mapping evolution outcomes into soft labels.

    ``p`` for a candidate is its holdout score when available (the honest
    quality signal), else the benchmark score shrunk toward 0.5 by
    ``train_score_shrinkage`` — a train-split score the gate optimized
    against is partially self-graded and gets less credence.
    """

    train_score_shrinkage: float = 0.5  # 0 = trust train score fully
    sigmoid_k: float = 2.0  # proxy calibration passed to ProxyComputer
