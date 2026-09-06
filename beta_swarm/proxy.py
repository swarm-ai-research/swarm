"""Build a Beta belief from observable signals.

The original ``ProxyComputer`` produced a single ``p`` from a weighted blend of
observables passed through a sigmoid. Here we keep that pipeline to set the
belief **mean**, and separately estimate the **concentration** from how much
evidence the observables represent. More observed signal (more rework cycles,
more verifier passes, stronger engagement) means a sharper belief; a sparse or
conflicting signal means a diffuse one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beta_swarm.belief import BetaBelief


def _sigmoid(x: float, k: float) -> float:
    return 1.0 / (1.0 + np.exp(-k * x))


@dataclass
class ProxyObservables:
    """Raw observable signals from an interaction (mirrors the scalar model)."""

    task_progress_delta: float = 0.0       # [-1, +1]
    rework_count: int = 0
    verifier_rejections: int = 0
    tool_misuse_flags: int = 0
    counterparty_engagement_delta: float = 0.0  # [-1, +1]


@dataclass
class BetaProxyComputer:
    """Maps observables to a :class:`BetaBelief`.

    Parameters
    ----------
    w_progress, w_rework, w_verifier, w_engagement:
        Non-negative weights blended into ``v_hat in [-1, +1]`` (normalized).
    sigmoid_k:
        Sharpness of the ``v_hat -> mean`` calibration.
    base_concentration:
        Prior pseudo-count when no evidence is present (the diffuse floor).
    evidence_scale:
        How strongly each observed signal sharpens the belief.
    max_concentration:
        Ceiling on the concentration a single interaction may assert. ``None``
        (default) leaves it uncapped — the configuration a volume-forgery attack
        exploits, fabricating event count to claim near-arbitrary sharpness.
        Setting a cap means high confidence must be *earned* through audits and
        reputation pooling, not claimed in one interaction.
    """

    w_progress: float = 0.4
    w_rework: float = 0.2
    w_verifier: float = 0.2
    w_engagement: float = 0.2
    sigmoid_k: float = 2.0
    base_concentration: float = 2.0
    evidence_scale: float = 1.5
    max_concentration: float | None = None

    def __post_init__(self) -> None:
        weights = np.array(
            [self.w_progress, self.w_rework, self.w_verifier, self.w_engagement]
        )
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if self.sigmoid_k <= 0:
            raise ValueError("sigmoid_k must be positive")
        if self.base_concentration <= 0 or self.evidence_scale < 0:
            raise ValueError("base_concentration > 0 and evidence_scale >= 0 required")
        total = weights.sum()
        self._w = weights / total if total > 0 else np.full(4, 0.25)

    @staticmethod
    def _count_signal(count: int, decay: float = 0.4) -> float:
        """Map a penalty count to ``[-1, +1]``: 0 -> +1, growing count -> -1."""
        return 1.0 if count == 0 else 2.0 * (decay ** count) - 1.0

    def compute_v_hat(self, obs: ProxyObservables) -> float:
        progress = float(np.clip(obs.task_progress_delta, -1.0, 1.0))
        rework = self._count_signal(obs.rework_count)
        rejection = self._count_signal(obs.verifier_rejections)
        misuse = self._count_signal(obs.tool_misuse_flags)
        verifier = 0.5 * (rejection + misuse)
        engagement = float(np.clip(obs.counterparty_engagement_delta, -1.0, 1.0))
        v_hat = (
            self._w[0] * progress
            + self._w[1] * rework
            + self._w[2] * verifier
            + self._w[3] * engagement
        )
        return float(np.clip(v_hat, -1.0, 1.0))

    def compute_concentration(self, obs: ProxyObservables) -> float:
        """Concentration grows with the amount of observed evidence.

        We count every signal the interaction actually exercised — a nonzero
        progress delta, each rework cycle, each verifier event, engagement — as
        evidence. A bare interaction with no signal stays at ``base_concentration``
        (a diffuse, "we don't know yet" belief).
        """
        evidence = (
            abs(obs.task_progress_delta)
            + obs.rework_count
            + obs.verifier_rejections
            + obs.tool_misuse_flags
            + abs(obs.counterparty_engagement_delta)
        )
        conc = self.base_concentration + self.evidence_scale * evidence
        if self.max_concentration is not None:
            conc = min(conc, self.max_concentration)
        return conc

    def compute_belief(self, obs: ProxyObservables) -> BetaBelief:
        """Full pipeline: observables -> (mean via sigmoid, concentration via evidence)."""
        v_hat = self.compute_v_hat(obs)
        mean = _sigmoid(v_hat, self.sigmoid_k)
        concentration = self.compute_concentration(obs)
        return BetaBelief.from_mean_concentration(mean, concentration)
