"""Proxy computer for v_hat computation from downstream observables."""

import logging
from typing import Any, Optional

from pydantic import BaseModel, field_validator

from swarm.core.sigmoid import _sigmoid_fast

logger = logging.getLogger(__name__)


class ProxyWeights(BaseModel):
    """
    Weights for combining downstream proxy signals into v_hat.

    Default values:
    - task_progress: 0.4 (primary signal — the only directly measured outcome)
    - rework_penalty: 0.2 (quality signal — indirect, from rework cycles)
    - verifier_penalty: 0.2 (safety signal — indirect, averages verifier rejections and tool misuse)
    - engagement_signal: 0.2 (social signal — indirect, from counterparty response)

    Design rationale: task_progress receives double weight (0.4 vs 0.2) because
    it is the sole direct outcome measure. The three indirect channels share the
    remaining weight equally (0.2 each) to avoid privileging one failure-detection
    mode over another without domain-specific calibration data.

    All weights are normalized to sum to 1.0 by ProxyComputer.__init__.
    All weights must be non-negative.

    These weights propagate into every downstream metric (toxicity, quality gap,
    payoffs, governance decisions) via the v_hat -> p pipeline. Document any
    non-default weights when reporting results.
    """

    task_progress: float = 0.4
    rework_penalty: float = 0.2
    verifier_penalty: float = 0.2
    engagement_signal: float = 0.2

    model_config = {"frozen": False}

    @field_validator("task_progress", "rework_penalty", "verifier_penalty", "engagement_signal")
    @classmethod
    def validate_non_negative(cls, v: float, info) -> float:
        """Validate that weights are non-negative."""
        if v < 0:
            raise ValueError(
                f"{info.field_name} must be non-negative, got {v}. "
                "Negative weights can lead to unintended behavior in proxy computation."
            )
        return v

    def normalize(self) -> "ProxyWeights":
        """Return normalized weights that sum to 1."""
        total = (
            self.task_progress
            + self.rework_penalty
            + self.verifier_penalty
            + self.engagement_signal
        )

        if total == 0:
            return ProxyWeights(
                task_progress=0.25,
                rework_penalty=0.25,
                verifier_penalty=0.25,
                engagement_signal=0.25,
            )

        return ProxyWeights(
            task_progress=self.task_progress / total,
            rework_penalty=self.rework_penalty / total,
            verifier_penalty=self.verifier_penalty / total,
            engagement_signal=self.engagement_signal / total,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = self.model_dump()
        return result


class ProxyObservables(BaseModel):
    """
    Raw observable signals from an interaction.

    These are transformed into the v_hat proxy score.
    """

    task_progress_delta: float = 0.0  # [-1, +1] normalized progress
    rework_count: int = 0  # number of rework cycles
    verifier_rejections: int = 0  # number of verifier rejections
    tool_misuse_flags: int = 0  # number of tool misuse flags
    counterparty_engagement_delta: float = 0.0  # [-1, +1] engagement change

    model_config = {"frozen": False}

    @classmethod
    def from_interaction(cls, interaction: Any) -> "ProxyObservables":
        """Extract observables from a SoftInteraction."""
        return cls(
            task_progress_delta=interaction.task_progress_delta,
            rework_count=interaction.rework_count,
            verifier_rejections=interaction.verifier_rejections,
            tool_misuse_flags=interaction.tool_misuse_flags,
            counterparty_engagement_delta=interaction.counterparty_engagement_delta,
        )


class ProxyComputer:
    """
    Computes v_hat proxy scores from downstream observables.

    v_hat is a weighted combination of normalized signals, resulting
    in a score in [-1, +1]. This is then converted to p via calibrated sigmoid.
    """

    def __init__(
        self,
        weights: Optional[ProxyWeights] = None,
        sigmoid_k: float = 2.0,
        rework_decay: float = 0.3,
        rejection_decay: float = 0.4,
        misuse_decay: float = 0.5,
    ):
        """
        Initialize proxy computer.

        Args:
            weights: Weights for combining signals (default: ProxyWeights())
            sigmoid_k: Calibration sharpness for p computation (must be > 0 and <= 100)
            rework_decay: Decay factor per rework cycle (must be in (0, 1))
            rejection_decay: Decay factor per verifier rejection (must be in (0, 1))
            misuse_decay: Decay factor per tool misuse flag (must be in (0, 1))

        Raises:
            ValueError: If any parameter is out of valid range
        """
        # Validate sigmoid_k
        if sigmoid_k <= 0:
            raise ValueError(f"sigmoid_k must be positive, got {sigmoid_k}")
        if sigmoid_k > 100:
            raise ValueError(
                f"sigmoid_k is extremely large ({sigmoid_k}), which may indicate a bug. "
                "Values above 100 are rejected."
            )

        # Validate decay parameters
        for param_name, param_value in [
            ("rework_decay", rework_decay),
            ("rejection_decay", rejection_decay),
            ("misuse_decay", misuse_decay),
        ]:
            if param_value <= 0:
                raise ValueError(
                    f"{param_name} must be positive, got {param_value}. "
                    "Decay values <= 0 would cause incorrect signal computation."
                )
            if param_value >= 1:
                raise ValueError(
                    f"{param_name} must be less than 1, got {param_value}. "
                    "Decay values >= 1 would cause signals to increase with penalties."
                )
            if param_value < 0.01:
                raise ValueError(
                    f"{param_name} is extremely small ({param_value}), which may indicate a bug. "
                    "Values below 0.01 cause signals to decay too rapidly."
                )

        self.weights = (weights or ProxyWeights()).normalize()
        self.sigmoid_k = sigmoid_k
        self.rework_decay = rework_decay
        self.rejection_decay = rejection_decay
        self.misuse_decay = misuse_decay

    def _normalize_progress(self, delta: float) -> float:
        """Normalize progress delta to [-1, +1]."""
        return max(-1.0, min(1.0, delta))

    def _decay_signal(self, count: int, decay: float) -> float:
        """
        Convert a penalty count (rework, rejection, misuse) to a signal
        in [-1, +1] via exponential decay: signal = 2 * decay^count - 1.

        Zero penalties give +1; each additional penalty decays the signal
        towards -1. Sharper penalties use a smaller decay factor.
        """
        if count == 0:
            return 1.0

        return 2.0 * decay**count - 1.0

    def _normalize_engagement(self, delta: float) -> float:
        """Normalize engagement delta to [-1, +1]."""
        return max(-1.0, min(1.0, delta))

    def compute_v_hat(self, observables: ProxyObservables) -> float:
        """
        Compute v_hat from downstream observables.

        Args:
            observables: Raw observable signals

        Returns:
            v_hat: Proxy score in [-1, +1]
        """
        # Compute individual signals
        progress_signal = self._normalize_progress(observables.task_progress_delta)
        rework_signal = self._decay_signal(observables.rework_count, self.rework_decay)
        rejection_signal = self._decay_signal(
            observables.verifier_rejections, self.rejection_decay
        )
        misuse_signal = self._decay_signal(
            observables.tool_misuse_flags, self.misuse_decay
        )
        engagement_signal = self._normalize_engagement(
            observables.counterparty_engagement_delta
        )

        # Combine rejection and misuse into a single "verifier" signal
        verifier_signal = (rejection_signal + misuse_signal) / 2.0

        # Weighted combination
        v_hat = (
            self.weights.task_progress * progress_signal
            + self.weights.rework_penalty * rework_signal
            + self.weights.verifier_penalty * verifier_signal
            + self.weights.engagement_signal * engagement_signal
        )

        # Clamp to [-1, +1] and warn if clamping occurs
        original_v_hat = v_hat
        v_hat = max(-1.0, min(1.0, v_hat))
        if v_hat != original_v_hat:
            logger.warning(
                "v_hat clamped from %.4f to %.4f in compute_v_hat. "
                "This may indicate incorrect weight normalization or signal computation. "
                "Observables: progress=%.2f, rework=%d, rejections=%d, misuse=%d, engagement=%.2f",
                original_v_hat,
                v_hat,
                observables.task_progress_delta,
                observables.rework_count,
                observables.verifier_rejections,
                observables.tool_misuse_flags,
                observables.counterparty_engagement_delta,
            )

        return v_hat

    def compute_p(self, v_hat: float) -> float:
        """
        Convert v_hat to probability p via calibrated sigmoid.

        Uses the fast-path sigmoid since sigmoid_k is validated at __init__
        time and v_hat arrives here already clamped by compute_v_hat().

        Args:
            v_hat: Proxy score in [-1, +1]

        Returns:
            p: P(v = +1) in [0, 1]
        """
        result: float = _sigmoid_fast(v_hat, self.sigmoid_k)
        return result

    def compute_labels(self, observables: ProxyObservables) -> tuple[float, float]:
        """
        Compute both v_hat and p from observables.

        Args:
            observables: Raw observable signals

        Returns:
            (v_hat, p): Proxy score and probability
        """
        v_hat = self.compute_v_hat(observables)
        p = self.compute_p(v_hat)
        return v_hat, p
