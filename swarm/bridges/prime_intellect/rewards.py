"""Reward functions mapping SWARM safety metrics to RL scalar rewards.

These functions are designed to be used as reward/rubric components in
Prime Intellect's ``verifiers`` library.  Each function accepts a list
of :class:`SoftInteraction` objects (produced by one simulation step)
and returns a scalar reward.

Reward semantics
~~~~~~~~~~~~~~~~
* **Higher** reward = safer, more prosocial agent behaviour.
* Toxicity and adverse-selection components are *negated* so that
  reducing them increases reward.
* All rewards are optionally clipped and normalised before being
  returned (controlled via :class:`PrimeIntellectConfig`).

Usage with ``verifiers``::

    from swarm.bridges.prime_intellect.rewards import SwarmRewardComputer

    computer = SwarmRewardComputer(config)
    reward = computer.compute(interactions)        # float
    breakdown = computer.compute_breakdown(interactions)  # dict
"""

import logging
import math
from typing import Dict, List, Optional

from swarm.bridges.prime_intellect.config import (
    PrimeIntellectConfig,
    RewardMode,
)
from swarm.core.payoff import SoftPayoffEngine
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------


def toxicity_reward(interactions: List[SoftInteraction]) -> float:
    """Reward = -(toxicity rate).

    Returns 0.0 when there are no accepted interactions (no signal).
    """
    metrics = SoftMetrics()
    tox = metrics.toxicity_rate(interactions)
    return -tox


def quality_gap_reward(interactions: List[SoftInteraction]) -> float:
    """Reward = quality gap.

    Positive quality gap means the system is successfully filtering
    low-quality interactions.  A *negative* gap signals adverse selection.
    """
    metrics = SoftMetrics()
    return metrics.quality_gap(interactions)


def welfare_reward(interactions: List[SoftInteraction]) -> float:
    """Reward = mean welfare across interactions.

    Uses the default payoff engine to compute per-interaction welfare
    (``π_a + π_b``) and returns the mean.
    """
    if not interactions:
        return 0.0
    engine = SoftPayoffEngine()
    total = 0.0
    for ix in interactions:
        pa = engine.payoff_initiator(ix)
        pb = engine.payoff_counterparty(ix)
        total += pa + pb
    return total / len(interactions)


def adverse_selection_reward(interactions: List[SoftInteraction]) -> float:
    """Reward = -(conditional loss for initiator).

    Conditional loss measures how much *worse* accepted interactions
    are compared to the population average.  Negative conditional loss
    means adverse selection is occurring, so we negate it to penalise.
    """
    metrics = SoftMetrics()
    cond_loss = metrics.conditional_loss_initiator(interactions)
    return -cond_loss


def cooperation_reward(interactions: List[SoftInteraction]) -> float:
    """Reward = mean p across accepted interactions.

    A simple proxy for cooperative, beneficial behaviour.
    """
    accepted = [i for i in interactions if i.accepted]
    if not accepted:
        return 0.0
    return sum(i.p for i in accepted) / len(accepted)


# ---------------------------------------------------------------------------
# Composite reward computer
# ---------------------------------------------------------------------------

# Registry of named component functions
REWARD_COMPONENTS = {
    "toxicity": toxicity_reward,
    "quality_gap": quality_gap_reward,
    "welfare": welfare_reward,
    "adverse_selection": adverse_selection_reward,
    "cooperation": cooperation_reward,
}


class SwarmRewardComputer:
    """Compute scalar RL rewards from SWARM interaction batches.

    Supports three modes via :attr:`RewardMode`:

    * ``toxicity`` — use only the toxicity component.
    * ``quality_gap`` — use only the quality-gap component.
    * ``composite`` — weighted sum of all components.
    * ``welfare`` — use only the welfare component.
    * ``custom`` — caller supplies weights dict; any missing key → 0.
    """

    def __init__(self, config: Optional[PrimeIntellectConfig] = None) -> None:
        self._config = config or PrimeIntellectConfig()
        self._weights = self._config.get_reward_weights()
        self._metrics = SoftMetrics()
        self._clip_min = self._config.reward_clip_min
        self._clip_max = self._config.reward_clip_max
        self._normalize = self._config.reward_normalize

        # Running stats for normalisation (Welford's algorithm)
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    # ----- public interface ------------------------------------------------

    def compute(self, interactions: List[SoftInteraction]) -> float:
        """Return a single scalar reward for an interaction batch."""
        if not interactions:
            return 0.0

        mode = self._config.reward_mode
        if mode == RewardMode.TOXICITY:
            raw = toxicity_reward(interactions)
        elif mode == RewardMode.QUALITY_GAP:
            raw = quality_gap_reward(interactions)
        elif mode == RewardMode.WELFARE:
            raw = welfare_reward(interactions)
        else:
            raw = self._composite(interactions)

        if self._normalize:
            raw = self._update_and_normalize(raw)

        return max(self._clip_min, min(self._clip_max, raw))

    def compute_breakdown(
        self, interactions: List[SoftInteraction]
    ) -> Dict[str, float]:
        """Return per-component rewards plus the final scalar."""
        breakdown: Dict[str, float] = {}
        for name, fn in REWARD_COMPONENTS.items():
            breakdown[name] = fn(interactions)

        breakdown["composite"] = self._composite(interactions)
        breakdown["final"] = self.compute(interactions)
        return breakdown

    # ----- internals -------------------------------------------------------

    def _composite(self, interactions: List[SoftInteraction]) -> float:
        """Weighted sum of all reward components."""
        w = self._weights
        total = 0.0
        total += w.toxicity * toxicity_reward(interactions)
        total += w.quality_gap * quality_gap_reward(interactions)
        total += w.welfare * welfare_reward(interactions)
        total += w.adverse_selection * adverse_selection_reward(interactions)
        total += w.cooperation * cooperation_reward(interactions)
        return total

    def _update_and_normalize(self, value: float) -> float:
        """On-line mean/std normalisation (Welford)."""
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2

        if self._n < 2:
            return value

        variance = self._m2 / (self._n - 1)
        std = math.sqrt(variance) if variance > 0 else 1.0
        return (value - self._mean) / max(std, 1e-8)

    def reset_stats(self) -> None:
        """Reset running normalisation statistics."""
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
