"""Incoherence-triggered circuit breaker lever."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from swarm.env.state import EnvState
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction

if TYPE_CHECKING:
    from swarm.governance.config import GovernanceConfig


@dataclass
class _IncoherenceTracker:
    """Tracks rolling incoherence proxy and freeze state for an agent."""

    freeze_until_epoch: int = -1
    rolling_uncertainty: List[float] = field(default_factory=list)

    def add(self, uncertainty: float, window_size: int = 10) -> None:
        self.rolling_uncertainty.append(uncertainty)
        if len(self.rolling_uncertainty) > window_size:
            self.rolling_uncertainty.pop(0)

    def avg_uncertainty(self) -> float:
        if not self.rolling_uncertainty:
            return 0.0
        return sum(self.rolling_uncertainty) / len(self.rolling_uncertainty)


class IncoherenceCircuitBreakerLever(GovernanceLever):
    """
    Freeze agents that exhibit persistent high-uncertainty behavior.

    Uses uncertainty proxy `1 - abs(2p - 1)`, which peaks near p=0.5.
    """

    def __init__(self, config: GovernanceConfig) -> None:
        super().__init__(config)
        self._trackers: Dict[str, _IncoherenceTracker] = {}

    reduces_variance = True

    @property
    def name(self) -> str:
        return "incoherence_breaker"

    def _tracker(self, agent_id: str) -> _IncoherenceTracker:
        if agent_id not in self._trackers:
            self._trackers[agent_id] = _IncoherenceTracker()
        return self._trackers[agent_id]

    def on_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> LeverEffect:
        if not self.config.incoherence_breaker_enabled:
            return LeverEffect(lever_name=self.name)

        agents_to_unfreeze = set()
        for agent_id, tracker in self._trackers.items():
            if tracker.freeze_until_epoch >= 0 and epoch >= tracker.freeze_until_epoch:
                tracker.freeze_until_epoch = -1
                agents_to_unfreeze.add(agent_id)

        return LeverEffect(
            lever_name=self.name,
            agents_to_unfreeze=agents_to_unfreeze,
            details={"unfrozen_agents": list(agents_to_unfreeze)},
        )

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        if not self.config.incoherence_breaker_enabled:
            return LeverEffect(lever_name=self.name)

        agent_id = interaction.initiator
        tracker = self._tracker(agent_id)
        uncertainty = 1.0 - abs(2 * interaction.p - 1.0)
        tracker.add(uncertainty)

        avg_uncertainty = tracker.avg_uncertainty()
        agents_to_freeze = set()
        if avg_uncertainty >= self.config.incoherence_breaker_threshold:
            tracker.freeze_until_epoch = (
                state.current_epoch + self.config.freeze_duration_epochs
            )
            agents_to_freeze.add(agent_id)

        return LeverEffect(
            lever_name=self.name,
            agents_to_freeze=agents_to_freeze,
            details={
                "agent_id": agent_id,
                "uncertainty": uncertainty,
                "avg_uncertainty": avg_uncertainty,
                "triggered": bool(agents_to_freeze),
            },
        )
