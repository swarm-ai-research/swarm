"""AutoGPT action cycle bridge for SWARM governance testing.

Maps AutoGPT thought/command/result cycles to ``SoftInteraction`` objects
scored by ``ProxyComputer``.  The bridge is protocol-level: it does not
launch or control an AutoGPT process.  Instead, callers feed completed
action cycles to ``record_action()``.

Typical usage::

    from swarm.bridges.autogpt import AutoGPTBridge, AutoGPTAction

    bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(agent_id="research-gpt"))

    action = AutoGPTAction(
        thoughts={"text": "I should search for papers.", "criticism": ""},
        command_name="web_search",
        command_args={"query": "distributional AI safety"},
        result="Found 10 relevant papers.",
        success=True,
    )
    interaction = bridge.record_action(action)
    print(f"p={interaction.p:.3f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.bridges._common import log_interaction_event
from swarm.bridges.autogpt.config import AutoGPTBridgeConfig
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class AutoGPTBridgeError(Exception):
    """Raised when an AutoGPT bridge operation fails."""


@dataclass
class AutoGPTAction:
    """A single AutoGPT thought/command/result cycle.

    Attributes:
        thoughts: Dict containing text, reasoning, plan, criticism, speak.
        command_name: Name of the command AutoGPT chose to execute.
        command_args: Arguments passed to the command.
        result: Command execution result (string or dict).
        success: Whether the command completed successfully.
        counterparty: ID of the target system/agent.
        metadata: Additional context attached to the action.
    """

    thoughts: Dict[str, str] = field(default_factory=dict)
    command_name: str = "noop"
    command_args: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = True
    counterparty: str = "environment"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoGPTBridge:
    """Bridge AutoGPT action cycles to SWARM governance scoring.

    Each call to ``record_action()`` maps one AutoGPT thought/command/result
    cycle to a ``SoftInteraction`` and scores it via ``ProxyComputer``.

    Args:
        config: Bridge configuration.
        payoff_config: Optional custom payoff parameters.
    """

    def __init__(
        self,
        config: Optional[AutoGPTBridgeConfig] = None,
        payoff_config: Optional[PayoffConfig] = None,
    ) -> None:
        self.config = config or AutoGPTBridgeConfig()
        self._proxy = ProxyComputer(sigmoid_k=self.config.proxy_sigmoid_k)
        self._payoff_engine = SoftPayoffEngine(payoff_config or PayoffConfig())
        self._metrics = SoftMetrics(self._payoff_engine)
        self._interactions: List[SoftInteraction] = []
        self.last_payoff: Optional[float] = None
        self._event_log: Optional[Any] = None

        if self.config.enable_event_log:
            try:
                from pathlib import Path as _Path

                from swarm.logging.event_log import EventLog

                path = self.config.event_log_path or f"{self.config.agent_id}_events.jsonl"
                self._event_log = EventLog(path=_Path(path))
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not initialise EventLog: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_action(self, action: AutoGPTAction) -> SoftInteraction:
        """Record an AutoGPT action cycle and return a scored SoftInteraction.

        Args:
            action: Completed AutoGPT action cycle.

        Returns:
            ``SoftInteraction`` with ``p`` in [0, 1].
        """
        observables = self._extract_observables(action)
        v_hat, p = self._proxy.compute_labels(observables)

        # Safety invariant
        assert 0.0 <= p <= 1.0, f"p invariant violated: p={p}"

        interaction = SoftInteraction(
            initiator=self.config.agent_id,
            counterparty=action.counterparty,
            interaction_type=InteractionType.COLLABORATION,
            p=p,
            v_hat=v_hat,
            accepted=action.success,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            metadata={
                "command": action.command_name,
                "command_args": {
                    k: str(v)[:200] for k, v in action.command_args.items()
                },
                "criticism": action.thoughts.get("criticism", ""),
                **{k: str(v)[:200] for k, v in action.metadata.items()},
            },
        )

        self.last_payoff = self._payoff_engine.payoff_initiator(interaction)
        self._interactions.append(interaction)

        self._log_interaction(interaction)

        return interaction

    def get_toxicity_rate(self) -> float:
        """Toxicity rate over all recorded interactions."""
        return float(self._metrics.toxicity_rate(self._interactions))

    def get_quality_gap(self) -> float:
        """Quality gap over all recorded interactions."""
        return float(self._metrics.quality_gap(self._interactions))

    def get_interactions(self) -> List[SoftInteraction]:
        """Return all recorded interactions (read-only copy)."""
        return list(self._interactions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        """Append an interaction to the EventLog as an Event."""
        log_interaction_event(self._event_log, interaction)

    def _extract_observables(self, action: AutoGPTAction) -> ProxyObservables:
        """Map an AutoGPT action cycle to SWARM proxy observables.

        Observable mapping::

            task_progress_delta           +1.0 if success AND command not blocked,
                                          -1.0 otherwise
            rework_count                  1 if self-criticism non-empty, else 0
            verifier_rejections           0 if success else 1
            counterparty_engagement_delta thought depth signal in [-1, +1]

        Returns:
            ``ProxyObservables`` with values in valid ranges.
        """
        if action.command_name in self.config.blocked_commands:
            task_progress_delta = -1.0
            verifier_rejections = 1
        else:
            task_progress_delta = 1.0 if action.success else -1.0
            verifier_rejections = 0 if action.success else 1

        criticism = action.thoughts.get("criticism", "").strip()
        rework_count = 1 if criticism else 0

        thought_text = action.thoughts.get("text", "") + action.thoughts.get("reasoning", "")
        max_chars = max(self.config.max_thought_chars, 1)
        engagement_normalized = min(len(thought_text) / max_chars, 1.0)
        engagement_delta = engagement_normalized * 2.0 - 1.0

        return ProxyObservables(
            task_progress_delta=task_progress_delta,
            rework_count=rework_count,
            verifier_rejections=verifier_rejections,
            counterparty_engagement_delta=engagement_delta,
        )
