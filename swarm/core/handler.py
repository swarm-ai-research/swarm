"""Base class for environment handlers in the orchestration layer.

All handlers share a common pattern:
- Accept an ``event_bus: EventBus`` for append-only event logging
- Declare which ``ActionType`` values they own via ``handled_action_types``
- Dispatch agent actions via ``handle_action``
- Build per-agent observation fields via ``build_observation_fields``

The ``HandlerActionResult`` model provides a unified return type that
the orchestrator uses for proxy computation and interaction finalization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, FrozenSet, Optional, cast

from pydantic import BaseModel, ConfigDict, Field

from swarm.logging.event_bus import EventBus
from swarm.models.agent import AgentType
from swarm.models.artifact import Artifact
from swarm.models.events import Event
from swarm.models.interaction import InteractionType


class HandlerActionResult(BaseModel):
    """Unified result type for all handler actions.

    All handlers return this from ``handle_action()``.  The orchestrator
    uses the common fields for proxy computation and interaction
    finalization.  Handler-specific data goes in ``metadata``.

    When ``observables`` is ``None`` the orchestrator treats the action
    as a simple success/failure (no proxy computation or interaction
    finalization).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    observables: Optional[Any] = None  # ProxyObservables when set
    initiator_id: str = ""
    counterparty_id: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    accepted: bool = True

    # What InteractionType to use for the SoftInteraction
    interaction_type: InteractionType = InteractionType.COLLABORATION

    # Transfer amount (tau) for the SoftInteraction
    tau: float = 0.0

    # Ground truth label (-1 or 1), or None for no ground truth
    ground_truth: Optional[int] = None

    # Artifact layer: typed outputs produced and inputs consumed
    produced_artifacts: list[Artifact] = Field(default_factory=list)
    consumed_artifact_ids: list[str] = Field(default_factory=list)


class Handler(ABC):
    """Abstract base for environment handlers (plugin protocol).

    Subclasses must implement:

    - ``handled_action_types``: declares which ``ActionType`` values this
      handler exclusively owns.
    - ``handle_action``: dispatch a single agent action.

    Optional overrides:

    - ``build_observation_fields``: contribute to per-agent observations.
    - ``observation_field_mapping``: remap handler keys to ``Observation`` fields.
    - ``on_epoch_start`` / ``on_epoch_end``: epoch lifecycle hooks.
    - ``on_step``: per-step tick hook.
    - ``on_pre_observation``: called before observation building (e.g. compaction).
    - ``post_finalize``: handler-specific post-processing after finalization.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
    ) -> None:
        self._event_bus = event_bus
        self._emit_event: Callable[[Event], None] = event_bus.emit

    @staticmethod
    @abstractmethod
    def handled_action_types() -> FrozenSet:
        """Return the set of ``ActionType`` values this handler exclusively owns.

        This is the registry key.  Each ``ActionType`` must map to exactly
        one handler.  The ``HandlerRegistry`` validates no overlaps at
        registration time.
        """
        ...

    @abstractmethod
    def handle_action(self, action: Any, state: Any) -> Any:
        """Dispatch a single agent action and return a result.

        New-style handlers should return ``HandlerActionResult``.
        Legacy handlers may still return domain-specific result types;
        the orchestrator will fall back to the old code path.
        """

    def build_observation_fields(
        self, agent_id: str, state: Any
    ) -> Dict[str, Any]:
        """Return handler-specific observation fields for *agent_id*.

        The default implementation returns an empty dict.  Handlers that
        contribute to agent observations should override this.
        """
        return {}

    def observation_field_mapping(self) -> Dict[str, str]:
        """Map keys from ``build_observation_fields`` to ``Observation`` field names.

        Default: identity mapping (key name == ``Observation`` field name).
        Override when the handler's internal key names differ from the
        ``Observation`` dataclass field names.

        Example::

            {"published_posts": "moltbook_published_posts"}
        """
        return {}

    def on_epoch_start(self, state: Any) -> None:  # noqa: B027
        """Hook called at the start of each epoch.  Override if needed."""

    def on_epoch_end(self, state: Any) -> None:  # noqa: B027
        """Hook called at the end of each epoch.  Override if needed."""

    def on_step(self, state: Any, step: int) -> None:  # noqa: B027
        """Hook called at the start of each step.  Override for per-step logic.

        Use cases: ``MoltbookHandler.tick()``, rate-limit decay.
        """

    def on_pre_observation(self, agent_id: str, state: Any) -> None:  # noqa: B027
        """Hook called per-agent before observation building.

        Use case: ``MemoryHandler.maybe_compaction(agent_id, state)``.
        """

    def post_finalize(  # noqa: B027
        self,
        result: Any,
        interaction: Any,
        gov_effect: Any,
        state: Any,
    ) -> None:
        """Hook called after ``_finalize_interaction`` completes.

        Handler-specific post-processing goes here instead of in the
        orchestrator.  Examples:

        - ``MoltipediaHandler``: record points, emit governance events.
        - ``MemoryHandler``: revert promotion if governance blocked it.
        """

    def _get_agent_type(self, agent_id: str, state: Any) -> AgentType:
        """Look up agent type from state, defaulting to HONEST if not found.

        Args:
            agent_id: The agent ID to look up.
            state: Environment state with get_agent method.

        Returns:
            The agent's AgentType, or AgentType.HONEST if agent not found.
        """
        agent_state = state.get_agent(agent_id)
        if agent_state is None:
            return AgentType.HONEST
        return cast(AgentType, agent_state.agent_type)
