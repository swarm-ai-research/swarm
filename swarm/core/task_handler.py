"""Handler for task pool actions (CLAIM_TASK, SUBMIT_OUTPUT).

Extracted from ``Orchestrator._handle_core_action`` to follow the
handler plugin pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler, HandlerActionResult

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.env.tasks import TaskPool
    from swarm.logging.event_bus import EventBus


class TaskHandler(Handler):
    """Handles CLAIM_TASK and SUBMIT_OUTPUT actions against the task pool."""

    def __init__(
        self,
        *,
        task_pool: TaskPool,
        event_bus: EventBus,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._task_pool = task_pool

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return frozenset({ActionType.CLAIM_TASK, ActionType.SUBMIT_OUTPUT})

    def handle_action(self, action: Action, state: EnvState) -> HandlerActionResult:
        agent_id = action.agent_id

        if action.action_type == ActionType.CLAIM_TASK:
            agent_state = state.get_agent(agent_id)
            if not agent_state:
                return HandlerActionResult(success=False)

            rate_limit = state.get_rate_limit_state(agent_id)
            success = self._task_pool.claim_task(
                task_id=action.target_id,
                agent_id=agent_id,
                agent_reputation=agent_state.reputation,
                current_epoch=state.current_epoch,
            )
            if success:
                rate_limit.record_task_claim()
            return HandlerActionResult(success=success)

        elif action.action_type == ActionType.SUBMIT_OUTPUT:
            task = self._task_pool.get_task(action.target_id)
            if task and task.claimed_by == agent_id:
                task.submit_output(agent_id, action.content)
                return HandlerActionResult(success=True)
            return HandlerActionResult(success=False)

        return HandlerActionResult(success=False)
