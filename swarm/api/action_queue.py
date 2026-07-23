"""Per-agent action queue that bridges async API and threaded orchestrator."""

from __future__ import annotations

import asyncio


class AsyncActionQueue:
    """Per-agent action queue that bridges async API and threaded orchestrator.

    Each agent that needs to act gets a Future created via ``wait_for_action``.
    The API layer resolves that Future via ``submit_action`` when the agent's
    HTTP request arrives.  If no request arrives within ``timeout_ms``, the
    Future resolves to ``None`` so the orchestrator can substitute a NOOP.

    Rate-limiting: Each agent is limited to ``_max_actions_per_step`` per step.
    Call ``reset_step()`` at the start of each step to clear per-agent counts.
    """

    def __init__(self, timeout_ms: int = 5000) -> None:
        self._timeout_ms = timeout_ms
        self._pending: dict[str, asyncio.Future[dict | None]] = {}
        self._lock = asyncio.Lock()
        self._action_counts: dict[str, int] = {}
        self._max_actions_per_step: int = 10

    async def wait_for_action(self, agent_id: str) -> dict | None:
        """Create a Future for *agent_id* and wait up to *timeout_ms*.

        Returns the submitted action dict, or ``None`` on timeout.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict | None] = loop.create_future()

        async with self._lock:
            # Cancel any stale future for the same agent.
            old = self._pending.pop(agent_id, None)
            if old is not None and not old.done():
                old.cancel()
            self._pending[agent_id] = future

        try:
            return await asyncio.wait_for(
                future, timeout=self._timeout_ms / 1000.0
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None
        finally:
            async with self._lock:
                # Only remove if it's still *our* future.
                if self._pending.get(agent_id) is future:
                    self._pending.pop(agent_id, None)

    async def submit_action(self, agent_id: str, action: dict) -> bool:
        """Resolve the pending Future for *agent_id* with *action*.

        Returns ``True`` if there was a waiter, ``False`` otherwise.
        Rate-limits each agent to ``_max_actions_per_step`` per step.

        Note: Rate-limit counter is persistent across calls. Call ``reset_step()``
        at the start of each step to clear per-agent counts, otherwise the limit
        will be enforced across step boundaries.
        """
        # Rate-limit check.
        count = self._action_counts.get(agent_id, 0)
        if count >= self._max_actions_per_step:
            return False

        async with self._lock:
            future = self._pending.pop(agent_id, None)

        if future is None or future.done():
            return False

        future.set_result(action)
        self._action_counts[agent_id] = count + 1
        return True

    async def cancel_all(self) -> int:
        """Cancel all pending futures.  Returns count cancelled."""
        async with self._lock:
            pending = dict(self._pending)
            self._pending.clear()

        cancelled = 0
        for future in pending.values():
            if not future.done():
                future.cancel()
                cancelled += 1
        return cancelled

    def reset_step(self) -> None:
        """Clear per-agent action counts (call at the start of each step)."""
        self._action_counts.clear()

    @property
    def pending_count(self) -> int:
        """Number of agents currently being waited on."""
        return len(self._pending)
