"""In-memory event bus for simulation events.

Thread-safe: ``publish`` / ``publish_sync`` can be called from any
thread and will correctly wake asyncio consumers on the subscriber's
event loop.
"""

import asyncio
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_SUBSCRIPTIONS_PER_AGENT = 2


class SimEventType(str, Enum):
    OBSERVATION_READY = "observation_ready"
    STEP_COMPLETE = "step_complete"
    EPOCH_COMPLETE = "epoch_complete"
    SIMULATION_COMPLETE = "simulation_complete"
    ACTION_TIMEOUT = "action_timeout"


@dataclass
class SimEvent:
    event_type: SimEventType
    simulation_id: str
    data: dict[str, Any] = field(default_factory=dict)
    agent_id: str | None = None  # None means broadcast to all


class _Subscriber:
    """Wraps an asyncio.Queue with its owning event loop for thread-safe put."""

    __slots__ = ("agent_id", "queue", "loop")

    def __init__(self, agent_id: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self.agent_id = agent_id
        self.queue = queue
        self.loop = loop


class EventBus:
    """In-memory pub/sub event bus for simulation events.

    Subscribers receive events via ``asyncio.Queue``. Publishing is
    thread-safe: if the caller is on a different thread than the
    subscriber's event loop, ``run_coroutine_threadsafe`` schedules a
    non-blocking queue put on the subscriber loop so waiters are notified
    while preserving drop-on-full semantics.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[_Subscriber]] = {}
        self._lock = threading.Lock()

    def subscribe(self, simulation_id: str, agent_id: str) -> asyncio.Queue:
        """Subscribe to events for a simulation. Returns a Queue to read from.

        Raises ValueError if the agent already has
        ``MAX_SUBSCRIPTIONS_PER_AGENT`` active subscriptions for this
        simulation.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            if simulation_id not in self._subscribers:
                self._subscribers[simulation_id] = []

            existing = sum(
                1 for s in self._subscribers[simulation_id] if s.agent_id == agent_id
            )
            if existing >= MAX_SUBSCRIPTIONS_PER_AGENT:
                raise ValueError(
                    f"Agent {agent_id} already has {existing} subscription(s) "
                    f"for simulation {simulation_id}"
                )

            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            self._subscribers[simulation_id].append(_Subscriber(agent_id, q, loop))
        return q

    def unsubscribe(
        self, simulation_id: str, agent_id: str, queue: asyncio.Queue
    ) -> None:
        """Remove a subscription."""
        with self._lock:
            subs = self._subscribers.get(simulation_id, [])
            self._subscribers[simulation_id] = [
                s for s in subs if not (s.agent_id == agent_id and s.queue is queue)
            ]

    def _put_event(self, sub: _Subscriber, event: SimEvent) -> bool:
        """Put an event into a subscriber's queue, thread-safely.

        Returns True if the event was delivered, False if the queue is full.
        """
        try:
            sub.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    async def _put_event_async(self, sub: _Subscriber, event: SimEvent) -> bool:
        """Put an event from another thread without blocking on full queues."""
        return self._put_event(sub, event)

    async def publish(self, event: SimEvent) -> int:
        """Publish an event (async). Returns number of subscribers notified.

        Safe to call from any thread — delegates to ``publish_sync``, which
        never blocks the event loop: the lock is held only to snapshot the
        subscriber list, same-loop delivery uses ``put_nowait`` (dropping on
        full queues), and cross-loop delivery schedules via
        ``run_coroutine_threadsafe`` without awaiting the result.
        """
        return self.publish_sync(event)

    def publish_sync(self, event: SimEvent) -> int:
        """Publish an event (sync). Returns number of subscribers notified.

        Safe to call from any thread.
        """
        with self._lock:
            subs = list(self._subscribers.get(event.simulation_id, []))

        # Determine if we're running inside an event loop
        try:
            calling_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            calling_loop = None

        count = 0
        for sub in subs:
            if event.agent_id is not None and event.agent_id != sub.agent_id:
                continue

            if calling_loop is sub.loop:
                # Same loop -- direct put is safe
                if self._put_event(sub, event):
                    count += 1
            else:
                # Different thread/loop -- use run_coroutine_threadsafe
                # so that asyncio tasks waiting on queue.get() are
                # properly woken up (call_soon_threadsafe + put_nowait
                # can miss wakeups on Python 3.12+), but keep put_nowait
                # semantics so full queues drop events instead of
                # accumulating pending Queue.put tasks.
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._put_event_async(sub, event), sub.loop
                    )
                    count += 1
                except RuntimeError:
                    pass  # Loop closed
        return count

    def subscriber_count(self, simulation_id: str) -> int:
        """Number of active subscribers for a simulation."""
        with self._lock:
            return len(self._subscribers.get(simulation_id, []))


# Global singleton
event_bus = EventBus()
