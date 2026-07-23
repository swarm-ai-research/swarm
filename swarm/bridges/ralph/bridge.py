"""Main bridge connecting Ralph exports to SWARM."""

import json
from pathlib import Path
from typing import List, Optional

from swarm.bridges._common import trim_to_half
from swarm.bridges.ralph.config import RalphConfig
from swarm.bridges.ralph.events import RalphEvent
from swarm.bridges.ralph.mapper import RalphMapper
from swarm.core.proxy import ProxyComputer
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import SoftInteraction


class RalphBridge:
    """Bridge that incrementally ingests Ralph JSONL events."""

    def __init__(
        self,
        config: RalphConfig,
        event_log: Optional[EventLog] = None,
    ) -> None:
        self._config = config
        self._event_log = event_log
        self._path = Path(config.events_path)
        self._mapper = RalphMapper(proxy=ProxyComputer(sigmoid_k=config.proxy_sigmoid_k))
        self._offset = 0
        self._interactions: List[SoftInteraction] = []
        self._events: List[RalphEvent] = []

    def poll(self) -> List[SoftInteraction]:
        """Read newly appended events and return mapped interactions."""
        if not self._path.exists():
            return []

        stat = self._path.stat()
        if stat.st_size < self._offset:
            # File was rotated/truncated; restart from beginning.
            self._offset = 0

        new_interactions: List[SoftInteraction] = []
        with self._path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue

                event = RalphEvent.from_dict(payload)
                self._record_event(event)

                actor_id = self._config.agent_role_map.get(event.actor_id, event.actor_id)
                interaction = self._mapper.map_event(
                    event=event,
                    initiator=self._config.orchestrator_id,
                    counterparty=actor_id,
                )
                self._record_interaction(interaction)
                new_interactions.append(interaction)

            self._offset = handle.tell()

        return new_interactions

    def get_events(self) -> List[RalphEvent]:
        """Return all observed Ralph events."""
        return list(self._events)

    def get_interactions(self) -> List[SoftInteraction]:
        """Return all interactions emitted by this bridge."""
        return list(self._interactions)

    def _record_event(self, event: RalphEvent) -> None:
        self._events = trim_to_half(self._events, self._config.max_events)
        self._events.append(event)

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        self._interactions = trim_to_half(self._interactions, self._config.max_interactions)
        self._interactions.append(interaction)
        if self._event_log is not None:
            self._event_log.append(
                Event(
                    event_type=EventType.INTERACTION_COMPLETED,
                    interaction_id=interaction.interaction_id,
                    initiator_id=interaction.initiator,
                    counterparty_id=interaction.counterparty,
                    payload=interaction.metadata,
                )
            )
