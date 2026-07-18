"""Helpers shared across framework bridges."""

import logging
from typing import Optional

from swarm.logging.event_log import EventLog
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


def log_interaction_event(
    event_log: Optional[EventLog], interaction: SoftInteraction
) -> None:
    """Append an interaction to the EventLog as an INTERACTION_COMPLETED event.

    No-op when event_log is None; write failures are logged, never raised —
    bridge telemetry must not break the host framework's run.
    """
    if event_log is None:
        return
    try:
        from swarm.models.events import Event, EventType

        event = Event(
            event_type=EventType.INTERACTION_COMPLETED,
            interaction_id=interaction.interaction_id,
            initiator_id=interaction.initiator,
            counterparty_id=interaction.counterparty,
            payload={
                "p": interaction.p,
                "v_hat": interaction.v_hat,
                "accepted": interaction.accepted,
                "metadata": interaction.metadata,
            },
        )
        event_log.append(event)
    except Exception as exc:  # pragma: no cover
        logger.warning("EventLog write failed: %s", exc)
