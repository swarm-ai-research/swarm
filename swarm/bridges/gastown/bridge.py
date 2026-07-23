"""Main bridge connecting a GasTown workspace to SWARM.

GasTownBridge polls the beads SQLite database and git worktrees,
converts lifecycle events into SWARM SoftInteractions, and logs
them to the SWARM event pipeline.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from swarm.bridges._common import trim_to_half
from swarm.bridges.gastown.beads import BeadsClient
from swarm.bridges.gastown.config import GasTownConfig
from swarm.bridges.gastown.events import GasTownEvent
from swarm.bridges.gastown.git_observer import GitObserver
from swarm.bridges.gastown.mapper import GasTownMapper
from swarm.core.proxy import ProxyComputer
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


class GasTownBridge:
    """Bridge between a GasTown workspace and the SWARM framework.

    Lifecycle::

        bridge = GasTownBridge(config)
        interactions = bridge.poll()  # call periodically
        bridge.shutdown()

    The bridge translates GasTown signals into SWARM's data model:
    - Bead state transitions become SoftInteraction records
    - Git PR stats map to ProxyObservables
    - All events are logged to SWARM's append-only event log
    """

    def __init__(
        self,
        config: GasTownConfig,
        event_log: Optional[EventLog] = None,
    ) -> None:
        self._config = config
        self._event_log = event_log

        # Resolve beads DB path
        db_path = config.beads_db_path or os.path.join(
            config.workspace_path, ".beads", "beads.db"
        )
        self._beads_client = BeadsClient(db_path)
        self._git_observer = GitObserver(config.workspace_path)
        self._mapper = GasTownMapper(
            proxy=ProxyComputer(sigmoid_k=config.proxy_sigmoid_k)
        )

        self._interactions: List[SoftInteraction] = []
        self._events: List[GasTownEvent] = []
        # Start from epoch so the first poll picks up all existing beads.
        self._last_poll = datetime(2000, 1, 1, tzinfo=timezone.utc)

    # --- Main polling interface ---

    def poll(self) -> List[SoftInteraction]:
        """Poll beads + git for new events and return new SoftInteractions.

        Workflow:
        1. Query BeadsClient for state changes since last poll.
        2. For each event, try worktree stats first, then fall back to
           matching a remote feature branch for per-agent git stats.
        3. Map to SoftInteractions via GasTownMapper.
        4. Append to internal store and log to EventLog.
        5. Update last poll timestamp.
        """
        now = datetime.now(timezone.utc)
        new_events = self._beads_client.poll_changes(self._last_poll)
        new_interactions: List[SoftInteraction] = []

        base = self._config.base_branch
        worktrees = self._git_observer.get_agent_worktrees()

        # Build a lookup from agent name → best matching branch for agents
        # that don't have a local worktree.
        branch_map: Dict[str, str] = {}
        for info in self._git_observer.get_feature_branches(base):
            agent = info["agent"]
            if agent not in worktrees:
                branch_map.setdefault(agent, info["branch"])

        for event in new_events:
            self._record_event(event)

            agent_name = event.agent_name
            agent_id = self._config.agent_role_map.get(agent_name, agent_name)

            # Get git stats: prefer worktree, fall back to branch
            wt = worktrees.get(agent_name, "")
            if wt:
                git_stats = self._git_observer.get_pr_stats(wt)
            elif agent_name in branch_map:
                git_stats = self._git_observer.get_branch_stats(
                    branch_map[agent_name], base
                )
            else:
                git_stats = {}

            # Build a bead dict from the event payload for the mapper
            bead = {
                "id": event.bead_id or "",
                "status": event.payload.get("status", "open"),
                "title": event.payload.get("title", ""),
                "assignee": agent_name,
            }

            interaction = self._mapper.map_bead_completion(
                bead=bead,
                git_stats=git_stats,
                agent_id=agent_id,
            )
            self._record_interaction(interaction)
            new_interactions.append(interaction)

        self._last_poll = now
        return new_interactions

    def poll_branches(self) -> List[SoftInteraction]:
        """Discover unmerged feature branches and score each one.

        Each branch becomes a SoftInteraction with per-branch git stats.
        Branches are treated as in-progress work units; the agent is
        inferred from the branch prefix (e.g. ``claude/``, ``codex/``).

        Returns only *new* branch interactions (branches already seen
        in a previous call are skipped).
        """
        base = self._config.base_branch
        branches = self._git_observer.get_feature_branches(base)
        new_interactions: List[SoftInteraction] = []

        seen_branches = {
            i.metadata.get("branch")
            for i in self._interactions
            if i.metadata.get("source") == "branch"
        }

        for info in branches:
            ref = info["branch"]
            if ref in seen_branches:
                continue

            git_stats = self._git_observer.get_branch_stats(ref, base)
            agent_id = self._config.agent_role_map.get(
                info["agent"], info["agent"]
            )

            bead = {
                "id": ref,
                "status": "in_progress",
                "title": info["slug"],
                "assignee": info["agent"],
            }

            interaction = self._mapper.map_bead_completion(
                bead=bead,
                git_stats=git_stats,
                agent_id=agent_id,
            )
            # Tag with branch metadata so we can deduplicate later
            interaction.metadata["source"] = "branch"
            interaction.metadata["branch"] = ref

            self._record_interaction(interaction)
            new_interactions.append(interaction)

        return new_interactions

    # --- Accessors ---

    def get_interactions(self) -> List[SoftInteraction]:
        """Return all interactions recorded by this bridge."""
        return list(self._interactions)

    def get_events(self) -> List[GasTownEvent]:
        """Return all GasTown events observed by this bridge."""
        return list(self._events)

    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Return summary stats for an agent."""
        agent_id = self._config.agent_role_map.get(agent_name, agent_name)
        agent_interactions = [
            i for i in self._interactions if i.counterparty == agent_id
        ]
        if not agent_interactions:
            return {"agent_id": agent_id, "interactions": 0}
        avg_p = sum(i.p for i in agent_interactions) / len(agent_interactions)
        return {
            "agent_id": agent_id,
            "interactions": len(agent_interactions),
            "avg_p": avg_p,
        }

    # --- Lifecycle ---

    def shutdown(self) -> None:
        """Close the beads DB connection."""
        self._beads_client.close()

    # --- Internal helpers ---

    def _record_event(self, event: GasTownEvent) -> None:
        self._events = trim_to_half(self._events, self._config.max_events)
        self._events.append(event)

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        self._interactions = trim_to_half(self._interactions, self._config.max_interactions)
        self._interactions.append(interaction)
        self._log_interaction(interaction)

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        if self._event_log is None:
            return
        event = Event(
            event_type=EventType.INTERACTION_COMPLETED,
            interaction_id=interaction.interaction_id,
            initiator_id=interaction.initiator,
            counterparty_id=interaction.counterparty,
            payload={
                "accepted": interaction.accepted,
                "v_hat": interaction.v_hat,
                "p": interaction.p,
                "bridge": "gastown",
                "metadata": interaction.metadata,
            },
        )
        self._event_log.append(event)
