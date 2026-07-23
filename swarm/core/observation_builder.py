"""Observation construction for agents.

Extracted from ``Orchestrator`` to isolate the per-agent observation
assembly into a focused, testable component.  The orchestrator
delegates to an ``ObservationBuilder`` instance, passing shared
mutable references so all state reads are live.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

from swarm.agents.base import Observation
from swarm.core.handler_registry import HandlerRegistry
from swarm.core.spawn import SpawnTree
from swarm.env.feed import Feed
from swarm.env.network import AgentNetwork
from swarm.env.state import EnvState
from swarm.env.tasks import TaskPool, TaskStatus
from swarm.models.agent import AgentState

logger = logging.getLogger(__name__)


class ObservationBuilder:
    """Builds per-agent ``Observation`` objects from environment state.

    Responsibilities:
    - Gather visible posts, pending proposals, tasks, and agent info
    - Filter agents by network topology when enabled
    - Collect handler-contributed observation fields via the registry
    - Apply configurable observation noise
    """

    def __init__(
        self,
        config: Any,  # OrchestratorConfig (avoid circular import)
        state: EnvState,
        feed: Feed,
        task_pool: TaskPool,
        network: Optional[AgentNetwork],
        handler_registry: HandlerRegistry,
        rng: random.Random,
        spawn_tree: Optional[SpawnTree] = None,
    ) -> None:
        self._config = config
        self._state = state
        self._feed = feed
        self._task_pool = task_pool
        self._network = network
        self._handler_registry = handler_registry
        self._rng = rng
        self._spawn_tree = spawn_tree
        # Per-step artifact cache; keyed by (step, registry_version) so
        # mid-step publishes/consumes invalidate stale views.
        self._artifact_cache_key: tuple[int, int] = (-1, -1)
        self._artifact_cache: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, agent_id: str) -> Observation:
        """Build an ``Observation`` for *agent_id*."""
        agent_state = self._state.get_agent(agent_id)
        rate_limit = self._state.get_rate_limit_state(agent_id)

        visible_posts = [p.to_dict() for p in self._feed.get_ranked_posts(limit=20)]

        pending_proposals = [
            {
                "proposal_id": p.proposal_id,
                "initiator_id": p.initiator_id,
                "interaction_type": p.interaction_type,
                "content": p.content,
                "offered_transfer": p.metadata.get("offered_transfer", 0),
            }
            for p in self._state.get_proposals_for_agent(agent_id)
        ]

        available_tasks = [
            t.to_dict()
            for t in self._task_pool.get_claimable_tasks(
                agent_reputation=agent_state.reputation if agent_state else 0,
                current_epoch=self._state.current_epoch,
            )
        ]

        active_tasks = [
            t.to_dict()
            for t in self._task_pool.get_tasks_for_agent(agent_id)
            if t.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS)
        ]

        active_agents = self._state.get_active_agents()

        if self._network is not None:
            neighbor_ids = set(self._network.neighbors(agent_id))
            active_agents = [s for s in active_agents if s.agent_id in neighbor_ids]

        visible_agents = [
            {
                "agent_id": s.agent_id,
                "name": s.name,
                "agent_type": s.agent_type.value,
                "reputation": s.reputation,
                "resources": s.resources,
                "edge_weight": self._network.edge_weight(agent_id, s.agent_id)
                if self._network
                else 1.0,
            }
            for s in active_agents
            if s.agent_id != agent_id
        ]

        visible_agents = [self.apply_noise(record) for record in visible_agents]

        # Collect handler observation fields via registry
        handler_fields: Dict[str, Any] = {}
        for handler in self._handler_registry.all_handlers():
            try:
                handler.on_pre_observation(agent_id, self._state)
                fields = handler.build_observation_fields(agent_id, self._state)
            except Exception as e:
                logger.exception(
                    "Handler %s failed to build observation fields for agent %s",
                    type(handler).__name__,
                    agent_id,
                    exc_info=e,
                )
                continue
            mapping = handler.observation_field_mapping()
            for key, value in fields.items():
                obs_key = mapping.get(key, key)
                if obs_key in handler_fields:
                    raise ValueError(
                        f"Observation field '{obs_key}' returned by "
                        f"{type(handler).__name__} conflicts with a field "
                        f"already set by another handler"
                    )
                handler_fields[obs_key] = value

        # Compute spawn fields
        can_spawn = False
        spawn_depth = 0
        spawn_children_count = 0
        if self._spawn_tree is not None:
            spawn_depth = self._spawn_tree.get_depth(agent_id)
            spawn_children_count = len(self._spawn_tree.get_children(agent_id))
            global_step = (
                self._state.current_epoch * self._config.steps_per_epoch
                + self._state.current_step
            )
            can_spawn_result, _ = self._spawn_tree.can_spawn(
                agent_id,
                global_step,
                agent_state.resources if agent_state else 0.0,
            )
            can_spawn = can_spawn_result

        return Observation(
            agent_state=agent_state or AgentState(),
            current_epoch=self._state.current_epoch,
            current_step=self._state.current_step,
            can_post=rate_limit.can_post(self._state.rate_limits)
            if self._config.enable_rate_limits
            else True,
            can_interact=rate_limit.can_interact(self._state.rate_limits)
            if self._config.enable_rate_limits
            else True,
            can_vote=rate_limit.can_vote(self._state.rate_limits)
            if self._config.enable_rate_limits
            else True,
            can_claim_task=rate_limit.can_claim_task(self._state.rate_limits)
            if self._config.enable_rate_limits
            else True,
            visible_posts=visible_posts,
            pending_proposals=pending_proposals,
            available_tasks=available_tasks,
            active_tasks=active_tasks,
            visible_agents=visible_agents,
            # Marketplace fields
            available_bounties=handler_fields.get("available_bounties", []),
            active_bids=handler_fields.get("active_bids", []),
            active_escrows=handler_fields.get("active_escrows", []),
            pending_bid_decisions=handler_fields.get("pending_bid_decisions", []),
            # Moltipedia fields
            contested_pages=handler_fields.get("contested_pages", []),
            search_results=handler_fields.get("search_results", []),
            random_pages=handler_fields.get("random_pages", []),
            leaderboard=handler_fields.get("leaderboard", []),
            agent_points=handler_fields.get("agent_points", 0.0),
            heartbeat_status=handler_fields.get("heartbeat_status", {}),
            ecosystem_metrics=self.apply_noise(
                self._state.get_epoch_metrics_snapshot()
            ),
            # Moltbook fields
            moltbook_published_posts=handler_fields.get("moltbook_published_posts", []),
            moltbook_pending_posts=handler_fields.get("moltbook_pending_posts", []),
            moltbook_rate_limits=handler_fields.get("moltbook_rate_limits", {}),
            moltbook_karma=handler_fields.get("moltbook_karma", 0.0),
            # Memory fields
            memory_hot_cache=handler_fields.get("memory_hot_cache", []),
            memory_pending_promotions=handler_fields.get("memory_pending_promotions", []),
            memory_search_results=handler_fields.get("memory_search_results", []),
            memory_challenged_entries=handler_fields.get("memory_challenged_entries", []),
            memory_entry_counts=handler_fields.get("memory_entry_counts", {}),
            memory_writes_remaining=handler_fields.get("memory_writes_remaining", 0),
            # Scholar fields
            scholar_query=handler_fields.get("scholar_query"),
            scholar_passage_pool=handler_fields.get("scholar_passage_pool", []),
            scholar_draft_citations=handler_fields.get("scholar_draft_citations", []),
            scholar_citation_to_verify=handler_fields.get("scholar_citation_to_verify"),
            scholar_synthesis_result=handler_fields.get("scholar_synthesis_result"),
            # Kernel fields
            kernel_available_challenges=handler_fields.get("kernel_available_challenges", []),
            kernel_pending_submissions=handler_fields.get("kernel_pending_submissions", []),
            kernel_submissions_to_verify=handler_fields.get("kernel_submissions_to_verify", []),
            kernel_submission_history=handler_fields.get("kernel_submission_history", []),
            # Rivals (Team-of-Rivals) fields
            rivals_assignments=handler_fields.get("rivals_assignments", []),
            # AWM (Agent World Model) fields
            awm_task=handler_fields.get("awm_task"),
            awm_available_tools=handler_fields.get("awm_available_tools", []),
            # Spawn fields
            can_spawn=can_spawn,
            spawn_depth=spawn_depth,
            spawn_children_count=spawn_children_count,
            # Artifact layer fields (use per-step cache to avoid O(agents × artifacts))
            available_artifacts=self._get_artifacts_for_agent(agent_id),
            artifact_pressure=self._state.artifact_registry.pressure_scores(),
        )

    def _get_artifacts_for_agent(self, agent_id: str) -> list:
        """Return available artifacts for *agent_id*, using a cached index.

        The cache is rebuilt when either the simulation step advances or
        the registry's mutation version changes.  This keeps the O(producers)
        per-agent lookup while preventing stale views when an earlier agent
        in the same step publishes/consumes an artifact.
        """
        registry = self._state.artifact_registry
        current_step = (
            self._state.current_epoch * self._config.steps_per_epoch
            + self._state.current_step
        )
        key = (current_step, registry.version())
        if self._artifact_cache_key != key:
            self._artifact_cache = registry.fresh_artifact_dicts(current_step)
            self._artifact_cache_key = key

        return registry.match_for_agent(
            agent_id, current_step, _cache=self._artifact_cache,
        )

    def apply_noise(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configurable gaussian noise to numeric observation fields.

        Noise is only applied when enabled and never to boolean fields.
        """
        if (
            self._config.observation_noise_probability <= 0
            or self._config.observation_noise_std <= 0
        ):
            return record

        noisy: Dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, bool):
                noisy[key] = value
                continue
            if isinstance(value, (int, float)):
                if self._rng.random() < self._config.observation_noise_probability:
                    noisy[key] = float(value) + self._rng.gauss(
                        0.0, self._config.observation_noise_std
                    )
                else:
                    noisy[key] = value
                continue
            noisy[key] = value
        return noisy
