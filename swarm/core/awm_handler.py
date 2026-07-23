"""Handler for AWM (Agent World Model) database-backed task environments.

Manages the lifecycle of AWM tasks within SWARM simulations:
- Assigns AWM environments to agents
- Processes AWM_EXECUTE_TASK actions (tool call sequences)
- Maps tool-use traces to ProxyObservables
- Runs AWM verification and converts to soft labels
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from typing import Any, Dict, FrozenSet, List, Optional, Set

from swarm.agents.base import Action, ActionType
from swarm.bridges.awm.config import AWMConfig
from swarm.bridges.awm.mcp_client import (
    AWMEpisodeTrace,
    AWMMCPSyncClient,
    ToolCallRecord,
)
from swarm.bridges.awm.observable_mapper import AWMObservableMapper
from swarm.bridges.awm.server_manager import AWMServerManager
from swarm.bridges.awm.verifier_bridge import AWMVerifierBridge
from swarm.core.handler import Handler, HandlerActionResult
from swarm.env.state import EnvState
from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType

logger = logging.getLogger(__name__)


class AWMHandler(Handler):
    """Handler for AWM database-backed task environments.

    Each agent can be assigned an AWM task. When the agent submits
    an AWM_EXECUTE_TASK action with tool calls, this handler:
    1. Records tool calls into an episode trace
    2. Simulates execution (in Phase 1, without a live server)
    3. Runs verification via AWMVerifierBridge
    4. Maps the trace to ProxyObservables via AWMObservableMapper
    """

    def __init__(
        self,
        *,
        config: AWMConfig,
        event_bus: EventBus,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.config = config
        self._rng = random.Random(seed)
        self._mapper = AWMObservableMapper(
            error_weight=config.error_weight,
            misuse_weight=config.misuse_weight,
        )
        self._verifier = AWMVerifierBridge(
            confidence=config.verification_confidence,
        )

        # Active task assignments: agent_id -> task info
        self._assignments: Dict[str, Dict[str, Any]] = {}

        # Episode traces: agent_id -> current trace
        self._traces: Dict[str, AWMEpisodeTrace] = {}

        # Completed episodes for metrics
        self._completed_episodes: List[AWMEpisodeTrace] = []

        # Available tools (populated from environment metadata)
        self._available_tools: List[Dict[str, Any]] = []

        # Task queue for the current epoch
        self._task_queue: List[Dict[str, Any]] = []

        # Phase 3: Multi-turn last results per agent
        self._last_results: Dict[str, Dict[str, Any]] = {}

        # Phase 2: Live mode support
        self._server_manager: Optional[AWMServerManager] = None
        self._clients: Dict[str, AWMMCPSyncClient] = {}
        if config.live_mode:
            self._server_manager = AWMServerManager(config)

        # Phase 4: Shared-state multi-agent coordination
        self._write_sets: Dict[str, Set[str]] = {}
        self._agent_transactions: Dict[str, bool] = {}

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return frozenset({
            ActionType.AWM_EXECUTE_TASK,
            ActionType.AWM_TOOL_CALL,
            ActionType.AWM_FINISH_TASK,
        })

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from synchronous handler methods."""
        try:
            asyncio.get_running_loop()
            # Already inside a running event loop — run in a dedicated thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            # No running event loop — asyncio.run() is safe
            return asyncio.run(coro)

    def _ensure_server(self, agent_id: str) -> Optional[str]:
        """Ensure a live AWM server is running for the agent.

        Returns the base_url on success, None on failure.
        """
        if self._server_manager is None:
            return None

        server = self._run_async(
            self._server_manager.start_server(agent_id)
        )
        if server is None:
            return None

        # Create/cache sync client for this agent
        base_url: str = server.base_url
        if agent_id not in self._clients:
            self._clients[agent_id] = AWMMCPSyncClient(
                base_url=base_url,
                timeout=self.config.verification_timeout,
            )
        return base_url

    def on_epoch_start(self, state: EnvState) -> None:
        """Generate AWM task assignments for the epoch."""
        self._assignments.clear()
        self._traces.clear()

        # In live mode, ensure servers are started for all agents
        if self.config.live_mode and self._server_manager is not None:
            for agent_id in state.agents:
                self._ensure_server(agent_id)

        # Generate synthetic tasks for this epoch
        self._task_queue = self._generate_tasks()

        # Assign tasks to agents round-robin
        agent_ids = list(state.agents.keys())
        if not agent_ids or not self._task_queue:
            return

        for i, task in enumerate(self._task_queue):
            agent_id = agent_ids[i % len(agent_ids)]
            self._assignments[agent_id] = task
            self._traces[agent_id] = AWMEpisodeTrace(
                agent_id=agent_id,
                environment_id=self.config.environment_id,
                task_description=task.get("description", ""),
                max_steps=self.config.max_steps_per_task,
            )

            self._emit_event(Event(
                event_type=EventType.AWM_TASK_ASSIGNED,
                agent_id=agent_id,
                payload={
                    "task_id": task.get("task_id", ""),
                    "environment_id": self.config.environment_id,
                    "description": task.get("description", ""),
                },
            ))

    def on_epoch_end(self, state: EnvState) -> None:
        """Finalize any incomplete episodes."""
        for _agent_id, trace in list(self._traces.items()):
            if trace.verified is None:
                # Mark unfinished episodes as failed
                trace.verified = False
                self._completed_episodes.append(trace)

        self._assignments.clear()
        self._traces.clear()
        self._last_results.clear()
        self._write_sets.clear()
        self._agent_transactions.clear()

        # In live mode, reset all DBs between epochs
        if (
            self.config.live_mode
            and self.config.reset_between_epochs
            and self._server_manager is not None
        ):
            self._run_async(self._server_manager.reset_all())

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        """Generate synthetic AWM tasks for the epoch."""
        tasks = []
        for _ in range(self.config.max_tasks_per_epoch):
            tasks.append({
                "task_id": str(uuid.uuid4()),
                "environment_id": self.config.environment_id,
                "description": (
                    f"Complete the assigned task in the "
                    f"{self.config.environment_id} environment."
                ),
                "tools": self._get_available_tools(),
            })
        return tasks

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools for the current environment."""
        if self._available_tools:
            return self._available_tools

        # In live mode, try to fetch tools from a running server
        if self.config.live_mode and self._clients:
            client = next(iter(self._clients.values()))
            try:
                tools = client.list_tools()
                if tools:
                    self._available_tools = tools
                    return self._available_tools
            except (
                ConnectionError,
                ImportError,
                RuntimeError,
                ValueError,
                TypeError,
            ) as exc:
                logger.debug(
                    "Failed to fetch live tools, using defaults: %s",
                    exc,
                )

        # Default tools (synthetic)
        return [
            {"name": "query_database", "description": "Execute a SQL query"},
            {"name": "update_record", "description": "Update a database record"},
            {"name": "create_record", "description": "Create a new record"},
            {"name": "delete_record", "description": "Delete a record"},
            {"name": "list_tables", "description": "List available tables"},
        ]

    def build_observation_fields(
        self, agent_id: str, state: Any
    ) -> Dict[str, Any]:
        """Provide AWM task and tool information to agents."""
        result: Dict[str, Any] = {}

        task = self._assignments.get(agent_id)
        if task:
            result["awm_task"] = task
            result["awm_available_tools"] = task.get(
                "tools", self._get_available_tools()
            )

        # Phase 3: Multi-turn observation fields
        trace = self._traces.get(agent_id)
        if trace is not None:
            result["awm_episode_active"] = True
            result["awm_steps_remaining"] = max(
                0, trace.max_steps - trace.steps_used
            )
            last = self._last_results.get(agent_id)
            if last is not None:
                result["awm_last_result"] = last

        return result

    def observation_field_mapping(self) -> Dict[str, str]:
        return {
            "awm_task": "awm_task",
            "awm_available_tools": "awm_available_tools",
            "awm_last_result": "awm_last_result",
            "awm_episode_active": "awm_episode_active",
            "awm_steps_remaining": "awm_steps_remaining",
        }

    def handle_action(
        self, action: Action, state: EnvState
    ) -> HandlerActionResult:
        """Dispatch AWM actions to the appropriate handler method."""
        if action.action_type == ActionType.AWM_EXECUTE_TASK:
            return self._handle_batch_mode(action, state)
        elif action.action_type == ActionType.AWM_TOOL_CALL:
            return self._handle_tool_call(action, state)
        elif action.action_type == ActionType.AWM_FINISH_TASK:
            return self._handle_finish_task(action, state)
        return HandlerActionResult(success=False)

    # ------------------------------------------------------------------
    # Phase 4: Shared-state coordination helpers
    # ------------------------------------------------------------------

    def _begin_agent_transaction(self, agent_id: str) -> None:
        """Begin a transaction for an agent in shared-database mode."""
        if not self.config.shared_database:
            return
        if self.config.isolation_level == "none":
            return

        self._agent_transactions[agent_id] = True

        # In live mode, send HTTP begin
        if self.config.live_mode and agent_id in self._clients:
            self._clients[agent_id].begin_transaction()

    def _end_agent_transaction(
        self, agent_id: str, commit: bool = True
    ) -> None:
        """End a transaction for an agent in shared-database mode."""
        if not self.config.shared_database:
            return
        if self.config.isolation_level == "none":
            return
        if not self._agent_transactions.get(agent_id, False):
            return

        self._agent_transactions[agent_id] = False

        # In live mode, send HTTP end
        if self.config.live_mode and agent_id in self._clients:
            self._clients[agent_id].end_transaction(commit)

        self._emit_event(Event(
            event_type=EventType.AWM_TRANSACTION_COMPLETED,
            agent_id=agent_id,
            payload={"committed": commit},
        ))

    @staticmethod
    def _infer_write_tables(tool_name: str) -> Set[str]:
        """Heuristic: which tables a tool writes to.

        Write tools (update, create, delete) → ``{"default_table"}``.
        Read-only tools (query, list) → empty set.
        """
        write_prefixes = ("update_", "create_", "delete_", "insert_")
        if any(tool_name.startswith(p) for p in write_prefixes):
            return {"default_table"}
        return set()

    def _check_write_conflict(
        self, agent_id: str, tool_name: str
    ) -> bool:
        """Simulated conflict detection via write-set overlap + RNG.

        Returns True if a conflict is detected.
        """
        if not self.config.shared_database:
            return False

        tables = self._infer_write_tables(tool_name)
        if not tables:
            return False  # Read-only, no conflict possible

        # Check overlap with other agents' write sets
        for other_id, other_tables in self._write_sets.items():
            if other_id == agent_id:
                continue
            if tables & other_tables:
                # Overlap exists — probabilistic conflict
                if self._rng.random() < self.config.conflict_probability:
                    return True

        # Track this agent's writes
        if agent_id not in self._write_sets:
            self._write_sets[agent_id] = set()
        self._write_sets[agent_id] |= tables
        return False

    def _run_verification_and_map(
        self, trace: AWMEpisodeTrace, agent_id: str
    ) -> tuple:
        """Run verification and observable mapping for a completed episode.

        Returns:
            Tuple of (p, observables) where p is the soft label probability.
        """
        verification_result = self._execute_verification(trace, agent_id)
        trace.verified = verification_result.get("passed", False)
        trace.verification_details = verification_result

        p = self._verifier.verify_and_score(verification_result)
        observables = self._mapper.map(trace)
        return p, observables

    def _emit_task_completed_event(
        self, agent_id: str, trace: AWMEpisodeTrace, p: float
    ) -> None:
        """Emit an AWM_TASK_COMPLETED event with standard payload."""
        self._emit_event(Event(
            event_type=EventType.AWM_TASK_COMPLETED,
            agent_id=agent_id,
            payload={
                "episode_id": trace.episode_id,
                "verified": trace.verified,
                "steps_used": trace.steps_used,
                "error_count": trace.error_count,
                "malformed_count": trace.malformed_count,
                "soft_p": p,
            },
        ))

    def _create_completion_result(
        self, agent_id: str, trace: AWMEpisodeTrace, p: float,
        observables: Any
    ) -> HandlerActionResult:
        """Create a completion result for a finished AWM episode."""
        return HandlerActionResult(
            success=True,
            observables=observables,
            initiator_id=agent_id,
            counterparty_id=agent_id,
            metadata={
                "episode_id": trace.episode_id,
                "verified": trace.verified,
                "soft_p": p,
                "steps_used": trace.steps_used,
            },
        )

    def _handle_batch_mode(
        self, action: Action, state: EnvState
    ) -> HandlerActionResult:
        """Process an AWM_EXECUTE_TASK action (batch mode)."""
        agent_id = action.agent_id
        trace = self._traces.get(agent_id)

        if trace is None:
            return HandlerActionResult(success=False)

        if trace.steps_used >= self.config.max_steps_per_task:
            return HandlerActionResult(
                success=False,
                metadata={"error": "max_steps_exceeded"},
            )

        # Phase 4: begin transaction in shared mode
        self._begin_agent_transaction(agent_id)

        conflict_occurred = False
        tool_calls = action.metadata.get("tool_calls", [])
        for tc_data in tool_calls:
            record = self._execute_tool_call(tc_data, agent_id)
            trace.tool_calls.append(record)
            trace.steps_used += 1

            if record.conflict_detected:
                conflict_occurred = True
                break

            if trace.steps_used >= self.config.max_steps_per_task:
                break

        # Phase 4: end transaction (rollback on conflict)
        self._end_agent_transaction(agent_id, commit=not conflict_occurred)

        # Run verification and map observables
        p, observables = self._run_verification_and_map(trace, agent_id)

        self._completed_episodes.append(trace)
        self._traces.pop(agent_id, None)
        self._assignments.pop(agent_id, None)

        self._emit_task_completed_event(agent_id, trace, p)
        return self._create_completion_result(agent_id, trace, p, observables)

    def _handle_tool_call(
        self, action: Action, state: EnvState
    ) -> HandlerActionResult:
        """Process a single AWM_TOOL_CALL action (multi-turn step mode)."""
        agent_id = action.agent_id
        trace = self._traces.get(agent_id)

        if trace is None:
            return HandlerActionResult(success=False)

        if trace.steps_used >= trace.max_steps:
            return HandlerActionResult(
                success=False,
                metadata={"error": "max_steps_exceeded"},
            )

        # Execute the single tool call
        tc_data = {
            "tool_name": action.metadata.get("tool_name", ""),
            "arguments": action.metadata.get("arguments", {}),
        }
        record = self._execute_tool_call(tc_data, agent_id)
        trace.tool_calls.append(record)
        trace.steps_used += 1

        # Store result for next observation
        self._last_results[agent_id] = {
            "tool_name": record.tool_name,
            "success": record.success,
            "result": record.result,
            "error": record.error,
            "is_error_response": record.is_error_response,
        }

        # Emit tool call event
        self._emit_event(Event(
            event_type=EventType.AWM_TOOL_CALL_EXECUTED,
            agent_id=agent_id,
            payload={
                "episode_id": trace.episode_id,
                "tool_name": record.tool_name,
                "success": record.success,
                "step": trace.steps_used,
            },
        ))

        # No observables — episode continues
        return HandlerActionResult(success=True, observables=None)

    def _handle_finish_task(
        self, action: Action, state: EnvState
    ) -> HandlerActionResult:
        """Process an AWM_FINISH_TASK action (finalize multi-turn episode)."""
        agent_id = action.agent_id
        trace = self._traces.get(agent_id)

        if trace is None:
            return HandlerActionResult(success=False)

        # Phase 4: end any open transaction before verification
        self._end_agent_transaction(agent_id, commit=True)

        # Run verification and map observables
        p, observables = self._run_verification_and_map(trace, agent_id)

        # Track completed episode
        self._completed_episodes.append(trace)

        # Clean up
        self._traces.pop(agent_id, None)
        self._assignments.pop(agent_id, None)
        self._last_results.pop(agent_id, None)

        self._emit_task_completed_event(agent_id, trace, p)
        return self._create_completion_result(agent_id, trace, p, observables)

    def _execute_tool_call(
        self, tc_data: Dict[str, Any], agent_id: str
    ) -> ToolCallRecord:
        """Dispatch tool call to live or simulated implementation.

        In shared-database mode, checks for simulated write conflicts
        before dispatch and detects SQLITE_BUSY errors after live calls.
        """
        tool_name = tc_data.get("tool_name", "")

        # Phase 4: simulated conflict check (before dispatch)
        if self._check_write_conflict(agent_id, tool_name):
            record = ToolCallRecord(
                tool_name=tool_name,
                arguments=tc_data.get("arguments", {}),
            )
            record.conflict_detected = True
            record.is_error_response = True
            record.error = "write_conflict"
            self._emit_event(Event(
                event_type=EventType.AWM_CONFLICT_DETECTED,
                agent_id=agent_id,
                payload={
                    "tool_name": tool_name,
                    "conflict_type": "simulated_write_set_overlap",
                },
            ))
            return record

        if self.config.live_mode and agent_id in self._clients:
            record = self._live_tool_call(tc_data, agent_id)
        else:
            record = self._simulate_tool_call(tc_data)

        # Phase 4: detect SQLITE_BUSY from live call errors
        if (
            self.config.shared_database
            and record.is_error_response
            and record.error
            and "SQLITE_BUSY" in record.error.upper()
        ):
            record.conflict_detected = True
            self._emit_event(Event(
                event_type=EventType.AWM_CONFLICT_DETECTED,
                agent_id=agent_id,
                payload={
                    "tool_name": tool_name,
                    "conflict_type": "sqlite_busy",
                },
            ))

        return record

    def _live_tool_call(
        self, tc_data: Dict[str, Any], agent_id: str
    ) -> ToolCallRecord:
        """Execute a tool call via the live AWM server."""
        tool_name = tc_data.get("tool_name", "")
        arguments = tc_data.get("arguments", {})
        client = self._clients[agent_id]
        try:
            return client.call_tool(tool_name, arguments)
        except (
            ConnectionError,
            ImportError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning(
                "Live tool call failed for agent=%s tool=%s: %s",
                agent_id,
                tool_name,
                exc,
            )
            record = ToolCallRecord(tool_name=tool_name, arguments=arguments)
            record.is_error_response = True
            record.error = str(exc)
            return record

    def _execute_verification(
        self, trace: AWMEpisodeTrace, agent_id: str
    ) -> Dict[str, Any]:
        """Dispatch verification to live or simulated implementation."""
        if self.config.live_mode and agent_id in self._clients:
            return self._live_verification(trace, agent_id)
        return self._simulate_verification(trace)

    def _live_verification(
        self, trace: AWMEpisodeTrace, agent_id: str
    ) -> Dict[str, Any]:
        """Run verification via the live AWM server."""
        client = self._clients[agent_id]
        try:
            result: Dict[str, Any] = client.verify()
            return result
        except (
            ConnectionError,
            ImportError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning(
                "Live verification failed for agent=%s: %s", agent_id, exc
            )
            return {"passed": False, "error": str(exc)}

    def _simulate_tool_call(
        self, tc_data: Dict[str, Any]
    ) -> ToolCallRecord:
        """Simulate a tool call for Phase 1 (no live server).

        In Phase 2+, this will be replaced by actual MCP HTTP calls.
        """
        tool_name = tc_data.get("tool_name", "")
        arguments = tc_data.get("arguments", {})

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
        )

        # Validate tool name exists
        available = {t["name"] for t in self._get_available_tools()}
        if tool_name not in available:
            record.is_malformed = True
            record.is_error_response = True
            record.error = f"Unknown tool: {tool_name}"
            return record

        # Simulate success/failure based on RNG
        if self._rng.random() < 0.85:
            record.success = True
            record.result = {"status": "ok", "rows_affected": 1}
        else:
            record.is_error_response = True
            record.error = "simulated_error"

        return record

    def _simulate_verification(
        self, trace: AWMEpisodeTrace
    ) -> Dict[str, Any]:
        """Simulate AWM verification for Phase 1.

        Verification success correlates with:
        - Fewer errors
        - Fewer malformed calls
        - More total tool calls (agent actually tried)
        """
        if trace.total_calls == 0:
            return {"passed": False, "reason": "no_tool_calls"}

        # Success probability based on trace quality
        error_ratio = trace.error_count / max(trace.total_calls, 1)
        malformed_ratio = trace.malformed_count / max(trace.total_calls, 1)
        success_prob = max(0.0, 0.8 - error_ratio - malformed_ratio * 1.5)

        passed = self._rng.random() < success_prob

        return {
            "passed": passed,
            "confidence": self.config.verification_confidence,
            "error_ratio": error_ratio,
            "malformed_ratio": malformed_ratio,
        }

    def get_completed_episodes(self) -> List[AWMEpisodeTrace]:
        """Return all completed episodes."""
        return list(self._completed_episodes)
