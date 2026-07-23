"""Hardware root-of-trust rejection handling governance lever.

When a hardware root of trust issues a Rejection/Halt command, this lever
treats it as a first-class governance signal.  The orchestration layer:

1. Immediately freezes all unsafe action pathways.
2. Preserves task state and records the causal trace that triggered the halt.
3. Propagates a signed stop token across dependent agents so the swarm
   converges to a safe quiescent state instead of fragmenting.
4. Downgrades affected agents into constrained recovery mode (no irreversible
   actions, no new external side effects, bounded local reasoning only).
5. Safe-resume occurs only after hardware clears the condition, at which
   point agents reload the last admissible state and continue from a
   verified checkpoint.
"""

import copy
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.events import Event, EventType

logger = logging.getLogger(__name__)


class HaltReason(Enum):
    """Reason categories for hardware halt signals."""

    INTEGRITY_VIOLATION = "integrity_violation"
    ATTESTATION_FAILURE = "attestation_failure"
    TAMPER_DETECTED = "tamper_detected"
    POLICY_BREACH = "policy_breach"
    EXTERNAL_REVOCATION = "external_revocation"


class RecoveryMode(Enum):
    """Agent recovery mode after a hardware halt.

    Recovery budget is **step-based**, not action-based: each call to
    ``on_step`` increments the counter regardless of whether the agent
    actually acted during that step.  This is intentional — idle silence
    under a halt should not buy unlimited time.  If time-gated recovery
    is undesirable for a scenario, set ``hardware_trust_recovery_max_steps``
    high enough to effectively disable escalation.
    """

    NORMAL = "normal"
    CONSTRAINED = "constrained"  # No irreversible actions, no external side effects
    FROZEN = "frozen"  # Completely halted


@dataclass
class HaltRecord:
    """Record of a hardware halt event."""

    halt_id: str
    timestamp: datetime
    reason: HaltReason
    source_agent_id: Optional[str]  # Agent whose action triggered the halt (if any)
    causal_trace: List[str]  # Chain of event IDs leading to the halt
    affected_agents: Set[str]  # Agents frozen by this halt
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleared: bool = False
    cleared_at: Optional[datetime] = None


@dataclass
class AgentRecoveryState:
    """Tracks recovery state for an agent under hardware halt."""

    mode: RecoveryMode = RecoveryMode.NORMAL
    halt_id: Optional[str] = None
    checkpoint_epoch: Optional[int] = None
    checkpoint_step: Optional[int] = None
    recovery_steps_taken: int = 0
    task_state_preserved: Dict[str, Any] = field(default_factory=dict)


# Actions explicitly allowed in constrained recovery mode.
# Everything NOT on this list is blocked — this fails safe when new
# action types are added to the codebase.
REVERSIBLE_ACTIONS: FrozenSet[str] = frozenset({
    "noop",
    "vote",
    "reply",
    "reject_interaction",
    "reject_bid",
    "withdraw_bid",
    "verify_output",
    "file_objection",
    "file_dispute",
})

# A sample of actions blocked in constrained mode, retained because tests
# import it. Enforcement uses the REVERSIBLE_ACTIONS allowlist above —
# anything not on that list is blocked, not just the actions named here.
IRREVERSIBLE_ACTIONS: FrozenSet[str] = frozenset({
    "propose_interaction",
    "submit_output",
    "post_bounty",
    "accept_bid",
    "create_page",
    "edit_page",
    "moltbook_post",
    "write_memory",
    "promote_memory",
    "submit_kernel",
    "spawn_subagent",
    "awm_execute_task",
    "awm_tool_call",
})


class HardwareTrustLever(GovernanceLever):
    """Governance lever for hardware root-of-trust rejection handling.

    This lever intercepts hardware halt signals, freezes affected agents,
    propagates stop tokens to dependents, and manages constrained recovery.

    .. note:: Thread safety

       All internal state (``_halt_records``, ``_recovery_states``,
       ``_dependency_graph``, ``_active_halt_ids``) is mutated without
       locking.  The lever assumes single-threaded access from the
       orchestrator's step loop.  If concurrent access is ever needed,
       wrap ``receive_halt`` / ``clear_halt`` / ``on_step`` with an
       external lock.
    """

    def __init__(self, config: GovernanceConfig):
        super().__init__(config)
        self._halt_records: Dict[str, HaltRecord] = {}
        self._recovery_states: Dict[str, AgentRecoveryState] = {}
        self._active_halt_ids: List[str] = []  # Ordered list for deterministic iteration
        self._dependency_graph: Dict[str, Set[str]] = {}  # agent -> dependents
        self._event_log: List[Event] = []  # Accumulated events for JSONL emission

    @property
    def name(self) -> str:
        return "hardware_trust"

    def register_dependency(self, agent_id: str, dependent_id: str) -> None:
        """Register that *dependent_id* depends on *agent_id*.

        Used for stop-token propagation: when *agent_id* is halted,
        *dependent_id* will also be frozen.
        """
        if agent_id not in self._dependency_graph:
            self._dependency_graph[agent_id] = set()
        self._dependency_graph[agent_id].add(dependent_id)

    def set_dependency_graph(self, graph: Dict[str, Set[str]]) -> None:
        """Replace the entire dependency graph."""
        self._dependency_graph = {k: set(v) for k, v in graph.items()}

    # ------------------------------------------------------------------
    # Halt lifecycle
    # ------------------------------------------------------------------

    def receive_halt(
        self,
        halt_id: str,
        reason: HaltReason,
        state: EnvState,
        source_agent_id: Optional[str] = None,
        causal_trace: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LeverEffect:
        """Process a hardware halt signal.

        This is the main entry point called when the hardware root of trust
        issues a Rejection/Halt command.

        Args:
            halt_id: Unique identifier for this halt event.
            reason: Why the halt was issued.
            state: Current environment state.
            source_agent_id: Agent whose action triggered the halt (if any).
            causal_trace: Event IDs forming the causal chain.
            metadata: Additional halt metadata.

        Returns:
            LeverEffect with agents to freeze.
        """
        if not self.config.hardware_trust_enabled:
            return LeverEffect(lever_name=self.name)

        now = datetime.now(tz=timezone.utc)

        # Determine which agents to freeze
        agents_to_freeze: Set[str] = set()
        if source_agent_id:
            agents_to_freeze.add(source_agent_id)

        # Propagate to all dependents if propagation enabled
        if self.config.hardware_trust_propagation_enabled:
            propagated = self._compute_propagation_set(agents_to_freeze, state)
            if propagated:
                self._emit(
                    EventType.HARDWARE_HALT_PROPAGATED,
                    payload={
                        "halt_id": halt_id,
                        "source_agents": sorted(agents_to_freeze),
                        "propagated_to": sorted(propagated),
                    },
                    epoch=state.current_epoch,
                    step=state.current_step,
                )
            agents_to_freeze |= propagated

        # If no specific agent, freeze everyone (global halt)
        if not agents_to_freeze:
            agents_to_freeze = set(state.agents.keys())

        # Record the halt
        record = HaltRecord(
            halt_id=halt_id,
            timestamp=now,
            reason=reason,
            source_agent_id=source_agent_id,
            causal_trace=list(causal_trace or []),
            affected_agents=set(agents_to_freeze),
            metadata=metadata or {},
        )
        self._halt_records[halt_id] = record
        if halt_id not in self._active_halt_ids:
            self._active_halt_ids.append(halt_id)

        # Emit HARDWARE_HALT_RECEIVED event
        self._emit(
            EventType.HARDWARE_HALT_RECEIVED,
            agent_id=source_agent_id,
            payload={
                "halt_id": halt_id,
                "reason": reason.value,
                "affected_agents": sorted(agents_to_freeze),
                "causal_trace": list(causal_trace or []),
            },
            epoch=state.current_epoch,
            step=state.current_step,
        )

        # Set up recovery state for each affected agent.
        # Don't overwrite existing recovery entries — keep the more
        # restrictive state if an agent is already tracked by a prior halt.
        for agent_id in agents_to_freeze:
            existing = self._recovery_states.get(agent_id)
            if existing is not None and existing.mode in (
                RecoveryMode.CONSTRAINED,
                RecoveryMode.FROZEN,
            ):
                # Already under a halt — keep existing (more restrictive) state.
                # Record this halt in the affected_agents of the new record,
                # but don't reset the agent's recovery progress.
                continue
            self._recovery_states[agent_id] = AgentRecoveryState(
                mode=RecoveryMode.CONSTRAINED,
                halt_id=halt_id,
                checkpoint_epoch=state.current_epoch,
                checkpoint_step=state.current_step,
                recovery_steps_taken=0,
            )
            self._emit(
                EventType.HARDWARE_RECOVERY_ENTERED,
                agent_id=agent_id,
                payload={
                    "halt_id": halt_id,
                    "mode": RecoveryMode.CONSTRAINED.value,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )

        # Constrained agents are NOT returned in agents_to_freeze —
        # they can still act (with action-level filtering via
        # is_action_allowed). The state-level freeze is reserved for
        # agents that escalate to FROZEN mode.
        return LeverEffect(
            lever_name=self.name,
            details={
                "halt_id": halt_id,
                "reason": reason.value,
                "source_agent_id": source_agent_id,
                "affected_count": len(agents_to_freeze),
                "affected_agents": sorted(agents_to_freeze),
                "causal_trace": list(causal_trace or []),
            },
        )

    def clear_halt(
        self,
        halt_id: str,
        state: EnvState,
    ) -> LeverEffect:
        """Clear a hardware halt condition, allowing safe resume.

        Called when the hardware root of trust signals the condition is
        resolved. Agents that were frozen by this halt are unfrozen and
        can resume from their last admissible checkpoint.

        **Checkpoint restore contract**: The returned ``LeverEffect.details``
        contains a ``resume_details`` dict keyed by agent_id.  Each entry
        provides ``checkpoint_epoch``, ``checkpoint_step``, and
        ``recovery_steps_taken``.  The orchestrator should call
        :meth:`get_preserved_task_state` for each resumed agent and
        reinject the snapshot into the agent's next turn context.  If no
        task state was preserved (empty dict), the agent starts fresh
        from the recorded checkpoint coordinates.

        Args:
            halt_id: ID of the halt to clear.
            state: Current environment state.

        Returns:
            LeverEffect with agents to unfreeze.
        """
        if not self.config.hardware_trust_enabled:
            return LeverEffect(lever_name=self.name)

        record = self._halt_records.get(halt_id)
        if record is None or record.cleared:
            return LeverEffect(lever_name=self.name)

        record.cleared = True
        record.cleared_at = datetime.now(tz=timezone.utc)

        agents_to_unfreeze: Set[str] = set()
        resume_details: Dict[str, Any] = {}

        for agent_id in record.affected_agents:
            recovery = self._recovery_states.get(agent_id)
            if recovery and recovery.halt_id == halt_id:
                agents_to_unfreeze.add(agent_id)
                resume_details[agent_id] = {
                    "checkpoint_epoch": recovery.checkpoint_epoch,
                    "checkpoint_step": recovery.checkpoint_step,
                    "recovery_steps_taken": recovery.recovery_steps_taken,
                }
                # Reset to normal mode
                recovery.mode = RecoveryMode.NORMAL
                recovery.halt_id = None

        if halt_id in self._active_halt_ids:
            self._active_halt_ids.remove(halt_id)

        # Emit HARDWARE_CONDITION_CLEARED event
        self._emit(
            EventType.HARDWARE_CONDITION_CLEARED,
            payload={
                "halt_id": halt_id,
                "resumed_agents": sorted(agents_to_unfreeze),
            },
            epoch=state.current_epoch,
            step=state.current_step,
        )

        return LeverEffect(
            agents_to_unfreeze=agents_to_unfreeze,
            lever_name=self.name,
            details={
                "halt_id": halt_id,
                "cleared": True,
                "resumed_count": len(agents_to_unfreeze),
                "resume_details": resume_details,
            },
        )

    # ------------------------------------------------------------------
    # GovernanceLever hooks
    # ------------------------------------------------------------------

    def on_step(self, state: EnvState, step: int) -> LeverEffect:
        """Track recovery steps for constrained agents."""
        if not self.config.hardware_trust_enabled:
            return LeverEffect(lever_name=self.name)
        if not self._active_halt_ids:
            return LeverEffect(lever_name=self.name)

        agents_to_freeze: Set[str] = set()
        for agent_id, recovery in self._recovery_states.items():
            if recovery.mode == RecoveryMode.CONSTRAINED:
                recovery.recovery_steps_taken += 1
                # If recovery budget exhausted, escalate to full freeze
                if (
                    recovery.recovery_steps_taken
                    >= self.config.hardware_trust_recovery_max_steps
                ):
                    recovery.mode = RecoveryMode.FROZEN
                    agents_to_freeze.add(agent_id)

        if not agents_to_freeze:
            return LeverEffect(lever_name=self.name)

        return LeverEffect(
            agents_to_freeze=agents_to_freeze,
            lever_name=self.name,
            details={
                "escalated_to_frozen": sorted(agents_to_freeze),
                "reason": "recovery_budget_exhausted",
            },
        )

    def can_agent_act(self, agent_id: str, state: EnvState) -> bool:
        """Block agents that are in FROZEN recovery mode.

        Agents in CONSTRAINED mode can still act (with action filtering),
        but agents in FROZEN mode cannot act at all.
        """
        if not self.config.hardware_trust_enabled:
            return True
        recovery = self._recovery_states.get(agent_id)
        if recovery is None:
            return True
        return recovery.mode != RecoveryMode.FROZEN

    def is_action_allowed(self, agent_id: str, action_type: str) -> bool:
        """Check if a specific action is allowed for an agent.

        In CONSTRAINED recovery mode, irreversible actions are blocked.
        In NORMAL mode, all actions are allowed.

        Args:
            agent_id: Agent attempting the action.
            action_type: The action type string (e.g. "propose_interaction").

        Returns:
            True if the action is permitted.
        """
        if not self.config.hardware_trust_enabled:
            return True
        recovery = self._recovery_states.get(agent_id)
        if recovery is None or recovery.mode == RecoveryMode.NORMAL:
            return True
        if recovery.mode == RecoveryMode.FROZEN:
            return False
        # CONSTRAINED: only allow explicitly reversible actions (fail-safe)
        return action_type in REVERSIBLE_ACTIONS

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def _compute_propagation_set(
        self,
        initial: Set[str],
        state: EnvState,
    ) -> Set[str]:
        """BFS over the dependency graph to find all transitive dependents."""
        visited: Set[str] = set(initial)
        queue: deque[str] = deque(initial)
        while queue:
            agent_id = queue.popleft()
            for dep in self._dependency_graph.get(agent_id, set()):
                if dep not in visited and dep in state.agents:
                    visited.add(dep)
                    queue.append(dep)
        return visited - initial  # Only the newly discovered dependents

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit(
        self,
        event_type: EventType,
        agent_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Emit a hardware trust event to the internal log."""
        event = Event(
            event_type=event_type,
            agent_id=agent_id,
            payload=payload or {},
            epoch=epoch,
            step=step,
        )
        self._event_log.append(event)

    def drain_events(self) -> List[Event]:
        """Return and clear accumulated events for external consumption."""
        events = list(self._event_log)
        self._event_log.clear()
        return events

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    @property
    def is_halted(self) -> bool:
        """Return True if there is at least one active (uncleared) halt."""
        return len(self._active_halt_ids) > 0

    def get_active_halt(self) -> Optional[HaltRecord]:
        """Return the most recent active halt record, if any."""
        if not self._active_halt_ids:
            return None
        # Return the latest active halt (by insertion order of the id list;
        # fall back to iterating halt records for the most recent timestamp).
        latest: Optional[HaltRecord] = None
        for halt_id in self._active_halt_ids:
            record = self._halt_records.get(halt_id)
            if record is not None:
                if latest is None or record.timestamp > latest.timestamp:
                    latest = record
        return latest

    def get_active_halts(self) -> List[HaltRecord]:
        """Return all active (uncleared) halt records."""
        return [
            self._halt_records[hid]
            for hid in self._active_halt_ids
            if hid in self._halt_records
        ]

    def get_recovery_state(self, agent_id: str) -> Optional[AgentRecoveryState]:
        """Get the recovery state for an agent."""
        return self._recovery_states.get(agent_id)

    def get_halt_record(self, halt_id: str) -> Optional[HaltRecord]:
        """Get a specific halt record."""
        return self._halt_records.get(halt_id)

    def get_all_halt_records(self) -> List[HaltRecord]:
        """Return all halt records (including cleared ones)."""
        return list(self._halt_records.values())

    def get_constrained_agents(self) -> Set[str]:
        """Return agents currently in constrained recovery mode."""
        return {
            agent_id
            for agent_id, recovery in self._recovery_states.items()
            if recovery.mode == RecoveryMode.CONSTRAINED
        }

    def get_frozen_agents(self) -> Set[str]:
        """Return agents currently in frozen recovery mode."""
        return {
            agent_id
            for agent_id, recovery in self._recovery_states.items()
            if recovery.mode == RecoveryMode.FROZEN
        }

    def preserve_task_state(
        self, agent_id: str, task_state: Dict[str, Any]
    ) -> None:
        """Preserve task state for an agent in recovery.

        Called by the orchestration layer to snapshot in-progress work
        before the agent is fully frozen.
        """
        recovery = self._recovery_states.get(agent_id)
        if recovery is not None:
            recovery.task_state_preserved = copy.deepcopy(task_state)

    def get_preserved_task_state(self, agent_id: str) -> Dict[str, Any]:
        """Retrieve preserved task state for safe-resume."""
        recovery = self._recovery_states.get(agent_id)
        if recovery is None:
            return {}
        return copy.deepcopy(recovery.task_state_preserved)
