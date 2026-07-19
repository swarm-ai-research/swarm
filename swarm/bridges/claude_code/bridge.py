"""Main bridge connecting Claude Code controller to SWARM.

ClaudeCodeBridge is the central adapter that:
1. Receives events from the controller (via ClaudeCodeClient)
2. Converts them to SWARM's SoftInteraction + ProxyObservables
3. Routes governance decisions through GovernancePolicy
4. Feeds results into SWARM's logging and metrics pipeline
"""

import logging
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional

from swarm.bridges.claude_code.client import ClaudeCodeClient, ClientConfig
from swarm.bridges.claude_code.events import (
    BridgeEvent,
    BridgeEventType,
    MessageEvent,
    PermissionRequest,
    PlanApprovalRequest,
)
from swarm.bridges.claude_code.policy import (
    GovernancePolicy,
    PolicyDecision,
)
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.governance.config import GovernanceConfig
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


# Map controller tool names to risk-relevant SWARM observables
TOOL_RISK_MAP: Dict[str, float] = {
    "Bash": 0.8,
    "Write": 0.5,
    "Edit": 0.3,
    "NotebookEdit": 0.4,
    "WebFetch": 0.3,
    "Read": 0.1,
    "Glob": 0.1,
    "Grep": 0.1,
    "WebSearch": 0.1,
}


@dataclass
class BridgeConfig:
    """Configuration for the Claude Code bridge."""

    client_config: ClientConfig = field(default_factory=ClientConfig)
    governance_config: GovernanceConfig = field(default_factory=GovernanceConfig)
    proxy_sigmoid_k: float = 2.0
    event_poll_interval: float = 1.0
    auto_respond_governance: bool = True
    tool_allowlist: Optional[Dict[str, List[str]]] = None
    max_interactions: int = 50000
    max_bridge_events: int = 50000


class ClaudeCodeBridge:
    """Bridge between Claude Code controller and SWARM framework.

    Lifecycle:
        bridge = ClaudeCodeBridge(config)
        bridge.spawn_agent("worker_1", system_prompt="...")
        result = bridge.dispatch_task("worker_1", "Review this code")
        interactions = bridge.get_interactions()
        bridge.shutdown()

    The bridge translates controller events into SWARM's data model:
    - Messages become SoftInteraction records
    - Tool usage maps to ProxyObservables (misuse flags, progress)
    - Plan/permission events feed the governance policy
    - All events are logged to SWARM's append-only event log
    """

    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        event_log: Optional[EventLog] = None,
    ):
        self._config = config or BridgeConfig()
        self._client = ClaudeCodeClient(self._config.client_config)
        self._policy = GovernancePolicy(
            governance_config=self._config.governance_config,
            tool_allowlist=self._config.tool_allowlist,
        )
        self._proxy = ProxyComputer(sigmoid_k=self._config.proxy_sigmoid_k)
        self._event_log = event_log
        self._interactions: List[SoftInteraction] = []
        self._bridge_events: List[BridgeEvent] = []
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._last_event_id: Optional[str] = None

        self._session_initialized = False

        # Register internal event handlers
        self._client.on(
            BridgeEventType.PLAN_APPROVAL_REQUEST,
            self._handle_plan_request,
        )
        self._client.on(
            BridgeEventType.PERMISSION_REQUEST,
            self._handle_permission_request,
        )

    # --- Session lifecycle ---

    def init_session(
        self,
        team_name: str = "swarm",
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Initialize the controller session.

        Must be called before spawning agents.

        Args:
            team_name: Name for the agent team
            cwd: Working directory for agents

        Returns:
            Session initialization response
        """
        result = self._client.init_session(team_name=team_name, cwd=cwd)
        self._session_initialized = result.get("initialized", False)
        return result

    def ensure_session(
        self,
        team_name: str = "swarm",
        cwd: Optional[str] = None,
    ) -> None:
        """Ensure session is initialized, initializing if needed."""
        if not self._session_initialized:
            status = self._client.get_session_status()
            if not status.get("initialized"):
                self.init_session(team_name=team_name, cwd=cwd)
            else:
                self._session_initialized = True

    # --- Agent lifecycle ---

    def spawn_agent(
        self,
        agent_id: str,
        system_prompt: str = "",
        allowed_tools: Optional[List[str]] = None,
        model: str = "sonnet",
        budget_tool_calls: int = 100,
        budget_cost_usd: float = 10.0,
    ) -> Dict[str, Any]:
        """Spawn a Claude Code agent and register it in the bridge.

        Args:
            agent_id: Unique agent identifier
            system_prompt: System prompt defining behavior
            allowed_tools: Tool allowlist
            model: Model to use ("sonnet", "opus", "haiku")
            budget_tool_calls: Max tool invocations for this agent
            budget_cost_usd: Max cost budget

        Returns:
            Controller response with agent metadata
        """
        # Ensure session is initialized
        self.ensure_session()

        # Configure governance budget
        self._policy.set_agent_budget(
            agent_id,
            max_tool_calls=budget_tool_calls,
            max_cost_usd=budget_cost_usd,
        )

        # Set tool allowlist if provided
        if allowed_tools is not None:
            self._policy.tool_allowlist[agent_id] = allowed_tools

        # Track agent state
        self._agent_states[agent_id] = {
            "model": model,
            "system_prompt": system_prompt,
            "spawned_at": time.time(),
            "interactions": 0,
            "reputation": 0.0,
        }

        result = self._client.spawn_agent(
            agent_id=agent_id,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            model=model,
        )

        # Log spawn event
        self._record_bridge_event(
            BridgeEvent(
                event_type=BridgeEventType.AGENT_SPAWNED,
                agent_id=agent_id,
                payload={"model": model, "tools": allowed_tools or []},
            )
        )

        return result

    def shutdown_agent(self, agent_id: str) -> None:
        """Shut down a single agent."""
        self._client.shutdown_agent(agent_id)
        self._record_bridge_event(
            BridgeEvent(
                event_type=BridgeEventType.AGENT_SHUTDOWN,
                agent_id=agent_id,
            )
        )

    def shutdown(self) -> None:
        """Shut down all agents managed by this bridge."""
        for agent_id in list(self._agent_states.keys()):
            try:
                self.shutdown_agent(agent_id)
            except Exception:
                logger.exception("Error shutting down agent %s", agent_id)

    # --- Task dispatch ---

    def dispatch_task(
        self,
        agent_id: str,
        prompt: str,
        counterparty_id: str = "swarm_orchestrator",
        interaction_type: InteractionType = InteractionType.COLLABORATION,
    ) -> SoftInteraction:
        """Dispatch a task to an agent and score the result.

        This is the primary interface: send a prompt, get back a
        SoftInteraction with computed p and observables.

        Args:
            agent_id: Target agent ID
            prompt: The task/prompt to send
            counterparty_id: Who is "requesting" this work
            interaction_type: Type of interaction

        Returns:
            SoftInteraction with observables and soft label p
        """
        start_time = time.monotonic()

        # Send prompt and wait for the real response via the event stream.
        # ask() returns None on timeout; failing loudly here is deliberate —
        # logging a fabricated response would corrupt the observational
        # record (see beads 2331).
        message = self._client.ask(agent_id, prompt)
        if message is None:
            raise TimeoutError(
                f"No response from agent {agent_id!r} within the ask() "
                "timeout; refusing to record a fabricated interaction"
            )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Extract observables from the response
        observables = self._extract_observables(message, agent_id)

        # Compute proxy labels
        v_hat, p = self._proxy.compute_labels(observables)

        # Build interaction record
        agent_state = self._agent_states.get(agent_id, {})
        interaction = SoftInteraction(
            initiator=counterparty_id,
            counterparty=agent_id,
            interaction_type=interaction_type,
            accepted=True,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "claude_code",
                "model": agent_state.get("model", "unknown"),
                "duration_ms": elapsed_ms,
                "token_count": message.token_count,
                "cost_usd": message.cost_usd,
                "tool_calls": len(message.tool_calls),
                "prompt_len": len(prompt),
                "response_len": len(message.content),
                "prompt_preview": prompt[:200],
                "response_preview": message.content[:200],
            },
        )

        self._record_interaction(interaction)

        # Update agent state
        if agent_id in self._agent_states:
            self._agent_states[agent_id]["interactions"] += 1

        # Log to SWARM event log
        self._log_interaction(interaction)

        return interaction

    def dispatch_inbox_task(
        self,
        subject: str,
        description: str,
        owner: str,
        counterparty_id: str = "swarm_orchestrator",
        poll_interval: float = 2.0,
        max_wait: float = 300.0,
    ) -> SoftInteraction:
        """Dispatch a longer-horizon task via the inbox protocol.

        Uses createTask + waitForTask for multi-step work.

        Args:
            subject: Task subject line
            description: Full task description
            owner: Agent ID that should work on this
            counterparty_id: Who is requesting this work
            poll_interval: Seconds between polls
            max_wait: Maximum wait time

        Returns:
            SoftInteraction with observables and soft label p
        """
        start_time = time.monotonic()

        # Create and wait for task
        task = self._client.create_task(
            subject=subject,
            description=description,
            owner=owner,
        )

        completed = self._client.wait_for_task(
            task_id=task.task_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Build observables from task result
        success = completed.status == "completed"
        observables = ProxyObservables(
            task_progress_delta=0.8 if success else -0.5,
            rework_count=0,
            verifier_rejections=0 if success else 1,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.5 if success else -0.3,
        )

        v_hat, p = self._proxy.compute_labels(observables)

        interaction = SoftInteraction(
            initiator=counterparty_id,
            counterparty=owner,
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "claude_code",
                "task_id": completed.task_id,
                "task_subject": subject,
                "task_status": completed.status,
                "duration_ms": elapsed_ms,
                "tools_used": completed.tools_used,
            },
        )

        self._record_interaction(interaction)
        self._log_interaction(interaction)
        return interaction

    # --- Observable extraction ---

    def _extract_observables(
        self,
        message: MessageEvent,
        agent_id: str,
    ) -> ProxyObservables:
        """Extract SWARM-compatible observables from a controller message.

        Maps Claude Code tool usage and response quality to SWARM's
        proxy observable format.

        Args:
            message: The response message from the controller
            agent_id: The responding agent's ID

        Returns:
            ProxyObservables ready for ProxyComputer
        """
        tool_calls = message.tool_calls

        # Task progress: positive if response has content, scaled by length
        content_len = len(message.content)
        progress = min(1.0, content_len / 500.0) if content_len > 0 else -0.3

        # Tool misuse: count high-risk tool calls
        misuse_flags = 0
        for tc in tool_calls:
            tool_name = tc.get("tool", tc.get("name", ""))
            risk = TOOL_RISK_MAP.get(tool_name, 0.2)
            if risk > 0.5:
                # Check if the agent had permission
                allowlist = self._policy.tool_allowlist.get(agent_id)
                if allowlist is not None and tool_name not in allowlist:
                    misuse_flags += 1

        # Engagement: based on response quality signals
        has_content = content_len > 50
        has_tool_use = len(tool_calls) > 0
        engagement = 0.0
        if has_content:
            engagement += 0.4
        if has_tool_use:
            engagement += 0.3
        if message.token_count > 100:
            engagement += 0.3

        return ProxyObservables(
            task_progress_delta=progress,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=misuse_flags,
            counterparty_engagement_delta=min(1.0, engagement),
        )

    # --- Governance event handling ---

    def _handle_plan_request(self, event: BridgeEvent) -> None:
        """Handle a plan approval request from the controller."""
        if not self._config.auto_respond_governance:
            return

        request = PlanApprovalRequest.from_dict(event.payload)
        agent_rep = self._agent_states.get(request.agent_id, {}).get("reputation", 0.0)

        result = self._policy.evaluate_plan(request, agent_rep)

        approved = result.decision in (
            PolicyDecision.APPROVE,
            PolicyDecision.REQUIRE_STAKE,
        )

        self._client.respond_to_plan(
            request_id=request.request_id,
            approved=approved,
            reason=result.reason,
        )

        logger.info(
            "Plan %s for agent %s: %s (%s)",
            "approved" if approved else "denied",
            request.agent_id,
            result.decision.value,
            result.reason,
        )

    def _handle_permission_request(self, event: BridgeEvent) -> None:
        """Handle a tool permission request from the controller."""
        if not self._config.auto_respond_governance:
            return

        request = PermissionRequest.from_dict(event.payload)
        agent_rep = self._agent_states.get(request.agent_id, {}).get("reputation", 0.0)

        result = self._policy.evaluate_permission(request, agent_rep)

        granted = result.decision in (
            PolicyDecision.APPROVE,
            PolicyDecision.REQUIRE_STAKE,
        )

        self._client.respond_to_permission(
            request_id=request.request_id,
            granted=granted,
            reason=result.reason,
        )

    # --- Event logging ---

    def _record_bridge_event(self, event: BridgeEvent) -> None:
        """Record a bridge event to the internal log."""
        if len(self._bridge_events) >= self._config.max_bridge_events:
            self._bridge_events = self._bridge_events[
                -self._config.max_bridge_events // 2 :
            ]
        self._bridge_events.append(event)
        self._client._dispatch_event(event)

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        """Record an interaction, enforcing the configured cap."""
        if len(self._interactions) >= self._config.max_interactions:
            self._interactions = self._interactions[
                -self._config.max_interactions // 2 :
            ]
        self._interactions.append(interaction)

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        """Log an interaction to SWARM's event log."""
        if self._event_log is None:
            return

        # Avoid persisting potentially sensitive prompt/response content by default.
        # Keep lengths + stable hashes for debugging, while dropping raw previews.
        metadata = dict(interaction.metadata or {})
        prompt_preview = metadata.pop("prompt_preview", None)
        response_preview = metadata.pop("response_preview", None)
        if isinstance(prompt_preview, str) and prompt_preview:
            metadata["prompt_preview_sha256"] = sha256(
                prompt_preview.encode("utf-8")
            ).hexdigest()
        if isinstance(response_preview, str) and response_preview:
            metadata["response_preview_sha256"] = sha256(
                response_preview.encode("utf-8")
            ).hexdigest()

        event = Event(
            event_type=EventType.INTERACTION_COMPLETED,
            interaction_id=interaction.interaction_id,
            initiator_id=interaction.initiator,
            counterparty_id=interaction.counterparty,
            payload={
                "accepted": interaction.accepted,
                "v_hat": interaction.v_hat,
                "p": interaction.p,
                "bridge": "claude_code",
                "metadata": metadata,
            },
        )
        self._event_log.append(event)

    # --- Polling loop ---

    def poll_and_process(self) -> List[BridgeEvent]:
        """Poll the controller for new events and process them.

        Returns:
            List of new events received
        """
        events = self._client.poll_events(since_event_id=self._last_event_id)
        for event in events:
            self._record_bridge_event(event)
            self._last_event_id = event.event_id
        return events

    # --- Accessors ---

    def get_interactions(self) -> List[SoftInteraction]:
        """Get all interactions recorded by this bridge."""
        return list(self._interactions)

    def get_bridge_events(self) -> List[BridgeEvent]:
        """Get all bridge events."""
        return list(self._bridge_events)

    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get the tracked state for an agent."""
        return self._agent_states.get(agent_id, {})

    def update_agent_reputation(self, agent_id: str, reputation: float) -> None:
        """Update the reputation score for an agent.

        Called by SWARM's orchestrator after computing payoffs.
        """
        if agent_id in self._agent_states:
            self._agent_states[agent_id]["reputation"] = reputation

    @property
    def policy(self) -> GovernancePolicy:
        """Access the governance policy."""
        return self._policy

    @property
    def client(self) -> ClaudeCodeClient:
        """Access the underlying client."""
        return self._client
