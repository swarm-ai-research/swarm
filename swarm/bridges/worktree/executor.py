"""SandboxExecutor — command execution with boundary enforcement.

Integrates with PolicyEngine, FlowTracker, and LeakageDetector to
enforce all SWARM boundary invariants on every command execution.
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

from swarm.boundaries.information_flow import (
    FlowDirection,
    FlowTracker,
    FlowType,
    InformationFlow,
)
from swarm.boundaries.leakage import LeakageDetector
from swarm.boundaries.policies import PolicyEngine
from swarm.bridges.worktree.config import WorktreeConfig
from swarm.bridges.worktree.events import WorktreeEvent, WorktreeEventType
from swarm.bridges.worktree.policy import WorktreePolicy
from swarm.bridges.worktree.sandbox import SandboxManager
from swarm.bridges.worktree.sandbox_launch import detect_backend, wrap_command

logger = logging.getLogger(__name__)

_LEAKAGE_BLOCK_THRESHOLD = 0.7


@dataclass
class CommandResult:
    """Result of a sandbox command execution."""

    sandbox_id: str
    agent_id: str
    command: List[str]
    allowed: bool
    deny_reason: str = ""
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    leakage_blocked: bool = False
    leakage_details: List[str] = field(default_factory=list)
    resource_violation: Optional[str] = None
    duration_seconds: float = 0.0
    isolation: str = "none"


class SandboxExecutor:
    """Executes commands in sandbox worktrees with full boundary enforcement.

    Integration points:
    - WorktreePolicy: command allowlisting + risk gating
    - PolicyEngine: boundary crossing evaluation
    - FlowTracker: INBOUND (command) + OUTBOUND (response) per execution
    - LeakageDetector: scans all stdout/stderr, blocks at severity >= 0.7
    - SandboxManager: resource limit checks pre/post execution

    Security invariants:
    - NO shell=True ever
    - COMMAND-ALLOWLIST enforced before subprocess
    - LEAKAGE-BLOCK on all outbound content
    - AUDIT-TRAIL via events
    """

    def __init__(
        self,
        config: WorktreeConfig,
        sandbox_manager: SandboxManager,
        worktree_policy: WorktreePolicy,
        policy_engine: Optional[PolicyEngine] = None,
        flow_tracker: Optional[FlowTracker] = None,
        leakage_detector: Optional[LeakageDetector] = None,
    ) -> None:
        self._config = config
        self._sandbox_mgr = sandbox_manager
        self._wt_policy = worktree_policy
        self._policy_engine = policy_engine or PolicyEngine()
        self._flow_tracker = flow_tracker or FlowTracker()
        self._leakage_detector = leakage_detector or LeakageDetector()

    def execute(
        self,
        sandbox_id: str,
        agent_id: str,
        command: List[str],
    ) -> tuple[CommandResult, List[WorktreeEvent]]:
        """Execute a command in a sandbox with full boundary enforcement.

        Args:
            sandbox_id: Target sandbox.
            agent_id: Agent requesting execution.
            command: Command as a list of arguments (no shell expansion).

        Returns:
            (CommandResult, list of WorktreeEvents produced).
        """
        events: List[WorktreeEvent] = []

        # Event: command requested
        events.append(
            WorktreeEvent(
                event_type=WorktreeEventType.COMMAND_REQUESTED,
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                payload={"command": command},
            )
        )

        # --- Pre-execution checks ---

        # 1. Command allowlist via WorktreePolicy
        decision = self._wt_policy.evaluate_command(agent_id, command)
        if not decision.allowed:
            events.append(
                WorktreeEvent(
                    event_type=WorktreeEventType.COMMAND_DENIED,
                    agent_id=agent_id,
                    sandbox_id=sandbox_id,
                    payload={
                        "command": command,
                        "reason": decision.reason,
                    },
                )
            )
            return (
                CommandResult(
                    sandbox_id=sandbox_id,
                    agent_id=agent_id,
                    command=command,
                    allowed=False,
                    deny_reason=decision.reason,
                ),
                events,
            )

        # 2. PolicyEngine boundary check (inbound command)
        cmd_str = " ".join(command)
        crossing = self._policy_engine.evaluate(
            agent_id=agent_id,
            direction="inbound",
            flow_type="command",
            content=cmd_str,
        )
        if not crossing.allowed:
            events.append(
                WorktreeEvent(
                    event_type=WorktreeEventType.COMMAND_DENIED,
                    agent_id=agent_id,
                    sandbox_id=sandbox_id,
                    payload={
                        "command": command,
                        "reason": crossing.reason,
                        "policy": crossing.policy_name,
                    },
                )
            )
            return (
                CommandResult(
                    sandbox_id=sandbox_id,
                    agent_id=agent_id,
                    command=command,
                    allowed=False,
                    deny_reason=crossing.reason,
                ),
                events,
            )

        # 3. Resource limits pre-check
        violation = self._sandbox_mgr.check_resource_limits(sandbox_id)
        if violation:
            events.append(
                WorktreeEvent(
                    event_type=WorktreeEventType.RESOURCE_LIMIT_HIT,
                    agent_id=agent_id,
                    sandbox_id=sandbox_id,
                    payload={"violation": violation},
                )
            )
            return (
                CommandResult(
                    sandbox_id=sandbox_id,
                    agent_id=agent_id,
                    command=command,
                    allowed=False,
                    deny_reason=f"Resource limit: {violation}",
                    resource_violation=violation,
                ),
                events,
            )

        # --- Resolve OS-level isolation before admitting the command ---
        sandbox_path = self._sandbox_mgr.get_sandbox_path(sandbox_id)
        if sandbox_path is None:
            return (
                CommandResult(
                    sandbox_id=sandbox_id,
                    agent_id=agent_id,
                    command=command,
                    allowed=False,
                    deny_reason=f"Sandbox {sandbox_id} not found",
                ),
                events,
            )

        isolation = "none"
        exec_argv = command
        if self._config.os_isolation_enabled:
            backend = detect_backend()
            if backend == "none" and self._config.require_os_isolation:
                reason = "OS isolation required but no backend available"
                events.append(
                    WorktreeEvent(
                        event_type=WorktreeEventType.COMMAND_DENIED,
                        agent_id=agent_id,
                        sandbox_id=sandbox_id,
                        payload={"command": command, "reason": reason},
                    )
                )
                return (
                    CommandResult(
                        sandbox_id=sandbox_id,
                        agent_id=agent_id,
                        command=command,
                        allowed=False,
                        deny_reason=reason,
                    ),
                    events,
                )
            if backend != "none":
                exec_argv = wrap_command(
                    command,
                    sandbox_path=sandbox_path,
                    backend=backend,
                    allow_network=self._config.os_isolation_allow_network,
                )
                isolation = backend

        # Record inbound flow (command into sandbox)
        inbound_flow = InformationFlow.create(
            direction=FlowDirection.INBOUND,
            flow_type=FlowType.COMMAND,
            source_id=agent_id,
            destination_id=f"sandbox:{sandbox_id}",
            content=cmd_str,
            sensitivity_score=decision.risk_score,
        )
        self._flow_tracker.record_flow(inbound_flow)

        events.append(
            WorktreeEvent(
                event_type=WorktreeEventType.COMMAND_ALLOWED,
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                payload={"command": command},
            )
        )

        # --- Execution ---
        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                exec_argv,
                capture_output=True,
                text=True,
                timeout=self._config.command_timeout_seconds,
                cwd=sandbox_path,
            )
            return_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = -1
            stdout = ""
            stderr = "Command timed out"
            events.append(
                WorktreeEvent(
                    event_type=WorktreeEventType.COMMAND_TIMEOUT,
                    agent_id=agent_id,
                    sandbox_id=sandbox_id,
                    payload={
                        "command": command,
                        "timeout": self._config.command_timeout_seconds,
                    },
                )
            )
        except (FileNotFoundError, OSError) as exc:
            return_code = -1
            stdout = ""
            stderr = str(exc)

        duration = time.monotonic() - start

        # Touch sandbox to update last-active time
        self._sandbox_mgr.touch_sandbox(sandbox_id)

        # --- Post-execution checks ---

        # Scan outbound content for leakage
        outbound_content = stdout + "\n" + stderr
        leakage_blocked = False
        leakage_details: List[str] = []

        leak_events = self._leakage_detector.scan(
            content=outbound_content,
            agent_id=agent_id,
            destination_id="external",
            flow_id=inbound_flow.flow_id,
        )

        for le in leak_events:
            if le.severity >= _LEAKAGE_BLOCK_THRESHOLD:
                leakage_blocked = True
                leakage_details.append(
                    f"{le.leakage_type.value}: {le.description} "
                    f"(severity={le.severity:.2f})"
                )
                le.blocked = True

        if leakage_blocked:
            events.append(
                WorktreeEvent(
                    event_type=WorktreeEventType.LEAKAGE_DETECTED,
                    agent_id=agent_id,
                    sandbox_id=sandbox_id,
                    payload={
                        "command": command,
                        "leakage_details": leakage_details,
                        "action": "output_blocked",
                    },
                )
            )
            # Redact the output
            stdout = "[REDACTED: leakage detected]"
            stderr = "[REDACTED: leakage detected]"

        # Record outbound flow (response from sandbox)
        outbound_flow = InformationFlow.create(
            direction=FlowDirection.OUTBOUND,
            flow_type=FlowType.RESPONSE,
            source_id=f"sandbox:{sandbox_id}",
            destination_id=agent_id,
            content=outbound_content if not leakage_blocked else "[REDACTED]",
            sensitivity_score=max(
                (le.severity for le in leak_events), default=0.0
            ),
        )
        if leakage_blocked:
            outbound_flow.blocked = True
            outbound_flow.block_reason = "; ".join(leakage_details)
        self._flow_tracker.record_flow(outbound_flow)

        # Resource limits post-check
        post_violation = self._sandbox_mgr.check_resource_limits(sandbox_id)
        if post_violation:
            events.append(
                WorktreeEvent(
                    event_type=WorktreeEventType.RESOURCE_LIMIT_HIT,
                    agent_id=agent_id,
                    sandbox_id=sandbox_id,
                    payload={"violation": post_violation, "phase": "post"},
                )
            )

        events.append(
            WorktreeEvent(
                event_type=WorktreeEventType.COMMAND_COMPLETED,
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                payload={
                    "command": command,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "leakage_blocked": leakage_blocked,
                    "duration_seconds": round(duration, 3),
                    "isolation": isolation,
                },
            )
        )

        return (
            CommandResult(
                sandbox_id=sandbox_id,
                agent_id=agent_id,
                command=command,
                allowed=True,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                leakage_blocked=leakage_blocked,
                leakage_details=leakage_details,
                resource_violation=post_violation,
                duration_seconds=round(duration, 3),
                isolation=isolation,
            ),
            events,
        )
