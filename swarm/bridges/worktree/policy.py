"""Command evaluation policy for worktree sandboxes.

Provides command allowlisting, risk scoring, and env injection rules.
SSH/SCP and network-reaching git operations are unconditionally blocked.
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from swarm.bridges.worktree.config import WorktreeConfig

if TYPE_CHECKING:
    from swarm.agentgit.identity import DelegationChain

logger = logging.getLogger(__name__)

# Risk scores for known commands (0.0 = safe, 1.0 = always blocked)
COMMAND_RISK_MAP: Dict[str, float] = {
    "git": 0.2,
    "python": 0.5,
    "pytest": 0.3,
    "ruff": 0.2,
    "mypy": 0.2,
    "pip": 0.4,
    "ls": 0.1,
    "cat": 0.1,
    "head": 0.1,
    "tail": 0.1,
    "wc": 0.1,
    "diff": 0.1,
    "grep": 0.1,
    "find": 0.2,
    "echo": 0.1,
    "mkdir": 0.2,
    "touch": 0.1,
    "rm": 0.9,
    "ssh": 1.0,
    "scp": 1.0,
    "curl": 0.8,
    "wget": 0.8,
}

# Git subcommands that are unconditionally blocked (network-reaching)
_BLOCKED_GIT_SUBCOMMANDS: Set[str] = {"push", "fetch", "pull", "clone"}

# Credential-like env var patterns
_CREDENTIAL_ENV_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(password|passwd|secret|token|api.?key|private.?key)", re.IGNORECASE),
    re.compile(r"^AWS_", re.IGNORECASE),
    re.compile(r"^GH_TOKEN$|^GITHUB_TOKEN$", re.IGNORECASE),
]


@dataclass
class CommandDecision:
    """Result of a command policy evaluation."""

    allowed: bool
    reason: str
    risk_score: float = 0.0
    command: str = ""
    agent_id: str = ""


class WorktreePolicy:
    """Evaluates commands and env injection requests against policy rules.

    Security invariants:
    - SSH/SCP always blocked (NO-SSH)
    - git push/fetch/pull/clone always blocked (NO-SSH)
    - Only allowlisted commands execute (COMMAND-ALLOWLIST)
    - Only explicitly allowlisted env vars are injected
    - Credential-pattern env vars are always blocked
    """

    def __init__(self, config: WorktreeConfig) -> None:
        self._config = config
        self._command_budgets: Dict[str, int] = {}  # agent_id -> remaining commands

    def apply_delegation(
        self,
        agent_id: str,
        chain: "DelegationChain",
        *,
        expected_subject_did: str,
        context: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """Derive ``agent_id``'s command allowlist from a delegation chain.

        Binds the cryptographically delegated capabilities (enqe) to what the
        sandbox physically allows: the per-agent allowlist becomes exactly the
        commands authorized by the chain's verified permissions. A chain that
        fails verification installs an empty allowlist (deny-all) so the agent
        can run nothing until a valid delegation is supplied.

        ``expected_subject_did`` is required and binds the chain to this agent:
        a chain whose final subject is a different DID is rejected, so a valid
        delegation issued to one identity cannot be installed under another
        sandbox name. Returns ``(ok, errors)``.
        """

        # Imported lazily to avoid a hard import cycle at module load.
        from swarm.agentgit.capabilities import enforced_allowlist_for_chain

        allowlist, errors = enforced_allowlist_for_chain(
            chain, expected_subject_did=expected_subject_did, context=context
        )
        self._config.agent_command_allowlists[agent_id] = allowlist
        if errors:
            logger.warning(
                "Delegation for agent %s failed verification; denying all "
                "commands: %s",
                agent_id,
                "; ".join(errors),
            )
        return not errors, errors

    def evaluate_command(
        self,
        agent_id: str,
        command: List[str],
        reputation: float = 0.5,
    ) -> CommandDecision:
        """Evaluate whether a command should be allowed.

        Args:
            agent_id: The agent requesting execution.
            command: The command as a list of arguments.
            reputation: Agent's current reputation score [0, 1].

        Returns:
            CommandDecision indicating allow/deny.
        """
        if not command:
            return CommandDecision(
                allowed=False,
                reason="Empty command",
                command="",
                agent_id=agent_id,
            )

        binary = command[0]
        cmd_str = " ".join(command)

        # Unconditionally block SSH/SCP
        if binary in ("ssh", "scp"):
            return CommandDecision(
                allowed=False,
                reason=f"Command '{binary}' is unconditionally blocked (NO-SSH)",
                risk_score=1.0,
                command=cmd_str,
                agent_id=agent_id,
            )

        # Block network-reaching git subcommands
        if binary == "git" and len(command) > 1:
            subcommand = command[1]
            if subcommand in _BLOCKED_GIT_SUBCOMMANDS:
                return CommandDecision(
                    allowed=False,
                    reason=f"git {subcommand} is unconditionally blocked (NO-SSH)",
                    risk_score=1.0,
                    command=cmd_str,
                    agent_id=agent_id,
                )

        # Check allowlist
        allowlist = self._get_allowlist(agent_id)
        if binary not in allowlist:
            return CommandDecision(
                allowed=False,
                reason=f"Command '{binary}' not in allowlist",
                risk_score=COMMAND_RISK_MAP.get(binary, 0.5),
                command=cmd_str,
                agent_id=agent_id,
            )

        risk = COMMAND_RISK_MAP.get(binary, 0.5)
        return CommandDecision(
            allowed=True,
            reason="Command allowed by policy",
            risk_score=risk,
            command=cmd_str,
            agent_id=agent_id,
        )

    def evaluate_env_injection(
        self,
        agent_id: str,
        env_vars: Dict[str, str],
    ) -> Tuple[Dict[str, str], List[str]]:
        """Filter env vars, returning only safe allowlisted ones.

        Args:
            agent_id: The agent the env is injected for.
            env_vars: Candidate env vars to inject.

        Returns:
            (allowed_vars, blocked_reasons): The filtered env dict and
            reasons for each blocked var.
        """
        allowed: Dict[str, str] = {}
        blocked_reasons: List[str] = []

        for key, value in env_vars.items():
            # Check credential patterns first
            if self._is_credential_var(key):
                blocked_reasons.append(
                    f"{key}: matches credential pattern (always blocked)"
                )
                continue

            # Only allow vars explicitly in the allowlist
            if key in self._config.env_allowlist:
                allowed[key] = value
            else:
                blocked_reasons.append(f"{key}: not in env_allowlist")

        return allowed, blocked_reasons

    def get_env_blocklist_patterns(self) -> List[str]:
        """Return glob patterns for env files to delete from sandboxes."""
        return list(self._config.env_blocklist_patterns)

    def _get_allowlist(self, agent_id: str) -> List[str]:
        """Get the command allowlist for an agent."""
        if agent_id in self._config.agent_command_allowlists:
            return self._config.agent_command_allowlists[agent_id]
        return self._config.default_command_allowlist

    @staticmethod
    def _is_credential_var(key: str) -> bool:
        """Check if an env var name matches credential patterns."""
        for pattern in _CREDENTIAL_ENV_PATTERNS:
            if pattern.search(key):
                return True
        return False
