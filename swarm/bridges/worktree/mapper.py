"""WorktreeMapper — convert sandbox signals to SoftInteractions.

Translates command execution results and git diff stats into SWARM's
ProxyObservables, then uses ProxyComputer to derive v_hat and p.
"""

import logging
from typing import Dict, Optional

from swarm.bridges.worktree.executor import CommandResult
from swarm.bridges.worktree.sandbox import _run_git
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class WorktreeMapper:
    """Maps sandbox command results and git stats to SoftInteractions.

    Observable mapping:

    ====================================== ============================= ==========================
    Sandbox Signal                         ProxyObservable               Formula
    ====================================== ============================= ==========================
    git diff --stat files changed          task_progress_delta           min(1.0, files / 10)
    Commands denied by policy              tool_misuse_flags             direct count
    pytest exit code != 0                  verifier_rejections           direct count
    git reflog amend/reset operations      rework_count                  direct count
    Successful commands / total            counterparty_engagement_delta ratio - 0.5 scaled
    ====================================== ============================= ==========================
    """

    def __init__(self, proxy: Optional[ProxyComputer] = None) -> None:
        self._proxy = proxy or ProxyComputer()

    def map_command_result(
        self,
        result: CommandResult,
        sandbox_path: Optional[str] = None,
        agent_stats: Optional[Dict[str, int]] = None,
    ) -> SoftInteraction:
        """Map a command execution result to a SoftInteraction.

        Args:
            result: The command execution result.
            sandbox_path: Path to the sandbox (for git diff stats).
            agent_stats: Running counts for the agent:
                - denied_count: commands denied by policy
                - test_failures: pytest failures
                - rework_ops: amend/reset operations
                - total_commands: total commands dispatched
                - successful_commands: commands with return_code == 0

        Returns:
            A SoftInteraction with computed v_hat and p.
        """
        stats = agent_stats or {}

        # task_progress_delta from git diff if sandbox available
        progress = 0.0
        if sandbox_path:
            progress = self._get_diff_progress(sandbox_path)

        # tool_misuse_flags from denied commands
        misuse = stats.get("denied_count", 0)
        if not result.allowed:
            misuse += 1

        # verifier_rejections from test failures
        rejections = stats.get("test_failures", 0)
        if (
            result.allowed
            and result.command
            and result.command[0] == "pytest"
            and result.return_code != 0
        ):
            rejections += 1

        # rework_count from reflog
        rework = stats.get("rework_ops", 0)
        if sandbox_path:
            rework += self._get_rework_count(sandbox_path)

        # engagement from success ratio
        total = stats.get("total_commands", 1)
        successful = stats.get("successful_commands", 0)
        if result.allowed and result.return_code == 0:
            successful += 1
        total += 1
        engagement = (successful / total) * 2.0 - 1.0  # scale to [-1, 1]

        observables = ProxyObservables(
            task_progress_delta=progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator="worktree_orchestrator",
            counterparty=result.agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=result.allowed,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "worktree",
                "sandbox_id": result.sandbox_id,
                "command": " ".join(result.command),
                "return_code": result.return_code,
                "timed_out": result.timed_out,
                "leakage_blocked": result.leakage_blocked,
            },
        )

    def map_sandbox_observation(
        self,
        sandbox_id: str,
        agent_id: str,
        sandbox_path: str,
        agent_stats: Optional[Dict[str, int]] = None,
    ) -> SoftInteraction:
        """Map a periodic sandbox observation to a SoftInteraction.

        Used during poll() to observe sandbox state without a specific command.
        Observations are passive status checks, not acceptance/rejection outcomes.
        """
        stats = agent_stats or {}
        progress = self._get_diff_progress(sandbox_path)
        rework = self._get_rework_count(sandbox_path)

        total = max(stats.get("total_commands", 0), 1)
        successful = stats.get("successful_commands", 0)
        engagement = (successful / total) * 2.0 - 1.0

        observables = ProxyObservables(
            task_progress_delta=progress,
            rework_count=rework,
            verifier_rejections=stats.get("test_failures", 0),
            tool_misuse_flags=stats.get("denied_count", 0),
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator="worktree_orchestrator",
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=False,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "worktree",
                "sandbox_id": sandbox_id,
                "observation": "poll",
            },
        )

    @staticmethod
    def _get_diff_progress(sandbox_path: str) -> float:
        """Get task progress from git diff --stat."""
        result = _run_git(["diff", "--stat", "HEAD"], cwd=sandbox_path)
        if not result or result.returncode != 0:
            return 0.0
        # Count changed files from diff --stat output
        lines = result.stdout.strip().splitlines()
        if not lines:
            return 0.0
        # Last line is summary: "N files changed, ..."
        # Count non-summary lines as changed files
        file_lines = [ln for ln in lines[:-1] if "|" in ln] if len(lines) > 1 else []
        files_changed = len(file_lines)
        return min(1.0, files_changed / 10.0)

    @staticmethod
    def _get_rework_count(sandbox_path: str) -> int:
        """Count amend/reset operations from git reflog."""
        result = _run_git(
            ["reflog", "--format=%gs", "HEAD"], cwd=sandbox_path
        )
        if not result or result.returncode != 0:
            return 0
        count = 0
        for line in result.stdout.strip().splitlines():
            lower = line.lower()
            if "reset" in lower or "amend" in lower:
                count += 1
        return count
