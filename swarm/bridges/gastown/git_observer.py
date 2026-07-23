"""GitObserver — extract PR/commit observables from git repos.

All git calls use ``subprocess.run`` with timeouts.  Errors are caught
and logged rather than raised so a single bad worktree never crashes
the bridge polling loop.
"""

import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

from swarm.bridges.gastown.events import GasTownEvent, GasTownEventType

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 10  # seconds


def _run_git(
    args: List[str],
    cwd: str,
    timeout: int = _SUBPROCESS_TIMEOUT,
) -> Optional[subprocess.CompletedProcess[str]]:
    """Run a git command, returning ``None`` on failure."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.warning("git %s failed in %s: %s", " ".join(args), cwd, exc)
        return None


class GitObserver:
    """Extracts PR/commit observables from a GasTown workspace."""

    def __init__(self, workspace_path: str) -> None:
        self._workspace_path = workspace_path

    def get_pr_stats(self, worktree: str, base: str = "main") -> dict:
        """Compute PR-related observables for a worktree/branch.

        Args:
            worktree: Path to the worktree.
            base: Base branch to compare against (default: "main").

        Returns a dict with keys:
            commit_count, files_changed, review_iterations,
            ci_failures, time_to_merge_hours.
        """
        stats: dict = {
            "commit_count": 0,
            "files_changed": 0,
            "review_iterations": 0,
            "ci_failures": 0,
            "time_to_merge_hours": None,
        }

        # Commit count since divergence from base
        result = _run_git(["rev-list", "--count", "HEAD", f"^{base}"], cwd=worktree)
        if result and result.returncode == 0:
            try:
                stats["commit_count"] = int(result.stdout.strip())
            except ValueError:
                pass

        # Files changed
        result = _run_git(
            ["diff", "--name-only", f"{base}...HEAD"], cwd=worktree
        )
        if result and result.returncode == 0:
            files = [f for f in result.stdout.strip().splitlines() if f]
            stats["files_changed"] = len(files)

        # Review iterations (force-push count via reflog)
        result = _run_git(["reflog", "--format=%gs", "HEAD"], cwd=worktree)
        if result and result.returncode == 0:
            pushes = [
                line
                for line in result.stdout.strip().splitlines()
                if "reset" in line.lower() or "amend" in line.lower()
            ]
            stats["review_iterations"] = len(pushes)

        # CI failures (convention: commits with "[ci-fail]" in message)
        result = _run_git(
            ["log", "--oneline", "--fixed-strings", "--grep=[ci-fail]", f"{base}..HEAD"],
            cwd=worktree,
        )
        if result and result.returncode == 0:
            failures = [line for line in result.stdout.strip().splitlines() if line]
            stats["ci_failures"] = len(failures)

        # Time to merge (if branch has been merged into base)
        first = _run_git(
            ["log", "--reverse", "--format=%aI", f"{base}..HEAD"],
            cwd=worktree,
        )
        merge = _run_git(
            ["log", "-1", "--merges", "--format=%aI", "HEAD"],
            cwd=worktree,
        )
        if (
            first
            and first.returncode == 0
            and merge
            and merge.returncode == 0
        ):
            first_line = first.stdout.strip().splitlines()
            merge_line = merge.stdout.strip()
            if first_line and merge_line:
                try:
                    t0 = datetime.fromisoformat(first_line[0])
                    t1 = datetime.fromisoformat(merge_line)
                    stats["time_to_merge_hours"] = (
                        t1 - t0
                    ).total_seconds() / 3600.0
                except (ValueError, IndexError):
                    pass

        return stats

    def get_agent_worktrees(self) -> Dict[str, str]:
        """Map agent names to their worktree paths.

        Scans the workspace for git worktrees and infers agent names
        from the worktree directory names.
        """
        result = _run_git(
            ["worktree", "list", "--porcelain"], cwd=self._workspace_path
        )
        if not result or result.returncode != 0:
            return {}

        worktrees: Dict[str, str] = {}
        current_path: Optional[str] = None
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                current_path = line[len("worktree ") :]
            elif line.startswith("branch ") and current_path:
                branch = line[len("branch refs/heads/") :]
                # Convention: branch names like "agent/<name>/..."
                parts = branch.split("/")
                if len(parts) >= 2:
                    agent_name = parts[1] if parts[0] == "agent" else parts[0]
                else:
                    agent_name = branch
                worktrees[agent_name] = current_path
                current_path = None
        return worktrees

    # --- Branch-based observables ---

    def get_feature_branches(
        self, base: str = "origin/main"
    ) -> List[Dict[str, str]]:
        """Return unmerged remote feature branches.

        Each entry contains:
            branch: full ref name (e.g. ``origin/claude/fix-bug``)
            agent:  inferred agent name from first path segment after remote
            slug:   remaining path after agent prefix
        """
        result = _run_git(
            ["branch", "-r", "--no-merged", base, "--format=%(refname:short)"],
            cwd=self._workspace_path,
        )
        if not result or result.returncode != 0:
            return []

        branches: List[Dict[str, str]] = []
        for line in result.stdout.strip().splitlines():
            ref = line.strip()
            if not ref or "->" in ref:
                continue
            # Strip remote prefix (e.g. "origin/")
            without_remote = ref.split("/", 1)[1] if "/" in ref else ref
            parts = without_remote.split("/", 1)
            agent = parts[0]
            slug = parts[1] if len(parts) > 1 else without_remote
            branches.append({"branch": ref, "agent": agent, "slug": slug})
        return branches

    def get_branch_stats(
        self, branch: str, base: str = "origin/main"
    ) -> dict:
        """Compute PR-related observables for a remote branch vs *base*.

        Same return schema as :meth:`get_pr_stats` so the mapper can
        consume either interchangeably.
        """
        cwd = self._workspace_path
        stats: dict = {
            "commit_count": 0,
            "files_changed": 0,
            "review_iterations": 0,
            "ci_failures": 0,
            "time_to_merge_hours": None,
        }

        # Commit count since divergence from base
        result = _run_git(
            ["rev-list", "--count", branch, f"^{base}"], cwd=cwd
        )
        if result and result.returncode == 0:
            try:
                stats["commit_count"] = int(result.stdout.strip())
            except ValueError:
                pass

        # Files changed
        result = _run_git(
            ["diff", "--name-only", f"{base}...{branch}"], cwd=cwd
        )
        if result and result.returncode == 0:
            files = [f for f in result.stdout.strip().splitlines() if f]
            stats["files_changed"] = len(files)

        # Review iterations (fixup/amend/fix commits)
        result = _run_git(
            ["log", "--oneline", f"{base}..{branch}"], cwd=cwd
        )
        if result and result.returncode == 0:
            rework = 0
            for line in result.stdout.strip().splitlines():
                low = line.lower()
                if any(
                    k in low
                    for k in ("fixup!", "squash!", "amend", "fix:", "fix ")
                ):
                    rework += 1
            stats["review_iterations"] = rework

        # CI failures (commits with "[ci-fail]" in message)
        result = _run_git(
            ["log", "--oneline", "--fixed-strings", "--grep=[ci-fail]", f"{base}..{branch}"],
            cwd=cwd,
        )
        if result and result.returncode == 0:
            failures = [
                line for line in result.stdout.strip().splitlines() if line
            ]
            stats["ci_failures"] = len(failures)

        # Time span: first commit to last commit on branch
        result = _run_git(
            ["log", "--format=%aI", f"{base}..{branch}"], cwd=cwd
        )
        if result and result.returncode == 0:
            timestamps = [
                line.strip()
                for line in result.stdout.strip().splitlines()
                if line.strip()
            ]
            if len(timestamps) >= 2:
                try:
                    t0 = datetime.fromisoformat(timestamps[-1])
                    t1 = datetime.fromisoformat(timestamps[0])
                    stats["time_to_merge_hours"] = (
                        t1 - t0
                    ).total_seconds() / 3600.0
                except (ValueError, IndexError):
                    pass

        return stats

    def get_recent_activity(
        self, agent_name: str, since: datetime
    ) -> List[GasTownEvent]:
        """Emit PR-related events for an agent since *since*."""
        worktrees = self.get_agent_worktrees()
        wt = worktrees.get(agent_name)
        if not wt:
            return []

        events: List[GasTownEvent] = []
        since_iso = since.isoformat()

        result = _run_git(
            ["log", f"--since={since_iso}", "--format=%H %s", "HEAD"],
            cwd=wt,
        )
        if result and result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if not line:
                    continue
                parts = line.split(" ", 1)
                sha = parts[0]
                msg = parts[1] if len(parts) > 1 else ""
                event_type = GasTownEventType.PR_OPENED
                if "merge" in msg.lower():
                    event_type = GasTownEventType.PR_MERGED
                elif "[ci-fail]" in msg:
                    event_type = GasTownEventType.CI_FAILED
                events.append(
                    GasTownEvent(
                        event_type=event_type,
                        agent_name=agent_name,
                        payload={"sha": sha, "message": msg},
                    )
                )
        return events
