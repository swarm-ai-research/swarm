"""SandboxManager — git worktree lifecycle for agent sandboxes.

Pure infrastructure: creates, destroys, lists, and garbage-collects
git worktree sandboxes.  No SWARM domain concepts.
"""

import fnmatch
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from swarm.bridges.worktree.config import WorktreeConfig
from swarm.bridges.worktree.events import WorktreeEvent, WorktreeEventType

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


class SandboxManager:
    """Manages git worktree sandboxes for agents.

    Security invariants:
    - ENV-NEVER-PROPAGATE: .env* files deleted before any agent code runs
    - PATH-GUARD: rmtree only on resolved paths strictly under sandbox_root
    - RESOURCE-BOUNDED: max_sandboxes enforced
    """

    def __init__(self, config: WorktreeConfig) -> None:
        self._config = config
        self._repo_path = str(Path(config.repo_path).resolve())
        self._sandbox_root = str(Path(config.sandbox_root).resolve())
        self._sandbox_meta: Dict[str, float] = {}  # sandbox_id -> last_active_time

        # Ensure sandbox root exists
        os.makedirs(self._sandbox_root, exist_ok=True)

    def discover_existing(self) -> Dict[str, str]:
        """Discover existing worktree sandboxes on disk.

        Walks ``sandbox_root``, checks each subdirectory is a git worktree
        via ``git rev-parse``, and populates ``_sandbox_meta`` from directory
        mtime so the CLI can pick up sandboxes across invocations.

        Returns:
            Mapping of sandbox_id -> absolute path for all discovered sandboxes.
        """
        discovered: Dict[str, str] = {}
        if not os.path.isdir(self._sandbox_root):
            return discovered

        for entry in os.scandir(self._sandbox_root):
            if not entry.is_dir(follow_symlinks=False):
                continue

            resolved = str(Path(entry.path).resolve())
            # Verify it's actually a git worktree
            result = _run_git(
                ["rev-parse", "--is-inside-work-tree"],
                cwd=resolved,
            )
            if result is None or result.returncode != 0:
                continue

            sid = entry.name
            discovered[sid] = resolved

            # Populate metadata from directory mtime if not already tracked
            if sid not in self._sandbox_meta:
                try:
                    mtime = os.path.getmtime(resolved)
                except OSError:
                    mtime = time.time()
                self._sandbox_meta[sid] = mtime

        return discovered

    def create_sandbox(
        self,
        sandbox_id: str,
        branch: Optional[str] = None,
    ) -> str:
        """Create a new git worktree sandbox.

        Args:
            sandbox_id: Unique identifier for the sandbox.
            branch: Optional branch to check out (creates new branch if needed).

        Returns:
            The absolute path to the new sandbox.

        Raises:
            RuntimeError: If max_sandboxes exceeded or git operation fails.
        """
        if len(self._sandbox_meta) >= self._config.max_sandboxes:
            raise RuntimeError(
                f"Max sandboxes ({self._config.max_sandboxes}) reached"
            )

        sandbox_path = str(Path(self._sandbox_root, sandbox_id).resolve())

        # Ensure the resolved path is strictly under sandbox_root
        self._validate_path_under_root(sandbox_path)

        if os.path.exists(sandbox_path):
            raise RuntimeError(f"Sandbox path already exists: {sandbox_path}")

        # Build git worktree add command
        args = ["worktree", "add", sandbox_path]
        if branch:
            args.extend(["-b", branch])

        result = _run_git(args, cwd=self._repo_path)
        if result is None or result.returncode != 0:
            stderr = result.stderr if result else "timeout"
            raise RuntimeError(f"git worktree add failed: {stderr}")

        # ENV-NEVER-PROPAGATE: delete all .env* files
        self._scrub_env_files(sandbox_path)

        # Inject allowlisted env vars as a .env.sandbox file
        if self._config.env_allowlist:
            env_path = os.path.join(sandbox_path, ".env.sandbox")
            with open(env_path, "w") as f:
                for key, value in self._config.env_allowlist.items():
                    f.write(f"{key}={value}\n")

        self._sandbox_meta[sandbox_id] = time.time()
        return sandbox_path

    def destroy_sandbox(self, sandbox_id: str) -> WorktreeEvent:
        """Destroy a sandbox by removing its worktree.

        Args:
            sandbox_id: The sandbox to destroy.

        Returns:
            A WorktreeEvent recording the destruction.

        Raises:
            ValueError: If sandbox path is outside sandbox_root (path traversal).
            RuntimeError: If git worktree remove fails.
        """
        sandbox_path = str(Path(self._sandbox_root, sandbox_id).resolve())

        # PATH-GUARD: resolve symlinks and validate strictly under root
        self._validate_path_under_root(sandbox_path)

        # Try git worktree remove first (clean)
        _run_git(
            ["worktree", "remove", "--force", sandbox_path],
            cwd=self._repo_path,
        )

        # If git remove failed but path exists, force-clean
        if os.path.exists(sandbox_path):
            # Double-check the path guard before any rmtree
            self._validate_path_under_root(
                str(Path(sandbox_path).resolve())
            )
            shutil.rmtree(sandbox_path)

        # Also prune stale worktree references
        _run_git(["worktree", "prune"], cwd=self._repo_path)

        self._sandbox_meta.pop(sandbox_id, None)

        return WorktreeEvent(
            event_type=WorktreeEventType.SANDBOX_DESTROYED,
            sandbox_id=sandbox_id,
            payload={"path": sandbox_path},
        )

    def list_sandboxes(self) -> Dict[str, str]:
        """List all active sandboxes.

        Returns:
            Mapping of sandbox_id to absolute path.
        """
        result = _run_git(
            ["worktree", "list", "--porcelain"], cwd=self._repo_path
        )
        if not result or result.returncode != 0:
            return {}

        worktrees: Dict[str, str] = {}
        current_path: Optional[str] = None

        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                current_path = line[len("worktree "):]
            elif line == "" and current_path:
                # End of worktree entry
                resolved = str(Path(current_path).resolve())
                if resolved.startswith(self._sandbox_root + os.sep):
                    sid = os.path.basename(resolved)
                    worktrees[sid] = resolved
                current_path = None

        return worktrees

    def gc_stale(self) -> List[WorktreeEvent]:
        """Destroy sandboxes inactive longer than gc_stale_seconds.

        Returns:
            List of GC collection events.
        """
        now = time.time()
        stale_ids = [
            sid
            for sid, last_active in self._sandbox_meta.items()
            if now - last_active > self._config.gc_stale_seconds
        ]

        events: List[WorktreeEvent] = []
        for sid in stale_ids:
            try:
                event = self.destroy_sandbox(sid)
                event.event_type = WorktreeEventType.SANDBOX_GC_COLLECTED
                events.append(event)
                logger.info("GC collected stale sandbox: %s", sid)
            except (ValueError, RuntimeError) as exc:
                logger.warning("GC failed for sandbox %s: %s", sid, exc)

        return events

    def check_resource_limits(self, sandbox_id: str) -> Optional[str]:
        """Check if a sandbox exceeds resource limits.

        Returns:
            None if within limits, or a description of the violation.
        """
        sandbox_path = str(Path(self._sandbox_root, sandbox_id).resolve())
        self._validate_path_under_root(sandbox_path)

        if not os.path.exists(sandbox_path):
            return f"Sandbox {sandbox_id} does not exist"

        # Count files
        file_count = sum(
            len(files) for _, _, files in os.walk(sandbox_path)
        )
        if file_count > self._config.max_files_per_sandbox:
            return (
                f"File count {file_count} exceeds limit "
                f"{self._config.max_files_per_sandbox}"
            )

        # Check disk usage (approximate via walking)
        total_bytes = 0
        for dirpath, _, filenames in os.walk(sandbox_path):
            for fname in filenames:
                try:
                    total_bytes += os.path.getsize(
                        os.path.join(dirpath, fname)
                    )
                except OSError:
                    pass
        if total_bytes > self._config.max_disk_bytes:
            return (
                f"Disk usage {total_bytes} bytes exceeds limit "
                f"{self._config.max_disk_bytes}"
            )

        return None

    def touch_sandbox(self, sandbox_id: str) -> None:
        """Update the last-active timestamp for a sandbox."""
        if sandbox_id in self._sandbox_meta:
            self._sandbox_meta[sandbox_id] = time.time()

    def get_sandbox_path(self, sandbox_id: str) -> Optional[str]:
        """Get the absolute path for a sandbox, or None if not tracked."""
        if sandbox_id not in self._sandbox_meta:
            return None
        path = str(Path(self._sandbox_root, sandbox_id).resolve())
        self._validate_path_under_root(path)
        return path if os.path.isdir(path) else None

    def _scrub_env_files(self, sandbox_path: str) -> List[str]:
        """Delete all .env* files from a sandbox directory.

        Returns:
            List of deleted file paths.
        """
        deleted: List[str] = []
        for root, _, files in os.walk(sandbox_path):
            for fname in files:
                for pattern in self._config.env_blocklist_patterns:
                    if fnmatch.fnmatch(fname, pattern):
                        filepath = os.path.join(root, fname)
                        try:
                            os.remove(filepath)
                            deleted.append(filepath)
                            logger.info("Scrubbed env file: %s", filepath)
                        except OSError as exc:
                            logger.warning(
                                "Failed to scrub %s: %s", filepath, exc
                            )
                        break  # don't match same file twice
        return deleted

    def _validate_path_under_root(self, path: str) -> None:
        """Validate that path is under or equal to sandbox_root.

        Raises:
            ValueError: If path is not under sandbox_root.
        """
        resolved = str(Path(path).resolve())
        root = self._sandbox_root
        if not resolved.startswith(root + os.sep) and resolved != root:
            raise ValueError(
                f"Path traversal detected: {resolved} is not under {root}"
            )
