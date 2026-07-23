"""Configuration for the SWARM-Worktree sandbox bridge."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WorktreeConfig:
    """Configuration for the Worktree sandbox bridge.

    Attributes:
        repo_path: Path to the main git repository.
        sandbox_root: Directory where worktree sandboxes are created.
        max_sandboxes: Maximum concurrent sandboxes.
        max_files_per_sandbox: File count limit per sandbox.
        max_disk_bytes: Disk usage limit per sandbox in bytes.
        command_timeout_seconds: Default subprocess timeout.
        gc_stale_seconds: Destroy sandboxes inactive longer than this.
        env_allowlist: Explicit env vars to inject into sandboxes.
        env_blocklist_patterns: Glob patterns for env files to delete.
        default_command_allowlist: Commands any agent may execute.
        agent_command_allowlists: Per-agent command overrides.
        proxy_sigmoid_k: Sigmoid sharpness for ProxyComputer.
        max_interactions: Cap on stored interactions.
        max_events: Cap on stored events.
    """

    repo_path: str = "."
    sandbox_root: str = "/tmp/swarm-sandboxes"
    max_sandboxes: int = 10
    max_files_per_sandbox: int = 5000
    max_disk_bytes: int = 500_000_000  # 500 MB
    command_timeout_seconds: int = 60
    gc_stale_seconds: int = 3600  # 1 hour

    env_allowlist: Dict[str, str] = field(default_factory=dict)
    env_blocklist_patterns: List[str] = field(
        default_factory=lambda: [".env", ".env.*", ".env.local", ".env.production"]
    )

    default_command_allowlist: List[str] = field(
        default_factory=lambda: [
            "git",
            "python",
            "pytest",
            "ruff",
            "mypy",
            "pip",
            "ls",
            "cat",
            "head",
            "tail",
            "wc",
            "diff",
            "grep",
            "find",
            "echo",
            "mkdir",
            "touch",
        ]
    )
    agent_command_allowlists: Dict[str, List[str]] = field(default_factory=dict)

    proxy_sigmoid_k: float = 2.0
    max_interactions: int = 50000
    max_events: int = 50000
    agent_role_map: Dict[str, str] = field(default_factory=dict)
    poll_interval_seconds: float = 5.0

    # OS-level command isolation (opt-in). When enabled, executed commands are
    # wrapped in sandbox-exec (macOS) / bwrap (Linux) to confine writes to the
    # sandbox dir and block network. If no backend is available the command
    # still runs and the result records isolation="none" (fail-open) unless
    # require_os_isolation is set, in which case execution is denied.
    os_isolation_enabled: bool = False
    require_os_isolation: bool = False
    os_isolation_allow_network: bool = False

    # Secret read-confinement (7ge5 slice 3). When OS isolation is active,
    # deny reads of credential material even though general reads stay open
    # (interpreters need their stdlib). Defaults cover the standard homedir
    # credential locations; extend per deployment via the extra list. Paths
    # are ~-expanded and existence-filtered at execution time.
    os_isolation_deny_secret_reads: bool = True
    os_isolation_extra_secret_paths: List[str] = field(default_factory=list)
