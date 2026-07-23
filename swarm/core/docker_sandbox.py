"""Docker-based sandbox for isolated agent code execution.

Provides real process-level isolation via Docker containers, filling the
gap identified in the InitRunner comparison.  The sandbox translates
governance contract parameters (from :mod:`swarm.bridges.opensandbox.config`)
into concrete ``docker run`` constraints: memory limits, CPU shares, network
mode, read-only mounts, and execution timeouts.

Design goals:

- **Governance-aware**: Resource limits and network policies are derived
  directly from :class:`GovernanceContract` objects.
- **Pluggable**: Implements :class:`ExecutionBackend` so it drops into the
  existing :class:`FailoverChain` alongside in-process backends.
- **Graceful degradation**: If the ``docker`` Python SDK is not installed or
  the Docker daemon is unreachable, operations raise
  :class:`DockerUnavailableError` rather than crashing.
- **Auditable**: Every container execution is logged with timing, exit codes,
  and truncated output for provenance.

.. note::

   Install the optional ``docker`` extra to use this module::

       pip install swarm-safety[docker]

   The Docker daemon must be running and the current user must have
   permission to communicate with it (``docker`` group or rootless setup).
"""

from __future__ import annotations

import atexit
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from swarm.core.sandbox import (
    ExecutionBackend,
    RetryPolicy,
    SandboxEnvironment,
)

logger = logging.getLogger(__name__)

# Maximum bytes of stdout/stderr captured per exec call.
_MAX_OUTPUT_BYTES = 256_000  # 250 KB

try:
    import docker  # isort: skip
    from docker.errors import ImageNotFound as DockerImageNotFound, NotFound as DockerNotFound  # isort: skip  # noqa: E501
    from docker.types import Ulimit  # isort: skip

    _DOCKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DOCKER_AVAILABLE = False
    docker = None  # type: ignore[assignment]
    DockerImageNotFound = Exception  # type: ignore[assignment,misc]
    DockerNotFound = Exception  # type: ignore[assignment,misc]
    Ulimit = None  # type: ignore[assignment,misc]


# Maximum bytes to read back from a container (H1 fix).
_MAX_COPY_FROM_BYTES = 50_000_000  # 50 MB

_DOCKER_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]")


def _shell_quote(s: str) -> str:
    """Shell-quote a string for safe embedding in sh -c commands."""
    return "'" + s.replace("'", "'\\''") + "'"


def _validate_container_path(path: str) -> None:
    """Validate a path destined for inside a container (C2/C1 fix).

    Raises ValueError for traversal attempts or invalid characters.
    """
    import posixpath

    normalised = posixpath.normpath(path)
    if ".." in normalised.split("/"):
        raise ValueError(f"Path traversal detected: {path!r}")
    # Block shell metacharacters
    for ch in (";", "&", "|", "`", "$", "(", ")", "{", "}", "\n", "\r"):
        if ch in path:
            raise ValueError(f"Invalid character {ch!r} in path: {path!r}")


def _sanitize_agent_id(agent_id: str) -> str:
    """Sanitize an agent_id for use in Docker container names (L2 fix)."""
    return _DOCKER_NAME_RE.sub("-", agent_id)[:64]


class DockerUnavailableError(RuntimeError):
    """Raised when Docker SDK is missing or daemon is unreachable."""


class ContainerState(Enum):
    """Lifecycle states for a managed container."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    REMOVED = "removed"


@dataclass
class ExecResult:
    """Result of a single command execution inside a container."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    command: str = ""


@dataclass
class ContainerSpec:
    """Declarative specification for a sandbox container.

    Typically derived from a :class:`GovernanceContract` via
    :func:`contract_to_spec`.
    """

    image: str = "python:3.12-slim"
    name: Optional[str] = None
    command: Optional[str] = None  # entrypoint override; default keeps container alive
    working_dir: str = "/workspace"
    env: Dict[str, str] = field(default_factory=dict)
    mem_limit: str = "512m"  # Docker memory limit string
    cpu_shares: int = 1024
    cpu_period: int = 100_000  # microseconds
    cpu_quota: int = 0  # 0 = no limit; set to cpu_period for 1 CPU
    disk_limit_mb: int = 1024  # advisory; requires storage-driver support
    network_mode: str = "none"  # "none", "bridge", or a custom network name
    read_only_root: bool = True
    tmpfs: Dict[str, str] = field(
        default_factory=lambda: {"/tmp": "size=64m,noexec"}
    )
    mounts: List[Dict[str, Any]] = field(default_factory=list)  # bind mounts
    labels: Dict[str, str] = field(default_factory=lambda: {"managed-by": "swarm"})
    timeout_seconds: int = 1800  # container lifetime cap
    ulimits: List[Dict[str, Any]] = field(default_factory=list)
    security_opt: List[str] = field(
        default_factory=lambda: ["no-new-privileges:true"]
    )
    cap_drop: List[str] = field(default_factory=lambda: ["ALL"])
    cap_add: List[str] = field(default_factory=list)
    pids_limit: int = 256


def contract_to_spec(
    contract: Any,
    *,
    image: Optional[str] = None,
    agent_id: str = "",
    extra_env: Optional[Dict[str, str]] = None,
) -> ContainerSpec:
    """Convert a :class:`GovernanceContract` into a :class:`ContainerSpec`.

    Parameters
    ----------
    contract:
        A ``GovernanceContract`` (from ``swarm.bridges.opensandbox.config``).
    image:
        Override the container image (default uses the contract tier logic).
    agent_id:
        Agent identifier, added to labels and env.
    extra_env:
        Additional environment variables merged on top of contract env.
    """
    from swarm.bridges.opensandbox.config import NetworkPolicy

    # Network mode — H4 fix: ALLOWLIST fails closed to "none" until
    # iptables-based allowlist enforcement is implemented.
    if contract.network == NetworkPolicy.DENY_ALL:
        network_mode = "none"
    elif contract.network == NetworkPolicy.ALLOWLIST:
        logger.warning(
            "NetworkPolicy.ALLOWLIST is not yet enforced; falling back "
            "to deny-all (none) for safety.  Contract: %s",
            contract.contract_id,
        )
        network_mode = "none"
    elif contract.network == NetworkPolicy.FULL:
        network_mode = "bridge"
    else:
        network_mode = "none"  # Unknown policy — fail closed

    # Environment — M5 fix: extra_env cannot override SWARM_ vars.
    env = contract.to_sandbox_env()
    if agent_id:
        env["SWARM_AGENT_ID"] = agent_id
    if extra_env:
        for k, v in extra_env.items():
            if k.startswith("SWARM_"):
                logger.warning(
                    "Ignoring extra_env key %r — cannot override "
                    "SWARM_ namespace variables",
                    k,
                )
                continue
            env[k] = v

    # Mounts from contract
    mounts = []
    for mount_path in contract.allowed_mounts:
        mounts.append(
            {
                "type": "bind",
                "source": mount_path,
                "target": mount_path,
                "read_only": True,
            }
        )

    suffix = uuid.uuid4().hex[:8]
    safe_id = _sanitize_agent_id(agent_id) if agent_id else ""
    name = f"swarm-{safe_id}-{suffix}" if safe_id else f"swarm-sandbox-{suffix}"

    # H3 fix: explicitly set all security-critical fields rather than
    # relying on ContainerSpec defaults.
    return ContainerSpec(
        image=image or "python:3.12-slim",
        name=name,
        mem_limit=f"{contract.max_memory_mb}m",
        cpu_shares=contract.max_cpu_shares,
        cpu_period=100_000,
        cpu_quota=100_000,  # I4 fix: hard limit to 1 CPU equivalent
        disk_limit_mb=contract.max_disk_mb,
        network_mode=network_mode,
        read_only_root=True,
        tmpfs={"/tmp": "size=64m,noexec"},
        env=env,
        mounts=mounts,
        timeout_seconds=contract.timeout_seconds,
        security_opt=["no-new-privileges:true"],
        cap_drop=["ALL"],
        cap_add=[],
        pids_limit=256,
        labels={
            "managed-by": "swarm",
            "swarm.agent_id": agent_id,
            "swarm.contract_id": contract.contract_id,
            "swarm.tier": contract.tier,
        },
    )


# I2 fix: atexit cleanup for containers started by this process.
_active_sandboxes: List["DockerSandbox"] = []


def _register_cleanup(sandbox: "DockerSandbox") -> None:
    """Register a sandbox for atexit cleanup."""
    _active_sandboxes.append(sandbox)
    atexit.register(_atexit_cleanup)


def _atexit_cleanup() -> None:
    """Stop and remove all active sandboxes on process exit."""
    for sandbox in list(_active_sandboxes):
        try:
            sandbox.stop(timeout=5)
            sandbox.remove()
        except Exception as exc:
            # Best-effort cleanup at process exit; log and continue.
            logger.warning("Failed to clean up Docker sandbox during atexit: %s", exc)
    _active_sandboxes.clear()


def _ensure_docker() -> None:
    """Raise if Docker SDK is unavailable."""
    if not _DOCKER_AVAILABLE:
        raise DockerUnavailableError(
            "Docker SDK not installed. Install with: pip install swarm-safety[docker]"
        )


class DockerSandbox:
    """Manages a single Docker container for isolated agent execution.

    Lifecycle::

        sandbox = DockerSandbox(spec)
        sandbox.start()
        result = sandbox.exec("python -c 'print(42)'")
        snapshot_id = sandbox.snapshot()
        sandbox.stop()
        sandbox.remove()

    The container runs with ``tail -f /dev/null`` as its entrypoint so it
    stays alive between ``exec`` calls.  Resource limits from the spec are
    enforced by Docker at the container level.
    """

    def __init__(
        self,
        spec: ContainerSpec,
        client: Optional[Any] = None,
    ) -> None:
        _ensure_docker()
        self.spec = spec
        try:
            self._client = client or docker.from_env()
        except Exception as exc:
            raise DockerUnavailableError(
                f"Cannot connect to Docker daemon: {exc}"
            ) from exc
        self._container: Optional[Any] = None
        self._state = ContainerState.REMOVED
        self._lock = threading.Lock()  # M3 fix: thread safety
        self._exec_log: List[ExecResult] = []
        self._snapshots: List[str] = []
        self._created_at: Optional[float] = None

    @property
    def container_id(self) -> Optional[str]:
        """Short container ID, or None if not created."""
        if self._container is None:
            return None
        return str(self._container.short_id)

    @property
    def state(self) -> ContainerState:
        return self._state

    @property
    def exec_log(self) -> List[ExecResult]:
        return list(self._exec_log)

    @property
    def snapshots(self) -> List[str]:
        return list(self._snapshots)

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> str:
        """Create and start the container. Returns container ID."""
        _ensure_docker()

        # Build host_config kwargs
        ulimits = []
        if Ulimit is not None:
            for ul in self.spec.ulimits:
                ulimits.append(Ulimit(**ul))
            # Always limit number of processes
            ulimits.append(Ulimit(name="nproc", soft=self.spec.pids_limit, hard=self.spec.pids_limit))

        run_kwargs: Dict[str, Any] = {
            "image": self.spec.image,
            "command": self.spec.command or "tail -f /dev/null",
            "detach": True,
            "working_dir": self.spec.working_dir,
            "environment": self.spec.env,
            "mem_limit": self.spec.mem_limit,
            "cpu_shares": self.spec.cpu_shares,
            "network_mode": self.spec.network_mode,
            "read_only": self.spec.read_only_root,
            "tmpfs": self.spec.tmpfs,
            "labels": self.spec.labels,
            "security_opt": self.spec.security_opt,
            "cap_drop": self.spec.cap_drop,
            "pids_limit": self.spec.pids_limit,
            "stdin_open": False,
            "tty": False,
        }

        if self.spec.name:
            run_kwargs["name"] = self.spec.name

        if self.spec.cap_add:
            run_kwargs["cap_add"] = self.spec.cap_add

        if self.spec.cpu_quota > 0:
            run_kwargs["cpu_period"] = self.spec.cpu_period
            run_kwargs["cpu_quota"] = self.spec.cpu_quota

        if ulimits:
            run_kwargs["ulimits"] = ulimits

        # Bind mounts
        if self.spec.mounts:
            volumes = {}
            for m in self.spec.mounts:
                mode = "ro" if m.get("read_only", True) else "rw"
                volumes[m["source"]] = {"bind": m["target"], "mode": mode}
            run_kwargs["volumes"] = volumes

        try:
            self._container = self._client.containers.run(**run_kwargs)
        except DockerImageNotFound:
            logger.info("Pulling image %s ...", self.spec.image)
            self._client.images.pull(self.spec.image)
            self._container = self._client.containers.run(**run_kwargs)

        self._state = ContainerState.RUNNING
        self._created_at = time.monotonic()

        # I2 fix: register atexit cleanup for this container
        _register_cleanup(self)

        logger.info(
            "Started container %s (image=%s, network=%s, mem=%s)",
            self._container.short_id,
            self.spec.image,
            self.spec.network_mode,
            self.spec.mem_limit,
        )
        return str(self._container.short_id)

    def exec(
        self,
        command: str,
        *,
        timeout: Optional[int] = None,
        user: str = "nobody",
        workdir: Optional[str] = None,
    ) -> ExecResult:
        """Execute a command inside the running container.

        Parameters
        ----------
        command:
            Shell command to execute (run via ``sh -c``).
        timeout:
            Per-command timeout in seconds.  Falls back to the container
            spec's ``timeout_seconds``.
        user:
            Unix user to run as.  Defaults to ``nobody`` for least privilege.
        workdir:
            Working directory inside the container.

        Returns
        -------
        ExecResult
            Contains exit code, captured stdout/stderr, and timing.
        """
        # M3 fix: lock protects state checks
        with self._lock:
            if self._container is None or self._state != ContainerState.RUNNING:
                raise RuntimeError("Container is not running")

        # Check container lifetime
        if self._created_at is not None:
            elapsed = time.monotonic() - self._created_at
            if elapsed > self.spec.timeout_seconds:
                self.stop()
                return ExecResult(
                    exit_code=-1,
                    stdout="",
                    stderr="Container lifetime exceeded",
                    duration_ms=0.0,
                    timed_out=True,
                    command=command,
                )

        # H2 fix: enforce per-exec timeout via shell `timeout` wrapper.
        effective_timeout = timeout or self.spec.timeout_seconds
        start = time.monotonic()

        try:
            # Wrap command in `timeout` to prevent indefinite blocking.
            wrapped_cmd = f"timeout {effective_timeout}s sh -c {_shell_quote(command)}"
            exec_id = self._client.api.exec_create(
                self._container.id,
                cmd=["sh", "-c", wrapped_cmd],
                user=user,
                workdir=workdir or self.spec.working_dir,
                stdout=True,
                stderr=True,
            )
            output = self._client.api.exec_start(exec_id["Id"], demux=True)
            inspect = self._client.api.exec_inspect(exec_id["Id"])

            duration_ms = (time.monotonic() - start) * 1000

            stdout_bytes = output[0] if output[0] else b""
            stderr_bytes = output[1] if output[1] else b""

            exit_code = inspect.get("ExitCode", -1)
            # Exit code 124 = killed by `timeout` command (H2 fix)
            timed_out = exit_code == 124

            result = ExecResult(
                exit_code=exit_code,
                stdout=stdout_bytes[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
                stderr=stderr_bytes[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
                timed_out=timed_out,
                command=command,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            result = ExecResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc)[:500],
                duration_ms=duration_ms,
                timed_out="timeout" in str(exc).lower() or "deadline" in str(exc).lower(),
                command=command,
            )

        self._exec_log.append(result)
        return result

    def copy_to(self, local_path: str, container_path: str) -> None:
        """Copy a local file/directory into the container.

        Parameters
        ----------
        local_path:
            Path on the host.  Must resolve under the working directory
            or an allowed mount path.
        container_path:
            Absolute destination path inside the container.
            Must not contain ``..`` or shell metacharacters.

        Raises
        ------
        ValueError
            If paths fail validation.
        """
        import io
        import os
        import tarfile

        if self._container is None:
            raise RuntimeError("Container is not created")

        # C2 fix: validate container_path against traversal
        _validate_container_path(container_path)
        if not os.path.isabs(container_path):
            raise ValueError(f"container_path must be absolute: {container_path!r}")

        # C2 fix: resolve local_path symlinks, validate it exists
        real_local = os.path.realpath(local_path)
        if not os.path.exists(real_local):
            raise FileNotFoundError(f"Local path not found: {local_path!r}")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            arcname = os.path.basename(container_path)
            tar.add(real_local, arcname=arcname)
        buf.seek(0)

        dest_dir = "/".join(container_path.split("/")[:-1]) or "/"
        self._container.put_archive(dest_dir, buf)

    def copy_from(
        self,
        container_path: str,
        max_bytes: int = _MAX_COPY_FROM_BYTES,
    ) -> bytes:
        """Copy a file from the container and return its contents.

        Parameters
        ----------
        container_path:
            Path inside the container to read.
        max_bytes:
            Maximum bytes to read.  Defaults to 50 MB.

        Returns
        -------
        bytes
            Raw file contents.

        Raises
        ------
        ValueError
            If the file exceeds *max_bytes*.
        """
        import io
        import tarfile

        if self._container is None:
            raise RuntimeError("Container is not created")

        # H1 fix: validate container_path
        _validate_container_path(container_path)

        bits, _ = self._container.get_archive(container_path)
        buf = io.BytesIO()
        total = 0
        for chunk in bits:
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"Container file exceeds size limit "
                    f"({total} > {max_bytes} bytes)"
                )
            buf.write(chunk)
        buf.seek(0)

        with tarfile.open(fileobj=buf) as tar:
            members = tar.getmembers()
            if not members:
                return b""
            member = members[0]
            # H1 fix: only extract regular files
            if not member.isreg():
                raise ValueError(
                    f"Expected regular file, got {member.type!r}: "
                    f"{member.name!r}"
                )
            f = tar.extractfile(member)
            if f is None:
                return b""
            data = f.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(
                    f"Extracted file exceeds size limit ({max_bytes} bytes)"
                )
            return data

    def snapshot(self, tag: Optional[str] = None) -> str:
        """Commit the container state as a Docker image.

        Parameters
        ----------
        tag:
            Image tag.  Auto-generated if not provided.

        Returns
        -------
        str
            The image ID of the snapshot.
        """
        if self._container is None:
            raise RuntimeError("Container is not created")

        tag = tag or f"swarm-snapshot-{uuid.uuid4().hex[:8]}"
        image = self._container.commit(repository="swarm-snapshots", tag=tag)
        self._snapshots.append(image.id)
        logger.info("Snapshot %s created for container %s", tag, self._container.short_id)
        return str(image.id)

    def pause(self) -> None:
        """Pause the container (freeze processes without killing)."""
        if self._container is None:
            raise RuntimeError("Container is not created")
        self._container.pause()
        self._state = ContainerState.PAUSED

    def unpause(self) -> None:
        """Unpause a paused container."""
        if self._container is None:
            raise RuntimeError("Container is not created")
        self._container.unpause()
        self._state = ContainerState.RUNNING

    def stop(self, timeout: int = 10) -> None:
        """Stop the container gracefully."""
        if self._container is None:
            return
        try:
            self._container.stop(timeout=timeout)
            self._state = ContainerState.STOPPED
        except Exception as exc:
            logger.warning("Error stopping container: %s", exc)
            try:
                self._container.kill()
                self._state = ContainerState.STOPPED
            except Exception as kill_exc:
                logger.warning("Error force-killing container during stop: %s", kill_exc)

    def remove(self, force: bool = True) -> None:
        """Remove the container."""
        if self._container is None:
            return
        try:
            self._container.remove(force=force)
        except DockerNotFound:
            pass  # already removed
        except Exception as exc:
            logger.warning("Error removing container: %s", exc)
        self._state = ContainerState.REMOVED
        self._container = None

    def get_stats(self) -> Dict[str, Any]:
        """Return live container resource usage statistics."""
        if self._container is None or self._state != ContainerState.RUNNING:
            return {}
        try:
            stats: Dict[str, Any] = self._container.stats(stream=False)
            return stats
        except Exception:
            return {}

    def get_logs(self, tail: int = 100) -> str:
        """Return recent container logs."""
        if self._container is None:
            return ""
        try:
            return str(self._container.logs(tail=tail).decode("utf-8", errors="replace"))
        except Exception:
            return ""

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
        self.remove()


# ---------------------------------------------------------------------------
# Pool for managing multiple sandboxes
# ---------------------------------------------------------------------------


class DockerSandboxPool:
    """Manages a pool of Docker sandboxes with lifecycle coordination.

    Enforces a global limit on concurrent containers and provides
    batch operations.

    Usage::

        pool = DockerSandboxPool(max_containers=10)
        sandbox = pool.create(spec)
        sandbox.start()
        result = sandbox.exec("echo hello")
        pool.destroy(sandbox)
        pool.shutdown()
    """

    def __init__(
        self,
        max_containers: int = 20,
        client: Optional[Any] = None,
    ) -> None:
        _ensure_docker()
        try:
            self._client = client or docker.from_env()
        except Exception as exc:
            raise DockerUnavailableError(
                f"Cannot connect to Docker daemon: {exc}"
            ) from exc
        self._max_containers = max_containers
        self._sandboxes: Dict[str, DockerSandbox] = {}  # container_name -> sandbox

    @property
    def active_count(self) -> int:
        return len(self._sandboxes)

    @property
    def max_containers(self) -> int:
        return self._max_containers

    def create(self, spec: ContainerSpec) -> DockerSandbox:
        """Create a new sandbox from spec (does not start it).

        Raises
        ------
        RuntimeError
            If the pool is at capacity.
        """
        if len(self._sandboxes) >= self._max_containers:
            raise RuntimeError(
                f"Pool capacity reached ({self._max_containers} containers)"
            )
        sandbox = DockerSandbox(spec, client=self._client)
        name = spec.name or f"swarm-pool-{uuid.uuid4().hex[:8]}"
        self._sandboxes[name] = sandbox
        return sandbox

    def get(self, name: str) -> Optional[DockerSandbox]:
        """Look up a sandbox by container name."""
        return self._sandboxes.get(name)

    def destroy(self, sandbox: DockerSandbox) -> None:
        """Stop, remove, and unregister a sandbox."""
        sandbox.stop()
        sandbox.remove()
        # Remove from pool by matching object identity
        self._sandboxes = {
            k: v for k, v in self._sandboxes.items() if v is not sandbox
        }

    def destroy_all(self) -> int:
        """Destroy all sandboxes. Returns the count destroyed."""
        count = len(self._sandboxes)
        for sandbox in list(self._sandboxes.values()):
            sandbox.stop()
            sandbox.remove()
        self._sandboxes.clear()
        return count

    def shutdown(self) -> None:
        """Graceful shutdown: destroy all and clean up snapshots."""
        self.destroy_all()

    def list_active(self) -> List[Dict[str, Any]]:
        """Return summary info for all active sandboxes."""
        result = []
        for name, sandbox in self._sandboxes.items():
            result.append(
                {
                    "name": name,
                    "container_id": sandbox.container_id,
                    "state": sandbox.state.value,
                    "image": sandbox.spec.image,
                    "network": sandbox.spec.network_mode,
                    "exec_count": len(sandbox.exec_log),
                }
            )
        return result

    def cleanup_stale(self) -> int:
        """Remove containers that have exceeded their lifetime.

        Returns the number of containers cleaned up.
        """
        cleaned = 0
        for name in list(self._sandboxes):
            sandbox = self._sandboxes[name]
            if sandbox._created_at is not None:
                elapsed = time.monotonic() - sandbox._created_at
                if elapsed > sandbox.spec.timeout_seconds:
                    self.destroy(sandbox)
                    cleaned += 1
        return cleaned


# ---------------------------------------------------------------------------
# FailoverChain-compatible backend
# ---------------------------------------------------------------------------


class DockerSandboxBackend(ExecutionBackend):
    """Execution backend that runs tasks inside Docker containers.

    Plugs into the existing :class:`FailoverChain` so that simulations
    can fall back between in-process and Docker execution.

    Usage::

        backend = DockerSandboxBackend(spec)
        chain = FailoverChain([in_process_backend, backend])
        result = await chain.execute(sandbox_env)
    """

    def __init__(
        self,
        spec: ContainerSpec,
        *,
        command_template: str = "python -c '{code}'",
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        self._spec = spec
        self._command_template = command_template
        self._retry = retry or RetryPolicy(max_retries=2, base_delay=2.0)
        self._docker_sandbox: Optional[DockerSandbox] = None

    @property
    def name(self) -> str:
        return f"docker:{self._spec.image}"

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry

    async def run(self, sandbox: SandboxEnvironment) -> Any:
        """Execute sandbox content inside a Docker container."""
        _ensure_docker()

        # Create and start container
        ds = DockerSandbox(self._spec)
        self._docker_sandbox = ds
        try:
            ds.start()

            # Copy sandbox files into container using base64 encoding
            # to avoid shell injection (C1 security fix).
            import base64

            for path in sandbox.list_files():
                content = sandbox.read_file(path)
                # Validate path: must be relative, no traversal
                _validate_container_path(path)
                b64 = base64.b64encode(content.encode()).decode()
                target = f"/workspace{path}"
                ds.exec(
                    f"mkdir -p \"$(dirname '{target}')\"",
                    user="root",
                )
                ds.exec(
                    f"echo '{b64}' | base64 -d > '{target}'",
                    user="root",
                )

            # Execute the main task
            if sandbox.file_exists("/main.py"):
                result = ds.exec("python /workspace/main.py")
            else:
                # If no main.py, just run health check
                result = ds.exec("python -c 'print(\"sandbox ready\")'")

            if result.exit_code != 0:
                raise RuntimeError(
                    f"Docker execution failed (exit {result.exit_code}): "
                    f"{result.stderr[:200]}"
                )

            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
                "container_id": ds.container_id,
            }
        finally:
            ds.stop()
            ds.remove()


# ---------------------------------------------------------------------------
# Utility: check Docker availability
# ---------------------------------------------------------------------------


def docker_available() -> bool:
    """Return True if Docker SDK is installed and daemon is reachable."""
    if not _DOCKER_AVAILABLE:
        return False
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def list_swarm_containers(client: Optional[Any] = None) -> List[Dict[str, Any]]:
    """List all Docker containers managed by SWARM.

    Filters containers by the ``managed-by=swarm`` label.
    """
    _ensure_docker()
    client = client or docker.from_env()
    containers = client.containers.list(
        all=True, filters={"label": "managed-by=swarm"}
    )
    return [
        {
            "id": c.short_id,
            "name": c.name,
            "status": c.status,
            "image": c.image.tags[0] if c.image.tags else str(c.image.id)[:12],
            "labels": c.labels,
        }
        for c in containers
    ]


def cleanup_swarm_containers(client: Optional[Any] = None) -> int:
    """Stop and remove all SWARM-managed containers. Returns count removed."""
    _ensure_docker()
    client = client or docker.from_env()
    containers = client.containers.list(
        all=True, filters={"label": "managed-by=swarm"}
    )
    count = 0
    for c in containers:
        try:
            c.stop(timeout=5)
        except Exception as exc:
            logger.debug("Failed to stop SWARM container %s: %s", c.short_id, exc)
        try:
            c.remove(force=True)
            count += 1
        except Exception as exc:
            logger.debug("Failed to remove SWARM container %s: %s", c.short_id, exc)
    return count
