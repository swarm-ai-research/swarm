"""OS-level sandbox launchers for worktree command execution.

The worktree executor runs allowlisted commands with
``subprocess.run(command, cwd=sandbox_path)``. ``cwd`` is only a working
directory, not a jail: an allowlisted ``python`` can still write anywhere on
disk and open network sockets. This module wraps a command's argv in a real OS
confinement so a granted command cannot **write outside its sandbox** or
**reach the network**.

Two backends are supported, selected by host:

- ``sandbox-exec`` (macOS): an SBPL profile that denies file writes outside the
  sandbox (plus temp/dev) and denies network access.
- ``bwrap`` (Linux, bubblewrap): a read-only root bind with a read-write bind on
  only the sandbox subtree, and a private (empty) network namespace.

This is **write + network** confinement, plus **targeted secret-read denial**.
It is not a full read-jail: interpreters need their standard library, so reads
default to permissive. But a granted command has no business reading credential
material, so ``deny_read_paths`` layers a precise deny-list on top of the
permissive default — the ``.env`` / key / cloud-credential paths a
buggy-or-compromised command would exfiltrate. The functions here only build
argv — they never execute — so they are pure and easy to test. No
``shell=True`` is ever introduced, preserving the executor's NO-shell invariant.
"""

from __future__ import annotations

import os
import platform
import shutil
from typing import List, Optional, Sequence

Backend = str  # "sandbox-exec" | "bwrap" | "none"

# Paths an otherwise-confined process still legitimately needs to write to.
_MACOS_WRITABLE_SUBPATHS = ("/private/tmp", "/private/var/folders", "/tmp")
_MACOS_WRITABLE_LITERALS = ("/dev/null", "/dev/stdout", "/dev/stderr")

# Credential material a granted command should never read. Home-relative
# secret stores plus any dotenv/key files. Deny-by-pattern where the backend
# supports it (SBPL regex); bwrap denies the concrete home-relative dirs by
# masking them with an empty tmpfs. Kept aligned with the policy denied_paths
# and scan_secrets patterns.
_SECRET_HOME_SUBPATHS = (
    ".ssh",
    ".aws",
    ".config/gcloud",
    ".gnupg",
    ".netrc",
    ".git-credentials",
    ".config/gh",
)
# Filename regex fragments (SBPL) for secret-looking files anywhere on read.
_SECRET_NAME_REGEXES = (
    r".*/\.env($|\..*)",
    r".*\.pem$",
    r".*\.key$",
    r".*/id_rsa.*",
    r".*/credentials($|\..*)",
)


def default_deny_read_paths() -> List[str]:
    """Concrete home-relative secret directories/files to deny reads of."""

    home = os.path.expanduser("~")
    return [os.path.join(home, sub) for sub in _SECRET_HOME_SUBPATHS]


def detect_backend() -> Backend:
    """Return the OS sandbox backend available on this host, or ``"none"``."""

    system = platform.system()
    if system == "Darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec"
    if system == "Linux" and shutil.which("bwrap"):
        return "bwrap"
    return "none"


def _sbpl_string(path: str) -> str:
    r"""Quote a path as an SBPL string literal (escape ``\`` and ``"``)."""

    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _sbpl_regex(pattern: str) -> str:
    r"""Quote a regex as an SBPL ``#"..."`` literal.

    Regexes must NOT go through :func:`_sbpl_string` — its backslash
    escaping turns ``\.env`` into a literal-backslash pattern that never
    matches (found live: ``.env``/``*.pem`` reads sailed through while the
    profile "contained" the deny lines). SBPL regex literals take the
    pattern verbatim; the only unrepresentable character is ``"``.
    """

    if '"' in pattern:
        raise ValueError(f"SBPL regex literal cannot contain a quote: {pattern!r}")
    return f'#"{pattern}"'


def _build_macos_profile(
    sandbox_path: str,
    *,
    allow_network: bool,
    deny_read_paths: Sequence[str] = (),
) -> str:
    """Build an SBPL profile: permissive reads, writes confined, network gated.

    Secret paths are denied for reads *after* ``(allow default)`` — in SBPL the
    last matching rule wins, so these denies override the permissive default
    while leaving every other read (stdlib, interpreters) untouched. The
    sandbox subtree itself stays readable even if it sits under a denied path.
    """

    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        f"(allow file-write* (subpath {_sbpl_string(sandbox_path)}))",
    ]
    for sub in _MACOS_WRITABLE_SUBPATHS:
        lines.append(f"(allow file-write* (subpath {_sbpl_string(sub)}))")
    for lit in _MACOS_WRITABLE_LITERALS:
        lines.append(f"(allow file-write* (literal {_sbpl_string(lit)}))")
    for path in deny_read_paths:
        lines.append(f"(deny file-read* (subpath {_sbpl_string(path)}))")
    for regex in _SECRET_NAME_REGEXES:
        lines.append(f"(deny file-read* (regex {_sbpl_regex(regex)}))")
    # The command's own sandbox must stay readable even under a denied prefix.
    lines.append(f"(allow file-read* (subpath {_sbpl_string(sandbox_path)}))")
    if not allow_network:
        lines.append("(deny network*)")
    return "".join(lines)


def wrap_command(
    command: List[str],
    *,
    sandbox_path: str,
    backend: Backend,
    allow_network: bool = False,
    deny_read_paths: Optional[Sequence[str]] = None,
) -> List[str]:
    """Wrap ``command`` argv for ``backend``, confining writes + network + reads.

    ``deny_read_paths`` denies reads of the given paths (default:
    :func:`default_deny_read_paths` — the host's home-relative credential
    stores); secret-*named* files (``.env``, ``*.pem``, ``id_rsa*``, …) are
    always denied on the macOS backend regardless. ``None`` uses the default
    set; pass ``[]`` to disable path-based read denial.

    ``backend == "none"`` returns the command unchanged (caller is responsible
    for recording that no isolation was applied).
    """

    if not command:
        return list(command)

    # macOS temp sandboxes live under /var/... which symlinks to
    # /private/var/...; SBPL matches against real paths, so an unresolved
    # sandbox_path silently voids its own read/write allow-subpath rules
    # (found live: the sandbox's own .env was unreadable while writes only
    # survived via the /private/var/folders blanket allow).
    sandbox_path = os.path.realpath(sandbox_path)

    deny_reads = (
        default_deny_read_paths() if deny_read_paths is None else list(deny_read_paths)
    )

    if backend == "sandbox-exec":
        profile = _build_macos_profile(
            sandbox_path, allow_network=allow_network, deny_read_paths=deny_reads
        )
        return ["sandbox-exec", "-p", profile, *command]

    if backend == "bwrap":
        argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
        # Mask each secret dir with an empty tmpfs so reads see nothing. These
        # come BEFORE the sandbox --bind so that a secret path *inside* the
        # sandbox stays visible — the task's own bind is applied last and wins.
        for path in deny_reads:
            argv.extend(["--tmpfs", path])
        argv.extend(["--bind", sandbox_path, sandbox_path, "--chdir", sandbox_path])
        if not allow_network:
            argv.append("--unshare-net")
        argv.append("--")
        argv.extend(command)
        return argv

    return list(command)
