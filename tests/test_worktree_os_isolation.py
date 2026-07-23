"""Tests for OS-level command isolation (7ge5, slice 2)."""

from pathlib import Path

import pytest

from swarm.bridges.worktree.bridge import WorktreeBridge
from swarm.bridges.worktree.config import WorktreeConfig
from swarm.bridges.worktree.events import WorktreeEventType
from swarm.bridges.worktree.sandbox_launch import detect_backend, wrap_command

CMD = ["python", "-c", "print(1)"]


# --- argv construction (pure, runs on every platform) ---


def test_macos_wrap_structure():
    argv = wrap_command(CMD, sandbox_path="/box", backend="sandbox-exec")
    assert argv[0] == "sandbox-exec"
    assert argv[1] == "-p"
    profile = argv[2]
    assert "(deny network*)" in profile
    assert "(deny file-write*)" in profile
    assert '(allow file-write* (subpath "/box"))' in profile
    assert argv[3:] == CMD  # command appended verbatim


def test_macos_allow_network_omits_network_deny():
    profile = wrap_command(
        CMD, sandbox_path="/box", backend="sandbox-exec", allow_network=True
    )[2]
    assert "(deny network*)" not in profile
    assert "(deny file-write*)" in profile  # writes still confined


def test_macos_profile_escapes_path():
    profile = wrap_command(
        CMD, sandbox_path='/weird "quoted" \\path', backend="sandbox-exec"
    )[2]
    assert '/weird \\"quoted\\" \\\\path' in profile


def test_bwrap_wrap_structure():
    argv = wrap_command(CMD, sandbox_path="/box", backend="bwrap")
    assert argv[0] == "bwrap"
    assert "--unshare-net" in argv
    assert "--ro-bind" in argv
    # rw bind on the sandbox subtree
    i = argv.index("--bind")
    assert argv[i + 1] == "/box" and argv[i + 2] == "/box"
    # command after the -- separator
    sep = argv.index("--")
    assert argv[sep + 1 :] == CMD


def test_bwrap_allow_network_omits_unshare():
    argv = wrap_command(CMD, sandbox_path="/box", backend="bwrap", allow_network=True)
    assert "--unshare-net" not in argv


def test_backend_none_returns_command_unchanged():
    assert wrap_command(CMD, sandbox_path="/box", backend="none") == CMD


def test_empty_command_is_passed_through():
    assert wrap_command([], sandbox_path="/box", backend="sandbox-exec") == []


def test_detect_backend_is_valid():
    assert detect_backend() in {"sandbox-exec", "bwrap", "none"}


# --- executor integration ---


def _bridge(tmp_path: Path, **cfg) -> tuple[WorktreeBridge, str]:
    bridge = WorktreeBridge(
        WorktreeConfig(
            repo_path=str(tmp_path / "repo"),
            sandbox_root=str(tmp_path / "boxes"),
            **cfg,
        )
    )
    # A minimal git repo so a worktree sandbox can be created.
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init"],
        ["config", "user.email", "a@b.c"],
        ["config", "user.name", "t"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True
    )
    sandbox_id = bridge.create_agent_sandbox("codex")
    return bridge, sandbox_id


def test_disabled_isolation_runs_unwrapped(tmp_path):
    bridge, sid = _bridge(tmp_path)  # os_isolation_enabled defaults False
    result, _ = bridge._executor.execute(sid, "codex", ["echo", "hi"])
    assert result.allowed
    assert result.isolation == "none"


def test_fail_closed_denies_when_no_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "swarm.bridges.worktree.executor.detect_backend", lambda: "none"
    )
    bridge, sid = _bridge(
        tmp_path, os_isolation_enabled=True, require_os_isolation=True
    )
    result, events = bridge._executor.execute(sid, "codex", ["echo", "hi"])
    assert not result.allowed
    assert "OS isolation required" in result.deny_reason
    assert any(e.event_type == WorktreeEventType.COMMAND_DENIED for e in events)


def test_fail_open_runs_and_labels_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "swarm.bridges.worktree.executor.detect_backend", lambda: "none"
    )
    bridge, sid = _bridge(
        tmp_path, os_isolation_enabled=True, require_os_isolation=False
    )
    result, _ = bridge._executor.execute(sid, "codex", ["echo", "hi"])
    assert result.allowed
    assert result.isolation == "none"


# --- real-backend enforcement (skipped where no backend exists, e.g. CI) ---

_BACKEND = detect_backend()
_needs_backend = pytest.mark.skipif(
    _BACKEND == "none", reason="no OS sandbox backend on this host"
)


@_needs_backend
def test_network_egress_blocked_under_isolation(tmp_path):
    bridge, sid = _bridge(tmp_path, os_isolation_enabled=True)
    code = (
        "import socket,sys\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1',80),2).close()\n"
        "    print('CONNECTED')\n"
        "except Exception:\n"
        "    sys.exit(7)\n"
    )
    result, _ = bridge._executor.execute(sid, "codex", ["python", "-c", code])
    assert result.isolation == _BACKEND
    assert "CONNECTED" not in result.stdout
    assert result.return_code != 0


@_needs_backend
def test_write_outside_sandbox_blocked(tmp_path):
    bridge, sid = _bridge(tmp_path, os_isolation_enabled=True)
    # Target a genuinely non-temp location: the home dir. (pytest's tmp_path
    # lives under a permitted temp subpath, so it can't serve as "outside".)
    # The write must be blocked while writes inside the sandbox succeed.
    target = Path.home() / ".swarm-sandbox-isolation-test-outside"
    code = (
        "import sys\n"
        f"try:\n"
        f"    open({str(target)!r},'w').write('x')\n"
        f"    print('WROTE')\n"
        f"except Exception:\n"
        f"    sys.exit(8)\n"
    )
    try:
        result, _ = bridge._executor.execute(sid, "codex", ["python", "-c", code])
        assert result.isolation == _BACKEND
        assert "WROTE" not in result.stdout
        assert not target.exists()
    finally:
        target.unlink(missing_ok=True)


# --- secret-read confinement (7ge5) ---


def test_macos_denies_secret_named_reads_by_default():
    from swarm.bridges.worktree.sandbox_launch import wrap_command as wc

    profile = wc(CMD, sandbox_path="/box", backend="sandbox-exec")[2]
    # secret-named files denied on read regardless of path
    assert "(deny file-read* (regex" in profile
    # SBPL string-escapes the backslash, so the regex appears double-escaped
    assert "env" in profile and "pem" in profile
    # sandbox stays readable even under a denied prefix
    assert '(allow file-read* (subpath "/box"))' in profile


def test_macos_denies_default_home_credential_dirs():
    import os

    from swarm.bridges.worktree.sandbox_launch import wrap_command as wc

    profile = wc(CMD, sandbox_path="/box", backend="sandbox-exec")[2]
    ssh = os.path.join(os.path.expanduser("~"), ".ssh")
    assert f'(deny file-read* (subpath "{ssh}"))' in profile


def test_deny_read_paths_empty_list_disables_path_denial():
    from swarm.bridges.worktree.sandbox_launch import wrap_command as wc

    profile = wc(
        CMD, sandbox_path="/box", backend="sandbox-exec", deny_read_paths=[]
    )[2]
    # no subpath read-denies, but name-regex denies still present (belt+braces)
    assert "(deny file-read* (subpath" not in profile
    assert "(deny file-read* (regex" in profile


def test_bwrap_masks_secret_dirs_with_tmpfs():
    from swarm.bridges.worktree.sandbox_launch import wrap_command as wc

    argv = wc(
        CMD, sandbox_path="/box", backend="bwrap",
        deny_read_paths=["/home/u/.ssh", "/home/u/.aws"],
    )
    joined = " ".join(argv)
    assert "--tmpfs /home/u/.ssh" in joined
    assert "--tmpfs /home/u/.aws" in joined
    # tmpfs masks come BEFORE the sandbox bind so a secret path inside the
    # sandbox stays visible (the task's own bind is applied last, wins)
    assert argv.index("--tmpfs") < argv.index("--bind")


def test_none_backend_unchanged_with_deny_reads():
    from swarm.bridges.worktree.sandbox_launch import wrap_command as wc

    assert wc(CMD, sandbox_path="/box", backend="none") == CMD


def test_executor_resolves_secret_denials_from_config(tmp_path):
    """Config gates + extends the deny list; nonexistent paths are dropped
    (bwrap would fail mounting tmpfs over a missing target)."""
    from swarm.bridges.worktree.config import WorktreeConfig
    from swarm.bridges.worktree.executor import SandboxExecutor
    from swarm.bridges.worktree.policy import WorktreePolicy
    from swarm.bridges.worktree.sandbox import SandboxManager

    real = tmp_path / "real-secret"
    real.mkdir()
    missing = tmp_path / "not-there"

    cfg = WorktreeConfig(
        os_isolation_extra_secret_paths=[str(real), str(missing)],
    )
    ex = SandboxExecutor(cfg, SandboxManager(cfg), WorktreePolicy(cfg))
    resolved = ex._resolve_secret_read_denials()
    assert str(real) in resolved
    assert str(missing) not in resolved

    cfg_off = WorktreeConfig(os_isolation_deny_secret_reads=False)
    ex_off = SandboxExecutor(cfg_off, SandboxManager(cfg_off), WorktreePolicy(cfg_off))
    assert ex_off._resolve_secret_read_denials() == []


@_needs_backend
def test_secret_named_read_blocked_live(tmp_path):
    """The dotenv/pem name-regex denial must actually block, not merely
    appear in the profile — the profile-text assertion previously passed
    while _sbpl_string's backslash escaping made the regex inert."""
    import subprocess

    from swarm.bridges.worktree.sandbox_launch import wrap_command as wc

    outside = tmp_path / "secrets"
    outside.mkdir()
    envf = outside / ".env"
    envf.write_text("SECRET=1")
    box = tmp_path / "box"
    box.mkdir()

    argv = wc(["cat", str(envf)], sandbox_path=str(box), backend=_BACKEND,
              deny_read_paths=[])
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert r.returncode != 0
    assert "SECRET" not in r.stdout


@_needs_backend
def test_sandbox_own_dotenv_stays_readable_live(tmp_path):
    """The task's own files win over the name denial — requires the profile
    to realpath the sandbox (macOS /var -> /private/var symlink previously
    voided the sandbox's read allow)."""
    import subprocess

    from swarm.bridges.worktree.sandbox_launch import wrap_command as wc

    box = tmp_path / "box"
    box.mkdir()
    own = box / ".env"
    own.write_text("TASKENV=1")
    argv = wc(["cat", str(own)], sandbox_path=str(box), backend=_BACKEND,
              deny_read_paths=[])
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "TASKENV" in r.stdout


@_needs_backend
def test_default_secret_dir_read_blocked_live(tmp_path):
    """deny_read_paths blocks reads of a (stand-in) credential dir."""
    import subprocess

    from swarm.bridges.worktree.sandbox_launch import wrap_command as wc

    creds = tmp_path / "fake-ssh"
    creds.mkdir()
    key = creds / "id_ed25519"
    key.write_text("PRIVATE")
    box = tmp_path / "box"
    box.mkdir()
    argv = wc(["cat", str(key)], sandbox_path=str(box), backend=_BACKEND,
              deny_read_paths=[str(creds)])
    r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert r.returncode != 0
    assert "PRIVATE" not in r.stdout
