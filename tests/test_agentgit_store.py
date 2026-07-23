"""Tests for durable AgentGit provenance storage (event log + git notes)."""

import json
import subprocess
from pathlib import Path

from swarm.agentgit.__main__ import main
from swarm.agentgit.bundle import build_bundle
from swarm.agentgit.policy import AgentGitPolicy
from swarm.agentgit.store import (
    append_to_log,
    attach_note,
    list_noted_commits,
    read_log,
    read_notes,
    store_bundle,
    verify_log,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "agentgit@example.com")
    _git(repo, "config", "user.name", "AgentGit Test")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def _policy() -> AgentGitPolicy:
    return AgentGitPolicy(
        allowed_paths=["**"],
        denied_paths=[".env*"],
        max_changed_files=10,
        max_added_lines=100,
        required_checks=[],
    )


def _make_bundle(repo: Path, task_id: str = "issue-1") -> dict:
    (repo / "change.py").write_text("print('ok')\n")
    return build_bundle(
        repo=repo, task_id=task_id, agent_id="codex", policy=_policy()
    )


def test_append_and_verify_log_chain(tmp_path):
    repo = _init_repo(tmp_path)
    log_path = repo / ".agentgit" / "log.jsonl"

    first = append_to_log(_make_bundle(repo, "issue-1"), log_path)
    second = append_to_log(_make_bundle(repo, "issue-2"), log_path)

    assert first["seq"] == 0
    assert second["seq"] == 1
    assert second["prev_hash"] == first["entry_hash"]

    ok, errors = verify_log(log_path)
    assert ok, errors


def test_verify_log_detects_edited_entry(tmp_path):
    repo = _init_repo(tmp_path)
    log_path = repo / ".agentgit" / "log.jsonl"
    append_to_log(_make_bundle(repo, "issue-1"), log_path)
    append_to_log(_make_bundle(repo, "issue-2"), log_path)

    lines = log_path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["bundle"]["task"]["task_id"] = "issue-FORGED"
    lines[0] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    log_path.write_text("\n".join(lines) + "\n")

    ok, errors = verify_log(log_path)
    assert not ok
    assert any("entry 0" in err for err in errors)


def test_verify_log_detects_truncation(tmp_path):
    repo = _init_repo(tmp_path)
    log_path = repo / ".agentgit" / "log.jsonl"
    append_to_log(_make_bundle(repo, "issue-1"), log_path)
    append_to_log(_make_bundle(repo, "issue-2"), log_path)

    # Drop the first entry: the survivor's seq/prev_hash no longer line up.
    lines = log_path.read_text().splitlines()
    log_path.write_text(lines[1] + "\n")

    ok, errors = verify_log(log_path)
    assert not ok


def test_verify_log_checks_bundle_signatures_with_key(tmp_path):
    repo = _init_repo(tmp_path)
    log_path = repo / ".agentgit" / "log.jsonl"
    entry = append_to_log(_make_bundle(repo), log_path)

    # Re-chain a forged bundle consistently: the chain passes, but the
    # bundle's receipt signature no longer matches under the signing key.
    entry["bundle"]["task"]["task_id"] = "issue-FORGED"
    from swarm.agentgit.store import _entry_hash, bundle_digest

    entry["bundle_sha256"] = bundle_digest(entry["bundle"])
    entry["entry_hash"] = _entry_hash(entry)
    log_path.write_text(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")

    ok_chain_only, _ = verify_log(log_path)
    assert ok_chain_only  # chain is internally consistent

    ok_signed, errors = verify_log(log_path, signing_key="0" * 64)
    assert not ok_signed
    assert any("payload_hash" in err for err in errors)


def test_attach_and_read_notes(tmp_path):
    repo = _init_repo(tmp_path)
    bundle = _make_bundle(repo, "issue-1")
    head = _git(repo, "rev-parse", "HEAD")

    sha = attach_note(repo, bundle)
    assert sha == head

    bundles = read_notes(repo, head)
    assert len(bundles) == 1
    assert bundles[0]["task"]["task_id"] == "issue-1"

    # Appending a second attestation accumulates rather than overwrites.
    attach_note(repo, _make_bundle(repo, "issue-2"))
    assert [b["task"]["task_id"] for b in read_notes(repo, head)] == [
        "issue-1",
        "issue-2",
    ]
    assert list_noted_commits(repo) == [head]


def test_notes_survive_clone(tmp_path):
    """Provenance notes travel with git history when the notes ref is pushed."""

    repo = _init_repo(tmp_path)
    attach_note(repo, _make_bundle(repo, "issue-1"))
    head = _git(repo, "rev-parse", "HEAD")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(repo), str(clone))
    _git(clone, "fetch", "origin", "refs/notes/agentgit:refs/notes/agentgit")

    bundles = read_notes(clone, head)
    assert len(bundles) == 1
    assert bundles[0]["task"]["task_id"] == "issue-1"


def test_store_bundle_dispatches_backends(tmp_path):
    repo = _init_repo(tmp_path)
    bundle = _make_bundle(repo)

    summary = store_bundle(bundle, repo=repo, backends=["log", "git-notes"])

    assert summary["log"]["seq"] == 0
    assert summary["git-notes"]["commit"] == _git(repo, "rev-parse", "HEAD")
    assert len(read_log(repo / ".agentgit" / "log.jsonl")) == 1
    assert len(read_notes(repo, "HEAD")) == 1


def test_cli_attest_store_and_history_verify(tmp_path, capsys):
    repo = _init_repo(tmp_path)
    (repo / "change.py").write_text("print('ok')\n")
    policy_path = repo / "policy.yaml"
    policy_path.write_text(
        "allowed_paths: ['**']\n"
        "denied_paths: []\n"
        "max_changed_files: 10\n"
        "max_added_lines: 100\n"
        "required_checks: []\n"
    )

    rc = main(
        [
            "attest",
            "--repo",
            str(repo),
            "--task",
            "issue-1",
            "--agent",
            "codex",
            "--policy",
            str(policy_path),
            "--output",
            str(repo / ".agentgit" / "provenance.json"),
            "--store",
            "log",
            "--store",
            "git-notes",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "stored [log]" in out
    assert "stored [git-notes]" in out

    rc = main(["history", "--repo", str(repo), "--verify"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS log chain verify" in out
    assert "[verified]" in out


def test_cli_history_verify_fails_on_tampered_log(tmp_path, capsys):
    repo = _init_repo(tmp_path)
    log_path = repo / ".agentgit" / "log.jsonl"
    append_to_log(_make_bundle(repo), log_path)

    entry = json.loads(log_path.read_text())
    entry["bundle"]["agent"]["agent_id"] = "impostor"
    log_path.write_text(json.dumps(entry) + "\n")

    rc = main(["history", "--repo", str(repo), "--verify"])
    assert rc == 1
    assert "FAIL log chain verify" in capsys.readouterr().out


def test_attach_note_requires_commit(tmp_path):
    repo = _init_repo(tmp_path)
    bundle = _make_bundle(repo)
    bundle["git"]["head_commit"] = None

    import pytest

    with pytest.raises(ValueError, match="head_commit"):
        attach_note(repo, bundle)
