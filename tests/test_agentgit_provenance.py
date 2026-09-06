"""Tests for the enriched provenance block (issue 8ll9)."""

import subprocess
from pathlib import Path

from swarm.agentgit.bundle import (
    SCHEMA_V1,
    CommandRecord,
    build_bundle,
    verify_bundle,
)
from swarm.agentgit.policy import AgentGitPolicy


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "a@b.c")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _allowed_diff(repo: Path) -> None:
    src = repo / "swarm"
    src.mkdir()
    (src / "ok.py").write_text("print('ok')\n")


def _bundle(repo: Path, **kwargs):
    return build_bundle(
        repo=repo,
        task_id="issue-prov",
        agent_id="codex",
        policy=AgentGitPolicy(allowed_paths=["swarm/**", "requirements.txt", "pyproject.toml"]),
        **kwargs,
    )


# --- CommandRecord ---


def test_command_record_to_dict_roundtrip():
    rec = CommandRecord(command=["pytest", "-q"], return_code=0, isolation="bwrap")
    d = rec.to_dict()
    assert d["command"] == ["pytest", "-q"]
    assert d["return_code"] == 0
    assert d["isolation"] == "bwrap"
    assert d["timed_out"] is False


def test_command_record_from_command_result_duck_typed():
    class FakeResult:
        command = ["python", "-c", "x"]
        return_code = 1
        isolation = "sandbox-exec"
        duration_seconds = 0.4
        timed_out = False

    rec = CommandRecord.from_command_result(FakeResult())
    assert list(rec.command) == ["python", "-c", "x"]
    assert rec.isolation == "sandbox-exec"
    assert rec.return_code == 1


# --- provenance block ---


def test_bundle_defaults_to_schema_v1_with_empty_provenance(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    bundle = _bundle(repo)
    assert bundle["schema_version"] == SCHEMA_V1
    prov = bundle["provenance"]
    assert prov["commands"] == []
    assert prov["dependency_changes"] == []
    assert set(prov) == {
        "commands",
        "environment",
        "dependency_changes",
        "sources",
        "reviews",
        "overrides",
    }
    ok, errors = verify_bundle(bundle)
    assert ok, errors


def test_provenance_records_all_inputs(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    bundle = _bundle(
        repo,
        commands=[CommandRecord(command=["pytest"], return_code=0, isolation="bwrap")],
        environment={"model": "claude-opus-4-7", "runtime": "python3.13"},
        sources=["https://example.com/spec"],
        reviews=[{"reviewer": "security", "decision": "approve"}],
        overrides=[{"by": "alice", "reason": "false positive"}],
    )
    prov = bundle["provenance"]
    assert prov["commands"][0]["command"] == ["pytest"]
    assert prov["commands"][0]["isolation"] == "bwrap"
    assert prov["environment"]["model"] == "claude-opus-4-7"
    assert prov["sources"] == ["https://example.com/spec"]
    assert prov["reviews"][0]["decision"] == "approve"
    assert prov["overrides"][0]["by"] == "alice"
    ok, errors = verify_bundle(bundle)
    assert ok, errors


def test_dependency_changes_detected_automatically(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    (repo / "requirements.txt").write_text("requests>=2\n")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    bundle = _bundle(repo)
    paths = {c["path"] for c in bundle["provenance"]["dependency_changes"]}
    assert "requirements.txt" in paths
    assert "pyproject.toml" in paths
    # A non-manifest file is not flagged as a dependency change.
    assert "swarm/ok.py" not in paths


# --- tamper-evidence: provenance is in the signed payload ---


def test_tampering_with_commands_fails_verification(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    bundle = _bundle(
        repo, commands=[CommandRecord(command=["pytest"], return_code=0)]
    )
    ok, _ = verify_bundle(bundle)
    assert ok
    # Forge the command log; the receipt payload_hash should no longer match.
    bundle["provenance"]["commands"][0]["command"] = ["rm", "-rf", "/"]
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("payload_hash does not match" in e for e in errors)


def test_verify_handles_malformed_receipt_gracefully():
    # verify_bundle runs over untrusted input; a bad receipt must return
    # (False, errors), not raise a pydantic ValidationError.
    bundle = {
        "schema_version": SCHEMA_V1,
        "task": {"task_id": "x"},
        "agent": {"agent_id": "y"},
        "git": {},
        "policy": {"passed": True, "decisions": []},
        "checks": {},
        "provenance": {},
        "receipt": {"not": "a valid receipt"},
    }
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("receipt failed schema validation" in e for e in errors)


def test_tampering_with_dependency_changes_fails_verification(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    (repo / "requirements.txt").write_text("requests>=2\n")
    bundle = _bundle(repo)
    ok, _ = verify_bundle(bundle)
    assert ok
    bundle["provenance"]["dependency_changes"] = []  # hide the dep change
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("payload_hash does not match" in e for e in errors)


# --- v0 backward compatibility ---


def test_legacy_v0_bundle_still_verifies(tmp_path):
    """A v0 bundle was hashed without provenance; reconstruction must match."""
    from dataclasses import asdict

    from swarm.agentgit.bundle import _seal_receipt, _snapshot_dict
    from swarm.agentgit.git import collect_snapshot

    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    policy = AgentGitPolicy(allowed_paths=["swarm/**"])
    snapshot = collect_snapshot(repo, base_ref="HEAD")
    decisions = policy.evaluate(snapshot, check_results={})
    git_dict = _snapshot_dict(snapshot)
    decision_dicts = [asdict(d) for d in decisions]

    # Hash a v0-style payload (no provenance) and assemble a v0 bundle by hand.
    v0_payload = {
        "task_id": "issue-legacy",
        "agent_id": "codex",
        "git": git_dict,
        "policy_decisions": decision_dicts,
        "checks": {},
    }
    receipt = _seal_receipt(
        payload=v0_payload,
        agent_id="codex",
        signing_key="0" * 64,
        decisions=decisions,
    )
    bundle = {
        "schema_version": "agentgit.provenance.v0",
        "task": {"task_id": "issue-legacy"},
        "agent": {"agent_id": "codex"},
        "git": git_dict,
        "policy": {"passed": True, "decisions": decision_dicts},
        "checks": {},
        "receipt": receipt.model_dump(mode="json"),
    }
    ok, errors = verify_bundle(bundle)
    assert ok, errors
