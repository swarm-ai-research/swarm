"""Tests for track-record reputation (ledger, keys, trust, policy feed)."""

import json
import subprocess
from pathlib import Path

import pytest

from swarm.agentgit.__main__ import main
from swarm.agentgit.bundle import build_bundle
from swarm.agentgit.policy import AgentGitPolicy, ConditionalRule, gate_bundle
from swarm.agentgit.reputation import (
    EVENT_MERGED,
    EVENT_ROLLED_BACK,
    EVENT_SECURITY_INCIDENT,
    ReputationKey,
    ReputationLedger,
)


def _bundle(agent_id="codex", model="fable-5", config_version="v3", **over):
    return {
        "task": {"task_id": over.get("task_id", "issue-1")},
        "agent": {"agent_id": agent_id},
        "git": {
            "repo_path": "/repo",
            "changed_files": over.get(
                "changed_files",
                [{"path": "swarm/core/proxy.py"}, {"path": "docs/notes.md"}],
            ),
        },
        "policy": {"passed": over.get("policy_passed", True)},
        "provenance": {
            "environment": {
                "model": model,
                "runtime": "python3.13",
                "toolchain": "claude-code-2.1",
                "config_version": config_version,
            },
            "reviews": over.get("reviews", []),
        },
    }


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------
def test_key_includes_runtime_and_config_version():
    a = ReputationKey.from_bundle(_bundle(config_version="v3"))
    b = ReputationKey.from_bundle(_bundle(config_version="v7"))
    assert a.agent_id == b.agent_id == "codex"
    assert a.key_id != b.key_id  # same agent, different config → different actor
    assert a.key_id.startswith("codex@")


def test_key_tolerates_sparse_bundles():
    key = ReputationKey.from_bundle({"agent": {"agent_id": "bare"}})
    assert key.agent_id == "bare"
    assert key.model == ""
    assert key.key_id.startswith("bare@")


# ---------------------------------------------------------------------------
# ledger + track record
# ---------------------------------------------------------------------------
@pytest.fixture
def ledger(tmp_path):
    return ReputationLedger(tmp_path / "reputation.jsonl")


def test_attestation_derives_facts_from_bundle(ledger):
    reviews = [
        {"reviewer": "security", "decision": "block",
         "findings": [{"severity": "blocking"}, {"severity": "warning"}]},
        {"reviewer": "test-coverage", "decision": "warn",
         "findings": [{"severity": "warning"}]},
    ]
    key = ledger.record_attestation(_bundle(reviews=reviews))

    record = ledger.track_record(key.key_id)
    assert record.attestations == 1
    assert record.policy_passes == 1
    assert record.review_findings == 3
    assert record.security_findings == 1  # only security + blocking
    assert record.domains == {"swarm": 1, "docs": 1}
    assert record.key["config_version"] == "v3"


def test_outcomes_move_trust(ledger):
    key = ledger.record_attestation(_bundle())
    neutral = ledger.track_record(key.key_id).trust

    ledger.record_outcome(key.key_id, EVENT_MERGED, task_id="issue-1")
    ledger.record_outcome(key.key_id, EVENT_MERGED, task_id="issue-2")
    up = ledger.track_record(key.key_id).trust
    assert up > neutral

    ledger.record_outcome(key.key_id, EVENT_ROLLED_BACK, task_id="issue-2")
    ledger.record_outcome(key.key_id, EVENT_SECURITY_INCIDENT, task_id="issue-2")
    down = ledger.track_record(key.key_id).trust
    assert down < up

    record = ledger.track_record(key.key_id)
    assert record.merges == 2
    assert record.rollbacks == 1
    assert record.security_incidents == 1
    assert record.rollback_rate == 0.5


def test_unknown_actor_starts_neutral_and_bad_history_sinks_below(ledger):
    key = ledger.record_attestation(_bundle(policy_passed=True))
    assert 0.4 < ledger.track_record(key.key_id).trust <= 0.6
    for _ in range(3):
        ledger.record_outcome(key.key_id, EVENT_SECURITY_INCIDENT)
    assert ledger.track_record(key.key_id).trust < 0.2


def test_invalid_outcome_rejected(ledger):
    with pytest.raises(ValueError, match="invalid outcome"):
        ledger.record_outcome("codex@abc", "agentgit_vibes")


def test_ledger_is_append_only_jsonl(ledger):
    key = ledger.record_attestation(_bundle())
    ledger.record_outcome(key.key_id, EVENT_MERGED)
    lines = ledger.path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "agentgit_attested"


def test_keys_for_agent_groups_variants(ledger):
    ledger.record_attestation(_bundle(config_version="v3"))
    ledger.record_attestation(_bundle(config_version="v7"))
    ledger.record_attestation(_bundle(agent_id="other"))
    assert len(ledger.keys_for_agent("codex")) == 2
    assert len(ledger.keys_for_agent("other")) == 1


# ---------------------------------------------------------------------------
# policy feed: low rep -> restricted scope
# ---------------------------------------------------------------------------
def _trust_policy(threshold=0.5):
    return AgentGitPolicy(
        rules=[
            ConditionalRule.from_dict(
                {"id": "low-trust-needs-review", "when": {"trust_below": threshold},
                 "action": "require_review"}
            )
        ]
    )


def test_trust_below_fires_for_unknown_actor():
    ok, decisions = gate_bundle(_bundle(), _trust_policy(), agent_trust=None)
    assert not ok
    fired = {d.policy_id: d for d in decisions}["rule:low-trust-needs-review"]
    assert not fired.passed  # unknown trust fails closed


def test_trust_below_respects_known_scores():
    ok, _ = gate_bundle(_bundle(), _trust_policy(0.5), agent_trust=0.8)
    assert ok
    ok, _ = gate_bundle(_bundle(), _trust_policy(0.5), agent_trust=0.2)
    assert not ok


def test_trust_below_rejects_bool_threshold():
    with pytest.raises(ValueError, match="number"):
        ConditionalRule.from_dict(
            {"id": "bad", "when": {"trust_below": True}, "action": "deny"}
        )


def test_ledger_trust_feeds_gate(ledger):
    bundle = _bundle()
    key = ledger.record_attestation(bundle)
    for _ in range(5):
        ledger.record_outcome(key.key_id, EVENT_MERGED)

    trust = ledger.trust_for_bundle(bundle)
    assert trust is not None and trust > 0.5
    ok, _ = gate_bundle(bundle, _trust_policy(0.5), agent_trust=trust)
    assert ok
    # A different config version has no history: unknown → gate fails.
    other = _bundle(config_version="v99")
    assert ledger.trust_for_bundle(other) is None


# ---------------------------------------------------------------------------
# CLI (record from a real attested bundle → outcome → show → gate)
# ---------------------------------------------------------------------------
def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_cli_end_to_end(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "T")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    (repo / "change.py").write_text("x = 1\n")

    bundle = build_bundle(
        repo=repo, task_id="issue-9", agent_id="codex",
        policy=AgentGitPolicy(allowed_paths=["**"]),
        environment={"model": "fable-5", "runtime": "py313", "config_version": "v3"},
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    ledger_path = str(tmp_path / "rep.jsonl")

    assert main(["reputation", "record", "--bundle", str(bundle_path),
                 "--ledger", ledger_path]) == 0
    key_id = capsys.readouterr().out.split()[-1]

    assert main(["reputation", "outcome", key_id, "--merged",
                 "--task", "issue-9", "--ledger", ledger_path]) == 0
    capsys.readouterr()

    assert main(["reputation", "show", "--ledger", ledger_path]) == 0
    out = capsys.readouterr().out
    assert key_id in out and "merges=1" in out

    # Exactly-one-outcome validation.
    assert main(["reputation", "outcome", key_id, "--merged", "--rolled-back",
                 "--ledger", ledger_path]) == 1


def test_cli_gate_reads_reputation_ledger(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AGENTGIT_SIGNING_KEY", "ab" * 32)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "T")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    (repo / "change.py").write_text("x = 1\n")

    bundle = build_bundle(
        repo=repo, task_id="issue-9", agent_id="codex",
        policy=AgentGitPolicy(allowed_paths=["**"]),
        signing_key="ab" * 32,
        environment={"model": "fable-5", "config_version": "v3"},
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "rules:\n"
        "  - id: low-trust-needs-review\n"
        "    when: {trust_below: 0.5}\n"
        "    action: require_review\n"
    )
    ledger_path = tmp_path / "rep.jsonl"

    # No history → unknown trust → rule fires → gate fails.
    rc = main(["gate", "--bundle", str(bundle_path), "--policy", str(policy_path),
               "--reputation-ledger", str(ledger_path)])
    assert rc == 1
    assert "low-trust-needs-review" in capsys.readouterr().out

    # Build a good record, then the same gate passes.
    ledger = ReputationLedger(ledger_path)
    key = ledger.record_attestation(bundle)
    for _ in range(5):
        ledger.record_outcome(key.key_id, EVENT_MERGED)
    rc = main(["gate", "--bundle", str(bundle_path), "--policy", str(policy_path),
               "--reputation-ledger", str(ledger_path)])
    assert rc == 0
