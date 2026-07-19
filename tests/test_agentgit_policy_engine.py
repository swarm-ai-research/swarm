"""Tests for the conditional policy engine + CI gate (issue b61g)."""

import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from swarm.agentgit.bundle import build_bundle, verify_bundle, write_bundle
from swarm.agentgit.policy import (
    AgentGitPolicy,
    ConditionalRule,
    PolicyFacts,
    decisions_passed,
    gate_bundle,
)

# The gate now requires an explicit signing key (no dev-key fallback), so CLI
# tests pass the dev key build_bundle signs with by default.
_KEY = "0" * 64


def _facts(**kw) -> PolicyFacts:
    base = {
        "changed_paths": [],
        "added_lines": 0,
        "changed_files": 0,
        "checks": {},
        "dependency_paths": [],
        "overridden_rules": set(),
    }
    base.update(kw)
    return PolicyFacts(**base)


# --- condition matching ---


def test_paths_match_condition():
    rule = ConditionalRule(id="r", when={"paths_match": ["**/auth/**"]}, action="deny")
    assert not rule.evaluate(_facts(changed_paths=["swarm/auth/x.py"])).passed
    assert rule.evaluate(_facts(changed_paths=["swarm/util/x.py"])).passed  # inert


def test_dependency_changed_condition():
    rule = ConditionalRule(
        id="r", when={"dependency_changed": True}, action="require_check", check="scan"
    )
    fired = rule.evaluate(_facts(dependency_paths=["requirements.txt"]))
    assert not fired.passed  # scan check missing
    assert rule.evaluate(_facts(dependency_paths=[])).passed  # inert


def test_added_lines_and_changed_files_conditions():
    big = ConditionalRule(id="b", when={"added_lines_gt": 100}, action="deny")
    assert not big.evaluate(_facts(added_lines=200)).passed
    assert big.evaluate(_facts(added_lines=50)).passed

    many = ConditionalRule(id="m", when={"changed_files_gt": 5}, action="deny")
    assert not many.evaluate(_facts(changed_files=10)).passed
    assert many.evaluate(_facts(changed_files=3)).passed


def test_check_failed_and_passed_conditions():
    on_fail = ConditionalRule(id="f", when={"check_failed": "pytest"}, action="deny")
    assert not on_fail.evaluate(_facts(checks={"pytest": False})).passed  # fired
    assert not on_fail.evaluate(_facts(checks={})).passed  # missing == failed
    assert on_fail.evaluate(_facts(checks={"pytest": True})).passed  # inert


def test_conditions_are_anded():
    rule = ConditionalRule(
        id="r",
        when={"paths_match": ["**/auth/**"], "added_lines_gt": 100},
        action="deny",
    )
    # Path matches but lines too few -> inert (passes).
    assert rule.evaluate(_facts(changed_paths=["swarm/auth/x.py"], added_lines=10)).passed
    # Both -> fires.
    assert not rule.evaluate(
        _facts(changed_paths=["swarm/auth/x.py"], added_lines=500)
    ).passed


# --- actions ---


def test_require_check_satisfied_by_passing_check():
    rule = ConditionalRule(
        id="r", when={"dependency_changed": True}, action="require_check", check="scan"
    )
    assert rule.evaluate(
        _facts(dependency_paths=["go.sum"], checks={"scan": True})
    ).passed


def test_require_review_needs_override():
    rule = ConditionalRule(id="r", when={"paths_match": ["*"]}, action="require_review")
    assert not rule.evaluate(_facts(changed_paths=["x"])).passed
    assert rule.evaluate(
        _facts(changed_paths=["x"], overridden_rules={"r"})
    ).passed


def test_override_unblocks_deny_by_rule_id():
    rule = ConditionalRule(id="dangerous", when={"paths_match": ["*"]}, action="deny")
    decision = rule.evaluate(_facts(changed_paths=["x"], overridden_rules={"dangerous"}))
    assert decision.passed
    assert decision.details["overridden"] is True
    # An override for a *different* rule id does not unblock this one.
    assert not rule.evaluate(
        _facts(changed_paths=["x"], overridden_rules={"other"})
    ).passed


def test_warning_severity_rule_does_not_block():
    rule = ConditionalRule(
        id="w", when={"paths_match": ["*"]}, action="deny", severity="warning"
    )
    decisions = [rule.evaluate(_facts(changed_paths=["x"]))]
    assert not decisions[0].passed
    assert decisions_passed(decisions)  # warnings never block


# --- yaml parsing ---


def test_from_yaml_parses_rules(tmp_path):
    policy_path = tmp_path / "p.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "allowed_paths: ['**']",
                "rules:",
                "  - id: deps-scan",
                "    when: {dependency_changed: true}",
                "    action: require_check",
                "    check: supply-chain",
                "  - id: big",
                "    when: {added_lines_gt: 1000}",
                "    action: require_review",
            ]
        )
        + "\n"
    )
    policy = AgentGitPolicy.from_yaml(policy_path)
    assert [r.id for r in policy.rules] == ["deps-scan", "big"]
    assert policy.rules[0].action == "require_check"
    assert policy.rules[0].check == "supply-chain"


# --- build-time integration + gate, end to end ---


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "a@b.c")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def test_build_time_rule_blocks_and_override_unblocks(tmp_path):
    repo = _repo(tmp_path)
    (repo / "requirements.txt").write_text("requests>=2\n")
    policy = AgentGitPolicy(
        allowed_paths=["**"],
        rules=[
            ConditionalRule(
                id="deps-scan",
                when={"dependency_changed": True},
                action="require_check",
                check="supply-chain",
            )
        ],
    )
    # No supply-chain check -> rule fires, bundle fails policy.
    bundle = build_bundle(repo=repo, task_id="t", agent_id="a", policy=policy)
    assert bundle["policy"]["passed"] is False
    assert any(d["policy_id"] == "rule:deps-scan" for d in bundle["policy"]["decisions"])
    verify_ok, _ = verify_bundle(bundle, require_policy_pass=True)
    assert not verify_ok

    # A recorded human override for the rule unblocks it.
    bundle_ok = build_bundle(
        repo=repo,
        task_id="t",
        agent_id="a",
        policy=policy,
        overrides=[{"rule": "deps-scan", "by": "alice", "reason": "vendored, scanned offline"}],
    )
    assert bundle_ok["policy"]["passed"] is True


def test_ci_gate_catches_violation_from_bundle_facts(tmp_path):
    repo = _repo(tmp_path)
    (repo / "auth.py").write_text("def login(): ...\n")

    # Agent self-attests with a lax policy (no rules) -> bundle passes.
    lax = AgentGitPolicy(allowed_paths=["**"])
    bundle = build_bundle(repo=repo, task_id="t", agent_id="a", policy=lax)
    assert bundle["policy"]["passed"] is True

    # CI owns a stricter policy: auth changes need security review.
    strict = AgentGitPolicy(
        allowed_paths=["**"],
        rules=[
            ConditionalRule(
                id="auth-review", when={"paths_match": ["*auth*"]}, action="require_review"
            )
        ],
    )
    ok, decisions = gate_bundle(bundle, strict)
    assert not ok  # the gate catches it from the bundle's recorded facts
    assert any(d.policy_id == "rule:auth-review" and not d.passed for d in decisions)


def test_ci_gate_passes_clean_bundle(tmp_path):
    repo = _repo(tmp_path)
    (repo / "swarm").mkdir()
    (repo / "swarm" / "ok.py").write_text("x = 1\n")
    policy = AgentGitPolicy(allowed_paths=["**"], rules=[])
    bundle = build_bundle(repo=repo, task_id="t", agent_id="a", policy=policy)
    ok, _ = gate_bundle(bundle, policy)
    assert ok


def test_from_dict_validates_action_severity_and_check():
    with pytest.raises(ValueError):
        ConditionalRule.from_dict({"id": "r", "action": "nope"})
    with pytest.raises(ValueError):
        ConditionalRule.from_dict({"id": "r", "severity": "loud"})
    with pytest.raises(ValueError):
        ConditionalRule.from_dict({"id": "r", "action": "require_check"})  # no check


def test_gate_ignores_bundle_supplied_overrides(tmp_path):
    # The bypass codex flagged: a bundle can pre-populate provenance.overrides
    # for the exact CI rule meant to catch it. The gate must NOT honor those.
    repo = _repo(tmp_path)
    (repo / "auth.py").write_text("def login(): ...\n")
    bundle = build_bundle(
        repo=repo,
        task_id="t",
        agent_id="a",
        policy=AgentGitPolicy(allowed_paths=["**"]),
        overrides=[{"rule": "auth-review", "by": "agent", "reason": "trust me"}],
    )
    ci_policy = AgentGitPolicy(
        allowed_paths=["**"],
        rules=[
            ConditionalRule(
                id="auth-review", when={"paths_match": ["*auth*"]}, action="require_review"
            )
        ],
    )
    # Bundle-supplied override is ignored -> still blocked.
    ok, _ = gate_bundle(bundle, ci_policy)
    assert not ok
    # Only an explicit CI-trusted override unblocks it.
    ok2, _ = gate_bundle(bundle, ci_policy, trusted_overrides=["auth-review"])
    assert ok2


def test_gate_ignores_bundle_supplied_checks(tmp_path):
    # The check-trust asymmetry: a bundle's `checks` are agent-authored, so a
    # check-based CI rule (tests-must-pass) must NOT trust them — otherwise an
    # agent defeats the gate by self-attesting checks={"pytest": True}.
    repo = _repo(tmp_path)
    (repo / "swarm").mkdir()
    (repo / "swarm" / "ok.py").write_text("x = 1\n")
    bundle = build_bundle(
        repo=repo,
        task_id="t",
        agent_id="a",
        policy=AgentGitPolicy(allowed_paths=["**"]),
        check_results={"pytest": True},  # agent claims tests passed
    )
    assert bundle["checks"] == {"pytest": True}
    ci_policy = AgentGitPolicy(
        allowed_paths=["**"],
        rules=[
            ConditionalRule(id="tests-must-pass", when={"check_failed": "pytest"}, action="deny")
        ],
    )
    # Bundle-supplied check is ignored -> unknown check fails closed -> blocked.
    ok, _ = gate_bundle(bundle, ci_policy)
    assert not ok
    # Only a CI-authoritative check result clears the rule.
    ok2, _ = gate_bundle(bundle, ci_policy, trusted_checks={"pytest": True})
    assert ok2


def test_gate_cli_uses_ci_authoritative_checks(tmp_path, capsys):
    from swarm.agentgit.__main__ import main

    repo = _repo(tmp_path)
    (repo / "swarm").mkdir()
    (repo / "swarm" / "ok.py").write_text("x = 1\n")
    bundle = build_bundle(
        repo=repo,
        task_id="t",
        agent_id="a",
        policy=AgentGitPolicy(allowed_paths=["**"]),
        check_results={"pytest": True},  # agent self-attests a pass
    )
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle, bundle_path)
    policy_path = tmp_path / "p.yaml"
    policy_path.write_text(
        'allowed_paths: ["**"]\n'
        "rules:\n"
        "  - id: tests-must-pass\n"
        '    when: {check_failed: pytest}\n'
        "    action: deny\n"
    )
    # Without a CI-supplied check, the gate fails closed despite the bundle's claim.
    rc = main(
        ["gate", "--bundle", str(bundle_path), "--policy", str(policy_path),
         "--signing-key", _KEY]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "rule:tests-must-pass" in out
    # CI supplies the real result -> rule clears.
    rc2 = main(
        [
            "gate",
            "--bundle",
            str(bundle_path),
            "--policy",
            str(policy_path),
            "--signing-key",
            _KEY,
            "--check",
            "pytest=pass",
        ]
    )
    assert rc2 == 0


def test_gate_cli_fails_closed_on_tampered_bundle(tmp_path, capsys):
    from swarm.agentgit.__main__ import main

    repo = _repo(tmp_path)
    (repo / "swarm").mkdir()
    (repo / "swarm" / "ok.py").write_text("x = 1\n")
    bundle = build_bundle(
        repo=repo, task_id="t", agent_id="a", policy=AgentGitPolicy(allowed_paths=["**"])
    )
    # Tamper with the recorded facts the gate would read.
    bundle["git"]["changed_files"].append(
        {"path": "secret.env", "status": "A", "additions": 1, "deletions": 0}
    )
    bundle_path = tmp_path / "tampered.json"
    write_bundle(bundle, bundle_path)
    policy_path = tmp_path / "p.yaml"
    policy_path.write_text('allowed_paths: ["**"]\n')

    rc = main(
        ["gate", "--bundle", str(bundle_path), "--policy", str(policy_path),
         "--signing-key", _KEY]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "failed verification" in out


def test_gate_cli_exit_codes(tmp_path, capsys):
    from swarm.agentgit.__main__ import main

    repo = _repo(tmp_path)
    (repo / "auth.py").write_text("def login(): ...\n")
    bundle = build_bundle(
        repo=repo, task_id="t", agent_id="a", policy=AgentGitPolicy(allowed_paths=["**"])
    )
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle, bundle_path)

    strict = tmp_path / "strict.yaml"
    strict.write_text(
        'allowed_paths: ["**"]\n'
        "rules:\n"
        "  - id: auth-review\n"
        '    when: {paths_match: ["*auth*"]}\n'
        "    action: require_review\n"
    )
    rc = main(
        ["gate", "--bundle", str(bundle_path), "--policy", str(strict),
         "--signing-key", _KEY]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL agentgit gate" in out
    assert "rule:auth-review" in out

    lax = tmp_path / "lax.yaml"
    lax.write_text('allowed_paths: ["**"]\n')
    assert main(
        ["gate", "--bundle", str(bundle_path), "--policy", str(lax),
         "--signing-key", _KEY]
    ) == 0


def test_gate_derives_dependencies_from_signed_diff_not_provenance(tmp_path):
    # A dependency_changed rule must fire from the signed diff, not the
    # provenance block (which is unsigned for schema v0 / can be stripped).
    repo = _repo(tmp_path)
    (repo / "requirements.txt").write_text("requests==2.0\n")
    bundle = build_bundle(
        repo=repo, task_id="t", agent_id="a", policy=AgentGitPolicy(allowed_paths=["**"])
    )
    # Simulate a bundle whose provenance hides the dependency change (e.g. a v0
    # bundle, or a stripped provenance block) — the diff still records it.
    bundle["provenance"]["dependency_changes"] = []
    assert any(
        f["path"] == "requirements.txt" for f in bundle["git"]["changed_files"]
    )

    ci_policy = AgentGitPolicy(
        allowed_paths=["**"],
        rules=[
            ConditionalRule(
                id="deps-need-supply-chain-scan",
                when={"dependency_changed": True},
                action="require_check",
                check="supply-chain-scan",
            )
        ],
    )
    # The rule still fires (scan check unsatisfied) despite the emptied provenance.
    ok, decisions = gate_bundle(bundle, ci_policy)
    assert not ok
    assert any(
        d.policy_id == "rule:deps-need-supply-chain-scan" and not d.passed
        for d in decisions
    )


def test_gate_cli_fails_closed_without_signing_key(tmp_path, capsys, monkeypatch):
    from swarm.agentgit.__main__ import main

    monkeypatch.delenv("AGENTGIT_SIGNING_KEY", raising=False)
    repo = _repo(tmp_path)
    (repo / "swarm").mkdir()
    (repo / "swarm" / "ok.py").write_text("x = 1\n")
    bundle = build_bundle(
        repo=repo, task_id="t", agent_id="a", policy=AgentGitPolicy(allowed_paths=["**"])
    )
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle, bundle_path)
    policy_path = tmp_path / "p.yaml"
    policy_path.write_text('allowed_paths: ["**"]\n')

    # No --signing-key and no AGENTGIT_SIGNING_KEY -> fail closed (no dev-key fallback).
    rc = main(["gate", "--bundle", str(bundle_path), "--policy", str(policy_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "no signing key" in out
    # With the key supplied, the same bundle gates normally.
    assert main(
        ["gate", "--bundle", str(bundle_path), "--policy", str(policy_path),
         "--signing-key", _KEY]
    ) == 0


def test_from_dict_validates_when_conditions():
    # Unknown condition key -> would silently drop scope and fire on everything.
    with pytest.raises(ValueError, match="unknown condition"):
        ConditionalRule.from_dict(
            {"id": "r", "when": {"path_match": ["*auth*"]}, "action": "deny"}
        )
    # Non-hashable value that would crash _matches at evaluate time.
    with pytest.raises(ValueError, match="check_failed"):
        ConditionalRule.from_dict(
            {"id": "r", "when": {"check_failed": ["pytest"]}, "action": "deny"}
        )
    # Wrong type for a list-valued condition.
    with pytest.raises(ValueError, match="paths_match"):
        ConditionalRule.from_dict(
            {"id": "r", "when": {"paths_match": "*auth*"}, "action": "deny"}
        )
    # bool is not a valid integer threshold.
    with pytest.raises(ValueError, match="added_lines_gt"):
        ConditionalRule.from_dict(
            {"id": "r", "when": {"added_lines_gt": True}, "action": "deny"}
        )
    # Valid conditions parse cleanly.
    rule = ConditionalRule.from_dict(
        {
            "id": "r",
            "when": {"paths_match": ["*auth*"], "added_lines_gt": 10},
            "action": "deny",
        }
    )
    assert rule.when == {"paths_match": ["*auth*"], "added_lines_gt": 10}


# --- attest-time vs CI-gate-time verdict parity (issue b61g) ---
#
# One rule engine serves both places: `AgentGitPolicy.evaluate` folds decisions
# into the signed bundle at attest time, and `gate_bundle` re-derives the same
# facts from the bundle at CI time. These tests pin the parity contract: given
# the same policy and the same trusted inputs, every decision — id, verdict,
# reason, details — is identical, so an agent can never see a different verdict
# locally than CI enforces.


def _parity_policy() -> AgentGitPolicy:
    return AgentGitPolicy(
        allowed_paths=["**"],
        max_changed_files=10,
        max_added_lines=1000,
        required_checks=["pytest"],
        rules=[
            ConditionalRule(
                id="deps-scan",
                when={"dependency_changed": True},
                action="require_check",
                check="supply-chain-scan",
            ),
            ConditionalRule(
                id="auth-review", when={"paths_match": ["*auth*"]}, action="require_review"
            ),
            ConditionalRule(
                id="tests-must-pass", when={"check_failed": "pytest"}, action="deny"
            ),
            ConditionalRule(
                id="large-warn",
                when={"added_lines_gt": 500},
                action="require_review",
                severity="warning",
            ),
        ],
    )


def _parity_repo(tmp_path: Path) -> Path:
    # A diff that exercises path-, dependency-, and check-conditioned rules.
    repo = _repo(tmp_path)
    (repo / "auth.py").write_text("def login(): ...\n")
    (repo / "requirements.txt").write_text("requests>=2\n")
    return repo


def test_attest_and_gate_verdicts_match_when_blocking(tmp_path):
    # No checks or overrides on either side: both paths must block, with
    # decision-for-decision identical output.
    repo = _parity_repo(tmp_path)
    policy = _parity_policy()
    bundle = build_bundle(repo=repo, task_id="t", agent_id="a", policy=policy)
    assert bundle["policy"]["passed"] is False

    ok, decisions = gate_bundle(bundle, policy)
    assert ok is bundle["policy"]["passed"]
    assert [asdict(d) for d in decisions] == bundle["policy"]["decisions"]


def test_attest_and_gate_verdicts_match_when_clean(tmp_path):
    # The same checks and overrides supplied on both sides (self-attested at
    # attest time, CI-trusted at gate time) yield identical passing verdicts.
    repo = _parity_repo(tmp_path)
    policy = _parity_policy()
    checks = {"pytest": True, "supply-chain-scan": True}
    overrides = [{"rule": "auth-review", "by": "alice", "reason": "reviewed"}]
    bundle = build_bundle(
        repo=repo,
        task_id="t",
        agent_id="a",
        policy=policy,
        check_results=checks,
        overrides=overrides,
    )
    assert bundle["policy"]["passed"] is True

    ok, decisions = gate_bundle(
        bundle, policy, trusted_checks=checks, trusted_overrides=["auth-review"]
    )
    assert ok is True
    assert [asdict(d) for d in decisions] == bundle["policy"]["decisions"]
