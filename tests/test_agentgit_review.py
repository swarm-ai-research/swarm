"""Tests for the multi-agent review layer (panel, reviewers, synthesis)."""

import subprocess
from pathlib import Path

import pytest

from swarm.agentgit.__main__ import main
from swarm.agentgit.bundle import verify_bundle
from swarm.agentgit.git import collect_snapshot
from swarm.agentgit.review import (
    LOW_CONFIDENCE,
    VERDICT_APPROVE,
    VERDICT_BLOCK,
    VERDICT_WARN,
    DependencyReviewer,
    ReviewContext,
    ReviewerReport,
    ReviewFinding,
    SecurityReviewer,
    TestCoverageReviewer,
    collect_added_lines,
    run_review_panel,
    synthesize,
    verdict_for,
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


def _ctx(repo: Path) -> ReviewContext:
    return ReviewContext(
        snapshot=collect_snapshot(repo),
        added_lines=collect_added_lines(repo),
    )


# ---------------------------------------------------------------------------
# reviewers
# ---------------------------------------------------------------------------
def test_test_coverage_flags_untested_source_change(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "module.py").write_text("def f():\n    return 1\n")

    findings = TestCoverageReviewer().review(_ctx(repo))
    assert [f.rule for f in findings] == ["untested-change"]
    assert findings[0].severity == "warning"


def test_test_coverage_approves_source_with_tests(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "module.py").write_text("def f():\n    return 1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_module.py").write_text("def test_f():\n    pass\n")

    assert TestCoverageReviewer().review(_ctx(repo)) == []


def test_test_coverage_blocks_test_deletion(tmp_path):
    repo = _init_repo(tmp_path)
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_module.py").write_text("def test_f():\n    pass\n")
    _git(repo, "add", "tests")
    _git(repo, "commit", "-m", "add test")
    (tests_dir / "test_module.py").unlink()

    findings = TestCoverageReviewer().review(_ctx(repo))
    rules = {f.rule: f for f in findings}
    assert rules["test-deletion"].severity == "blocking"


def test_dependency_reviewer_flags_lockfile_only_change(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "package-lock.json").write_text("{}\n")

    findings = DependencyReviewer().review(_ctx(repo))
    rules = {f.rule for f in findings}
    assert rules == {"dependency-change", "lockfile-only-change"}


def test_dependency_reviewer_manifest_plus_lockfile_not_blocking(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "package.json").write_text("{}\n")
    (repo / "package-lock.json").write_text("{}\n")

    findings = DependencyReviewer().review(_ctx(repo))
    assert [f.rule for f in findings] == ["dependency-change"]
    assert verdict_for(findings) == VERDICT_WARN


def test_security_reviewer_flags_secret_material_and_paths(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / ".env").write_text("SECRET=1\n")
    (repo / "config.py").write_text('api_key = "sk-live-abcdef123456"\n')

    findings = SecurityReviewer().review(_ctx(repo))
    rules = {f.rule for f in findings}
    assert "sensitive-path" in rules
    assert "hardcoded-credential" in rules
    assert verdict_for(findings) == VERDICT_BLOCK


def test_security_reviewer_flags_risky_calls_as_warning(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "runner.py").write_text(
        "import subprocess\nsubprocess.run(cmd, shell=True)\n"
    )

    findings = SecurityReviewer().review(_ctx(repo))
    assert {f.rule for f in findings} == {"shell-true"}
    assert findings[0].severity == "warning"


def test_security_reviewer_only_sees_added_lines(tmp_path):
    """Pre-existing risky code must not be attributed to the new diff."""

    repo = _init_repo(tmp_path)
    (repo / "legacy.py").write_text("import os\nos.system('ls')\n")
    _git(repo, "add", "legacy.py")
    _git(repo, "commit", "-m", "legacy")
    (repo / "clean.py").write_text("x = 1\n")

    assert SecurityReviewer().review(_ctx(repo)) == []


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
def _report(name: str, verdict: str, findings=()) -> ReviewerReport:
    return ReviewerReport(reviewer=name, verdict=verdict, findings=tuple(findings))


def test_synthesize_surfaces_disagreement():
    blocker = ReviewFinding(
        reviewer="security",
        rule="secret",
        severity="blocking",
        confidence=0.9,
        paths=("a.py",),
        summary="secret found",
    )
    synthesis = synthesize(
        [
            _report("test-coverage", VERDICT_APPROVE),
            _report("dependency", VERDICT_APPROVE),
            _report("security", VERDICT_BLOCK, [blocker]),
        ]
    )
    assert synthesis.verdict == VERDICT_BLOCK
    assert len(synthesis.disagreements) == 1
    sides = synthesis.disagreements[0]["sides"]
    assert sides[VERDICT_APPROVE] == ["dependency", "test-coverage"]
    assert sides[VERDICT_BLOCK] == ["security"]
    assert synthesis.blocking == (blocker,)


def test_synthesize_unanimous_approve_has_no_disagreement():
    synthesis = synthesize(
        [_report("a", VERDICT_APPROVE), _report("b", VERDICT_APPROVE)]
    )
    assert synthesis.verdict == VERDICT_APPROVE
    assert synthesis.disagreements == ()
    assert synthesis.blocking == ()
    assert synthesis.attention == ()


def test_synthesize_attention_includes_low_confidence_findings():
    unsure = ReviewFinding(
        reviewer="security",
        rule="eval-exec",
        severity="warning",
        confidence=LOW_CONFIDENCE - 0.1,
        paths=("a.py",),
        summary="eval added",
    )
    synthesis = synthesize([_report("security", VERDICT_WARN, [unsure])])
    assert synthesis.attention == (unsure,)
    assert synthesis.blocking == ()


def test_finding_validation():
    with pytest.raises(ValueError, match="severity"):
        ReviewFinding(
            reviewer="x", rule="r", severity="fatal", confidence=0.5,
            paths=(), summary="s",
        )
    with pytest.raises(ValueError, match="confidence"):
        ReviewFinding(
            reviewer="x", rule="r", severity="info", confidence=1.5,
            paths=(), summary="s",
        )


# ---------------------------------------------------------------------------
# panel + CLI
# ---------------------------------------------------------------------------
def test_run_review_panel_is_deterministic(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "module.py").write_text("def f():\n    return 1\n")
    (repo / "package-lock.json").write_text("{}\n")

    first = run_review_panel(repo).to_dict()
    second = run_review_panel(repo).to_dict()
    assert first == second
    assert first["verdict"] == VERDICT_BLOCK  # lockfile-only-change


def test_cli_review_text_and_exit_codes(tmp_path, capsys):
    repo = _init_repo(tmp_path)
    (repo / ".env").write_text("SECRET=1\n")

    assert main(["review", "--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "panel verdict: BLOCK" in out
    assert "sensitive-path" in out

    assert main(["review", "--repo", str(repo), "--warn-only"]) == 0


def test_cli_review_clean_diff_approves(tmp_path, capsys):
    repo = _init_repo(tmp_path)
    (repo / "notes.md").write_text("docs only\n")

    assert main(["review", "--repo", str(repo)]) == 0
    assert "panel verdict: APPROVE" in capsys.readouterr().out


def test_cli_attest_review_folds_panel_into_signed_bundle(tmp_path, capsys, monkeypatch):
    # bead qkzc: attest signs with AGENTGIT_SIGNING_KEY when set (a local .env
    # leaks it into xdist workers) while verify_bundle below uses the dev key.
    # Pin both sides to the dev key so the test is env-independent.
    monkeypatch.delenv("AGENTGIT_SIGNING_KEY", raising=False)
    repo = _init_repo(tmp_path)
    (repo / "module.py").write_text("def f():\n    return 1\n")
    policy_path = repo / "policy.yaml"
    policy_path.write_text("allowed_paths: ['**']\n")
    out_path = repo / ".agentgit" / "provenance.json"

    rc = main(
        [
            "attest",
            "--repo", str(repo),
            "--task", "issue-1",
            "--agent", "codex",
            "--policy", str(policy_path),
            "--output", str(out_path),
            "--review",
        ]
    )
    assert rc == 0

    import json

    bundle = json.loads(out_path.read_text())
    reviews = bundle["provenance"]["reviews"]
    reviewers = {r["reviewer"] for r in reviews}
    assert {"test-coverage", "dependency", "security", "panel-synthesis"} <= reviewers

    # The recorded panel outcome is inside the signed payload: tamper-evident.
    ok, _ = verify_bundle(bundle)
    assert ok
    reviews[0]["decision"] = "approve-FORGED"
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("payload_hash" in err for err in errors)
