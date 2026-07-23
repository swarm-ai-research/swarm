"""Tests for structured, versioned, scoped operational memory."""

import json

import pytest

from swarm.agentgit.__main__ import main
from swarm.agentgit.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    return MemoryStore(repo, agent="session-1", home=tmp_path / "home")


# ---------------------------------------------------------------------------
# structured + versioned
# ---------------------------------------------------------------------------
def test_remember_and_recall(store):
    store.remember(
        scope="repo",
        kind="fragility",
        subject="swarm/core/proxy.py",
        body="sigmoid calibration breaks if weights are renormalized",
        author="session-1",
    )
    entries = store.recall("swarm/core/proxy.py")
    assert len(entries) == 1
    assert entries[0].kind == "fragility"
    assert entries[0].version == 1


def test_update_appends_new_version_and_keeps_history(store):
    first = store.remember(
        scope="repo", kind="constraint", subject="deploy",
        body="run migrations before deploy", author="session-1",
    )
    second = store.remember(
        scope="repo", kind="constraint", subject="deploy",
        body="run migrations before deploy, IN ORDER", author="session-2",
        entry_id=first.id,
    )
    assert second.version == 2
    assert second.created_at == first.created_at

    current = store.recall("deploy")
    assert len(current) == 1
    assert "IN ORDER" in current[0].body

    versions = store.history(first.id, scope="repo")
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].author == "session-1"


def test_storage_is_append_only_jsonl(store, tmp_path):
    entry = store.remember(
        scope="repo", kind="practice", subject="tests",
        body="prefer determinism over loosened assertions", author="s1",
    )
    store.remember(
        scope="repo", kind="practice", subject="tests",
        body="updated", author="s1", entry_id=entry.id,
    )
    lines = (tmp_path / "repo" / ".agentgit" / "memory.jsonl").read_text().splitlines()
    assert len(lines) == 2  # two versions, nothing rewritten
    assert json.loads(lines[0])["version"] == 1
    assert json.loads(lines[1])["version"] == 2


def test_retire_tombstones_but_preserves_history(store):
    entry = store.remember(
        scope="repo", kind="constraint", subject="lib-y",
        body="never use lib Y here", author="s1",
    )
    assert store.retire(scope="repo", entry_id=entry.id, author="s2", reason="lib fixed")
    assert store.recall("lib-y") == []
    versions = store.history(entry.id, scope="repo")
    assert versions[-1].retired
    assert versions[-1].details["retire_reason"] == "lib fixed"
    # Retiring twice is a no-op.
    assert not store.retire(scope="repo", entry_id=entry.id, author="s2")


def test_invalid_scope_and_kind_rejected(store):
    with pytest.raises(ValueError, match="scope"):
        store.remember(scope="global", kind="context", subject="x", body="b", author="a")
    with pytest.raises(ValueError, match="kind"):
        store.remember(scope="repo", kind="vibes", subject="x", body="b", author="a")


def test_agent_scope_requires_agent(tmp_path):
    store = MemoryStore(tmp_path, home=tmp_path / "home")  # no agent
    with pytest.raises(ValueError, match="agent"):
        store.remember(scope="agent", kind="context", subject="x", body="b", author="a")


# ---------------------------------------------------------------------------
# scoped
# ---------------------------------------------------------------------------
def test_scopes_are_separate_files_and_ordered_by_specificity(store, tmp_path):
    store.remember(scope="org", kind="practice", subject="swarm/core",
                   body="org note", author="s1")
    store.remember(scope="repo", kind="practice", subject="swarm/core",
                   body="repo note", author="s1")
    store.remember(scope="agent", kind="practice", subject="swarm/core",
                   body="agent note", author="s1")

    entries = store.recall("swarm/core")
    assert [e.scope for e in entries] == ["agent", "repo", "org"]
    assert (tmp_path / "home" / ".agentgit" / "org-memory.jsonl").exists()
    assert (
        tmp_path / "home" / ".agentgit" / "agent-memory" / "session-1.jsonl"
    ).exists()


def test_org_memory_shared_across_repos(tmp_path):
    home = tmp_path / "home"
    (tmp_path / "repo-a").mkdir()
    (tmp_path / "repo-b").mkdir()
    a = MemoryStore(tmp_path / "repo-a", home=home)
    b = MemoryStore(tmp_path / "repo-b", home=home)

    a.remember(scope="org", kind="constraint", subject="ci",
               body="deploys fail if migration order changes", author="s1")
    assert [e.body for e in b.recall("ci")] == [
        "deploys fail if migration order changes"
    ]
    # Repo-scoped memories do NOT leak across repos.
    a.remember(scope="repo", kind="context", subject="ci", body="repo-a only",
               author="s1")
    assert len(b.recall("ci")) == 1


# ---------------------------------------------------------------------------
# subject-aware recall
# ---------------------------------------------------------------------------
def test_recall_matches_hierarchically(store):
    store.remember(scope="repo", kind="fragility", subject="swarm/core",
                   body="module-level note", author="s1")

    # Asking about a file under the module surfaces the module note...
    assert len(store.recall("swarm/core/proxy.py")) == 1
    # ...and asking about the parent surfaces child-subject notes too.
    assert len(store.recall("swarm")) == 1
    # Unrelated paths (including name-prefix siblings) do not match.
    assert store.recall("swarm/core_extras") == []
    assert store.recall("docs") == []


def test_recall_for_paths_dedupes_across_changed_files(store):
    store.remember(scope="repo", kind="fragility", subject="swarm/core",
                   body="fragile", author="s1")
    store.remember(scope="repo", kind="failure", subject="swarm/metrics",
                   body="previous agent broke the reporters here", author="s1")

    entries = store.recall_for_paths(
        ["swarm/core/proxy.py", "swarm/core/payoff.py", "swarm/metrics/reporters.py"]
    )
    assert sorted(e.kind for e in entries) == ["failure", "fragility"]


def test_recall_filters_by_kind(store):
    store.remember(scope="repo", kind="fragility", subject="a", body="x", author="s1")
    store.remember(scope="repo", kind="practice", subject="b", body="y", author="s1")
    assert [e.kind for e in store.recall(kind="practice")] == ["practice"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_memory_roundtrip(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    home = str(tmp_path / "home")
    base = ["--repo", str(repo), "--home", home, "--agent", "s1"]

    rc = main([
        "memory", "remember", "swarm/core",
        "--kind", "fragility", "--body", "handle with care", *base,
    ])
    assert rc == 0

    rc = main(["memory", "recall", "swarm/core/proxy.py", *base])
    assert rc == 0
    out = capsys.readouterr().out
    assert "handle with care" in out
    assert "[repo/fragility]" in out

    rc = main(["memory", "history", "--id", "fragility-swarm-core", "--scope", "repo", *base])
    assert rc == 0
    assert "v1" in capsys.readouterr().out

    rc = main([
        "memory", "retire", "--id", "fragility-swarm-core",
        "--reason", "refactored", *base,
    ])
    assert rc == 0
    main(["memory", "recall", "swarm/core", *base])
    assert "no memories found" in capsys.readouterr().out


def test_cli_memory_remember_requires_subject_and_body(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    rc = main(["memory", "remember", "--repo", str(repo), "--home", str(tmp_path)])
    assert rc == 1
    assert "required" in capsys.readouterr().out
