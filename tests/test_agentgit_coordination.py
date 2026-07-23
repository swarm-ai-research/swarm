"""Tests for machine-speed coordination primitives (CoordinationBoard)."""

import concurrent.futures

import pytest

from swarm.agentgit.__main__ import main
from swarm.agentgit.coordination import CoordinationBoard


@pytest.fixture
def board(tmp_path):
    with CoordinationBoard(tmp_path / "coord.db") as b:
        yield b


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------
def test_claim_is_exclusive(board):
    assert board.claim("session-1", "beads-42").ok
    contested = board.claim("session-2", "beads-42")
    assert not contested.ok
    assert contested.holder == "session-1"


def test_reclaim_by_holder_is_idempotent(board):
    first = board.claim("session-1", "beads-42")
    again = board.claim("session-1", "beads-42")
    assert again.ok
    assert again.claim_id == first.claim_id


def test_yield_frees_the_task(board):
    board.claim("session-1", "beads-42")
    assert board.yield_claim("session-1", "beads-42")
    assert board.claim("session-2", "beads-42").ok


def test_complete_frees_and_records(board):
    board.claim("session-1", "beads-42")
    assert board.complete("session-1", "beads-42")
    assert board.active_claims() == []


def test_cannot_yield_unheld_task(board):
    assert not board.yield_claim("session-1", "beads-42")
    board.claim("session-1", "beads-42")
    assert not board.yield_claim("session-2", "beads-42")  # not the holder


# ---------------------------------------------------------------------------
# locks
# ---------------------------------------------------------------------------
def test_lock_conflicts_with_overlapping_prefix(board):
    assert board.lock("session-1", "swarm/core", reason="refactor").ok

    file_lock = board.lock("session-2", "swarm/core/proxy.py")
    assert not file_lock.ok
    assert file_lock.conflicts[0]["agent"] == "session-1"

    parent_lock = board.lock("session-2", "swarm")
    assert not parent_lock.ok


def test_sibling_locks_do_not_conflict(board):
    assert board.lock("session-1", "swarm/core").ok
    assert board.lock("session-2", "swarm/metrics").ok
    # And a name-prefix (not path-prefix) sibling is not an overlap:
    assert board.lock("session-3", "swarm/core_extras").ok


def test_own_overlapping_lock_is_allowed(board):
    assert board.lock("session-1", "swarm/core").ok
    assert board.lock("session-1", "swarm/core/proxy.py").ok


def test_release_unblocks_other_agents(board):
    board.lock("session-1", "swarm/core")
    assert board.release("session-1", "swarm/core")
    assert board.lock("session-2", "swarm/core/proxy.py").ok


def test_release_requires_holding(board):
    board.lock("session-1", "swarm/core")
    assert not board.release("session-2", "swarm/core")


# ---------------------------------------------------------------------------
# proposals / review / delegation
# ---------------------------------------------------------------------------
def test_propose_and_respond_first_wins(board):
    pid = board.propose(
        "session-1", "review", "please review the proxy change",
        task_id="beads-42", to_agent="session-2",
    )
    assert board.respond("session-2", pid, accept=True, response="lgtm")
    # Already resolved: second responder loses.
    assert not board.respond("session-3", pid, accept=False)


def test_open_proposals_addressing(board):
    board.propose("session-1", "plan", "split the epic", to_agent="session-2")
    board.propose("session-1", "delegate", "run the sweep")  # broadcast

    for_2 = board.open_proposals("session-2")
    assert [p["kind"] for p in for_2] == ["plan", "delegate"]
    for_3 = board.open_proposals("session-3")
    assert [p["kind"] for p in for_3] == ["delegate"]


def test_propose_rejects_unknown_kind(board):
    with pytest.raises(ValueError, match="kind"):
        board.propose("session-1", "vibe", "??")


# ---------------------------------------------------------------------------
# conflict detection + audit stream
# ---------------------------------------------------------------------------
def test_detect_conflicts_before_work_collides(board):
    board.lock("session-1", "swarm/core", reason="payoff refactor")

    conflicts = board.detect_conflicts(
        "session-2", ["swarm/core/proxy.py", "docs/notes.md"]
    )
    assert len(conflicts) == 1
    assert conflicts[0]["agent"] == "session-1"
    assert conflicts[0]["paths"] == ["swarm/core/proxy.py"]

    # The lock holder sees no conflict with itself.
    assert board.detect_conflicts("session-1", ["swarm/core/proxy.py"]) == []


def test_events_are_append_only_and_replayable(board):
    board.claim("session-1", "beads-42")
    board.lock("session-1", "swarm/core")
    board.complete("session-1", "beads-42")

    events = board.events()
    assert [e["action"] for e in events] == ["claim", "lock", "done"]
    assert events[0]["payload"] == {"task_id": "beads-42"}
    # Incremental reads resume from an id watermark.
    assert board.events(since_id=events[1]["id"]) == events[2:]


def _race_claim(db_path: str, agent: str) -> bool:
    with CoordinationBoard(db_path) as b:
        return b.claim(agent, "beads-race").ok


def test_claim_is_atomic_across_processes(tmp_path):
    db = str(tmp_path / "coord.db")
    CoordinationBoard(db).close()  # create schema first
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(_race_claim, [db] * 8, [f"session-{i}" for i in range(8)])
        )
    assert sum(results) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_coord_claim_and_contention(tmp_path, capsys):
    db = str(tmp_path / "coord.db")
    assert main(["coord", "claim", "beads-42", "--agent", "s1", "--db", db]) == 0
    assert main(["coord", "claim", "beads-42", "--agent", "s2", "--db", db]) == 1
    assert "HELD beads-42 by s1" in capsys.readouterr().out


def test_cli_coord_conflicts_and_status(tmp_path, capsys):
    db = str(tmp_path / "coord.db")
    main(["coord", "lock", "swarm/core", "--agent", "s1", "--db", db, "--reason", "wip"])
    rc = main(["coord", "conflicts", "swarm/core/proxy.py", "--agent", "s2", "--db", db])
    assert rc == 1
    assert "CONFLICT swarm/core held by s1" in capsys.readouterr().out

    assert main(["coord", "status", "--agent", "s2", "--db", db]) == 0
    assert "lock: swarm/core by s1: wip" in capsys.readouterr().out


def test_cli_coord_requires_agent_identity(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SESSION_ID", raising=False)
    db = str(tmp_path / "coord.db")
    assert main(["coord", "status", "--db", db]) == 1
    assert "no agent identity" in capsys.readouterr().out


def test_cli_coord_uses_session_id_env(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SESSION_ID", "session-env")
    db = str(tmp_path / "coord.db")
    assert main(["coord", "claim", "beads-42", "--db", db]) == 0
    with CoordinationBoard(db) as board:
        assert board.active_claims()[0]["agent"] == "session-env"


def test_cli_coord_propose_respond_roundtrip(tmp_path, capsys):
    db = str(tmp_path / "coord.db")
    main([
        "coord", "propose", "review the payoff change",
        "--kind", "review", "--to", "s2", "--agent", "s1", "--db", db,
    ])
    out = capsys.readouterr().out
    assert "PROPOSED review #1 -> s2" in out
    assert main([
        "coord", "respond", "1", "--accept", "--agent", "s2", "--db", db,
    ]) == 0


# ---------------------------------------------------------------------------
# atomic work-start claim entry point (duplication prevention)
# ---------------------------------------------------------------------------
def test_claim_refuses_task_held_by_another_session(tmp_path, monkeypatch, capsys):
    from swarm.agentgit.__main__ import main

    db = str(tmp_path / "runs.db")
    monkeypatch.setenv("MAIN_REPO_ROOT", str(tmp_path))

    monkeypatch.setenv("SESSION_ID", "session-A")
    assert main(["claim", "claim", "task-dup", "--db", db]) == 0
    capsys.readouterr()

    # A different session must be REFUSED (exit 2) — the duplication gate.
    monkeypatch.setenv("SESSION_ID", "session-B")
    assert main(["claim", "claim", "task-dup", "--db", db]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_claim_check_detects_foreign_holder(tmp_path, monkeypatch, capsys):
    from swarm.agentgit.__main__ import main

    db = str(tmp_path / "runs.db")
    monkeypatch.setenv("MAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("SESSION_ID", "session-A")
    main(["claim", "claim", "task-c", "--db", db])
    capsys.readouterr()

    # holder's own check passes; a foreign session's check fails (collision).
    assert main(["claim", "check", "task-c", "--db", db]) == 0
    monkeypatch.setenv("SESSION_ID", "session-B")
    assert main(["claim", "check", "task-c", "--db", db]) == 1
    assert "COLLISION" in capsys.readouterr().out


def test_claim_marker_written_and_cleared(tmp_path, monkeypatch):
    from swarm.agentgit.__main__ import main
    from swarm.agentgit.coordination import read_claim_marker

    db = str(tmp_path / "runs.db")
    monkeypatch.setenv("MAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("SESSION_ID", "session-A")

    main(["claim", "claim", "task-m", "--db", db])
    marker = read_claim_marker(tmp_path)
    assert marker == {"task_id": "task-m", "agent": "session-A"}

    main(["claim", "release", "task-m", "--db", db])
    assert read_claim_marker(tmp_path) is None


def test_release_lets_another_session_claim(tmp_path, monkeypatch):
    from swarm.agentgit.__main__ import main

    db = str(tmp_path / "runs.db")
    monkeypatch.setenv("MAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("SESSION_ID", "session-A")
    main(["claim", "claim", "task-r", "--db", db])
    main(["claim", "release", "task-r", "--db", db])

    monkeypatch.setenv("SESSION_ID", "session-B")
    assert main(["claim", "claim", "task-r", "--db", db]) == 0


def test_resolve_shared_db_prefers_main_repo_root(tmp_path, monkeypatch):
    from swarm.agentgit.coordination import resolve_shared_db_path

    monkeypatch.setenv("MAIN_REPO_ROOT", str(tmp_path))
    assert resolve_shared_db_path() == tmp_path / "runs" / "runs.db"
