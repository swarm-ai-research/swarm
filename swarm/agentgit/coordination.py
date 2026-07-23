"""Machine-speed coordination primitives for concurrent agent sessions.

GitHub issues and PRs are human-shaped: coarse, slow, and prose-first. When
15+ agent sessions work one repo concurrently, they need primitives that
operate at machine speed with machine semantics. This module extends the
``agent_messages`` convention (shared SQLite in ``runs/runs.db``) into real
coordination:

- **claim / yield / complete** — exclusive task ownership with atomic
  claim-or-tell-me-who-has-it semantics (no more ``CLAIM:`` message strings
  that two sessions can race past each other).
- **lock / release** — advisory locks on files, directories, or modules.
  Locks on a directory conflict with locks on anything under it.
- **propose / respond** — structured plan proposals, review requests, and
  subtask delegations addressed to another agent (or anyone).
- **detect_conflicts** — given the paths an agent is about to touch (e.g.
  from a :class:`~swarm.agentgit.git.GitSnapshot`), surface other agents'
  overlapping active locks *before* the work collides in a merge.

Every state transition is also appended to ``coordination_events`` — an
append-only audit stream, replayable per the repo's event-log invariants.
Writes use ``BEGIN IMMEDIATE`` transactions so claim/lock acquisition is
atomic across processes.

Merging compatible diffs is deliberately out of scope for v1: locks and
early conflict detection make merges boring; a diff-merging primitive can
layer on later without changing this schema.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_DB_PATH = "runs/runs.db"
# Marker written on claim / cleared on release, so the pre-commit gate can
# verify the committing session actually holds the task it is committing.
CLAIM_MARKER = ".agentgit/current-claim.json"


def resolve_shared_db_path() -> Path:
    """The one coordination DB every worktree of this repo shares.

    Claims must be visible across sessions, and concurrent sessions run in
    separate worktrees (each with its own `runs/`). Anchoring the DB at the
    repo's *common* git dir parent (``MAIN_REPO_ROOT``) gives all worktrees a
    single source of truth; falls back to the local default outside git.
    """

    main_root = os.environ.get("MAIN_REPO_ROOT")
    if main_root:
        return Path(main_root) / DEFAULT_DB_PATH
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if common:
            return Path(common).parent / DEFAULT_DB_PATH
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return Path(DEFAULT_DB_PATH)


def resolve_agent_id() -> str:
    """This session's coordination identity.

    ``SESSION_ID`` (set per worktree by detect-session.sh) is authoritative.
    Outside a worktree we fall back to a per-checkout id — concurrent code
    work in one checkout is prevented by the pre-commit worktree tripwire, so
    a shared fallback id there is acceptable (and makes the un-isolated case
    visibly a single "main-checkout" actor rather than silently forgeable).
    """

    sid = os.environ.get("SESSION_ID")
    if sid:
        return sid
    return f"main-checkout@{socket.gethostname()}"


def write_claim_marker(repo_root: Path, task_id: str, agent: str) -> None:
    marker = Path(repo_root) / CLAIM_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"task_id": task_id, "agent": agent}, sort_keys=True))


def read_claim_marker(repo_root: Path) -> Optional[Dict[str, str]]:
    marker = Path(repo_root) / CLAIM_MARKER
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text())
        return {"task_id": str(data["task_id"]), "agent": str(data["agent"])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def clear_claim_marker(repo_root: Path) -> None:
    (Path(repo_root) / CLAIM_MARKER).unlink(missing_ok=True)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  agent TEXT NOT NULL,
  task_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active'  -- active | yielded | done
);
CREATE INDEX IF NOT EXISTS idx_agent_claims_task_active
  ON agent_claims (task_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS agent_locks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  agent TEXT NOT NULL,
  resource TEXT NOT NULL,          -- normalized path/module prefix
  reason TEXT NOT NULL DEFAULT '',
  released INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_agent_locks_active
  ON agent_locks (resource) WHERE released = 0;

CREATE TABLE IF NOT EXISTS agent_proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  agent TEXT NOT NULL,             -- proposer
  kind TEXT NOT NULL,              -- plan | review | delegate
  task_id TEXT NOT NULL DEFAULT '',
  to_agent TEXT NOT NULL DEFAULT '#swarm',
  body TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',  -- open | accepted | rejected
  responder TEXT NOT NULL DEFAULT '',
  response TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_agent_proposals_open
  ON agent_proposals (to_agent) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS coordination_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  agent TEXT NOT NULL,
  action TEXT NOT NULL,
  payload TEXT NOT NULL            -- JSON
);
"""

_PROPOSAL_KINDS = {"plan", "review", "delegate"}


@dataclass(frozen=True)
class ClaimResult:
    ok: bool
    task_id: str
    holder: Optional[str] = None
    claim_id: Optional[int] = None


@dataclass(frozen=True)
class LockResult:
    ok: bool
    resource: str
    lock_id: Optional[int] = None
    conflicts: Sequence[Dict[str, Any]] = ()


def _normalize_resource(resource: str) -> str:
    """Normalize to a comparable prefix: posix separators, no trailing slash."""

    return resource.replace("\\", "/").rstrip("/")


def _overlaps(a: str, b: str) -> bool:
    """Two resources conflict when one is the other (or a parent of it)."""

    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


class CoordinationBoard:
    """Shared coordination state for concurrent agent sessions.

    One board per repo, backed by SQLite (defaults to the same ``runs/runs.db``
    the ``agent_messages`` table lives in). Safe for cross-process use: every
    acquire runs in a ``BEGIN IMMEDIATE`` transaction.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CoordinationBoard":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- claims -------------------------------------------------------------
    def claim(self, agent: str, task_id: str) -> ClaimResult:
        """Atomically claim ``task_id``; on contention, report the holder.

        Re-claiming a task you already hold is idempotent and succeeds.
        """

        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT id, agent FROM agent_claims "
                "WHERE task_id = ? AND status = 'active'",
                (task_id,),
            ).fetchone()
            if row is not None:
                if row["agent"] == agent:
                    return ClaimResult(True, task_id, holder=agent, claim_id=row["id"])
                return ClaimResult(False, task_id, holder=row["agent"])
            cur = self._conn.execute(
                "INSERT INTO agent_claims (agent, task_id) VALUES (?, ?)",
                (agent, task_id),
            )
            self._log(agent, "claim", {"task_id": task_id})
            return ClaimResult(True, task_id, holder=agent, claim_id=cur.lastrowid)

    def yield_claim(self, agent: str, task_id: str) -> bool:
        """Yield ownership so another agent can claim the task."""

        return self._end_claim(agent, task_id, "yielded")

    def complete(self, agent: str, task_id: str) -> bool:
        """Mark a claimed task done (also releases the claim)."""

        return self._end_claim(agent, task_id, "done")

    def _end_claim(self, agent: str, task_id: str, status: str) -> bool:
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            cur = self._conn.execute(
                "UPDATE agent_claims SET status = ? "
                "WHERE task_id = ? AND agent = ? AND status = 'active'",
                (status, task_id, agent),
            )
            if cur.rowcount:
                self._log(agent, status, {"task_id": task_id})
            return bool(cur.rowcount)

    def active_claims(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM agent_claims WHERE status = 'active' ORDER BY ts, id"
        ).fetchall()
        return [dict(r) for r in rows]

    def claim_holder(self, task_id: str) -> Optional[str]:
        """Agent currently holding ``task_id`` (read-only), or ``None``."""

        row = self._conn.execute(
            "SELECT agent FROM agent_claims WHERE task_id = ? AND status = 'active'",
            (task_id,),
        ).fetchone()
        return row["agent"] if row else None

    # -- locks --------------------------------------------------------------
    def lock(self, agent: str, resource: str, reason: str = "") -> LockResult:
        """Acquire an advisory lock on a file/directory/module prefix.

        Fails (without waiting) when another agent holds an overlapping active
        lock — a lock on ``swarm/core`` conflicts with one on
        ``swarm/core/proxy.py`` and vice versa. An agent's own overlapping
        locks do not conflict.
        """

        resource = _normalize_resource(resource)
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            conflicts = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT id, agent, resource, reason, ts FROM agent_locks "
                    "WHERE released = 0 AND agent != ?",
                    (agent,),
                )
                if _overlaps(_normalize_resource(row["resource"]), resource)
            ]
            if conflicts:
                return LockResult(False, resource, conflicts=conflicts)
            cur = self._conn.execute(
                "INSERT INTO agent_locks (agent, resource, reason) VALUES (?, ?, ?)",
                (agent, resource, reason),
            )
            self._log(agent, "lock", {"resource": resource, "reason": reason})
            return LockResult(True, resource, lock_id=cur.lastrowid)

    def release(self, agent: str, resource: str) -> bool:
        """Release this agent's active lock(s) on ``resource``."""

        resource = _normalize_resource(resource)
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            cur = self._conn.execute(
                "UPDATE agent_locks SET released = 1 "
                "WHERE agent = ? AND resource = ? AND released = 0",
                (agent, resource),
            )
            if cur.rowcount:
                self._log(agent, "release", {"resource": resource})
            return bool(cur.rowcount)

    def active_locks(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM agent_locks WHERE released = 0 ORDER BY ts, id"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- proposals / review / delegation ------------------------------------
    def propose(
        self,
        agent: str,
        kind: str,
        body: str,
        *,
        task_id: str = "",
        to_agent: str = "#swarm",
    ) -> int:
        """File a plan proposal, review request, or subtask delegation."""

        if kind not in _PROPOSAL_KINDS:
            raise ValueError(
                f"invalid proposal kind {kind!r}; expected one of {sorted(_PROPOSAL_KINDS)}"
            )
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            cur = self._conn.execute(
                "INSERT INTO agent_proposals (agent, kind, task_id, to_agent, body) "
                "VALUES (?, ?, ?, ?, ?)",
                (agent, kind, task_id, to_agent, body),
            )
            self._log(
                agent, "propose",
                {"proposal_id": cur.lastrowid, "kind": kind, "to_agent": to_agent},
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def respond(
        self, agent: str, proposal_id: int, accept: bool, response: str = ""
    ) -> bool:
        """Accept or reject an open proposal (first responder wins)."""

        status = "accepted" if accept else "rejected"
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            cur = self._conn.execute(
                "UPDATE agent_proposals SET status = ?, responder = ?, response = ? "
                "WHERE id = ? AND status = 'open'",
                (status, agent, response, proposal_id),
            )
            if cur.rowcount:
                self._log(
                    agent, "respond",
                    {"proposal_id": proposal_id, "status": status},
                )
            return bool(cur.rowcount)

    def open_proposals(self, for_agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Open proposals addressed to ``for_agent`` (plus broadcast), or all."""

        if for_agent is None:
            rows = self._conn.execute(
                "SELECT * FROM agent_proposals WHERE status = 'open' ORDER BY ts, id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM agent_proposals "
                "WHERE status = 'open' AND to_agent IN (?, '#swarm') ORDER BY ts, id",
                (for_agent,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- conflict detection --------------------------------------------------
    def detect_conflicts(
        self, agent: str, paths: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Other agents' active locks overlapping the paths this agent touches.

        Call before (or during) work with the changed paths of the working
        diff — e.g. ``[f.path for f in snapshot.changed_files]`` — to catch
        colliding work while it is still cheap to coordinate.
        """

        normalized = [_normalize_resource(p) for p in paths]
        conflicts: List[Dict[str, Any]] = []
        for row in self._conn.execute(
            "SELECT id, agent, resource, reason, ts FROM agent_locks "
            "WHERE released = 0 AND agent != ?",
            (agent,),
        ):
            locked = _normalize_resource(row["resource"])
            overlapping = [p for p in normalized if _overlaps(locked, p)]
            if overlapping:
                conflicts.append({**dict(row), "paths": overlapping})
        return conflicts

    # -- audit stream --------------------------------------------------------
    def events(self, since_id: int = 0) -> List[Dict[str, Any]]:
        """Read the append-only audit stream (optionally after ``since_id``)."""

        rows = self._conn.execute(
            "SELECT * FROM coordination_events WHERE id > ? ORDER BY id",
            (since_id,),
        ).fetchall()
        out = []
        for row in rows:
            entry = dict(row)
            entry["payload"] = json.loads(entry["payload"])
            out.append(entry)
        return out

    def _log(self, agent: str, action: str, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO coordination_events (agent, action, payload) VALUES (?, ?, ?)",
            (agent, action, json.dumps(payload, sort_keys=True)),
        )
