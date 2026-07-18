"""SQLite-backed persistence for Agent API runs and posts.

Replaces the in-memory dicts with a durable store that survives restarts.
Uses a write-through cache for active (non-terminal) runs so hot-path
reads don't hit disk.

Schema is created lazily on first access (no migration tooling needed).
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from swarm.api.models.post import PostResponse
from swarm.api.models.run import (
    RunResponse,
    RunStatus,
    RunSummaryMetrics,
    RunVisibility,
)
from swarm.api.models.scenario import ScenarioResponse, ScenarioStatus
from swarm.api.models.simulation import (
    SimulationMode,
    SimulationResponse,
    SimulationStatus,
)

logger = logging.getLogger(__name__)

def _connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """Open a WAL-mode connection with busy timeout, FK enforcement, Row rows."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


# Default DB path — sits next to the runs/ directory at repo root.
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "runs" / "agent_api.db"

_SCHEMA_VERSION = 1

_CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    scenario_id   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',
    visibility    TEXT NOT NULL DEFAULT 'private',
    agent_id      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    completed_at  TEXT,
    progress      REAL DEFAULT 0.0,
    summary_json  TEXT,
    status_url    TEXT NOT NULL,
    public_url    TEXT,
    error         TEXT,
    warnings_json TEXT
);
"""

_CREATE_POSTS_TABLE = """
CREATE TABLE IF NOT EXISTS posts (
    post_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    agent_id      TEXT NOT NULL,
    title         TEXT NOT NULL,
    blurb         TEXT NOT NULL,
    key_metrics   TEXT NOT NULL DEFAULT '{}',
    tags          TEXT NOT NULL DEFAULT '[]',
    published_at  TEXT NOT NULL,
    run_url       TEXT,
    upvotes       INTEGER NOT NULL DEFAULT 0,
    downvotes     INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_VOTES_TABLE = """
CREATE TABLE IF NOT EXISTS votes (
    vote_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id   TEXT NOT NULL REFERENCES posts(post_id),
    agent_id  TEXT NOT NULL,
    direction INTEGER NOT NULL,  -- +1 upvote, -1 downvote
    voted_at  TEXT NOT NULL,
    UNIQUE(post_id, agent_id)
);
"""

_CREATE_SCENARIOS_TABLE = """
CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id        TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    description        TEXT NOT NULL,
    status             TEXT NOT NULL,
    validation_errors  TEXT NOT NULL DEFAULT '[]',
    submitted_at       TEXT NOT NULL,
    tags               TEXT NOT NULL DEFAULT '[]',
    resource_estimate  TEXT,
    yaml_content       TEXT
);
"""

_CREATE_PROPOSALS_TABLE = """
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id        TEXT PRIMARY KEY,
    title              TEXT NOT NULL,
    description        TEXT NOT NULL,
    policy_declaration TEXT NOT NULL DEFAULT '{}',
    target_scenarios   TEXT NOT NULL DEFAULT '[]',
    status             TEXT NOT NULL,
    proposer_id        TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    votes_for          INTEGER NOT NULL DEFAULT 0,
    votes_against      INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_PROPOSAL_VOTES_TABLE = """
CREATE TABLE IF NOT EXISTS proposal_votes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id  TEXT NOT NULL REFERENCES proposals(proposal_id),
    agent_id     TEXT NOT NULL,
    direction    INTEGER NOT NULL,
    voted_at     TEXT NOT NULL,
    UNIQUE(proposal_id, agent_id)
);
"""

_CREATE_SIMULATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS simulations (
    simulation_id       TEXT PRIMARY KEY,
    scenario_id         TEXT NOT NULL,
    status              TEXT NOT NULL,
    mode                TEXT NOT NULL,
    max_participants    INTEGER NOT NULL,
    current_participants INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    join_deadline       TEXT NOT NULL,
    config_overrides    TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_SIM_PARTICIPANTS_TABLE = """
CREATE TABLE IF NOT EXISTS simulation_participants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id   TEXT NOT NULL REFERENCES simulations(simulation_id),
    agent_id        TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'participant',
    joined_at       TEXT NOT NULL
);
"""

_CREATE_SIM_ACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS simulation_actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id   TEXT NOT NULL REFERENCES simulations(simulation_id),
    action_id       TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    step            INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,
    accepted        INTEGER NOT NULL DEFAULT 1,
    source          TEXT
);
"""

_CREATE_SIM_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS simulation_results (
    simulation_id   TEXT PRIMARY KEY REFERENCES simulations(simulation_id),
    results_json    TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_SIM_EXEC_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS simulation_execution_state (
    simulation_id   TEXT PRIMARY KEY REFERENCES simulations(simulation_id),
    state_json      TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);",
    "CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_posts_published ON posts(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_posts_agent ON posts(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_votes_post ON votes(post_id);",
    "CREATE INDEX IF NOT EXISTS idx_scenarios_status ON scenarios(status);",
    "CREATE INDEX IF NOT EXISTS idx_scenarios_submitted ON scenarios(submitted_at);",
    "CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);",
    "CREATE INDEX IF NOT EXISTS idx_proposals_created ON proposals(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_proposal_votes_proposal ON proposal_votes(proposal_id);",
    "CREATE INDEX IF NOT EXISTS idx_simulations_status ON simulations(status);",
    "CREATE INDEX IF NOT EXISTS idx_simulations_created ON simulations(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_sim_participants_sim ON simulation_participants(simulation_id);",
    "CREATE INDEX IF NOT EXISTS idx_sim_actions_sim ON simulation_actions(simulation_id);",
]


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class RunStore:
    """SQLite-backed run storage with in-memory cache for active runs."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Write-through cache: only active (non-terminal) runs stay here.
        self._cache: dict[str, RunResponse] = {}

        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS_TABLE)
            for idx in _CREATE_INDEXES:
                if "runs" in idx:
                    conn.execute(idx)
            # Migrate: add warnings_json column for existing DBs that predate it
            try:
                conn.execute("ALTER TABLE runs ADD COLUMN warnings_json TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists

    def _connect(self) -> sqlite3.Connection:
        return _connect_sqlite(self._db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, run: RunResponse) -> None:
        """Upsert a run into both cache and DB.

        Lock covers both cache and DB write to prevent inconsistency
        (security fix 1.7).
        """
        summary_json = (
            run.summary_metrics.model_dump_json() if run.summary_metrics else None
        )
        warnings_json = json.dumps(run.warnings) if run.warnings else None

        with self._lock:
            # Update cache for active runs
            terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
            if run.status not in terminal:
                self._cache[run.run_id] = run
            else:
                self._cache.pop(run.run_id, None)

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO runs
                        (run_id, scenario_id, status, visibility, agent_id,
                         created_at, started_at, completed_at, progress,
                         summary_json, status_url, public_url, error, warnings_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status=excluded.status,
                        started_at=excluded.started_at,
                        completed_at=excluded.completed_at,
                        progress=excluded.progress,
                        summary_json=excluded.summary_json,
                        error=excluded.error,
                        warnings_json=excluded.warnings_json
                    """,
                    (
                        run.run_id,
                        run.scenario_id,
                        run.status.value,
                        run.visibility.value,
                        run.agent_id,
                        _iso(run.created_at),
                        _iso(run.started_at),
                        _iso(run.completed_at),
                        run.progress,
                        summary_json,
                        run.status_url,
                        run.public_url,
                        run.error,
                        warnings_json,
                    ),
                )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> Optional[RunResponse]:
        """Fetch a run by ID. Checks cache first, then DB."""
        with self._lock:
            cached = self._cache.get(run_id)
            if cached is not None:
                return cached

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_by_agent(self, agent_id: str) -> list[RunResponse]:
        """List all runs for an agent (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def count_active(self, agent_id: str) -> int:
        """Count queued + running runs for an agent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE agent_id = ? AND status IN ('queued', 'running')",
                (agent_id,),
            ).fetchone()
        return row[0] if row else 0

    def total_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return row[0] if row else 0

    def get_multiple(self, run_ids: list[str]) -> list[RunResponse]:
        """Fetch multiple runs by IDs."""
        if not run_ids:
            return []
        placeholders = ",".join("?" * len(run_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs WHERE run_id IN ({placeholders})",
                run_ids,
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_run(self, row: sqlite3.Row) -> RunResponse:
        summary = None
        if row["summary_json"]:
            try:
                summary = RunSummaryMetrics.model_validate_json(row["summary_json"])
            except Exception:
                logger.warning("Corrupt summary_json for run %s", row["run_id"])
        warnings: Optional[list[str]] = None
        warnings_json = row["warnings_json"] if "warnings_json" in row.keys() else None
        if warnings_json:
            try:
                warnings = json.loads(warnings_json)
            except Exception:
                logger.warning("Corrupt warnings_json for run %s", row["run_id"])
        return RunResponse(
            run_id=row["run_id"],
            scenario_id=row["scenario_id"],
            status=RunStatus(row["status"]),
            visibility=RunVisibility(row["visibility"]),
            agent_id=row["agent_id"],
            created_at=_parse_dt(row["created_at"]),  # type: ignore[arg-type]
            started_at=_parse_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
            progress=row["progress"],
            summary_metrics=summary,
            status_url=row["status_url"],
            public_url=row["public_url"],
            error=row["error"],
            warnings=warnings,
        )


class PostStore:
    """SQLite-backed post storage with voting."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_POSTS_TABLE)
            conn.execute(_CREATE_VOTES_TABLE)
            for idx in _CREATE_INDEXES:
                if ("posts" in idx or "votes" in idx) and "proposal" not in idx:
                    conn.execute(idx)

    def _connect(self) -> sqlite3.Connection:
        return _connect_sqlite(self._db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, post: PostResponse) -> None:
        """Insert a new post."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO posts
                    (post_id, run_id, agent_id, title, blurb,
                     key_metrics, tags, published_at, run_url,
                     upvotes, downvotes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    upvotes=excluded.upvotes,
                    downvotes=excluded.downvotes
                """,
                (
                    post.post_id,
                    post.run_id,
                    post.agent_id,
                    post.title,
                    post.blurb,
                    json.dumps(post.key_metrics, default=str),
                    json.dumps(post.tags),
                    _iso(post.published_at),
                    post.run_url,
                    getattr(post, "upvotes", 0),
                    getattr(post, "downvotes", 0),
                ),
            )

    def vote(self, post_id: str, agent_id: str, direction: int) -> dict:
        """Cast or change a vote. direction: +1 (up) or -1 (down).

        Uses BEGIN IMMEDIATE to take a write lock upfront, preventing
        race conditions between concurrent voters (security fix 1.2).

        Vote counts are recomputed from the votes table rather than
        maintained as denormalized counters, eliminating drift.

        Returns {"upvotes": int, "downvotes": int, "your_vote": int|None}.
        """
        if direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")

        now = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            # BEGIN IMMEDIATE acquires a write lock upfront, preventing
            # concurrent vote operations from interleaving.
            conn.execute("BEGIN IMMEDIATE")

            existing = conn.execute(
                "SELECT direction FROM votes WHERE post_id = ? AND agent_id = ?",
                (post_id, agent_id),
            ).fetchone()

            if existing:
                if existing["direction"] == direction:
                    # Toggle off (remove vote)
                    conn.execute(
                        "DELETE FROM votes WHERE post_id = ? AND agent_id = ?",
                        (post_id, agent_id),
                    )
                    your_vote = None
                else:
                    # Switch vote
                    conn.execute(
                        "UPDATE votes SET direction = ?, voted_at = ? WHERE post_id = ? AND agent_id = ?",
                        (direction, now, post_id, agent_id),
                    )
                    your_vote = direction
            else:
                # New vote
                conn.execute(
                    "INSERT INTO votes (post_id, agent_id, direction, voted_at) VALUES (?, ?, ?, ?)",
                    (post_id, agent_id, direction, now),
                )
                your_vote = direction

            # Recompute counts from ground truth (votes table) to prevent drift
            up_row = conn.execute(
                "SELECT COUNT(*) FROM votes WHERE post_id = ? AND direction = 1",
                (post_id,),
            ).fetchone()
            down_row = conn.execute(
                "SELECT COUNT(*) FROM votes WHERE post_id = ? AND direction = -1",
                (post_id,),
            ).fetchone()

            upvotes = up_row[0] if up_row else 0
            downvotes = down_row[0] if down_row else 0

            conn.execute(
                "UPDATE posts SET upvotes = ?, downvotes = ? WHERE post_id = ?",
                (upvotes, downvotes, post_id),
            )

            conn.execute("COMMIT")

        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

        return {
            "post_id": post_id,
            "upvotes": upvotes,
            "downvotes": downvotes,
            "your_vote": your_vote,
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, post_id: str) -> Optional[PostResponse]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM posts WHERE post_id = ?", (post_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_post(row)

    def list_posts(
        self,
        *,
        tag: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[PostResponse]:
        """List posts newest first, with optional filters.

        Tag filtering is pushed into the SQL query using json_each()
        to avoid post-LIMIT filtering issues (security fix 3.2).
        """
        conditions = []
        params: list = []

        if agent_id:
            conditions.append("p.agent_id = ?")
            params.append(agent_id)

        if tag:
            # Use a subquery with json_each to filter tags in SQL
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(p.tags) WHERE json_each.value = ?)"
            )
            params.append(tag)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT p.* FROM posts p {where} ORDER BY p.published_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_post(r) for r in rows]

    def total_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM posts").fetchone()
        return row[0] if row else 0

    def get_vote(self, post_id: str, agent_id: str) -> Optional[int]:
        """Get an agent's vote on a post (+1, -1, or None)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT direction FROM votes WHERE post_id = ? AND agent_id = ?",
                (post_id, agent_id),
            ).fetchone()
        return row["direction"] if row else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_post(self, row: sqlite3.Row) -> PostResponse:
        try:
            key_metrics = json.loads(row["key_metrics"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt key_metrics for post %s", row["post_id"])
            key_metrics = {}
        try:
            tags = json.loads(row["tags"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt tags for post %s", row["post_id"])
            tags = []

        return PostResponse(
            post_id=row["post_id"],
            run_id=row["run_id"],
            agent_id=row["agent_id"],
            title=row["title"],
            blurb=row["blurb"],
            key_metrics=key_metrics,
            tags=tags,
            published_at=_parse_dt(row["published_at"]),  # type: ignore[arg-type]
            run_url=row["run_url"],
        )


class ScenarioStore:
    """SQLite-backed scenario storage."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_SCENARIOS_TABLE)
            for idx in _CREATE_INDEXES:
                if "scenarios" in idx:
                    conn.execute(idx)

    def _connect(self) -> sqlite3.Connection:
        return _connect_sqlite(self._db_path)

    def save(self, scenario: ScenarioResponse, yaml_content: Optional[str] = None) -> None:
        """Upsert a scenario, optionally storing raw YAML content."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scenarios
                    (scenario_id, name, description, status,
                     validation_errors, submitted_at, tags, resource_estimate,
                     yaml_content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scenario_id) DO UPDATE SET
                    name=excluded.name,
                    description=excluded.description,
                    status=excluded.status,
                    validation_errors=excluded.validation_errors,
                    tags=excluded.tags,
                    resource_estimate=excluded.resource_estimate,
                    yaml_content=excluded.yaml_content
                """,
                (
                    scenario.scenario_id,
                    scenario.name,
                    scenario.description,
                    scenario.status.value,
                    json.dumps(scenario.validation_errors),
                    _iso(scenario.submitted_at),
                    json.dumps(scenario.tags),
                    json.dumps(scenario.resource_estimate) if scenario.resource_estimate else None,
                    yaml_content,
                ),
            )

    def get(self, scenario_id: str) -> Optional[ScenarioResponse]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_scenario(row)

    def get_yaml(self, scenario_id: str) -> Optional[str]:
        """Return stored YAML content for a VALID scenario, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT yaml_content FROM scenarios WHERE scenario_id = ? AND status = ?",
                (scenario_id, ScenarioStatus.VALID.value),
            ).fetchone()
        if row is None:
            return None
        result: Optional[str] = row["yaml_content"]
        return result

    def list_scenarios(
        self,
        *,
        status: Optional[ScenarioStatus] = None,
        tag: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ScenarioResponse]:
        conditions: list[str] = []
        params: list = []

        if status is not None:
            conditions.append("s.status = ?")
            params.append(status.value)

        if tag is not None:
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(s.tags) WHERE json_each.value = ?)"
            )
            params.append(tag)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT s.* FROM scenarios s {where} ORDER BY s.submitted_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_scenario(r) for r in rows]

    def _row_to_scenario(self, row: sqlite3.Row) -> ScenarioResponse:
        try:
            validation_errors = json.loads(row["validation_errors"])
        except (json.JSONDecodeError, TypeError):
            validation_errors = []
        try:
            tags = json.loads(row["tags"])
        except (json.JSONDecodeError, TypeError):
            tags = []
        resource_estimate = None
        if row["resource_estimate"]:
            try:
                resource_estimate = json.loads(row["resource_estimate"])
            except (json.JSONDecodeError, TypeError):
                pass
        return ScenarioResponse(
            scenario_id=row["scenario_id"],
            name=row["name"],
            description=row["description"],
            status=ScenarioStatus(row["status"]),
            validation_errors=validation_errors,
            submitted_at=_parse_dt(row["submitted_at"]),  # type: ignore[arg-type]
            tags=tags,
            resource_estimate=resource_estimate,
        )


class ProposalStore:
    """SQLite-backed governance proposal storage with voting."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_PROPOSALS_TABLE)
            conn.execute(_CREATE_PROPOSAL_VOTES_TABLE)
            for idx in _CREATE_INDEXES:
                if "proposal" in idx:
                    conn.execute(idx)

    def _connect(self) -> sqlite3.Connection:
        return _connect_sqlite(self._db_path)

    def save(self, proposal: Any) -> None:
        """Upsert a proposal."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO proposals
                    (proposal_id, title, description, policy_declaration,
                     target_scenarios, status, proposer_id, created_at,
                     votes_for, votes_against)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    title=excluded.title,
                    description=excluded.description,
                    status=excluded.status,
                    votes_for=excluded.votes_for,
                    votes_against=excluded.votes_against
                """,
                (
                    proposal.proposal_id,
                    proposal.title,
                    proposal.description,
                    json.dumps(proposal.policy_declaration),
                    json.dumps(proposal.target_scenarios),
                    proposal.status.value if hasattr(proposal.status, "value") else proposal.status,
                    proposal.proposer_id,
                    _iso(proposal.created_at),
                    proposal.votes_for,
                    proposal.votes_against,
                ),
            )

    def get(self, proposal_id: str) -> Optional[dict]:
        """Fetch a proposal as a dict (matches ProposalResponse fields)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_proposals(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM proposals {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def vote(self, proposal_id: str, agent_id: str, direction: int) -> Optional[dict]:
        """Cast a vote. Returns updated counts or None if duplicate."""
        now = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")

            existing = conn.execute(
                "SELECT direction FROM proposal_votes WHERE proposal_id = ? AND agent_id = ?",
                (proposal_id, agent_id),
            ).fetchone()

            if existing:
                conn.execute("ROLLBACK")
                return None  # duplicate vote

            conn.execute(
                "INSERT INTO proposal_votes (proposal_id, agent_id, direction, voted_at) VALUES (?, ?, ?, ?)",
                (proposal_id, agent_id, direction, now),
            )

            # Recompute counts from ground truth
            up_row = conn.execute(
                "SELECT COUNT(*) FROM proposal_votes WHERE proposal_id = ? AND direction = 1",
                (proposal_id,),
            ).fetchone()
            down_row = conn.execute(
                "SELECT COUNT(*) FROM proposal_votes WHERE proposal_id = ? AND direction = -1",
                (proposal_id,),
            ).fetchone()

            votes_for = up_row[0] if up_row else 0
            votes_against = down_row[0] if down_row else 0

            conn.execute(
                "UPDATE proposals SET votes_for = ?, votes_against = ? WHERE proposal_id = ?",
                (votes_for, votes_against, proposal_id),
            )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

        return {
            "proposal_id": proposal_id,
            "votes_for": votes_for,
            "votes_against": votes_against,
        }

    def get_vote(self, proposal_id: str, agent_id: str) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT direction FROM proposal_votes WHERE proposal_id = ? AND agent_id = ?",
                (proposal_id, agent_id),
            ).fetchone()
        return row["direction"] if row else None

    def close_proposal(self, proposal_id: str, status: str) -> bool:
        """Update a proposal's status. Returns True if a row was updated."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE proposals SET status = ? WHERE proposal_id = ?",
                (status, proposal_id),
            )
            return cur.rowcount > 0

    def list_votes(self, proposal_id: str) -> list[dict]:
        """Return all votes for a proposal."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent_id, direction, voted_at FROM proposal_votes WHERE proposal_id = ? ORDER BY id",
                (proposal_id,),
            ).fetchall()
        return [
            {"agent_id": r["agent_id"], "direction": r["direction"], "voted_at": r["voted_at"]}
            for r in rows
        ]

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        try:
            policy_declaration = json.loads(row["policy_declaration"])
        except (json.JSONDecodeError, TypeError):
            policy_declaration = {}
        try:
            target_scenarios = json.loads(row["target_scenarios"])
        except (json.JSONDecodeError, TypeError):
            target_scenarios = []
        return {
            "proposal_id": row["proposal_id"],
            "title": row["title"],
            "description": row["description"],
            "policy_declaration": policy_declaration,
            "target_scenarios": target_scenarios,
            "status": row["status"],
            "proposer_id": row["proposer_id"],
            "created_at": row["created_at"],
            "votes_for": row["votes_for"],
            "votes_against": row["votes_against"],
        }


class SimulationStore:
    """SQLite-backed simulation storage with write-through cache."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, SimulationResponse] = {}
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_SIMULATIONS_TABLE)
            conn.execute(_CREATE_SIM_PARTICIPANTS_TABLE)
            conn.execute(_CREATE_SIM_ACTIONS_TABLE)
            conn.execute(_CREATE_SIM_RESULTS_TABLE)
            conn.execute(_CREATE_SIM_EXEC_STATE_TABLE)
            for idx in _CREATE_INDEXES:
                if "simulation" in idx or "sim_" in idx:
                    conn.execute(idx)

    def _connect(self) -> sqlite3.Connection:
        return _connect_sqlite(self._db_path)

    # ------------------------------------------------------------------
    # Simulation CRUD
    # ------------------------------------------------------------------

    def save(self, sim: SimulationResponse) -> None:
        """Upsert a simulation into both cache and DB."""
        with self._lock:
            terminal = {SimulationStatus.COMPLETED, SimulationStatus.CANCELLED}
            if sim.status not in terminal:
                self._cache[sim.simulation_id] = sim
            else:
                self._cache.pop(sim.simulation_id, None)

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO simulations
                        (simulation_id, scenario_id, status, mode,
                         max_participants, current_participants,
                         created_at, join_deadline, config_overrides)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(simulation_id) DO UPDATE SET
                        status=excluded.status,
                        current_participants=excluded.current_participants,
                        config_overrides=excluded.config_overrides,
                        join_deadline=excluded.join_deadline
                    """,
                    (
                        sim.simulation_id,
                        sim.scenario_id,
                        sim.status.value,
                        sim.mode.value,
                        sim.max_participants,
                        sim.current_participants,
                        _iso(sim.created_at),
                        _iso(sim.join_deadline),
                        json.dumps(
                            sim.config_overrides.model_dump()
                            if hasattr(sim.config_overrides, "model_dump")
                            else sim.config_overrides
                        ),
                    ),
                )

    def get(self, simulation_id: str) -> Optional[SimulationResponse]:
        """Fetch a simulation by ID. Cache-first."""
        with self._lock:
            cached = self._cache.get(simulation_id)
            if cached is not None:
                return cached

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM simulations WHERE simulation_id = ?", (simulation_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_sim(row)

    def list_simulations(
        self,
        *,
        status: Optional[SimulationStatus] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SimulationResponse]:
        conditions: list[str] = []
        params: list = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM simulations {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_sim(r) for r in rows]

    def count_active(self) -> int:
        """Count simulations in waiting or running state."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM simulations WHERE status IN (?, ?)",
                (SimulationStatus.WAITING.value, SimulationStatus.RUNNING.value),
            ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Participants
    # ------------------------------------------------------------------

    def add_participant(
        self, simulation_id: str, agent_id: str, role: str, joined_at: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO simulation_participants (simulation_id, agent_id, role, joined_at) VALUES (?, ?, ?, ?)",
                (simulation_id, agent_id, role, joined_at),
            )

    def get_participants(self, simulation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent_id, role, joined_at FROM simulation_participants WHERE simulation_id = ?",
                (simulation_id,),
            ).fetchall()
        return [{"agent_id": r["agent_id"], "role": r["role"], "joined_at": r["joined_at"]} for r in rows]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def add_action(self, simulation_id: str, record: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO simulation_actions
                    (simulation_id, action_id, agent_id, action_type, step, timestamp, accepted, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    simulation_id,
                    record["action_id"],
                    record["agent_id"],
                    record["action_type"],
                    record["step"],
                    record["timestamp"],
                    1 if record.get("accepted", True) else 0,
                    record.get("source"),
                ),
            )

    def get_action_history(self, simulation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT action_id, agent_id, action_type, step, timestamp, accepted, source "
                "FROM simulation_actions WHERE simulation_id = ? ORDER BY id",
                (simulation_id,),
            ).fetchall()
        return [
            {
                "action_id": r["action_id"],
                "agent_id": r["agent_id"],
                "action_type": r["action_type"],
                "step": r["step"],
                "timestamp": r["timestamp"],
                "accepted": bool(r["accepted"]),
                "source": r["source"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def save_results(self, simulation_id: str, results: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO simulation_results (simulation_id, results_json)
                   VALUES (?, ?)
                   ON CONFLICT(simulation_id) DO UPDATE SET results_json=excluded.results_json""",
                (simulation_id, json.dumps(results)),
            )

    def get_results(self, simulation_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT results_json FROM simulation_results WHERE simulation_id = ?",
                (simulation_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            result: dict[str, Any] = json.loads(row["results_json"])
            return result
        except (json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Execution state
    # ------------------------------------------------------------------

    def save_execution_state(self, simulation_id: str, state: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO simulation_execution_state (simulation_id, state_json)
                   VALUES (?, ?)
                   ON CONFLICT(simulation_id) DO UPDATE SET state_json=excluded.state_json""",
                (simulation_id, json.dumps(state)),
            )

    def get_execution_state(self, simulation_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM simulation_execution_state WHERE simulation_id = ?",
                (simulation_id,),
            ).fetchone()
        if row is None:
            return {}
        try:
            state: dict[str, Any] = json.loads(row["state_json"])
            return state
        except (json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_sim(self, row: sqlite3.Row) -> SimulationResponse:
        try:
            config_overrides = json.loads(row["config_overrides"])
        except (json.JSONDecodeError, TypeError):
            config_overrides = {}
        return SimulationResponse(
            simulation_id=row["simulation_id"],
            scenario_id=row["scenario_id"],
            status=SimulationStatus(row["status"]),
            mode=SimulationMode(row["mode"]),
            max_participants=row["max_participants"],
            current_participants=row["current_participants"],
            created_at=_parse_dt(row["created_at"]),  # type: ignore[arg-type]
            join_deadline=_parse_dt(row["join_deadline"]),  # type: ignore[arg-type]
            config_overrides=config_overrides,
        )
