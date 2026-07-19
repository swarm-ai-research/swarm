"""BeadsClient — read-only access to the GasTown .beads/ SQLite database.

Discovers bead lifecycle events by polling the beads database for state
transitions since the last poll.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

from swarm.bridges.gastown.events import GasTownEvent, GasTownEventType

logger = logging.getLogger(__name__)

# Allowlist of valid table names for SQL query interpolation.
# Table names cannot be parameterized in SQLite, so we use f-string interpolation
# with strict validation to prevent SQL injection.
_VALID_TABLES = frozenset({"issues", "beads"})

# Mapping from bead status strings to event types for state transitions.
_STATE_EVENT_MAP: Dict[str, GasTownEventType] = {
    "open": GasTownEventType.BEAD_CREATED,
    "assigned": GasTownEventType.BEAD_ASSIGNED,
    "in_progress": GasTownEventType.BEAD_IN_PROGRESS,
    "done": GasTownEventType.BEAD_COMPLETED,
    "closed": GasTownEventType.BEAD_COMPLETED,
    "blocked": GasTownEventType.BEAD_BLOCKED,
}


class BeadsClient:
    """Read-only client for the GasTown beads SQLite database.

    Opens the database in read-only mode and provides polling-based
    change detection.  Never writes to the database.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        uri = f"file:{db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._conn.row_factory = sqlite3.Row
        self._table = self._detect_table()
        # Validate table name is from allowlist (SQL injection prevention).
        # f-string interpolation on lines 78, 82, 88 is safe because _table
        # is guaranteed to be "issues" or "beads" (cannot be parameterized).
        if self._table not in _VALID_TABLES:
            raise ValueError(f"Invalid table name: {self._table!r}")
        # Track last-seen state per bead to detect transitions.
        self._last_states: Dict[str, str] = {}

    def _detect_table(self) -> str:
        """Return the table name holding bead/issue records.

        Real beads databases use an ``issues`` table; the original
        GasTown schema used ``beads``.  Try ``issues`` first.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('issues', 'beads')"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        if "issues" in tables:
            return "issues"
        if "beads" in tables:
            return "beads"
        raise RuntimeError(
            f"beads DB at {self._db_path} has neither 'issues' nor 'beads' table"
        )

    def get_beads(self, since: Optional[datetime] = None) -> List[dict]:
        """Return beads modified since *since* (or all if ``None``)."""
        cursor = self._conn.cursor()
        if since is not None:
            ts = since.isoformat()
            cursor.execute(
                f"SELECT * FROM {self._table} WHERE updated_at >= ? ORDER BY updated_at",
                (ts,),
            )
        else:
            cursor.execute(f"SELECT * FROM {self._table} ORDER BY updated_at")
        return [dict(row) for row in cursor.fetchall()]

    def get_bead(self, bead_id: str) -> Optional[dict]:
        """Look up a single bead by ID."""
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT * FROM {self._table} WHERE id = ?", (bead_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def poll_changes(self, last_poll: datetime) -> List[GasTownEvent]:
        """Diff bead state since *last_poll* and emit events for transitions."""
        beads = self.get_beads(since=last_poll)
        events: List[GasTownEvent] = []
        for bead in beads:
            bead_id = str(bead.get("id", ""))
            status = str(bead.get("status", "open"))
            prev = self._last_states.get(bead_id)
            if prev == status:
                continue  # No state change
            self._last_states[bead_id] = status
            event_type = _STATE_EVENT_MAP.get(status, GasTownEventType.BEAD_CREATED)
            events.append(
                GasTownEvent(
                    event_type=event_type,
                    agent_name=str(bead.get("assignee", "")),
                    bead_id=bead_id,
                    payload={
                        "status": status,
                        "title": bead.get("title", ""),
                        "previous_status": prev,
                    },
                )
            )
        return events

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
