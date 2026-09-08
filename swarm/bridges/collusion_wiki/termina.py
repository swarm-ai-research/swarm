"""Load the swarm.termina.digital incident database (bead lnaf).

The *ai-safety-lab incident db* (https://swarm.termina.digital/db/, CC0,
schema v9 on 2026-09-08) is a secondary synthesis over the same wikis the
export covers plus the paste sites, URL shorteners and sister wikis found
afterwards: 91,320 ``record`` rows across 154 venues, with actors, claims,
evidence and record-to-record edges. It carries **pointers, not bodies**
(``body_sha256`` / ``body_len`` only), so the reading pack stays the body
source (``reading_pack.py``). What it adds over the export:

* **post-export rows.** The export ends 2026-07-02; the db's ``live-rc``
  recent-changes rows run to 2026-09-07, including the moderator clean-up
  after the ZZZ Pages post and the first post-disclosure edits.
* **the wider venue set.** Every venue id maps to the bridge's wiki name
  through :func:`venue_to_wiki`; for the export's wikis the two agree.
* **row-level status.** ``claim.status`` distinguishes ``verified``,
  ``inferred``, ``reported`` and ``contradicted``; they are not
  interchangeable and are preserved on every :class:`TerminaClaim`.

The db's ``phase`` column is measured against each incident's own disclosure
date (dsewiki-2026-05 was disclosed 2026-09-03), so rows from July and August
are still ``pre-disclosure``. Callers wanting "what the export lacks" should
filter on time (``after=EXPORT_END``), not on phase.

Count mismatch to expect: the db has 54,612 dse rows against the export's
13,403 dse revisions, because it also holds each revision's ``save`` event,
5,217 deletions, 101 probes and 22,484 recent-changes rows. Filter on
``kind`` before comparing; ``kind="revision"`` matches the export one-to-one.

The snapshot is pinned by sha256 the way the scenario pins the export's
``db_sha256``; :func:`verify_snapshot` checks the file and the manifest's row
counts. Every text field is untrusted content: do not execute it or fetch
URLs it contains.

This module is a standalone loader. The collusion_wiki CLI and replay
pipeline do not load the Termina snapshot yet; the scenario's ``termina:``
block pins hashes for callers of this module.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from swarm.bridges.collusion_wiki.loader import WikiRevision

SQLITE_NAME = "incidents.sqlite"
MANIFEST_NAME = "manifest.json"

# Last revision in the collusion.wiki export the scenario pins (probier).
EXPORT_END = datetime(2026, 7, 2, 17, 51, 22, tzinfo=timezone.utc)

# Venue ids whose name in the db differs from the export's ``wiki`` field.
# The export's own wikis (dse, probier, fractal, dorfwiki) use the same id
# in both; anything not a wiki maps to "".
_VENUE_ALIASES: Dict[str, str] = {}

RECORD_KINDS = frozenset({
    "revision", "save", "delete", "revert", "probe", "rc-row",
    "paste", "shortlink", "package-file",
})

CLAIM_STATUSES = ("verified", "inferred", "reported", "contradicted")


@dataclass(frozen=True)
class TerminaVenue:
    id: str
    host: str
    path: str
    software: str
    kind: str  # wiki | paste | shortener | ...
    status: str  # live | archived | wiped | gone | candidate
    caveat: str


@dataclass(frozen=True)
class TerminaRecord:
    """One ``record`` row. Times are UTC; ``time`` is None when unparseable."""

    id: str
    venue: str
    wiki: str  # bridge wiki name; "" for non-wiki venues
    kind: str
    title: str
    actor_id: str  # "handle:dse:Name", "human:dse:118", ...; "" when unknown
    ip16: str  # first two octets from ip_actor_id; "" when absent
    time: Optional[datetime]
    time_precision: str  # second | minute | hour | day | month | ""
    phase: str  # pre-disclosure | post-report | post-press | unknown
    status: str  # live | deleted | wiped | placeholder
    source: str  # collusion-export | live-rc | shellac-pack | ...
    body_sha256: str
    body_len: Optional[int]
    content_kind: str
    incident_id: str
    campaign_id: str
    cluster_id: str

    @property
    def page_id(self) -> str:
        """Matches the export's ``page_id`` (``wiki/Page``) for wiki rows."""
        return f"{self.wiki}/{self.title}" if self.wiki and self.title else ""

    @property
    def actor_label(self) -> str:
        """The handle or human id without its kind/venue prefix."""
        return self.actor_id.rsplit(":", 1)[-1] if self.actor_id else ""

    @property
    def post_export(self) -> bool:
        return self.time is not None and self.time > EXPORT_END


@dataclass(frozen=True)
class TerminaActor:
    id: str
    kind: str  # handle | ip | ip16 | asn | model | developer | human | tracker
    name: str
    venue: str
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    notes: str


@dataclass(frozen=True)
class TerminaClaim:
    id: str
    subject_kind: str  # incident | campaign | cluster | venue | actor | record | edge
    subject_id: str
    text: str
    status: str  # verified | inferred | reported | contradicted
    basis: str
    made_by: str  # evidence id; "" = unsourced
    checked_by: str
    notes: str


def resolve_sqlite(db: Path) -> Path:
    """Accept the snapshot directory or the sqlite file itself."""
    if db.is_dir():
        db = db / SQLITE_NAME
    if not db.exists():
        raise FileNotFoundError(f"{SQLITE_NAME} not found at {db}")
    return db


def _connect(db: Path) -> sqlite3.Connection:
    # as_uri() percent-encodes spaces etc.; a hand-built file:{path} does not
    return sqlite3.connect(resolve_sqlite(db).resolve().as_uri() + "?mode=ro", uri=True)


def _quote_ident(name: str) -> str:
    """Double-quote a SQLite identifier, escaping embedded ``"``."""
    return '"' + name.replace('"', '""') + '"'


def _sql_in(column: str, values: Iterable[str], where: List[str], params: List[str]) -> bool:
    """Append ``column IN (...)``. Return False if ``values`` is empty.

    SQLite rejects ``IN ()``; an empty filter list is an empty result set.
    """
    items = sorted(set(values))
    if not items:
        return False
    where.append(f"{column} IN ({','.join('?' * len(items))})")
    params.extend(items)
    return True


def parse_time(s: Optional[str]) -> Optional[datetime]:
    """ISO-8601 at any precision the db uses (``2026-05``, ``...T20:23Z``, ...)."""
    if not s:
        return None
    raw = s[:-1] if s.endswith("Z") else s
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        if len(raw) == 7:  # YYYY-MM
            try:
                t = datetime.strptime(raw, "%Y-%m")
            except ValueError:
                return None
        else:
            return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def venue_to_wiki(venue: TerminaVenue) -> str:
    """The bridge's wiki name for a venue; "" for non-wiki venues."""
    if venue.kind != "wiki":
        return ""
    return _VENUE_ALIASES.get(venue.id, venue.id)


def load_venues(db: Path) -> Dict[str, TerminaVenue]:
    con = _connect(db)
    try:
        rows = con.execute(
            "SELECT id, host, path, software, kind, status, caveat FROM venue ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    return {
        r[0]: TerminaVenue(
            id=str(r[0]), host=str(r[1] or ""), path=str(r[2] or ""),
            software=str(r[3] or ""), kind=str(r[4] or ""), status=str(r[5] or ""),
            caveat=str(r[6] or ""),
        )
        for r in rows
    }


def wiki_venues(db: Path) -> Dict[str, str]:
    """venue id -> bridge wiki name, for wiki venues only."""
    out = {}
    for vid, v in load_venues(db).items():
        w = venue_to_wiki(v)
        if w:
            out[vid] = w
    return out


def iter_records(
    db: Path,
    *,
    venues: Optional[Sequence[str]] = None,
    wikis: Optional[Sequence[str]] = None,
    kinds: Optional[Sequence[str]] = None,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    phases: Optional[Sequence[str]] = None,
) -> Iterator[TerminaRecord]:
    """Records in ``(observed_time, id)`` order, filtered in SQL where possible.

    ``wikis`` selects by bridge wiki name (via :func:`wiki_venues`);
    ``venues`` by db venue id. ``after`` keeps ``time > after``; ``before``
    keeps ``time < before``; rows with no parseable time are dropped whenever
    either bound is given.
    """
    venue_wiki = wiki_venues(db)
    want: Optional[set[str]] = None
    if venues is not None:
        want = set(venues)
    if wikis is not None:
        chosen = {vid for vid, w in venue_wiki.items() if w in set(wikis)}
        want = chosen if want is None else want & chosen
    sql = (
        "SELECT id, venue_id, kind, title, actor_id, ip_actor_id, observed_time, "
        "time_precision, phase, status, source, body_sha256, body_len, content_kind, "
        "incident_id, campaign_id, cluster_id FROM record"
    )
    where: List[str] = []
    params: List[str] = []
    if want is not None and not _sql_in("venue_id", want, where, params):
        return
    if kinds is not None and not _sql_in("kind", kinds, where, params):
        return
    if phases is not None and not _sql_in("phase", phases, where, params):
        return
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY observed_time, id"
    con = _connect(db)
    try:
        for r in con.execute(sql, params):
            t = parse_time(r[6])
            if (after is not None or before is not None) and t is None:
                continue
            if after is not None and t is not None and not t > after:
                continue
            if before is not None and t is not None and not t < before:
                continue
            ip = str(r[5] or "")
            yield TerminaRecord(
                id=str(r[0]), venue=str(r[1]), wiki=venue_wiki.get(str(r[1]), ""),
                kind=str(r[2]), title=str(r[3] or ""), actor_id=str(r[4] or ""),
                ip16=ip.split(":", 1)[1] if ip.startswith("ip16:") else "",
                time=t, time_precision=str(r[7] or ""), phase=str(r[8] or ""),
                status=str(r[9] or ""), source=str(r[10] or ""),
                body_sha256=str(r[11] or ""), body_len=int(r[12]) if r[12] is not None else None,
                content_kind=str(r[13] or ""), incident_id=str(r[14] or ""),
                campaign_id=str(r[15] or ""), cluster_id=str(r[16] or ""),
            )
    finally:
        con.close()


def load_records(db: Path, **kw) -> List[TerminaRecord]:
    return list(iter_records(db, **kw))


def post_export_records(db: Path, *, wikis: Optional[Sequence[str]] = None,
                        kinds: Optional[Sequence[str]] = None) -> List[TerminaRecord]:
    """Wiki rows observed after the export's last revision.

    Defaults to every wiki venue. These are mostly ``rc-row`` entries from
    live recent-changes polling (minute precision, human moderator ids) plus
    the export's late deletions; there are no post-export ``revision`` rows.
    """
    ws = list(wikis) if wikis is not None else sorted(set(wiki_venues(db).values()))
    return load_records(db, wikis=ws, kinds=kinds, after=EXPORT_END)


def iter_actors(db: Path, *, venues: Optional[Sequence[str]] = None,
                kinds: Optional[Sequence[str]] = None) -> Iterator[TerminaActor]:
    sql = "SELECT id, kind, name, venue_id, first_seen, last_seen, notes FROM actor"
    where: List[str] = []
    params: List[str] = []
    if venues is not None and not _sql_in("venue_id", venues, where, params):
        return
    if kinds is not None and not _sql_in("kind", kinds, where, params):
        return
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"
    con = _connect(db)
    try:
        for r in con.execute(sql, params):
            yield TerminaActor(
                id=str(r[0]), kind=str(r[1]), name=str(r[2] or ""), venue=str(r[3] or ""),
                first_seen=parse_time(r[4]), last_seen=parse_time(r[5]), notes=str(r[6] or ""),
            )
    finally:
        con.close()


def iter_claims(db: Path, *, subject_kind: Optional[str] = None,
                subject_id: Optional[str] = None,
                statuses: Optional[Sequence[str]] = None) -> Iterator[TerminaClaim]:
    """Claims with their status preserved; never collapse statuses when counting."""
    sql = ("SELECT id, subject_kind, subject_id, text, status, basis, made_by, "
           "checked_by, notes FROM claim")
    where: List[str] = []
    params: List[str] = []
    if subject_kind is not None:
        where.append("subject_kind = ?")
        params.append(subject_kind)
    if subject_id is not None:
        where.append("subject_id = ?")
        params.append(subject_id)
    if statuses is not None and not _sql_in("status", statuses, where, params):
        return
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY subject_kind, subject_id, id"
    con = _connect(db)
    try:
        for r in con.execute(sql, params):
            yield TerminaClaim(
                id=str(r[0]), subject_kind=str(r[1]), subject_id=str(r[2]), text=str(r[3] or ""),
                status=str(r[4]), basis=str(r[5] or ""), made_by=str(r[6] or ""),
                checked_by=str(r[7] or ""), notes=str(r[8] or ""),
            )
    finally:
        con.close()


def claim_status_counts(claims: Sequence[TerminaClaim]) -> Dict[str, int]:
    """Per status, in the db's fixed order, zeros included."""
    out = dict.fromkeys(CLAIM_STATUSES, 0)
    for c in claims:
        out[c.status] = out.get(c.status, 0) + 1
    return out


def join_revisions(
    revs: Sequence[WikiRevision], records: Sequence[TerminaRecord],
) -> Dict[str, TerminaRecord]:
    """rev_id -> db ``revision`` row, keyed on ``(page_id, time)`` to the second.

    Same caveat as the reading pack: several export revisions saved in the
    same second on one page collapse onto one db row.
    """
    by_key: Dict[Tuple[str, datetime], TerminaRecord] = {}
    for rec in records:
        if rec.kind == "revision" and rec.time is not None and rec.page_id:
            by_key.setdefault((rec.page_id, rec.time), rec)
    out: Dict[str, TerminaRecord] = {}
    for r in revs:
        hit = by_key.get((r.page_id, r.time))
        if hit is not None:
            out[r.rev_id] = hit
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_snapshot(
    db: Path, *, sqlite_sha256: Optional[str] = None, manifest: Optional[Path] = None,
) -> Dict[str, object]:
    """Check the snapshot against a pinned sha256 and its manifest's row counts.

    ``manifest`` defaults to ``manifest.json`` beside the sqlite file and is
    skipped when absent. The manifest describes the JSONL export of the same
    build, so each ``<table>.jsonl`` row count must equal the table's row
    count. Raises ``ValueError`` on any mismatch; returns what was checked.
    """
    path = resolve_sqlite(db)
    report: Dict[str, object] = {"sqlite": str(path), "sqlite_sha256": sha256_of(path)}
    if sqlite_sha256 is not None and report["sqlite_sha256"] != sqlite_sha256:
        raise ValueError(
            f"{path.name}: sha256 {report['sqlite_sha256']} != pinned {sqlite_sha256}"
        )
    mpath = manifest if manifest is not None else path.parent / MANIFEST_NAME
    if not mpath.exists():
        report["manifest"] = None
        return report
    m = json.loads(mpath.read_text(encoding="utf-8"))
    report["manifest"] = str(mpath)
    report["schema_version"] = m.get("schema_version")
    report["generated_at"] = m.get("generated_at")
    con = _connect(path)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts: Dict[str, int] = {}
        for fname, meta in sorted(m.get("files", {}).items()):
            table = fname[:-len(".jsonl")] if fname.endswith(".jsonl") else fname
            if table not in tables:
                continue
            n = con.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0]
            if n != meta.get("rows"):
                raise ValueError(f"{table}: {n} rows in sqlite, manifest says {meta.get('rows')}")
            counts[table] = n
    finally:
        con.close()
    report["rows"] = counts
    return report
