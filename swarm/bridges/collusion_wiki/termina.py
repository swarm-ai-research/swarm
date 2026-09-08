"""Load the ai-safety-lab incident db from swarm.termina.digital (bead lnaf).

``https://swarm.termina.digital/pub/`` publishes the incident db as one
JSONL file per table plus a ``manifest.json`` of row counts and sha256
hashes, and the same tables as a single ``incidents.sqlite`` (schema
version 9 on 2026-09-08; 22 tables; CC0-1.0, no keys). This loader reads
the sqlite: the ``venue``, ``record``, ``actor`` and ``claim`` tables.

Its value for this bridge is **time**. The collusion.wiki export ends on
2026-07-02; the db's ``record`` table carries the same 14,591 export
revisions (``source = collusion-export``) *and* live RecentChanges rows
(``rc-row``, ``source = live-rc``) that run to 2026-09-07, so the wikis'
post-disclosure activity is visible here and nowhere else in the bridge.
Wiki venue ids (``dse``, ``probier``, ``fractal``, ``dorfwiki``) are the
export's ``wiki`` names, so records project onto ``WikiRevision`` and the
existing identity modes and detectors run unchanged.

Read the counts with care. The db has 54,612 ``dse`` records against the
export's 13,403 revisions because it holds one row per RecentChanges
listing (minute precision, ``rc-row``), one per export revision (twice:
``revision`` and ``save``), and one per deletion. Bodies are **not**
carried (``body_sha256`` / ``body_path`` are pointers); the reading pack
(``reading_pack.py``) stays the body source. ``observed_time`` is UTC
throughout; ``time_zone`` records the source clock's zone, not a pending
correction. Every ``claim`` row has a ``status`` (verified, inferred,
reported, contradicted) and the manifest's ``provenance_rule`` says a row
with no evidence link is an assertion awaiting one: carry that status
through, never flatten it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

from swarm.bridges.collusion_wiki.loader import WikiRevision

log = logging.getLogger(__name__)

SQLITE_NAME = "incidents.sqlite"
MANIFEST_NAME = "manifest.json"

# Last revision in the collusion.wiki export the scenario pins; records
# observed after this are the db's post-disclosure contribution.
EXPORT_END = datetime(2026, 7, 2, 23, 59, 59, tzinfo=timezone.utc)

# record.kind values that are one saved edit of one page. ``save`` rows
# duplicate ``revision`` rows one-to-one (same export revision, no actor),
# so they are left out of the projection.
REVISION_KINDS = frozenset({"revision", "rc-row"})

# actor.kind values whose name is an editor label rather than an address.
LABEL_ACTOR_KINDS = frozenset({"handle", "human"})

# venue.id -> export wiki name; identity unless listed here.
VENUE_WIKI: Dict[str, str] = {}


@dataclass(frozen=True)
class Venue:
    id: str
    host: str
    path: str
    kind: str  # wiki | paste | shortener | url-as-storage | counter | ...
    software: str
    status: str  # live | archived | wiped | gone | candidate
    first_seen: str
    last_seen: str

    @property
    def wiki(self) -> str:
        return VENUE_WIKI.get(self.id, self.id)


@dataclass(frozen=True)
class TerminaRecord:
    id: str
    venue: str  # venue.id
    kind: str  # revision | rc-row | save | delete | paste | shortlink | probe | ...
    title: str
    time: Optional[datetime]  # observed_time, UTC; None when the db has none
    time_precision: str  # second | minute | hour | day | month | ""
    actor_kind: str  # handle | ip | ip16 | human | "" (joined from actor)
    actor: str  # actor.name; "" when unattributed
    ip16: str  # first two octets when the db knows them, else ""
    status: str  # live | deleted | wiped | placeholder
    phase: str  # pre-disclosure | post-report | post-press | unknown
    source: str  # collusion-export | live-rc | shellac-pack | ...
    body_sha256: str
    body_len: int
    external_id: str
    source_ref: str

    @property
    def wiki(self) -> str:
        return VENUE_WIKI.get(self.venue, self.venue)

    @property
    def page_id(self) -> str:
        return f"{self.wiki}/{self.title}"

    @property
    def post_disclosure(self) -> bool:
        return self.time is not None and self.time > EXPORT_END


@dataclass(frozen=True)
class Claim:
    id: str
    subject_kind: str  # incident | campaign | cluster | venue | actor | record | edge
    subject_id: str
    text: str
    status: str  # verified | inferred | reported | contradicted
    basis: str
    made_by: str  # evidence.id or ""
    checked_by: str  # evidence.id or ""
    notes: str

    @property
    def sourced(self) -> bool:
        return bool(self.made_by)


def resolve_sqlite(path: Path) -> Path:
    """Accept the bundle directory or the sqlite file itself."""
    if path.is_dir():
        path = path / SQLITE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{SQLITE_NAME} not found at {path}")
    return path


def _connect(path: Path) -> sqlite3.Connection:
    # as_uri() percent-encodes spaces etc.; a hand-built file:{path} does not
    return sqlite3.connect(resolve_sqlite(path).resolve().as_uri() + "?mode=ro", uri=True)


def parse_time(s: Optional[str]) -> Optional[datetime]:
    """The db mixes ``2026-05-24T06:05Z``, ``2026-05-12T00:53``, second and
    fractional-second forms and bare dates. All are UTC."""
    if not s:
        return None
    t = datetime.fromisoformat(s[:-1] if s.endswith("Z") else s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


# --- manifest ---------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(bundle: Path) -> dict:
    """``manifest.json`` beside the sqlite (or the bundle directory)."""
    d = bundle if bundle.is_dir() else bundle.parent
    doc = json.loads((d / MANIFEST_NAME).read_text(encoding="utf-8"))
    return dict(doc) if isinstance(doc, dict) else {}


def verify_pins(bundle: Path, pins: Mapping[str, str]) -> Dict[str, str]:
    """Compare a scenario's pinned sha256s against the bundle.

    ``pins`` keys are manifest file names (``record.jsonl`` ...) checked
    against the manifest's hashes, plus optionally ``incidents.sqlite``,
    checked against the file on disk. Returns ``name -> "ok" | "mismatch"
    | "absent"``; the caller decides whether a mismatch is fatal.
    """
    out: Dict[str, str] = {}
    try:
        files = load_manifest(bundle).get("files", {})
    except FileNotFoundError:
        files = {}
    for name, want in pins.items():
        if name == SQLITE_NAME:
            try:
                have: Optional[str] = sha256_of(resolve_sqlite(bundle))
            except FileNotFoundError:
                have = None
        else:
            have = (files.get(name) or {}).get("sha256")
        out[name] = "absent" if have is None else ("ok" if have == want else "mismatch")
    return out


# --- tables -----------------------------------------------------------------

def load_venues(bundle: Path, *, kind: Optional[str] = None) -> List[Venue]:
    con = _connect(bundle)
    try:
        sql = ("SELECT id, host, path, kind, software, status, first_seen, last_seen "
               "FROM venue")
        params: list = []
        if kind is not None:
            sql += " WHERE kind = ?"
            params = [kind]
        return [
            Venue(*(str(c or "") for c in row))
            for row in con.execute(sql + " ORDER BY id", params)
        ]
    finally:
        con.close()


def wiki_venues(bundle: Path) -> Dict[str, str]:
    """venue.id -> export wiki name for every ``kind = 'wiki'`` venue."""
    return {v.id: v.wiki for v in load_venues(bundle, kind="wiki")}


def _ip16(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]) else ""


def iter_records(
    bundle: Path,
    *,
    venues: Optional[Sequence[str]] = None,
    kinds: Optional[Sequence[str]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    phases: Optional[Sequence[str]] = None,
) -> Iterator[TerminaRecord]:
    """Records joined to their actor, optionally by venue / kind / phase / time.

    ``since`` is exclusive and ``until`` inclusive so that
    ``since=EXPORT_END`` yields exactly the rows the export lacks.
    """
    con = _connect(bundle)
    try:
        sql = (
            "SELECT r.id, r.venue_id, r.kind, r.title, r.observed_time, "
            "r.time_precision, a.kind, a.name, ipa.name, r.status, r.phase, "
            "r.source, r.body_sha256, r.body_len, r.external_id, r.source_ref "
            "FROM record r LEFT JOIN actor a ON a.id = r.actor_id "
            "LEFT JOIN actor ipa ON ipa.id = r.ip_actor_id"
        )
        where: List[str] = []
        params: list = []
        for col, vals in (("r.venue_id", venues), ("r.kind", kinds), ("r.phase", phases)):
            if vals is not None:
                where.append(f"{col} IN ({','.join('?' * len(vals))})")
                params.extend(vals)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY r.observed_time, r.id"
        for row in con.execute(sql, params):
            t = parse_time(row[4])
            if since is not None and (t is None or t <= since):
                continue
            if until is not None and (t is None or t > until):
                continue
            actor_kind = str(row[6] or "")
            actor = str(row[7] or "")
            ip16 = _ip16(str(row[8] or ""))
            if not ip16 and actor_kind in ("ip", "ip16"):
                ip16 = _ip16(actor)
            yield TerminaRecord(
                id=str(row[0]), venue=str(row[1]), kind=str(row[2]),
                title=str(row[3] or ""), time=t, time_precision=str(row[5] or ""),
                actor_kind=actor_kind, actor=actor, ip16=ip16,
                status=str(row[9] or ""), phase=str(row[10] or ""),
                source=str(row[11] or ""), body_sha256=str(row[12] or ""),
                body_len=int(row[13] or 0), external_id=str(row[14] or ""),
                source_ref=str(row[15] or ""),
            )
    finally:
        con.close()


def load_records(bundle: Path, **kw) -> List[TerminaRecord]:
    return list(iter_records(bundle, **kw))


def post_disclosure_records(
    bundle: Path,
    *,
    venues: Optional[Sequence[str]] = None,
    export_end: datetime = EXPORT_END,
) -> List[TerminaRecord]:
    """Saved edits on the wiki venues observed after the export ends.

    Defaults to every ``kind = 'wiki'`` venue and to ``REVISION_KINDS``, so
    deletions and the duplicate ``save`` rows are not counted as activity.
    """
    if venues is None:
        venues = sorted(wiki_venues(bundle))
    return load_records(bundle, venues=venues, kinds=sorted(REVISION_KINDS),
                        since=export_end)


def load_claims(
    bundle: Path,
    *,
    subject_kind: Optional[str] = None,
    subject_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Claim]:
    con = _connect(bundle)
    try:
        sql = ("SELECT id, subject_kind, subject_id, text, status, basis, made_by, "
               "checked_by, notes FROM claim")
        where: List[str] = []
        params: list = []
        for col, val in (("subject_kind", subject_kind), ("subject_id", subject_id),
                         ("status", status)):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [
            Claim(*(str(c or "") for c in row))
            for row in con.execute(sql + " ORDER BY subject_kind, subject_id, id", params)
        ]
    finally:
        con.close()


# --- projection -------------------------------------------------------------

def to_revisions(records: Sequence[TerminaRecord]) -> List[WikiRevision]:
    """Project saved edits onto the export's ``WikiRevision`` shape.

    ``label`` is the actor name when the actor is a handle or one of the db's
    numbered ``human`` ids (the wiki's own anonymous-editor number); an IP
    actor gives ``ip16`` and an empty label, so the ``label`` identity mode
    does not silently treat addresses as editors. ``rc-row`` times are minute
    precision: reply windows tighter than 60 s are meaningless on them.
    ``change_summary`` is empty and ``page_created`` is unknown (False).
    """
    out: List[WikiRevision] = []
    for r in records:
        if r.kind not in REVISION_KINDS or r.time is None or not r.title:
            continue
        out.append(WikiRevision(
            rev_id=r.id, wiki=r.wiki, page_id=r.page_id,
            label=r.actor if r.actor_kind in LABEL_ACTOR_KINDS else "",
            ip16=r.ip16, time=r.time, body_len=r.body_len,
            change_summary="", page_created=False,
        ))
    out.sort(key=lambda x: (x.time, x.rev_id))
    return out


def summary(bundle: Path, *, export_end: datetime = EXPORT_END) -> dict:
    """Per wiki venue: rows by kind, time span, and post-export saved edits."""
    wikis = wiki_venues(bundle)
    per: Dict[str, dict] = {}
    for r in iter_records(bundle, venues=sorted(wikis)):
        d = per.setdefault(r.venue, {"wiki": r.wiki, "by_kind": {}, "first": None,
                                     "last": None, "post_export_edits": 0})
        d["by_kind"][r.kind] = d["by_kind"].get(r.kind, 0) + 1
        if r.time is not None:
            iso = r.time.strftime("%Y-%m-%dT%H:%M:%SZ")
            d["first"] = iso if d["first"] is None or iso < d["first"] else d["first"]
            d["last"] = iso if d["last"] is None or iso > d["last"] else d["last"]
            if r.kind in REVISION_KINDS and r.time > export_end:
                d["post_export_edits"] += 1
    manifest = load_manifest(bundle)
    return {
        "schema_version": manifest.get("schema_version"),
        "generated_at": manifest.get("generated_at"),
        "export_end": export_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "venues": dict(sorted(per.items())),
    }
