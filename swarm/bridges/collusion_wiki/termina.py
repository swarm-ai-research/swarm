"""Load the swarm.termina.digital incident db as a third source (bead lnaf).

``https://swarm.termina.digital/pub/`` publishes the *ai-safety-lab incident
db* (CC0) as one ``incidents.sqlite`` plus per-table JSONL and a
``manifest.json`` with row counts and sha256s. Its ``record`` table extends
the wikis the collusion.wiki export covers (which ends 2026-07-02) with the
wikis' live RecentChanges listings (``kind='rc-row'``, ``source='live-rc'``)
through the disclosure week, and adds the venues the export never had
(usemod.org, Wiki4D, the wider ProWiki farm).

What the db carries and what it does not:

- **No bodies.** ``body_len`` / ``body_sha256`` / ``body_path`` are pointers;
  the reading pack (``reading_pack.py``) stays the body source.
- **rc-rows are minute-precision** and are the *listing*, not the revision
  store: a moderator deletion shows as two rows, and a signed edit carries a
  handle but no address (``ip_actor_id`` is null on the wikiservice.at
  venues). Anonymous edits carry an ``ip:`` actor instead, from which a /16
  is recovered when the address is dotted (``ip:84.115.212.x`` -> ``84.115``).
- **Export revisions are duplicated** inside the db (``source=
  'collusion-export'``, ids identical to the export's ``rev_id``), so a
  window that overlaps the export must pick one ``record_kinds`` set.
- **Phase** is termina's: ``pre-disclosure`` runs to 2026-09-02,
  ``post-report`` is 09-03 (the collusion.wiki write-up), ``post-press``
  09-04 onward. The replay's "post-disclosure" window (July 3 to the
  snapshot) is a different cut and is set in the scenario, not read from
  ``phase``.
- **Actor kinds** ``handle`` / ``human`` / ``ip`` are termina's own
  classification (``human:<venue>:<n>`` is a numbered wiki regular). The
  bridge keeps them as ``actor_kind`` on the revision so a replay can drop
  the moderators.

Every string in the db is untrusted content from public wikis: titles,
summaries and handles are shown, never executed, and no URL in them is
fetched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from swarm.bridges.collusion_wiki.loader import WikiRevision

log = logging.getLogger(__name__)

SQLITE_NAME = "incidents.sqlite"
MANIFEST_NAME = "manifest.json"

#: termina venue id -> the bridge's wiki name (the export's ``wiki`` field).
#: Venues not listed keep their termina id.
VENUE_WIKI: Dict[str, str] = {
    "dse": "dse",
    "probier": "probier",
    "fractal": "fractal",
    "dorfwiki": "dorfwiki",
    "usemod-org": "usemod",
    "wiki4d": "wiki4d",
}

#: The venues the post-disclosure replay reads by default: the three export
#: wikis plus the three boards the field-evidence note saw reused after
#: disclosure.
DEFAULT_VENUES: Tuple[str, ...] = ("dse", "probier", "fractal", "usemod-org", "wiki4d", "dorfwiki")

DEFAULT_RECORD_KINDS: Tuple[str, ...] = ("rc-row",)

#: The June swarm's handle grammar as the census scanner matches it: one to
#: four CamelCase words ending in a role noun, optional trailing tag.
HANDLE_GRAMMAR = re.compile(
    r"^(?:[A-Z][a-z0-9]+){1,4}"
    r"(?:Agent|Bot|Helper|Researcher|Worker|Fleet|Swarm|Relay|Probe|Bridge|"
    r"Scout|Runner|Tester|Person|Visitor|Creator|Node)[A-Za-z0-9_]*$"
)

#: The June swarm's page-title grammar: an ``Agent`` / ``OpenAI`` / ``OAI`` /
#: ``ZZZ`` prefix, or a CamelCase title ending in a 10-digit epoch or a
#: cohort tag such as ``X1`` / ``Z9`` / ``M2``.
TITLE_GRAMMAR = re.compile(
    r"^(?:Agent|OpenAI|OAI|ZZZ|Loop)[A-Za-z0-9]*$"
    r"|^(?:[A-Z][A-Za-z0-9]*?)(?:1[678]\d{8}|[A-Z]\d{1,2})[a-z0-9]*$"
)

_EPOCH = re.compile(r"(1[678]\d{8})")
_DOTTED_IP = re.compile(r"^ip:(\d{1,3})\.(\d{1,3})\.")
_HOST_CLOUD = (
    ("amazonaws", "aws"),
    ("ec2-", "aws"),
    ("googleusercontent", "gcp"),
    ("cloudapp.azure", "azure"),
    ("azure", "azure"),
)


# ---------------------------------------------------------------------------
# manifest / pin
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminaManifest:
    schema_version: str
    generated_at: str
    rows: Dict[str, int]
    sha256: Dict[str, str]
    licence: str


def read_manifest(path: Path) -> TerminaManifest:
    """``manifest.json`` (or the directory holding it)."""
    if path.is_dir():
        path = path / MANIFEST_NAME
    doc = json.loads(path.read_text())
    files = doc.get("files", {}) or {}
    return TerminaManifest(
        schema_version=str(doc.get("schema_version", "")),
        generated_at=str(doc.get("generated_at", "")),
        rows={k: int(v.get("rows", 0)) for k, v in files.items()},
        sha256={k: str(v.get("sha256", "")) for k, v in files.items()},
        licence=str(doc.get("licence", "")),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pin(db: Path, expected_sha256: Optional[str]) -> bool:
    """Warn (do not fail) when the db differs from the scenario's pin.

    The manifest lists sha256s for the JSONL files only, so the sqlite is
    pinned by its own digest in the scenario, the way the export scenario
    pins ``db_sha256``.
    """
    if not expected_sha256:
        return True
    got = sha256_file(db)
    if got != expected_sha256:
        log.warning(
            "%s sha256 %s… differs from the scenario pin %s…; the snapshot has "
            "moved on and counts will not match the note",
            db, got[:12], expected_sha256[:12],
        )
        return False
    return True


# ---------------------------------------------------------------------------
# records -> revisions
# ---------------------------------------------------------------------------


def resolve_db(path: Path) -> Path:
    if path.is_dir():
        path = path / SQLITE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{SQLITE_NAME} not found at {path}")
    return path


def parse_time(s: Optional[str]) -> Optional[datetime]:
    """termina times: ``2026-07-03T08:05Z`` (minute), ``…:55Z`` (second),
    ``2026-03-11T11:09`` (no zone; treated as UTC), ``2026-08-30`` (day)."""
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def actor_parts(actor_id: Optional[str]) -> Tuple[str, str]:
    """``(kind, name)`` from ``handle:dse:AiraBot`` / ``human:dse:118`` /
    ``ip:84.115.212.x``; ``("", "")`` when null."""
    if not actor_id:
        return "", ""
    kind, _, rest = actor_id.partition(":")
    if kind in ("handle", "human"):
        _venue, _, name = rest.partition(":")
        return kind, (f"human-{name}" if kind == "human" else name)
    return kind, rest


def ip16_of(actor_id: Optional[str], ip_actor_id: Optional[str]) -> str:
    if ip_actor_id and ip_actor_id.startswith("ip16:"):
        return ip_actor_id[5:]
    for cand in (ip_actor_id, actor_id):
        if cand:
            m = _DOTTED_IP.match(cand)
            if m:
                return f"{m.group(1)}.{m.group(2)}"
    return ""


def cloud_of(actor_id: Optional[str]) -> str:
    """Provider from a reverse-DNS actor name; ``""`` when the address is
    bare (numeric /16s are not mapped to providers here)."""
    if not actor_id or not actor_id.startswith("ip:"):
        return ""
    host = actor_id[3:].lower()
    for needle, cloud in _HOST_CLOUD:
        if needle in host:
            return cloud
    return ""


def _sql_in(column: str, values: Iterable[str], where: List[str], params: List[str]) -> bool:
    """Append ``column IN (...)``. Return False if ``values`` is empty.

    SQLite rejects ``IN ()``; an empty filter list is an empty result set
    (a misconfigured ``venues: []`` / ``record_kinds: []`` in YAML).
    """
    items = sorted(set(values))
    if not items:
        return False
    where.append(f"{column} IN ({','.join('?' * len(items))})")
    params.extend(items)
    return True


def iter_revisions(
    db: Path,
    *,
    venues: Sequence[str] = DEFAULT_VENUES,
    record_kinds: Sequence[str] = DEFAULT_RECORD_KINDS,
    window: Optional[Tuple[datetime, datetime]] = None,
    exclude_actor_kinds: Iterable[str] = (),
) -> Iterable[WikiRevision]:
    """Records of the given venues and kinds as ``WikiRevision`` rows.

    ``window`` keeps ``window[0] <= observed_time < window[1]``; records with
    no parseable time are dropped (and counted in the log). Titles with a
    ``" / "`` sub-page separator keep the full path as the page name.
    Empty ``venues`` or ``record_kinds`` yields no rows (SQLite rejects ``IN ()``).
    """
    path = resolve_db(db)
    excl = set(exclude_actor_kinds)
    where: List[str] = []
    params: List[str] = []
    if not _sql_in("r.venue_id", venues, where, params):
        return
    if not _sql_in("r.kind", record_kinds, where, params):
        return
    sql = (
        "SELECT r.id, r.venue_id, r.kind, r.external_id, r.title, r.summary, "
        "r.actor_id, r.ip_actor_id, r.observed_time, r.body_len, r.phase, "
        "r.source, r.campaign_id, r.content_kind, r.status "
        "FROM record r WHERE " + " AND ".join(where)
    )
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        n_untimed = 0
        for row in con.execute(sql, params):
            (rid, venue, kind, ext, title, summary, actor, ip_actor, t_raw,
             body_len, phase, source, campaign, content_kind, status) = row
            t = parse_time(t_raw)
            if t is None:
                n_untimed += 1
                continue
            if window is not None and not (window[0] <= t < window[1]):
                continue
            a_kind, a_name = actor_parts(actor)
            if a_kind in excl:
                continue
            wiki = VENUE_WIKI.get(venue, venue)
            title = str(title or "")
            yield WikiRevision(
                rev_id=str(rid),
                wiki=wiki,
                page_id=f"{wiki}/{title}",
                label=a_name if a_kind in ("handle", "human") else "",
                ip16=ip16_of(actor, ip_actor),
                time=t,
                body_len=int(body_len or 0),
                change_summary=str(summary or ""),
                page_created=str(ext or "") == "1.1",
                source=f"termina:{source}",
                phase=str(phase or ""),
                campaign=str(campaign or ""),
                actor_kind=a_kind,
                actor_raw=str(actor or ""),
                content_kind=str(content_kind or ""),
                record_kind=str(kind or ""),
                record_status=str(status or ""),
            )
        if n_untimed:
            log.info("termina: %d records without a parseable observed_time dropped", n_untimed)
    finally:
        con.close()


def load_revisions(db: Path, **kw: Any) -> List[WikiRevision]:
    """Sorted by time then id, like the export loader."""
    revs = list(iter_revisions(db, **kw))
    revs.sort(key=lambda r: (r.time, r.rev_id))
    return revs


def venue_table(db: Path, venues: Sequence[str] = DEFAULT_VENUES) -> List[Dict[str, Any]]:
    """termina's own venue rows (status, moderator note, first/last seen)."""
    path = resolve_db(db)
    where: List[str] = []
    params: List[str] = []
    if not _sql_in("id", venues, where, params):
        return []
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        sql = (
            "SELECT id, host, path, software, status, moderator, first_seen, last_seen, "
            "caveat FROM venue WHERE " + " AND ".join(where) + " ORDER BY id"
        )
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# fingerprint: does a window carry the June swarm's grammar and cohort marks?
# ---------------------------------------------------------------------------


def _share(hit: int, n: int) -> Optional[float]:
    return round(hit / n, 4) if n else None


def fingerprint(
    revs: Sequence[WikiRevision],
    baseline: Sequence[WikiRevision] = (),
) -> Dict[str, Any]:
    """Grammar and reuse marks of one window, optionally against a baseline.

    Reuse shares are measured on *distinct* handles, titles and /16s so a
    single busy actor cannot carry the share. ``title_epoch_in_baseline`` is
    the share of distinct titles carrying a 10-digit epoch that falls inside
    the baseline's time span, which is how the June pages were named
    (``Agent010LeminoDirect1781807128`` is 2026-06-18).
    """
    handles = {r.label for r in revs if r.actor_kind == "handle" and r.label}
    humans = {r.label for r in revs if r.actor_kind == "human"}
    titles = {r.page_id.split("/", 1)[1] for r in revs}
    ip16s = {r.ip16 for r in revs if r.ip16}
    b_handles = {r.label for r in baseline if r.actor_kind == "handle" and r.label}
    b_titles = {r.page_id.split("/", 1)[1] for r in baseline}
    b_ip16s = {r.ip16 for r in baseline if r.ip16}
    b_span: Optional[Tuple[datetime, datetime]] = (
        (min(r.time for r in baseline), max(r.time for r in baseline)) if baseline else None
    )

    def epoch_in_baseline(title: str) -> bool:
        if b_span is None:
            return False
        for m in _EPOCH.findall(title):
            t = datetime.fromtimestamp(int(m), tz=timezone.utc)
            if b_span[0] <= t <= b_span[1]:
                return True
        return False

    kinds: Dict[str, int] = {}
    clouds: Dict[str, int] = {}
    campaigns: Dict[str, int] = {}
    days: Dict[str, int] = {}
    for r in revs:
        kinds[r.actor_kind or "none"] = kinds.get(r.actor_kind or "none", 0) + 1
        c = cloud_of(r.actor_raw)
        if c:
            clouds[c] = clouds.get(c, 0) + 1
        campaigns[r.campaign or "(none)"] = campaigns.get(r.campaign or "(none)", 0) + 1
        d = r.time.strftime("%Y-%m-%d")
        days[d] = days.get(d, 0) + 1
    peak_day = max(days.items(), key=lambda kv: (kv[1], kv[0])) if days else (None, 0)

    return {
        "n_rows": len(revs),
        "n_days_active": len(days),
        "peak_day": peak_day[0],
        "peak_day_rows": peak_day[1],
        "actor_kinds": dict(sorted(kinds.items())),
        "cloud_from_rdns": dict(sorted(clouds.items())),
        "termina_campaigns": dict(sorted(campaigns.items())),
        "n_handles": len(handles),
        "n_humans": len(humans),
        "n_titles": len(titles),
        "n_ip16": len(ip16s),
        "handle_grammar_share": _share(sum(1 for h in handles if HANDLE_GRAMMAR.match(h)), len(handles)),
        "title_grammar_share": _share(sum(1 for t in titles if TITLE_GRAMMAR.match(t)), len(titles)),
        "handle_in_baseline": _share(len(handles & b_handles), len(handles)) if baseline else None,
        "title_in_baseline": _share(len(titles & b_titles), len(titles)) if baseline else None,
        "title_epoch_in_baseline": (
            _share(sum(1 for t in titles if epoch_in_baseline(t)), len(titles)) if baseline else None
        ),
        "ip16_in_baseline": _share(len(ip16s & b_ip16s), len(ip16s)) if baseline else None,
        "rows_per_handle": round(len(revs) / len(handles), 2) if handles else None,
        "rows_per_ip16": round(sum(1 for r in revs if r.ip16) / len(ip16s), 2) if ip16s else None,
    }
