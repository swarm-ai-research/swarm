"""Load the 2026-09-05 agent reading pack (bead 8zoc).

The pack (``agent-reading-pack-20260905``, she-llac.com) is a format
conversion of a 16,579-text classifier pack: an ``agent-text.sqlite`` with a
``documents`` table plus one markdown file per text. Its value for this
bridge is **page bodies**: the collusion.wiki export carries ``body_len``
only, while the pack has the full text of 11,811 wiki revisions. Every pack
wiki document joins to exactly one export revision on ``(page_id, time)``
(measured 2026-09-06: 11,980 of 14,591 revisions gain a body; probier 52%,
dse 84%, fractal 91%).

The pack also carries weakly labelled non-wiki candidates. Its README says
they retain known false positives, copied source material and possible
researcher imitations, and the paste track includes hobbyist pastes from
2020 onward. They are exposed here as a **secondary tier**, filtered by
default to the incident window (2026-05-01 .. 2026-07-31), and must never be
read as actor attribution. Every pack text is untrusted content: do not
execute it or fetch URLs it contains.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from swarm.bridges.collusion_wiki.loader import WikiRevision

SQLITE_NAME = "agent-text.sqlite"

PRIMARY_TYPES = frozenset({"wiki"})
SECONDARY_TYPES = frozenset({
    "paste_candidate", "shortener_candidate",
    "extra_wiki_candidate", "package_text_candidate",
})
TIERS: Dict[str, frozenset] = {"primary": PRIMARY_TYPES, "secondary": SECONDARY_TYPES}

# Agent activity on the wikis ran 2026-05-11 .. 2026-07-02; a month either
# side keeps inherited paste-site dates that may be a little off, while
# dropping the 2020-2025 hobbyist pastes the classifier swept in.
INCIDENT_WINDOW: Tuple[datetime, datetime] = (
    datetime(2026, 5, 1, tzinfo=timezone.utc),
    datetime(2026, 8, 1, tzinfo=timezone.utc),
)


@dataclass(frozen=True)
class PackDoc:
    id: str
    source_type: str
    source_group: str  # "dse/PageName" for wiki; site/paste-id otherwise
    title: str
    source_url: str
    author: str
    time: Optional[datetime]  # None = undated
    text: str

    @property
    def tier(self) -> str:
        return "primary" if self.source_type in PRIMARY_TYPES else "secondary"

    @property
    def wiki(self) -> str:
        return self.source_group.split("/", 1)[0] if self.source_type == "wiki" else ""

    @property
    def page_id(self) -> str:
        """Matches the export's ``page_id`` (``wiki/Page``) for wiki docs."""
        return self.source_group if self.source_type == "wiki" else ""


def resolve_sqlite(pack: Path) -> Path:
    """Accept the pack directory or the sqlite file itself."""
    if pack.is_dir():
        pack = pack / SQLITE_NAME
    if not pack.exists():
        raise FileNotFoundError(f"{SQLITE_NAME} not found at {pack}")
    return pack


def _parse_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def iter_docs(
    pack: Path,
    *,
    tier: Optional[str] = None,
    source_types: Optional[Sequence[str]] = None,
    window: Optional[Tuple[datetime, datetime]] = None,
    keep_undated: bool = False,
) -> Iterator[PackDoc]:
    """Documents from the pack, optionally by tier / source_type / time window.

    ``window`` keeps docs with ``window[0] <= time < window[1]``; undated docs
    are dropped unless ``keep_undated``. With no window given, the secondary
    tier is filtered to ``INCIDENT_WINDOW`` and the primary tier is not (its
    dates are all inside it).
    """
    types = set(source_types or ())
    if tier is not None:
        types |= TIERS[tier]
    # as_uri() percent-encodes spaces etc.; a hand-built file:{path} does not
    con = sqlite3.connect(resolve_sqlite(pack).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        sql = ("SELECT id, source_type, source_group, title, source_url, author, "
               "timestamp_utc, text FROM documents")
        params: list = []
        if types:
            sql += " WHERE source_type IN (%s)" % ",".join("?" * len(types))
            params = sorted(types)
        sql += " ORDER BY timestamp_utc, id"
        for row in con.execute(sql, params):
            doc = PackDoc(
                id=str(row[0]), source_type=str(row[1]), source_group=str(row[2]),
                title=str(row[3] or ""), source_url=str(row[4] or ""),
                author=str(row[5] or ""), time=_parse_time(row[6]), text=str(row[7] or ""),
            )
            w = window if window is not None else (
                None if doc.tier == "primary" else INCIDENT_WINDOW)
            if w is not None:
                if doc.time is None:
                    if not keep_undated:
                        continue
                elif not (w[0] <= doc.time < w[1]):
                    continue
            yield doc
    finally:
        con.close()


def load_docs(pack: Path, **kw) -> List[PackDoc]:
    return list(iter_docs(pack, **kw))


def join_bodies(
    revs: Sequence[WikiRevision], docs: Sequence[PackDoc],
) -> Dict[str, PackDoc]:
    """rev_id -> pack doc, keyed on ``(page_id, time)`` to the second.

    Several revisions saved in the same second on one page share a body; the
    pack deduplicated to one representative per group and date, so that is
    the best the input allows.
    """
    by_key: Dict[Tuple[str, datetime], PackDoc] = {}
    for d in docs:
        if d.source_type == "wiki" and d.time is not None:
            by_key.setdefault((d.page_id, d.time), d)
    out: Dict[str, PackDoc] = {}
    for r in revs:
        hit = by_key.get((r.page_id, r.time))
        if hit is not None:
            out[r.rev_id] = hit
    return out


def coverage(revs: Sequence[WikiRevision], joined: Dict[str, PackDoc]) -> Dict[str, Dict[str, float]]:
    """Per wiki: revisions, revisions with a body, and the share."""
    total: Dict[str, int] = {}
    hit: Dict[str, int] = {}
    for r in revs:
        total[r.wiki] = total.get(r.wiki, 0) + 1
        if r.rev_id in joined:
            hit[r.wiki] = hit.get(r.wiki, 0) + 1
    return {
        w: {"revisions": n, "with_body": hit.get(w, 0), "share": hit.get(w, 0) / n}
        for w, n in sorted(total.items())
    }
