"""Load the collusion.wiki export (revisions.jsonl, events.jsonl).

Download page: https://collusion.wiki/explorer/download.html
Files are gzipped JSONL; this loader accepts either ``.jsonl`` or
``.jsonl.gz``. The export redacts the low half of every IP (``ip16`` is
the first two octets) and replaces user names with opaque labels, so
nothing here is more identifying than what the site itself publishes.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

_ISO = "%Y-%m-%dT%H:%M:%SZ"


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, _ISO).replace(tzinfo=timezone.utc)


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _resolve(data_dir: Path, stem: str) -> Path:
    for cand in (data_dir / f"{stem}.jsonl", data_dir / f"{stem}.jsonl.gz"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{stem}.jsonl[.gz] not found under {data_dir}")


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WikiRevision:
    """One saved edit. Field names follow the export's ``revisions.jsonl``."""

    rev_id: str
    wiki: str
    page_id: str
    label: str  # opaque editor handle; "" when the export had none
    ip16: str  # first two octets of the source IP
    time: datetime
    body_len: int
    change_summary: str
    page_created: bool

    @property
    def editor_label(self) -> str:
        return self.label or "(unlabeled)"


@dataclass(frozen=True)
class WikiEvent:
    """A non-save event from ``events.jsonl`` (delete, revert, probe)."""

    event_type: str
    wiki: str
    page: Optional[str]
    time: datetime
    actor_label: str


def iter_revisions(data_dir: Path) -> Iterator[WikiRevision]:
    with _open(_resolve(data_dir, "revisions")) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            yield WikiRevision(
                rev_id=str(d["rev_id"]),
                wiki=str(d["wiki"]),
                page_id=str(d["page_id"]),
                label=str(d.get("label") or ""),
                ip16=str(d.get("ip16") or ""),
                time=_parse_ts(d["time"]),
                body_len=int(d.get("body_len") or 0),
                change_summary=str(d.get("change_summary") or ""),
                page_created=d.get("diff_base_reason") == "page_created",
            )


PLACEHOLDER_BODY_LEN = 27  # len("Describe the new page here.")


def placeholder_body_len_share(revs: List[WikiRevision]) -> Dict[str, float]:
    """Per wiki, the share of revisions whose body_len is the placeholder length.

    The published export reports ``body_len`` 27 for 555 of 1,013 ProbierWiki
    revisions whose live bodies are hundreds to thousands of characters (bead
    m6tu, field-evidence 2026-09-05). Byte-volume figures for such a wiki are
    not derivable from the export.
    """
    total: Dict[str, int] = {}
    hit: Dict[str, int] = {}
    for r in revs:
        total[r.wiki] = total.get(r.wiki, 0) + 1
        if r.body_len == PLACEHOLDER_BODY_LEN:
            hit[r.wiki] = hit.get(r.wiki, 0) + 1
    return {w: hit.get(w, 0) / n for w, n in total.items()}


def load_revisions(data_dir: Path) -> List[WikiRevision]:
    """All revisions sorted by time (ties broken by rev_id for determinism).

    Warns when a wiki's ``body_len`` is the placeholder length for more than half
    its revisions, so no caller trusts byte volume for that wiki.
    """
    revs = list(iter_revisions(data_dir))
    revs.sort(key=lambda r: (r.time, r.rev_id))
    for wiki, share in placeholder_body_len_share(revs).items():
        if share > 0.5:
            log.warning(
                "%s: body_len is the %d-byte placeholder for %.0f%% of revisions; "
                "do not derive byte volume for this wiki from the export",
                wiki, PLACEHOLDER_BODY_LEN, 100 * share,
            )
    return revs


def load_events(data_dir: Path, *, types: Optional[set] = None) -> List[WikiEvent]:
    """Non-save events, optionally filtered by type. Missing file -> []."""
    try:
        path = _resolve(data_dir, "events")
    except FileNotFoundError:
        return []
    out: List[WikiEvent] = []
    with _open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            et = str(d.get("event_type"))
            if et == "save" or (types is not None and et not in types):
                continue
            out.append(
                WikiEvent(
                    event_type=et,
                    wiki=str(d.get("wiki") or ""),
                    page=d.get("page"),
                    time=_parse_ts(d["time"]),
                    actor_label=str(d.get("actor_label") or ""),
                )
            )
    out.sort(key=lambda e: e.time)
    return out
