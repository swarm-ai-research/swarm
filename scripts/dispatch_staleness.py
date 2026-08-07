#!/usr/bin/env python3
"""Effective-staleness report for dispatch hygiene (beads-y0cx).

``bd``'s ``updated_at`` does not bump on comments, and run artifacts
live outside bd entirely, so a staleness pass that reads ``updated_at``
alone cries zombie on actively-worked beads — k2yr was flagged
stale-46d in dispatch r4 despite a same-day arm A run and fresh
comments. This tool computes

    effective last activity = max(updated_at,
                                  newest comment created_at,
                                  newest runs/ artifact referencing
                                  the bead)

and reports staleness against that, so the dispatch hygiene pass
(``/bv-dispatch`` step 5) flags only genuinely idle beads.

Usage:
    python scripts/dispatch_staleness.py                 # in_progress beads
    python scripts/dispatch_staleness.py --status open
    python scripts/dispatch_staleness.py --threshold-days 7 --json

A bead "references" a run when the bead id (or its short suffix, e.g.
``k2yr``) appears — as a whole word — in the run directory's name or in
small text/metadata files (json/md/yaml/txt) up to two levels deep.
Run-dir mtimes stand in for artifact timestamps.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_METADATA_SUFFIXES = {".json", ".md", ".yaml", ".yml", ".txt", ".csv"}
_MAX_METADATA_BYTES = 1_000_000


def short_id(bead_id: str) -> str:
    """``distributional-agi-safety-k2yr`` -> ``k2yr``."""
    return bead_id.rsplit("-", 1)[-1]


def parse_ts(raw: str) -> Optional[datetime]:
    """Parse bd timestamps (ISO 8601, offset or ``Z``) to aware UTC."""
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _dir_references_bead(run_dir: Path, pattern: re.Pattern) -> bool:
    if pattern.search(run_dir.name):
        return True
    try:
        candidates = [
            p for depth_glob in ("*", "*/*")
            for p in run_dir.glob(depth_glob)
            if p.is_file() and p.suffix.lower() in _METADATA_SUFFIXES
            and p.stat().st_size <= _MAX_METADATA_BYTES
        ]
    except OSError:
        return False
    for path in candidates:
        try:
            if pattern.search(path.read_text(errors="ignore")):
                return True
        except OSError:
            continue
    return False


def latest_run_activity(
    bead_id: str, runs_root: Path
) -> Optional[datetime]:
    """Newest mtime among run dirs that reference ``bead_id``.

    Word-boundary match on the full id or its short suffix keeps 4-char
    suffixes from lighting up on random substrings.
    """
    if not runs_root.is_dir():
        return None
    sid = short_id(bead_id)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:%s|%s)(?![A-Za-z0-9])"
        % (re.escape(bead_id), re.escape(sid))
    )
    newest: Optional[datetime] = None
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        if not _dir_references_bead(run_dir, pattern):
            continue
        mtime = datetime.fromtimestamp(
            run_dir.stat().st_mtime, tz=timezone.utc
        )
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def effective_activity(
    updated_at: Optional[datetime],
    last_comment_at: Optional[datetime],
    last_run_at: Optional[datetime],
) -> Tuple[Optional[datetime], str]:
    """Latest of the three signals and which one won."""
    candidates = [
        (updated_at, "updated_at"),
        (last_comment_at, "comment"),
        (last_run_at, "run_artifact"),
    ]
    live = [(ts, src) for ts, src in candidates if ts is not None]
    if not live:
        return None, "none"
    return max(live, key=lambda pair: pair[0])


def build_report(
    issues: Sequence[Dict],
    comments_by_id: Dict[str, List[Dict]],
    runs_root: Path,
    now: datetime,
) -> List[Dict]:
    """One row per issue with effective staleness, sorted stalest-first."""
    rows: List[Dict] = []
    for issue in issues:
        bead_id = issue["id"]
        updated = parse_ts(issue.get("updated_at", ""))
        comment_ts = [
            ts for c in comments_by_id.get(bead_id, [])
            if (ts := parse_ts(c.get("created_at", ""))) is not None
        ]
        last_comment = max(comment_ts) if comment_ts else None
        last_run = latest_run_activity(bead_id, runs_root)
        effective, source = effective_activity(updated, last_comment, last_run)
        naive_days = (
            (now - updated).total_seconds() / 86400 if updated else None
        )
        stale_days = (
            (now - effective).total_seconds() / 86400 if effective else None
        )
        rows.append({
            "id": bead_id,
            "title": issue.get("title", ""),
            "updated_at": updated.isoformat() if updated else None,
            "last_comment_at": (
                last_comment.isoformat() if last_comment else None
            ),
            "last_run_artifact_at": last_run.isoformat() if last_run else None,
            "effective_last_activity": (
                effective.isoformat() if effective else None
            ),
            "activity_source": source,
            "naive_stale_days": (
                round(naive_days, 1) if naive_days is not None else None
            ),
            "stale_days": (
                round(stale_days, 1) if stale_days is not None else None
            ),
        })
    rows.sort(key=lambda r: -(r["stale_days"] or 0.0))
    return rows


def _bd_json(args: List[str]) -> List[Dict]:
    out = subprocess.run(
        ["bd", *args, "--json"], capture_output=True, text=True, timeout=60
    )
    if out.returncode != 0:
        raise RuntimeError(f"bd {' '.join(args)} failed: {out.stderr.strip()}")
    parsed: List[Dict] = json.loads(out.stdout or "[]")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", default="in_progress",
                    help="bd status filter (default: in_progress)")
    ap.add_argument("--threshold-days", type=float, default=14.0,
                    help="flag beads whose effective staleness exceeds this")
    ap.add_argument("--runs-dir", default="runs",
                    help="runs directory to scan for bead references")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit the full report as JSON")
    opts = ap.parse_args(argv)

    try:
        issues = _bd_json(["list", "--status", opts.status])
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    comments_by_id: Dict[str, List[Dict]] = {}
    for issue in issues:
        try:
            comments_by_id[issue["id"]] = _bd_json(["comments", issue["id"]])
        except (RuntimeError, OSError, json.JSONDecodeError):
            comments_by_id[issue["id"]] = []

    now = datetime.now(timezone.utc)
    rows = build_report(issues, comments_by_id, Path(opts.runs_dir), now)

    if opts.as_json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print(f"no {opts.status} beads")
        return 0
    print(f"{'bead':14s} {'naive':>6s} {'eff.':>6s} {'source':12s} title")
    for r in rows:
        flag = "ZOMBIE " if (
            r["stale_days"] is not None
            and r["stale_days"] > opts.threshold_days
        ) else ""
        rescued = (
            " (rescued by %s)" % r["activity_source"]
            if r["activity_source"] != "updated_at"
            and r["naive_stale_days"] is not None
            and r["stale_days"] is not None
            and r["naive_stale_days"] - r["stale_days"] > 1.0
            else ""
        )
        print(
            f"{short_id(r['id']):14s} "
            f"{r['naive_stale_days'] if r['naive_stale_days'] is not None else '?':>6} "
            f"{r['stale_days'] if r['stale_days'] is not None else '?':>6} "
            f"{r['activity_source']:12s} "
            f"{flag}{r['title'][:60]}{rescued}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
