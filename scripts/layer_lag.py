#!/usr/bin/env python3
"""Layer lag: how far do the fast layers run ahead of the slow ones?

Pace layers (Brand; deepfates, Long Now 2026-10-22): fast layers disrupt,
slow layers govern, and the danger is tension that builds between them and
releases at once. The MUD LEDGER shearing check (bv-dispatch retro item 3)
asks whether a default contradicts a lock *now*. This tool asks how long
gaps stay open and whether they open faster than they close
(docs/research/pace-layers-swarm-lag.md). Every artifact gets a layer by
path:

  fast    .beads/ (bead traffic that reached git; since beads moved to
          Dolt this goes quiet, and the report says when it was last seen)
  middle  package code, tests, scripts, experiments, CI, scenarios
  slow    prereg docs, theory.md, CLAUDE.md, AGENTS.md

and the tool reports three things:

  CHURN       commits per ISO week touching each layer, and the smoothed
              fast:slow ratio (fast+1)/(slow+1).
  LAG         for each prereg doc, every later commit that changed a file
              the doc pins (a backticked repo path), and the time until a doc
              commit ratified it. Never ratified = OPEN. A pin that no longer
              exists at HEAD is reported as GONE.
  INCIDENTS   mean churn ratio in the weeks before each known shear versus
              all other weeks. The note's prediction is that shears follow
              rising ratios.

Usage:
    python scripts/layer_lag.py
    python scripts/layer_lag.py --scope dir --since 2026-06-01
    python scripts/layer_lag.py --incident 2026-06-05 --json

Ratification (--ratify strict, the default): a doc commit ratifies a pin
only if one of its changed lines names the pin's path or file name.
--ratify touch counts any doc edit. naa8 is why: d49daaa2 moved the rubric
default to v3 *and* added a v3 pilot pointer to the prereg in the same
commit, so "touch" scores the real incident as ratified at lag 0. Matching
lock/amend/deviation words was tried and dropped: an unrelated Arm A
"deviation analysis" line on 07-21 closed naa8 by accident. --scope dir widens each pin to its
directory, which is what it takes to see naa8 at all (the default lived
beside the pinned rubric file, not in it). LAG is a triage list for a human
read, not a verdict. Commits touching more than --sweep-files files are
skipped as sweeps.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import posixpath
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

LAYER_PATTERNS: Sequence[Tuple[str, Sequence[str]]] = (
    ("slow", (
        "CLAUDE.md",
        "AGENTS.md",
        "docs/research/theory.md",
        "docs/research/*prereg*.md",
    )),
    ("fast", (".beads/*",)),
    ("middle", (
        "swarm/*",
        "tests/*",
        "scripts/*",
        "experiments/*",
        "scenarios/*",
        ".github/*",
        ".claude/commands/*",
        ".claude/hooks/*",
        "pyproject.toml",
    )),
)
LAYERS = ("fast", "middle", "slow")
# naa8: DEFAULT_RUBRIC_VERSION moved to v3 in d49daaa2 after the 06-01 lock.
DEFAULT_INCIDENTS = ("2026-06-05",)

_PIN = re.compile(r"`([^`\s]+)`")


@dataclass(frozen=True)
class Commit:
    sha: str
    when: datetime
    files: Tuple[str, ...]


@dataclass(frozen=True)
class Change:
    doc: str
    sha: str
    when: datetime
    file: str
    ratified: Optional[datetime]

    def lag_days(self, now: datetime) -> float:
        end = self.ratified or now
        return (end - self.when).total_seconds() / 86400


def classify(path: str) -> Optional[str]:
    """Layer for ``path``, or None for unlayered files (most docs)."""
    for layer, patterns in LAYER_PATTERNS:
        if any(fnmatch.fnmatchcase(path, p) for p in patterns):
            return layer
    return None


def iso_week(when: datetime) -> str:
    year, week, _ = when.isocalendar()
    return f"{year}-W{week:02d}"


def week_start(week: str) -> date:
    year, w = week.split("-W")
    return date.fromisocalendar(int(year), int(w), 1)


def extract_pins(text: str, known: Set[str]) -> List[str]:
    """Backticked tokens in ``text`` that name a file git has ever tracked.

    ``path.py:63`` and ``path.md#anchor`` count as ``path.py`` / ``path.md``.
    """
    pins = set()
    for token in _PIN.findall(text):
        path = re.split(r"[:#]", token, maxsplit=1)[0].rstrip(".,;)")
        if path in known:
            pins.add(path)
    return sorted(pins)


def _matches(path: str, pins: Sequence[str], scope: str) -> Optional[str]:
    for pin in pins:
        if path == pin:
            return pin
        if scope == "dir" and posixpath.dirname(path) == posixpath.dirname(pin):
            return pin
    return None


def ratifies(changed_lines: Sequence[str], pin: Optional[str]) -> bool:
    """Does a doc commit with these changed lines ratify ``pin``?

    ``pin=None`` is --ratify touch: any edit counts.
    """
    if pin is None:
        return True
    name = posixpath.basename(pin)
    return any(pin in line or name in line for line in changed_lines)


def ratification_lags(
    commits: Sequence[Commit],
    doc: str,
    pins: Sequence[str],
    scope: str,
    doc_edits: Optional[Dict[str, List[str]]] = None,
) -> List[Change]:
    """Every pinned-file change after both ``doc`` and the pinned file exist,
    with the first doc commit at or after it that ratifies the pin. ``commits`` must be
    oldest first; ``doc_edits`` maps a doc commit sha to its changed lines
    (None = --ratify touch, any doc edit ratifies)."""
    doc_commits = [c for c in commits if doc in c.files]
    if not doc_commits:
        return []
    registered = doc_commits[0].when
    born = {p: next((c.when for c in commits if p in c.files), registered) for p in pins}
    changes: List[Change] = []
    for c in commits:
        if c.when <= registered:
            continue
        for f in c.files:
            pin = None if f == doc else _matches(f, pins, scope)
            if pin is None or c.when <= born[pin]:
                continue
            ratified = next(
                (d.when for d in doc_commits
                 if d.when >= c.when
                 and ratifies(doc_edits.get(d.sha, []) if doc_edits is not None else [],
                              pin if doc_edits is not None else None)),
                None,
            )
            changes.append(Change(doc, c.sha, c.when, f, ratified))
    return changes


def weekly_churn(commits: Iterable[Commit]) -> Dict[str, Dict[str, int]]:
    """Commits touching each layer per ISO week (a commit counts once per layer)."""
    weeks: Dict[str, Dict[str, int]] = defaultdict(lambda: dict.fromkeys(LAYERS, 0))
    for c in commits:
        layers = {classify(f) for f in c.files} - {None}
        row = weeks[iso_week(c.when)]
        for layer in layers:
            row[layer] += 1
    return dict(sorted(weeks.items()))


def churn_ratio(row: Dict[str, int]) -> float:
    return (row["fast"] + 1) / (row["slow"] + 1)


def incident_test(
    churn: Dict[str, Dict[str, int]], incidents: Sequence[date], window: int
) -> List[dict]:
    """Mean churn ratio in the ``window`` weeks before each incident vs the rest."""
    out = []
    for incident in incidents:
        before, rest = [], []
        for week, row in churn.items():
            start = week_start(week)
            in_window = incident - timedelta(weeks=window) <= start < incident
            (before if in_window else rest).append(churn_ratio(row))
        out.append({
            "incident": incident.isoformat(),
            "weeks_before": len(before),
            "ratio_before": round(mean(before), 2) if before else None,
            "ratio_rest": round(mean(rest), 2) if rest else None,
        })
    return out


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def read_commits(since: Optional[str], until: Optional[str]) -> List[Commit]:
    args = ["log", "--first-parent", "--reverse", "--no-renames", "--format=@@%H\t%aI", "--name-only"]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    commits: List[Commit] = []
    sha, when, files = None, None, []
    for line in _git(*args).splitlines():
        if line.startswith("@@"):
            if sha:
                commits.append(Commit(sha, when, tuple(files)))
            sha, iso = line[2:].split("\t")
            when, files = datetime.fromisoformat(iso), []
        elif line:
            files.append(line)
    if sha:
        commits.append(Commit(sha, when, tuple(files)))
    return commits


def read_doc_edits(doc: str) -> Dict[str, List[str]]:
    """Changed (added or removed) lines of every commit touching ``doc``."""
    edits: Dict[str, List[str]] = {}
    sha = None
    for line in _git("log", "--first-parent", "--no-renames", "-p", "-U0", "--format=@@%H", "--", doc).splitlines():
        if line.startswith("@@") and not line.startswith("@@ "):
            sha = line[2:]
            edits[sha] = []
        elif sha and line[:1] in "+-" and not line.startswith(("+++", "---")):
            edits[sha].append(line[1:])
    return edits


def build_report(args: argparse.Namespace) -> dict:
    commits = read_commits(args.since, args.until)
    all_commits = read_commits(None, args.until) if args.since else commits
    kept = [c for c in commits if len(c.files) <= args.sweep_files]
    head = set(_git("ls-files").splitlines())
    ever = {f for c in all_commits for f in c.files} | head
    now = max(c.when for c in all_commits)

    docs = sorted(f for f in head if fnmatch.fnmatchcase(f, "docs/research/*prereg*.md"))
    locks = []
    for doc in docs:
        pins = extract_pins(Path(doc).read_text(encoding="utf-8"), ever)
        changes = ratification_lags(
            [c for c in all_commits if len(c.files) <= args.sweep_files],
            doc, pins, args.scope,
            read_doc_edits(doc) if args.ratify == "strict" else None,
        )
        if args.since:
            cutoff = kept[0].when if kept else now
            changes = [ch for ch in changes if ch.when >= cutoff]
        closed = [ch.lag_days(now) for ch in changes if ch.ratified]
        open_ = [ch for ch in changes if not ch.ratified]
        locks.append({
            "doc": doc,
            "pins": pins,
            "gone": [p for p in pins if p not in head],
            "changes": len(changes),
            "ratified": len(closed),
            "median_lag_days": round(median(closed), 1) if closed else None,
            "open": [
                {"sha": ch.sha[:8], "date": ch.when.date().isoformat(),
                 "file": ch.file, "age_days": round(ch.lag_days(now), 1)}
                for ch in open_
            ],
        })

    churn = weekly_churn(kept)
    fast_seen = [c.when for c in kept if any(classify(f) == "fast" for f in c.files)]
    incidents = [date.fromisoformat(d) for d in (args.incident or DEFAULT_INCIDENTS)]
    return {
        "range": [kept[0].when.date().isoformat(), kept[-1].when.date().isoformat()] if kept else None,
        "commits": len(commits),
        "sweeps_skipped": len(commits) - len(kept),
        "scope": args.scope,
        "ratify": args.ratify,
        "fast_last_seen": fast_seen[-1].date().isoformat() if fast_seen else None,
        "churn": churn,
        "locks": locks,
        "incidents": incident_test(churn, incidents, args.window),
    }


def render(report: dict) -> str:
    lines = [
        f"Layer lag over {report['range']}: {report['commits']} commits, "
        f"{report['sweeps_skipped']} sweeps skipped, pin scope={report['scope']}, "
        f"ratify={report['ratify']}",
        "",
        "CHURN (commits per ISO week; ratio = (fast+1)/(slow+1))",
        f"  {'week':<10} {'fast':>5} {'middle':>7} {'slow':>5} {'ratio':>6}",
    ]
    for week, row in report["churn"].items():
        lines.append(
            f"  {week:<10} {row['fast']:>5} {row['middle']:>7} {row['slow']:>5} "
            f"{churn_ratio(row):>6.2f}"
        )
    lines.append(
        f"  fast layer last seen in git: {report['fast_last_seen']} "
        "(ratios after that undercount fast churn)"
    )
    lines += ["", "LAG (pinned-file changes after registration, time to ratification)"]
    for lock in report["locks"]:
        lines.append(
            f"  {lock['doc']}: {len(lock['pins'])} pins, {lock['changes']} changes, "
            f"{lock['ratified']} ratified, median lag {lock['median_lag_days']} d, "
            f"{len(lock['open'])} OPEN"
        )
        for pin in lock["gone"]:
            lines.append(f"    GONE  {pin} (pinned, not at HEAD)")
        for ch in lock["open"]:
            lines.append(f"    OPEN  {ch['date']} {ch['sha']} {ch['file']} ({ch['age_days']} d)")
    lines += ["", "INCIDENTS (mean ratio in the weeks before vs all other weeks)"]
    for inc in report["incidents"]:
        lines.append(
            f"  {inc['incident']}: before {inc['ratio_before']} "
            f"({inc['weeks_before']} wk) vs rest {inc['ratio_rest']}"
        )
    if len(report["incidents"]) < 3:
        lines.append("  fewer than 3 incidents: descriptive only, not a test")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--since", help="git date; churn and LAG start here")
    parser.add_argument("--until", help="git date; default HEAD")
    parser.add_argument("--scope", choices=("file", "dir"), default="file",
                        help="pin a file exactly, or widen each pin to its directory")
    parser.add_argument("--ratify", choices=("strict", "touch"), default="strict",
                        help="strict: a doc edit must name the pin or the lock; touch: any edit")
    parser.add_argument("--sweep-files", type=int, default=40,
                        help="skip commits touching more files than this")
    parser.add_argument("--incident", action="append",
                        help=f"known shear date, repeatable (default {DEFAULT_INCIDENTS})")
    parser.add_argument("--window", type=int, default=4,
                        help="weeks before an incident to compare (default 4)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
