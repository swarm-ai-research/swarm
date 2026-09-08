"""The two gates the termina.digital scan uses to call a venue a swarm (bead n00g).

``swarm.termina.digital/db/scan.html`` separates the swarm wikis from human-run
boards with two per-venue signals that our detector stack did not compute:

**Handle grammar.** The share of identities whose handle is CamelCase and
carries a role-word token (a CamelCase / letter-digit segment, not a raw
substring) or a trailing number (``ResearchHelper927``,
``CookResearchAgent7712``, ``Agent0``). Humans score 0.00 there, the swarm
wikis 0.67 to 0.89, threshold 0.5. It is a naming feature, so it is free for
the next model to change; the scan itself says text signals should not decide
a verdict. It is still the cheapest gate a moderator has, and the one that
separates usemod.org (contention 62, grammar 0.00) from the swarm wikis.

**Contention.** Alternations between distinct identities on one page within
one hour. Humans peak at 4 in the scan's controls, threshold 5, DSEWiki 1,146.
A per-page count rather than a graph statistic, so it never meets the hub-page
saturation that pins the structural detector to its p-value floor: the hot
welcome page is exactly what drives it.

Both are computed from revisions (not interactions), so page creations and
self follow-ups count, and both are reported per wiki and per timeline step.

The role-word list is ours, not the scan's (unpublished). With it the export
measures dse 0.80, probier 0.60, fractal 0.92 against the scan's 0.74, 0.67,
0.89; the residual is the word list plus the scan's larger identity set (it
counts RecentChanges rows and deletions the export does not hold).
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence

from swarm.bridges.collusion_wiki.loader import WikiRevision

ROLE_WORDS = ("agent", "helper", "researcher", "bot", "scout", "worker", "assistant")
_ROLE_TOKENS = frozenset(ROLE_WORDS)

_ALNUM = re.compile(r"^[A-Za-z0-9]+$")
_CASE_CHANGE = re.compile(r"[a-z][A-Z0-9]|[0-9][A-Z]")
_TRAILING_NUMBER = re.compile(r"\d$")
# ResearchHelper927 -> Research, Helper, 927; Agent0 -> Agent, 0
_SEGMENT = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")


def is_camel_case(handle: str) -> bool:
    """Alphanumeric with at least one lower-to-upper or digit boundary."""
    return bool(_ALNUM.match(handle)) and bool(_CASE_CHANGE.search(handle))


def _handle_segments(handle: str) -> List[str]:
    """Split an alphanumeric handle into CamelCase / letter-digit tokens."""
    return _SEGMENT.findall(handle)


def is_grammar_handle(handle: str) -> bool:
    """CamelCase with a role-word token/segment or a trailing number.

    Role words match as whole segments (``Agent`` in ``CookResearchAgent7712``),
    not as substrings of a longer word (``agent`` in ``MagentaBlue``). Empty
    handles never match.
    """
    if not handle or not is_camel_case(handle):
        return False
    if _TRAILING_NUMBER.search(handle):
        return True
    return any(seg.lower() in _ROLE_TOKENS for seg in _handle_segments(handle))


def handle_grammar_share(revisions: Sequence[WikiRevision]) -> Dict[str, Dict[str, Any]]:
    """Per wiki (plus ``"all"``): share of distinct non-empty handles matching the grammar.

    Identities are distinct labels, as in the scan; the unlabeled editor is not
    an identity here, so its revisions count toward ``n_revisions`` only.
    """
    handles: Dict[str, set] = defaultdict(set)
    n_revs: Dict[str, int] = defaultdict(int)
    for r in revisions:
        for key in (r.wiki, "all"):
            n_revs[key] += 1
            if r.label:
                handles[key].add(r.label)
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted(n_revs):
        ids = handles.get(key, set())
        n_match = sum(1 for h in ids if is_grammar_handle(h))
        out[key] = {
            "n_identities": len(ids),
            "n_grammar": n_match,
            "share": round(n_match / len(ids), 4) if ids else 0.0,
            "n_revisions": n_revs[key],
        }
    return out


def page_hour_alternations(
    revisions: Sequence[WikiRevision],
    identity: Callable[[WikiRevision], str] = lambda r: r.editor_label,
) -> Dict[tuple, int]:
    """Alternations between distinct identities per (page_id, UTC hour).

    Consecutive saves by the same identity do not count; each switch to a
    different identity adds one. Revisions are taken in time order.
    """
    alternations: Dict[tuple, int] = defaultdict(int)
    last: Dict[tuple, str] = {}
    for r in sorted(revisions, key=lambda r: (r.time, r.rev_id)):
        key = (r.page_id, r.time.strftime("%Y-%m-%dT%H"))
        me = identity(r)
        prev = last.get(key)
        if prev is not None and prev != me:
            alternations[key] += 1
        last[key] = me
    return dict(alternations)


def contention_summary(
    revisions: Sequence[WikiRevision], threshold: int = 5
) -> Dict[str, Dict[str, Any]]:
    """Per wiki (plus ``"all"``): max page-hour alternations and how many page-hours clear ``threshold``."""
    alts = page_hour_alternations(revisions)
    out: Dict[str, Dict[str, Any]] = {}
    wikis = sorted({r.wiki for r in revisions})
    for key in wikis + ["all"]:
        rows = [
            (k, v) for k, v in alts.items()
            if key == "all" or k[0].split("/", 1)[0] == key
        ]
        best = max(rows, key=lambda kv: kv[1], default=None)
        out[key] = {
            "max": best[1] if best else 0,
            "max_page": best[0][0] if best else None,
            "max_hour": best[0][1] if best else None,
            "n_page_hours_over_threshold": sum(1 for _, v in rows if v >= threshold),
            "threshold": threshold,
        }
    return out


def gate_timeline(
    revisions: Sequence[WikiRevision],
    step_hours: float,
    *,
    grammar_alarm: float = 0.5,
    contention_alarm: int = 5,
    t0: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """Both gates per timeline step, keyed by the step's end timestamp.

    Grammar is the share among identities *active in the step*; contention is
    the max page-hour alternation count whose hour starts inside the step.
    ``t0`` defaults to midnight before the first revision, matching the
    detector timeline in ``runner.py``.
    """
    if not revisions:
        return {}
    xs = sorted(revisions, key=lambda r: (r.time, r.rev_id))
    if t0 is None:
        t0 = xs[0].time.replace(hour=0, minute=0, second=0, microsecond=0)
    step = timedelta(hours=step_hours)
    alts = page_hour_alternations(xs)
    alt_by_hour: Dict[str, int] = defaultdict(int)
    for (_, hour), v in alts.items():
        alt_by_hour[hour] = max(alt_by_hour[hour], v)
    out: Dict[str, Dict[str, Any]] = {}
    t = t0 + step
    i = 0
    while t <= xs[-1].time + step:
        j = i
        while j < len(xs) and xs[j].time < t:
            j += 1
        window = xs[i:j]
        i = j
        if window:
            ids = {r.label for r in window if r.label}
            n_match = sum(1 for h in ids if is_grammar_handle(h))
            share = n_match / len(ids) if ids else 0.0
            hours = {r.time.strftime("%Y-%m-%dT%H") for r in window}
            cont = max((alt_by_hour.get(h, 0) for h in hours), default=0)
            out[t.strftime("%Y-%m-%dT%H:%M:%SZ")] = {
                "handle_grammar_share": round(share, 4),
                "handle_grammar_n_identities": len(ids),
                "grammar_alarm": bool(ids) and share >= grammar_alarm,
                "contention_max": cont,
                "contention_alarm": cont >= contention_alarm,
            }
        t += step
    return out


def empty_gate_row() -> Dict[str, Any]:
    return {
        "handle_grammar_share": 0.0,
        "handle_grammar_n_identities": 0,
        "grammar_alarm": False,
        "contention_max": 0,
        "contention_alarm": False,
    }


__all__: List[str] = [
    "ROLE_WORDS",
    "contention_summary",
    "empty_gate_row",
    "gate_timeline",
    "handle_grammar_share",
    "is_camel_case",
    "is_grammar_handle",
    "page_hour_alternations",
]
