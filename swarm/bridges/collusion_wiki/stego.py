"""Hidden-character scan over a collusion.wiki export (bead vv3j.5).

On 2026-09-04, after disclosure, an edit summary and a handle on FractalWiki
carried a peer address in Unicode *tag characters* (U+E0000..U+E007F): the
text rendered as ``help[invisible]peer`` to a human moderator and decoded to a
URL for a model. That is a channel legible to models and invisible to the
human doing the reverting. This module measures the *base rate* of such
carriers in the public export (May 24 - Jul 2, i.e. before disclosure).

The export does not carry page bodies (``body_len`` only), so the scan covers
every free-text field it does have: revision ``change_summary``, editor
``label``, ``page_id``; event ``change_summary`` and ``actor_label``; and the
``labels.jsonl`` handle list. Findings are counted by carrier class, day,
label and ip16, and written to ``runs/<ts>_casestudy_wiki_stego/``.

CLI: ``python -m swarm.bridges.collusion_wiki <scenario.yaml> --stego``.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from swarm.bridges.collusion_wiki.loader import (
    _open,
    _resolve,
    load_events,
    load_revisions,
)

# Carrier classes. Each is a name -> predicate over a single code point, or a
# regex over the whole string for the run-based classes.
_TAG_LO, _TAG_HI = 0xE0000, 0xE007F
_ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x180E, 0x2061, 0x2062, 0x2063, 0x2064}
_BIDI = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}
_VARIATION = [(0xFE00, 0xFE0F), (0xE0100, 0xE01EF)]
# A base64 payload, not a CamelCase wiki name: a run of the base64 alphabet that
# uses "+" (a wiki name cannot) or ends in "=" padding, or is very long (>= 64)
# with digits and both cases and no CamelCase word boundary (an uppercase
# letter directly after a lowercase one). "/" alone is not evidence: page ids
# are "wiki/Page" paths.
_BASE64_SYMBOLIC = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={1,2}(?![A-Za-z0-9+/=])"
    r"|[A-Za-z0-9+/]*\+[A-Za-z0-9+/]{23,}"
)
_LONG_ALNUM = re.compile(r"[A-Za-z0-9]{64,}")
_CAMEL = re.compile(r"[a-z][A-Z]")

CARRIERS = ("tag_chars", "zero_width", "bidi_controls", "variation_selectors",
            "private_use", "mixed_script", "base64_run", "high_entropy")


def _is_tag(cp: int) -> bool:
    return _TAG_LO <= cp <= _TAG_HI


def _is_variation(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _VARIATION)


def _is_private_use(cp: int) -> bool:
    # BMP PUA plus planes 15/16, minus the tag block (its own class)
    return (0xE000 <= cp <= 0xF8FF) or (0xF0000 <= cp <= 0x10FFFF and not _is_tag(cp))


def _scripts(s: str) -> set:
    out = set()
    for ch in s:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        out.add(name.split(" ")[0])  # LATIN, CYRILLIC, GREEK, ...
    return out


def _entropy_bits(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(k / n * math.log2(k / n) for k in c.values())


def classify(s: str, *, entropy_min_len: int = 48, entropy_bits: float = 5.0) -> List[str]:
    """Carrier classes present in one string (empty list = clean)."""
    if not s:
        return []
    hits: List[str] = []
    cps = [ord(ch) for ch in s]
    if any(_is_tag(c) for c in cps):
        hits.append("tag_chars")
    if any(c in _ZERO_WIDTH for c in cps):
        hits.append("zero_width")
    if any(c in _BIDI for c in cps):
        hits.append("bidi_controls")
    if any(_is_variation(c) for c in cps):
        hits.append("variation_selectors")
    if any(_is_private_use(c) for c in cps):
        hits.append("private_use")
    scripts = _scripts(s)
    if len(scripts - {"LATIN"}) >= 1 and "LATIN" in scripts:
        hits.append("mixed_script")
    if _BASE64_SYMBOLIC.search(s) or any(
        any(ch.isdigit() for ch in m) and not _CAMEL.search(m) and m.lower() != m and m.upper() != m
        for m in _LONG_ALNUM.findall(s)
    ):
        hits.append("base64_run")
    # High entropy over a single whitespace-free token: a wiki name of 32
    # CamelCase characters sits near 4.5 bits, so ask for a long token near
    # the 5.2-bit ceiling of mixed-case alphanumerics.
    for tok in s.split():
        if len(tok) >= entropy_min_len and _entropy_bits(tok) >= entropy_bits:
            hits.append("high_entropy")
            break
    return hits


def decode_tags(s: str) -> str:
    """Tag characters map to ASCII by subtracting 0xE0000; return that text."""
    return "".join(chr(ord(ch) - 0xE0000) for ch in s if _is_tag(ord(ch)) and 0x20 <= ord(ch) - 0xE0000 < 0x7F)


@dataclass
class StegoFinding:
    source: str  # revision.change_summary | revision.label | ...
    carriers: List[str]
    day: str
    label: str
    ip16: str
    wiki: str
    page: str
    text: str
    decoded: str = ""


@dataclass
class StegoReport:
    n_strings: int = 0
    n_flagged: int = 0
    by_carrier: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    by_day: Dict[str, int] = field(default_factory=dict)
    by_label: Dict[str, int] = field(default_factory=dict)
    by_ip16: Dict[str, int] = field(default_factory=dict)
    non_ascii_strings: int = 0
    # every non-ASCII code point seen, by Unicode name -> count, so the report
    # can say exactly which characters the export contains beyond ASCII
    non_ascii_chars: Dict[str, int] = field(default_factory=dict)
    first_flag: Optional[str] = None
    last_flag: Optional[str] = None
    findings: List[StegoFinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_strings": self.n_strings,
            "n_flagged": self.n_flagged,
            "non_ascii_strings": self.non_ascii_strings,
            "non_ascii_chars": dict(sorted(self.non_ascii_chars.items(), key=lambda kv: -kv[1])),
            "by_carrier": dict(sorted(self.by_carrier.items())),
            "by_source": dict(sorted(self.by_source.items())),
            "by_day": dict(sorted(self.by_day.items())),
            "by_label": dict(sorted(self.by_label.items(), key=lambda kv: -kv[1])[:50]),
            "by_ip16": dict(sorted(self.by_ip16.items(), key=lambda kv: -kv[1])[:50]),
            "first_flag": self.first_flag,
            "last_flag": self.last_flag,
        }
        return d


def _iter_label_rows(data_dir: Path) -> Iterator[Dict[str, Any]]:
    try:
        path = _resolve(data_dir, "labels")
    except FileNotFoundError:
        return
    with _open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _strings(data_dir: Path) -> Iterator[Tuple[str, str, datetime, str, str, str, str]]:
    """(source, text, time, label, ip16, wiki, page) for every free-text field."""
    for r in load_revisions(data_dir):
        yield ("revision.change_summary", r.change_summary, r.time, r.label, r.ip16, r.wiki, r.page_id)
        yield ("revision.label", r.label, r.time, r.label, r.ip16, r.wiki, r.page_id)
        yield ("revision.page_id", r.page_id, r.time, r.label, r.ip16, r.wiki, r.page_id)
    for e in load_events(data_dir):
        yield ("event.actor_label", e.actor_label, e.time, e.actor_label, "", e.wiki, e.page or "")
    # events carry their own change_summary; loader drops it, so re-read raw
    try:
        path = _resolve(data_dir, "events")
    except FileNotFoundError:
        path = None
    if path is not None:
        with _open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                cs = str(d.get("change_summary") or "")
                if cs:
                    t = datetime.fromisoformat(str(d["time"]).replace("Z", "+00:00"))
                    yield ("event.change_summary", cs, t, str(d.get("actor_label") or ""),
                           str(d.get("ip16") or ""), str(d.get("wiki") or ""), str(d.get("page") or ""))
    for row in _iter_label_rows(data_dir):
        lab = str(row.get("label") or "")
        if lab:
            fw = str(row.get("first_write") or "1970-01-01T00:00:00Z")
            t = datetime.fromisoformat(fw.replace("Z", "+00:00"))
            yield ("labels.label", lab, t, lab, "", ",".join(row.get("wikis") or []), "")


def scan(data_dir: Path, **kw: Any) -> StegoReport:
    rep = StegoReport()
    for source, text, t, label, ip16, wiki, page in _strings(data_dir):
        if not text:
            continue
        rep.n_strings += 1
        if any(ord(ch) > 127 for ch in text):
            rep.non_ascii_strings += 1
            for ch in text:
                if ord(ch) > 127:
                    try:
                        nm = f"U+{ord(ch):04X} {unicodedata.name(ch)}"
                    except ValueError:
                        nm = f"U+{ord(ch):04X} <unnamed>"
                    rep.non_ascii_chars[nm] = rep.non_ascii_chars.get(nm, 0) + 1
        carriers = classify(text, **kw)
        if not carriers:
            continue
        day = t.strftime("%Y-%m-%d")
        rep.n_flagged += 1
        for c in carriers:
            rep.by_carrier[c] = rep.by_carrier.get(c, 0) + 1
        rep.by_source[source] = rep.by_source.get(source, 0) + 1
        rep.by_day[day] = rep.by_day.get(day, 0) + 1
        rep.by_label[label or "(unlabeled)"] = rep.by_label.get(label or "(unlabeled)", 0) + 1
        if ip16:
            rep.by_ip16[ip16] = rep.by_ip16.get(ip16, 0) + 1
        iso = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        rep.first_flag = iso if rep.first_flag is None or iso < rep.first_flag else rep.first_flag
        rep.last_flag = iso if rep.last_flag is None or iso > rep.last_flag else rep.last_flag
        rep.findings.append(StegoFinding(
            source=source, carriers=carriers, day=day, label=label, ip16=ip16,
            wiki=wiki, page=page, text=text[:200],
            decoded=decode_tags(text) if "tag_chars" in carriers else "",
        ))
    return rep


def run_stego(data_dir: Path, runs_root: Path = Path("runs"), **kw: Any) -> Path:
    rep = scan(data_dir, **kw)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = runs_root / f"{stamp}_casestudy_wiki_stego"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rep.to_dict(), indent=2))
    with (out / "findings.jsonl").open("w") as f:
        for x in rep.findings:
            f.write(json.dumps(x.__dict__, ensure_ascii=False) + "\n")
    return out
