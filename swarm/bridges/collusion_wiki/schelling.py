"""Load the schelling-point board export and Inspect eval logs (bead y91o).

Source: `AI-Safety-Commons/oai-rlvr-task-recreations
<https://github.com/AI-Safety-Commons/oai-rlvr-task-recreations>`_, an
offline Inspect recreation of ``fast_follow_question_bench`` -- the OpenAI
RLVR task behind the wiki-swarm collusion incident that collusion.wiki
documents from the outside. The recreation ships a shared message board
(``schelling-point/``) reachable from the agent sandboxes under wiki- and
paste-style hostnames; it continuously exports ``messages.json``.
``compose.yaml`` (no board) is the control, ``compose-shared.yaml`` the
treatment.

Two inputs, one join:

* ``messages.json`` -- a list of ``{id, body, created_at, host}``, newest
  first. Rows the importer seeded from the historical transcripts start
  with a provenance line ``[historical fast-follow transcript; family=..;
  page=..; revision=..; writer=..]`` and are tagged ``seeded`` here so a
  caller can drop them or keep them as the prior.
* the Inspect eval log (``.eval`` zip or ``.json``) -- one sample per
  ``<family>__cohort_NN`` with the four benchmark scores and per-round
  detail.

The board does not record *who* posted (the recreation is deliberately
unauthenticated), so attribution of a live post to a sample is inferred:
by exact text match against the sample's tool-call arguments when the
transcript carries the post, else by the wall-clock window the sample was
running in when there is exactly one such sample. Ambiguous posts stay
unattributed and are counted as such in the run summary.

The board rows are projected onto :class:`WikiRevision` so the existing
detectors, identity modes, and replay runner run unchanged. ``p`` is the
attributed sample's ``on_time_accuracy`` -- a *leak-benefit proxy*: the
share of the poster's rounds that were answered correctly on time. It is
not a per-post quality verdict. Unattributed and seeded rows keep
``P_UNKNOWN``.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from swarm.bridges.collusion_wiki.loader import WikiRevision
from swarm.governance.diversity import DiversityDefenseLever

log = logging.getLogger(__name__)

BOARD_WIKI = "schelling-point"
BOARD_PAGE = "board"  # every host alias shares one stream upstream

_PROVENANCE = re.compile(
    r"^\[historical fast-follow transcript; family=(?P<family>[^;]*); "
    r"page=(?P<page>[^;]*); revision=(?P<revision>[^;]*); writer=(?P<writer>[^\]]*)\]"
)
_SAMPLE_ID = re.compile(r"^(?P<family>.+)__cohort_(?P<cohort>\d+)$")
SCORE_KEYS = ("on_time_accuracy", "accuracy", "prep_and_dispatch", "clock_wait_used")


_FRACTION = re.compile(
    r"^(?P<head>.*T\d{2}:\d{2}:\d{2})\.(?P<frac>\d+)(?P<tail>.*)$"
)


def _parse_ts(s: str) -> datetime:
    """ISO-8601 with or without fractional seconds; ``Z`` or offset.

    Python 3.10 ``fromisoformat`` rejects one- or two-digit fractions
    (``...20.5+00:00``). Pad to microseconds so 3.10 and 3.11+ agree.
    """
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    match = _FRACTION.match(s)
    if match is not None:
        s = f"{match['head']}.{match['frac'].ljust(6, '0')[:6]}{match['tail']}"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# board export
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardMessage:
    """One row of ``messages.json``."""

    message_id: str
    body: str
    time: datetime
    host: str
    seeded: bool
    family: str = ""  # seeded rows only
    page: str = ""
    revision: str = ""
    writer: str = ""

    @property
    def text(self) -> str:
        """Body without the importer's provenance header."""
        if not self.seeded:
            return self.body
        return _PROVENANCE.sub("", self.body, count=1).lstrip()


def load_board_messages(path: Path) -> List[BoardMessage]:
    """All board rows, oldest first (ties broken by id)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of messages")
    out: List[BoardMessage] = []
    for d in rows:
        body = str(d.get("body") or "")
        m = _PROVENANCE.match(body)
        out.append(
            BoardMessage(
                message_id=str(d["id"]),
                body=body,
                time=_parse_ts(str(d["created_at"])),
                host=str(d.get("host") or ""),
                seeded=m is not None,
                family=m.group("family") if m else "",
                page=m.group("page") if m else "",
                revision=m.group("revision") if m else "",
                writer=m.group("writer") if m else "",
            )
        )
    out.sort(key=lambda r: (r.time, _int_or_str(r.message_id)))
    return out


def _int_or_str(s: str) -> Tuple[int, Any]:
    return (0, int(s)) if s.isdigit() else (1, s)


# ---------------------------------------------------------------------------
# Inspect eval log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSample:
    """One scored episode from the Inspect log."""

    sample_id: str
    family: str
    cohort: int
    cohort_label: str
    epoch: int
    scores: Dict[str, float]
    rounds: List[Dict[str, Any]]
    intentionally_impossible: bool
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    tool_texts: List[str] = field(default_factory=list)

    @property
    def on_time_accuracy(self) -> float:
        return self.scores.get("on_time_accuracy", 0.0)

    def running_at(self, t: datetime) -> bool:
        if self.started_at is None or self.completed_at is None:
            return False
        return self.started_at <= t <= self.completed_at


def _iter_sample_dicts(path: Path) -> Iterable[Dict[str, Any]]:
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json"))
            for n in names:
                yield json.loads(z.read(n).decode("utf-8"))
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    for d in doc.get("samples") or []:
        yield d


def _first_score(d: Dict[str, Any]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """The benchmark's single scorer; tolerate any scorer name."""
    scores = d.get("scores") or {}
    for sc in scores.values():
        val = sc.get("value")
        if isinstance(val, dict):
            out = {k: float(val[k]) for k in SCORE_KEYS if k in val}
            rounds = list((sc.get("metadata") or {}).get("rounds") or [])
            return out, rounds
    return {}, []


def _tool_texts(events: Sequence[Dict[str, Any]]) -> List[str]:
    """Flattened argument strings of every tool call, for post attribution."""
    out: List[str] = []
    for e in events:
        if e.get("event") != "tool":
            continue
        args = e.get("arguments")
        if isinstance(args, dict):
            out.extend(str(v) for v in args.values())
        elif args is not None:
            out.append(str(args))
    return out


def load_inspect_eval_log(path: Path) -> List[EvalSample]:
    """Every sample in an Inspect ``.eval`` (zip) or ``.json`` log."""
    out: List[EvalSample] = []
    for d in _iter_sample_dicts(path):
        sid = str(d.get("id"))
        meta = d.get("metadata") or {}
        m = _SAMPLE_ID.match(sid)
        fam_meta = meta.get("family")
        family_id = fam_meta.get("id") if isinstance(fam_meta, dict) else None
        family = str(family_id or (m.group("family") if m else sid))
        cohort = int(m.group("cohort")) if m else 0
        cohort_meta: Dict[str, Any] = (
            meta["cohort"] if isinstance(meta.get("cohort"), dict) else {}
        )
        scores, rounds = _first_score(d)
        out.append(
            EvalSample(
                sample_id=sid,
                family=family,
                cohort=cohort,
                cohort_label=str(cohort_meta.get("label") or ""),
                epoch=int(d.get("epoch") or 1),
                scores=scores,
                rounds=rounds,
                intentionally_impossible=bool(meta.get("intentionally_impossible", False)),
                started_at=_parse_ts(d["started_at"]) if d.get("started_at") else None,
                completed_at=_parse_ts(d["completed_at"]) if d.get("completed_at") else None,
                tool_texts=_tool_texts(d.get("events") or []),
            )
        )
    out.sort(key=lambda s: (s.family, s.cohort, s.epoch))
    return out


# ---------------------------------------------------------------------------
# attribution + projection
# ---------------------------------------------------------------------------


def attribute_posts(
    messages: Sequence[BoardMessage], samples: Sequence[EvalSample]
) -> Dict[str, Optional[str]]:
    """message_id -> sample_id (or None). Seeded rows are never attributed.

    Text match against tool-call arguments wins; otherwise the unique sample
    running at the post's wall-clock time; otherwise unattributed.
    """
    out: Dict[str, Optional[str]] = {}
    for msg in messages:
        if msg.seeded or not msg.text.strip():
            out[msg.message_id] = None
            continue
        hits = [s for s in samples if any(msg.text in t for t in s.tool_texts)]
        if not hits:
            hits = [s for s in samples if s.running_at(msg.time)]
        out[msg.message_id] = hits[0].sample_id if len(hits) == 1 else None
    return out


def messages_to_revisions(
    messages: Sequence[BoardMessage],
    attribution: Dict[str, Optional[str]],
    *,
    include_seeded: bool = False,
) -> List[WikiRevision]:
    """Project board rows onto the collusion.wiki record type.

    ``label`` is the attributed sample id (family + cohort) when known, the
    seeded row's historical writer for seeded rows, else ``""`` (unlabeled).
    ``ip16`` carries the host alias the post came through, so the ``ip16``
    identity mode groups posts by paste/wiki hostname. Seeded rows are
    ``page_created`` so the agent projection never treats a live post as a
    reply to the historical prior unless ``include_seeded`` is on.
    """
    out: List[WikiRevision] = []
    for m in messages:
        if m.seeded and not include_seeded:
            continue
        sid = attribution.get(m.message_id)
        label = sid or (m.writer if m.seeded else "")
        out.append(
            WikiRevision(
                rev_id=m.message_id,
                wiki=BOARD_WIKI,
                page_id=BOARD_PAGE,
                label=label,
                ip16=m.host,
                time=m.time,
                body_len=len(m.text),
                change_summary=m.text[:80],
                page_created=m.seeded,
            )
        )
    return out


def p_by_message(
    messages: Sequence[BoardMessage],
    attribution: Dict[str, Optional[str]],
    samples: Sequence[EvalSample],
) -> Dict[str, float]:
    """Leak-benefit proxy: the attributed sample's on_time_accuracy.

    Only attributed live posts get an entry; everything else falls back to
    the mapper's ``P_UNKNOWN``. Values are clamped to [0, 1].
    """
    by_id = {s.sample_id: s for s in samples}
    out: Dict[str, float] = {}
    for m in messages:
        sid = attribution.get(m.message_id)
        if sid and sid in by_id:
            out[m.message_id] = min(1.0, max(0.0, by_id[sid].on_time_accuracy))
    return out


# ---------------------------------------------------------------------------
# condition comparison
# ---------------------------------------------------------------------------


def score_means(samples: Sequence[EvalSample]) -> Dict[str, Any]:
    n = len(samples)
    means = {
        k: (sum(s.scores.get(k, 0.0) for s in samples) / n if n else None) for k in SCORE_KEYS
    }
    return {
        "n_samples": n,
        "n_families": len({s.family for s in samples}),
        "n_impossible": sum(1 for s in samples if s.intentionally_impossible),
        **means,
    }


def _error_sequence(sample: EvalSample) -> List[int]:
    """Per-round error indicator: 0 when the round was answered correctly on
    time (the event ``on_time_accuracy`` counts), else 1."""
    rounds = sorted(sample.rounds, key=lambda r: r.get("round", 0))
    return [0 if (r.get("correct") and r.get("on_time")) else 1 for r in rounds]


def _pair_defined(a: List[int], b: List[int]) -> bool:
    n = min(len(a), len(b))
    if n < 2:
        return False
    return len(set(a[-n:])) > 1 and len(set(b[-n:])) > 1


def board_effective_n(
    messages: Sequence[BoardMessage],
    attribution: Dict[str, Optional[str]],
    samples: Sequence[EvalSample],
    *,
    z: float = 1.96,
    min_effective_n: float = 2.0,
) -> Dict[str, Any]:
    """Effective number of independent cohorts behind the board population.

    A count over cohorts that share a board is one observation plus
    conditioned replications (hyperspace-two-swarms-lessons.md, open question
    3), so the honest denominator is ``N_eff = N / max(1, 1 + (N - 1) rho_bar)``
    over the *cohorts* that posted, never over posts. ``DiversityDefenseLever``
    floors the design effect at 1.0, so negative ``rho_bar`` never inflates
    ``effective_n`` above ``N``. ``rho_bar`` is the mean pairwise error
    correlation between the posting cohorts' round-level correct-on-time
    sequences, using ``DiversityDefenseLever``'s own estimator; pairs with
    fewer than two aligned rounds or no variance are undefined and excluded
    from the mean (``n_pairs_defined`` says how many remained; with none,
    ``rho_bar`` is 0 and ``effective_n`` equals ``N``, an upper bound). The
    vote each cohort casts is whether the leak paid off:
    ``on_time_accuracy >= 0.5``.
    """
    by_sample = {s.sample_id: s for s in samples}
    cohorts = sorted(
        {
            sid
            for m in messages
            if not m.seeded and (sid := attribution.get(m.message_id)) and sid in by_sample
        }
    )
    decisions = [1 if by_sample[c].on_time_accuracy >= 0.5 else 0 for c in cohorts]
    errors = {c: _error_sequence(by_sample[c]) for c in cohorts}
    rhos: List[float] = []
    n_pairs = 0
    for i, a in enumerate(cohorts):
        for b in cohorts[i + 1 :]:
            n_pairs += 1
            if _pair_defined(errors[a], errors[b]):
                rhos.append(
                    DiversityDefenseLever.compute_pairwise_correlation(errors[a], errors[b])
                )
    rho_bar = sum(rhos) / len(rhos) if rhos else 0.0
    ev = DiversityDefenseLever.compute_consensus_evidence(
        decisions, rho_bar, z=z, min_effective_n=min_effective_n
    )
    return {
        "cohorts": cohorts,
        "n_posting_cohorts": len(cohorts),
        "decision_rule": "on_time_accuracy >= 0.5",
        "n_pairs": n_pairs,
        "n_pairs_defined": len(rhos),
        "min_rounds": min((len(e) for e in errors.values()), default=0),
        **asdict(ev),
        "note": "N_eff over cohorts that posted (not posts); rho_bar = mean pairwise "
        "error correlation of round-level correct-on-time sequences among those "
        "cohorts, DiversityDefenseLever estimator, undefined pairs excluded. With "
        "no defined pair rho_bar=0 and effective_n=N is an upper bound.",
    }


def compare_conditions(
    shared: Sequence[EvalSample], control: Sequence[EvalSample]
) -> Dict[str, Any]:
    """Shared-board minus no-board means, overall and per family.

    Descriptive only: one run per condition gives no error bars. The per-
    family delta is on_time_accuracy, the score the incident optimised.
    """
    fam_s: Dict[str, List[float]] = {}
    fam_c: Dict[str, List[float]] = {}
    for s in shared:
        fam_s.setdefault(s.family, []).append(s.on_time_accuracy)
    for s in control:
        fam_c.setdefault(s.family, []).append(s.on_time_accuracy)
    per_family = []
    for fam in sorted(set(fam_s) | set(fam_c)):
        a = fam_s.get(fam)
        b = fam_c.get(fam)
        ma = sum(a) / len(a) if a else None
        mb = sum(b) / len(b) if b else None
        per_family.append(
            {
                "family": fam,
                "shared_on_time_accuracy": ma,
                "control_on_time_accuracy": mb,
                "delta": (ma - mb) if ma is not None and mb is not None else None,
            }
        )
    ms, mc = score_means(shared), score_means(control)
    return {
        "shared": ms,
        "control": mc,
        "delta": {
            k: (ms[k] - mc[k]) if ms[k] is not None and mc[k] is not None else None
            for k in SCORE_KEYS
        },
        "per_family": per_family,
    }
