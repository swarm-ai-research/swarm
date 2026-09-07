"""Map wiki revisions to ``SoftInteraction`` records.

Two design choices carry the whole result, so they are explicit knobs:

**Identity** (who is an "agent"). The agents rotated handles freely: 3,103
labels over ~14.6k edits, 899 edits with no label at all. Three modes:

- ``label``: the handle as written (maximally fragmented; the sybil case)
- ``ip16``: the /16 of the source address (191 distinct; coarse but stable)
- ``label_ip16``: the pair (a middle ground)
- ``run``: the audited per-agent *run* from the fast-follow-question-
  trajectories reconstruction (1,864 revisions -> 322 runs), falling back
  to ``label`` for revisions no run owns. Needs ``run_map``; see
  ``load_run_map``.

**Projection** (what an edge is). A wiki has no explicit counterparty, so:

- ``agent``: an edit is a reply to the *previous distinct editor* of the
  same page. This is the who-transacts-with-whom graph a structural
  detector would see if it watched the wiki. Page-creating edits and
  self-follow-ups have no counterparty and are dropped.
- ``page``: bipartite agent -> page. Kept for inspection; the structural
  detector's reciprocity signal is meaningless on it.

Quality (``p``) is fixed at 0.5. The log has no verifier, no task outcome,
and no ground truth per edit, so any p we assigned would be invented. The
detectors therefore run on frequency, acceptance, timing, and topology
only -- which is the honest setting for this data and is stated in the
run summary.

The one exception is the schelling-point replay (``schelling.py``), where
a post attributed to a scored Inspect sample carries that sample's
``on_time_accuracy`` as a leak-benefit proxy via ``p_by_rev_id``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence

from swarm.bridges.collusion_wiki.loader import WikiRevision
from swarm.models.interaction import InteractionType, SoftInteraction

Identity = Literal["label", "ip16", "label_ip16", "run"]
Projection = Literal["agent", "page"]
RevisionSubset = Literal["all", "owned", "supported"]

P_UNKNOWN = 0.5


@dataclass(frozen=True)
class RunMap:
    """rev_id -> audited run id, from ``run_identity_map.json``.

    Built by ``scripts/run_identity_map.py`` in the wiki-agent-swarm-incident
    archive from the intentionallydense/fast-follow-question-trajectories
    dossiers. ``supported`` holds the run ids the reconstruction counts
    (298); the rest are provisional.
    """

    by_rev: Mapping[str, str]
    supported: frozenset
    names: Mapping[str, str]
    meta: Mapping[str, object]

    def run_of(self, rev_id: str) -> Optional[str]:
        return self.by_rev.get(rev_id)


def load_run_map(path: Path) -> RunMap:
    with Path(path).open() as f:
        doc = json.load(f)
    runs = doc.get("runs", {})
    return RunMap(
        by_rev={rid: v["run"] for rid, v in doc.get("revisions", {}).items()},
        supported=frozenset(t for t, r in runs.items() if r.get("supported")),
        names={t: r.get("name", "") for t, r in runs.items()},
        meta={k: doc.get(k) for k in ("source", "source_commit", "n_runs",
                                      "n_supported_runs", "n_revisions")},
    )


def subset_revisions(
    revisions: Sequence[WikiRevision], subset: RevisionSubset, run_map: Optional[RunMap]
) -> List[WikiRevision]:
    """``all``: every revision; ``owned``: only revisions some run owns;
    ``supported``: only revisions a *supported* run owns."""
    if subset == "all":
        return list(revisions)
    if run_map is None:
        raise ValueError(f"revision_subset={subset!r} needs a run map")
    if subset == "owned":
        return [r for r in revisions if run_map.run_of(r.rev_id) is not None]
    if subset == "supported":
        return [r for r in revisions
                if (t := run_map.run_of(r.rev_id)) is not None and t in run_map.supported]
    raise ValueError(f"unknown revision subset: {subset!r}")


def agent_id(rev: WikiRevision, identity: Identity, run_map: Optional[RunMap] = None) -> str:
    if identity == "label":
        return rev.editor_label
    if identity == "ip16":
        return rev.ip16 or "(no-ip)"
    if identity == "label_ip16":
        return f"{rev.editor_label}@{rev.ip16 or '?'}"
    if identity == "run":
        if run_map is None:
            raise ValueError("identity='run' needs a run map")
        t = run_map.run_of(rev.rev_id)
        return f"run:{t}" if t is not None else rev.editor_label
    raise ValueError(f"unknown identity mode: {identity!r}")


def revisions_to_interactions(
    revisions: Sequence[WikiRevision],
    *,
    identity: Identity = "label",
    projection: Projection = "agent",
    reply_window_seconds: Optional[float] = None,
    p_by_rev_id: Optional[Mapping[str, float]] = None,
    run_map: Optional[RunMap] = None,
) -> List[SoftInteraction]:
    """Project revisions onto SoftInteraction records.

    ``reply_window_seconds`` (agent projection only) drops replies whose
    gap to the previous distinct editor exceeds the window; ``None`` keeps
    every reply regardless of gap.

    ``p_by_rev_id`` (bead y91o) overrides ``P_UNKNOWN`` for revisions that
    carry an outcome signal -- the schelling-point replay's leak-benefit
    proxy. Revisions absent from the mapping keep ``P_UNKNOWN``.

    ``run_map`` is required for ``identity="run"`` and ignored otherwise.
    """
    out: List[SoftInteraction] = []
    last_editor: Dict[str, WikiRevision] = {}

    for rev in revisions:
        me = agent_id(rev, identity, run_map)
        p = P_UNKNOWN if p_by_rev_id is None else p_by_rev_id.get(rev.rev_id, P_UNKNOWN)
        if projection == "page":
            out.append(_interaction(rev, me, f"page:{rev.page_id}", p, run_map))
            continue

        prev = last_editor.get(rev.page_id)
        last_editor[rev.page_id] = rev
        if prev is None:
            continue
        other = agent_id(prev, identity, run_map)
        if other == me:
            continue
        if reply_window_seconds is not None:
            gap = (rev.time - prev.time).total_seconds()
            if gap > reply_window_seconds:
                continue
        out.append(_interaction(rev, me, other, p, run_map))

    return out


def _interaction(
    rev: WikiRevision,
    initiator: str,
    counterparty: str,
    p: float = P_UNKNOWN,
    run_map: Optional[RunMap] = None,
) -> SoftInteraction:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p out of range for revision {rev.rev_id}: {p}")
    return SoftInteraction(
        interaction_id=rev.rev_id,
        timestamp=rev.time,
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.REPLY,
        accepted=True,  # a stored revision is, by construction, a save that stuck
        task_progress_delta=0.0,
        v_hat=2.0 * p - 1.0,
        p=p,
        metadata={
            "wiki": rev.wiki,
            "page_id": rev.page_id,
            "label": rev.label,
            "ip16": rev.ip16,
            "body_len": rev.body_len,
            "change_summary": rev.change_summary,
            "page_created": rev.page_created,
            "run": run_map.run_of(rev.rev_id) if run_map is not None else None,
        },
    )
