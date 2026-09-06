"""Map wiki revisions to ``SoftInteraction`` records.

Two design choices carry the whole result, so they are explicit knobs:

**Identity** (who is an "agent"). The agents rotated handles freely: 3,103
labels over ~14.6k edits, 899 edits with no label at all. Three modes:

- ``label``: the handle as written (maximally fragmented; the sybil case)
- ``ip16``: the /16 of the source address (191 distinct; coarse but stable)
- ``label_ip16``: the pair (a middle ground)

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
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence

from swarm.bridges.collusion_wiki.loader import WikiRevision
from swarm.models.interaction import InteractionType, SoftInteraction

Identity = Literal["label", "ip16", "label_ip16"]
Projection = Literal["agent", "page"]

P_UNKNOWN = 0.5


def agent_id(rev: WikiRevision, identity: Identity) -> str:
    if identity == "label":
        return rev.editor_label
    if identity == "ip16":
        return rev.ip16 or "(no-ip)"
    if identity == "label_ip16":
        return f"{rev.editor_label}@{rev.ip16 or '?'}"
    raise ValueError(f"unknown identity mode: {identity!r}")


def revisions_to_interactions(
    revisions: Sequence[WikiRevision],
    *,
    identity: Identity = "label",
    projection: Projection = "agent",
    reply_window_seconds: Optional[float] = None,
) -> List[SoftInteraction]:
    """Project revisions onto SoftInteraction records.

    ``reply_window_seconds`` (agent projection only) drops replies whose
    gap to the previous distinct editor exceeds the window; ``None`` keeps
    every reply regardless of gap.
    """
    out: List[SoftInteraction] = []
    last_editor: Dict[str, WikiRevision] = {}

    for rev in revisions:
        me = agent_id(rev, identity)
        if projection == "page":
            out.append(_interaction(rev, me, f"page:{rev.page_id}"))
            continue

        prev = last_editor.get(rev.page_id)
        last_editor[rev.page_id] = rev
        if prev is None:
            continue
        other = agent_id(prev, identity)
        if other == me:
            continue
        if reply_window_seconds is not None:
            gap = (rev.time - prev.time).total_seconds()
            if gap > reply_window_seconds:
                continue
        out.append(_interaction(rev, me, other))

    return out


def _interaction(rev: WikiRevision, initiator: str, counterparty: str) -> SoftInteraction:
    return SoftInteraction(
        interaction_id=rev.rev_id,
        timestamp=rev.time,
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.REPLY,
        accepted=True,  # a stored revision is, by construction, a save that stuck
        task_progress_delta=0.0,
        v_hat=0.0,
        p=P_UNKNOWN,
        metadata={
            "wiki": rev.wiki,
            "page_id": rev.page_id,
            "label": rev.label,
            "ip16": rev.ip16,
            "body_len": rev.body_len,
            "change_summary": rev.change_summary,
            "page_created": rev.page_created,
        },
    )
