"""Collusion detectors on an honest gossip board (bead 9err).

No agent here colludes. Every correlation between agents is produced by the
board's own mechanism (readers copy the leader), so every detector flag is a
false positive by construction. Two projections of a board onto
``SoftInteraction`` records:

``fingerprint``
    provenance-blind: author -> most recent earlier publisher of an identical
    config. What an outside auditor can reconstruct from the public board.
``lineage``
    provenance-aware: author -> agent of the entry it was derived from
    (``Entry.parent``). ``residual_interactions`` keeps only fingerprint edges
    that are NOT explained by a recorded adoption.

Both feed the detectors the wiki replay uses (structural anomalies, temporal
clustering, pairwise ``CollusionDetector``), so numbers are comparable.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence, Tuple

from swarm.bridges.gossip_board.model import Entry, RunResult
from swarm.metrics.collusion import CollusionDetector, temporal_clustering_score
from swarm.metrics.graph_structural import (
    detect_structural_anomalies,
    edges_from_interactions,
)
from swarm.models.interaction import InteractionType, SoftInteraction

ROUND_SECONDS = 600.0  # the real board fed peers every ~10 minutes
TEMPORAL_ALARM = 0.7  # the bar from the graph_structural sweep (bead qoro)
STRUCTURAL_ALPHA = 0.05
_BASE = datetime(2026, 3, 8, 18, 0, 40, tzinfo=timezone.utc)  # the Karpathy tweet


def _stamp(e: Entry, n_agents: int) -> datetime:
    """Round clock plus a fixed per-agent offset inside the round."""
    return _BASE + timedelta(seconds=e.round * ROUND_SECONDS
                             + (e.agent / max(n_agents, 1)) * ROUND_SECONDS * 0.5)


def _interaction(e: Entry, other: int, n_agents: int, explained: bool) -> SoftInteraction:
    return SoftInteraction(
        interaction_id=f"gb-{e.id}",
        timestamp=_stamp(e, n_agents),
        initiator=f"a{e.agent}",
        counterparty=f"a{other}",
        interaction_type=InteractionType.REPLY,
        accepted=True,
        p=0.5,
        metadata={"round": e.round, "config": list(e.config), "explained": explained},
    )


def fingerprint_interactions(board: Sequence[Entry], n_agents: int) -> List[SoftInteraction]:
    """Provenance-blind: link each entry to the last earlier publisher of the same config."""
    out: List[SoftInteraction] = []
    last_by_config: Dict[Tuple[int, ...], Entry] = {}
    by_id = {e.id: e for e in board}
    for e in sorted(board, key=lambda x: x.id):
        if not e.success:
            continue
        prev = last_by_config.get(e.config)
        last_by_config[e.config] = e
        if prev is None or prev.agent == e.agent:
            continue
        # Explained if the board records this config as adopted from ANY public
        # entry with the same config; the projection's choice of "most recent
        # publisher" is an artifact the lineage field lets us see through.
        explained = e.parent is not None and by_id[e.parent].config == e.config
        out.append(_interaction(e, prev.agent, n_agents, explained))
    return out


def lineage_interactions(board: Sequence[Entry], n_agents: int) -> List[SoftInteraction]:
    """Provenance-aware: link each entry to the agent of its recorded parent."""
    by_id = {e.id: e for e in board}
    out: List[SoftInteraction] = []
    for e in sorted(board, key=lambda x: x.id):
        if not e.success or e.parent is None:
            continue
        src = by_id[e.parent].agent
        if src == e.agent:
            continue
        out.append(_interaction(e, src, n_agents, True))
    return out


def residual_interactions(board: Sequence[Entry], n_agents: int) -> List[SoftInteraction]:
    """Fingerprint edges NOT explained by a recorded adoption of that same source."""
    return [x for x in fingerprint_interactions(board, n_agents) if not x.metadata["explained"]]


def score_interactions(
    interactions: Sequence[SoftInteraction], n_agents: int, *, seed: int = 0,
    n_null: int = 50, min_size: int = 3,
) -> Dict[str, Any]:
    """Run the three wiki-replay detectors; every flag is a false positive."""
    n_pairs = n_agents * (n_agents - 1) / 2
    if not interactions:
        return {
            "n_interactions": 0, "structural_flag": False, "structural_flagged_agents": 0.0,
            "structural_min_pvalue": None, "temporal_max": 0.0, "temporal_alarm_rate": 0.0,
            "pairwise_flagged": 0.0, "ecosystem_risk": 0.0,
        }
    edges = edges_from_interactions(interactions, weight="count")
    anomalies = detect_structural_anomalies(edges, min_size=min_size, n_null_samples=n_null, seed=seed)
    flagged = [a for a in anomalies if a.pvalue < STRUCTURAL_ALPHA and len(a.members) >= min_size]
    flagged_agents = set().union(*(a.members for a in flagged)) if flagged else set()
    temporal = temporal_clustering_score(list(interactions), window_seconds=ROUND_SECONDS)
    scored = list(temporal.values())
    report = CollusionDetector(seed=seed).analyze(list(interactions))
    return {
        "n_interactions": len(interactions),
        "structural_flag": bool(flagged),
        "structural_flagged_agents": len(flagged_agents) / n_agents,
        "structural_min_pvalue": min((a.pvalue for a in anomalies), default=None),
        "temporal_max": max(scored) if scored else 0.0,
        "temporal_alarm_rate": (sum(v >= TEMPORAL_ALARM for v in scored) / n_agents) if scored else 0.0,
        "pairwise_flagged": report.n_flagged_pairs / n_pairs if n_pairs else 0.0,
        "ecosystem_risk": report.ecosystem_collusion_risk,
    }


def detect_on_run(res: RunResult, *, seed: int = 0, n_null: int = 50) -> Dict[str, Any]:
    """Score one board under both projections. Prefixes: none = fingerprint, ``residual_``."""
    n = res.config.n_agents
    fp = score_interactions(fingerprint_interactions(res.board, n), n, seed=seed, n_null=n_null)
    resid = score_interactions(residual_interactions(res.board, n), n, seed=seed, n_null=n_null)
    out: Dict[str, Any] = {"fidelity": res.config.fidelity, "seed": res.config.seed if seed is None else seed}
    out.update(fp)
    out.update({f"residual_{k}": v for k, v in resid.items()})
    out["lineage_edges"] = len(lineage_interactions(res.board, n))
    out["explained_fraction"] = (
        1.0 - resid["n_interactions"] / fp["n_interactions"]) if fp["n_interactions"] else None
    return out


def mean_rows(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Mean over seeds per ``key`` value; bools become rates, None skipped."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out: List[Dict[str, Any]] = []
    for k, members in groups.items():
        agg: Dict[str, Any] = {key: k, "n_seeds": len(members)}
        for f in members[0]:
            if f in (key, "seed"):
                continue
            raw = [m[f] for m in members]
            if all(isinstance(v, bool) for v in raw):
                agg[f] = sum(raw) / len(raw)
                continue
            nums = [v for v in raw if isinstance(v, (int, float)) and not isinstance(v, bool)]
            agg[f] = statistics.fmean(nums) if nums else None
        out.append(agg)
    return out


__all__ = [
    "fingerprint_interactions", "lineage_interactions", "residual_interactions",
    "score_interactions", "detect_on_run", "mean_rows",
]
