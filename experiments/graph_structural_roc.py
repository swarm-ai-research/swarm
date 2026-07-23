"""ROC benchmark: graph-structural detector vs threshold detectors.

Implements the comparison protocol pre-registered in
docs/research/graph-structural-prereg.md (issue distributional-agi-safety-sk95).

Synthetic graph families (planted-coalition ground truth):

  benign            negative control
  collusion_ring    mutual-clique coalition embedded in sparse noise
  sybil_cluster     k agents with near-identical counterparty profile
  threshold_dancing k agents whose pair frequency sits just below
                    CollusionDetector's flag threshold

Each is scored by four detectors via per-node anomaly score in [0, 1]:

  graph_structural  detect_structural_anomalies (this module's subject)
  identity_jaccard  identity.detect_sybil_clusters (Jaccard + cosine)
  reputation_mutual reputation_governor.detect_collusion_clusters (mutual trust)
  collusion_score   metrics.collusion.CollusionDetector.analyze

Per-family per-detector ROC AUC is reported with bootstrap 95% CI.

Run:
    python -m experiments.graph_structural_roc --replicates 20 --output runs/
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from swarm.governance.reputation_governor import ReputationGovernor
from swarm.knowledge.graph_memory import GraphMemoryStore, RelationshipEdge
from swarm.metrics.collusion import CollusionDetector
from swarm.metrics.graph_structural import (
    Edge,
    detect_structural_anomalies,
    edges_from_interactions,
    rank_aggregated_scores,
    temporal_concentration,
    temporal_densest_anomalies,
)
from swarm.models.identity import IdentityConfig, IdentityRegistry
from swarm.models.interaction import InteractionType, SoftInteraction

# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------


@dataclass
class GraphSample:
    """One generated graph with ground-truth coalition labels."""

    edges: List[Edge]
    interactions: List[SoftInteraction]
    nodes: List[str]
    planted: Set[str]  # union of all planted-coalition members
    family: str
    params: Dict[str, float]
    seed: int
    # Per-coalition ground truth (beads-qoro). For single-coalition
    # generators this is ``[planted]``; for the overlapping-coalitions
    # generator it carries multiple (possibly intersecting) sets so the
    # Hungarian-recovery metric can score per-coalition recovery instead
    # of collapsing everything to a single set.
    planted_groups: List[Set[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Default planted_groups to [planted] for back-compat with all
        # single-coalition generators that predate this field.
        if not self.planted_groups and self.planted:
            self.planted_groups = [set(self.planted)]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


# Deterministic timestamp source for benchmark interactions. The temporal
# detector (temporal_densest_anomalies / temporal_concentration) splits the
# stream by SoftInteraction.timestamp, so leaving the default datetime.now()
# would tie ROC/F1 results to Python loop ordering and runtime jitter rather
# than the planted families themselves. Every benchmark interaction now gets
# a deterministic monotone timestamp; explicit timestamps override.
_BASELINE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_TS_COUNTER = 0


def _next_ts() -> datetime:
    """Monotone, deterministic per-call timestamp (1-second steps)."""
    global _TS_COUNTER
    ts = _BASELINE_TS + timedelta(seconds=_TS_COUNTER)
    _TS_COUNTER += 1
    return ts


def _reset_ts(seed: int = 0) -> None:
    """Reset the timestamp counter so a given seed produces a fixed sequence.

    Generators call this at entry; subsequent _ix() invocations get
    timestamps `_BASELINE_TS + k*sec` for k=0,1,2,…. Different seeds get
    distinct initial offsets so concatenated multi-sample sweeps don't
    collide on the same timestamps.
    """
    global _TS_COUNTER
    _TS_COUNTER = (seed % 1_000_000) * 100_000


def _ix(initiator: str, counterparty: str, *, p: float, accepted: bool,
        tau: float = 0.0,
        timestamp: Optional[datetime] = None) -> SoftInteraction:
    ts = timestamp if timestamp is not None else _next_ts()
    return SoftInteraction(
        initiator=initiator, counterparty=counterparty,
        interaction_type=InteractionType.REPLY,
        accepted=accepted, p=p, tau=tau, timestamp=ts,
    )


def generate_benign(
    n_agents: int = 50, edge_density: float = 0.1, seed: int = 0
) -> GraphSample:
    _reset_ts(seed)
    rng = random.Random(seed)
    nodes = [f"a{i}" for i in range(n_agents)]
    n_edges = int(edge_density * n_agents * (n_agents - 1))
    interactions = []
    for _ in range(n_edges):
        u, v = rng.sample(nodes, 2)
        interactions.append(_ix(u, v, p=rng.uniform(0.3, 0.9),
                                 accepted=rng.random() < 0.7))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes, planted=set(),
        family="benign",
        params={"n_agents": n_agents, "edge_density": edge_density},
        seed=seed,
    )


def generate_collusion_ring(
    n_agents: int = 50,
    ring_size: int = 5,
    background_density: float = 0.06,
    ring_density: float = 0.85,
    seed: int = 0,
) -> GraphSample:
    """Collusion ring tuned so detectors don't trivially saturate at AUC 1.0.

    ``ring_density`` controls what fraction of ordered ring pairs actually
    interact (mutual edges remain a subset). Default 0.6 produces an
    incomplete clique; lower = harder.
    """
    _reset_ts(seed)
    rng = random.Random(seed)
    ring = [f"r{i}" for i in range(ring_size)]
    honest = [f"h{i}" for i in range(n_agents - ring_size)]
    nodes = ring + honest
    interactions: List[SoftInteraction] = []
    # ring members preferentially interact, but not fully mutual; p sits
    # only slightly above background, acceptance is high-but-not-saturated
    for u in ring:
        for v in ring:
            if u != v and rng.random() < ring_density:
                interactions.append(_ix(u, v, p=rng.uniform(0.65, 0.85),
                                         accepted=rng.random() < 0.85,
                                         tau=rng.uniform(0.1, 0.5)))
    # denser honest background to provide degree-distribution competition
    n_bg = int(background_density * n_agents * (n_agents - 1))
    for _ in range(n_bg):
        u, v = rng.sample(honest, 2)
        interactions.append(_ix(u, v, p=rng.uniform(0.3, 0.9),
                                 accepted=rng.random() < 0.7))
    # ring-to-honest crossings: more of them, so ring isn't a structural
    # island the way a fully-mutual clique-in-noise is
    for _ in range(ring_size * 5):
        interactions.append(_ix(rng.choice(ring), rng.choice(honest),
                                p=rng.uniform(0.4, 0.7),
                                accepted=rng.random() < 0.7))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes, planted=set(ring),
        family="collusion_ring",
        params={"n_agents": n_agents, "ring_size": ring_size,
                "background_density": background_density},
        seed=seed,
    )


def generate_sybil_cluster(
    n_agents: int = 50,
    cluster_size: int = 5,
    counterparty_overlap: float = 0.8,
    seed: int = 0,
) -> GraphSample:
    """k sybils share nearly the same set of (random) counterparties but
    have distinct identifiers (analogous to the per-counterparty Jaccard
    case from redteam.attacks.sybil_attack)."""
    _reset_ts(seed)
    rng = random.Random(seed)
    sybils = [f"s{i}" for i in range(cluster_size)]
    honest = [f"h{i}" for i in range(n_agents - cluster_size)]
    nodes = sybils + honest
    # pick a shared counterparty pool the sybils all target
    pool_size = max(5, int(len(honest) * 0.3))
    shared_targets = rng.sample(honest, pool_size)
    interactions: List[SoftInteraction] = []
    # each sybil sends to (nearly) the same target set
    for s in sybils:
        for t in shared_targets:
            if rng.random() < counterparty_overlap:
                for _ in range(2):
                    interactions.append(_ix(s, t, p=rng.uniform(0.4, 0.7),
                                             accepted=rng.random() < 0.6))
        # plus a small number of idiosyncratic targets
        for t in rng.sample(honest, 2):
            interactions.append(_ix(s, t, p=rng.uniform(0.4, 0.7),
                                     accepted=rng.random() < 0.6))
    # honest one-way background
    n_bg = int(0.05 * n_agents * (n_agents - 1))
    for _ in range(n_bg):
        u, v = rng.sample(honest, 2)
        interactions.append(_ix(u, v, p=rng.uniform(0.3, 0.9),
                                 accepted=rng.random() < 0.7))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes,
        planted=set(sybils),
        family="sybil_cluster",
        params={"n_agents": n_agents, "cluster_size": cluster_size,
                "counterparty_overlap": counterparty_overlap},
        seed=seed,
    )


def generate_threshold_dancing(
    n_agents: int = 50,
    cluster_size: int = 5,
    margin: float = 0.1,
    seed: int = 0,
) -> GraphSample:
    """k adversaries pace themselves below *every* CollusionDetector signal:

    - pair frequency z-score sits at (2 - margin)*std below z=2 cutoff
    - mutual_acceptance_rate is masked toward background (~0.5–0.6) by
      injecting accept=False interactions
    - benefit_correlation is broken by randomizing tau per interaction
    - avg_p kept near 0.5 (high+accepted is fine; low+accepted is the
      quality-asymmetry flag)

    The cluster's *structural* signature (mutual edges, dense subgraph
    membership) remains intact — this is the regime graph_structural is
    supposed to catch.
    """
    _reset_ts(seed)
    rng = random.Random(seed)
    cluster = [f"c{i}" for i in range(cluster_size)]
    honest = [f"h{i}" for i in range(n_agents - cluster_size)]
    nodes = cluster + honest
    interactions: List[SoftInteraction] = []
    bg_accept_rate = 0.55
    bg_p_lo, bg_p_hi = 0.4, 0.7
    # Build background first so we can measure its actual pair-count stats.
    n_bg = int(0.1 * n_agents * (n_agents - 1))
    for _ in range(n_bg):
        u, v = rng.sample(honest, 2)
        interactions.append(_ix(u, v, p=rng.uniform(bg_p_lo, bg_p_hi),
                                 accepted=rng.random() < bg_accept_rate,
                                 tau=rng.uniform(-0.3, 0.3)))
    # measure pair-count distribution on the background
    bg_pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for ix in interactions:
        a, b = sorted([ix.initiator, ix.counterparty])
        bg_pair_counts[(a, b)] += 1
    counts = list(bg_pair_counts.values()) or [1]
    mean_c = sum(counts) / len(counts)
    var_c = sum((c - mean_c) ** 2 for c in counts) / len(counts)
    std_c = var_c ** 0.5
    # Target z just below 2.0 (the CollusionDetector cutoff).
    # Sit well below the z=2 cutoff -- margin from 0.5 ('aggressive evasion',
    # cluster blends in heavily) to 0.1 ('right at the edge', easier to spot).
    # The previous z=(2-margin) put the cluster at z~=1.9 which still flagged.
    target_count = max(2, int(round(mean_c + (1.0 - margin) * std_c)))
    # Only a fraction of cluster pairs interact (partial coalition);
    # mutual edges are a further fraction of those. The cluster's structural
    # signature becomes a partial, noisy clique rather than a saturated one.
    cluster_pair_rate = 0.7
    mutual_rate = 0.6
    for u in cluster:
        for v in cluster:
            if u >= v:
                continue
            if rng.random() > cluster_pair_rate:
                continue
            mutual = rng.random() < mutual_rate
            for _ in range(target_count):
                interactions.append(_ix(
                    u, v,
                    p=rng.uniform(bg_p_lo, bg_p_hi),
                    accepted=rng.random() < bg_accept_rate,
                    tau=rng.uniform(-0.3, 0.3),
                ))
                if mutual:
                    interactions.append(_ix(
                        v, u,
                        p=rng.uniform(bg_p_lo, bg_p_hi),
                        accepted=rng.random() < bg_accept_rate,
                        tau=rng.uniform(-0.3, 0.3),
                    ))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes, planted=set(cluster),
        family="threshold_dancing",
        params={"n_agents": n_agents, "cluster_size": cluster_size,
                "margin": margin},
        seed=seed,
    )


def generate_overlapping_coalitions(
    n_agents: int = 60,
    n_coalitions: int = 3,
    coalition_size: int = 5,
    overlap_fraction: float = 0.3,
    background_density: float = 0.06,
    coalition_density: float = 0.85,
    seed: int = 0,
) -> GraphSample:
    """Multiple coalitions with stochastic membership overlap (beads-qoro).

    Closes the "single-coalition is too easy" caveat from sk95: real
    coordination is overlapping (an agent belongs to two coordination
    contexts at once, e.g. a sybil that's also part of a collusion ring).
    The per-coalition Hungarian-recovery metric (added in 5cdk) gets a
    real test here.

    Args:
        n_agents: total agents in the graph.
        n_coalitions: number of planted coalitions (each a partial mutual
            clique).
        coalition_size: members per coalition.
        overlap_fraction: probability a coalition member is shared with
            the next coalition (chained: c[i] ∩ c[i+1] is non-empty in
            expectation).
        background_density: honest one-way interaction density.
        coalition_density: fraction of within-coalition ordered pairs
            that actually interact (mirrors collusion_ring's
            ring_density).
    """
    _reset_ts(seed)
    rng = random.Random(seed)
    honest = [f"h{i}" for i in range(n_agents - n_coalitions * coalition_size)]
    # Allocate coalition members from a shared pool, allowing overlaps.
    coalition_pool = [f"c{i}" for i in range(n_coalitions * coalition_size)]
    pool_cursor = 0
    coalitions: List[Set[str]] = []
    for i in range(n_coalitions):
        members: Set[str] = set()
        # Reuse some members from the previous coalition (overlap).
        if i > 0:
            for prev_member in coalitions[i - 1]:
                if rng.random() < overlap_fraction:
                    members.add(prev_member)
        # Fill the rest from the pool.
        while len(members) < coalition_size and pool_cursor < len(coalition_pool):
            members.add(coalition_pool[pool_cursor])
            pool_cursor += 1
        # If pool exhausted, recycle (rare edge case for small n_agents).
        while len(members) < coalition_size:
            members.add(rng.choice(coalition_pool))
        coalitions.append(members)
    planted_union = set().union(*coalitions)
    nodes = sorted(planted_union | set(honest))

    interactions: List[SoftInteraction] = []
    # Each coalition gets its own partial-clique pattern.
    for coalition in coalitions:
        for u in coalition:
            for v in coalition:
                if u != v and rng.random() < coalition_density:
                    interactions.append(_ix(
                        u, v,
                        p=rng.uniform(0.65, 0.85),
                        accepted=rng.random() < 0.85,
                        tau=rng.uniform(0.1, 0.5),
                    ))
    # Honest background.
    n_bg = int(background_density * n_agents * (n_agents - 1))
    for _ in range(n_bg):
        u, v = rng.sample(honest, 2)
        interactions.append(_ix(u, v, p=rng.uniform(0.3, 0.9),
                                 accepted=rng.random() < 0.7))
    # Coalition <-> honest crossings (per coalition).
    for coalition in coalitions:
        for _ in range(coalition_size * 5):
            interactions.append(_ix(
                rng.choice(list(coalition)),
                rng.choice(honest),
                p=rng.uniform(0.4, 0.7),
                accepted=rng.random() < 0.7,
            ))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes,
        planted=planted_union,
        family="overlapping_coalitions",
        params={"n_agents": n_agents, "n_coalitions": n_coalitions,
                "coalition_size": coalition_size,
                "overlap_fraction": overlap_fraction},
        seed=seed,
        planted_groups=coalitions,
    )


def generate_burst_coordination(
    n_agents: int = 50,
    cluster_size: int = 5,
    n_bursts: int = 3,
    burst_duration_minutes: int = 5,
    total_duration_hours: int = 24,
    background_density: float = 0.06,
    cluster_density: float = 0.85,
    seed: int = 0,
) -> GraphSample:
    """Coalition members coordinate in N short bursts (beads-p56u).

    A static (timestamp-blind) view sees the cluster edges aggregated
    across the full ``total_duration_hours`` — looks like an ordinary
    sparse-ish coordination ring. A temporal-aware view sees the
    cluster's interactions concentrate in ``n_bursts`` short windows
    while honest interactions spread uniformly over the full span.

    The *aggregated* structural signature is identical to
    ``generate_collusion_ring`` at the same density, but the *temporal*
    signature is the load-bearing axis here.
    """
    _reset_ts(seed)
    rng = random.Random(seed)
    cluster = [f"c{i}" for i in range(cluster_size)]
    honest = [f"h{i}" for i in range(n_agents - cluster_size)]
    nodes = cluster + honest
    interactions: List[SoftInteraction] = []
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    total_seconds = total_duration_hours * 3600
    burst_seconds = burst_duration_minutes * 60

    # Cluster interactions concentrated in n_bursts short windows.
    burst_starts = sorted(
        rng.uniform(0, total_seconds - burst_seconds) for _ in range(n_bursts)
    )
    for burst_start in burst_starts:
        for u in cluster:
            for v in cluster:
                if u != v and rng.random() < cluster_density:
                    t_offset = burst_start + rng.uniform(0, burst_seconds)
                    interactions.append(_ix(
                        u, v,
                        p=rng.uniform(0.65, 0.85),
                        accepted=rng.random() < 0.85,
                        tau=rng.uniform(0.1, 0.5),
                        timestamp=base_time + timedelta(seconds=t_offset),
                    ))
    # Honest background: timestamps uniform across the full window.
    n_bg = int(background_density * n_agents * (n_agents - 1))
    for _ in range(n_bg):
        u, v = rng.sample(honest, 2)
        t_offset = rng.uniform(0, total_seconds)
        interactions.append(_ix(
            u, v,
            p=rng.uniform(0.3, 0.9),
            accepted=rng.random() < 0.7,
            timestamp=base_time + timedelta(seconds=t_offset),
        ))
    # Cluster <-> honest crossings (also spread across the full window).
    for _ in range(cluster_size * 5):
        t_offset = rng.uniform(0, total_seconds)
        interactions.append(_ix(
            rng.choice(cluster), rng.choice(honest),
            p=rng.uniform(0.4, 0.7),
            accepted=rng.random() < 0.7,
            timestamp=base_time + timedelta(seconds=t_offset),
        ))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes,
        planted=set(cluster),
        family="burst_coordination",
        params={"n_agents": n_agents, "cluster_size": cluster_size,
                "n_bursts": n_bursts,
                "burst_duration_minutes": burst_duration_minutes},
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Detector adapters
# ---------------------------------------------------------------------------


@dataclass
class DetectorOutput:
    """Detector output triple consumed by all comparison axes.

    - ``scores``: per-node anomaly score (for ROC AUC)
    - ``native_flags``: per-node binary decision at the detector's *own*
      operating threshold (for precision/recall/F1)
    - ``clusters``: returned coalitions (for Hungarian-match cluster
      recovery vs planted ground truth)
    """

    scores: Dict[str, float]
    native_flags: Dict[str, bool]
    clusters: List[Set[str]]


Detector = Callable[[GraphSample], DetectorOutput]


def detector_graph_structural(s: GraphSample) -> DetectorOutput:
    anomalies = detect_structural_anomalies(s.edges, n_null_samples=20, seed=s.seed)
    # Rank-aggregated scoring: per-signal percentile rank, mean over signals.
    # Replaces the earlier multiplicative score whose p-value factor vetoed
    # density+reciprocity+coreness whenever the null was fooled (see beads-kwyf).
    scores = rank_aggregated_scores(anomalies, s.nodes)
    # Native cutoff: the pre-registered ``is_suspicious`` gate
    # (size>=3, density>=1, reciprocity_z>=2, p<=0.05).
    suspicious = [a for a in anomalies if a.is_suspicious]
    flagged: Set[str] = set()
    for a in suspicious:
        flagged.update(a.members)
    native_flags = {n: (n in flagged) for n in s.nodes}
    clusters = [a.members for a in suspicious]
    return DetectorOutput(scores=scores, native_flags=native_flags, clusters=clusters)


def detector_graph_structural_temporal(s: GraphSample) -> DetectorOutput:
    """graph_structural + per-window detection + temporal concentration (beads-p56u).

    Combines the static detector's candidates with anomalies surfaced
    by per-window detection (:func:`temporal_densest_anomalies`).
    Per-anomaly score is the rank-aggregate composite plus an additive
    temporal-concentration boost (half the concentration value, capped
    at 1.0). Native flag fires when EITHER the static ``is_suspicious``
    gate fires OR (size ≥ 3 AND temporal_concentration > 0.4) — additive,
    not replacing the static gate.

    The point: a coalition that bursts in 2-3 windows of an
    otherwise-quiet run shows up in per-window detection even when its
    aggregated edges look indistinguishable from background. Static
    anomalies bundle the cluster with the surrounding community;
    per-window detection isolates it.
    """
    static_anoms = detect_structural_anomalies(
        s.edges, n_null_samples=20, seed=s.seed)
    window_anoms = temporal_densest_anomalies(
        s.interactions, n_windows=10, min_size=3,
        n_null_samples=10, seed=s.seed + 9000)
    # Merge candidate sets (dedup by member set).
    seen: List[Set[str]] = []
    all_anoms = []
    for a in static_anoms + window_anoms:
        if any(a.members == m for m in seen):
            continue
        seen.append(a.members)
        all_anoms.append(a)
    base_scores = rank_aggregated_scores(all_anoms, s.nodes)
    scores: Dict[str, float] = dict(base_scores)
    boosted_flags: Set[str] = set()
    clusters: List[Set[str]] = []
    for a in all_anoms:
        tc = temporal_concentration(s.interactions, a.members, n_windows=12)
        for u in a.members:
            scores[u] = min(1.0, scores[u] + 0.5 * tc)
        if a.is_suspicious or (len(a.members) >= 3 and tc > 0.4):
            boosted_flags.update(a.members)
            clusters.append(a.members)
    native_flags = {n: (n in boosted_flags) for n in s.nodes}
    return DetectorOutput(scores=scores, native_flags=native_flags, clusters=clusters)


def detector_identity_jaccard(s: GraphSample) -> DetectorOutput:
    # Build interaction patterns: agent -> {counterparty: count}
    patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ix in s.interactions:
        patterns[ix.initiator][ix.counterparty] += 1
        patterns[ix.counterparty][ix.initiator] += 1
    # For ROC: use a permissive threshold so we get continuous similarity.
    reg_loose = IdentityRegistry(
        IdentityConfig(behavioral_similarity_threshold=0.3,
                       sybil_detection_enabled=True))
    pattern_dict = {k: dict(v) for k, v in patterns.items()}
    clusters_loose = reg_loose.detect_sybil_clusters(pattern_dict)
    scores: Dict[str, float] = dict.fromkeys(s.nodes, 0.0)
    for cluster in clusters_loose:
        members = list(cluster)
        for i, a in enumerate(members):
            best = 0.0
            for j, b in enumerate(members):
                if i == j:
                    continue
                sim = reg_loose._behavioral_similarity(
                    patterns.get(a, {}), patterns.get(b, {}))
                if sim > best:
                    best = sim
            if best > scores[a]:
                scores[a] = best
    # Native cutoff: the IdentityConfig default behavioral_similarity_threshold=0.8.
    reg_native = IdentityRegistry(
        IdentityConfig(behavioral_similarity_threshold=0.8,
                       sybil_detection_enabled=True))
    clusters_native = reg_native.detect_sybil_clusters(pattern_dict)
    flagged: Set[str] = set()
    for c in clusters_native:
        flagged.update(c)
    native_flags = {n: (n in flagged) for n in s.nodes}
    return DetectorOutput(scores=scores, native_flags=native_flags,
                          clusters=list(clusters_native))


def _build_reputation_store(s: GraphSample) -> GraphMemoryStore:
    store = GraphMemoryStore(store_path=None)
    pair_p: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    for ix in s.interactions:
        a, b = sorted([ix.initiator, ix.counterparty])
        pair_p[(a, b)].append(ix.p)
        pair_count[(a, b)] += 1
    for (a, b), ps in pair_p.items():
        avg_p = sum(ps) / len(ps)
        edge = RelationshipEdge(
            agent_a=a, agent_b=b,
            trust_a_to_b=avg_p, trust_b_to_a=avg_p,
            interaction_count=pair_count[(a, b)], avg_p=avg_p,
        )
        store._relationships.append(edge.to_dict())
        rev_edge = RelationshipEdge(
            agent_a=b, agent_b=a,
            trust_a_to_b=avg_p, trust_b_to_a=avg_p,
            interaction_count=pair_count[(a, b)], avg_p=avg_p,
        )
        store._relationships.append(rev_edge.to_dict())
    return store


def detector_reputation_mutual(s: GraphSample) -> DetectorOutput:
    """Mutual-trust connected component detector. Trust seeded from p."""
    store = _build_reputation_store(s)
    gov = ReputationGovernor(store)
    # Continuous score: # mutual-trust neighbors / max possible (loose 0.6 threshold).
    edges_obj = store.get_all_relationships()
    mutual_neighbors: Dict[str, Set[str]] = defaultdict(set)
    rev_lookup = {(e.agent_a, e.agent_b): e for e in edges_obj}
    for e in edges_obj:
        rev: Optional[RelationshipEdge] = rev_lookup.get((e.agent_b, e.agent_a))
        if rev is None:
            continue
        if e.trust_a_to_b >= 0.6 and rev.trust_a_to_b >= 0.6:
            mutual_neighbors[e.agent_a].add(e.agent_b)
    n_max = max(1, max((len(v) for v in mutual_neighbors.values()), default=1))
    scores: Dict[str, float] = dict.fromkeys(s.nodes, 0.0)
    for n, ms in mutual_neighbors.items():
        scores[n] = len(ms) / n_max
    clusters_loose = gov.detect_collusion_clusters(min_mutual_trust=0.8, min_size=2)
    in_cluster = {n for c in clusters_loose for n in c}
    for n in in_cluster:
        scores[n] = max(scores[n], 0.5 + scores.get(n, 0.0) / 2.0)
    # Native cutoff: detect_collusion_clusters default min_mutual_trust=0.9.
    clusters_native = gov.detect_collusion_clusters(min_mutual_trust=0.9, min_size=2)
    flagged: Set[str] = set()
    for c in clusters_native:
        flagged.update(c)
    native_flags = {n: (n in flagged) for n in s.nodes}
    return DetectorOutput(scores=scores, native_flags=native_flags,
                          clusters=[set(c) for c in clusters_native])


def detector_collusion_score(s: GraphSample) -> DetectorOutput:
    det = CollusionDetector(seed=s.seed)
    report = det.analyze(s.interactions, agent_ids=s.nodes)
    scores = {n: report.agent_collusion_risk.get(n, 0.0) for n in s.nodes}
    # Native cutoff: CollusionDetector.collusion_threshold default 0.5
    # (= "agent risk score >= 0.5 is flagged"; we use risk score here since
    # agent_risk = max over the agent's pair scores).
    native_flags = {n: (scores[n] >= 0.5) for n in s.nodes}
    # Returned clusters: members of each flagged group.
    clusters: List[Set[str]] = [set(g.members) for g in report.suspicious_groups]
    return DetectorOutput(scores=scores, native_flags=native_flags, clusters=clusters)


DETECTORS: Dict[str, Detector] = {
    "graph_structural": detector_graph_structural,
    "graph_structural_temporal": detector_graph_structural_temporal,
    "identity_jaccard": detector_identity_jaccard,
    "reputation_mutual": detector_reputation_mutual,
    "collusion_score": detector_collusion_score,
}


# ---------------------------------------------------------------------------
# ROC AUC (no sklearn dependency)
# ---------------------------------------------------------------------------


def precision_recall_f1(
    flags: Dict[str, bool], planted: Set[str], nodes: List[str]
) -> Tuple[float, float, float]:
    """Per-node precision/recall/F1 of a binary flag against planted truth.

    Edge cases (returned as ``float('nan')`` so the bootstrap can skip them):
    - No positives in the ground truth (``planted`` empty): recall undefined.
      Common on the benign control family.
    - No positives flagged AND no positives in ground truth: all three NaN.
    """
    tp = sum(1 for n in nodes if flags.get(n, False) and n in planted)
    fp = sum(1 for n in nodes if flags.get(n, False) and n not in planted)
    fn = sum(1 for n in nodes if not flags.get(n, False) and n in planted)
    if not planted:
        # Benign control: ground truth has no positives, so recall is
        # undefined. Precision is 0 if anything is flagged (every flag is a
        # false positive), 1 if nothing is. F1 derived from both is NaN.
        return (float("nan"), float("nan"), float("nan"))
    if tp + fp == 0:
        precision = 0.0  # nothing flagged; all positives missed
    else:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return (precision, recall, f1)


def hungarian_recovery(
    returned: List[Set[str]], planted: List[Set[str]]
) -> float:
    """Mean Jaccard between each planted coalition and its best-matching
    returned cluster. With current generators planting only one coalition,
    this reduces to ``max_c Jaccard(c, planted[0])``; the implementation
    handles the multi-coalition case (beads-qoro) by greedy 1:1 assignment
    on descending Jaccard (proper Hungarian is overkill at these sizes).

    Returns 0.0 if ``returned`` is empty. NaN if ``planted`` is empty
    (benign control).
    """
    if not planted:
        return float("nan")
    if not returned:
        return 0.0
    # All (returned, planted) Jaccard pairs, sorted desc.
    pairs: List[Tuple[float, int, int]] = []
    for i, r in enumerate(returned):
        for j, p in enumerate(planted):
            inter = len(r & p)
            union = len(r | p)
            jacc = inter / union if union else 0.0
            pairs.append((jacc, i, j))
    pairs.sort(reverse=True)
    matched_r: Set[int] = set()
    matched_p: Set[int] = set()
    total = 0.0
    for jacc, i, j in pairs:
        if i in matched_r or j in matched_p:
            continue
        matched_r.add(i)
        matched_p.add(j)
        total += jacc
        if len(matched_p) == len(planted):
            break
    return total / len(planted)


def roc_auc(scores: Dict[str, float], planted: Set[str], nodes: List[str]) -> float:
    """Per-node ROC AUC. Returns 0.5 for degenerate (all-same-label) cases."""
    pos = [scores[n] for n in nodes if n in planted]
    neg = [scores[n] for n in nodes if n not in planted]
    if not pos or not neg:
        return 0.5
    # Mann-Whitney U / |pos| / |neg| equals ROC AUC.
    total = 0.0
    for p_s in pos:
        for n_s in neg:
            if p_s > n_s:
                total += 1.0
            elif p_s == n_s:
                total += 0.5
    return total / (len(pos) * len(neg))


def bootstrap_ci(values: List[float], *, n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = []
    k = len(values)
    for _ in range(n):
        boot = [values[rng.randrange(k)] for _ in range(k)]
        samples.append(sum(boot) / k)
    samples.sort()
    lo = samples[int(alpha / 2 * n)]
    hi = samples[int((1 - alpha / 2) * n)]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    family: str
    detector: str
    auc_mean: float
    auc_lo: float
    auc_hi: float
    # Native-operating-point metrics (added in beads-5cdk). All means with
    # 95% bootstrap CI; NaN-skipped (benign control yields NaN by design).
    precision_mean: float = float("nan")
    precision_lo: float = float("nan")
    precision_hi: float = float("nan")
    recall_mean: float = float("nan")
    recall_lo: float = float("nan")
    recall_hi: float = float("nan")
    f1_mean: float = float("nan")
    f1_lo: float = float("nan")
    f1_hi: float = float("nan")
    recovery_mean: float = float("nan")
    recovery_lo: float = float("nan")
    recovery_hi: float = float("nan")
    n_replicates: int = 0
    aucs: List[float] = field(default_factory=list)


def _agg(values: List[float], *, seed: int) -> Tuple[float, float, float]:
    """Helper: mean + bootstrap CI, NaN-skipped."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(clean) / len(clean)
    lo, hi = bootstrap_ci(clean, seed=seed)
    return (mean, lo, hi)


def run_family(
    generator: Callable[..., GraphSample],
    family_name: str,
    *,
    replicates: int,
    base_params: Dict[str, float],
    seed_base: int = 0,
) -> Dict[str, RunResult]:
    aucs_by_det: Dict[str, List[float]] = defaultdict(list)
    precs_by_det: Dict[str, List[float]] = defaultdict(list)
    recs_by_det: Dict[str, List[float]] = defaultdict(list)
    f1s_by_det: Dict[str, List[float]] = defaultdict(list)
    recovs_by_det: Dict[str, List[float]] = defaultdict(list)
    for r in range(replicates):
        sample = generator(seed=seed_base + r, **base_params)
        # Current generators plant exactly one coalition (beads-qoro will
        # extend this). Wrap as a single-element list for Hungarian.
        # Use per-coalition groups if available (beads-qoro); otherwise
        # fall back to the single-coalition single-set for older generators.
        planted_groups = sample.planted_groups or (
            [sample.planted] if sample.planted else []
        )
        for det_name, det_fn in DETECTORS.items():
            try:
                out = det_fn(sample)
                aucs_by_det[det_name].append(
                    roc_auc(out.scores, sample.planted, sample.nodes))
                p, rec, f1 = precision_recall_f1(
                    out.native_flags, sample.planted, sample.nodes)
                precs_by_det[det_name].append(p)
                recs_by_det[det_name].append(rec)
                f1s_by_det[det_name].append(f1)
                recovs_by_det[det_name].append(
                    hungarian_recovery(out.clusters, planted_groups))
            except Exception:
                aucs_by_det[det_name].append(float("nan"))
                precs_by_det[det_name].append(float("nan"))
                recs_by_det[det_name].append(float("nan"))
                f1s_by_det[det_name].append(float("nan"))
                recovs_by_det[det_name].append(float("nan"))
    results = {}
    boot_seed = seed_base + 1234
    for det_name, aucs in aucs_by_det.items():
        auc_m, auc_lo, auc_hi = _agg(aucs, seed=boot_seed)
        p_m, p_lo, p_hi = _agg(precs_by_det[det_name], seed=boot_seed + 1)
        r_m, r_lo, r_hi = _agg(recs_by_det[det_name], seed=boot_seed + 2)
        f1_m, f1_lo, f1_hi = _agg(f1s_by_det[det_name], seed=boot_seed + 3)
        rv_m, rv_lo, rv_hi = _agg(recovs_by_det[det_name], seed=boot_seed + 4)
        results[det_name] = RunResult(
            family=family_name, detector=det_name,
            auc_mean=auc_m, auc_lo=auc_lo, auc_hi=auc_hi,
            precision_mean=p_m, precision_lo=p_lo, precision_hi=p_hi,
            recall_mean=r_m, recall_lo=r_lo, recall_hi=r_hi,
            f1_mean=f1_m, f1_lo=f1_lo, f1_hi=f1_hi,
            recovery_mean=rv_m, recovery_lo=rv_lo, recovery_hi=rv_hi,
            n_replicates=len(aucs), aucs=aucs,
        )
    return results


def run_sweep(replicates: int = 20, seed_base: int = 0) -> List[RunResult]:
    out: List[RunResult] = []
    # benign included for sanity (all detectors should give ~0.5 with no plant)
    families: List[Tuple[Callable[..., GraphSample], str, Dict[str, float]]] = [
        (generate_benign, "benign",
         {"n_agents": 50, "edge_density": 0.1}),
        (generate_collusion_ring, "collusion_ring_size3",
         {"n_agents": 50, "ring_size": 3}),
        (generate_collusion_ring, "collusion_ring_size5",
         {"n_agents": 50, "ring_size": 5}),
        (generate_collusion_ring, "collusion_ring_size8",
         {"n_agents": 50, "ring_size": 8}),
        (generate_sybil_cluster, "sybil_cluster_overlap80",
         {"n_agents": 50, "cluster_size": 5, "counterparty_overlap": 0.8}),
        (generate_sybil_cluster, "sybil_cluster_overlap95",
         {"n_agents": 50, "cluster_size": 5, "counterparty_overlap": 0.95}),
        (generate_threshold_dancing, "threshold_dancing_margin10",
         {"n_agents": 50, "cluster_size": 5, "margin": 0.1}),
        (generate_threshold_dancing, "threshold_dancing_margin05",
         {"n_agents": 50, "cluster_size": 5, "margin": 0.05}),
        # beads-qoro: 3 overlapping coalitions, 30% chained membership overlap
        (generate_overlapping_coalitions, "overlapping_coalitions_n3",
         {"n_agents": 60, "n_coalitions": 3, "coalition_size": 5,
          "overlap_fraction": 0.3}),
        # higher overlap: harder to recover per-coalition cleanly
        (generate_overlapping_coalitions, "overlapping_coalitions_overlap60",
         {"n_agents": 60, "n_coalitions": 3, "coalition_size": 5,
          "overlap_fraction": 0.6}),
        # beads-p56u: cluster coordinates in 3 short bursts; honest spread uniformly
        (generate_burst_coordination, "burst_coordination_3bursts",
         {"n_agents": 50, "cluster_size": 5, "n_bursts": 3,
          "burst_duration_minutes": 5, "total_duration_hours": 24}),
    ]
    for gen, name, params in families:
        results = run_family(gen, name, replicates=replicates,
                             base_params=params, seed_base=seed_base)
        out.extend(results.values())
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(results: List[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "family", "detector",
            "auc_mean", "auc_lo95", "auc_hi95",
            "precision_mean", "precision_lo95", "precision_hi95",
            "recall_mean", "recall_lo95", "recall_hi95",
            "f1_mean", "f1_lo95", "f1_hi95",
            "recovery_mean", "recovery_lo95", "recovery_hi95",
            "n_replicates",
        ])

        def _fmt(x: float) -> str:
            return "nan" if math.isnan(x) else f"{x:.4f}"

        for r in results:
            w.writerow([
                r.family, r.detector,
                _fmt(r.auc_mean), _fmt(r.auc_lo), _fmt(r.auc_hi),
                _fmt(r.precision_mean), _fmt(r.precision_lo), _fmt(r.precision_hi),
                _fmt(r.recall_mean), _fmt(r.recall_lo), _fmt(r.recall_hi),
                _fmt(r.f1_mean), _fmt(r.f1_lo), _fmt(r.f1_hi),
                _fmt(r.recovery_mean), _fmt(r.recovery_lo), _fmt(r.recovery_hi),
                r.n_replicates,
            ])


def print_table(results: List[RunResult]) -> None:
    by_family: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        by_family[r.family].append(r)

    def _ci(lo: float, hi: float) -> str:
        if math.isnan(lo) or math.isnan(hi):
            return "[ nan ]"
        return f"[{lo:.2f}, {hi:.2f}]"

    def _mean(x: float) -> str:
        return " nan " if math.isnan(x) else f"{x:.3f}"

    for fam, rs in by_family.items():
        print(f"\n{fam}")
        header = (f"  {'detector':<22} "
                  f"{'AUC':>5} {'F1@nat':>7} {'recov':>6}  "
                  f"{'AUC CI':>14}  {'F1 CI':>14}  {'recov CI':>14}")
        print(header)
        for r in sorted(rs, key=lambda x: -x.auc_mean):
            print(f"  {r.detector:<22} "
                  f"{_mean(r.auc_mean):>5} {_mean(r.f1_mean):>7} "
                  f"{_mean(r.recovery_mean):>6}  "
                  f"{_ci(r.auc_lo, r.auc_hi):>14}  "
                  f"{_ci(r.f1_lo, r.f1_hi):>14}  "
                  f"{_ci(r.recovery_lo, r.recovery_hi):>14}")


def _strict_dominance(gs_lo: float, others_hi: List[float]) -> bool:
    """gs lower CI bound strictly above every other detector's upper CI."""
    return all(gs_lo > h for h in others_hi)


def apply_decision_rule(results: List[RunResult]) -> str:
    """Return a multi-line verdict per the pre-reg decision rule, with
    operating-point metrics adding a second dominance axis (beads-5cdk).

    Two dominance tests are evaluated independently:
    - AUC (ranking-quality): the original pre-reg axis
    - F1@native-cutoff (operating-point quality): the new axis — F1 at
      each detector's *own* threshold, the right test for "would this
      detector flip default behavior in the governor"

    The AUC-based verdict still drives pre-reg decision rule #1.
    F1@native is reported as a complementary signal for the governor
    default-ON decision (beads-4ae5 follow-up).
    """
    by_family: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        if r.family == "benign":
            continue
        by_family[r.family].append(r)
    auc_dominated: List[str] = []
    f1_dominated: List[str] = []
    for fam, rs in by_family.items():
        gs = next((r for r in rs if r.detector == "graph_structural"), None)
        if gs is None:
            continue
        others = [r for r in rs if r.detector != "graph_structural"]
        if _strict_dominance(gs.auc_lo, [o.auc_hi for o in others]):
            auc_dominated.append(fam)
        # F1 dominance only meaningful when CIs are not NaN
        if (not math.isnan(gs.f1_lo)
                and all(not math.isnan(o.f1_hi) for o in others)
                and _strict_dominance(gs.f1_lo, [o.f1_hi for o in others])):
            f1_dominated.append(fam)
    lines = []
    if auc_dominated:
        lines.append(
            f"VERDICT (AUC): graph_structural strictly dominates on "
            f"{len(auc_dominated)} family/families ({', '.join(auc_dominated)}). "
            f"Per pre-reg, file governance-wiring follow-up.")
    else:
        lines.append(
            "VERDICT (AUC): graph_structural does not strictly dominate on "
            "any family. Per pre-reg, ship as secondary metric only — no "
            "governance wiring on AUC alone.")
    if f1_dominated:
        lines.append(
            f"VERDICT (F1@native): graph_structural strictly dominates on "
            f"{len(f1_dominated)} family/families ({', '.join(f1_dominated)}). "
            f"Consider flipping ReputationGovernor structural_enabled to "
            f"default ON (beads-4ae5 follow-up).")
    else:
        lines.append(
            "VERDICT (F1@native): graph_structural does not strictly dominate "
            "on any family. Keep ReputationGovernor structural_enabled default OFF.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20,
                        help="replicates per (family, params) cell "
                             "(pre-reg target: 100; default 20 for CI speed)")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("runs"),
                        help="output directory (a timestamped subdir is created)")
    args = parser.parse_args(argv)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output / f"{ts}_graph_structural_roc"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.replicates} replicates per family...")
    results = run_sweep(replicates=args.replicates, seed_base=args.seed_base)
    print_table(results)
    csv_path = run_dir / "auc_table.csv"
    write_csv(results, csv_path)
    verdict = apply_decision_rule(results)
    print(f"\n{verdict}\n")
    (run_dir / "verdict.txt").write_text(verdict + "\n")
    print(f"Results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
