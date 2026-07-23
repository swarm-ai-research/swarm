"""Graph-structural sybil/coordination detectors.

Existing detectors in swarm/ (identity.detect_sybil_clusters,
reputation_governor.detect_collusion_clusters, collusion.CollusionDetector)
score pairs/groups against a threshold. This module flags coalitions by
*topological* structure relative to a degree-preserving null model:

    * Densest subgraph extraction (Charikar 1/2-approx peeling).
    * k-core decomposition.
    * Reciprocity z-score vs configuration model.
    * Label-propagation community detection.
    * Configuration-model null sampler -> p-values, not magic thresholds.

Zero external dependency (matches swarm/analysis/network.py convention).
Inputs are a list of weighted directed edges, easily produced from
``SoftInteraction`` via ``edges_from_interactions``. Weights are
exposed via two ``DiGraph`` methods — ``induced_edge_weight`` (sum of
in-cluster weight) and ``weighted_reciprocity`` (min/max strength of
mutuality per unordered pair) — and surface as
``StructuralAnomaly.total_internal_weight`` and ``weighted_reciprocity``
(beads-f970). They are **not** included in ``rank_aggregated_scores``;
see that function's docstring for the empirical rationale. Downstream
consumers (governance, custom scoring, inspection) can read them
directly.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from swarm.models.interaction import SoftInteraction

Edge = Tuple[str, str, float]


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------


@dataclass
class DiGraph:
    """Minimal directed multigraph keyed by string node ids.

    Stores aggregated edge weight w(u, v) = sum of weights of all u->v edges.
    """

    nodes: Set[str] = field(default_factory=set)
    out: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    in_: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )

    @classmethod
    def from_edges(cls, edges: Iterable[Edge]) -> "DiGraph":
        g = cls()
        for u, v, w in edges:
            if u == v or w <= 0.0:
                # ignore self-loops and non-positive-weight edges (a p=0.0
                # SoftInteraction in p-weighted mode must not contribute
                # topology, since downstream metrics use edge presence)
                continue
            g.nodes.add(u)
            g.nodes.add(v)
            g.out[u][v] += w
            g.in_[v][u] += w
        return g

    def undirected_neighbors(self, u: str) -> Set[str]:
        return set(self.out.get(u, {})) | set(self.in_.get(u, {}))

    def undirected_degree(self, u: str) -> int:
        return len(self.undirected_neighbors(u))

    def induced_edge_count(self, subset: Set[str]) -> int:
        """Count directed edges with both endpoints in ``subset``."""
        n = 0
        for u in subset:
            for v in self.out.get(u, {}):
                if v in subset:
                    n += 1
        return n

    def induced_edge_weight(self, subset: Set[str]) -> float:
        """Sum of edge weights for directed edges with both endpoints in
        ``subset`` (beads-f970). Counterpart to :meth:`induced_edge_count`
        that propagates the weighting mode chosen in
        :func:`edges_from_interactions`:

        - ``"count"``: total *interactions* inside the subset (each
          aggregated (u, v) edge weight = number of interactions on that
          pair), so this can exceed the unique-edge count when pairs
          repeat — it is NOT just ``induced_edge_count`` as a float.
        - ``"p"``: sum of interaction probabilities (a tight clique of
          low-p mutual interactions scores low here even though it
          scores high on count-based density).
        - ``"mutual_benefit"``: count of mutually-accepted interactions.
        """
        w = 0.0
        for u in subset:
            for v, weight in self.out.get(u, {}).items():
                if v in subset:
                    w += weight
        return w

    def reciprocity(self, subset: Optional[Set[str]] = None) -> float:
        """Fraction of directed edges (u, v) for which (v, u) also exists."""
        nodes = self.nodes if subset is None else subset
        total = 0
        reciprocated = 0
        for u in nodes:
            for v in self.out.get(u, {}):
                if subset is not None and v not in subset:
                    continue
                total += 1
                if u in self.out.get(v, {}):
                    reciprocated += 1
        return reciprocated / total if total else 0.0

    def weighted_reciprocity(self, subset: Optional[Set[str]] = None) -> float:
        """Weighted mutuality, computed once per unordered pair {u, v}:
        sum of min(w(u,v), w(v,u)) over sum of max(w(u,v), w(v,u))
        (beads-f970).

        Captures *strength* of mutuality, not just presence: two agents
        exchanging 10 interactions in each direction score higher than
        two agents with 1 forward + 10 reverse. Each unordered pair is
        counted once (not double-counted across directions). Returns
        0.0 if no edges. Counterpart to :meth:`reciprocity`
        (presence-based).
        """
        nodes = self.nodes if subset is None else subset
        sum_min = 0.0
        sum_max = 0.0
        seen: Set[Tuple[str, str]] = set()
        for u in nodes:
            for v, w_uv in self.out.get(u, {}).items():
                if subset is not None and v not in subset:
                    continue
                key = (u, v) if u < v else (v, u)
                if key in seen:
                    continue
                seen.add(key)
                w_vu = self.out.get(v, {}).get(u, 0.0)
                sum_min += min(w_uv, w_vu)
                sum_max += max(w_uv, w_vu)
        return sum_min / sum_max if sum_max > 0 else 0.0


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------


def edges_from_interactions(
    interactions: Sequence[SoftInteraction],
    *,
    weight: str = "count",
) -> List[Edge]:
    """Aggregate ``SoftInteraction`` records into weighted directed edges.

    Weights surface as ``DiGraph.induced_edge_weight`` and
    ``DiGraph.weighted_reciprocity`` (and on ``StructuralAnomaly`` as
    ``total_internal_weight`` / ``weighted_reciprocity``) for
    inspection or custom scoring. They are NOT folded into
    ``rank_aggregated_scores`` — see that function's docstring for the
    empirical rationale (beads-f970).

    Mode choice still acts as a positivity filter in
    ``DiGraph.from_edges`` (a ``p=0`` interaction in ``"p"`` mode drops
    out — see beads-kwyf for why), so the inverse blind spot (a mutual
    clique of accepted low-quality interactions in ``"p"`` mode) is
    still a follow-up worth filing if it bites.

    Args:
        interactions: list of SoftInteraction records.
        weight: one of ``"count"`` (1 per interaction), ``"p"`` (sum of p), or
            ``"mutual_benefit"`` (sum of accepted=True interactions only).
    """
    agg: Dict[Tuple[str, str], float] = defaultdict(float)
    for x in interactions:
        if not x.initiator or not x.counterparty:
            continue
        key = (x.initiator, x.counterparty)
        if weight == "count":
            agg[key] += 1.0
        elif weight == "p":
            agg[key] += x.p
        elif weight == "mutual_benefit":
            if x.accepted:
                agg[key] += 1.0
        else:
            raise ValueError(f"unknown weight mode: {weight!r}")
    return [(u, v, w) for (u, v), w in agg.items()]


# ---------------------------------------------------------------------------
# k-core
# ---------------------------------------------------------------------------


def k_core_decomposition(g: DiGraph) -> Dict[str, int]:
    """Coreness on the undirected projection (Batagelj-Zaversnik style peel).

    Not the strict O(m + n) bucket-array variant — degree updates here use a
    bounded backward scan rather than constant-time bucket pointers, so
    worst-case is super-linear. Fine for the graph sizes this detector runs on.
    """
    deg = {u: g.undirected_degree(u) for u in g.nodes}
    neigh = {u: g.undirected_neighbors(u) for u in g.nodes}
    # Tie-break by node id for determinism.
    order: List[str] = sorted(g.nodes, key=lambda u: (deg[u], u))
    pos = {u: i for i, u in enumerate(order)}
    core: Dict[str, int] = {}
    for u in order:
        core[u] = deg[u]
        for v in neigh[u]:
            if v in core:
                continue  # already peeled
            if deg[v] > deg[u]:
                dv = deg[v]
                # swap v with the first node of degree dv to keep order sorted
                first_idx = pos[v]
                # find the leftmost index with degree dv
                while first_idx > 0 and deg[order[first_idx - 1]] == dv:
                    first_idx -= 1
                w = order[first_idx]
                if w != v:
                    order[first_idx], order[pos[v]] = v, w
                    pos[v], pos[w] = first_idx, pos[v]
                deg[v] -= 1
    return core


# ---------------------------------------------------------------------------
# Densest subgraph (Charikar 1/2-approx)
# ---------------------------------------------------------------------------


def densest_subgraph(g: DiGraph) -> Tuple[Set[str], float]:
    """Charikar peeling: repeatedly remove the min-degree node, return the
    snapshot maximizing |E(S)| / |S| (directed edges, undirected support).
    """
    deg = {u: g.undirected_degree(u) for u in g.nodes}
    neigh = {u: set(g.undirected_neighbors(u)) for u in g.nodes}
    remaining = set(g.nodes)
    edge_count = g.induced_edge_count(remaining)

    best_set = set(remaining)
    best_density = edge_count / len(remaining) if remaining else 0.0

    while len(remaining) > 1:
        # pick min-degree node (tie-break by id for determinism)
        u = min(remaining, key=lambda x: (deg[x], x))
        # count directed edges incident to u within `remaining`
        removed = 0
        for v in g.out.get(u, {}):
            if v in remaining and v != u:
                removed += 1
        for v in g.in_.get(u, {}):
            if v in remaining and v != u:
                removed += 1
        edge_count -= removed
        remaining.discard(u)
        for v in neigh[u]:
            if v in remaining:
                deg[v] -= 1
        if remaining:
            density = edge_count / len(remaining)
            if density > best_density:
                best_density = density
                best_set = set(remaining)
    return best_set, best_density


# ---------------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------------


def label_propagation(
    g: DiGraph, *, max_iter: int = 30, seed: int = 0
) -> Dict[str, str]:
    """Asynchronous label propagation on the undirected projection.

    Returns ``node -> community_label``. Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    labels: Dict[str, str] = {u: u for u in g.nodes}
    nodes = sorted(g.nodes)  # deterministic order; rng.shuffle reseeds each iter
    for _ in range(max_iter):
        rng.shuffle(nodes)
        changed = False
        for u in nodes:
            neigh = g.undirected_neighbors(u)
            if not neigh:
                continue
            counts: Dict[str, int] = defaultdict(int)
            for v in neigh:
                counts[labels[v]] += 1
            top = max(counts.values())
            candidates = sorted(lab for lab, c in counts.items() if c == top)
            new = candidates[0]  # deterministic tie-break
            if labels[u] != new:
                labels[u] = new
                changed = True
        if not changed:
            break
    return labels


# ---------------------------------------------------------------------------
# Configuration-model null
# ---------------------------------------------------------------------------


def configuration_model_null(g: DiGraph, *, seed: int = 0) -> DiGraph:
    """Generate a directed graph with the same in/out degree sequence.

    Stubs are randomly matched; self-loops and multi-edges are merged into
    weighted edges (matching DiGraph aggregation semantics).
    """
    rng = random.Random(seed)
    out_stubs: List[str] = []
    in_stubs: List[str] = []
    # Iterate in sorted order so the (seed, graph) pair is reproducible
    # across PYTHONHASHSEED values; raw set iteration is not.
    for u in sorted(g.nodes):
        out_stubs.extend([u] * sum(1 for _ in g.out.get(u, {})))
        in_stubs.extend([u] * sum(1 for _ in g.in_.get(u, {})))
    rng.shuffle(out_stubs)
    rng.shuffle(in_stubs)
    edges: List[Edge] = []
    for u, v in zip(out_stubs, in_stubs, strict=False):
        if u != v:
            edges.append((u, v, 1.0))
    return DiGraph.from_edges(edges)


def reciprocity_zscore(
    g: DiGraph,
    subset: Optional[Set[str]] = None,
    *,
    n_samples: int = 50,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return (observed_reciprocity, z_score) for ``subset`` vs null model."""
    observed = g.reciprocity(subset)
    samples = []
    for i in range(n_samples):
        null = configuration_model_null(g, seed=seed + i)
        samples.append(null.reciprocity(subset if subset is None else subset & null.nodes))
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    std = var ** 0.5
    # Floor std so a tight null (all zeros) against a positive observed
    # still scores as anomalous rather than collapsing to 0.
    z = (observed - mean) / max(std, 0.05)
    return observed, z


def density_pvalue(
    g: DiGraph,
    subset: Set[str],
    *,
    n_samples: int = 50,
    seed: int = 0,
) -> float:
    """Empirical p-value that the *same nodes* in a degree-matched null
    graph have internal density >= the observed density of ``subset``.

    This is the right test for "is THIS coalition denser than chance".
    The earlier implementation took the null's *globally densest*
    subgraph (any size) and compared against that — a different
    hypothesis ("does the null contain any dense subgraph anywhere?"),
    which saturates at p≈1 whenever the null happens to produce any
    small dense pocket. That bug killed AUC on the sk95 ROC benchmark's
    threshold_dancing family (see beads-kwyf).

    The configuration model preserves degree sequence, so a candidate
    subset whose nodes have high observed degree will *also* have many
    stubs in the null — but those stubs distribute randomly across all
    nodes, so the expected number landing back inside the subset is
    governed by the subset's degree-share of the total, not by the
    subset's own internal-edge count. Real coalitions whose internal
    edges far exceed degree-share expectation light up.
    """
    if not subset:
        return 1.0
    observed_edges = g.induced_edge_count(subset)
    observed_density = observed_edges / len(subset)
    hits = 0
    for i in range(n_samples):
        null = configuration_model_null(g, seed=seed + i)
        # density on the SAME nodes in the null (correct subset-conditioned test)
        live = subset & null.nodes
        null_edges = null.induced_edge_count(live)
        null_density = null_edges / max(1, len(live))
        if null_density >= observed_density:
            hits += 1
    return (hits + 1) / (n_samples + 1)


# ---------------------------------------------------------------------------
# Top-level detector
# ---------------------------------------------------------------------------


@dataclass
class StructuralAnomaly:
    """Suspicious subgraph flagged by structural signals."""

    members: Set[str]
    n_internal_edges: int
    density: float  # edges per node (directed, presence-based)
    k_core: int  # min coreness within members
    reciprocity: float
    reciprocity_z: float
    modularity_label: Optional[str] = None
    pvalue: float = 1.0
    # Weighted-graph signals added in beads-f970. ``total_internal_weight``
    # is the sum of edge weights inside ``members`` (in the weighting mode
    # the caller chose for :func:`edges_from_interactions`); when all
    # weights are 1.0 (the ``"count"`` default) this just equals
    # ``n_internal_edges``. ``weighted_reciprocity`` measures *strength* of
    # mutuality (min/max ratio per pair), not just presence.
    total_internal_weight: float = 0.0
    weighted_reciprocity: float = 0.0

    @property
    def edge_probability(self) -> float:
        """Fraction of possible directed edges present within ``members``.

        Scales density by max-possible (n*(n-1)), so a 5-node clique and a
        43-node graph at 12% edge probability are correctly ordered (clique
        first). Using raw ``density = edges/nodes`` for ranking favors large
        dense communities over tight small cliques — exactly the failure
        mode that gave threshold_dancing AUC < 0.5 before beads-kwyf.
        """
        n = len(self.members)
        if n < 2:
            return 0.0
        return self.n_internal_edges / (n * (n - 1))

    @property
    def is_suspicious(self) -> bool:
        # Pre-registered combination: dense AND reciprocal AND rare under null.
        return (
            len(self.members) >= 3
            and self.density >= 1.0
            and self.reciprocity_z >= 2.0
            and self.pvalue <= 0.05
        )


def detect_structural_anomalies(
    edges: Sequence[Edge],
    *,
    min_size: int = 3,
    n_null_samples: int = 50,
    seed: int = 0,
) -> List[StructuralAnomaly]:
    """End-to-end detector: build graph, run all four signals, return
    one anomaly record per candidate cluster.

    Candidates come from (a) Charikar densest subgraph and (b) each label-
    propagation community of size >= ``min_size``. Duplicates are merged.
    """
    g = DiGraph.from_edges(edges)
    if len(g.nodes) < min_size:
        return []

    coreness = k_core_decomposition(g)
    communities: Dict[str, Set[str]] = defaultdict(set)
    for node, node_label in label_propagation(g, seed=seed).items():
        communities[node_label].add(node)

    candidates: List[Set[str]] = []
    dense_set, _ = densest_subgraph(g)
    if len(dense_set) >= min_size:
        candidates.append(dense_set)
    for _lab, members in communities.items():
        if len(members) >= min_size and not any(members == c for c in candidates):
            candidates.append(members)

    results: List[StructuralAnomaly] = []
    for cand in candidates:
        n_edges = g.induced_edge_count(cand)
        density = n_edges / len(cand)
        rec, z = reciprocity_zscore(g, cand, n_samples=n_null_samples, seed=seed)
        pval = density_pvalue(g, cand, n_samples=n_null_samples, seed=seed)
        # find community label, if any
        lab: Optional[str] = next(
            (cl for cl, m in communities.items() if m == cand), None
        )
        results.append(
            StructuralAnomaly(
                members=cand,
                n_internal_edges=n_edges,
                density=density,
                k_core=min(coreness[u] for u in cand),
                reciprocity=rec,
                reciprocity_z=z,
                modularity_label=lab,
                pvalue=pval,
                total_internal_weight=g.induced_edge_weight(cand),
                weighted_reciprocity=g.weighted_reciprocity(cand),
            )
        )
    return results


def rank_aggregated_scores(
    anomalies: Sequence[StructuralAnomaly],
    nodes: Sequence[str],
) -> Dict[str, float]:
    """Per-node anomaly score in [0, 1] aggregated by signal rank.

    Each anomaly is ranked across all candidates on four independent
    signals: density, reciprocity_z, k_core, and -log10(pvalue+eps).
    Its composite score is the mean percentile rank. Per-node score is
    the max composite over anomalies containing that node.

    Rationale (vs. the earlier multiplicative product): no single signal
    can veto the others. A fooled p-value (the failure mode that killed
    AUC on threshold_dancing in the sk95 benchmark) drops the pvalue
    rank to ~0 but leaves density / reciprocity / coreness intact. See
    beads-kwyf for the design discussion.
    """
    score_by_node = dict.fromkeys(nodes, 0.0)
    if not anomalies:
        return score_by_node

    n = len(anomalies)
    if n == 1:
        # Single candidate: percentile rank is degenerate; fall back to
        # a presence indicator so the lone anomaly still scores something.
        a = anomalies[0]
        composite = 1.0 if a.density > 0 else 0.0
        for u in a.members:
            score_by_node[u] = max(score_by_node[u], composite)
        return score_by_node

    def _ranks(values: List[float]) -> List[float]:
        """Fractional ranks in [0, 1]; ties get average rank."""
        order = sorted(range(n), key=lambda i: values[i])
        ranks_out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2  # 0-indexed average
            for k in range(i, j + 1):
                ranks_out[order[k]] = avg_rank / max(1, n - 1)
            i = j + 1
        return ranks_out

    eps = 1e-9
    # Use edge_probability (in [0, 1]) rather than raw density: a small
    # clique correctly outranks a large-but-sparse community.
    density_r = _ranks([a.edge_probability for a in anomalies])
    rec_r = _ranks([max(0.0, a.reciprocity_z) for a in anomalies])
    # k-core normalized by anomaly size — coreness of 4 in a 5-clique is
    # saturated; coreness of 4 in a 50-node graph is weak.
    core_r = _ranks([
        a.k_core / max(1, len(a.members) - 1) for a in anomalies
    ])
    sig_r = _ranks([-math.log10(a.pvalue + eps) for a in anomalies])

    # Why weighted_reciprocity is NOT in this composite:
    # beads-f970 added ``total_internal_weight`` and
    # ``weighted_reciprocity`` to the dataclass and exposes them via
    # ``DiGraph.induced_edge_weight`` and
    # ``DiGraph.weighted_reciprocity`` — they are real, inspectable
    # signals. Naively averaging weighted_reciprocity into the rank
    # composite regressed sybil-family AUC (0.754 -> 0.641 on the sk95
    # overlap-0.95 sybil family) because sybils are *designed* to be
    # low-mutuality and the new signal correctly scored them low — but
    # at the cost of pulling their composite below honest random
    # communities that happened to have small random mutuality.
    # Until per-family signal routing is in place, weights remain
    # available to downstream consumers (governance / inspection /
    # custom scoring) but are not auto-folded into the rank composite.

    for idx, a in enumerate(anomalies):
        composite = (density_r[idx] + rec_r[idx] + core_r[idx] + sig_r[idx]) / 4.0
        for u in a.members:
            if composite > score_by_node[u]:
                score_by_node[u] = composite
    return score_by_node


# ---------------------------------------------------------------------------
# Temporal coordination signal (beads-p56u)
# ---------------------------------------------------------------------------


def temporal_concentration(
    interactions: Sequence[SoftInteraction],
    members: Set[str],
    *,
    n_windows: int = 10,
) -> float:
    """Normalized concentration of within-``members`` interactions over time.

    Coordination bursts (cluster members coordinate during short windows
    and stay quiet otherwise) show high concentration; honest agents who
    interact uniformly over time show low concentration.

    Implementation: bin the timestamps of interactions where both
    ``initiator`` and ``counterparty`` are in ``members`` into
    ``n_windows`` equal-width windows spanning the *full* observed
    time range (NOT just the cluster's range, so a cluster's silence
    relative to background activity scores high). Returns
    ``1 - H(p) / log(n_windows)`` ∈ [0, 1] where ``H`` is Shannon
    entropy and ``p`` is the per-window count distribution. Uniform
    spread → 0, single-window concentration → 1.

    Returns 0.0 if there are no in-cluster interactions or only one
    interaction in the whole graph (no time axis to concentrate on).

    The "honest baseline" subtraction is the caller's job — this
    function returns the raw concentration; the benchmark adapter
    pairs it with a z-score against the rest-of-graph baseline.
    """
    if not interactions or n_windows < 2:
        return 0.0
    # Use the full graph's time range so a cluster confined to a
    # narrow window scores high even if its absolute timespan is
    # short.
    times = [ix.timestamp for ix in interactions]
    t_min = min(times)
    t_max = max(times)
    span = (t_max - t_min).total_seconds()
    if span <= 0:
        return 0.0  # everything happened at the same instant
    cluster_counts = [0] * n_windows
    n_in_cluster = 0
    for ix in interactions:
        if ix.initiator not in members or ix.counterparty not in members:
            continue
        n_in_cluster += 1
        offset = (ix.timestamp - t_min).total_seconds()
        bin_idx = min(n_windows - 1, int(n_windows * offset / span))
        cluster_counts[bin_idx] += 1
    if n_in_cluster < 2:
        return 0.0
    # Shannon entropy of the per-window probability distribution.
    entropy = 0.0
    for c in cluster_counts:
        if c == 0:
            continue
        p = c / n_in_cluster
        entropy -= p * math.log(p)
    max_entropy = math.log(n_windows)
    return 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0


def temporal_densest_anomalies(
    interactions: Sequence[SoftInteraction],
    *,
    n_windows: int = 10,
    min_size: int = 3,
    n_null_samples: int = 20,
    seed: int = 0,
) -> List[StructuralAnomaly]:
    """Per-window structural anomaly detection (beads-p56u).

    Splits ``interactions`` into ``n_windows`` equal-width time bins
    and runs :func:`detect_structural_anomalies` on each window's
    induced subgraph. Returns the union of resulting anomalies,
    de-duplicated by member set (different windows surfacing the
    same subset are kept once).

    The point: a coalition that bursts in 2-3 windows of an
    otherwise-quiet 10-window run produces small, dense anomalies in
    those windows even when its aggregated edges look like just
    another sparse part of the full graph. The static detector sees
    aggregated edges and bundles the coalition with the surrounding
    background community; per-window detection isolates the burst.
    """
    if not interactions or n_windows < 1:
        return []
    times = [ix.timestamp for ix in interactions]
    t_min = min(times)
    t_max = max(times)
    span = (t_max - t_min).total_seconds()
    if span <= 0:
        return detect_structural_anomalies(
            edges_from_interactions(interactions),
            min_size=min_size, n_null_samples=n_null_samples, seed=seed,
        )
    window_buckets: List[List[SoftInteraction]] = [[] for _ in range(n_windows)]
    for ix in interactions:
        offset = (ix.timestamp - t_min).total_seconds()
        bin_idx = min(n_windows - 1, int(n_windows * offset / span))
        window_buckets[bin_idx].append(ix)
    seen: List[Set[str]] = []
    merged: List[StructuralAnomaly] = []
    for w_idx, window_ix in enumerate(window_buckets):
        if not window_ix:
            continue
        window_edges = edges_from_interactions(window_ix)
        # Per-window detection uses a window-local seed so the
        # configuration-model null is reproducible across windows.
        window_anoms = detect_structural_anomalies(
            window_edges,
            min_size=min_size,
            n_null_samples=n_null_samples,
            seed=seed + w_idx,
        )
        for a in window_anoms:
            if any(a.members == s for s in seen):
                continue
            seen.append(a.members)
            merged.append(a)
    return merged
