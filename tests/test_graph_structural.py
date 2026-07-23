"""Tests for graph-structural sybil/coordination detectors."""

from __future__ import annotations

import random

import pytest

from swarm.metrics.graph_structural import (
    DiGraph,
    StructuralAnomaly,
    densest_subgraph,
    density_pvalue,
    detect_structural_anomalies,
    edges_from_interactions,
    k_core_decomposition,
    label_propagation,
    rank_aggregated_scores,
    reciprocity_zscore,
)
from swarm.models.interaction import SoftInteraction


def _clique_edges(nodes, weight=1.0):
    return [(u, v, weight) for u in nodes for v in nodes if u != v]


def _ring_edges(nodes, weight=1.0):
    n = len(nodes)
    return [(nodes[i], nodes[(i + 1) % n], weight) for i in range(n)]


class TestDiGraph:
    def test_from_edges_aggregates_weights(self):
        g = DiGraph.from_edges([("a", "b", 1.0), ("a", "b", 2.5)])
        assert g.out["a"]["b"] == pytest.approx(3.5)
        assert g.nodes == {"a", "b"}

    def test_self_loops_dropped(self):
        g = DiGraph.from_edges([("a", "a", 1.0), ("a", "b", 1.0)])
        assert "a" not in g.out.get("a", {})

    def test_reciprocity(self):
        g = DiGraph.from_edges([("a", "b", 1), ("b", "a", 1), ("a", "c", 1)])
        # 2 of 3 directed edges have a reverse counterpart
        assert g.reciprocity() == pytest.approx(2 / 3)


class TestEdgesFromInteractions:
    def test_count_mode(self):
        ix = [
            SoftInteraction(initiator="a", counterparty="b", p=0.9, accepted=True),
            SoftInteraction(initiator="a", counterparty="b", p=0.8, accepted=False),
        ]
        edges = edges_from_interactions(ix, weight="count")
        assert edges == [("a", "b", 2.0)]

    def test_p_mode_sums_probabilities(self):
        ix = [
            SoftInteraction(initiator="a", counterparty="b", p=0.9),
            SoftInteraction(initiator="a", counterparty="b", p=0.3),
        ]
        edges = edges_from_interactions(ix, weight="p")
        assert edges[0][2] == pytest.approx(1.2)

    def test_mutual_benefit_only_counts_accepted(self):
        ix = [
            SoftInteraction(initiator="a", counterparty="b", accepted=True),
            SoftInteraction(initiator="a", counterparty="b", accepted=False),
        ]
        edges = edges_from_interactions(ix, weight="mutual_benefit")
        assert edges[0][2] == pytest.approx(1.0)


class TestKCore:
    def test_clique_is_its_own_core(self):
        nodes = [f"n{i}" for i in range(5)]
        g = DiGraph.from_edges(_clique_edges(nodes))
        core = k_core_decomposition(g)
        # A 5-clique on undirected projection has coreness 4 for every node.
        assert all(c == 4 for c in core.values())

    def test_leaf_has_low_core(self):
        # 4-clique plus one pendant leaf
        clique = ["a", "b", "c", "d"]
        edges = _clique_edges(clique) + [("a", "leaf", 1), ("leaf", "a", 1)]
        g = DiGraph.from_edges(edges)
        core = k_core_decomposition(g)
        assert core["leaf"] == 1
        assert all(core[n] >= 3 for n in clique)


class TestDensestSubgraph:
    def test_clique_dominates(self):
        clique = ["c1", "c2", "c3", "c4"]
        # clique embedded in sparse noise
        edges = _clique_edges(clique) + [
            ("p1", "p2", 1),
            ("p2", "p3", 1),
            ("p3", "p4", 1),
        ]
        g = DiGraph.from_edges(edges)
        best_set, _ = densest_subgraph(g)
        assert set(clique).issubset(best_set)


class TestLabelPropagation:
    def test_two_cliques_two_labels(self):
        c1 = ["a", "b", "c", "d"]
        c2 = ["w", "x", "y", "z"]
        edges = _clique_edges(c1) + _clique_edges(c2) + [("d", "w", 1)]
        g = DiGraph.from_edges(edges)
        labels = label_propagation(g, seed=0)
        # Members within each clique should share a label.
        c1_labs = {labels[n] for n in c1}
        c2_labs = {labels[n] for n in c2}
        assert len(c1_labs) == 1
        assert len(c2_labs) == 1
        assert c1_labs != c2_labs


class TestReciprocityZscore:
    def test_mutual_ring_is_anomalous(self):
        # Build a base of one-way random edges plus a fully mutual triangle.
        rng = random.Random(0)
        nodes = [f"n{i}" for i in range(40)]
        edges = []
        for _ in range(120):
            u, v = rng.sample(nodes, 2)
            edges.append((u, v, 1.0))
        triangle = ["t1", "t2", "t3"]
        edges += _clique_edges(triangle)  # fully mutual
        g = DiGraph.from_edges(edges)
        _, z = reciprocity_zscore(g, set(triangle), n_samples=20, seed=1)
        assert z > 2.0


class TestEndToEndDetector:
    def test_flags_collusion_ring(self):
        """Synthetic collusion ring: ring members are tightly mutually
        connected; rest of the network is sparse one-way."""
        rng = random.Random(0)
        ring = [f"r{i}" for i in range(5)]
        honest = [f"h{i}" for i in range(40)]
        edges = _clique_edges(ring)  # dense, mutual
        # honest agents have sparse, one-way interactions
        for _ in range(80):
            u, v = rng.sample(honest, 2)
            edges.append((u, v, 1.0))
        # a few crossings so the ring isn't a disconnected component
        for _ in range(5):
            edges.append((rng.choice(ring), rng.choice(honest), 1.0))

        anomalies = detect_structural_anomalies(
            edges, min_size=3, n_null_samples=20, seed=42
        )
        flagged = [a for a in anomalies if a.is_suspicious]
        assert flagged, "expected detector to flag collusion ring"
        # at least one anomaly should be predominantly ring members
        best = max(flagged, key=lambda a: len(a.members & set(ring)))
        assert len(best.members & set(ring)) >= 4

    def test_benign_random_graph_few_false_positives(self):
        """Erdos-Renyi-like random graph should not look like a coalition."""
        rng = random.Random(1)
        nodes = [f"n{i}" for i in range(40)]
        edges = []
        for _ in range(120):
            u, v = rng.sample(nodes, 2)
            edges.append((u, v, 1.0))
        anomalies = detect_structural_anomalies(
            edges, min_size=3, n_null_samples=20, seed=7
        )
        # A random graph may still produce candidate clusters, but none
        # should pass the combined density+reciprocity+pvalue gate.
        flagged = [a for a in anomalies if a.is_suspicious]
        assert len(flagged) == 0, f"unexpected false positives: {flagged}"


class TestDensityPvalueSubsetConditioned:
    def test_planted_clique_significant(self):
        """A planted 5-clique should get a small p-value (high density on
        SAME subset in null is rare)."""
        rng = random.Random(0)
        clique = [f"c{i}" for i in range(5)]
        background = [f"b{i}" for i in range(40)]
        edges = [(u, v, 1.0) for u in clique for v in clique if u != v]
        for _ in range(150):
            u, v = rng.sample(background, 2)
            edges.append((u, v, 1.0))
        g = DiGraph.from_edges(edges)
        pval = density_pvalue(g, set(clique), n_samples=30, seed=1)
        assert pval < 0.1

    def test_arbitrary_subset_not_significant(self):
        """A random 5-node subset of an Erdos-Renyi graph should NOT be
        significant -- guards against the old global-densest bug where
        any random subset got the same saturated p-value."""
        rng = random.Random(2)
        nodes = [f"n{i}" for i in range(45)]
        edges = []
        for _ in range(200):
            u, v = rng.sample(nodes, 2)
            edges.append((u, v, 1.0))
        g = DiGraph.from_edges(edges)
        arbitrary = set(rng.sample(nodes, 5))
        pval = density_pvalue(g, arbitrary, n_samples=30, seed=3)
        assert pval > 0.2


class TestRankAggregatedScores:
    def _anom(self, members, density=1.0, rec_z=0.0, k_core=1, pval=0.5):
        return StructuralAnomaly(
            members=set(members),
            n_internal_edges=int(density * len(members)),
            density=density,
            k_core=k_core,
            reciprocity=0.5,
            reciprocity_z=rec_z,
            pvalue=pval,
        )

    def test_empty_anomalies(self):
        scores = rank_aggregated_scores([], ["a", "b"])
        assert scores == {"a": 0.0, "b": 0.0}

    def test_no_multiplicative_veto(self):
        """The fix: a high-density, high-recip, high-core anomaly whose
        p-value is saturated (~1.0) still gets a high composite score."""
        anoms = [
            self._anom(["a", "b", "c"], density=10.0, rec_z=8.0, k_core=5,
                       pval=1.0),  # the threshold-dancer profile
            self._anom(["d", "e", "f"], density=1.0, rec_z=0.1, k_core=1,
                       pval=0.5),
            self._anom(["g", "h", "i"], density=0.5, rec_z=0.0, k_core=1,
                       pval=0.9),
        ]
        scores = rank_aggregated_scores(anoms, list("abcdefghi"))
        # the high-signal anomaly should win on 3 of 4 ranks
        assert scores["a"] > scores["d"]
        assert scores["a"] > scores["g"]
        # and should be ≥ 0.5 even with pvalue=1 (3/4 signals max-ranked)
        assert scores["a"] >= 0.5

    def test_per_node_max_over_anomalies(self):
        anoms = [
            self._anom(["a"], density=1.0, rec_z=1.0, k_core=1, pval=0.5),
            self._anom(["a", "b"], density=5.0, rec_z=5.0, k_core=3, pval=0.1),
        ]
        scores = rank_aggregated_scores(anoms, ["a", "b"])
        # a appears in both; b only in the higher-scoring one
        assert scores["a"] == scores["b"]
        assert scores["b"] > 0.5


class TestWeightedMetrics:
    """Weight-aware metrics added in beads-f970."""

    def test_induced_edge_weight_counts_sum(self):
        g = DiGraph.from_edges([
            ("a", "b", 3.0), ("b", "a", 5.0), ("a", "c", 2.0),
        ])
        assert g.induced_edge_weight({"a", "b"}) == pytest.approx(8.0)
        # c is in the subset but only has incoming a->c (weight 2.0):
        # induced_edge_weight counts directed edges with BOTH endpoints
        # in subset, so a->c at weight 2 + a->b at weight 3 + b->a at 5 = 10
        assert g.induced_edge_weight({"a", "b", "c"}) == pytest.approx(10.0)

    def test_weighted_reciprocity_balanced_mutual(self):
        # Equal weight in both directions -> reciprocity 1.0
        g = DiGraph.from_edges([("a", "b", 5.0), ("b", "a", 5.0)])
        assert g.weighted_reciprocity() == pytest.approx(1.0)

    def test_weighted_reciprocity_imbalanced(self):
        # Asymmetric: min/max = 1/5
        g = DiGraph.from_edges([("a", "b", 1.0), ("b", "a", 5.0)])
        assert g.weighted_reciprocity() == pytest.approx(0.2)

    def test_weighted_reciprocity_one_way_is_zero(self):
        g = DiGraph.from_edges([("a", "b", 5.0)])  # no b->a
        assert g.weighted_reciprocity() == 0.0

    def test_weighted_reciprocity_empty(self):
        g = DiGraph()
        assert g.weighted_reciprocity() == 0.0

    def test_anomaly_carries_weighted_fields(self):
        """detect_structural_anomalies populates the new fields."""
        clique = [f"c{i}" for i in range(5)]
        edges = [(u, v, 3.0) for u in clique for v in clique if u != v]
        anomalies = detect_structural_anomalies(edges, n_null_samples=10, seed=0)
        # Full directed clique: each ordered pair has weight 3.0; 5*4 = 20 directed
        # edges so total weight = 60. Weighted reciprocity should be 1.0.
        clique_anom = next(a for a in anomalies if a.members == set(clique))
        assert clique_anom.total_internal_weight == pytest.approx(60.0)
        assert clique_anom.weighted_reciprocity == pytest.approx(1.0)

    def test_weighted_signals_intentionally_not_in_composite(self):
        """weighted_reciprocity is exposed as data but is intentionally
        NOT folded into the rank composite (see beads-f970 rationale in
        rank_aggregated_scores). Two anomalies identical on the 4
        composite signals but differing only on weighted_reciprocity
        should tie, not separate — naive inclusion regressed sybil-family
        AUC because sybils are designed to be low-mutuality."""
        from swarm.metrics.graph_structural import StructuralAnomaly
        balanced = StructuralAnomaly(
            members={"a", "b", "c"}, n_internal_edges=6, density=2.0,
            k_core=2, reciprocity=1.0, reciprocity_z=2.0, pvalue=0.01,
            total_internal_weight=12.0, weighted_reciprocity=1.0,
        )
        imbalanced = StructuralAnomaly(
            members={"x", "y", "z"}, n_internal_edges=6, density=2.0,
            k_core=2, reciprocity=1.0, reciprocity_z=2.0, pvalue=0.01,
            total_internal_weight=12.0, weighted_reciprocity=0.1,
        )
        scores = rank_aggregated_scores(
            [balanced, imbalanced], ["a", "b", "c", "x", "y", "z"]
        )
        assert scores["a"] == scores["x"]
