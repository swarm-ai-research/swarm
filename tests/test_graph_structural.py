"""Tests for graph-structural sybil/coordination detectors."""

from __future__ import annotations

import random

import pytest

from swarm.metrics.graph_structural import (
    DiGraph,
    StructuralAnomaly,
    bipartite_null,
    densest_subgraph,
    density_pvalue,
    detect_structural_anomalies,
    edges_from_interactions,
    k_core_decomposition,
    label_propagation,
    project_incidence,
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
    def test_reciprocity_treats_missing_nodes_as_isolates(self):
        """A node omitted from the graph (isolate) must not change
        reciprocity of the remaining edges — same node-set comparison."""
        g = DiGraph.from_edges([("a", "b", 1.0), ("b", "a", 1.0)])
        assert g.reciprocity({"a", "b", "c"}) == pytest.approx(g.reciprocity({"a", "b"}))
        assert g.induced_edge_count({"a", "b", "c"}) == g.induced_edge_count({"a", "b"})

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

    def test_null_isolates_do_not_shrink_subset(self, monkeypatch):
        """DiGraph omits isolates. Intersecting the candidate with
        null_g.nodes inflates null density (smaller denominator) and
        can turn a miss into a hit. Density must stay over len(subset)."""
        subset = {"a", "b", "c", "d"}
        # observed_edges=4, density=1.0. Isolate-null has a<->b only and
        # omits c, d. Old live={a,b}: 2/2=1.0 >= 1.0 → hit every sample.
        # New: 2/4=0.5 < 1.0 → miss every sample → p = 1/(n+1).
        g = DiGraph.from_edges([
            ("a", "b", 1.0), ("b", "a", 1.0),
            ("c", "d", 1.0), ("d", "c", 1.0),
        ])
        isolate_null = DiGraph.from_edges([("a", "b", 1.0), ("b", "a", 1.0)])
        assert isolate_null.nodes == {"a", "b"}

        def fake_null(*_args, **_kwargs):
            return isolate_null

        monkeypatch.setattr("swarm.metrics.graph_structural._null_graph", fake_null)
        pval = density_pvalue(g, subset, n_samples=20, seed=0)
        assert pval == pytest.approx(1 / 21)


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


# ---------------------------------------------------------------------------
# Hub-aware bipartite nulls (beads-y2t2)
# ---------------------------------------------------------------------------


def _hub_incidence(n_agents=40, n_edits=400, seed=7):
    rng = random.Random(seed)
    agents = [f"h{i}" for i in range(n_agents)]
    return [(rng.choice(agents), "welcome") for _ in range(n_edits)]


class TestProjectIncidence:
    def test_sequential_is_reply_to_previous_distinct_agent(self):
        inc = [("a", "p"), ("b", "p"), ("b", "p"), ("a", "p"), ("c", "q")]
        # b replies to a; b's self follow-up is dropped; a replies to b;
        # c creates q and has no counterparty.
        assert sorted(project_incidence(inc)) == [("a", "b", 1.0), ("b", "a", 1.0)]

    def test_sequential_reproduces_projected_graph(self):
        inc = _hub_incidence(n_agents=6, n_edits=40)
        g = DiGraph.from_edges(project_incidence(inc))
        # re-projecting the same incidence is idempotent on the graph
        g2 = DiGraph.from_edges(project_incidence(inc))
        assert g.out == g2.out

    def test_co_membership_is_symmetric_clique_per_object(self):
        inc = [("a", "p"), ("b", "p"), ("c", "p"), ("a", "q"), ("b", "q")]
        edges = {(u, v): w for u, v, w in project_incidence(inc, projection="co_membership")}
        assert edges[("a", "b")] == 2.0 and edges[("b", "a")] == 2.0
        assert edges[("a", "c")] == 1.0 and ("c", "b") in edges

    def test_unknown_projection(self):
        with pytest.raises(ValueError):
            project_incidence([("a", "p")], projection="nope")


class TestBipartiteNull:
    def test_preserves_agent_and_object_counts(self):
        inc = _hub_incidence(n_agents=5, n_edits=30) + [("x", "side")] * 3
        # Count what the null projects from, not the projected edges: the
        # sampler is deterministic per seed, so rebuild its shuffled incidence.
        rng = random.Random(3)
        agents = [a for a, _ in inc]
        rng.shuffle(agents)
        from collections import Counter
        assert Counter(agents) == Counter(a for a, _ in inc)
        null = bipartite_null(inc, seed=3)
        assert null.nodes <= {a for a, _ in inc}

    def test_membership_variant_keeps_per_object_counts(self):
        from collections import Counter
        inc = _hub_incidence(n_agents=5, n_edits=30) + [("x", "side"), ("y", "side"), ("x", "side")]
        # Under preserve_membership the per-object multiset of agents is
        # fixed, so "side" can only ever project x<->y edges and the hub
        # never gains x or y.
        for seed in range(5):
            null = bipartite_null(inc, seed=seed, preserve_membership=True)
            assert not (null.undirected_neighbors("x") - {"y"})
        assert Counter(a for a, o in inc if o == "side") == Counter({"x": 2, "y": 1})

    def test_requires_incidence(self):
        g_edges = project_incidence(_hub_incidence(n_agents=6, n_edits=30))
        with pytest.raises(ValueError):
            detect_structural_anomalies(g_edges, null="bipartite")
        with pytest.raises(ValueError):
            density_pvalue(DiGraph.from_edges(g_edges), {"h0", "h1"}, null="membership")

    def test_unknown_null(self):
        g_edges = project_incidence(_hub_incidence(n_agents=6, n_edits=30))
        with pytest.raises(ValueError):
            detect_structural_anomalies(g_edges, null="lattice")

    def test_default_null_unchanged(self):
        inc = _hub_incidence()
        edges = project_incidence(inc)
        a = detect_structural_anomalies(edges, n_null_samples=20, seed=0)
        b = detect_structural_anomalies(edges, n_null_samples=20, seed=0, null="configuration")
        assert [(x.members, x.pvalue, x.reciprocity_z) for x in a] == [
            (x.members, x.pvalue, x.reciprocity_z) for x in b
        ]

    def test_hub_page_saturates_configuration_but_not_bipartite(self):
        """The acceptance case: a hot hub page makes every co-editor a
        reply neighbour. The configuration model cannot reproduce that and
        pins every hub community at the p-value floor; both bipartite nulls
        reproduce the board and rank nothing on it."""
        inc = _hub_incidence()
        edges = project_incidence(inc)
        conf = detect_structural_anomalies(edges, n_null_samples=50, seed=0)
        assert conf and all(a.pvalue <= 0.05 for a in conf)
        for null in ("bipartite", "membership"):
            hub_aware = detect_structural_anomalies(
                edges, n_null_samples=50, seed=0, null=null, incidence=inc
            )
            assert [a.members for a in hub_aware] == [a.members for a in conf]
            assert all(a.pvalue > 0.3 for a in hub_aware), null

    def test_relay_on_the_hub_survives_membership_null(self):
        """Coordination beyond membership: four agents who only ever post
        in consecutive runs on the busy page. Their internal edges are far
        denser than a random ordering of the same per-page edit counts
        predicts, while the hub community around them is not."""
        rng = random.Random(11)
        inc = _hub_incidence()
        block = [(c, "welcome") for c in ("c0", "c1", "c2", "c3", "c0", "c2")]
        for _ in range(4):
            pos = rng.randrange(len(inc))
            inc[pos:pos] = block
        g = DiGraph.from_edges(project_incidence(inc))
        coalition = {"c0", "c1", "c2", "c3"}
        p_relay = density_pvalue(g, coalition, n_samples=50, seed=0, null="membership", incidence=inc)
        assert p_relay <= 0.05
        hub = {f"h{i}" for i in range(40)}
        p_hub = density_pvalue(g, hub, n_samples=50, seed=0, null="membership", incidence=inc)
        assert p_hub > 0.05
        # Presence-based density saturates: give the relay enough edits and
        # every ordered pair among four agents appears by chance in the null
        # too, so the coalition stops being distinguishable (12 blocks: p≈0.67).
        # A weighted density is the follow-up if that bites on real data.

    def test_private_page_pair_is_invisible_to_membership_null(self):
        """Documented blind spot: two agents alternating alone on their
        own page produce order-invariant edges, so the membership null
        reports them as expected; the bipartite configuration model, which
        also randomises page choice, still flags them."""
        inc = _hub_incidence()
        for _ in range(8):
            inc += [("c0", "p0"), ("c1", "p0"), ("c2", "p1"), ("c3", "p1"), ("c0", "p2"), ("c2", "p2")]
        inc += [("c1", "p2"), ("c3", "p0")]
        random.Random(5).shuffle(inc)
        g = DiGraph.from_edges(project_incidence(inc))
        coalition = {"c0", "c1", "c2", "c3"}
        p_member = density_pvalue(g, coalition, n_samples=50, seed=0, null="membership", incidence=inc)
        p_bip = density_pvalue(g, coalition, n_samples=50, seed=0, null="bipartite", incidence=inc)
        assert p_bip <= 0.05 < p_member
