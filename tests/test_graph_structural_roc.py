"""Smoke tests for the graph-structural ROC benchmark harness."""

from __future__ import annotations

import pytest

from experiments.graph_structural_roc import (
    DETECTORS,
    apply_decision_rule,
    generate_benign,
    generate_collusion_ring,
    generate_sybil_cluster,
    generate_threshold_dancing,
    roc_auc,
    run_family,
    run_sweep,
)


class TestGenerators:
    def test_benign_no_plant(self):
        s = generate_benign(n_agents=20, edge_density=0.1, seed=0)
        assert s.planted == set()
        assert len(s.nodes) == 20

    def test_collusion_ring_plants_ring(self):
        s = generate_collusion_ring(n_agents=30, ring_size=4, seed=1)
        assert s.planted == {"r0", "r1", "r2", "r3"}

    def test_sybil_cluster_plants_sybils(self):
        s = generate_sybil_cluster(n_agents=30, cluster_size=4, seed=2)
        assert s.planted == {"s0", "s1", "s2", "s3"}

    def test_threshold_dancing_plants_cluster(self):
        s = generate_threshold_dancing(n_agents=30, cluster_size=4, seed=3)
        assert s.planted == {"c0", "c1", "c2", "c3"}


class TestROCAUC:
    def test_perfect_separation(self):
        # planted nodes get score 1.0, rest 0.0
        scores = {"p1": 1.0, "p2": 1.0, "n1": 0.0, "n2": 0.0}
        auc = roc_auc(scores, {"p1", "p2"}, list(scores))
        assert auc == pytest.approx(1.0)

    def test_random_scores(self):
        scores = {"p1": 0.5, "p2": 0.5, "n1": 0.5, "n2": 0.5}
        auc = roc_auc(scores, {"p1", "p2"}, list(scores))
        assert auc == pytest.approx(0.5)

    def test_no_positives_returns_half(self):
        scores = {"a": 0.9, "b": 0.1}
        assert roc_auc(scores, set(), list(scores)) == 0.5


class TestDetectorAdapters:
    @pytest.mark.parametrize("name", list(DETECTORS))
    def test_returns_detector_output(self, name):
        s = generate_collusion_ring(n_agents=20, ring_size=3, seed=4)
        out = DETECTORS[name](s)
        # every node appears in scores and native_flags
        assert set(out.scores) == set(s.nodes)
        assert set(out.native_flags) == set(s.nodes)
        # scores in [0, 1] (loose; allow slight overshoot from raw similarity)
        for v in out.scores.values():
            assert 0.0 <= v <= 1.1
        # native_flags are booleans
        for v in out.native_flags.values():
            assert isinstance(v, bool)
        # clusters are sets of node ids drawn from s.nodes
        node_set = set(s.nodes)
        for c in out.clusters:
            assert isinstance(c, set)
            assert c.issubset(node_set)


class TestRunFamily:
    def test_small_run_completes(self):
        results = run_family(
            generate_collusion_ring,
            "collusion_ring_small",
            replicates=2,
            base_params={"n_agents": 20, "ring_size": 3},
            seed_base=10,
        )
        assert set(results) == set(DETECTORS)
        for r in results.values():
            assert r.n_replicates == 2


class TestDecisionRule:
    def _rr(self, det, auc_mean, auc_lo, auc_hi, f1_mean=float("nan"),
            f1_lo=float("nan"), f1_hi=float("nan")):
        from experiments.graph_structural_roc import RunResult
        return RunResult(
            family="f1", detector=det,
            auc_mean=auc_mean, auc_lo=auc_lo, auc_hi=auc_hi,
            f1_mean=f1_mean, f1_lo=f1_lo, f1_hi=f1_hi,
            n_replicates=10,
        )

    def test_auc_dominance_recommends_wiring(self):
        rs = [
            self._rr("graph_structural", 0.9, 0.85, 0.95),
            self._rr("identity_jaccard", 0.6, 0.55, 0.65),
            self._rr("reputation_mutual", 0.5, 0.45, 0.55),
            self._rr("collusion_score", 0.7, 0.65, 0.75),
        ]
        verdict = apply_decision_rule(rs)
        assert "VERDICT (AUC)" in verdict
        assert "strictly dominates" in verdict

    def test_tie_recommends_secondary_metric(self):
        rs = [
            self._rr("graph_structural", 0.7, 0.65, 0.75),
            self._rr("identity_jaccard", 0.7, 0.65, 0.75),
            self._rr("reputation_mutual", 0.7, 0.65, 0.75),
            self._rr("collusion_score", 0.7, 0.65, 0.75),
        ]
        verdict = apply_decision_rule(rs)
        assert "no governance wiring" in verdict.lower()

    def test_f1_dominance_suggests_default_on(self):
        rs = [
            self._rr("graph_structural", 0.7, 0.65, 0.75,
                     f1_mean=0.9, f1_lo=0.85, f1_hi=0.95),
            self._rr("identity_jaccard", 0.7, 0.65, 0.75,
                     f1_mean=0.5, f1_lo=0.45, f1_hi=0.55),
            self._rr("reputation_mutual", 0.7, 0.65, 0.75,
                     f1_mean=0.5, f1_lo=0.45, f1_hi=0.55),
            self._rr("collusion_score", 0.7, 0.65, 0.75,
                     f1_mean=0.5, f1_lo=0.45, f1_hi=0.55),
        ]
        verdict = apply_decision_rule(rs)
        assert "VERDICT (F1@native)" in verdict
        assert "default ON" in verdict


@pytest.mark.slow
class TestSweep:
    def test_full_sweep_smoke(self):
        results = run_sweep(replicates=2, seed_base=0)
        # 11 families * 5 detectors = 55
        # (8 single-coalition + 2 overlapping + 1 burst; +temporal detector)
        assert len(results) == 11 * 5


class TestOverlappingCoalitions:
    def test_planted_groups_has_n_coalitions(self):
        from experiments.graph_structural_roc import (
            generate_overlapping_coalitions,
        )
        s = generate_overlapping_coalitions(
            n_agents=40, n_coalitions=3, coalition_size=5,
            overlap_fraction=0.3, seed=0)
        assert len(s.planted_groups) == 3
        for g in s.planted_groups:
            assert len(g) == 5
        # planted is the union; with overlaps the union can be smaller
        # than n_coalitions * coalition_size.
        assert s.planted == set().union(*s.planted_groups)
        assert len(s.planted) <= 15

    def test_overlap_actually_overlaps(self):
        from experiments.graph_structural_roc import (
            generate_overlapping_coalitions,
        )
        # With overlap_fraction=1.0 every member of c[i] is shared with
        # c[i+1], so the union should be much smaller than n*size.
        s = generate_overlapping_coalitions(
            n_agents=40, n_coalitions=3, coalition_size=5,
            overlap_fraction=1.0, seed=0)
        # When overlap is total, all three coalitions are the same set.
        assert s.planted_groups[0] == s.planted_groups[1] == s.planted_groups[2]
        assert len(s.planted) == 5

    def test_default_planted_groups_for_single_coalition(self):
        """Back-compat: pre-qoro generators don't set planted_groups, but
        GraphSample.__post_init__ should default it to [planted]."""
        s = generate_collusion_ring(n_agents=20, ring_size=4, seed=1)
        assert s.planted_groups == [s.planted]


from experiments.graph_structural_roc import (  # noqa: E402
    hungarian_recovery,
    precision_recall_f1,
)


class TestPrecisionRecallF1:
    def test_perfect(self):
        flags = {"a": True, "b": True, "c": False, "d": False}
        p, r, f1 = precision_recall_f1(flags, {"a", "b"}, list(flags))
        assert (p, r, f1) == (1.0, 1.0, 1.0)

    def test_all_false_positive(self):
        flags = {"a": True, "b": True, "c": False, "d": False}
        p, r, f1 = precision_recall_f1(flags, {"c", "d"}, list(flags))
        # 0 TP, 2 FP, 2 FN -> precision 0, recall 0, f1 0
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_benign_returns_nan(self):
        flags = {"a": False, "b": False}
        p, r, f1 = precision_recall_f1(flags, set(), list(flags))
        import math as _m
        assert _m.isnan(p) and _m.isnan(r) and _m.isnan(f1)


class TestHungarianRecovery:
    def test_perfect_recovery(self):
        planted = [{"a", "b", "c"}]
        returned = [{"a", "b", "c"}]
        assert hungarian_recovery(returned, planted) == 1.0

    def test_partial_recovery(self):
        planted = [{"a", "b", "c", "d"}]
        returned = [{"a", "b"}]  # half
        # Jaccard(2/4) = 0.5
        assert hungarian_recovery(returned, planted) == pytest.approx(0.5)

    def test_no_returned_means_zero(self):
        assert hungarian_recovery([], [{"a", "b"}]) == 0.0

    def test_benign_returns_nan(self):
        import math as _m
        assert _m.isnan(hungarian_recovery([{"a"}], []))

    def test_greedy_assignment_multi_coalition(self):
        # Two planted coalitions; returned has a perfect and a partial match
        planted = [{"a", "b", "c"}, {"x", "y", "z"}]
        returned = [{"x", "y", "z"}, {"a", "b"}]
        # planted[0] best matches returned[1] (J=2/3); planted[1] best matches
        # returned[0] (J=1.0). Greedy: assigns J=1.0 first, then J=2/3.
        # Mean = (1.0 + 2/3) / 2 = 5/6
        assert hungarian_recovery(returned, planted) == pytest.approx(5/6)


class TestTemporalCoordination:
    def test_burst_generator_plants_cluster(self):
        from experiments.graph_structural_roc import generate_burst_coordination
        s = generate_burst_coordination(n_agents=30, cluster_size=4,
                                         n_bursts=2, seed=0)
        assert s.planted == {"c0", "c1", "c2", "c3"}
        assert s.family == "burst_coordination"

    def test_temporal_concentration_high_on_burst(self):
        """Cluster-internal interactions confined to a tiny window should
        score high concentration; honest spread should score near zero."""
        from datetime import datetime, timedelta

        from swarm.metrics.graph_structural import temporal_concentration
        from swarm.models.interaction import SoftInteraction
        base = datetime(2026, 1, 1)
        cluster = {"a", "b", "c"}
        # Cluster: 10 interactions all in the first 10 minutes
        # Honest: 20 interactions spread over 24 hours
        ixs = []
        for i in range(10):
            ixs.append(SoftInteraction(
                initiator="a", counterparty="b", p=0.8,
                timestamp=base + timedelta(seconds=i * 60)))
        for i in range(20):
            ixs.append(SoftInteraction(
                initiator="h0", counterparty="h1", p=0.6,
                timestamp=base + timedelta(seconds=i * 4000)))
        # Honest extends the time window; cluster is concentrated.
        conc = temporal_concentration(ixs, cluster, n_windows=12)
        assert conc > 0.7, f"expected high concentration, got {conc}"

    def test_temporal_concentration_low_on_uniform(self):
        from datetime import datetime, timedelta

        from swarm.metrics.graph_structural import temporal_concentration
        from swarm.models.interaction import SoftInteraction
        base = datetime(2026, 1, 1)
        cluster = {"a", "b"}
        # Cluster interactions uniform across full 1-hour window
        ixs = [
            SoftInteraction(
                initiator="a", counterparty="b", p=0.8,
                timestamp=base + timedelta(seconds=i * 300))
            for i in range(12)
        ]
        conc = temporal_concentration(ixs, cluster, n_windows=12)
        # Uniform-across-windows -> concentration near 0
        assert conc < 0.1, f"expected low concentration, got {conc}"

    def test_temporal_concentration_edge_cases(self):
        from swarm.metrics.graph_structural import temporal_concentration
        assert temporal_concentration([], {"a"}, n_windows=10) == 0.0
        # Single interaction -> n_in_cluster < 2 -> 0.0
        from datetime import datetime

        from swarm.models.interaction import SoftInteraction
        single = [SoftInteraction(
            initiator="a", counterparty="b", p=0.5,
            timestamp=datetime(2026, 1, 1))]
        assert temporal_concentration(single, {"a", "b"}, n_windows=10) == 0.0
