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

    def test_persistence_high_on_recurring_cluster(self):
        """A cluster active in every window is a standing channel:
        persistence ~1.0. Complement of the concentration signal."""
        from datetime import datetime, timedelta

        from swarm.metrics.graph_structural import cross_window_persistence
        from swarm.models.interaction import SoftInteraction
        base = datetime(2026, 1, 1)
        cluster = {"a", "b"}
        # One cluster-internal interaction every 5 minutes for an hour
        ixs = [
            SoftInteraction(
                initiator="a", counterparty="b", p=0.8,
                timestamp=base + timedelta(seconds=i * 300))
            for i in range(12)
        ]
        pers = cross_window_persistence(ixs, cluster, n_windows=12)
        assert pers > 0.9, f"expected ~1.0 persistence, got {pers}"

    def test_persistence_low_on_one_flare(self):
        """A single-window flare against an always-active background
        scores 1/n_active_windows — the shape persistence is built to
        deprioritize relative to recurrence."""
        from datetime import datetime, timedelta

        from swarm.metrics.graph_structural import cross_window_persistence
        from swarm.models.interaction import SoftInteraction
        base = datetime(2026, 1, 1)
        cluster = {"a", "b"}
        ixs = []
        # Cluster: 10 interactions inside one minute
        for i in range(10):
            ixs.append(SoftInteraction(
                initiator="a", counterparty="b", p=0.8,
                timestamp=base + timedelta(seconds=i * 6)))
        # Background keeps every window active over 2 hours
        for i in range(120):
            ixs.append(SoftInteraction(
                initiator="h0", counterparty="h1", p=0.6,
                timestamp=base + timedelta(seconds=i * 60)))
        pers = cross_window_persistence(ixs, cluster, n_windows=12)
        assert pers <= 1 / 12 + 1e-9, f"expected one-window persistence, got {pers}"

    def test_persistence_denominator_is_active_windows(self):
        """Globally quiet windows must not dilute persistence: cluster
        active in both of the only two active windows scores 1.0."""
        from datetime import datetime, timedelta

        from swarm.metrics.graph_structural import cross_window_persistence
        from swarm.models.interaction import SoftInteraction
        base = datetime(2026, 1, 1)
        cluster = {"a", "b"}
        # Activity only at the very start and very end of the range;
        # the 8 middle windows are empty for everyone.
        ixs = [
            SoftInteraction(initiator="a", counterparty="b", p=0.8,
                            timestamp=base),
            SoftInteraction(initiator="a", counterparty="b", p=0.8,
                            timestamp=base + timedelta(hours=10)),
        ]
        pers = cross_window_persistence(ixs, cluster, n_windows=10)
        assert pers == pytest.approx(1.0)

    def test_persistence_edge_cases(self):
        from datetime import datetime

        from swarm.metrics.graph_structural import cross_window_persistence
        from swarm.models.interaction import SoftInteraction
        assert cross_window_persistence([], {"a"}, n_windows=10) == 0.0
        # All timestamps equal -> no time axis -> 0.0
        same = [
            SoftInteraction(initiator="a", counterparty="b", p=0.5,
                            timestamp=datetime(2026, 1, 1))
            for _ in range(3)
        ]
        assert cross_window_persistence(same, {"a", "b"}, n_windows=10) == 0.0

    def test_temporal_anomalies_annotated_with_shape_signals(self):
        """temporal_densest_anomalies fills temporal_concentration and
        temporal_persistence on every returned anomaly; the static
        detector leaves the 0.0 defaults."""
        from experiments.graph_structural_roc import generate_burst_coordination
        from swarm.metrics.graph_structural import (
            detect_structural_anomalies,
            temporal_densest_anomalies,
        )
        s = generate_burst_coordination(n_agents=30, cluster_size=4,
                                        n_bursts=2, seed=0)
        window_anoms = temporal_densest_anomalies(
            s.interactions, n_windows=10, min_size=3, n_null_samples=5, seed=1)
        assert window_anoms, "burst sample should surface per-window anomalies"
        cluster_anoms = [
            a for a in window_anoms if a.members <= s.planted
        ]
        assert cluster_anoms, "planted cluster should appear as a window anomaly"
        for a in cluster_anoms:
            # Bursty cluster: concentrated (few windows), not persistent.
            assert a.temporal_concentration > 0.4
            assert a.temporal_persistence < 0.5
        static = detect_structural_anomalies(s.edges, n_null_samples=5, seed=1)
        for a in static:
            assert a.temporal_concentration == 0.0
            assert a.temporal_persistence == 0.0

    def test_planted_burst_cluster_recurs_across_windows(self):
        """The planted coalition re-surfaces as (Jaccard-matched) the
        same member set in multiple windows; that recurrence count is
        the signal separating it from one-window honest noise."""
        from experiments.graph_structural_roc import generate_burst_coordination
        from swarm.metrics.graph_structural import temporal_densest_anomalies
        s = generate_burst_coordination(n_agents=50, cluster_size=5,
                                        n_bursts=3, seed=0)
        anoms = temporal_densest_anomalies(
            s.interactions, n_windows=10, min_size=3, n_null_samples=5, seed=1)
        planted_reps = [
            a for a in anoms
            if len(a.members & s.planted) / len(a.members | s.planted) >= 0.6
        ]
        assert planted_reps, "planted cluster should surface as a group"
        assert max(a.n_windows_surfaced for a in planted_reps) >= 2

    def test_temporal_detector_flags_cluster_not_everything(self):
        """Regression for the flag-everything failure (beads-p56u): the
        first-cut detector flagged all 50 nodes on its own burst family
        (every one-window candidate is window-concentrated by
        construction; whole-graph candidates pass the persistence gate
        trivially). The recurrence + minority gates must keep the flag
        set tight around the planted cluster."""
        from experiments.graph_structural_roc import (
            detector_graph_structural_temporal,
            generate_burst_coordination,
        )
        s = generate_burst_coordination(n_agents=50, cluster_size=5,
                                        n_bursts=3, seed=0)
        out = detector_graph_structural_temporal(s)
        flagged = {n for n, f in out.native_flags.items() if f}
        assert s.planted <= flagged, "planted cluster must be flagged"
        assert len(flagged) <= 3 * len(s.planted), (
            f"flag set should be tight around the cluster, got {len(flagged)}"
        )


# ---------------------------------------------------------------------------
# Overlapping-coalition ground truth, per-coalition TPR, overlap sweep (qoro)
# ---------------------------------------------------------------------------

from experiments.graph_structural_roc import (  # noqa: E402
    CRITERION_OVERLAP,
    CRITERION_TPR,
    RunResult,
    disjoint_ground_truth,
    overlap_criterion_verdict,
    per_coalition_tpr,
    run_overlap_sweep,
)


class TestDisjointGroundTruth:
    def test_oldest_coalition_wins(self):
        # "b" is in both; coalition 0 was joined first, so it keeps "b".
        planted = [{"a", "b"}, {"b", "c"}]
        assert disjoint_ground_truth(planted) == [{"a", "b"}, {"c"}]

    def test_result_is_a_partition(self):
        planted = [{"a", "b", "c"}, {"c", "d"}, {"a", "d", "e"}]
        out = disjoint_ground_truth(planted)
        # No member appears twice, and nothing is lost.
        flat = [m for g in out for m in g]
        assert len(flat) == len(set(flat))
        assert set(flat) == set().union(*planted)

    def test_fully_absorbed_coalition_is_dropped(self):
        # At overlap 1.0 every coalition is the same set; only one survives,
        # rather than leaving empty groups that would score a free 0 Jaccard.
        planted = [{"a", "b"}, {"a", "b"}, {"a", "b"}]
        assert disjoint_ground_truth(planted) == [{"a", "b"}]

    def test_disjoint_input_is_unchanged(self):
        planted = [{"a", "b"}, {"c", "d"}]
        assert disjoint_ground_truth(planted) == planted

    def test_empty(self):
        assert disjoint_ground_truth([]) == []

    def test_sample_property_matches_helper(self):
        from experiments.graph_structural_roc import (
            generate_overlapping_coalitions,
        )
        s = generate_overlapping_coalitions(
            n_agents=40, n_coalitions=3, coalition_size=5,
            overlap_fraction=0.5, seed=3)
        assert s.planted_groups_disjoint == disjoint_ground_truth(
            s.planted_groups)
        # Partition of the same union the overlapping truth covers.
        flat = [m for g in s.planted_groups_disjoint for m in g]
        assert len(flat) == len(set(flat))
        assert set(flat) == s.planted

    def test_single_coalition_generators_unaffected(self):
        s = generate_collusion_ring(n_agents=20, ring_size=4, seed=1)
        assert s.planted_groups_disjoint == [s.planted]


class TestPerCoalitionTPR:
    def test_perfect(self):
        planted = [{"a", "b"}, {"x", "y"}]
        assert per_coalition_tpr([{"a", "b"}, {"x", "y"}], planted) == (1.0, 1.0)

    def test_min_is_the_worst_coalition_not_the_mean(self):
        # One coalition fully recovered, one half — mean 0.75, min 0.5.
        planted = [{"a", "b"}, {"x", "y"}]
        returned = [{"a", "b"}, {"x"}]
        mean, mn = per_coalition_tpr(returned, planted)
        assert mean == pytest.approx(0.75)
        assert mn == pytest.approx(0.5)

    def test_unmatched_coalition_scores_zero(self):
        # Only one returned cluster for two planted coalitions.
        planted = [{"a", "b"}, {"x", "y"}]
        mean, mn = per_coalition_tpr([{"a", "b"}], planted)
        assert mean == pytest.approx(0.5)
        assert mn == 0.0

    def test_no_clusters_returned(self):
        assert per_coalition_tpr([], [{"a", "b"}]) == (0.0, 0.0)

    def test_benign_returns_nan(self):
        import math as _m
        mean, mn = per_coalition_tpr([{"a"}], [])
        assert _m.isnan(mean) and _m.isnan(mn)

    def test_merging_two_coalitions_is_penalized_twice(self):
        """Why 1:1 matching matters for the merged-blob case.

        A detector that lumps two coalitions into one cluster fully contains
        the coalition it matches (TPR 1.0 there), but 1:1 matching means the
        blob can only claim *one* — the second coalition is left unmatched at
        0.0. So min TPR is 0.0, not 1.0: merging is scored as missing a
        coalition, which is the behavior the criterion should reward avoiding.
        TPR still isn't a restatement of Jaccard — mean TPR 0.5 vs recovery
        0.25, since Jaccard also charges for the members the blob adds.
        """
        planted = [{"a", "b"}, {"x", "y"}]
        blob = [{"a", "b", "x", "y"}]
        mean, mn = per_coalition_tpr(blob, planted)
        assert mn == 0.0
        assert mean == pytest.approx(0.5)
        assert hungarian_recovery(blob, planted) == pytest.approx(0.25)

    def test_agrees_with_hungarian_on_exact_recovery(self):
        planted = [{"a", "b", "c"}, {"x", "y", "z"}]
        returned = [{"x", "y", "z"}, {"a", "b", "c"}]
        assert per_coalition_tpr(returned, planted) == (1.0, 1.0)
        assert hungarian_recovery(returned, planted) == 1.0


def _cell(**kw):
    """Build a {detector: RunResult} cell for verdict tests."""
    return {name: RunResult(family="overlap_0.40", detector=name,
                            auc_mean=0.5, auc_lo=0.4, auc_hi=0.6, **vals)
            for name, vals in kw.items()}


class TestOverlapCriterionVerdict:
    def test_met_when_all_clustering_detectors_clear_the_bar(self):
        sweep = {CRITERION_OVERLAP: _cell(
            good={"tpr_min": 0.9, "cluster_rate": 1.0},
            alsogood={"tpr_min": CRITERION_TPR, "cluster_rate": 0.8},
        )}
        v = overlap_criterion_verdict(sweep)
        assert "MET by all 2" in v
        assert "NOT MET" not in v

    def test_not_met_names_the_failing_detector_and_its_tpr(self):
        sweep = {CRITERION_OVERLAP: _cell(
            weak={"tpr_min": 0.42, "cluster_rate": 1.0},
        )}
        v = overlap_criterion_verdict(sweep)
        assert "NOT MET" in v
        assert "weak (min TPR 0.420)" in v

    def test_non_clustering_detector_is_excluded_not_failed(self):
        """The load-bearing distinction: a detector that never returns a
        cluster scores TPR 0 on every family, so counting it as an overlap
        failure would manufacture evidence about overlap it cannot supply."""
        sweep = {CRITERION_OVERLAP: _cell(
            clusters={"tpr_min": 0.9, "cluster_rate": 1.0},
            abstains={"tpr_min": 0.0, "cluster_rate": 0.0},
        )}
        v = overlap_criterion_verdict(sweep)
        assert "MET by all 1" in v
        assert "Not evaluated (1): abstains" in v
        assert "NOT MET" not in v

    def test_all_abstain_is_not_evaluable(self):
        sweep = {CRITERION_OVERLAP: _cell(
            a={"tpr_min": 0.0, "cluster_rate": 0.0},
            b={"tpr_min": 0.0, "cluster_rate": float("nan")},
        )}
        assert "NOT EVALUABLE" in overlap_criterion_verdict(sweep)

    def test_missing_criterion_cell_is_reported(self):
        assert "not in the swept grid" in overlap_criterion_verdict({0.1: {}})


class TestOverlapSweep:
    def test_sweep_covers_the_grid_and_criterion_point(self):
        sweep = run_overlap_sweep(replicates=1, overlaps=[0.0, CRITERION_OVERLAP])
        assert set(sweep) == {0.0, CRITERION_OVERLAP}
        for cell in sweep.values():
            assert set(cell) == set(DETECTORS)
            for r in cell.values():
                assert 0.0 <= r.cluster_rate <= 1.0

    def test_overlap_is_recorded_in_the_family_name(self):
        sweep = run_overlap_sweep(replicates=1, overlaps=[0.3])
        assert all(r.family == "overlap_0.30" for r in sweep[0.3].values())
