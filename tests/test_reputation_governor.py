"""Tests for ReputationGovernor reputation-based governance."""

import tempfile
from pathlib import Path

import pytest

from swarm.governance.reputation_governor import ReputationGovernor
from swarm.knowledge.graph_memory import GraphMemoryStore


@pytest.fixture
def temp_store():
    """Create a temporary graph memory store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "test_graph_memory.json"
        store = GraphMemoryStore(str(store_path))
        yield store


class TestReputationScoreComputation:
    """Test reputation score calculation."""

    def test_single_agent_no_relationships(self, temp_store):
        """Test reputation score when agent has no relationships."""
        governor = ReputationGovernor(temp_store)

        # Add a relationship edge manually (since no agent interactions yet)
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.8,
                "payoff_a": 10.0,
                "payoff_b": 8.0,
                "epoch": 1,
                "trust_a_to_b": 0.5,
                "trust_b_to_a": 0.5,
            },
        )

        scores = governor.compute_reputation_scores()

        # agent_1 has no incoming trust (no one trusts A->1, only A->2)
        # agent_2 has incoming trust from agent_1 (0.5)
        assert "agent_1" in scores
        assert "agent_2" in scores
        assert 0.0 <= scores["agent_1"] <= 1.0
        assert 0.0 <= scores["agent_2"] <= 1.0

    def test_known_topology_reputation(self, temp_store):
        """Test reputation computation with known trust topology.

        Setup:
        - agent_1 -> agent_2: trust=0.9, count=2
        - agent_2 -> agent_1: trust=0.1, count=2
        - agent_3 -> agent_2: trust=0.8, count=1

        Expected:
        - agent_2 has incoming: 0.9 (weight sqrt(2)) + 0.8 (weight sqrt(1))
        - agent_2 reputation = (0.9*sqrt(2) + 0.8*1) / (sqrt(2) + 1)
        """
        # A1 -> A2
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.9,
                "payoff_a": 10.0,
                "payoff_b": 8.0,
                "epoch": 1,
                "trust_a_to_b": 0.9,
            },
        )
        # Add interaction 2 to boost count
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.85,
                "payoff_a": 9.0,
                "payoff_b": 7.0,
                "epoch": 2,
            },
        )

        # A2 -> A1
        temp_store.add_relationship_event(
            "agent_2",
            "agent_1",
            {
                "p": 0.2,
                "payoff_a": 2.0,
                "payoff_b": 5.0,
                "epoch": 1,
                "trust_a_to_b": 0.1,
            },
        )
        # Add interaction 2 to boost count
        temp_store.add_relationship_event(
            "agent_2",
            "agent_1",
            {
                "p": 0.25,
                "payoff_a": 2.5,
                "payoff_b": 4.5,
                "epoch": 2,
            },
        )

        # A3 -> A2
        temp_store.add_relationship_event(
            "agent_3",
            "agent_2",
            {
                "p": 0.8,
                "payoff_a": 8.0,
                "payoff_b": 6.0,
                "epoch": 1,
                "trust_a_to_b": 0.8,
            },
        )

        governor = ReputationGovernor(temp_store)
        scores = governor.compute_reputation_scores()

        # agent_2 should have high reputation (trusted by A1 and A3)
        assert scores["agent_2"] > 0.5
        # agent_1 should have low reputation (distrusted by A2)
        assert scores["agent_1"] < 0.5

    def test_reputation_bounds(self, temp_store):
        """Test that all reputation scores are in [0, 1]."""
        # Create several relationships with varying trust values
        for i in range(5):
            temp_store.add_relationship_event(
                f"agent_{i}",
                f"agent_{i+1}",
                {
                    "p": 0.5 + (i * 0.1),
                    "payoff_a": 10.0,
                    "payoff_b": 10.0,
                    "epoch": 1,
                    "trust_a_to_b": 0.2 + (i * 0.15),
                },
            )

        governor = ReputationGovernor(temp_store)
        scores = governor.compute_reputation_scores()

        for agent_id, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score for {agent_id} out of bounds: {score}"

    def test_weighted_by_interaction_count(self, temp_store):
        """Test that reputation weighting respects interaction count.

        Setup:
        - A1 -> A2: trust=0.5, count=1
        - A3 -> A2: trust=0.9, count=10

        Expected:
        - A2's reputation should be closer to 0.9 because A3 has more interactions.
        """
        # A1 -> A2 (low trust, few interactions)
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.5,
                "payoff_a": 5.0,
                "payoff_b": 5.0,
                "epoch": 1,
                "trust_a_to_b": 0.5,
            },
        )

        # A3 -> A2 (high trust, many interactions)
        for i in range(10):
            temp_store.add_relationship_event(
                "agent_3",
                "agent_2",
                {
                    "p": 0.9,
                    "payoff_a": 9.0,
                    "payoff_b": 9.0,
                    "epoch": i,
                    "trust_a_to_b": 0.9,
                },
            )

        governor = ReputationGovernor(temp_store)
        scores = governor.compute_reputation_scores()

        # agent_2's reputation should lean toward 0.9 (weighted by sqrt(10))
        assert scores["agent_2"] > 0.7

    def test_empty_store(self, temp_store):
        """Test reputation computation with empty graph memory."""
        governor = ReputationGovernor(temp_store)
        scores = governor.compute_reputation_scores()

        assert scores == {}


class TestGovernanceRecommendations:
    """Test governance recommendation generation."""

    def test_normal_recommendation(self, temp_store):
        """Test normal recommendation for high-reputation agent."""
        # Create high-trust relationship
        temp_store.add_relationship_event(
            "agent_1",
            "agent_good",
            {
                "p": 0.9,
                "payoff_a": 10.0,
                "payoff_b": 10.0,
                "epoch": 1,
                "trust_a_to_b": 0.9,
            },
        )

        governor = ReputationGovernor(temp_store)
        recs = governor.get_governance_recommendations(threshold=0.3)

        assert recs["agent_good"] == "normal"

    def test_monitor_recommendation(self, temp_store):
        """Test monitor recommendation for medium-reputation agent."""
        # Create medium-trust relationship
        temp_store.add_relationship_event(
            "agent_1",
            "agent_medium",
            {
                "p": 0.5,
                "payoff_a": 5.0,
                "payoff_b": 5.0,
                "epoch": 1,
                "trust_a_to_b": 0.4,
            },
        )

        governor = ReputationGovernor(temp_store)
        recs = governor.get_governance_recommendations(threshold=0.5)

        assert recs["agent_medium"] == "monitor"

    def test_restrict_recommendation(self, temp_store):
        """Test restrict recommendation for low-reputation agent."""
        # Create low-trust relationship
        temp_store.add_relationship_event(
            "agent_1",
            "agent_bad",
            {
                "p": 0.2,
                "payoff_a": 2.0,
                "payoff_b": 2.0,
                "epoch": 1,
                "trust_a_to_b": 0.1,
            },
        )

        governor = ReputationGovernor(temp_store)
        recs = governor.get_governance_recommendations(threshold=0.3)

        assert recs["agent_bad"] == "restrict"

    def test_recommendations_with_custom_threshold(self, temp_store):
        """Test recommendations respect custom threshold."""
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.5,
                "payoff_a": 5.0,
                "payoff_b": 5.0,
                "epoch": 1,
                "trust_a_to_b": 0.5,
            },
        )

        governor = ReputationGovernor(temp_store)

        # With low threshold, should be normal
        recs_low = governor.get_governance_recommendations(threshold=0.2)
        assert recs_low["agent_2"] == "normal"

        # With high threshold, should be monitor
        recs_high = governor.get_governance_recommendations(threshold=0.7)
        assert recs_high["agent_2"] == "monitor"


class TestTrustWeightedFees:
    """Test trust-weighted fee calculation."""

    def test_high_reputation_discount(self, temp_store):
        """Test fee discount for high-reputation agents."""
        # Create high-trust relationship
        temp_store.add_relationship_event(
            "agent_1",
            "agent_good",
            {
                "p": 0.9,
                "payoff_a": 10.0,
                "payoff_b": 10.0,
                "epoch": 1,
                "trust_a_to_b": 0.9,
            },
        )

        governor = ReputationGovernor(temp_store)
        fee = governor.compute_trust_weighted_fee("agent_good", base_fee=100.0)

        assert fee == 80.0  # 0.8x discount

    def test_low_reputation_surcharge(self, temp_store):
        """Test fee surcharge for low-reputation agents."""
        # Create low-trust relationship
        temp_store.add_relationship_event(
            "agent_1",
            "agent_bad",
            {
                "p": 0.1,
                "payoff_a": 1.0,
                "payoff_b": 1.0,
                "epoch": 1,
                "trust_a_to_b": 0.1,
            },
        )

        governor = ReputationGovernor(temp_store)
        fee = governor.compute_trust_weighted_fee("agent_bad", base_fee=100.0)

        assert fee == 150.0  # 1.5x surcharge

    def test_medium_reputation_base_fee(self, temp_store):
        """Test base fee for medium-reputation agents."""
        temp_store.add_relationship_event(
            "agent_1",
            "agent_medium",
            {
                "p": 0.5,
                "payoff_a": 5.0,
                "payoff_b": 5.0,
                "epoch": 1,
                "trust_a_to_b": 0.5,
            },
        )

        governor = ReputationGovernor(temp_store)
        fee = governor.compute_trust_weighted_fee("agent_medium", base_fee=100.0)

        assert fee == 100.0  # Base fee (no adjustment)

    def test_unknown_agent_neutral_fee(self, temp_store):
        """Test fee for unknown agent (default neutral reputation)."""
        governor = ReputationGovernor(temp_store)
        fee = governor.compute_trust_weighted_fee("agent_unknown", base_fee=100.0)

        # Default reputation is 0.5, so should get base fee
        assert fee == 100.0


class TestCollusionDetection:
    """Test collusion cluster detection."""

    def test_simple_pair_collusion(self, temp_store):
        """Test detection of simple 2-agent collusion."""
        # Create mutual high trust between A1 and A2
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.9,
                "payoff_a": 10.0,
                "payoff_b": 10.0,
                "epoch": 1,
                "trust_a_to_b": 0.95,
            },
        )
        temp_store.add_relationship_event(
            "agent_2",
            "agent_1",
            {
                "p": 0.9,
                "payoff_a": 10.0,
                "payoff_b": 10.0,
                "epoch": 1,
                "trust_a_to_b": 0.95,
            },
        )

        governor = ReputationGovernor(temp_store)
        clusters = governor.detect_collusion_clusters(min_mutual_trust=0.9)

        assert len(clusters) == 1
        assert sorted(clusters[0]) == ["agent_1", "agent_2"]

    def test_no_collusion_below_threshold(self, temp_store):
        """Test that low-trust pairs don't form collusion clusters."""
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.5,
                "payoff_a": 5.0,
                "payoff_b": 5.0,
                "epoch": 1,
                "trust_a_to_b": 0.5,
            },
        )
        temp_store.add_relationship_event(
            "agent_2",
            "agent_1",
            {
                "p": 0.5,
                "payoff_a": 5.0,
                "payoff_b": 5.0,
                "epoch": 1,
                "trust_a_to_b": 0.5,
            },
        )

        governor = ReputationGovernor(temp_store)
        clusters = governor.detect_collusion_clusters(min_mutual_trust=0.9)

        assert len(clusters) == 0

    def test_larger_collusion_cluster(self, temp_store):
        """Test detection of larger collusion clusters."""
        # Create 3-agent ring with high mutual trust
        agents = ["agent_1", "agent_2", "agent_3"]

        for i in range(len(agents)):
            a = agents[i]
            b = agents[(i + 1) % len(agents)]
            # A -> B
            temp_store.add_relationship_event(
                a,
                b,
                {
                    "p": 0.95,
                    "payoff_a": 10.0,
                    "payoff_b": 10.0,
                    "epoch": 1,
                    "trust_a_to_b": 0.95,
                },
            )
            # B -> A
            temp_store.add_relationship_event(
                b,
                a,
                {
                    "p": 0.95,
                    "payoff_a": 10.0,
                    "payoff_b": 10.0,
                    "epoch": 1,
                    "trust_a_to_b": 0.95,
                },
            )

        governor = ReputationGovernor(temp_store)
        clusters = governor.detect_collusion_clusters(min_mutual_trust=0.9)

        assert len(clusters) == 1
        assert len(clusters[0]) == 3
        assert set(clusters[0]) == set(agents)

    def test_collusion_respects_min_size(self, temp_store):
        """Test that clusters below min_size are filtered out."""
        # Create pair with high trust
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.95,
                "payoff_a": 10.0,
                "payoff_b": 10.0,
                "epoch": 1,
                "trust_a_to_b": 0.95,
            },
        )
        temp_store.add_relationship_event(
            "agent_2",
            "agent_1",
            {
                "p": 0.95,
                "payoff_a": 10.0,
                "payoff_b": 10.0,
                "epoch": 1,
                "trust_a_to_b": 0.95,
            },
        )

        governor = ReputationGovernor(temp_store)

        # With min_size=3, pair should not be detected
        clusters = governor.detect_collusion_clusters(min_mutual_trust=0.9, min_size=3)
        assert len(clusters) == 0

        # With min_size=2, pair should be detected
        clusters = governor.detect_collusion_clusters(min_mutual_trust=0.9, min_size=2)
        assert len(clusters) == 1

    def test_no_collusion_unidirectional_trust(self, temp_store):
        """Test that unidirectional trust doesn't create collusion clusters."""
        # A1 -> A2 high trust, but A2 -> A1 low trust
        temp_store.add_relationship_event(
            "agent_1",
            "agent_2",
            {
                "p": 0.95,
                "payoff_a": 10.0,
                "payoff_b": 10.0,
                "epoch": 1,
                "trust_a_to_b": 0.95,
            },
        )
        temp_store.add_relationship_event(
            "agent_2",
            "agent_1",
            {
                "p": 0.2,
                "payoff_a": 2.0,
                "payoff_b": 2.0,
                "epoch": 1,
                "trust_a_to_b": 0.1,
            },
        )

        governor = ReputationGovernor(temp_store)
        clusters = governor.detect_collusion_clusters(min_mutual_trust=0.9)

        assert len(clusters) == 0

    def test_empty_store_no_clusters(self, temp_store):
        """Test that empty store has no collusion clusters."""
        governor = ReputationGovernor(temp_store)
        clusters = governor.detect_collusion_clusters()

        assert clusters == []


def _record_event(store, a, b, p=0.7, trust_ab=0.7, trust_ba=0.7, epoch=1):
    """Helper: add a single relationship event with trust on both sides."""
    store.add_relationship_event(a, b, {
        "p": p, "payoff_a": 1.0, "payoff_b": 1.0, "epoch": epoch,
        "trust_a_to_b": trust_ab, "trust_b_to_a": trust_ba,
    })


class TestStructuralCoalitionDetection:
    """Test ReputationGovernor's graph-structural coalition wiring (beads-4ae5)."""

    def test_explicit_off_preserves_pure_reputation(self, temp_store):
        """structural_enabled=False (legacy) — recommendations come from
        reputation alone, so a high-trust mutual clique stays "normal"
        even though it would be flagged structurally."""
        # Plant a mutual clique with high p (so trust=p is high)
        ring = ["r0", "r1", "r2", "r3"]
        for u in ring:
            for v in ring:
                if u != v:
                    _record_event(temp_store, u, v, p=0.9,
                                  trust_ab=0.9, trust_ba=0.9)
        governor = ReputationGovernor(temp_store)
        recs = governor.get_governance_recommendations(structural_enabled=False)
        for r in ring:
            assert recs[r] == "normal"

    def test_default_on_flags_planted_ring(self, temp_store):
        """Default behavior (post-beads-2iok): structural_enabled=True is
        the default, so a planted mutual ring gets bumped off "normal"
        without the caller passing any flag."""
        ring = ["r0", "r1", "r2", "r3", "r4"]
        honest = [f"h{i}" for i in range(15)]
        for u in ring:
            for v in ring:
                if u != v:
                    _record_event(temp_store, u, v, p=0.9,
                                  trust_ab=0.9, trust_ba=0.9)
        # Honest one-way background (no reverse edge)
        import random as _r
        rng = _r.Random(0)
        for _ in range(40):
            a, b = rng.sample(honest, 2)
            _record_event(temp_store, a, b, p=0.6,
                          trust_ab=0.6, trust_ba=0.0)

        governor = ReputationGovernor(temp_store)
        # No flag passed -> uses the new default (True since beads-2iok).
        recs_default = governor.get_governance_recommendations(
            structural_null_samples=20, structural_seed=1)
        recs_off = governor.get_governance_recommendations(
            structural_enabled=False)

        bumped = [r for r in ring if recs_default[r] != "normal"]
        assert len(bumped) >= 3, f"expected ≥3 bumped, got {bumped}"
        for r in ring:
            assert recs_off[r] == "normal"

    def test_detect_structural_coalitions_returns_anomalies(self, temp_store):
        ring = ["r0", "r1", "r2", "r3", "r4"]
        for u in ring:
            for v in ring:
                if u != v:
                    _record_event(temp_store, u, v, p=0.9,
                                  trust_ab=0.9, trust_ba=0.9)
        governor = ReputationGovernor(temp_store)
        anomalies = governor.detect_structural_coalitions(
            min_size=3, n_null_samples=20, seed=0)
        # At least one anomaly should be the ring.
        ring_set = set(ring)
        best = max((len(a.members & ring_set) for a in anomalies), default=0)
        assert best >= 4
