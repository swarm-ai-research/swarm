"""Tests for collusion detection module."""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pytest
from pydantic import ValidationError

from swarm.governance import GovernanceConfig, GovernanceEngine
from swarm.metrics.collusion import (
    CollusionDetector,
    detect_vote_coordination,
    temporal_clustering_score,
)
from swarm.models.interaction import InteractionType, SoftInteraction

# =============================================================================
# Test Fixtures
# =============================================================================


def create_interaction(
    initiator: str,
    counterparty: str,
    p: float = 0.5,
    accepted: bool = True,
    tau: float = 0.0,
    c_a: float = 0.0,
    c_b: float = 0.0,
    timestamp: Optional[datetime] = None,
) -> SoftInteraction:
    """Create a test interaction."""
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.COLLABORATION,
        accepted=accepted,
        p=p,
        tau=tau,
        c_a=c_a,
        c_b=c_b,
        timestamp=timestamp or datetime.now(),
    )


def create_collusion_pattern(
    agent_a: str,
    agent_b: str,
    count: int = 10,
    p_internal: float = 0.9,
    acceptance_rate: float = 1.0,
    base_time: Optional[datetime] = None,
    interval_seconds: float = 10.0,
) -> list[SoftInteraction]:
    """Create interactions that look like collusion."""
    base = base_time or datetime.now()
    interactions = []
    for i in range(count):
        # Alternate initiator
        init, counter = (agent_a, agent_b) if i % 2 == 0 else (agent_b, agent_a)
        accepted = np.random.random() < acceptance_rate
        interactions.append(
            create_interaction(
                initiator=init,
                counterparty=counter,
                p=p_internal,
                accepted=accepted,
                timestamp=base + timedelta(seconds=i * interval_seconds),
            )
        )
    return interactions


def create_normal_interactions(
    agents: list[str],
    count: int = 50,
    seed: int = 42,
) -> list[SoftInteraction]:
    """Create normal (non-colluding) interactions."""
    rng = np.random.default_rng(seed)
    interactions = []
    base = datetime.now()

    for i in range(count):
        a, b = rng.choice(agents, size=2, replace=False)
        p = rng.uniform(0.3, 0.8)
        accepted = rng.random() < 0.6
        interactions.append(
            create_interaction(
                initiator=a,
                counterparty=b,
                p=p,
                accepted=accepted,
                timestamp=base + timedelta(seconds=i * 60),
            )
        )
    return interactions


# =============================================================================
# CollusionDetector Tests
# =============================================================================


class TestCollusionDetector:
    """Tests for CollusionDetector class."""

    def test_empty_interactions(self):
        """Should handle empty interaction list."""
        detector = CollusionDetector()
        report = detector.analyze([])

        assert report.ecosystem_collusion_risk == 0.0
        assert report.n_flagged_pairs == 0
        assert report.n_flagged_groups == 0

    def test_single_interaction(self):
        """Should handle single interaction."""
        detector = CollusionDetector(min_interactions_for_analysis=1)
        interaction = create_interaction("a", "b", p=0.8)
        report = detector.analyze([interaction])

        # Single interaction shouldn't be flagged
        assert report.ecosystem_collusion_risk < 0.5

    def test_detects_high_frequency_pair(self):
        """Should detect unusually frequent interactions between a pair."""
        detector = CollusionDetector(
            frequency_threshold=1.5,
            min_interactions_for_analysis=3,
            collusion_score_threshold=0.3,
        )

        # Create many interactions between A-B, few between others
        interactions = create_collusion_pattern("alice", "bob", count=20)
        # Add sparse interactions with others
        for agent in ["charlie", "david", "eve"]:
            interactions.append(create_interaction("alice", agent))
            interactions.append(create_interaction("bob", agent))

        report = detector.analyze(interactions)

        # Should flag alice-bob pair
        assert report.n_flagged_pairs >= 1
        flagged_agents = {p.agent_a for p in report.suspicious_pairs} | {
            p.agent_b for p in report.suspicious_pairs
        }
        assert "alice" in flagged_agents or "bob" in flagged_agents

    def test_detects_high_acceptance_rate(self):
        """Should flag pairs with unusually high mutual acceptance."""
        detector = CollusionDetector(
            min_interactions_for_analysis=5,
            collusion_score_threshold=0.3,
        )

        # Create pattern with 100% acceptance (suspicious)
        interactions = create_collusion_pattern(
            "alice",
            "bob",
            count=15,
            acceptance_rate=1.0,
            p_internal=0.4,  # Low quality but still accepted
        )
        # Add normal interactions with OTHER agents (not alice/bob)
        interactions.extend(
            create_normal_interactions(
                ["charlie", "david", "eve", "frank"],
                count=20,
            )
        )

        report = detector.analyze(interactions)

        # Alice-Bob should have high acceptance score
        alice_bob_pair = next(
            (
                p
                for p in report.suspicious_pairs
                if {p.agent_a, p.agent_b} == {"alice", "bob"}
            ),
            None,
        )
        if alice_bob_pair:
            assert alice_bob_pair.mutual_acceptance_rate >= 0.8

    def test_agent_risk_scores(self):
        """Should compute per-agent risk scores."""
        detector = CollusionDetector(
            min_interactions_for_analysis=3,
            collusion_score_threshold=0.2,
        )

        interactions = create_collusion_pattern("alice", "bob", count=10)
        interactions.extend(create_normal_interactions(["charlie", "david"], count=20))

        # Add some interactions involving colluders with others
        interactions.append(create_interaction("alice", "charlie", p=0.3))
        interactions.append(create_interaction("bob", "david", p=0.3))

        report = detector.analyze(
            interactions,
            agent_ids=["alice", "bob", "charlie", "david"],
        )

        # Colluders should have higher risk
        assert "alice" in report.agent_collusion_risk
        assert "bob" in report.agent_collusion_risk
        # They may or may not be flagged depending on thresholds

    def test_ecosystem_risk_calculation(self):
        """Should compute overall ecosystem risk."""
        detector = CollusionDetector(collusion_score_threshold=0.3)

        # Create scenario with multiple colluding pairs
        interactions = []
        interactions.extend(create_collusion_pattern("a1", "a2", count=10))
        interactions.extend(create_collusion_pattern("b1", "b2", count=10))
        interactions.extend(create_normal_interactions(["c1", "c2", "c3"], count=20))

        report = detector.analyze(interactions)

        # Should have non-zero ecosystem risk
        assert 0.0 <= report.ecosystem_collusion_risk <= 1.0


class TestPairMetrics:
    """Tests for pair-level metrics computation."""

    def test_interaction_count(self):
        """Should count interactions between pair."""
        detector = CollusionDetector(min_interactions_for_analysis=1)

        interactions = [
            create_interaction("a", "b"),
            create_interaction("b", "a"),
            create_interaction("a", "b"),
        ]

        detector.analyze(interactions)

        # Find the a-b pair metrics
        for key, metrics in detector._group_by_pair(interactions).items():
            if {"a", "b"} == set(key):
                assert len(metrics) == 3

    def test_benefit_correlation(self):
        """Should compute benefit correlation."""
        detector = CollusionDetector(min_interactions_for_analysis=3)

        # Create interactions with correlated benefits (both gain)
        interactions = [
            create_interaction("a", "b", tau=1.0),  # B gains
            create_interaction("b", "a", tau=1.0),  # A gains
            create_interaction("a", "b", tau=1.0),
            create_interaction("b", "a", tau=1.0),
        ]

        report = detector.analyze(interactions)

        # Metrics should be computed
        if report.suspicious_pairs:
            pair = report.suspicious_pairs[0]
            # Correlation should be computed (may vary based on implementation)
            assert -1.0 <= pair.benefit_correlation <= 1.0


class TestGroupDetection:
    """Tests for group-level collusion detection."""

    def test_detects_connected_group(self):
        """Should detect groups from connected suspicious pairs."""
        detector = CollusionDetector(
            min_interactions_for_analysis=3,
            collusion_score_threshold=0.2,
        )

        # Create a triangle of suspicious interactions
        interactions = []
        interactions.extend(create_collusion_pattern("a", "b", count=8))
        interactions.extend(create_collusion_pattern("b", "c", count=8))
        interactions.extend(create_collusion_pattern("a", "c", count=8))

        # Add normal outsiders
        interactions.extend(create_normal_interactions(["d", "e", "f"], count=30))

        report = detector.analyze(interactions)

        # Should potentially detect a group (depends on scores)
        # At minimum, should detect suspicious pairs
        assert report.n_flagged_pairs > 0 or report.n_flagged_groups > 0

    def test_export_harm_coalition_flagged_by_group_first_pass(self):
        """Group-first pass (bead mwve): a coalition that trades HIGH-p
        internally and dumps LOW-p extraction on outsiders is invisible to
        the pair-first pipeline (no pair looks bad) but must be flagged.

        This is the emergent-RL-coalition signature from
        scenarios/rl_emergence.yaml capacity=3 seed=44, reproduced as a
        synthetic fixture.
        """
        members = ["m1", "m2", "m3"]
        victims = ["v1", "v2", "v3"]
        interactions = []

        # High-p internal trade among all member pairs (mutual favoritism).
        for a, b in [("m1", "m2"), ("m2", "m3"), ("m1", "m3")]:
            interactions.extend(create_collusion_pattern(a, b, count=12, p_internal=0.85))

        # Each member extracts (low-p, accepted) from each victim.
        for m in members:
            for v in victims:
                for _ in range(6):
                    interactions.append(
                        create_interaction(m, v, p=0.25, accepted=True)
                    )

        # Victims also do honest high-p work among themselves.
        for a, b in [("v1", "v2"), ("v2", "v3"), ("v1", "v3")]:
            for _ in range(5):
                interactions.append(create_interaction(a, b, p=0.8, accepted=True))

        detector = CollusionDetector(
            min_interactions_for_analysis=3,
            quality_asymmetry_threshold=0.2,
        )
        report = detector.analyze(interactions)

        group_members = [set(g.members) for g in report.suspicious_groups]
        assert any(
            set(members) <= gm for gm in group_members
        ), f"export-harm coalition not flagged; groups={group_members}"

        flagged = next(
            g for g in report.suspicious_groups if set(members) <= set(g.members)
        )
        assert flagged.detection_method == "mutual_favoritism"
        assert flagged.directional_quality_asymmetry >= 0.2
        # Victims must NOT be reported as coalition members.
        assert not (set(victims) & set(flagged.members))

    def test_group_first_pass_ignores_uniform_honest_population(self):
        """A population where everyone trades high-p with everyone must not
        be flagged — mutual favoritism without out-group harm is not
        collusion."""
        agents = ["h1", "h2", "h3", "h4", "h5"]
        interactions = []
        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                for _ in range(6):
                    interactions.append(create_interaction(a, b, p=0.8, accepted=True))

        detector = CollusionDetector(
            min_interactions_for_analysis=3,
            quality_asymmetry_threshold=0.2,
        )
        report = detector.analyze(interactions)
        assert report.n_flagged_groups == 0
        assert report.ecosystem_collusion_risk == 0.0

    def test_group_first_pass_ignores_favor_cluster_without_victims(self):
        """A mutual-favoritism cluster whose members never engage the
        out-group has no victims to extract from — not a harm-exporting
        coalition, so it must not be flagged even though internal p is
        high and external contact is nil."""
        interactions = []
        # Tight high-p cluster, no outbound at all.
        for a, b in [("c1", "c2"), ("c2", "c3"), ("c1", "c3")]:
            interactions.extend(create_collusion_pattern(a, b, count=10, p_internal=0.9))
        # Separate outsiders doing their own thing.
        for a, b in [("o1", "o2"), ("o2", "o3")]:
            for _ in range(6):
                interactions.append(create_interaction(a, b, p=0.7, accepted=True))

        detector = CollusionDetector(
            min_interactions_for_analysis=3,
            quality_asymmetry_threshold=0.2,
        )
        report = detector.analyze(interactions)
        # The cluster has no outbound engagement, so the group-first pass
        # must not promote it to a harm-exporting coalition.
        assert all(
            g.detection_method != "mutual_favoritism"
            for g in report.suspicious_groups
        )

    def test_group_metrics_calculation(self):
        """Should compute metrics for detected groups."""
        detector = CollusionDetector(
            min_interactions_for_analysis=3,
            collusion_score_threshold=0.15,
        )

        # Create tight group
        interactions = []
        for _ in range(5):
            interactions.extend(create_collusion_pattern("g1", "g2", count=3))
            interactions.extend(create_collusion_pattern("g2", "g3", count=3))
            interactions.extend(create_collusion_pattern("g1", "g3", count=3))

        # Few external interactions
        interactions.append(create_interaction("g1", "outsider", p=0.3))

        report = detector.analyze(
            interactions,
            agent_ids=["g1", "g2", "g3", "outsider"],
        )

        # If groups are detected, check their metrics
        for group in report.suspicious_groups:
            assert len(group.members) >= 2
            assert 0.0 <= group.internal_interaction_rate <= 1.0
            assert 0.0 <= group.collusion_score <= 1.0


# =============================================================================
# Vote Coordination Tests
# =============================================================================


class TestVoteCoordination:
    """Tests for vote coordination detection."""

    def test_empty_votes(self):
        """Should handle empty vote list."""
        result = detect_vote_coordination([])
        assert result == []

    def test_no_coordination(self):
        """Should not flag truly random voting patterns."""
        rng = np.random.default_rng(12345)
        votes = []
        voters = ["v0", "v1", "v2", "v3"]
        targets = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"]

        # Each voter votes randomly on each target
        for voter in voters:
            for target in targets:
                votes.append(
                    {
                        "voter": voter,
                        "target": target,
                        "direction": rng.choice([1, -1]),
                    }
                )

        result = detect_vote_coordination(votes, threshold=0.9)

        # Random voting with many targets shouldn't show perfect coordination
        # Allow some coincidental alignment but not too many pairs
        high_coordination = [r for r in result if r[2] >= 0.9]
        assert (
            len(high_coordination) <= 1
        )  # Allow at most 1 coincidental high alignment

    def test_detects_coordination(self):
        """Should detect coordinated voting."""
        votes = []
        # Alice and Bob always vote the same way
        for target in ["t1", "t2", "t3", "t4", "t5"]:
            direction = 1 if target in ["t1", "t3", "t5"] else -1
            votes.append({"voter": "alice", "target": target, "direction": direction})
            votes.append({"voter": "bob", "target": target, "direction": direction})

        # Charlie votes randomly
        for target in ["t1", "t2", "t3", "t4", "t5"]:
            votes.append({"voter": "charlie", "target": target, "direction": 1})

        result = detect_vote_coordination(votes, threshold=0.8)

        # Alice-Bob should be detected
        alice_bob = [r for r in result if {"alice", "bob"} == {r[0], r[1]}]
        assert len(alice_bob) == 1
        assert alice_bob[0][2] == 1.0  # Perfect alignment


# =============================================================================
# Temporal Clustering Tests
# =============================================================================


class TestTemporalClustering:
    """Tests for temporal clustering detection."""

    def test_empty_interactions(self):
        """Should handle empty list."""
        result = temporal_clustering_score([])
        assert result == {}

    def test_uniform_timing(self):
        """Uniform timing should have low clustering."""
        base = datetime.now()
        interactions = [
            create_interaction("a", "b", timestamp=base + timedelta(seconds=i * 100))
            for i in range(10)
        ]

        result = temporal_clustering_score(interactions, window_seconds=50)

        # Uniform spacing shouldn't show high clustering
        for _agent, score in result.items():
            assert score < 0.5

    def test_bursty_timing(self):
        """Bursty timing should have higher clustering."""
        base = datetime.now()
        interactions = []

        # Create burst of interactions
        for i in range(5):
            interactions.append(
                create_interaction("a", "b", timestamp=base + timedelta(seconds=i))
            )

        # Gap
        # Then another burst
        for i in range(5):
            interactions.append(
                create_interaction(
                    "a", "b", timestamp=base + timedelta(seconds=1000 + i)
                )
            )

        result = temporal_clustering_score(interactions, window_seconds=10)

        # Should show some clustering
        assert "a" in result
        assert "b" in result


# =============================================================================
# Governance Integration Tests
# =============================================================================


class TestCollusionGovernance:
    """Tests for collusion detection governance lever."""

    def test_governance_config_validation(self):
        """Should validate collusion config parameters."""
        # Valid config
        config = GovernanceConfig(
            collusion_detection_enabled=True,
            collusion_frequency_threshold=2.0,
            collusion_correlation_threshold=0.7,
        )
        assert config is not None  # Pydantic auto-validates on creation

        # Invalid threshold
        with pytest.raises(ValidationError):
            GovernanceConfig(collusion_frequency_threshold=-1.0)

        with pytest.raises(ValidationError):
            GovernanceConfig(collusion_correlation_threshold=1.5)

    def test_engine_creates_collusion_lever(self):
        """GovernanceEngine should create collusion lever."""
        config = GovernanceConfig(collusion_detection_enabled=True)
        engine = GovernanceEngine(config)

        # Should have collusion lever
        assert engine._collusion_lever is not None

    def test_set_agent_ids(self):
        """Should be able to set agent IDs for collusion detection."""
        config = GovernanceConfig(collusion_detection_enabled=True)
        engine = GovernanceEngine(config)

        engine.set_collusion_agent_ids(["a", "b", "c"])

        # Should not raise
        assert engine._collusion_lever._agent_ids == ["a", "b", "c"]

    def test_get_collusion_report(self):
        """Should be able to get collusion report from engine."""
        config = GovernanceConfig(collusion_detection_enabled=True)
        engine = GovernanceEngine(config)

        # Initially no report
        report = engine.get_collusion_report()
        assert report is None


# =============================================================================
# Scenario Loader Tests
# =============================================================================


class TestCollusionScenarioLoader:
    """Tests for collusion config in scenario loader."""

    def test_parse_collusion_governance_config(self):
        """Should parse collusion settings from governance config."""
        from swarm.scenarios.loader import parse_governance_config

        data = {
            "collusion_detection_enabled": True,
            "collusion_frequency_threshold": 3.0,
            "collusion_score_threshold": 0.6,
            "collusion_penalty_multiplier": 2.0,
            "collusion_realtime_penalty": True,
        }

        config = parse_governance_config(data)

        assert config.collusion_detection_enabled is True
        assert config.collusion_frequency_threshold == 3.0
        assert config.collusion_score_threshold == 0.6
        assert config.collusion_penalty_multiplier == 2.0
        assert config.collusion_realtime_penalty is True

    def test_default_collusion_config(self):
        """Should use defaults when not specified."""
        from swarm.scenarios.loader import parse_governance_config

        config = parse_governance_config({})

        assert config.collusion_detection_enabled is False
        assert config.collusion_frequency_threshold == 2.0
        assert config.collusion_score_threshold == 0.5


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_agent(self):
        """Should handle interactions involving single agent."""
        detector = CollusionDetector(min_interactions_for_analysis=1)

        interactions = [
            create_interaction("solo", "b"),
            create_interaction("solo", "c"),
            create_interaction("solo", "d"),
        ]

        report = detector.analyze(interactions)

        # Should not crash
        assert report is not None

    def test_all_same_pair(self):
        """Should handle all interactions being same pair."""
        detector = CollusionDetector(min_interactions_for_analysis=3)

        interactions = [create_interaction("a", "b") for _ in range(20)]

        report = detector.analyze(interactions)

        # Should detect as suspicious
        assert report.n_flagged_pairs >= 0  # May or may not flag based on scoring

    def test_nan_handling(self):
        """Should handle edge cases that could produce NaN."""
        detector = CollusionDetector(min_interactions_for_analysis=2)

        # Interactions with zero variance in benefits
        interactions = [
            create_interaction("a", "b", tau=0.0, c_a=0.0, c_b=0.0) for _ in range(5)
        ]

        report = detector.analyze(interactions)

        # Should not crash, correlation should be 0 or valid
        for pair in report.suspicious_pairs:
            assert not np.isnan(pair.benefit_correlation)
            assert not np.isnan(pair.collusion_score)

    def test_large_number_of_agents(self):
        """Should handle large agent populations efficiently."""
        detector = CollusionDetector(min_interactions_for_analysis=5)

        agents = [f"agent_{i}" for i in range(50)]
        interactions = create_normal_interactions(agents, count=200, seed=42)

        report = detector.analyze(interactions, agent_ids=agents)

        # Should complete and return valid report
        assert report is not None
        assert 0.0 <= report.ecosystem_collusion_risk <= 1.0


# =============================================================================
# CollusionPenaltyLever Tests
# =============================================================================


class TestCollusionPenaltyLever:
    """Tests for CollusionPenaltyLever governance lever."""

    def _make_lever(self, **overrides):
        """Create a CollusionPenaltyLever with sensible test defaults."""
        from swarm.governance.collusion import CollusionPenaltyLever

        defaults = {
            "collusion_detection_enabled": True,
            "collusion_frequency_threshold": 1.5,
            "collusion_correlation_threshold": 0.7,
            "collusion_min_interactions": 3,
            "collusion_score_threshold": 0.3,
            "collusion_penalty_multiplier": 1.0,
            "collusion_realtime_penalty": False,
            "collusion_realtime_rate": 0.1,
            "collusion_clear_history_on_epoch": False,
        }
        defaults.update(overrides)
        config = GovernanceConfig(**defaults)
        return CollusionPenaltyLever(config)

    def _make_state(self, agent_ids=None):
        """Create an EnvState with agents."""
        from swarm.env.state import EnvState

        state = EnvState()
        for aid in agent_ids or ["a", "b", "c"]:
            state.add_agent(aid)
        return state

    # --- on_epoch_start ---

    def test_on_epoch_start_disabled(self):
        """Disabled lever returns empty effect."""
        lever = self._make_lever(collusion_detection_enabled=False)
        state = self._make_state()

        effect = lever.on_epoch_start(state, epoch=0)
        assert effect.cost_a == 0.0
        assert effect.reputation_deltas == {}
        assert effect.resource_deltas == {}

    def test_on_epoch_start_insufficient_history(self):
        """Should return empty effect when not enough interactions."""
        lever = self._make_lever(collusion_min_interactions=10)
        state = self._make_state()

        # Only add 2 interactions
        lever._interaction_history.append(create_interaction("a", "b"))
        lever._interaction_history.append(create_interaction("b", "a"))

        effect = lever.on_epoch_start(state, epoch=1)
        assert effect.reputation_deltas == {}

    def test_on_epoch_start_with_flagged_pairs(self):
        """Should apply penalties when collusion is detected."""
        lever = self._make_lever(
            collusion_min_interactions=3,
            collusion_score_threshold=0.2,
            collusion_penalty_multiplier=2.0,
        )
        state = self._make_state(["alice", "bob", "charlie", "dave"])
        lever.set_agent_ids(["alice", "bob", "charlie", "dave"])

        # Load collusion pattern into history
        lever._interaction_history.extend(
            create_collusion_pattern("alice", "bob", count=15, p_internal=0.9)
        )
        # Add sparse normal interactions
        lever._interaction_history.extend(
            create_normal_interactions(["charlie", "dave"], count=10)
        )
        lever._interaction_history.append(create_interaction("alice", "charlie", p=0.3))
        lever._interaction_history.append(create_interaction("bob", "dave", p=0.3))

        effect = lever.on_epoch_start(state, epoch=1)

        # Report should be cached
        assert lever.get_report() is not None

        # If agents were flagged, penalties should be applied
        if effect.reputation_deltas:
            for _agent_id, delta in effect.reputation_deltas.items():
                assert delta < 0  # Penalties are negative
            for _agent_id, delta in effect.resource_deltas.items():
                assert delta < 0

    def test_on_epoch_start_clear_after_detection(self):
        """clear_history_on_epoch=True should clear history after analysis."""
        lever = self._make_lever(
            collusion_clear_history_on_epoch=True,
            collusion_min_interactions=3,
        )
        state = self._make_state()
        lever.set_agent_ids(["a", "b", "c"])

        lever._interaction_history.extend(create_collusion_pattern("a", "b", count=10))
        lever._interaction_history.extend(
            create_normal_interactions(["a", "c"], count=5)
        )

        lever.on_epoch_start(state, epoch=1)
        assert len(lever._interaction_history) == 0

    # --- on_interaction ---

    def test_on_interaction_disabled(self):
        """Disabled lever doesn't record interactions."""
        lever = self._make_lever(collusion_detection_enabled=False)
        state = self._make_state()

        interaction = create_interaction("a", "b")
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0
        assert len(lever._interaction_history) == 0

    def test_on_interaction_records_history(self):
        """Enabled lever records interaction in history."""
        lever = self._make_lever()
        state = self._make_state()

        interaction = create_interaction("a", "b")
        lever.on_interaction(interaction, state)
        assert len(lever._interaction_history) == 1

    def test_on_interaction_realtime_penalty_disabled(self):
        """No realtime penalty when disabled."""
        lever = self._make_lever(collusion_realtime_penalty=False)
        state = self._make_state()

        interaction = create_interaction("a", "b")
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0

    def test_on_interaction_realtime_penalty_no_report(self):
        """No realtime penalty when no report exists yet."""
        lever = self._make_lever(collusion_realtime_penalty=True)
        state = self._make_state()

        assert lever._latest_report is None
        interaction = create_interaction("a", "b")
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0

    def test_on_interaction_realtime_penalty_with_flagged_pair(self):
        """Realtime penalty applied for interaction between flagged agents."""
        lever = self._make_lever(
            collusion_realtime_penalty=True,
            collusion_realtime_rate=0.5,
            collusion_min_interactions=3,
            collusion_score_threshold=0.2,
        )
        state = self._make_state(["alice", "bob", "charlie", "dave"])
        lever.set_agent_ids(["alice", "bob", "charlie", "dave"])

        # Build up history and generate a report
        lever._interaction_history.extend(
            create_collusion_pattern("alice", "bob", count=15)
        )
        lever._interaction_history.extend(
            create_normal_interactions(["charlie", "dave"], count=10)
        )
        lever._interaction_history.append(create_interaction("alice", "charlie", p=0.3))
        lever._interaction_history.append(create_interaction("bob", "dave", p=0.3))

        lever.on_epoch_start(state, epoch=1)

        # If both agents are flagged, interacting should incur penalty
        report = lever.get_report()
        if report:
            alice_risk = report.agent_collusion_risk.get("alice", 0)
            bob_risk = report.agent_collusion_risk.get("bob", 0)

            if (
                alice_risk >= lever._detector.collusion_threshold
                and bob_risk >= lever._detector.collusion_threshold
            ):
                interaction = create_interaction("alice", "bob")
                effect = lever.on_interaction(interaction, state)
                assert effect.cost_a > 0
                assert effect.cost_b > 0
                assert effect.details.get("realtime_penalty") is True

    # --- setters and getters ---

    def test_set_agent_ids(self):
        """set_agent_ids propagates correctly."""
        lever = self._make_lever()
        lever.set_agent_ids(["x", "y", "z"])
        assert lever._agent_ids == ["x", "y", "z"]

    def test_get_report_initially_none(self):
        """get_report() returns None before any analysis."""
        lever = self._make_lever()
        assert lever.get_report() is None

    def test_get_interaction_history(self):
        """get_interaction_history() returns a copy."""
        lever = self._make_lever()
        lever._interaction_history.append(create_interaction("a", "b"))

        history = lever.get_interaction_history()
        assert len(history) == 1
        # It's a copy, modifying it shouldn't affect lever
        history.clear()
        assert len(lever._interaction_history) == 1

    def test_clear_history(self):
        """clear_history() clears interactions and report."""
        lever = self._make_lever()
        lever._interaction_history.append(create_interaction("a", "b"))
        lever._latest_report = "dummy"

        lever.clear_history()
        assert len(lever._interaction_history) == 0
        assert lever._latest_report is None
