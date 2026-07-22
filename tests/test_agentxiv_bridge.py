"""Tests for the AgentXiv bridge (annotator, scenario_gen, validation)."""

import pytest
import yaml

from swarm.research.annotator import (
    PaperAnnotation,
    PaperAnnotator,
    RiskProfile,
    VerifiableClaim,
)
from swarm.research.platforms import Paper
from swarm.research.scenario_gen import ScenarioGenerator
from swarm.research.validation import ValidationResult, ValidationWorkflow

# ---------------------------------------------------------------------------
# RiskProfile tests
# ---------------------------------------------------------------------------


class TestRiskProfile:
    def test_to_dict_roundtrip(self):
        rp = RiskProfile(
            interaction_density="high",
            failure_modes=["collusion", "deception"],
            assumptions=["assumes_honest_majority"],
        )
        data = rp.to_dict()
        restored = RiskProfile.from_dict(data)
        assert restored.interaction_density == "high"
        assert restored.failure_modes == ["collusion", "deception"]
        assert restored.assumptions == ["assumes_honest_majority"]

    def test_defaults(self):
        rp = RiskProfile()
        assert rp.interaction_density == "medium"
        assert rp.failure_modes == []
        assert rp.assumptions == []


# ---------------------------------------------------------------------------
# VerifiableClaim tests
# ---------------------------------------------------------------------------


class TestVerifiableClaim:
    def test_to_dict_roundtrip(self):
        tc = VerifiableClaim(
            claim="governance reduces toxicity",
            testable=True,
            metric="toxicity_rate",
            expected="negative",
            parameters={"threshold": 0.3},
        )
        data = tc.to_dict()
        restored = VerifiableClaim.from_dict(data)
        assert restored.claim == "governance reduces toxicity"
        assert restored.metric == "toxicity_rate"
        assert restored.expected == "negative"
        assert restored.parameters == {"threshold": 0.3}


# ---------------------------------------------------------------------------
# PaperAnnotation tests
# ---------------------------------------------------------------------------


class TestPaperAnnotation:
    def test_to_dict_roundtrip(self):
        ann = PaperAnnotation(
            paper_id="test-123",
            arxiv_id="2401.00001",
            title="Test Paper",
            risk_profile=RiskProfile(
                interaction_density="high",
                failure_modes=["collusion"],
            ),
            claims=[
                VerifiableClaim(claim="welfare increases", metric="total_welfare", expected="positive"),
            ],
            swarm_scenarios=["baseline"],
        )
        data = ann.to_dict()
        restored = PaperAnnotation.from_dict(data)
        assert restored.paper_id == "test-123"
        assert restored.title == "Test Paper"
        assert restored.risk_profile.interaction_density == "high"
        assert len(restored.claims) == 1
        assert restored.claims[0].metric == "total_welfare"

    def test_yaml_roundtrip(self):
        ann = PaperAnnotation(
            paper_id="yaml-test",
            title="YAML Round Trip",
            risk_profile=RiskProfile(failure_modes=["deception"]),
        )
        yaml_str = ann.to_yaml()
        restored = PaperAnnotation.from_yaml(yaml_str)
        assert restored.paper_id == "yaml-test"
        assert restored.risk_profile.failure_modes == ["deception"]


# ---------------------------------------------------------------------------
# PaperAnnotator tests
# ---------------------------------------------------------------------------


class TestPaperAnnotator:
    def test_estimate_interaction_density_high(self):
        annotator = PaperAnnotator()
        text = " ".join(["agent interaction exchange trade"] * 10)
        result = annotator._estimate_interaction_density(text)
        assert result == "high"

    def test_estimate_interaction_density_low(self):
        annotator = PaperAnnotator()
        text = "this paper studies mathematics and proofs"
        result = annotator._estimate_interaction_density(text)
        assert result == "low"

    def test_estimate_interaction_density_medium(self):
        annotator = PaperAnnotator()
        # 8-19 keyword occurrences = medium
        text = "agent agent agent interaction interaction exchange trade communicate cooperat"
        result = annotator._estimate_interaction_density(text)
        assert result == "medium"

    def test_detect_failure_modes(self):
        annotator = PaperAnnotator()
        text = "we study collusion between agents and deceptive behavior"
        modes = annotator._detect_failure_modes(text)
        assert "collusion" in modes
        assert "deception" in modes

    def test_detect_failure_modes_empty(self):
        annotator = PaperAnnotator()
        text = "this paper studies benign cooperative systems"
        modes = annotator._detect_failure_modes(text)
        assert modes == []

    def test_detect_assumptions(self):
        annotator = PaperAnnotator()
        text = "we assume an honest majority of participants in a fixed population"
        assumptions = annotator._detect_assumptions(text)
        assert "assumes_honest_majority" in assumptions
        assert "fixed_population" in assumptions

    def test_extract_claims_basic(self):
        annotator = PaperAnnotator()
        paper = Paper(
            abstract="We show that governance reduces toxic interactions significantly.",
            source="",
        )
        claims = annotator._extract_claims(paper)
        assert len(claims) >= 1
        assert any("governance" in c.claim.lower() for c in claims)

    def test_extract_claims_find_pattern(self):
        annotator = PaperAnnotator()
        paper = Paper(
            abstract="We find that welfare improves under cooperative conditions.",
            source="",
        )
        claims = annotator._extract_claims(paper)
        assert len(claims) >= 1

    def test_annotate_paper(self):
        annotator = PaperAnnotator()
        paper = Paper(
            paper_id="test-paper",
            title="Multi-Agent Collusion Study",
            abstract=(
                "We study collusion in multi-agent systems with many agent "
                "interactions. We show that governance reduces toxic behavior."
            ),
            source="",
        )
        annotation = annotator.annotate_paper(paper)
        assert annotation.paper_id == "test-paper"
        assert annotation.title == "Multi-Agent Collusion Study"
        assert "collusion" in annotation.risk_profile.failure_modes

    def test_annotate_missing_paper(self):
        """annotate() with no platforms returns empty annotation."""
        annotator = PaperAnnotator(platforms=[])
        annotation = annotator.annotate("nonexistent")
        assert annotation.paper_id == "nonexistent"
        assert annotation.title == ""


# ---------------------------------------------------------------------------
# ScenarioGenerator tests
# ---------------------------------------------------------------------------


class TestScenarioGenerator:
    def test_from_paper_has_required_keys(self):
        gen = ScenarioGenerator()
        annotation = PaperAnnotation(
            paper_id="test",
            title="Test",
            risk_profile=RiskProfile(
                interaction_density="medium",
                failure_modes=["collusion"],
            ),
        )
        scenario = gen.from_paper(annotation)

        # Check all required top-level keys from baseline.yaml
        required_keys = {
            "scenario_id",
            "description",
            "motif",
            "agents",
            "governance",
            "simulation",
            "rate_limits",
            "payoff",
            "success_criteria",
            "outputs",
        }
        assert required_keys.issubset(scenario.keys())

    def test_from_paper_valid_yaml(self):
        gen = ScenarioGenerator()
        annotation = PaperAnnotation(
            paper_id="yaml-test",
            title="YAML Test",
            risk_profile=RiskProfile(interaction_density="low"),
        )
        yaml_str = gen.from_paper_to_yaml(annotation)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["scenario_id"] == "agentxiv_yaml-test"

    def test_map_agent_composition_collusion(self):
        gen = ScenarioGenerator()
        annotation = PaperAnnotation(
            risk_profile=RiskProfile(
                interaction_density="medium",
                failure_modes=["collusion"],
            ),
        )
        agents = gen._map_agent_composition(annotation)
        types = {a["type"] for a in agents}
        assert "honest" in types
        assert "deceptive" in types

    def test_map_agent_composition_adverse_selection(self):
        gen = ScenarioGenerator()
        annotation = PaperAnnotation(
            risk_profile=RiskProfile(
                interaction_density="medium",
                failure_modes=["adverse_selection"],
            ),
        )
        agents = gen._map_agent_composition(annotation)
        types = {a["type"] for a in agents}
        assert "opportunistic" in types

    def test_map_agent_composition_rejects_unknown_density(self):
        gen = ScenarioGenerator()
        annotation = PaperAnnotation(
            risk_profile=RiskProfile(interaction_density="hgih"),
        )
        with pytest.raises(ValueError, match="interaction_density"):
            gen._map_agent_composition(annotation)

    def test_map_agent_composition_high_density_scales(self):
        gen = ScenarioGenerator()
        ann_low = PaperAnnotation(
            risk_profile=RiskProfile(
                interaction_density="low",
                failure_modes=["collusion"],
            ),
        )
        ann_high = PaperAnnotation(
            risk_profile=RiskProfile(
                interaction_density="high",
                failure_modes=["collusion"],
            ),
        )
        agents_low = gen._map_agent_composition(ann_low)
        agents_high = gen._map_agent_composition(ann_high)
        total_low = sum(a["count"] for a in agents_low)
        total_high = sum(a["count"] for a in agents_high)
        assert total_high > total_low

    def test_map_governance_minimal(self):
        gen = ScenarioGenerator()
        annotation = PaperAnnotation(
            risk_profile=RiskProfile(assumptions=["assumes_honest_majority"]),
        )
        governance = gen._map_governance(annotation)
        assert governance["audit_enabled"] is False
        assert governance["circuit_breaker_enabled"] is False

    def test_map_governance_with_collusion(self):
        gen = ScenarioGenerator()
        annotation = PaperAnnotation(
            risk_profile=RiskProfile(failure_modes=["collusion"]),
        )
        governance = gen._map_governance(annotation)
        assert governance["audit_enabled"] is True

    def test_map_simulation_params_density(self):
        gen = ScenarioGenerator()
        for density, expected_steps in [("low", 5), ("medium", 10), ("high", 15)]:
            annotation = PaperAnnotation(
                risk_profile=RiskProfile(interaction_density=density),
            )
            params = gen._map_simulation_params(annotation)
            assert params["steps_per_epoch"] == expected_steps

    def test_payoff_defaults_match_baseline(self):
        gen = ScenarioGenerator()
        annotation = PaperAnnotation()
        payoff = gen._map_payoff_params(annotation)
        assert payoff["s_plus"] == 2.0
        assert payoff["s_minus"] == 1.0
        assert payoff["h"] == 2.0
        assert payoff["theta"] == 0.5


# ---------------------------------------------------------------------------
# ValidationWorkflow tests
# ---------------------------------------------------------------------------


class TestValidationWorkflow:
    def _mock_simulation(self, scenario: dict, seed: int) -> dict:
        """Mock simulation that returns predictable metrics."""
        return {
            "toxicity_rate": 0.1 * (seed - 41),
            "quality_gap": -0.05,
            "total_welfare": 5.0 + seed * 0.1,
        }

    def test_compare_claims_positive(self):
        workflow = ValidationWorkflow(simulation_fn=self._mock_simulation)
        claims = [
            VerifiableClaim(
                claim="welfare increases",
                testable=True,
                metric="total_welfare",
                expected="positive",
            ),
        ]
        results = [{"total_welfare": 5.0}, {"total_welfare": 6.0}]
        claim_results = workflow._compare_claims(claims, results)
        assert len(claim_results) == 1
        assert claim_results[0].matched is True
        assert claim_results[0].mean_value == pytest.approx(5.5)

    def test_compare_claims_negative(self):
        workflow = ValidationWorkflow()
        claims = [
            VerifiableClaim(
                claim="toxicity decreases",
                testable=True,
                metric="toxicity_rate",
                expected="negative",
            ),
        ]
        results = [{"toxicity_rate": -0.3}, {"toxicity_rate": -0.1}]
        claim_results = workflow._compare_claims(claims, results)
        assert claim_results[0].matched is True

    def test_compare_claims_untestable(self):
        workflow = ValidationWorkflow()
        claims = [
            VerifiableClaim(claim="vague claim", testable=False, metric=""),
        ]
        results = [{"toxicity_rate": 0.1}]
        claim_results = workflow._compare_claims(claims, results)
        assert claim_results[0].matched is False

    def test_compare_claims_missing_metric(self):
        workflow = ValidationWorkflow()
        claims = [
            VerifiableClaim(
                claim="unmeasured",
                testable=True,
                metric="nonexistent_metric",
                expected="positive",
            ),
        ]
        results = [{"toxicity_rate": 0.1}]
        claim_results = workflow._compare_claims(claims, results)
        assert claim_results[0].matched is False

    def test_validate_from_annotation_full(self):
        def sim_fn(scenario, seed):
            return {"total_welfare": 5.0, "toxicity_rate": -0.2}

        workflow = ValidationWorkflow(
            generator=ScenarioGenerator(),
            simulation_fn=sim_fn,
            num_runs=3,
        )
        annotation = PaperAnnotation(
            paper_id="test",
            title="Test Paper",
            risk_profile=RiskProfile(interaction_density="medium"),
            claims=[
                VerifiableClaim(
                    claim="welfare is positive",
                    testable=True,
                    metric="total_welfare",
                    expected="positive",
                ),
            ],
        )
        result = workflow.validate_from_annotation(annotation)
        assert result.paper_id == "test"
        assert len(result.run_results) == 3
        assert result.overall_verdict == "supported"
        assert "MATCHED" in result.report

    def test_validate_no_simulation_fn(self):
        workflow = ValidationWorkflow(
            generator=ScenarioGenerator(),
            simulation_fn=None,
            num_runs=3,
        )
        annotation = PaperAnnotation(
            paper_id="test",
            claims=[
                VerifiableClaim(
                    claim="claim",
                    testable=True,
                    metric="toxicity_rate",
                    expected="negative",
                ),
            ],
        )
        result = workflow.validate_from_annotation(annotation)
        assert result.overall_verdict == "not_supported"

    def test_validate_no_testable_claims(self):
        def sim_fn(scenario, seed):
            return {"total_welfare": 5.0}

        workflow = ValidationWorkflow(
            generator=ScenarioGenerator(),
            simulation_fn=sim_fn,
            num_runs=1,
        )
        annotation = PaperAnnotation(paper_id="test", claims=[])
        result = workflow.validate_from_annotation(annotation)
        assert result.overall_verdict == "no_testable_claims"

    def test_validation_result_to_dict(self):
        result = ValidationResult(
            paper_id="test",
            overall_verdict="supported",
            report="Test report",
        )
        data = result.to_dict()
        assert data["paper_id"] == "test"
        assert data["overall_verdict"] == "supported"
