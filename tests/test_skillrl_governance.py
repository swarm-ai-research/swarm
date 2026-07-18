"""Tests for SkillRL governance integration: RefinementProposal and Two-Gate evaluation."""

import pytest

from swarm.governance.config import GovernanceConfig
from swarm.governance.refinement_proposal import RefinementProposal
from swarm.governance.self_modification import (
    ModificationState,
    RiskTier,
    SelfModificationLever,
)
from swarm.skills.evolution import EvolutionConfig, SkillEvolutionEngine
from swarm.skills.library import SkillLibrary
from swarm.skills.model import Skill, SkillDomain, SkillType

# ---------------------------------------------------------------------------
# RefinementProposal tests
# ---------------------------------------------------------------------------


class TestRefinementProposal:
    """Test RefinementProposal creation and properties."""

    def test_default_initialization(self):
        """Test that RefinementProposal can be created with defaults."""
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
        )
        assert proposal.skill_id == "skill_1"
        assert proposal.agent_id == "agent_1"
        assert proposal.complexity_weight >= 1.0
        assert proposal.state == ModificationState.PROPOSED

    def test_compute_hash_deterministic(self):
        """Test that compute_hash is deterministic."""
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            refined_version=2,
            timestamp=1000.0,
        )
        h1 = proposal.compute_hash()
        h2 = proposal.compute_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_strategy_skill_weight(self):
        """Test that STRATEGY skills get correct base weight."""
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            skill_type="STRATEGY",
        )
        assert proposal.complexity_weight == 1.0

    def test_lesson_skill_weight(self):
        """Test that LESSON skills get correct base weight."""
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            skill_type="LESSON",
        )
        assert proposal.complexity_weight == 0.8

    def test_composite_skill_weight(self):
        """Test that COMPOSITE skills get correct base weight."""
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            skill_type="COMPOSITE",
        )
        assert proposal.complexity_weight == 1.5

    def test_large_effect_delta_increases_weight(self):
        """Test that large effect changes increase complexity weight."""
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            skill_type="STRATEGY",
            effect_delta={
                "acceptance_threshold_delta": (-0.05, 0.15),  # delta = 0.2
            },
        )
        # Base weight 1.0 * 1.2 = 1.2
        assert proposal.complexity_weight > 1.0


# ---------------------------------------------------------------------------
# Risk classification tests
# ---------------------------------------------------------------------------


class TestRefinementRiskClassification:
    """Test risk tier classification for refinement proposals."""

    def test_small_changes_are_low_risk(self):
        """Test that small changes are classified as LOW risk."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            condition_delta={
                "min_p": (0.4, 0.42),
                "max_p": (0.6, 0.58),
            },
            effect_delta={
                "acceptance_threshold_delta": (-0.05, -0.04),
            },
        )
        tier = lever._classify_refinement_risk(proposal)
        assert tier == RiskTier.LOW

    def test_medium_condition_changes_are_medium_risk(self):
        """Test that moderate condition changes are MEDIUM risk."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            condition_delta={
                "min_p": (0.3, 0.5),  # delta = 0.2 > 0.15
            },
            effect_delta={
                "acceptance_threshold_delta": (-0.05, -0.02),  # delta = 0.03 < 0.05
            },
        )
        tier = lever._classify_refinement_risk(proposal)
        # condition_delta > 0.15 but effect_delta < 0.05 -> LOW
        assert tier == RiskTier.LOW

    def test_medium_effect_changes_are_high_risk(self):
        """Test that medium effect changes are HIGH risk."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            effect_delta={
                "acceptance_threshold_delta": (-0.05, 0.08),  # delta = 0.13 > 0.1
            },
        )
        tier = lever._classify_refinement_risk(proposal)
        assert tier == RiskTier.HIGH

    def test_large_effect_changes_are_critical(self):
        """Test that large effect changes are CRITICAL risk."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            effect_delta={
                "acceptance_threshold_delta": (-0.05, 0.20),  # delta = 0.25 > 0.2
            },
        )
        tier = lever._classify_refinement_risk(proposal)
        assert tier == RiskTier.CRITICAL

    def test_composite_skills_are_critical(self):
        """Test that COMPOSITE skills are always CRITICAL."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            skill_type="COMPOSITE",
            effect_delta={
                "acceptance_threshold_delta": (-0.05, -0.04),  # small change
            },
        )
        tier = lever._classify_refinement_risk(proposal)
        assert tier == RiskTier.CRITICAL


# ---------------------------------------------------------------------------
# Tau gate tests
# ---------------------------------------------------------------------------


class TestTauGateRefinement:
    """Test tau gate evaluation for refinements."""

    def test_safe_direction_and_justified_passes(self):
        """Test that refinements with safe direction and justification pass."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            effect_delta={
                "acceptance_threshold_delta": (-0.05, 0.05),  # more conservative
            },
            perf_before={
                "success_rate": 0.3,  # low success rate justifies refinement
            },
        )
        proposal.risk_tier = lever._classify_refinement_risk(proposal)

        tau_result = lever._evaluate_tau_gate_for_refinement(
            proposal, {}, {}
        )

        assert tau_result.passed
        assert tau_result.value == 1.0  # Both checks pass

    def test_risky_direction_fails(self):
        """Test that refinements moving in risky direction fail."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            effect_delta={
                "acceptance_threshold_delta": (0.05, -0.05),  # less conservative
            },
            perf_before={
                "success_rate": 0.3,
            },
        )
        proposal.risk_tier = lever._classify_refinement_risk(proposal)

        tau_result = lever._evaluate_tau_gate_for_refinement(
            proposal, {}, {}
        )

        # Risky direction but justified -> tau = 0.5
        # For LOW risk tier, tau_min = -0.1, so it should pass
        assert tau_result.value == 0.5
        assert tau_result.passed  # LOW tier has tau_min = -0.1

    def test_unjustified_refinement_warning(self):
        """Test that refining well-performing skills gets warning."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            effect_delta={
                "acceptance_threshold_delta": (-0.05, 0.05),
            },
            perf_before={
                "success_rate": 0.8,  # high success rate
            },
        )
        proposal.risk_tier = lever._classify_refinement_risk(proposal)

        tau_result = lever._evaluate_tau_gate_for_refinement(
            proposal, {}, {}
        )

        assert "warning" in tau_result.details
        assert "well-performing" in tau_result.details["warning"].lower()

    def test_no_effect_change_is_neutral(self):
        """Test that refinements without effect changes are neutral."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            condition_delta={
                "min_p": (0.4, 0.45),
            },
            effect_delta={},  # No effect changes
            perf_before={
                "success_rate": 0.3,
            },
        )
        proposal.risk_tier = lever._classify_refinement_risk(proposal)

        tau_result = lever._evaluate_tau_gate_for_refinement(
            proposal, {}, {}
        )

        # No acceptance_threshold_delta -> neutral direction (True)
        # Low success rate -> justified (True)
        assert tau_result.value == 1.0
        assert tau_result.passed


# ---------------------------------------------------------------------------
# K_max gate tests
# ---------------------------------------------------------------------------


class TestKMaxGateRefinement:
    """Test K_max gate evaluation for refinements."""

    def test_under_budget_passes(self):
        """Test that refinements under budget pass K_max gate."""
        lever = SelfModificationLever(GovernanceConfig())
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            skill_type="STRATEGY",
        )
        proposal.risk_tier = lever._classify_refinement_risk(proposal)
        proposal.complexity_weight = 1.0

        # Agent has used 5.0, adding 1.0 = 6.0 < 20.0 (LOW tier limit)
        baseline = {}
        uncertainties = {}

        approved, tau_result, k_max_result = lever.evaluate_refinement(
            proposal, baseline, uncertainties
        )

        assert k_max_result.passed
        assert approved  # Both gates pass

    def test_over_budget_fails(self):
        """Test that refinements over budget fail K_max gate."""
        lever = SelfModificationLever(GovernanceConfig())

        # Pre-fill agent budget to near limit
        agent_id = "agent_1"
        lever._agent_budgets[agent_id] = 19.5  # LOW tier limit is 20.0

        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id=agent_id,
            skill_type="STRATEGY",
        )
        proposal.risk_tier = lever._classify_refinement_risk(proposal)
        proposal.complexity_weight = 1.0  # Would exceed 20.0

        baseline = {}
        uncertainties = {}

        approved, tau_result, k_max_result = lever.evaluate_refinement(
            proposal, baseline, uncertainties
        )

        assert not k_max_result.passed
        assert not approved

    def test_budget_accumulates(self):
        """Test that budget accumulates across refinements."""
        lever = SelfModificationLever(GovernanceConfig())
        agent_id = "agent_1"

        # Submit three refinements
        for i in range(3):
            proposal = RefinementProposal(
                skill_id=f"skill_{i}",
                agent_id=agent_id,
                skill_type="STRATEGY",
            )
            proposal.risk_tier = lever._classify_refinement_risk(proposal)
            proposal.complexity_weight = 1.0

            approved, _, _ = lever.evaluate_refinement(proposal, {}, {})
            assert approved

        # Budget should be 3.0
        assert lever._agent_budgets[agent_id] == 3.0

    def test_composite_refinement_higher_cost(self):
        """Test that COMPOSITE refinements cost more."""
        lever = SelfModificationLever(GovernanceConfig())

        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            skill_type="COMPOSITE",
        )
        # COMPOSITE skills are CRITICAL risk, which has K_max = -1.0 (always fail)
        proposal.risk_tier = lever._classify_refinement_risk(proposal)

        approved, tau_result, k_max_result = lever.evaluate_refinement(
            proposal, {}, {}
        )

        # COMPOSITE -> CRITICAL -> K_max = -1.0 -> always fail
        assert not approved
        assert not k_max_result.passed


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSkillRLGovernanceIntegration:
    """Test integration of SkillRL with governance."""

    def test_refine_skills_with_governance_collects_proposals(self):
        """Test that refine_skills_with_governance collects proposals."""
        config = EvolutionConfig(
            recursive_evolution_enabled=True,
            refinement_min_invocations=3,
            refinement_success_threshold=0.4,
        )
        engine = SkillEvolutionEngine(config=config)
        library = SkillLibrary(owner_id="agent_1")

        # Add an under-performing skill
        skill = Skill(
            name="test_skill",
            skill_type=SkillType.STRATEGY,
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.3, "max_p": 0.7},
            effect={"acceptance_threshold_delta": -0.05},
        )
        library.add_skill(skill)

        # Record invocations with poor performance
        for _ in range(5):
            library.record_invocation(skill.skill_id, payoff=-0.5, p=0.4)

        # Call refine_skills_with_governance
        refined_ids, proposals = engine.refine_skills_with_governance(
            library, "agent_1", governance_lever=None
        )

        # Should have collected one proposal
        assert len(proposals) == 1
        assert proposals[0].skill_id == skill.skill_id

        # Should have applied refinement (no governance)
        assert len(refined_ids) == 1

    def test_refine_skills_with_governance_no_proposals_when_disabled(self):
        """Test that no proposals are generated when recursive evolution is disabled."""
        config = EvolutionConfig(recursive_evolution_enabled=False)
        engine = SkillEvolutionEngine(config=config)
        library = SkillLibrary(owner_id="agent_1")

        refined_ids, proposals = engine.refine_skills_with_governance(
            library, "agent_1", governance_lever=None
        )

        assert len(proposals) == 0
        assert len(refined_ids) == 0

    def test_governance_lever_blocks_risky_refinement(self):
        """Test that governance lever blocks risky refinements."""
        lever = SelfModificationLever(GovernanceConfig())

        # Create a CRITICAL risk refinement
        proposal = RefinementProposal(
            skill_id="skill_1",
            agent_id="agent_1",
            skill_type="COMPOSITE",  # COMPOSITE -> CRITICAL
            effect_delta={
                "acceptance_threshold_delta": (-0.05, 0.25),  # Large change
            },
            perf_before={"success_rate": 0.3},
        )

        approved, tau_result, k_max_result = lever.evaluate_refinement(
            proposal, {}, {}
        )

        # CRITICAL tier should fail
        assert not approved

    def test_backward_compatibility_no_governance(self):
        """Test backward compatibility when governance_lever is None."""
        config = EvolutionConfig(
            recursive_evolution_enabled=True,
            refinement_min_invocations=2,
            refinement_success_threshold=0.4,
        )
        engine = SkillEvolutionEngine(config=config)
        library = SkillLibrary(owner_id="agent_1")

        # Add under-performing skill
        skill = Skill(
            name="test_skill",
            skill_type=SkillType.STRATEGY,
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.3, "max_p": 0.7},
            effect={"acceptance_threshold_delta": -0.05},
        )
        library.add_skill(skill)

        # Record poor performance
        for _ in range(3):
            library.record_invocation(skill.skill_id, payoff=-0.5, p=0.4)

        # Call with governance_lever=None (backward compatible)
        refined_ids, proposals = engine.refine_skills_with_governance(
            library, "agent_1", governance_lever=None
        )

        # Should still apply refinement directly
        assert len(refined_ids) == 1
        assert len(proposals) == 1

        # Skill should be refined
        refined_skill = library.get_skill(skill.skill_id)
        assert "refined" in refined_skill.tags
        assert refined_skill.version == 2


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestGovernanceWithoutEnvState:
    """Regression for beads 7209: a lever with no env_state must still gate.

    Previously _maybe_refine required BOTH a lever and env_state to run
    governance; with env_state=None every proposal was silently auto-
    approved past an attached lever — a governance bypass.
    """

    def _agent_with_rejecting_lever(self):
        from swarm.agents.skillrl_agent import SkillRLAgent

        agent = SkillRLAgent(
            agent_id="rl_alice",
            config={"refinement_interval": 1, "refinement_min_invocations": 1},
        )

        calls = []

        class _RejectingLever:
            def evaluate_refinement(self, proposal, baseline, uncertainties):
                calls.append((proposal, dict(baseline)))

                class _Gate:
                    passed = False

                return False, _Gate(), _Gate()

        agent.governance_lever = _RejectingLever()
        return agent, calls

    def _seed_refinable_skill(self, agent):
        from swarm.skills.model import Skill, SkillDomain, SkillTier, SkillType

        skill = Skill(
            skill_id="s1",
            name="s1",
            skill_type=SkillType.STRATEGY,
            tier=SkillTier.TASK_SPECIFIC,
            domain=SkillDomain.INTERACTION,
            condition={"min_p": 0.2, "max_p": 0.9},
            effect={"acceptance_threshold_delta": 0.0},
        )
        agent.skill_library.add_skill(skill)
        perf = agent.skill_library.get_performance("s1")
        # Eligible for refinement: invoked, low success rate.
        perf.invocations = 5
        perf.successes = 0
        return skill

    def test_lever_is_consulted_and_rejection_blocks_apply(self):
        agent, calls = self._agent_with_rejecting_lever()
        skill = self._seed_refinable_skill(agent)
        before = (dict(skill.condition), dict(skill.effect), skill.version)

        agent._maybe_refine(epoch=1, env_state=None)

        assert calls, "governance lever was never consulted (bypass)"
        # Default baselines used when env_state is unavailable.
        assert calls[0][1] == {"payoff": 0.0, "reputation": 0.5}
        after = (dict(skill.condition), dict(skill.effect), skill.version)
        assert after == before, "rejected refinement was applied anyway"
