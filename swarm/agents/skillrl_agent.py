"""SkillRL agent: full implementation of recursive skill-augmented RL.

Models the approach from:
    Xia, P. et al. (2026). SkillRL: Evolving Agents via Recursive
    Skill-Augmented Reinforcement Learning. arXiv:2602.08234 [cs.LG].

Key features beyond the base SkillEvolvingMixin:
- Hierarchical SkillBank with tiered retrieval (GENERAL / TASK_SPECIFIC)
- GRPO-style group-relative advantage for adaptive extraction thresholds
- Recursive skill evolution: refine under-performing skills by tightening
  conditions and adjusting effects based on validation-failure analysis
- Automatic tier promotion: task-specific skills that prove broadly useful
  are promoted to the GENERAL tier
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType
from swarm.skills.evolution import EvolutionConfig, SkillEvolutionEngine
from swarm.skills.library import SkillLibrary, SkillLibraryConfig
from swarm.skills.model import Skill, SkillDomain, SkillTier, SkillType

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig
    from swarm.env.state import EnvState
    from swarm.models.interaction import SoftInteraction


@dataclass
class SkillRLConfig:
    """Configuration for the SkillRL agent.

    Bundles the agent-level hyperparameters that are separate from the
    skill evolution engine's own config.
    """

    # Base acceptance threshold before skill modulation
    acceptance_threshold: float = 0.4

    # Weight given to trust signals in acceptance decision
    trust_weight: float = 0.3

    # Probabilities for spontaneous actions
    post_probability: float = 0.2
    vote_probability: float = 0.3
    interact_probability: float = 0.5

    # How many recent payoffs to track for policy gradient
    policy_window: int = 30

    # Exploration rate for skill selection (epsilon-greedy)
    skill_exploration_rate: float = 0.15

    # Epoch interval for running recursive refinement
    refinement_interval: int = 2

    # Epoch interval for running tier promotion
    promotion_interval: int = 5


@dataclass
class PolicyGradientState:
    """Tracks running statistics for GRPO-style policy updates."""

    payoff_history: deque = field(
        default_factory=lambda: deque(maxlen=30),
    )
    # Per-skill payoff tracking, bounded per-skill to match payoff_history
    skill_payoffs: Dict[str, deque] = field(default_factory=dict)
    # Cap the number of tracked skill IDs to prevent unbounded key growth
    _max_tracked_skills: int = 100

    @property
    def baseline(self) -> float:
        """Running mean payoff (group baseline)."""
        if not self.payoff_history:
            return 0.0
        return float(sum(self.payoff_history) / len(self.payoff_history))

    def advantage(self, payoff: float) -> float:
        """Advantage = payoff - baseline."""
        return payoff - self.baseline

    def record(self, payoff: float, skill_id: Optional[str] = None) -> None:
        self.payoff_history.append(payoff)
        if skill_id:
            if skill_id not in self.skill_payoffs:
                # Evict oldest-tracked skill if at capacity
                if len(self.skill_payoffs) >= self._max_tracked_skills:
                    oldest_key = next(iter(self.skill_payoffs))
                    del self.skill_payoffs[oldest_key]
                self.skill_payoffs[skill_id] = deque(maxlen=30)
            self.skill_payoffs[skill_id].append(payoff)


class SkillRLAgent(BaseAgent):
    """Agent implementing the full SkillRL pipeline.

    Combines:
    1. Hierarchical SkillBank (general + task-specific tiers)
    2. GRPO-style group-relative advantage for extraction
    3. Recursive skill evolution (validation-failure refinement)
    4. Automatic tier promotion
    5. Level-k style counterparty trust modelling

    Config (via ``config`` dict):
        acceptance_threshold, trust_weight, post_probability,
        vote_probability, interact_probability, skill_exploration_rate,
        refinement_interval, promotion_interval
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        library_config: Optional[SkillLibraryConfig] = None,
        evolution_config: Optional[EvolutionConfig] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,  # Base behaviour is cooperative
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

        cfg = config or {}
        self.skillrl_config = SkillRLConfig(
            acceptance_threshold=cfg.get("acceptance_threshold", 0.4),
            trust_weight=cfg.get("trust_weight", 0.3),
            post_probability=cfg.get("post_probability", 0.2),
            vote_probability=cfg.get("vote_probability", 0.3),
            interact_probability=cfg.get("interact_probability", 0.5),
            skill_exploration_rate=cfg.get("skill_exploration_rate", 0.15),
            refinement_interval=cfg.get("refinement_interval", 2),
            promotion_interval=cfg.get("promotion_interval", 5),
        )

        # --- Skill evolution engine with SkillRL extensions enabled ---
        if evolution_config is None:
            evolution_config = EvolutionConfig(
                recursive_evolution_enabled=True,
                grpo_enabled=True,
                auto_tier_promotion=True,
                # Moderate extraction thresholds
                success_payoff_threshold=0.5,
                failure_payoff_threshold=-0.3,
                min_p_for_strategy=0.55,
                # GRPO window
                grpo_window_size=cfg.get("grpo_window_size", 20),
                grpo_temperature=cfg.get("grpo_temperature", 1.0),
                # Recursive refinement
                refinement_min_invocations=cfg.get("refinement_min_invocations", 3),
                refinement_success_threshold=cfg.get(
                    "refinement_success_threshold", 0.4,
                ),
                max_refinements_per_epoch=cfg.get("max_refinements_per_epoch", 3),
                # Tier promotion
                tier_promotion_min_invocations=cfg.get(
                    "tier_promotion_min_invocations", 10,
                ),
                tier_promotion_min_success_rate=cfg.get(
                    "tier_promotion_min_success_rate", 0.6,
                ),
            )

        self.skill_evolution = SkillEvolutionEngine(config=evolution_config)
        self.skill_library = SkillLibrary(
            owner_id=agent_id,
            config=library_config,
        )

        # Active skills during the current interaction
        self._active_skill_ids: List[str] = []

        # Policy gradient tracker
        self._pg_state = PolicyGradientState(
            payoff_history=deque(
                maxlen=self.skillrl_config.policy_window,
            ),
        )

        # Epoch tracking for periodic refinement/promotion
        self._last_refinement_epoch: int = -1
        self._last_promotion_epoch: int = -1
        # Tracks the last epoch we forwarded to the evolution engine so
        # that on_epoch_start() is called exactly once per new epoch,
        # resetting the per-epoch rate limiters for extraction/refinement.
        self._last_seen_epoch: int = -1

    # ------------------------------------------------------------------
    # SkillBank helpers
    # ------------------------------------------------------------------

    def _select_skill(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[Skill]:
        """Select a skill using tiered retrieval."""
        context = {
            "reputation": observation.agent_state.reputation,
            "resources": observation.agent_state.resources,
            "counterparty_id": counterparty_id,
            "trust": self.compute_counterparty_trust(counterparty_id),
        }

        skill = self.skill_library.select_best_skill_tiered(
            domain=SkillDomain.INTERACTION,
            context=context,
            exploration_rate=self.skillrl_config.skill_exploration_rate,
        )

        self._active_skill_ids = [skill.skill_id] if skill else []
        return skill

    def _apply_skill_threshold(
        self,
        base_threshold: float,
        skill: Optional[Skill],
    ) -> float:
        """Adjust acceptance threshold using skill effect."""
        if skill is None:
            return base_threshold
        delta = skill.effect.get("acceptance_threshold_delta", 0.0)
        return float(max(0.0, min(1.0, base_threshold + delta)))

    def _apply_skill_trust_weight(
        self,
        base_weight: float,
        skill: Optional[Skill],
    ) -> float:
        """Adjust trust weight using skill effect."""
        if skill is None:
            return base_weight
        delta = skill.effect.get("trust_weight_delta", 0.0)
        return float(max(0.0, min(1.0, base_weight + delta)))

    # ------------------------------------------------------------------
    # Core agent interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Act with SkillRL-augmented decision making."""
        epoch = observation.current_epoch

        # Forward epoch transition to the evolution engine so it resets
        # per-epoch rate limiters (extractions, refinements).
        if epoch != self._last_seen_epoch:
            self._last_seen_epoch = epoch
            self.skill_evolution.on_epoch_start(epoch)

        # Run periodic refinement and promotion at epoch boundaries.
        # env_state is not available here; an attached governance_lever is
        # still honored — refinements are evaluated against default baseline
        # metrics rather than silently auto-approved.
        self._maybe_refine(epoch, env_state=None)
        self._maybe_promote(epoch)

        # Handle pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")
            skill = self._select_skill(observation, initiator_id)

            threshold = self._apply_skill_threshold(
                self.skillrl_config.acceptance_threshold, skill,
            )
            trust = self.compute_counterparty_trust(initiator_id)
            tw = self._apply_skill_trust_weight(
                self.skillrl_config.trust_weight, skill,
            )

            # Skill-augmented acceptance score
            effective_score = trust * tw + 0.5 * (1 - tw)

            if skill and skill.effect.get("avoid_action"):
                return self.create_reject_action(proposal["proposal_id"])

            if effective_score >= threshold:
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # Propose interactions
        if (
            observation.can_interact
            and observation.visible_agents
            and self._rng.random() < self.skillrl_config.interact_probability
        ):
            candidates = [
                a for a in observation.visible_agents
                if a.get("agent_id") != self.agent_id
            ]
            if candidates:
                # Pick the most trusted counterparty
                best = max(
                    candidates,
                    key=lambda a: self.compute_counterparty_trust(
                        a.get("agent_id", ""),
                    ),
                )
                cid = best.get("agent_id", "")
                self._select_skill(observation, cid)
                return self.create_propose_action(
                    counterparty_id=cid,
                    interaction_type=InteractionType.COLLABORATION,
                    content="SkillRL collaboration proposal.",
                )

        # Post
        if (
            observation.can_post
            and self._rng.random() < self.skillrl_config.post_probability
        ):
            return self.create_post_action("Sharing skill-augmented insights.")

        # Vote
        if (
            observation.can_vote
            and observation.visible_posts
            and self._rng.random() < self.skillrl_config.vote_probability
        ):
            post = self._rng.choice(observation.visible_posts)
            return self.create_vote_action(post.get("post_id", ""), 1)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Skill-augmented acceptance decision."""
        skill = self._select_skill(observation, proposal.initiator_id)
        threshold = self._apply_skill_threshold(
            self.skillrl_config.acceptance_threshold, skill,
        )
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        tw = self._apply_skill_trust_weight(
            self.skillrl_config.trust_weight, skill,
        )
        effective = trust * tw + 0.5 * (1 - tw)

        if skill and skill.effect.get("avoid_action"):
            return False

        return bool(effective >= threshold)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose interaction with skill guidance."""
        trust = self.compute_counterparty_trust(counterparty_id)
        if trust < 0.25:
            return None

        self._select_skill(observation, counterparty_id)

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="SkillRL collaboration proposal.",
        )

    def update_from_outcome(
        self,
        interaction: "SoftInteraction",
        payoff: float,
    ) -> None:
        """Update with SkillRL pipeline: record, extract, track advantage."""
        super().update_from_outcome(interaction, payoff)

        # Record invocations for active skills
        for skill_id in self._active_skill_ids:
            self.skill_evolution.record_invocation(
                skill_id=skill_id,
                agent_id=self.agent_id,
                interaction_id=interaction.interaction_id,
                epoch=self.skill_evolution._current_epoch,
                step=0,  # SkillRL records all invocations as epoch-level events
                payoff=payoff,
                p=interaction.p,
                library=self.skill_library,
            )

        # Attempt skill extraction (uses GRPO advantage internally)
        self.skill_evolution.extract_skill(
            agent_id=self.agent_id,
            interaction=interaction,
            payoff=payoff,
            library=self.skill_library,
            active_skill_ids=self._active_skill_ids,
        )

        # Track payoff for policy gradient
        active_id = self._active_skill_ids[0] if self._active_skill_ids else None
        self._pg_state.record(payoff, active_id)

        # Reset active skills
        self._active_skill_ids = []

    # ------------------------------------------------------------------
    # Periodic SkillRL lifecycle
    # ------------------------------------------------------------------

    def _maybe_refine(self, epoch: int, env_state: Optional["EnvState"] = None) -> None:
        """Run recursive skill refinement at configured intervals.

        NEW: Evaluates refinements against governance gates if available.

        Args:
            epoch: Current epoch number.
            env_state: Optional environment state for governance evaluation.
        """
        import logging
        logger = logging.getLogger(__name__)

        interval = self.skillrl_config.refinement_interval
        if interval <= 0:
            return
        if epoch <= self._last_refinement_epoch:
            return
        if epoch % interval != 0:
            return

        self._last_refinement_epoch = epoch

        governance_lever = getattr(self, "governance_lever", None)

        # Collect refinement proposals without auto-applying when governance exists.
        _, proposals = self.skill_evolution.refine_skills_with_governance(
            self.skill_library,
            self.agent_id,
            governance_lever=governance_lever,
        )

        if not proposals:
            return

        # Apply governance if enabled. A lever with no env_state still gates:
        # evaluation falls back to default baseline metrics below — silently
        # auto-approving self-modifications past an attached lever would be a
        # governance bypass.
        if governance_lever is not None:
            approved_proposals = []

            for proposal in proposals:
                # Get agent metrics from environment state (when available)
                agent_metrics = (
                    env_state.get_agent_metrics(self.agent_id)
                    if env_state is not None and hasattr(env_state, 'get_agent_metrics')
                    else {}
                )
                baseline = {k: getattr(v, 'mean', 0.0) for k, v in agent_metrics.items()}
                uncertainties = {k: getattr(v, 'std', 0.1) for k, v in agent_metrics.items()}

                # If no metrics available, use defaults
                if not baseline:
                    baseline = {"payoff": 0.0, "reputation": 0.5}
                    uncertainties = {"payoff": 0.1, "reputation": 0.1}

                approved, tau_result, k_max_result = governance_lever.evaluate_refinement(
                    proposal, baseline, uncertainties
                )

                if approved:
                    approved_proposals.append(proposal)
                else:
                    logger.info(
                        f"Skill refinement blocked: {proposal.skill_id} - "
                        f"{'tau gate failed' if not tau_result.passed else 'k_max gate failed'}"
                    )
        else:
            # No governance: approve all proposals (backward compatible)
            approved_proposals = proposals

        # Apply approved refinements
        for proposal in approved_proposals:
            skill = self.skill_library.get_skill(proposal.skill_id)
            if skill is not None:
                # Apply condition delta
                new_condition = dict(skill.condition)
                for key, (_old_val, new_val) in proposal.condition_delta.items():
                    if new_val is not None:
                        new_condition[key] = new_val
                skill.condition = new_condition

                # Apply effect delta
                new_effect = dict(skill.effect)
                for key, (_old_val, new_val) in proposal.effect_delta.items():
                    if new_val is not None:
                        new_effect[key] = new_val
                skill.effect = new_effect

                # Update version and tags
                skill.version = proposal.refined_version
                skill.tags = set(skill.tags) | {"refined"}
                self.skill_evolution._refinements_this_epoch += 1

    def _maybe_promote(self, epoch: int) -> None:
        """Run tier promotion at configured intervals."""
        interval = self.skillrl_config.promotion_interval
        if interval <= 0:
            return
        if epoch <= self._last_promotion_epoch:
            return
        if epoch % interval != 0:
            return

        self._last_promotion_epoch = epoch
        self.skill_evolution.maybe_promote_to_general(self.skill_library)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def skill_summary(self) -> Dict:
        """Return a diagnostic summary of the SkillRL agent state."""
        skills = self.skill_library.all_skills
        general = [s for s in skills if s.tier == SkillTier.GENERAL]
        task_specific = [s for s in skills if s.tier == SkillTier.TASK_SPECIFIC]
        refined = [s for s in skills if "refined" in s.tags]

        return {
            "total_skills": len(skills),
            "general_tier": len(general),
            "task_specific_tier": len(task_specific),
            "strategies": sum(
                1 for s in skills if s.skill_type == SkillType.STRATEGY
            ),
            "lessons": sum(
                1 for s in skills if s.skill_type == SkillType.LESSON
            ),
            "composites": sum(
                1 for s in skills if s.skill_type == SkillType.COMPOSITE
            ),
            "refined": len(refined),
            "pg_baseline": self._pg_state.baseline,
            "pg_window_size": len(self._pg_state.payoff_history),
        }
