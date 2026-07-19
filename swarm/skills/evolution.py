"""Skill evolution engine: extracts, scores, composes, and prunes skills.

Implements SkillRL-style recursive skill extraction from:
    Xia, P. et al. (2026). SkillRL: Evolving Agents via Recursive
    Skill-Augmented Reinforcement Learning. arXiv:2602.08234 [cs.LG].

Core loop:
- Success (payoff > 0) -> new strategy skill
- Failure (payoff < 0) -> new lesson skill
- Co-occurring successes -> composite skill candidates
- Recursive refinement: tighten failing skills via validation analysis
- GRPO advantage: score skills relative to group baseline
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from swarm.models.interaction import SoftInteraction
from swarm.skills.library import SkillLibrary
from swarm.skills.model import (
    Skill,
    SkillDomain,
    SkillInvocation,
    SkillTier,
    SkillType,
)

if TYPE_CHECKING:
    from swarm.governance.refinement_proposal import RefinementProposal
    from swarm.governance.self_modification import SelfModificationLever


@dataclass
class EvolutionConfig:
    """Configuration for the skill evolution engine."""

    # Extraction thresholds
    success_payoff_threshold: float = 0.5  # Min payoff to extract strategy
    failure_payoff_threshold: float = -0.3  # Max payoff to extract lesson
    min_p_for_strategy: float = 0.6  # Min interaction quality for strategy

    # Composition
    min_co_occurrences: int = 3  # Min times two skills succeed together
    composition_effectiveness_threshold: float = 0.6

    # Pruning frequency
    prune_every_n_epochs: int = 3

    # Skill naming
    auto_name: bool = True

    # Rate limiting
    max_extractions_per_epoch: int = 5

    # Memory bounds
    max_invocation_log_size: int = 5000
    max_co_occurrence_entries: int = 10000

    # Effect delta clamp range for composed skills
    max_effect_delta: float = 0.5

    # --- SkillRL extensions (Xia et al., 2026) ---

    # Recursive evolution: refine skills when they fail during validation
    recursive_evolution_enabled: bool = False
    # Minimum invocations before a skill is eligible for refinement
    refinement_min_invocations: int = 3
    # If a skill's success_rate drops below this after refinement_min_invocations,
    # tighten its condition or adjust its effect
    refinement_success_threshold: float = 0.4
    # How much to tighten condition bands on refinement (shrink ±p band)
    refinement_band_shrink: float = 0.05
    # Maximum number of refinements per epoch
    max_refinements_per_epoch: int = 3

    # GRPO-style advantage: use group-relative advantage instead of raw payoff
    grpo_enabled: bool = False
    # Window of recent payoffs for computing group baseline
    grpo_window_size: int = 20
    # Temperature for advantage normalisation
    grpo_temperature: float = 1.0

    # Tier assignment: skills with broad applicability become GENERAL
    auto_tier_promotion: bool = False
    # Minimum invocations across distinct domains for tier promotion
    tier_promotion_min_invocations: int = 10
    # Minimum success rate for tier promotion
    tier_promotion_min_success_rate: float = 0.6


class SkillEvolutionEngine:
    """Extracts, evolves, and composes skills from interaction outcomes.

    This is the core RL-style learning loop:
    1. After each interaction, check if outcome warrants a new skill
    2. Extract strategy (from success) or lesson (from failure) skills
    3. Track co-occurrence patterns for composite skill candidates
    4. Periodically prune underperforming skills
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
    ):
        self.config = config or EvolutionConfig()

        # Track co-occurrences: (skill_a, skill_b) -> success count
        self._co_occurrences: Dict[Tuple[str, str], int] = {}

        # Track extractions per epoch for rate limiting
        self._extractions_this_epoch: int = 0
        self._current_epoch: int = 0

        # Invocation log for analysis
        self._invocation_log: List[SkillInvocation] = []

        # --- SkillRL extensions ---
        # GRPO: rolling window of recent payoffs for group baseline
        self._grpo_payoff_buffer: deque = deque(
            maxlen=max(1, self.config.grpo_window_size),
        )
        # Refinement rate limiting
        self._refinements_this_epoch: int = 0

    def on_epoch_start(self, epoch: int) -> None:
        """Reset per-epoch counters."""
        self._current_epoch = epoch
        self._extractions_this_epoch = 0
        self._refinements_this_epoch = 0

    def extract_skill(
        self,
        agent_id: str,
        interaction: SoftInteraction,
        payoff: float,
        library: SkillLibrary,
        active_skill_ids: Optional[List[str]] = None,
    ) -> Optional[Skill]:
        """Attempt to extract a new skill from an interaction outcome.

        Args:
            agent_id: Agent that experienced the interaction.
            interaction: The completed interaction.
            payoff: Payoff received by the agent.
            library: The agent's skill library to add to.
            active_skill_ids: Skills that were active during this interaction.

        Returns:
            The newly created Skill, or None if no extraction warranted.
        """
        # Rate limit
        if self._extractions_this_epoch >= self.config.max_extractions_per_epoch:
            return None

        # Track co-occurrences of active skills
        if active_skill_ids and payoff > 0:
            self._record_co_occurrences(active_skill_ids)

        # If GRPO is enabled, use advantage for extraction decisions
        effective_payoff = payoff
        if self.config.grpo_enabled:
            effective_payoff = self.compute_grpo_advantage(payoff)

        # Determine if we should extract
        skill = None
        if effective_payoff >= self.config.success_payoff_threshold and interaction.p >= self.config.min_p_for_strategy:
            skill = self._extract_strategy(agent_id, interaction, payoff)
        elif effective_payoff <= self.config.failure_payoff_threshold:
            skill = self._extract_lesson(agent_id, interaction, payoff)

        if skill is not None:
            added = library.add_skill(skill)
            if added:
                self._extractions_this_epoch += 1
                return skill

        return None

    def try_compose(
        self,
        library: SkillLibrary,
        agent_id: str,
    ) -> Optional[Skill]:
        """Try to compose a new higher-order skill from co-occurring successes.

        Returns the composed skill if one was created.
        """
        for (sid_a, sid_b), count in self._co_occurrences.items():
            if count < self.config.min_co_occurrences:
                continue

            skill_a = library.get_skill(sid_a)
            skill_b = library.get_skill(sid_b)
            if skill_a is None or skill_b is None:
                continue

            perf_a = library.get_performance(sid_a)
            perf_b = library.get_performance(sid_b)
            if perf_a is None or perf_b is None:
                continue

            # Both need to be effective
            if (
                perf_a.effectiveness >= self.config.composition_effectiveness_threshold
                and perf_b.effectiveness >= self.config.composition_effectiveness_threshold
            ):
                composite = self._compose_skills(skill_a, skill_b, agent_id)
                if library.add_skill(composite):
                    # Reset co-occurrence counter
                    self._co_occurrences[(sid_a, sid_b)] = 0
                    return composite

        return None

    def record_invocation(
        self,
        skill_id: str,
        agent_id: str,
        interaction_id: str,
        epoch: int,
        step: int,
        payoff: float,
        p: float,
        library: SkillLibrary,
    ) -> SkillInvocation:
        """Record a skill invocation and update performance."""
        invocation = SkillInvocation(
            skill_id=skill_id,
            agent_id=agent_id,
            interaction_id=interaction_id,
            epoch=epoch,
            step=step,
            payoff=payoff,
            p=p,
        )
        self._invocation_log.append(invocation)
        # Cap invocation log to prevent unbounded memory growth
        if len(self._invocation_log) > self.config.max_invocation_log_size:
            self._invocation_log = self._invocation_log[
                -self.config.max_invocation_log_size:
            ]
        library.record_invocation(skill_id, payoff, p)
        return invocation

    @property
    def invocation_log(self) -> List[SkillInvocation]:
        """Access the full invocation log."""
        return self._invocation_log

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_strategy(
        self,
        agent_id: str,
        interaction: SoftInteraction,
        payoff: float,
    ) -> Skill:
        """Extract a strategy skill from a successful interaction."""
        domain = self._infer_domain(interaction)
        condition = self._build_condition_from_interaction(interaction)
        effect = self._build_strategy_effect(interaction, payoff)
        tier = self._assign_tier(interaction)

        name = ""
        if self.config.auto_name:
            name = (
                f"strategy_{domain.value}_p{interaction.p:.2f}"
                f"_pay{payoff:.1f}"
            )

        return Skill(
            name=name,
            skill_type=SkillType.STRATEGY,
            domain=domain,
            tier=tier,
            created_by=agent_id,
            condition=condition,
            effect=effect,
            source_interaction_ids=[interaction.interaction_id],
            tags={"auto_extracted", "strategy"},
        )

    def _extract_lesson(
        self,
        agent_id: str,
        interaction: SoftInteraction,
        payoff: float,
    ) -> Skill:
        """Extract a lesson skill from a failed interaction."""
        domain = self._infer_domain(interaction)
        condition = self._build_condition_from_interaction(interaction)
        effect = self._build_lesson_effect(interaction, payoff)
        tier = self._assign_tier(interaction)

        name = ""
        if self.config.auto_name:
            name = (
                f"lesson_{domain.value}_p{interaction.p:.2f}"
                f"_pay{payoff:.1f}"
            )

        return Skill(
            name=name,
            skill_type=SkillType.LESSON,
            domain=domain,
            tier=tier,
            created_by=agent_id,
            condition=condition,
            effect=effect,
            source_interaction_ids=[interaction.interaction_id],
            tags={"auto_extracted", "lesson"},
        )

    def _compose_skills(
        self,
        skill_a: Skill,
        skill_b: Skill,
        agent_id: str,
    ) -> Skill:
        """Compose two skills into a higher-order composite skill."""
        # Merge conditions (intersection / tightest bounds)
        merged_condition = self._merge_conditions(
            skill_a.condition, skill_b.condition
        )
        # Merge effects (additive deltas)
        merged_effect = self._merge_effects(skill_a.effect, skill_b.effect)

        name = ""
        if self.config.auto_name:
            name = f"composite_{skill_a.name}+{skill_b.name}"

        return Skill(
            name=name,
            skill_type=SkillType.COMPOSITE,
            domain=skill_a.domain,  # Use primary skill's domain
            created_by=agent_id,
            child_ids=[skill_a.skill_id, skill_b.skill_id],
            condition=merged_condition,
            effect=merged_effect,
            tags={"auto_composed", "composite"},
        )

    # ------------------------------------------------------------------
    # Context inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_domain(interaction: SoftInteraction) -> SkillDomain:
        """Infer the skill domain from interaction characteristics."""
        if interaction.interaction_type.value == "collaboration":
            return SkillDomain.COORDINATION
        if interaction.interaction_type.value == "trade":
            return SkillDomain.INTERACTION
        return SkillDomain.INTERACTION

    @staticmethod
    def _build_condition_from_interaction(interaction: SoftInteraction) -> Dict:
        """Build a skill condition that matches similar future interactions."""
        # Create a band around the observed p value
        p = interaction.p
        band = 0.15
        condition: Dict = {
            "min_p": max(0.0, p - band),
            "max_p": min(1.0, p + band),
            "interaction_types": [interaction.interaction_type.value],
        }
        return condition

    @staticmethod
    def _build_strategy_effect(interaction: SoftInteraction, payoff: float) -> Dict:
        """Build effect descriptor for a strategy skill."""
        # Strategies encourage similar behavior
        effect: Dict = {
            "acceptance_threshold_delta": -0.05,  # Be slightly more accepting
            "trust_weight_delta": 0.05,  # Lean more on trust signals
        }
        if payoff > 1.0:
            effect["acceptance_threshold_delta"] = -0.1
        return effect

    @staticmethod
    def _build_lesson_effect(interaction: SoftInteraction, payoff: float) -> Dict:
        """Build effect descriptor for a lesson skill."""
        # Lessons encourage avoidance
        effect: Dict = {
            "acceptance_threshold_delta": 0.1,  # Be more cautious
            "trust_weight_delta": -0.05,  # Rely less on trust alone
        }
        if payoff < -1.0:
            effect["acceptance_threshold_delta"] = 0.15
            effect["avoid_action"] = True
        return effect

    @staticmethod
    def _merge_conditions(cond_a: Dict, cond_b: Dict) -> Dict:
        """Merge two conditions using tightest bounds."""
        merged: Dict = {}

        # Numeric bounds: take tightest
        for key in ("min_p", "min_reputation", "min_trust"):
            vals = [c[key] for c in (cond_a, cond_b) if key in c]
            if vals:
                merged[key] = max(vals)

        for key in ("max_p", "max_reputation", "max_trust"):
            vals = [c[key] for c in (cond_a, cond_b) if key in c]
            if vals:
                merged[key] = min(vals)

        # Sets: intersection
        for key in ("interaction_types", "counterparty_types"):
            if key in cond_a and key in cond_b:
                merged[key] = list(
                    set(cond_a[key]) & set(cond_b[key])
                ) or list(set(cond_a[key]) | set(cond_b[key]))
            elif key in cond_a:
                merged[key] = cond_a[key]
            elif key in cond_b:
                merged[key] = cond_b[key]

        return merged

    def _merge_effects(self, eff_a: Dict, eff_b: Dict) -> Dict:
        """Merge two effects additively for numeric values.

        Clamps numeric deltas to [-max_effect_delta, +max_effect_delta]
        to prevent unbounded accumulation through recursive composition.
        """
        merged: Dict = {}
        all_keys = set(eff_a.keys()) | set(eff_b.keys())
        clamp = self.config.max_effect_delta

        for key in all_keys:
            va = eff_a.get(key)
            vb = eff_b.get(key)

            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                merged[key] = max(-clamp, min(clamp, va + vb))
            elif va is not None:
                merged[key] = va
            else:
                merged[key] = vb

        return merged

    # ------------------------------------------------------------------
    # GRPO-style group-relative advantage (Xia et al., 2026)
    # ------------------------------------------------------------------

    def compute_grpo_advantage(self, payoff: float) -> float:
        """Compute group-relative advantage for a payoff.

        Instead of using raw payoff to decide whether to extract a
        strategy or lesson, GRPO normalises against a running baseline
        of recent payoffs:

            advantage = (payoff - mean) / (std + temperature)

        This makes skill extraction adaptive: in a high-payoff
        environment the bar for "success" rises, and in a low-payoff
        environment small wins still count.
        """
        import math

        # Guard against NaN/Inf input
        if math.isnan(payoff) or math.isinf(payoff):
            return 0.0

        self._grpo_payoff_buffer.append(payoff)

        if len(self._grpo_payoff_buffer) < 2:
            return payoff  # Not enough data yet

        buf = list(self._grpo_payoff_buffer)
        mean = sum(buf) / len(buf)
        variance = sum((x - mean) ** 2 for x in buf) / len(buf)
        std = variance ** 0.5
        # Guard against zero denominator (temperature=0 + constant payoffs)
        denom = std + max(1e-8, self.config.grpo_temperature)
        return float((payoff - mean) / denom)

    # ------------------------------------------------------------------
    # Recursive skill evolution (validation-failure refinement)
    # ------------------------------------------------------------------

    def refine_skills(
        self,
        library: "SkillLibrary",
        agent_id: str,
    ) -> List[str]:
        """Refine under-performing skills by tightening conditions/effects.

        This implements the recursive evolution mechanism from SkillRL:
        after a skill has been invoked enough times, if it is still
        under-performing, we tighten its condition band so it fires
        in a narrower (more appropriate) context, and adjust its effect
        to be more conservative.

        Note: Mutations are done via copy-on-write (new dict objects) so
        that shared-library scenarios are safe from cross-agent side effects.

        Returns list of skill IDs that were refined.
        """
        if not self.config.recursive_evolution_enabled:
            return []

        refined: List[str] = []
        for skill in list(library.all_skills):
            if self._refinements_this_epoch >= self.config.max_refinements_per_epoch:
                break

            perf = library.get_performance(skill.skill_id)
            if perf is None or perf.invocations < self.config.refinement_min_invocations:
                continue

            if perf.success_rate >= self.config.refinement_success_threshold:
                continue  # Performing well enough

            # Composite skills are not directly refined
            if skill.skill_type == SkillType.COMPOSITE:
                continue

            # --- Tighten the condition band (copy-on-write) ---
            shrink = self.config.refinement_band_shrink
            cond = dict(skill.condition)
            if "min_p" in cond and "max_p" in cond:
                mid = (cond["min_p"] + cond["max_p"]) / 2.0
                half = max(0.05, (cond["max_p"] - cond["min_p"]) / 2.0 - shrink)
                cond["min_p"] = max(0.0, mid - half)
                cond["max_p"] = min(1.0, mid + half)
            skill.condition = cond

            # --- Adjust effect toward caution (copy-on-write) ---
            eff = dict(skill.effect)
            if skill.skill_type == SkillType.STRATEGY:
                # Make the strategy less aggressive
                delta = eff.get("acceptance_threshold_delta", 0.0)
                eff["acceptance_threshold_delta"] = max(-0.5, min(0.5, delta + 0.02))
            elif skill.skill_type == SkillType.LESSON:
                # Make the lesson more cautious
                delta = eff.get("acceptance_threshold_delta", 0.0)
                eff["acceptance_threshold_delta"] = max(-0.5, min(0.5, delta + 0.03))
            skill.effect = eff

            # Bump version; copy tags to avoid mutating a shared set
            skill.version += 1
            skill.tags = set(skill.tags) | {"refined"}

            self._refinements_this_epoch += 1
            refined.append(skill.skill_id)

        return refined

    def refine_skills_with_governance(
        self,
        library: "SkillLibrary",
        agent_id: str,
        governance_lever: Optional["SelfModificationLever"] = None,
    ) -> Tuple[List[str], List["RefinementProposal"]]:
        """Refine skills, returning both IDs and governance proposals.

        This method collects refinement proposals for skills that need
        refinement. When governance_lever is provided, it returns the
        proposals for evaluation by the governance lever. When governance_lever
        is None (backward compatible), it applies refinements directly to skills.

        Args:
            library: The agent's skill library.
            agent_id: ID of the agent proposing refinements.
            governance_lever: Optional governance lever for evaluation.
                When None, refinements are applied immediately.

        Returns:
            (refined_skill_ids, proposals_submitted_to_governance)
        """
        from swarm.governance.refinement_proposal import RefinementProposal

        if not self.config.recursive_evolution_enabled:
            return [], []

        proposals = []
        refined = []

        for skill in list(library.all_skills):
            if self._refinements_this_epoch >= self.config.max_refinements_per_epoch:
                break

            perf = library.get_performance(skill.skill_id)

            # Eligibility check
            if perf is None or perf.invocations < self.config.refinement_min_invocations:
                continue
            if perf.success_rate >= self.config.refinement_success_threshold:
                continue
            if skill.skill_type == SkillType.COMPOSITE:
                continue

            # COLLECT CHANGES (don't apply yet)
            old_condition = dict(skill.condition)
            old_effect = dict(skill.effect)

            # Compute refined condition
            shrink = self.config.refinement_band_shrink
            new_condition = dict(old_condition)
            if "min_p" in old_condition and "max_p" in old_condition:
                mid = (old_condition["min_p"] + old_condition["max_p"]) / 2.0
                half = max(0.05, (old_condition["max_p"] - old_condition["min_p"]) / 2.0 - shrink)
                new_condition["min_p"] = max(0.0, mid - half)
                new_condition["max_p"] = min(1.0, mid + half)

            # Compute refined effect
            new_effect = dict(old_effect)
            if skill.skill_type == SkillType.STRATEGY:
                delta = old_effect.get("acceptance_threshold_delta", 0.0)
                new_effect["acceptance_threshold_delta"] = max(-0.5, min(0.5, delta + 0.02))
            elif skill.skill_type == SkillType.LESSON:
                delta = old_effect.get("acceptance_threshold_delta", 0.0)
                new_effect["acceptance_threshold_delta"] = max(-0.5, min(0.5, delta + 0.03))

            # CREATE PROPOSAL
            proposal = RefinementProposal(
                skill_id=skill.skill_id,
                original_version=skill.version,
                refined_version=skill.version + 1,
                condition_delta={
                    "min_p": (
                        old_condition.get("min_p", 0.0),
                        new_condition.get("min_p", 0.0)
                    ),
                    "max_p": (
                        old_condition.get("max_p", 1.0),
                        new_condition.get("max_p", 1.0)
                    ),
                },
                effect_delta={
                    k: (old_effect.get(k), new_effect.get(k))
                    for k in set(old_effect.keys()) | set(new_effect.keys())
                },
                perf_before={
                    "success_rate": perf.success_rate,
                    "invocations": perf.invocations,
                    "avg_payoff": perf.avg_payoff,
                },
                skill_type=skill.skill_type.value,
                agent_id=agent_id,
            )
            proposals.append(proposal)

            # CONDITIONAL: Apply only if governance is None (backward compatible)
            if governance_lever is None:
                # No governance: apply immediately (backward compatible)
                skill.condition = new_condition
                skill.effect = new_effect
                skill.version += 1
                skill.tags = set(skill.tags) | {"refined"}
                self._refinements_this_epoch += 1
                refined.append(skill.skill_id)

        return refined, proposals

    # ------------------------------------------------------------------
    # Automatic tier promotion
    # ------------------------------------------------------------------

    def maybe_promote_to_general(
        self,
        library: "SkillLibrary",
    ) -> List[str]:
        """Promote high-performing task-specific skills to GENERAL tier.

        A skill that has been invoked many times with consistent success
        across interactions is likely a universal heuristic rather than a
        domain-specific one.  Promoting it to GENERAL makes it available
        as a fallback in all domains.
        """
        if not self.config.auto_tier_promotion:
            return []

        promoted: List[str] = []
        for skill in library.all_skills:
            if skill.tier != SkillTier.TASK_SPECIFIC:
                continue

            perf = library.get_performance(skill.skill_id)
            if perf is None:
                continue

            if (
                perf.invocations >= self.config.tier_promotion_min_invocations
                and perf.success_rate >= self.config.tier_promotion_min_success_rate
            ):
                skill.tier = SkillTier.GENERAL
                # Copy-on-write: remove domain-specific constraints
                cond = dict(skill.condition)
                cond.pop("interaction_types", None)
                skill.condition = cond
                skill.tags = set(skill.tags) | {"promoted_general"}
                promoted.append(skill.skill_id)

        return promoted

    # ------------------------------------------------------------------
    # Tier-aware extraction helpers
    # ------------------------------------------------------------------

    def _assign_tier(self, interaction: SoftInteraction) -> SkillTier:
        """Determine initial tier for a newly extracted skill.

        Skills from domain-specific interaction types start as
        TASK_SPECIFIC.  Skills from generic/uncategorised contexts
        default to GENERAL.
        """
        if interaction.interaction_type.value in ("collaboration", "trade"):
            return SkillTier.TASK_SPECIFIC
        return SkillTier.GENERAL

    # ------------------------------------------------------------------
    # Co-occurrence tracking
    # ------------------------------------------------------------------

    def _record_co_occurrences(self, skill_ids: List[str]) -> None:
        """Record co-occurrence of skills in a successful interaction."""
        for i, sid_a in enumerate(skill_ids):
            for sid_b in skill_ids[i + 1:]:
                key: tuple[str, str] = (min(sid_a, sid_b), max(sid_a, sid_b))
                self._co_occurrences[key] = self._co_occurrences.get(key, 0) + 1

        # Cap co-occurrence dict to prevent unbounded growth
        if len(self._co_occurrences) > self.config.max_co_occurrence_entries:
            # Keep only the entries with highest counts
            sorted_entries = sorted(
                self._co_occurrences.items(), key=lambda x: x[1], reverse=True
            )
            self._co_occurrences = dict(
                sorted_entries[: self.config.max_co_occurrence_entries]
            )
