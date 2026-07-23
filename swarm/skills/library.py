"""Skill library with per-agent and shared modes.

Supports four sharing architectures:
- PRIVATE: Each agent has an independent library
- COORDINATOR_ONLY: Only designated coordinators evolve skills
- SHARED_GATED: Single shared library with reputation-gated writes
- COMMUNICATION: Skills referenced by ID across agents
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from swarm.skills.model import (
    Skill,
    SkillDomain,
    SkillPerformance,
    SkillTier,
    SkillType,
    clamp_p,
)


class SharingMode(Enum):
    """How skills are shared across the swarm."""

    PRIVATE = "private"
    COORDINATOR_ONLY = "coordinator_only"
    SHARED_GATED = "shared_gated"
    COMMUNICATION = "communication"


@dataclass
class SkillLibraryConfig:
    """Configuration for the skill library."""

    sharing_mode: SharingMode = SharingMode.PRIVATE
    max_skills_per_agent: int = 50
    max_shared_skills: int = 200
    prune_threshold: float = 0.2  # Remove skills with effectiveness below this
    prune_min_invocations: int = 3  # Only prune after enough data
    decay_rate: float = 0.9  # Per-epoch EMA decay for unused skills
    min_reputation_to_write: float = 1.0  # For SHARED_GATED mode
    coordinator_agent_ids: Set[str] = field(default_factory=set)


class SkillLibrary:
    """Manages a collection of skills with performance tracking.

    Can operate as a per-agent private library or a shared swarm-level
    library depending on the SharingMode.
    """

    def __init__(
        self,
        owner_id: str = "shared",
        config: Optional[SkillLibraryConfig] = None,
    ):
        self.owner_id = owner_id
        self.config = config or SkillLibraryConfig()

        # Skills indexed by ID
        self._skills: Dict[str, Skill] = {}

        # Performance tracking per skill
        self._performance: Dict[str, SkillPerformance] = {}

        # Index: domain -> skill_ids for fast lookup
        self._domain_index: Dict[SkillDomain, Set[str]] = {
            d: set() for d in SkillDomain
        }

        # Write access control: agent_id -> bool
        self._write_permissions: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def add_skill(
        self,
        skill: Skill,
        author_reputation: float = 0.0,
    ) -> bool:
        """Add a skill to the library.

        Returns True if accepted, False if rejected (capacity or gate).
        """
        # Capacity check
        max_cap = (
            self.config.max_shared_skills
            if self.owner_id == "shared"
            else self.config.max_skills_per_agent
        )
        if len(self._skills) >= max_cap:
            # Evict worst-performing skill if at capacity
            if not self._evict_weakest():
                return False

        # Reputation gate for shared library
        if (
            self.config.sharing_mode == SharingMode.SHARED_GATED
            and self.owner_id == "shared"
        ):
            if author_reputation < self.config.min_reputation_to_write:
                return False

        # Coordinator gate
        if self.config.sharing_mode == SharingMode.COORDINATOR_ONLY:
            if (
                self.owner_id == "shared"
                and skill.created_by not in self.config.coordinator_agent_ids
            ):
                return False

        self._skills[skill.skill_id] = skill
        self._performance[skill.skill_id] = SkillPerformance(
            skill_id=skill.skill_id,
        )
        self._domain_index[skill.domain].add(skill.skill_id)
        return True

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by ID."""
        return self._skills.get(skill_id)

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill from the library."""
        skill = self._skills.pop(skill_id, None)
        if skill is None:
            return False
        self._performance.pop(skill_id, None)
        self._domain_index[skill.domain].discard(skill_id)
        return True

    def get_performance(self, skill_id: str) -> Optional[SkillPerformance]:
        """Get performance tracker for a skill."""
        return self._performance.get(skill_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_skills_by_domain(self, domain: SkillDomain) -> List[Skill]:
        """Get all skills in a given domain."""
        return [
            self._skills[sid]
            for sid in self._domain_index.get(domain, set())
            if sid in self._skills
        ]

    def get_skills_by_type(self, skill_type: SkillType) -> List[Skill]:
        """Get all skills of a given type."""
        return [s for s in self._skills.values() if s.skill_type == skill_type]

    def get_skills_by_tier(self, tier: SkillTier) -> List[Skill]:
        """Get all skills in a given tier (GENERAL or TASK_SPECIFIC)."""
        return [s for s in self._skills.values() if s.tier == tier]

    def select_best_skill_tiered(
        self,
        domain: SkillDomain,
        context: Dict,
        exploration_rate: float = 0.1,
    ) -> Optional[Skill]:
        """SkillRL-style tiered retrieval: task-specific first, general fallback.

        Retrieves the best applicable task-specific skill for the domain.
        If none is found, falls back to the best applicable general skill.
        """
        # Try task-specific skills in the requested domain
        task_applicable = [
            s for s in self.get_applicable_skills(domain, context)
            if s.tier == SkillTier.TASK_SPECIFIC
        ]
        if task_applicable:
            return self._epsilon_greedy_select(task_applicable, exploration_rate)

        # Fallback to general-tier skills (domain-agnostic)
        general_applicable = [
            s for s in self._skills.values()
            if s.tier == SkillTier.GENERAL and self._condition_matches(s, context)
        ]
        return self._epsilon_greedy_select(general_applicable, exploration_rate)

    def get_applicable_skills(
        self,
        domain: SkillDomain,
        context: Dict,
    ) -> List[Skill]:
        """Get skills whose conditions match the given context.

        Context keys: p, reputation, trust, interaction_type, counterparty_type
        """
        candidates = self.get_skills_by_domain(domain)
        result = []
        for skill in candidates:
            if self._condition_matches(skill, context):
                result.append(skill)
        return result

    def _epsilon_greedy_select(
        self,
        applicable_skills: List[Skill],
        exploration_rate: float = 0.1,
    ) -> Optional[Skill]:
        """Select a skill using epsilon-greedy exploration-exploitation.

        With probability exploration_rate, returns a random applicable skill.
        Otherwise returns the one with highest effectiveness.
        """
        if not applicable_skills:
            return None

        # Exploration: choose random skill
        if random.random() < exploration_rate:
            return random.choice(applicable_skills)

        # Exploitation: pick skill with highest effectiveness
        return max(
            applicable_skills,
            key=lambda s: self._performance.get(s.skill_id, SkillPerformance()).effectiveness,
        )

    def select_best_skill(
        self,
        domain: SkillDomain,
        context: Dict,
        exploration_rate: float = 0.1,
    ) -> Optional[Skill]:
        """Select the best applicable skill using epsilon-greedy.

        With probability exploration_rate, returns a random applicable skill.
        Otherwise returns the one with highest effectiveness.
        """
        applicable = self.get_applicable_skills(domain, context)
        return self._epsilon_greedy_select(applicable, exploration_rate)

    @property
    def size(self) -> int:
        """Number of skills in the library."""
        return len(self._skills)

    @property
    def all_skills(self) -> List[Skill]:
        """All skills in the library."""
        return list(self._skills.values())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def record_invocation(
        self,
        skill_id: str,
        payoff: float,
        p: float,
    ) -> None:
        """Record the outcome of invoking a skill."""
        p = clamp_p(p)
        perf = self._performance.get(skill_id)
        if perf:
            perf.record(payoff, p)

    def epoch_decay(self) -> None:
        """Apply per-epoch decay to all skills."""
        for perf in self._performance.values():
            perf.decay()

    def prune(self) -> List[str]:
        """Remove underperforming skills. Returns IDs of pruned skills."""
        to_remove = []
        for sid, perf in self._performance.items():
            if (
                perf.invocations >= self.config.prune_min_invocations
                and perf.effectiveness < self.config.prune_threshold
            ):
                to_remove.append(sid)

        for sid in to_remove:
            self.remove_skill(sid)

        return to_remove

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _condition_matches(skill: Skill, context: Dict) -> bool:
        """Check if a skill's condition matches the context."""
        cond = skill.condition
        if not cond:
            return True  # No condition = always applies

        p = context.get("p")
        if p is not None:
            if "min_p" in cond and p < cond["min_p"]:
                return False
            if "max_p" in cond and p > cond["max_p"]:
                return False

        rep = context.get("reputation")
        if rep is not None:
            if "min_reputation" in cond and rep < cond["min_reputation"]:
                return False
            if "max_reputation" in cond and rep > cond["max_reputation"]:
                return False

        trust = context.get("trust")
        if trust is not None:
            if "min_trust" in cond and trust < cond["min_trust"]:
                return False
            if "max_trust" in cond and trust > cond["max_trust"]:
                return False

        i_type = context.get("interaction_type")
        if i_type and "interaction_types" in cond:
            if i_type not in cond["interaction_types"]:
                return False

        cp_type = context.get("counterparty_type")
        if cp_type and "counterparty_types" in cond:
            if cp_type not in cond["counterparty_types"]:
                return False

        return True

    def _evict_weakest(self) -> bool:
        """Evict the weakest skill to make room. Returns False if nothing to evict."""
        if not self._performance:
            return False

        # Find skill with lowest effectiveness that has enough invocations
        candidates = [
            (sid, perf)
            for sid, perf in self._performance.items()
            if perf.invocations >= self.config.prune_min_invocations
        ]

        if not candidates:
            # Fall back to oldest skill
            oldest = min(self._skills.values(), key=lambda s: s.created_at)
            self.remove_skill(oldest.skill_id)
            return True

        worst_sid = min(candidates, key=lambda x: x[1].effectiveness)[0]
        self.remove_skill(worst_sid)
        return True

    def to_dict(self) -> Dict:
        """Serialize entire library."""
        return {
            "owner_id": self.owner_id,
            "skills": {sid: s.to_dict() for sid, s in self._skills.items()},
            "performance": {
                sid: {
                    "invocations": p.invocations,
                    "successes": p.successes,
                    "failures": p.failures,
                    "total_payoff": p.total_payoff,
                    "ema_payoff": p.ema_payoff,
                    "epochs_since_last_use": p.epochs_since_last_use,
                }
                for sid, p in self._performance.items()
            },
        }
