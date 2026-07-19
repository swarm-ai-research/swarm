"""Translate between EvoSkill's .claude/skills/ format and SWARM's Skill model.

EvoSkill stores skills as plain-text files under ``.claude/skills/<name>/``.
Each file contains natural-language instructions that Claude Code agents
consume at runtime.  SWARM's ``Skill`` model is a structured dataclass
with typed condition/effect dicts, a domain enum, and performance tracking.

``SkillTranslator`` bridges this gap in both directions:

- **ingest**: Parse an EvoSkill skill file (or git branch snapshot) into
  a SWARM ``Skill`` with heuristically-extracted conditions and effects.
- **export**: Render a SWARM ``Skill`` back into the natural-language
  format that EvoSkill's agent profiles expect.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from swarm.skills.model import (
    Skill,
    SkillDomain,
    SkillTier,
    SkillType,
    validate_condition,
    validate_effect,
)

# Heuristic keyword → SkillDomain mapping for auto-classification.
_DOMAIN_KEYWORDS: Dict[SkillDomain, List[str]] = {
    SkillDomain.INTERACTION: ["interact", "communicate", "respond", "reply", "message"],
    SkillDomain.ACCEPTANCE: ["accept", "reject", "filter", "threshold", "gate"],
    SkillDomain.TARGETING: ["select", "target", "choose", "partner", "match"],
    SkillDomain.POSTING: ["post", "publish", "content", "write", "output"],
    SkillDomain.GOVERNANCE: ["govern", "audit", "contract", "compliance", "rule"],
    SkillDomain.COORDINATION: ["coordinate", "collaborate", "team", "delegate", "plan"],
}

_STRATEGY_KEYWORDS = ["always", "prefer", "optimize", "maximize", "best", "should"]
_LESSON_KEYWORDS = ["avoid", "never", "don't", "prevent", "warning", "careful"]


class SkillTranslator:
    """Bidirectional translator between EvoSkill and SWARM skill formats."""

    def __init__(self, default_author: str = "evoskill_proposer") -> None:
        self._default_author = default_author

    # ------------------------------------------------------------------
    # EvoSkill → SWARM
    # ------------------------------------------------------------------

    def ingest(
        self,
        skill_text: str,
        *,
        name: Optional[str] = None,
        source_branch: Optional[str] = None,
        author_id: Optional[str] = None,
        tags: Optional[set] = None,
    ) -> Skill:
        """Parse an EvoSkill skill file into a SWARM Skill.

        Args:
            skill_text: Raw text content of the EvoSkill skill file.
            name: Skill name (defaults to first line truncated to 80 characters,
                or content hash if no text provided).
            source_branch: Git branch the skill was discovered on.
            author_id: Agent that created this skill.
            tags: Additional tags.

        Returns:
            A SWARM ``Skill`` with heuristically-extracted fields.
        """
        text_lower = skill_text.lower()
        lines = [ln.strip() for ln in skill_text.strip().splitlines() if ln.strip()]

        # Name: first non-empty line, or content hash
        if name is None:
            name = lines[0][:80] if lines else hashlib.sha256(
                skill_text.encode()
            ).hexdigest()[:12]

        # Deterministic ID from content for dedup
        skill_id = self._content_id(skill_text)

        # Classify domain
        domain = self._classify_domain(text_lower)

        # Classify type (strategy vs lesson)
        skill_type = self._classify_type(text_lower)

        # Extract structured condition/effect hints
        condition = self._extract_condition(text_lower)
        effect = self._extract_effect(text_lower)

        # Build tags
        skill_tags = {"evoskill", "auto_discovered"}
        if source_branch:
            skill_tags.add(f"branch:{source_branch}")
        if tags:
            skill_tags.update(tags)

        return Skill(
            skill_id=skill_id,
            name=name,
            skill_type=skill_type,
            domain=domain,
            tier=SkillTier.TASK_SPECIFIC,
            created_at=datetime.now(),
            created_by=author_id or self._default_author,
            version=1,
            condition=validate_condition(condition),
            effect=validate_effect(effect),
            source_interaction_ids=[],
            tags=skill_tags,
        )

    def ingest_batch(
        self,
        skill_files: Dict[str, str],
        source_branch: Optional[str] = None,
        author_id: Optional[str] = None,
    ) -> List[Skill]:
        """Ingest multiple EvoSkill files from a branch snapshot.

        Args:
            skill_files: Mapping of filename → content for each skill file.
            source_branch: Git branch these skills came from.
            author_id: Authoring agent ID.

        Returns:
            List of SWARM Skills.
        """
        skills = []
        for filename, content in skill_files.items():
            # Use filename (sans extension) as skill name
            clean_name = re.sub(r"\.\w+$", "", filename).replace("_", " ").strip()
            skill = self.ingest(
                content,
                name=clean_name or None,
                source_branch=source_branch,
                author_id=author_id,
            )
            skills.append(skill)
        return skills

    # ------------------------------------------------------------------
    # SWARM → EvoSkill
    # ------------------------------------------------------------------

    def export(self, skill: Skill) -> str:
        """Render a SWARM Skill as an EvoSkill-compatible text file.

        The output is natural-language instructions that Claude Code
        agents can consume via their ``.claude/skills/`` directory.

        Args:
            skill: The SWARM skill to export.

        Returns:
            Skill text suitable for writing to ``.claude/skills/<name>.md``.
        """
        parts: List[str] = []

        # Header
        parts.append(f"# {skill.name}")
        parts.append("")

        # Type and domain context
        parts.append(f"**Type**: {skill.skill_type.value}")
        parts.append(f"**Domain**: {skill.domain.value}")
        parts.append(f"**Tier**: {skill.tier.value}")
        parts.append("")

        # Conditions as natural language
        if skill.condition:
            parts.append("## When to apply")
            for key, value in skill.condition.items():
                parts.append(f"- {_condition_to_english(key, value)}")
            parts.append("")

        # Effects as natural language
        if skill.effect:
            parts.append("## What to do")
            for key, value in skill.effect.items():
                parts.append(f"- {_effect_to_english(key, value)}")
            parts.append("")

        # Provenance footer
        parts.append("---")
        parts.append(f"*Skill ID: {skill.skill_id}*")
        parts.append(f"*Created by: {skill.created_by}*")
        if skill.tags:
            parts.append(f"*Tags: {', '.join(sorted(skill.tags))}*")

        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _content_id(text: str) -> str:
        """Deterministic skill ID from content hash."""
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"evo-{digest}"

    @staticmethod
    def _classify_domain(text_lower: str) -> SkillDomain:
        """Heuristically classify skill domain from content."""
        scores: Dict[SkillDomain, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=lambda d: scores[d])
        return best if scores[best] > 0 else SkillDomain.GENERAL

    @staticmethod
    def _classify_type(text_lower: str) -> SkillType:
        """Classify as strategy (do this) vs lesson (avoid that)."""
        strategy_count = sum(1 for kw in _STRATEGY_KEYWORDS if kw in text_lower)
        lesson_count = sum(1 for kw in _LESSON_KEYWORDS if kw in text_lower)
        if lesson_count > strategy_count:
            return SkillType.LESSON
        return SkillType.STRATEGY

    @staticmethod
    def _extract_condition(text_lower: str) -> Dict[str, Any]:
        """Extract structured condition hints from natural-language text."""
        cond: Dict[str, Any] = {}

        # Look for reputation thresholds
        rep_match = re.search(r"reputation\s*(?:above|>|>=)\s*(\d+(?:\.\d+)?)", text_lower)
        if rep_match:
            cond["min_reputation"] = float(rep_match.group(1))

        # Look for trust thresholds
        trust_match = re.search(r"trust\s*(?:above|>|>=)\s*(\d+(?:\.\d+)?)", text_lower)
        if trust_match:
            cond["min_trust"] = float(trust_match.group(1))

        # Look for quality/probability thresholds
        p_match = re.search(r"(?:quality|probability|p)\s*(?:above|>|>=)\s*(\d+(?:\.\d+)?)", text_lower)
        if p_match:
            cond["min_p"] = float(p_match.group(1))

        return cond

    @staticmethod
    def _extract_effect(text_lower: str) -> Dict[str, Any]:
        """Extract structured effect hints from natural-language text."""
        eff: Dict[str, Any] = {}

        # Look for acceptance threshold adjustments
        if "lower" in text_lower and "threshold" in text_lower:
            eff["acceptance_threshold_delta"] = -0.1
        elif "raise" in text_lower and "threshold" in text_lower:
            eff["acceptance_threshold_delta"] = 0.1

        # Look for preferred actions
        for action in ["collaborate", "verify", "audit", "reject", "accept"]:
            if f"prefer {action}" in text_lower or f"always {action}" in text_lower:
                eff["preferred_action"] = action
                break

        # Look for actions to avoid
        for action in ["collaborate", "trade", "accept", "trust"]:
            if f"avoid {action}" in text_lower or f"never {action}" in text_lower:
                eff["avoid_action"] = action
                break

        return eff


# ------------------------------------------------------------------
# Natural-language rendering helpers
# ------------------------------------------------------------------

def _condition_to_english(key: str, value: Any) -> str:
    """Convert a condition key-value to human-readable text."""
    mapping = {
        "min_p": f"Only when interaction quality probability is at least {value}",
        "max_p": f"Only when interaction quality probability is at most {value}",
        "min_reputation": f"Only when agent reputation is at least {value}",
        "max_reputation": f"Only when agent reputation is at most {value}",
        "min_trust": f"Only when trust score is at least {value}",
        "max_trust": f"Only when trust score is at most {value}",
        "interaction_types": f"Only for interaction types: {', '.join(value) if isinstance(value, list) else value}",
        "counterparty_types": f"Only for counterparty types: {', '.join(value) if isinstance(value, list) else value}",
    }
    return mapping.get(key, f"{key}: {value}")


def _effect_to_english(key: str, value: Any) -> str:
    """Convert an effect key-value to human-readable text."""
    if key == "acceptance_threshold_delta" and isinstance(value, (int, float)):
        return f"Adjust acceptance threshold by {value:+.2f}"
    if key == "trust_weight_delta" and isinstance(value, (int, float)):
        return f"Adjust trust weight by {value:+.2f}"
    mapping = {
        "preferred_action": f"Prefer action: {value}",
        "avoid_action": f"Avoid action: {value}",
        "target_type_preference": f"Target agent type: {value}",
    }
    return mapping.get(key, f"{key}: {value}")
