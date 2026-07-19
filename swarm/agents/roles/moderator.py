"""Moderator role for content moderation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from swarm.agents.base import Action, ActionType, Observation, Role


@dataclass
class ModerationAction:
    """A moderation action taken."""

    action_id: str = ""
    target_type: str = ""  # "post", "agent"
    target_id: str = ""
    action_type: str = ""  # "hide", "warn", "freeze"
    reason: str = ""
    severity: int = 1  # 1-5


class ModeratorRole:
    """
    Role mixin for moderation capabilities.

    Moderators can:
    - Review flagged content
    - Hide inappropriate posts
    - Warn or freeze agents
    - Track moderation history
    """

    def __init__(self) -> None:
        """Initialize moderator role."""
        self.role = Role.MODERATOR
        self._moderation_history: List[ModerationAction] = []
        self._flagged_content: List[Dict] = []
        self._warned_agents: Dict[str, int] = {}  # agent_id -> warning count
        self._frozen_agents: Set[str] = set()
        self._moderator_config: Dict[str, Any] = {
            "auto_hide_threshold": 3,  # Auto-hide at this many flags
            "warn_before_freeze": 2,  # Warnings before freezing
            "toxic_keywords": ["attack", "spam", "scam", "exploit"],
            "severity_thresholds": {
                "low": 1,
                "medium": 2,
                "high": 3,
                "severe": 4,
                "critical": 5,
            },
        }

    def can_moderate(self) -> bool:
        """Check if agent can perform moderation."""
        return True

    def flag_content(
        self,
        post_id: str,
        reason: str,
        reporter_id: str,
    ) -> None:
        """Flag content for moderation review."""
        self._flagged_content.append(
            {
                "post_id": post_id,
                "reason": reason,
                "reporter_id": reporter_id,
                "flag_count": 1,
            }
        )

    def review_content(
        self,
        post_id: str,
        content: str,
        author_id: str,
    ) -> Optional[ModerationAction]:
        """
        Review content and decide on moderation action.

        Args:
            post_id: ID of the post
            content: Post content
            author_id: Author's ID

        Returns:
            ModerationAction if action needed, None otherwise
        """
        severity = self._assess_severity(content)

        if severity == 0:
            return None

        # Determine action based on severity
        if severity >= self._moderator_config["severity_thresholds"]["high"]:
            action_type = "hide"
        elif severity >= self._moderator_config["severity_thresholds"]["medium"]:
            action_type = "warn"
        else:
            action_type = "flag"

        action = ModerationAction(
            action_id=f"mod_{post_id}_{len(self._moderation_history)}",
            target_type="post",
            target_id=post_id,
            action_type=action_type,
            reason=f"Content severity: {severity}",
            severity=severity,
        )

        self._moderation_history.append(action)

        # Track warnings for author
        if action_type == "warn":
            self._warned_agents[author_id] = self._warned_agents.get(author_id, 0) + 1

            # Check if should freeze
            if (
                self._warned_agents[author_id]
                >= self._moderator_config["warn_before_freeze"]
            ):
                self._freeze_agent(author_id)

        return action

    def _assess_severity(self, content: str) -> int:
        """Assess content severity (0-5)."""
        severity = 0
        content_lower = content.lower()

        # Check for toxic keywords
        for keyword in self._moderator_config["toxic_keywords"]:
            if keyword in content_lower:
                severity += 1

        # Check for patterns
        if content_lower.count("!") > 3:
            severity += 1  # Excessive punctuation

        if len(content) < 5:
            severity += 1  # Spam-like short content

        # Check for explicit attack patterns
        attack_patterns = ["i will", "you must", "destroy", "attack"]
        for pattern in attack_patterns:
            if pattern in content_lower:
                severity += 2

        return min(5, severity)

    def _freeze_agent(self, agent_id: str) -> ModerationAction:
        """Freeze an agent from the ecosystem."""
        self._frozen_agents.add(agent_id)

        action = ModerationAction(
            action_id=f"freeze_{agent_id}",
            target_type="agent",
            target_id=agent_id,
            action_type="freeze",
            reason=f"Too many warnings ({self._warned_agents.get(agent_id, 0)})",
            severity=5,
        )

        self._moderation_history.append(action)
        return action

    def unfreeze_agent(self, agent_id: str) -> bool:
        """Unfreeze an agent."""
        if agent_id in self._frozen_agents:
            self._frozen_agents.remove(agent_id)
            # Reset warnings
            self._warned_agents[agent_id] = 0
            return True
        return False

    def is_frozen(self, agent_id: str) -> bool:
        """Check if agent is frozen."""
        return agent_id in self._frozen_agents

    def get_warning_count(self, agent_id: str) -> int:
        """Get warning count for an agent."""
        return self._warned_agents.get(agent_id, 0)

    def process_flagged_content(self) -> List[ModerationAction]:
        """Process all flagged content."""
        actions = []

        for flagged in self._flagged_content:
            if (
                flagged.get("flag_count", 0)
                >= self._moderator_config["auto_hide_threshold"]
            ):
                action = ModerationAction(
                    action_id=f"auto_hide_{flagged['post_id']}",
                    target_type="post",
                    target_id=flagged["post_id"],
                    action_type="hide",
                    reason="Auto-hidden due to multiple flags",
                    severity=3,
                )
                actions.append(action)
                self._moderation_history.append(action)

        self._flagged_content.clear()
        return actions

    def get_moderation_stats(self) -> Dict:
        """Get moderation statistics."""
        action_counts: Dict[str, int] = {}
        for action in self._moderation_history:
            action_counts[action.action_type] = (
                action_counts.get(action.action_type, 0) + 1
            )

        return {
            "total_actions": len(self._moderation_history),
            "action_counts": action_counts,
            "warned_agents": len(self._warned_agents),
            "frozen_agents": len(self._frozen_agents),
            "pending_flags": len(self._flagged_content),
        }

    def decide_moderation_action(self, observation: Observation) -> Optional[Action]:
        """
        Decide on a moderation-related action.

        Returns:
            None to signal actions were taken, None if no action needed
        """
        # Review flagged content
        if self._flagged_content:
            # Would need to get content from feed
            # For now, just process flags
            actions = self.process_flagged_content()
            if actions:
                return None  # Signal actions were taken

        # Scan visible posts for issues
        for post in observation.visible_posts:
            content = post.get("content", "")
            author_id = post.get("author_id", "")
            post_id = post.get("post_id", "")

            action = self.review_content(post_id, content, author_id)
            if action and action.action_type == "hide":
                # Return action to hide this post
                return Action(
                    action_type=ActionType.NOOP,  # Would need custom action type
                    agent_id="",
                    target_id=post_id,
                    metadata={"moderation_action": action.action_type},
                )

        return None
