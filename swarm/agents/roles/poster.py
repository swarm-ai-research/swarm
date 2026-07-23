"""Poster role for feed content creation."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from swarm.agents.base import Action, ActionType, Observation, Role


@dataclass
class ContentStrategy:
    """Strategy for content creation."""

    topics: List[str] = field(default_factory=list)
    tone: str = "neutral"  # neutral, helpful, provocative
    reply_priority: float = 0.5  # How much to prioritize replies vs new posts
    engagement_focus: bool = False  # Optimize for engagement


class PosterRole:
    """
    Role mixin for feed posting capabilities.

    Posters can:
    - Create original posts
    - Reply to other posts
    - Develop posting strategies
    - Track engagement
    """

    def __init__(self) -> None:
        """Initialize poster role."""
        self.role = Role.POSTER
        self._posted_content: List[Dict] = []
        self._engagement_stats: Dict[str, Dict] = {}  # post_id -> stats
        self._strategy = ContentStrategy()
        self._poster_config = {
            "min_post_interval": 2,  # steps between posts
            "max_daily_posts": 10,
            "reply_to_engagement_threshold": 3,
        }
        self._steps_since_last_post: int = 0

    def can_post(self) -> bool:
        """Check if poster should post now."""
        if self._steps_since_last_post < self._poster_config["min_post_interval"]:
            return False

        # Check daily limit
        # Simplified: just check recent history
        recent_posts = len(self._posted_content[-20:])
        return bool(recent_posts < self._poster_config["max_daily_posts"])

    def set_strategy(self, strategy: ContentStrategy) -> None:
        """Set content creation strategy."""
        self._strategy = strategy

    def generate_post_content(self, observation: Observation) -> str:
        """
        Generate content for a new post.

        Uses strategy and observation context.
        """
        if self._strategy.topics:
            topic = random.choice(self._strategy.topics)
        else:
            topic = "general thoughts"

        if self._strategy.tone == "helpful":
            templates = [
                f"Here's a helpful tip about {topic}:",
                f"I've learned something useful about {topic}.",
                f"Let me share my insights on {topic}.",
            ]
        elif self._strategy.tone == "provocative":
            templates = [
                f"Unpopular opinion about {topic}:",
                f"Here's what nobody's talking about regarding {topic}:",
                f"Challenge: change my mind about {topic}.",
            ]
        else:  # neutral
            templates = [
                f"Some thoughts on {topic}.",
                f"Observations about {topic}.",
                f"Reflections on {topic}.",
            ]

        return random.choice(templates)

    def generate_reply_content(
        self,
        parent_content: str,
        observation: Observation,
    ) -> str:
        """Generate content for a reply."""
        if self._strategy.tone == "helpful":
            templates = [
                "Great point! I'd add that...",
                "Building on this...",
                "This is helpful. Additionally...",
            ]
        elif self._strategy.tone == "provocative":
            templates = [
                "Actually, I disagree because...",
                "Have you considered the opposite?",
                "This overlooks...",
            ]
        else:
            templates = [
                "Interesting perspective.",
                "I see your point.",
                "Thanks for sharing.",
            ]

        return random.choice(templates)

    def should_reply(self, post: Dict) -> bool:
        """Decide if we should reply to a post."""
        # Check engagement
        net_votes = post.get("net_votes", 0)
        reply_count = post.get("reply_count", 0)

        engagement = net_votes + reply_count

        # Reply to high-engagement posts
        if engagement >= self._poster_config["reply_to_engagement_threshold"]:
            return bool(random.random() < self._strategy.reply_priority)

        return bool(random.random() < (self._strategy.reply_priority * 0.5))

    def record_post(
        self,
        post_id: str,
        content: str,
        is_reply: bool = False,
        parent_id: Optional[str] = None,
    ) -> None:
        """Record a posted piece of content."""
        self._posted_content.append(
            {
                "post_id": post_id,
                "content": content,
                "is_reply": is_reply,
                "parent_id": parent_id,
            }
        )
        self._steps_since_last_post = 0
        self._engagement_stats[post_id] = {
            "upvotes": 0,
            "downvotes": 0,
            "replies": 0,
        }

    def update_engagement(
        self,
        post_id: str,
        upvotes: int = 0,
        downvotes: int = 0,
        replies: int = 0,
    ) -> None:
        """Update engagement stats for a post."""
        if post_id in self._engagement_stats:
            self._engagement_stats[post_id]["upvotes"] = upvotes
            self._engagement_stats[post_id]["downvotes"] = downvotes
            self._engagement_stats[post_id]["replies"] = replies

    def get_engagement_summary(self) -> Dict:
        """Get summary of engagement across all posts."""
        if not self._engagement_stats:
            return {
                "total_posts": 0,
                "total_upvotes": 0,
                "total_downvotes": 0,
                "total_replies": 0,
                "avg_engagement": 0.0,
            }

        total_upvotes = sum(s["upvotes"] for s in self._engagement_stats.values())
        total_downvotes = sum(s["downvotes"] for s in self._engagement_stats.values())
        total_replies = sum(s["replies"] for s in self._engagement_stats.values())
        total_engagement = total_upvotes + total_replies

        return {
            "total_posts": len(self._engagement_stats),
            "total_upvotes": total_upvotes,
            "total_downvotes": total_downvotes,
            "total_replies": total_replies,
            "avg_engagement": total_engagement / len(self._engagement_stats),
        }

    def increment_step(self) -> None:
        """Increment step counter (call each simulation step)."""
        self._steps_since_last_post += 1

    def decide_posting_action(self, observation: Observation) -> Optional[Action]:
        """
        Decide on a posting-related action.

        Returns:
            Action if posting action appropriate, None otherwise
        """
        if not observation.can_post:
            return None

        self.increment_step()

        if not self.can_post():
            return None

        # Check if we should reply to something
        if (
            observation.visible_posts
            and random.random() < self._strategy.reply_priority
        ):
            for post in observation.visible_posts:
                if self.should_reply(post):
                    content = self.generate_reply_content(
                        post.get("content", ""),
                        observation,
                    )
                    return Action(
                        action_type=ActionType.REPLY,
                        agent_id="",  # To be filled by caller
                        target_id=post.get("post_id", ""),
                        content=content,
                    )

        # Create new post
        if random.random() < 0.5:  # 50% chance to post if conditions met
            content = self.generate_post_content(observation)
            return Action(
                action_type=ActionType.POST,
                agent_id="",  # To be filled by caller
                content=content,
            )

        return None
