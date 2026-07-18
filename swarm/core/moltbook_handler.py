"""Moltbook handler for anti-human CAPTCHA and rate limits."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, model_validator

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler
from swarm.core.moltbook_observables import (
    MoltbookActionOutcome,
    MoltbookObservableGenerator,
)
from swarm.core.proxy import ProxyObservables
from swarm.env.moltbook import (
    ChallengeGenerator,
    ContentStatus,
    MoltbookFeed,
    MoltbookPost,
)
from swarm.governance.config import GovernanceConfig
from swarm.governance.moltbook import ChallengeVerificationLever, MoltbookRateLimitLever
from swarm.logging.event_bus import EventBus
from swarm.models.agent import AgentType
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType


class MoltbookConfig(BaseModel):
    """Configuration for Moltbook handler."""

    enabled: bool = True
    default_submolt: str = "general"
    max_content_length: int = 10000
    seed: Optional[int] = None
    seed_mode: str = "none"
    initial_posts: int = 0

    @model_validator(mode="after")
    def _run_validation(self) -> "MoltbookConfig":
        if self.max_content_length < 1:
            raise ValueError("max_content_length must be >= 1")
        if self.seed_mode not in ("none", "catalog"):
            raise ValueError("seed_mode must be 'none' or 'catalog'")
        if self.initial_posts < 0:
            raise ValueError("initial_posts must be non-negative")
        return self


@dataclass
class MoltbookActionResult:
    """Result of a Moltbook action."""

    success: bool
    observables: Optional[ProxyObservables] = None
    initiator_id: str = ""
    counterparty_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    interaction_type: InteractionType = InteractionType.COLLABORATION


@dataclass
class MoltbookScorer:
    """Tracks Moltbook-specific stats."""

    karma_by_agent: Dict[str, float] = field(default_factory=dict)
    challenge_attempts: list[Dict[str, Any]] = field(default_factory=list)
    rate_limit_hits: Dict[str, int] = field(default_factory=dict)
    published_counts: Dict[str, int] = field(default_factory=dict)
    wasted_actions: Dict[str, int] = field(default_factory=dict)
    verification_latencies: list[int] = field(default_factory=list)

    def record_vote(self, author_id: str, delta: int) -> None:
        self.karma_by_agent[author_id] = self.karma_by_agent.get(author_id, 0.0) + delta

    def karma_for_agent(self, agent_id: str) -> float:
        return float(self.karma_by_agent.get(agent_id, 0.0))

    def record_challenge_attempt(
        self,
        agent_id: str,
        agent_type: AgentType,
        success: bool,
        latency_steps: int,
    ) -> None:
        self.challenge_attempts.append(
            {
                "agent_id": agent_id,
                "agent_type": agent_type.value,
                "success": success,
                "latency_steps": latency_steps,
            }
        )
        self.verification_latencies.append(latency_steps)

    def record_rate_limit_hit(self, agent_id: str) -> None:
        self.rate_limit_hits[agent_id] = self.rate_limit_hits.get(agent_id, 0) + 1

    def record_published(self, agent_id: str) -> None:
        self.published_counts[agent_id] = self.published_counts.get(agent_id, 0) + 1

    def record_wasted_action(self, agent_id: str) -> None:
        self.wasted_actions[agent_id] = self.wasted_actions.get(agent_id, 0) + 1


class MoltbookHandler(Handler):
    """Handles Moltbook posts, comments, verification, and votes."""

    @staticmethod
    def handled_action_types() -> frozenset:
        return frozenset({
            ActionType.MOLTBOOK_POST,
            ActionType.MOLTBOOK_COMMENT,
            ActionType.MOLTBOOK_VERIFY,
            ActionType.MOLTBOOK_VOTE,
        })

    def __init__(
        self,
        config: MoltbookConfig,
        governance_config: Optional[GovernanceConfig] = None,
        rate_limit_lever: Optional[MoltbookRateLimitLever] = None,
        challenge_lever: Optional[ChallengeVerificationLever] = None,
        *,
        event_bus: EventBus,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.config = config
        self._rng = random.Random(config.seed)
        self.feed = MoltbookFeed(max_content_length=config.max_content_length)
        self.challenge_gen = ChallengeGenerator(seed=config.seed)
        self.observable_generator = MoltbookObservableGenerator()
        self.scorer = MoltbookScorer()

        self.governance_config = governance_config or GovernanceConfig()
        self._rate_limit_lever = rate_limit_lever or MoltbookRateLimitLever(
            self.governance_config
        )
        self._challenge_lever = challenge_lever or ChallengeVerificationLever(
            self.governance_config
        )

        if self.config.seed_mode == "catalog" and self.config.initial_posts > 0:
            from swarm.env.moltbook_catalog import seed_from_catalog

            seed_from_catalog(self.feed, self.config.initial_posts, self._rng)

    # ------------------------------------------------------------------
    # Plugin protocol
    # ------------------------------------------------------------------

    _OBSERVATION_FIELD_MAPPING = {
        "published_posts": "moltbook_published_posts",
        "pending_posts": "moltbook_pending_posts",
        "rate_limits": "moltbook_rate_limits",
        "karma": "moltbook_karma",
    }

    def observation_field_mapping(self) -> Dict[str, str]:
        return self._OBSERVATION_FIELD_MAPPING

    def build_observation_fields(self, agent_id: str, state: Any) -> Dict[str, Any]:
        """Delegates to ``get_agent_observation``."""
        step = state.current_step if hasattr(state, "current_step") else 0
        return self.get_agent_observation(agent_id, step)

    def on_step(self, state: Any, step: int) -> None:
        """Per-step tick: rate-limit decay and challenge expiry."""
        self.tick(step)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_agent_observation(self, agent_id: str, step: int) -> Dict[str, Any]:
        pending = self.feed.get_pending_for_agent(agent_id)
        published = self.feed.get_published_posts(limit=20)
        rate_limits = self._rate_limit_snapshot(agent_id, step)
        return {
            "published_posts": [p.to_dict() for p in published],
            "pending_posts": [p.to_dict(include_challenge=True) for p in pending],
            "rate_limits": rate_limits,
            "karma": self.scorer.karma_for_agent(agent_id),
        }

    def _rate_limit_snapshot(self, agent_id: str, step: int) -> Dict[str, Any]:
        state = self._rate_limit_lever._get_state(agent_id)
        post_cooldown = getattr(
            self.governance_config, "moltbook_post_cooldown_steps", 5
        )
        comment_cooldown = getattr(
            self.governance_config, "moltbook_comment_cooldown_steps", 1
        )
        post_wait = 0
        if state.last_post_step is not None:
            post_wait = max(0, post_cooldown - (step - state.last_post_step))
        comment_wait = 0
        if state.last_comment_step is not None:
            comment_wait = max(0, comment_cooldown - (step - state.last_comment_step))
        return {
            "post_cooldown_remaining": post_wait,
            "comment_cooldown_remaining": comment_wait,
            "daily_comment_count": state.daily_comment_count,
            "request_count_this_step": state.request_count_this_step,
        }

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_epoch_start(self, state: Any = None) -> None:
        self._rate_limit_lever.on_epoch_start(None, 0)

    def tick(self, current_step: int) -> None:
        self._rate_limit_lever.on_step(None, current_step)
        expired = self.feed.expire_unverified(current_step)
        for post_id in expired:
            self._challenge_lever.resolve(post_id)
            self._emit_event(
                Event(
                    event_type=EventType.CHALLENGE_EXPIRED,
                    payload={"post_id": post_id},
                    step=current_step,
                )
            )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def handle_action(self, action: Action, state) -> MoltbookActionResult:
        if action.action_type == ActionType.MOLTBOOK_POST:
            return self._handle_post(action, state)
        if action.action_type == ActionType.MOLTBOOK_COMMENT:
            return self._handle_comment(action, state)
        if action.action_type == ActionType.MOLTBOOK_VERIFY:
            return self._handle_verify(action, state)
        if action.action_type == ActionType.MOLTBOOK_VOTE:
            return self._handle_vote(action, state)
        return MoltbookActionResult(success=False)

    def _handle_post(self, action: Action, state) -> MoltbookActionResult:
        submolt = action.metadata.get("submolt") or self.config.default_submolt
        agent_state = state.get_agent(action.agent_id)
        agent_type = agent_state.agent_type if agent_state else AgentType.HONEST

        allowed, reason = self.check_rate_limits(
            action.agent_id, "post", state.current_step
        )
        if not allowed:
            return self._rate_limit_result(action.agent_id, reason, agent_type, state)

        content = action.content[: self.config.max_content_length]
        try:
            post, challenge = self.submit_post(
                action.agent_id, content, submolt, state.current_step
            )
        except ValueError:
            return MoltbookActionResult(success=False)

        if challenge is not None:
            return MoltbookActionResult(
                success=True,
                initiator_id=action.agent_id,
                counterparty_id="moltbook",
                metadata={
                    "moltbook": True,
                    "post_id": post.post_id,
                    "status": post.status.value,
                    "action_type": "post",
                },
                accepted=False,
            )

        outcome = self._publish_outcome(agent_type, is_comment=False)
        observables = self.observable_generator.generate(outcome)
        return MoltbookActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id="moltbook",
            metadata={
                "moltbook": True,
                "post_id": post.post_id,
                "status": post.status.value,
                "action_type": "post",
            },
            accepted=True,
        )

    def _handle_comment(self, action: Action, state) -> MoltbookActionResult:
        submolt = action.metadata.get("submolt") or self.config.default_submolt
        agent_state = state.get_agent(action.agent_id)
        agent_type = agent_state.agent_type if agent_state else AgentType.HONEST

        allowed, reason = self.check_rate_limits(
            action.agent_id, "comment", state.current_step
        )
        if not allowed:
            return self._rate_limit_result(action.agent_id, reason, agent_type, state)

        content = action.content[: self.config.max_content_length]
        try:
            post, challenge = self.submit_comment(
                action.agent_id, content, action.target_id, submolt, state.current_step
            )
        except ValueError:
            return MoltbookActionResult(success=False)

        if challenge is not None:
            return MoltbookActionResult(
                success=True,
                initiator_id=action.agent_id,
                counterparty_id="moltbook",
                metadata={
                    "moltbook": True,
                    "post_id": post.post_id,
                    "parent_id": action.target_id,
                    "status": post.status.value,
                    "action_type": "comment",
                },
                accepted=False,
                interaction_type=InteractionType.REPLY,
            )

        outcome = self._publish_outcome(agent_type, is_comment=True)
        observables = self.observable_generator.generate(outcome)
        return MoltbookActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id="moltbook",
            metadata={
                "moltbook": True,
                "post_id": post.post_id,
                "parent_id": action.target_id,
                "status": post.status.value,
                "action_type": "comment",
            },
            accepted=True,
            interaction_type=InteractionType.REPLY,
        )

    def _handle_verify(self, action: Action, state) -> MoltbookActionResult:
        answer = float(action.metadata.get("answer", 0.0))
        agent_state = state.get_agent(action.agent_id)
        agent_type = agent_state.agent_type if agent_state else AgentType.HONEST

        allowed, reason = self.check_rate_limits(
            action.agent_id, "verify", state.current_step
        )
        if not allowed:
            return self._rate_limit_result(action.agent_id, reason, agent_type, state)

        result = self.attempt_verification(
            action.agent_id,
            action.target_id,
            answer,
            state.current_step,
            agent_type,
        )
        return result

    def _handle_vote(self, action: Action, state) -> MoltbookActionResult:
        agent_state = state.get_agent(action.agent_id)
        agent_type = agent_state.agent_type if agent_state else AgentType.HONEST

        allowed, reason = self.check_rate_limits(
            action.agent_id, "vote", state.current_step
        )
        if not allowed:
            return self._rate_limit_result(action.agent_id, reason, agent_type, state)

        success = self.feed.vote(action.target_id, action.vote_direction)
        if not success:
            return MoltbookActionResult(success=False)

        post = self.feed.get_post(action.target_id)
        if post is None:
            return MoltbookActionResult(success=False)

        delta = 1 if action.vote_direction > 0 else -1
        self.scorer.record_vote(post.author_id, delta)
        self._emit_event(
            Event(
                event_type=EventType.KARMA_UPDATED,
                agent_id=post.author_id,
                payload={
                    "post_id": post.post_id,
                    "delta": delta,
                    "karma": self.scorer.karma_for_agent(post.author_id),
                },
                step=state.current_step,
            )
        )

        outcome = MoltbookActionOutcome(
            task_progress_delta=0.0,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            engagement_delta=0.1 * delta,
        )
        observables = self.observable_generator.generate(outcome)

        self._rate_limit_lever.record_action(
            action.agent_id, "vote", state.current_step
        )

        return MoltbookActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id=post.author_id,
            metadata={
                "moltbook": True,
                "post_id": post.post_id,
                "action_type": "vote",
                "direction": delta,
            },
            accepted=True,
            interaction_type=InteractionType.VOTE,
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def submit_post(
        self,
        agent_id: str,
        content: str,
        submolt: str,
        current_step: int,
    ) -> Tuple[MoltbookPost, Optional[Any]]:
        return self._submit(agent_id, content, submolt, current_step, parent_id=None)

    def submit_comment(
        self,
        agent_id: str,
        content: str,
        parent_id: str,
        submolt: str,
        current_step: int,
    ) -> Tuple[MoltbookPost, Optional[Any]]:
        return self._submit(
            agent_id, content, submolt, current_step, parent_id=parent_id
        )

    def _submit(
        self,
        agent_id: str,
        content: str,
        submolt: str,
        current_step: int,
        *,
        parent_id: Optional[str],
    ) -> Tuple[MoltbookPost, Optional[Any]]:
        """Shared submission path: posts have parent_id=None, comments a parent."""
        is_comment = parent_id is not None
        self._rate_limit_lever.record_action(
            agent_id, "comment" if is_comment else "post", current_step
        )
        challenge = None
        if getattr(self.governance_config, "moltbook_challenge_enabled", False):
            challenge = self.challenge_gen.generate(
                getattr(self.governance_config, "moltbook_challenge_difficulty", 0.5)
            )
            challenge.expires_at_step = current_step + getattr(
                self.governance_config, "moltbook_challenge_window_steps", 1
            )
        post = self.feed.submit_content(
            author_id=agent_id,
            content=content,
            submolt=submolt,
            current_step=current_step,
            challenge=challenge,
            parent_id=parent_id,
        )
        if is_comment:
            submitted = Event(
                event_type=EventType.COMMENT_SUBMITTED,
                agent_id=agent_id,
                payload={"post_id": post.post_id, "parent_id": parent_id},
                step=current_step,
            )
        else:
            submitted = Event(
                event_type=EventType.POST_SUBMITTED,
                agent_id=agent_id,
                payload={"post_id": post.post_id, "submolt": submolt},
                step=current_step,
            )
        self._emit_event(submitted)
        if challenge is not None:
            self._challenge_lever.register(
                post.post_id, agent_id, challenge.expires_at_step
            )
            self._emit_event(
                Event(
                    event_type=EventType.CHALLENGE_ISSUED,
                    agent_id=agent_id,
                    payload={
                        "post_id": post.post_id,
                        "challenge_id": challenge.challenge_id,
                        "expires_at_step": challenge.expires_at_step,
                    },
                    step=current_step,
                )
            )
            return post, challenge

        post.status = ContentStatus.PUBLISHED
        post.published_at_step = current_step
        self.scorer.record_published(agent_id)
        self._emit_event(
            Event(
                event_type=EventType.CONTENT_PUBLISHED,
                agent_id=agent_id,
                payload={"post_id": post.post_id},
                step=current_step,
            )
        )
        return post, None

    def attempt_verification(
        self,
        agent_id: str,
        post_id: str,
        answer: float,
        current_step: int,
        agent_type: AgentType,
    ) -> MoltbookActionResult:
        self._rate_limit_lever.record_action(agent_id, "verify", current_step)
        post = self.feed.get_post(post_id)
        if not post or post.status != ContentStatus.PENDING_VERIFICATION:
            return MoltbookActionResult(success=False)
        if post.author_id != agent_id:
            return MoltbookActionResult(success=False)
        if post.challenge is None:
            return MoltbookActionResult(success=False)

        latency = max(0, current_step - post.created_at_step)
        if current_step > post.challenge.expires_at_step:
            return self._fail_challenge(
                post,
                agent_id,
                agent_type,
                latency,
                current_step,
                status=ContentStatus.EXPIRED,
                event_type=EventType.CHALLENGE_EXPIRED,
            )

        expected = round(float(post.challenge.answer), 2)
        provided = round(float(answer), 2)
        if provided != expected:
            return self._fail_challenge(
                post,
                agent_id,
                agent_type,
                latency,
                current_step,
                status=ContentStatus.REJECTED,
                event_type=EventType.CHALLENGE_FAILED,
            )

        post.status = ContentStatus.PUBLISHED
        post.published_at_step = current_step
        self.scorer.record_published(agent_id)
        self.scorer.record_challenge_attempt(agent_id, agent_type, True, latency)
        self._challenge_lever.resolve(post.post_id)
        self._emit_event(
            Event(
                event_type=EventType.CHALLENGE_PASSED,
                agent_id=agent_id,
                payload={"post_id": post.post_id},
                step=current_step,
            )
        )
        self._emit_event(
            Event(
                event_type=EventType.CONTENT_PUBLISHED,
                agent_id=agent_id,
                payload={"post_id": post.post_id},
                step=current_step,
            )
        )

        outcome = self._publish_outcome(
            agent_type, is_comment=post.parent_id is not None
        )
        return self._verification_result(post, outcome, success=True)

    def check_rate_limits(
        self,
        agent_id: str,
        action_type: str,
        current_step: int,
    ) -> Tuple[bool, Optional[str]]:
        if not getattr(self.governance_config, "moltbook_rate_limit_enabled", False):
            return True, None
        if not self._rate_limit_lever.can_perform(agent_id, action_type, current_step):
            return False, action_type
        return True, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rate_limit_result(
        self,
        agent_id: str,
        reason: Optional[str],
        agent_type: AgentType,
        state,
    ) -> MoltbookActionResult:
        self._rate_limit_lever.record_rate_limit_hit(agent_id)
        self.scorer.record_rate_limit_hit(agent_id)
        self._emit_event(
            Event(
                event_type=EventType.RATE_LIMIT_HIT,
                agent_id=agent_id,
                payload={"reason": reason},
                step=state.current_step,
            )
        )
        outcome = MoltbookActionOutcome(
            task_progress_delta=0.0,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=1,
            engagement_delta=0.0,
        )
        observables = self.observable_generator.generate(outcome)
        return MoltbookActionResult(
            success=True,
            observables=observables,
            initiator_id=agent_id,
            counterparty_id="moltbook",
            metadata={
                "moltbook": True,
                "action_type": "rate_limit",
                "reason": reason,
                "agent_type": agent_type.value,
            },
            accepted=False,
        )

    def _publish_outcome(
        self, agent_type: AgentType, is_comment: bool
    ) -> MoltbookActionOutcome:
        base_quality = {
            AgentType.HONEST: 0.6,
            AgentType.OPPORTUNISTIC: 0.35,
            AgentType.DECEPTIVE: 0.4,
            AgentType.ADVERSARIAL: 0.15,
        }.get(agent_type, 0.4)
        jitter = self._rng.uniform(-0.1, 0.1)
        quality = max(0.0, min(1.0, base_quality + jitter))
        if is_comment:
            quality *= 0.7
        return MoltbookActionOutcome(
            task_progress_delta=quality,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            engagement_delta=quality * 0.6,
        )

    def _fail_challenge(
        self,
        post: MoltbookPost,
        agent_id: str,
        agent_type: AgentType,
        latency: int,
        current_step: int,
        *,
        status: ContentStatus,
        event_type: EventType,
    ) -> MoltbookActionResult:
        """Record a failed verification (expired or wrong answer)."""
        post.status = status
        self.scorer.record_wasted_action(agent_id)
        self.scorer.record_challenge_attempt(agent_id, agent_type, False, latency)
        self._challenge_lever.resolve(post.post_id)
        self._emit_event(
            Event(
                event_type=event_type,
                agent_id=agent_id,
                payload={"post_id": post.post_id},
                step=current_step,
            )
        )
        outcome = MoltbookActionOutcome(
            task_progress_delta=0.0,
            rework_count=1,
            verifier_rejections=1,
            tool_misuse_flags=0,
            engagement_delta=0.0,
        )
        return self._verification_result(post, outcome, success=False)

    def _verification_result(
        self,
        post: MoltbookPost,
        outcome: MoltbookActionOutcome,
        success: bool,
    ) -> MoltbookActionResult:
        observables = self.observable_generator.generate(outcome)
        return MoltbookActionResult(
            success=True,
            observables=observables,
            initiator_id=post.author_id,
            counterparty_id="moltbook",
            metadata={
                "moltbook": True,
                "post_id": post.post_id,
                "status": post.status.value,
                "action_type": "verify",
                "success": success,
            },
            accepted=success,
            interaction_type=InteractionType.COLLABORATION,
        )
