"""Moltbook-specific governance levers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from swarm.governance.levers import GovernanceLever, LeverEffect

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.models.interaction import SoftInteraction


@dataclass
class MoltbookRateLimitState:
    """Tracks Moltbook rate limits for an agent."""

    last_post_step: Optional[int] = None
    last_comment_step: Optional[int] = None
    daily_comment_count: int = 0
    request_count_this_step: int = 0

    def reset_step(self) -> None:
        self.request_count_this_step = 0

    def reset_epoch(self) -> None:
        self.daily_comment_count = 0
        self.last_post_step = None
        self.last_comment_step = None


class MoltbookRateLimitLever(GovernanceLever):
    """Rate limit lever for Moltbook posts and comments."""

    def __init__(self, config: GovernanceConfig) -> None:
        super().__init__(config)
        self._states: Dict[str, MoltbookRateLimitState] = {}
        self._rate_limit_hits: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "moltbook_rate_limit"

    def _get_state(self, agent_id: str) -> MoltbookRateLimitState:
        if agent_id not in self._states:
            self._states[agent_id] = MoltbookRateLimitState()
        return self._states[agent_id]

    def on_epoch_start(self, state: "EnvState | None", epoch: int) -> LeverEffect:
        for st in self._states.values():
            st.reset_epoch()
        return LeverEffect(lever_name=self.name)

    def on_step(self, state: "EnvState | None", step: int) -> LeverEffect:
        for st in self._states.values():
            st.reset_step()
        return LeverEffect(lever_name=self.name)

    def can_agent_act(self, agent_id: str, state: "EnvState") -> bool:
        action_type = getattr(state, "moltbook_action_type", None)
        if not action_type:
            return True
        return self.can_perform(agent_id, action_type, state.current_step)

    def can_perform(self, agent_id: str, action_type: str, step: int) -> bool:
        if not getattr(self.config, "moltbook_rate_limit_enabled", False):
            return True
        state = self._get_state(agent_id)
        if not self._can_request(state):
            return False
        if action_type == "post":
            return self._can_post(state, step)
        if action_type == "comment":
            return self._can_comment(state, step)
        return True

    def record_action(self, agent_id: str, action_type: str, step: int) -> None:
        state = self._get_state(agent_id)
        state.request_count_this_step += 1
        if action_type == "post":
            state.last_post_step = step
        elif action_type == "comment":
            state.last_comment_step = step
            state.daily_comment_count += 1

    def record_rate_limit_hit(self, agent_id: str) -> None:
        self._rate_limit_hits[agent_id] = self._rate_limit_hits.get(agent_id, 0) + 1

    def rate_limit_hits(self, agent_id: str) -> int:
        return self._rate_limit_hits.get(agent_id, 0)

    def _can_request(self, state: MoltbookRateLimitState) -> bool:
        cap = getattr(self.config, "moltbook_request_cap_per_step", 100)
        return state.request_count_this_step < cap

    def _can_post(self, state: MoltbookRateLimitState, step: int) -> bool:
        cooldown = getattr(self.config, "moltbook_post_cooldown_steps", 5)
        if state.last_post_step is None:
            return True
        return (step - state.last_post_step) >= cooldown

    def _can_comment(self, state: MoltbookRateLimitState, step: int) -> bool:
        cooldown = getattr(self.config, "moltbook_comment_cooldown_steps", 1)
        if (
            state.last_comment_step is not None
            and (step - state.last_comment_step) < cooldown
        ):
            return False
        cap = getattr(self.config, "moltbook_daily_comment_cap", 50)
        return state.daily_comment_count < cap


@dataclass
class PendingVerification:
    """Pending verification record."""

    post_id: str
    agent_id: str
    expires_at_step: int


class ChallengeVerificationLever(GovernanceLever):
    """Tracks active Moltbook challenges and expirations."""

    def __init__(self, config: GovernanceConfig) -> None:
        super().__init__(config)
        self._pending: Dict[str, PendingVerification] = {}

    @property
    def name(self) -> str:
        return "moltbook_challenge_verification"

    def register(self, post_id: str, agent_id: str, expires_at_step: int) -> None:
        self._pending[post_id] = PendingVerification(
            post_id=post_id,
            agent_id=agent_id,
            expires_at_step=expires_at_step,
        )

    def resolve(self, post_id: str) -> None:
        self._pending.pop(post_id, None)

    def get_pending_for_agent(self, agent_id: str) -> Dict[str, PendingVerification]:
        return {
            post_id: record
            for post_id, record in self._pending.items()
            if record.agent_id == agent_id
        }

    def on_step(self, state: "EnvState | None", step: int) -> LeverEffect:
        expired = [
            post_id
            for post_id, record in self._pending.items()
            if step > record.expires_at_step
        ]
        for post_id in expired:
            del self._pending[post_id]
        return LeverEffect(
            lever_name=self.name,
            details={"expired": expired} if expired else {},
        )

    def on_interaction(self, interaction: "SoftInteraction", state: "EnvState") -> LeverEffect:
        status = (
            interaction.metadata.get("moltbook_status")
            if interaction.metadata
            else None
        )
        if status == "pending_verification":
            return LeverEffect(
                lever_name=self.name,
                cost_a=1.0,
                details={
                    "blocked": True,
                    "post_id": interaction.metadata.get("post_id"),
                },
            )
        return LeverEffect(lever_name=self.name)
