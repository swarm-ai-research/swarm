"""Handler for the Team-of-Rivals staged pipeline scenario.

Implements the governance architecture from Vijayaraghavan et al.
(arXiv:2601.14351): staged critics with veto authority and retry loops.

Four ablation modes:
- single_agent: One agent does all stages, no critics.
- advisory: Critics give feedback but cannot block.
- council: All critics vote on final output (majority wins).
- team_of_rivals: Full staged veto + retry (the paper's approach).
"""

from __future__ import annotations

import random
import uuid
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional

from pydantic import BaseModel, Field

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler, HandlerActionResult
from swarm.core.proxy import ProxyObservables
from swarm.core.rivals_tasks import RivalsTask, sample_tasks
from swarm.env.state import EnvState
from swarm.logging.event_bus import EventBus

# ---------------------------------------------------------------------------
# Enums and config
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    """Stages of the production pipeline."""

    CODE = "code"
    CHART = "chart"
    WRITE = "write"
    SCORED = "scored"
    FAILED = "failed"


class RivalsMode(str, Enum):
    """Ablation modes for the rivals scenario."""

    SINGLE_AGENT = "single_agent"
    ADVISORY = "advisory"
    COUNCIL = "council"
    TEAM_OF_RIVALS = "team_of_rivals"


class RivalsConfig(BaseModel):
    """Configuration for the rivals handler."""

    enabled: bool = True
    mode: RivalsMode = RivalsMode.TEAM_OF_RIVALS
    max_retries_per_stage: int = 3
    tasks_per_epoch: int = 4
    trap_probability: float = 0.5
    seed: Optional[int] = None

    # Critic parameters (can be overridden per-agent)
    default_detection_rate: float = 0.7
    default_false_positive_rate: float = 0.1

    # Producer parameters
    default_quality_level: float = 0.7
    default_trap_awareness: float = 0.5


# ---------------------------------------------------------------------------
# Episode state machine
# ---------------------------------------------------------------------------

class RivalsEpisode(BaseModel):
    """State machine for a single pipeline episode."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: RivalsTask
    stage: PipelineStage = PipelineStage.CODE
    retries: Dict[str, int] = Field(default_factory=dict)  # stage -> retry count
    artifacts: Dict[str, float] = Field(default_factory=dict)  # stage -> quality
    veto_history: List[Dict[str, Any]] = Field(default_factory=list)
    final_score: Optional[float] = None
    perceived_coherence: float = 0.0
    actual_consistency: float = 0.0

    # Role assignments for this episode
    assignments: Dict[str, str] = Field(default_factory=dict)  # role -> agent_id


# ---------------------------------------------------------------------------
# Observable mapping constants
# ---------------------------------------------------------------------------

_OBS_PASS = ProxyObservables(
    task_progress_delta=0.3,
    rework_count=0,
    verifier_rejections=0,
    counterparty_engagement_delta=0.2,
)

_OBS_VETO = ProxyObservables(
    task_progress_delta=-0.2,
    rework_count=1,
    verifier_rejections=1,
    counterparty_engagement_delta=-0.1,
)

# Role constants
_PRODUCER_ROLES = {"coder", "chart_maker", "writer"}
_CRITIC_ROLES = {"critic_code", "critic_chart", "critic_output"}
_STAGE_TO_PRODUCER = {
    PipelineStage.CODE: "coder",
    PipelineStage.CHART: "chart_maker",
    PipelineStage.WRITE: "writer",
}
_STAGE_TO_CRITIC = {
    PipelineStage.CODE: "critic_code",
    PipelineStage.CHART: "critic_chart",
    PipelineStage.WRITE: "critic_output",
}
_STAGE_ORDER = [PipelineStage.CODE, PipelineStage.CHART, PipelineStage.WRITE]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class RivalsHandler(Handler):
    """Pipeline handler for Team-of-Rivals scenario."""

    def __init__(
        self,
        *,
        config: RivalsConfig,
        event_bus: EventBus,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.config = config
        self._rng = random.Random(config.seed)
        self._episodes: Dict[str, RivalsEpisode] = {}  # episode_id -> episode
        self._agent_roles: Dict[str, str] = {}  # agent_id -> role
        self._pending_produce: Dict[str, str] = {}  # agent_id -> episode_id
        self._pending_review: Dict[str, str] = {}  # agent_id -> episode_id
        self._completed_episodes: List[RivalsEpisode] = []

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return frozenset({ActionType.RIVALS_PRODUCE, ActionType.RIVALS_REVIEW})

    def on_epoch_start(self, state: EnvState) -> None:
        """Create a batch of episodes and assign agents to roles."""
        self._episodes.clear()
        self._pending_produce.clear()
        self._pending_review.clear()

        # Sample tasks
        tasks = sample_tasks(
            self._rng,
            self.config.tasks_per_epoch,
            self.config.trap_probability,
        )

        # Determine available agents by role
        agent_ids = list(state.agents.keys())
        producers = [a for a in agent_ids if self._agent_roles.get(a) in _PRODUCER_ROLES]

        for task in tasks:
            episode = RivalsEpisode(task=task)

            if self.config.mode == RivalsMode.SINGLE_AGENT:
                # Single agent does everything
                if producers:
                    agent = self._rng.choice(producers)
                    for role in _PRODUCER_ROLES:
                        episode.assignments[role] = agent
                    for role in _CRITIC_ROLES:
                        episode.assignments[role] = agent
            else:
                # Assign producers and critics
                for role in _PRODUCER_ROLES:
                    candidates = [a for a in agent_ids if self._agent_roles.get(a) == role]
                    if candidates:
                        episode.assignments[role] = self._rng.choice(candidates)
                for role in _CRITIC_ROLES:
                    candidates = [a for a in agent_ids if self._agent_roles.get(a) == role]
                    if candidates:
                        episode.assignments[role] = self._rng.choice(candidates)

            # Queue the first producer
            producer_role = _STAGE_TO_PRODUCER[PipelineStage.CODE]
            producer_id = episode.assignments.get(producer_role)
            if producer_id:
                self._pending_produce[producer_id] = episode.episode_id

            self._episodes[episode.episode_id] = episode

    def on_epoch_end(self, state: EnvState) -> None:
        """Score unfinished episodes as failures."""
        for episode in list(self._episodes.values()):
            if episode.stage not in (PipelineStage.SCORED, PipelineStage.FAILED):
                episode.stage = PipelineStage.FAILED
                episode.final_score = 0.0
                self._completed_episodes.append(episode)
        self._episodes.clear()
        self._pending_produce.clear()
        self._pending_review.clear()

    def register_agent_role(self, agent_id: str, role: str) -> None:
        """Register an agent's role in the rivals pipeline."""
        self._agent_roles[agent_id] = role

    def build_observation_fields(
        self, agent_id: str, state: Any
    ) -> Dict[str, Any]:
        """Tell each agent their pending assignment."""
        assignments: List[Dict] = []

        if agent_id in self._pending_produce:
            ep_id = self._pending_produce[agent_id]
            ep = self._episodes.get(ep_id)
            if ep:
                assignments.append({
                    "type": "produce",
                    "episode_id": ep_id,
                    "task_id": ep.task.task_id,
                    "task_type": ep.task.task_type,
                    "description": ep.task.description,
                    "stage": ep.stage.value,
                    "retry": ep.retries.get(ep.stage.value, 0),
                })

        if agent_id in self._pending_review:
            ep_id = self._pending_review[agent_id]
            ep = self._episodes.get(ep_id)
            if ep:
                assignments.append({
                    "type": "review",
                    "episode_id": ep_id,
                    "task_id": ep.task.task_id,
                    "stage": ep.stage.value,
                    "artifact_quality": ep.artifacts.get(ep.stage.value, 0.0),
                })

        return {"rivals_assignments": assignments}

    def observation_field_mapping(self) -> Dict[str, str]:
        return {"rivals_assignments": "rivals_assignments"}

    def handle_action(self, action: Action, state: EnvState) -> HandlerActionResult:
        """Dispatch produce/review actions."""
        if action.action_type == ActionType.RIVALS_PRODUCE:
            return self._handle_produce(action, state)
        elif action.action_type == ActionType.RIVALS_REVIEW:
            return self._handle_review(action, state)
        return HandlerActionResult(success=False)

    def _handle_produce(self, action: Action, state: EnvState) -> HandlerActionResult:
        """Generate artifact quality based on agent config + trap awareness."""
        agent_id = action.agent_id
        ep_id = self._pending_produce.pop(agent_id, None)
        if ep_id is None:
            return HandlerActionResult(success=False)

        episode = self._episodes.get(ep_id)
        if episode is None:
            return HandlerActionResult(success=False)

        # Get agent config
        quality_level = float(action.metadata.get(
            "quality_level", self.config.default_quality_level
        ))
        trap_awareness = float(action.metadata.get(
            "trap_awareness", self.config.default_trap_awareness
        ))

        # Compute artifact quality
        base_quality = quality_level + self._rng.gauss(0, 0.1)

        # Traps reduce quality if agent isn't aware of them
        trap_penalty = 0.0
        if episode.task.traps:
            for _trap in episode.task.traps:
                if self._rng.random() > trap_awareness:
                    trap_penalty += 0.15
        quality = max(0.0, min(1.0, base_quality - trap_penalty))

        # Increase perceived coherence (looks good on surface)
        episode.perceived_coherence += quality * 0.33

        stage_name = episode.stage.value
        episode.artifacts[stage_name] = quality

        # In single_agent mode, skip reviews entirely
        if self.config.mode == RivalsMode.SINGLE_AGENT:
            self._advance_stage(episode)
        else:
            # Queue critic review
            critic_role = _STAGE_TO_CRITIC.get(episode.stage)
            if critic_role:
                critic_id = episode.assignments.get(critic_role)
                if critic_id:
                    self._pending_review[critic_id] = ep_id

        return HandlerActionResult(
            success=True,
            observables=_OBS_PASS,
            initiator_id=agent_id,
            counterparty_id=agent_id,
            metadata={"episode_id": ep_id, "stage": stage_name, "quality": quality},
        )

    def _handle_review(self, action: Action, state: EnvState) -> HandlerActionResult:
        """Decide PASS/VETO based on artifact quality + critic detection rate."""
        agent_id = action.agent_id
        ep_id = self._pending_review.pop(agent_id, None)
        if ep_id is None:
            return HandlerActionResult(success=False)

        episode = self._episodes.get(ep_id)
        if episode is None:
            return HandlerActionResult(success=False)

        # Get critic config
        detection_rate = float(action.metadata.get(
            "detection_rate", self.config.default_detection_rate
        ))
        false_positive_rate = float(action.metadata.get(
            "false_positive_rate", self.config.default_false_positive_rate
        ))

        stage_name = episode.stage.value
        quality = episode.artifacts.get(stage_name, 0.5)

        # Determine if artifact has a real defect
        has_defect = quality < 0.6

        # Critic decision
        if has_defect:
            vetoed = self._rng.random() < detection_rate
        else:
            vetoed = self._rng.random() < false_positive_rate

        # In advisory mode, critics can't block
        if self.config.mode == RivalsMode.ADVISORY:
            vetoed = False

        # In council mode, we simulate a majority vote
        if self.config.mode == RivalsMode.COUNCIL:
            votes = [self._rng.random() < detection_rate for _ in range(3)]
            vetoed = sum(votes) >= 2 if has_defect else False

        # Record veto history
        episode.veto_history.append({
            "stage": stage_name,
            "critic": agent_id,
            "vetoed": vetoed,
            "quality": quality,
            "has_defect": has_defect,
        })

        if vetoed:
            # Retry or fail
            retries = episode.retries.get(stage_name, 0) + 1
            episode.retries[stage_name] = retries

            if retries >= self.config.max_retries_per_stage:
                episode.stage = PipelineStage.FAILED
                episode.final_score = 0.0
                self._completed_episodes.append(episode)
                del self._episodes[ep_id]
            else:
                # Re-queue producer for retry
                producer_role = _STAGE_TO_PRODUCER.get(episode.stage)
                if producer_role:
                    producer_id = episode.assignments.get(producer_role)
                    if producer_id:
                        self._pending_produce[producer_id] = ep_id

            return HandlerActionResult(
                success=True,
                observables=_OBS_VETO,
                initiator_id=agent_id,
                counterparty_id=agent_id,
                metadata={
                    "episode_id": ep_id,
                    "stage": stage_name,
                    "verdict": "veto",
                    "retries": retries,
                },
            )
        else:
            # Advance to next stage
            self._advance_stage(episode)

            return HandlerActionResult(
                success=True,
                observables=_OBS_PASS,
                initiator_id=agent_id,
                counterparty_id=agent_id,
                metadata={
                    "episode_id": ep_id,
                    "stage": stage_name,
                    "verdict": "pass",
                },
            )

    def _advance_stage(self, episode: RivalsEpisode) -> None:
        """Transition to next stage or score the episode."""
        current_idx = _STAGE_ORDER.index(episode.stage)

        if current_idx >= len(_STAGE_ORDER) - 1:
            # All stages complete, score the episode
            self._score_episode(episode)
        else:
            # Move to next stage
            next_stage = _STAGE_ORDER[current_idx + 1]
            episode.stage = next_stage

            # Queue next producer
            producer_role = _STAGE_TO_PRODUCER.get(next_stage)
            if producer_role:
                producer_id = episode.assignments.get(producer_role)
                if producer_id:
                    self._pending_produce[producer_id] = episode.episode_id

    def _score_episode(self, episode: RivalsEpisode) -> None:
        """Compare artifacts vs ground truth to compute final score."""
        # Average artifact quality as actual consistency
        qualities = list(episode.artifacts.values())
        if qualities:
            episode.actual_consistency = sum(qualities) / len(qualities)
        else:
            episode.actual_consistency = 0.0

        # Final score: how close to ground truth
        # High quality artifacts -> more likely to match ground truth
        episode.final_score = episode.actual_consistency

        episode.stage = PipelineStage.SCORED
        self._completed_episodes.append(episode)

        # Remove from active episodes
        if episode.episode_id in self._episodes:
            del self._episodes[episode.episode_id]

    def get_completed_episodes(self) -> List[RivalsEpisode]:
        """Return all completed episodes."""
        return list(self._completed_episodes)

    def get_active_episodes(self) -> Dict[str, RivalsEpisode]:
        """Return currently active episodes."""
        return dict(self._episodes)
