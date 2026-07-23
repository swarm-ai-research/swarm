"""Maps AI-Scientist events to ProxyObservables and SoftInteraction objects."""

from __future__ import annotations

from swarm.bridges.ai_scientist.config import AIScientistConfig
from swarm.bridges.ai_scientist.events import (
    AIScientistEvent,
    AIScientistEventType,
    ExperimentRunEvent,
    IdeaEvent,
    ReviewEvent,
    WriteupEvent,
)
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction


class AIScientistMapper:
    """Converts AI-Scientist events into SoftInteraction objects."""

    def __init__(self, config: AIScientistConfig | None = None) -> None:
        self._config = config or AIScientistConfig()
        self._proxy = ProxyComputer(sigmoid_k=self._config.proxy_sigmoid_k)

    def _resolve_agent_id(self, role: str) -> str:
        return self._config.agent_role_map.get(role, role)

    def map_idea(
        self,
        event: AIScientistEvent,
        idea: IdeaEvent,
    ) -> SoftInteraction:
        """Map idea generation event to SoftInteraction."""
        mean_score = (idea.interestingness + idea.feasibility + idea.novelty_score) / 3.0
        task_progress = mean_score / 5.0 - 1.0  # scale 1-10 -> [-0.8, +1.0] (clamped to [-1, +1])

        observables = ProxyObservables(
            task_progress_delta=max(-1.0, min(1.0, task_progress)),
            counterparty_engagement_delta=0.0,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=self._config.orchestrator_id,
            counterparty=self._resolve_agent_id("ideation"),
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            task_progress_delta=observables.task_progress_delta,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "ai_scientist",
                "event_type": event.event_type.value,
                "idea_name": idea.idea_name,
                "phase": "idea",
            },
        )

    def map_novelty_check(
        self,
        event: AIScientistEvent,
        idea: IdeaEvent,
    ) -> SoftInteraction:
        """Map novelty check result to SoftInteraction (VOTE type)."""
        passed = event.event_type == AIScientistEventType.NOVELTY_CHECK_PASSED

        observables = ProxyObservables(
            task_progress_delta=0.4 if passed else -0.4,
            verifier_rejections=0 if passed else 1,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=self._resolve_agent_id("literature_checker"),
            counterparty=self._config.orchestrator_id,
            interaction_type=InteractionType.VOTE,
            accepted=passed,
            task_progress_delta=observables.task_progress_delta,
            rework_count=0,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.0,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "ai_scientist",
                "event_type": event.event_type.value,
                "idea_name": idea.idea_name,
                "novel": idea.novel,
                "phase": "idea",
            },
        )

    def map_experiment_run(
        self,
        event: AIScientistEvent,
        run: ExperimentRunEvent,
    ) -> SoftInteraction:
        """Map experiment run to SoftInteraction."""
        task_progress = 0.4 if run.success else -0.4

        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=run.retry_count,
            verifier_rejections=0 if run.success else 1,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=self._config.orchestrator_id,
            counterparty=self._resolve_agent_id("experiment"),
            interaction_type=InteractionType.COLLABORATION,
            accepted=run.success,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.0,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "ai_scientist",
                "event_type": event.event_type.value,
                "idea_name": event.idea_name,
                "run_index": run.run_index,
                "phase": "experiment",
            },
        )

    def map_writeup(
        self,
        event: AIScientistEvent,
        writeup: WriteupEvent,
    ) -> SoftInteraction:
        """Map writeup event to SoftInteraction."""
        if event.event_type == AIScientistEventType.WRITEUP_COMPILED:
            task_progress = 0.6
            verifier_rejections = 0
            accepted = True
        elif event.event_type == AIScientistEventType.WRITEUP_FAILED:
            task_progress = -0.4
            verifier_rejections = 1
            accepted = False
        elif event.event_type == AIScientistEventType.WRITEUP_SECTION:
            task_progress = 0.2
            verifier_rejections = 0
            accepted = True
        else:
            task_progress = 0.0
            verifier_rejections = 0
            accepted = True

        engagement = min(1.0, writeup.citation_count / 20.0)

        observables = ProxyObservables(
            task_progress_delta=task_progress,
            verifier_rejections=verifier_rejections,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=self._config.orchestrator_id,
            counterparty=self._resolve_agent_id("writeup"),
            interaction_type=InteractionType.COLLABORATION,
            accepted=accepted,
            task_progress_delta=observables.task_progress_delta,
            rework_count=0,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=0,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "ai_scientist",
                "event_type": event.event_type.value,
                "idea_name": event.idea_name,
                "section": writeup.section,
                "phase": "writeup",
            },
        )

    def map_review(
        self,
        event: AIScientistEvent,
        review: ReviewEvent,
    ) -> SoftInteraction:
        """Map review event to SoftInteraction (VOTE type)."""
        task_progress = review.overall_score / 5.0 - 1.0  # 1-10 -> [-0.8, +1.0] (clamped to [-1, +1])
        is_accept = review.decision.lower().startswith("accept")
        engagement = review.confidence / 5.0  # 1-5 -> [0, 1]

        observables = ProxyObservables(
            task_progress_delta=max(-1.0, min(1.0, task_progress)),
            verifier_rejections=0 if is_accept else 1,
            counterparty_engagement_delta=max(-1.0, min(1.0, engagement)),
        )
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=self._resolve_agent_id("reviewer"),
            counterparty=self._config.orchestrator_id,
            interaction_type=InteractionType.VOTE,
            accepted=is_accept,
            task_progress_delta=observables.task_progress_delta,
            rework_count=0,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=0,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "ai_scientist",
                "event_type": event.event_type.value,
                "idea_name": event.idea_name,
                "overall_score": review.overall_score,
                "decision": review.decision,
                "phase": "review",
            },
        )

    def map_event(self, event: AIScientistEvent) -> SoftInteraction | None:
        """Route an event to the appropriate mapper method."""
        if event.event_type == AIScientistEventType.IDEA_GENERATED:
            idea = IdeaEvent.from_dict(event.payload)
            return self.map_idea(event, idea)

        if event.event_type in (
            AIScientistEventType.NOVELTY_CHECK_PASSED,
            AIScientistEventType.NOVELTY_CHECK_FAILED,
        ):
            idea = IdeaEvent.from_dict(event.payload)
            return self.map_novelty_check(event, idea)

        if event.event_type in (
            AIScientistEventType.EXPERIMENT_RUN_COMPLETED,
            AIScientistEventType.EXPERIMENT_RUN_FAILED,
        ):
            run = ExperimentRunEvent.from_dict(event.payload)
            return self.map_experiment_run(event, run)

        if event.event_type in (
            AIScientistEventType.WRITEUP_SECTION,
            AIScientistEventType.WRITEUP_COMPILED,
            AIScientistEventType.WRITEUP_FAILED,
        ):
            writeup = WriteupEvent.from_dict(event.payload)
            return self.map_writeup(event, writeup)

        if event.event_type == AIScientistEventType.REVIEW_SUBMITTED:
            review = ReviewEvent.from_dict(event.payload)
            return self.map_review(event, review)

        # Events like EXPERIMENT_STARTED, CITATION_ADDED, COST_UPDATED,
        # IMPROVEMENT_APPLIED, etc. don't produce interactions
        return None
