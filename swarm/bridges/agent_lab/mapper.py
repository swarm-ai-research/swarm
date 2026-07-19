"""AgentLab event -> SoftInteraction mapper.

Maps AgentLaboratory signals to SWARM ProxyObservables:

| AgentLab Signal                         | ProxyObservable                  | Mapping              |
|-----------------------------------------|----------------------------------|----------------------|
| MLESolver/PaperSolver score (0-1)       | task_progress_delta              | score * 2 - 1        |
| Repair attempts + excess iterations     | rework_count                     | Direct count          |
| Code exec failures + review rejections  | verifier_rejections              | Count of failures     |
| Dialogue frequency + submission depth   | counterparty_engagement_delta    | Scaled to [-1, +1]    |
| Code success/failure + repair status    | tool_misuse_flags                | 1 if failed and no repair, else 0 (map_code_event only; 0 in other event types) |
"""

from swarm.bridges.agent_lab.config import AgentLabConfig
from swarm.bridges.agent_lab.events import (
    AgentLabEvent,
    DialogueEvent,
    ReviewEvent,
    SolverIterationEvent,
)
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction


class AgentLabMapper:
    """Translate AgentLab events into SWARM SoftInteractions."""

    def __init__(
        self,
        config: AgentLabConfig | None = None,
        proxy: ProxyComputer | None = None,
    ) -> None:
        self._config = config or AgentLabConfig()
        self._proxy = proxy or ProxyComputer(
            sigmoid_k=self._config.proxy_sigmoid_k
        )

    def _resolve_agent_id(self, role: str) -> str:
        """Map an AgentLab role name to a SWARM agent ID."""
        return self._config.agent_role_map.get(role, role)

    def _resolve_reviewer_id(self, index: int) -> str:
        """Map a reviewer index (0-2) to a SWARM agent ID."""
        return self._config.reviewer_map.get(
            index, f"agent_lab_reviewer_{index + 1}"
        )

    def map_solver_iteration(
        self,
        event: AgentLabEvent,
        solver: SolverIterationEvent,
    ) -> SoftInteraction:
        """Map a solver iteration to a COLLABORATION interaction.

        Score (0-1) maps to task_progress_delta via ``score * 2 - 1``.
        Repair attempts map to rework_count.
        Execution errors map to verifier_rejections.
        """
        task_progress = solver.score * 2.0 - 1.0  # [0,1] -> [-1,+1]
        rework = solver.repair_attempts
        rejections = 1 if solver.execution_error else 0
        engagement = min(1.0, max(-1.0, solver.score - 0.5))  # centered

        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=0,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        agent_id = self._resolve_agent_id(event.agent_role)
        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=self._config.orchestrator_id,
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=solver.execution_error is None,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "agent_lab",
                "event_type": event.event_type.value,
                "event_id": event.event_id,
                "phase": event.phase,
                "solver_type": solver.solver_type,
                "iteration_index": solver.iteration_index,
                "score": solver.score,
                "cost_usd": solver.cost_usd,
            },
        )

    def map_dialogue(
        self,
        event: AgentLabEvent,
        dialogue: DialogueEvent,
    ) -> SoftInteraction:
        """Map a dialogue exchange to a COLLABORATION interaction.

        Engagement scales with submission depth (has_submission = +0.3).
        """
        engagement = 0.3 if dialogue.has_submission else 0.1
        task_progress = 0.2 if dialogue.has_submission else 0.05

        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        speaker_id = self._resolve_agent_id(dialogue.speaker_role)
        listener_id = self._resolve_agent_id(dialogue.listener_role)
        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=speaker_id,
            counterparty=listener_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "agent_lab",
                "event_type": event.event_type.value,
                "event_id": event.event_id,
                "phase": dialogue.phase,
                "command_type": dialogue.command_type,
                "has_submission": dialogue.has_submission,
            },
        )

    def map_review(
        self,
        event: AgentLabEvent,
        review: ReviewEvent,
    ) -> SoftInteraction:
        """Map a review submission to a VOTE interaction.

        Overall score (1-10) maps to task_progress_delta via ``score / 5 - 1``.
        Low scores contribute verifier_rejections.
        """
        # Normalize 1-10 scale to [-1, +1]
        task_progress = max(-1.0, min(1.0, review.overall_score / 5.0 - 1.0))
        rejections = 1 if review.decision in ("reject", "weak_reject") else 0
        engagement = min(1.0, review.confidence / 5.0)

        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=0,
            verifier_rejections=rejections,
            tool_misuse_flags=0,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        reviewer_id = self._resolve_reviewer_id(review.reviewer_index)
        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=reviewer_id,
            counterparty=self._config.orchestrator_id,
            interaction_type=InteractionType.VOTE,
            accepted=review.decision not in ("reject", "weak_reject"),
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "agent_lab",
                "event_type": event.event_type.value,
                "event_id": event.event_id,
                "reviewer_index": review.reviewer_index,
                "overall_score": review.overall_score,
                "decision": review.decision,
                "confidence": review.confidence,
            },
        )

    def map_phase_completion(
        self,
        event: AgentLabEvent,
        success: bool = True,
        phase_score: float = 0.5,
    ) -> SoftInteraction:
        """Map a phase completion to a COLLABORATION interaction.

        phase_score (0-1) maps to task_progress_delta.
        """
        task_progress = phase_score * 2.0 - 1.0
        engagement = 0.5 if success else -0.3

        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=0 if success else 1,
            verifier_rejections=0 if success else 1,
            tool_misuse_flags=0,
            counterparty_engagement_delta=engagement,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        agent_id = self._resolve_agent_id(event.agent_role)
        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=self._config.orchestrator_id,
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=success,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "agent_lab",
                "event_type": event.event_type.value,
                "event_id": event.event_id,
                "phase": event.phase,
                "success": success,
                "phase_score": phase_score,
            },
        )

    def map_code_event(
        self,
        event: AgentLabEvent,
        success: bool,
        repair_attempt: bool = False,
    ) -> SoftInteraction:
        """Map a code generation/execution/failure event."""
        if success:
            task_progress = 0.4
            rejections = 0
            misuse = 0
        else:
            task_progress = -0.4
            rejections = 1
            misuse = 1 if not repair_attempt else 0

        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=1 if repair_attempt else 0,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=0.1 if success else -0.2,
        )
        v_hat, p = self._proxy.compute_labels(observables)

        agent_id = self._resolve_agent_id(event.agent_role)
        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=self._config.orchestrator_id,
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=success,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "agent_lab",
                "event_type": event.event_type.value,
                "event_id": event.event_id,
                "phase": event.phase,
                "success": success,
                "repair_attempt": repair_attempt,
            },
        )
