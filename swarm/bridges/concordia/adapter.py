"""Main adapter connecting Concordia narratives to SWARM's data model.

ConcordiaAdapter is the central bridge class that:
1. Takes narrative text from Concordia game steps
2. Evaluates it via LLMJudge to get scores
3. Converts scores to SWARM ProxyObservables
4. Computes v_hat and p via ProxyComputer
5. Records SoftInteraction objects
"""

import logging
from hashlib import sha256
from typing import List, Optional

from swarm.bridges.concordia.config import ConcordiaConfig
from swarm.bridges.concordia.events import (
    ConcordiaEvent,
    ConcordiaEventType,
    JudgeScores,
    NarrativeChunk,
)
from swarm.bridges.concordia.judge import LLMJudge
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.logging.event_log import EventLog
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class ConcordiaAdapter:
    """Bridge between Concordia narrative interactions and SWARM."""

    def __init__(
        self,
        config: Optional[ConcordiaConfig] = None,
        proxy_computer: Optional[ProxyComputer] = None,
        event_log: Optional[EventLog] = None,
        judge: Optional[LLMJudge] = None,
    ):
        self._config = config or ConcordiaConfig()
        self._proxy = proxy_computer or ProxyComputer(
            sigmoid_k=self._config.proxy_sigmoid_k
        )
        self._event_log = event_log
        self._judge = judge or LLMJudge(self._config.judge_config)
        self._interactions: List[SoftInteraction] = []
        self._events: List[ConcordiaEvent] = []
        self._metrics = SoftMetrics()

    def process_narrative(
        self,
        agent_ids: list[str],
        narrative_text: str,
        step: int = 0,
    ) -> list[SoftInteraction]:
        """Process a narrative chunk and produce SoftInteractions.

        Args:
            agent_ids: IDs of agents involved in the narrative
            narrative_text: The narrative text to evaluate
            step: Current simulation step

        Returns:
            List of SoftInteraction objects. For single agent, returns one
            self-interaction. For multiple agents, returns one per unique pair.
        """
        # Evaluate narrative via judge
        scores = self._judge.evaluate(narrative_text)

        self._record_event(
            ConcordiaEvent(
                event_type=ConcordiaEventType.JUDGE_EVALUATED,
                payload={
                    "agent_ids": agent_ids,
                    "progress": scores.progress,
                    "quality": scores.quality,
                    "cooperation": scores.cooperation,
                    "harm": scores.harm,
                    "cached": scores.cached,
                    "step": step,
                },
            )
        )

        # Convert scores to observables
        observables = self._scores_to_observables(scores)

        # Compute proxy labels
        v_hat, p = self._proxy.compute_labels(observables)

        # Create interactions for each agent pair
        interactions: list[SoftInteraction] = []

        if len(agent_ids) < 2:
            # Single agent: create self-interaction
            agent_id = agent_ids[0] if agent_ids else "unknown"
            interaction = SoftInteraction(
                initiator=agent_id,
                counterparty=agent_id,
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
                    "bridge": "concordia",
                    "step": step,
                    "narrative_preview": narrative_text[:200],
                },
            )
            interactions.append(interaction)
        else:
            # Create an interaction for each ordered pair
            for i, initiator in enumerate(agent_ids):
                for j, counterparty in enumerate(agent_ids):
                    if i >= j:
                        continue
                    interaction = SoftInteraction(
                        initiator=initiator,
                        counterparty=counterparty,
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
                            "bridge": "concordia",
                            "step": step,
                            "narrative_preview": narrative_text[:200],
                        },
                    )
                    interactions.append(interaction)

        for interaction in interactions:
            self._record_interaction(interaction)
            self._log_interaction(interaction)

        return interactions

    def process_narrative_batch(
        self, chunks: list[NarrativeChunk]
    ) -> list[SoftInteraction]:
        """Process a batch of narrative chunks."""
        all_interactions: list[SoftInteraction] = []
        for chunk in chunks:
            interactions = self.process_narrative(
                agent_ids=chunk.agent_ids,
                narrative_text=chunk.narrative_text,
                step=chunk.step_range[0],
            )
            all_interactions.extend(interactions)
        return all_interactions

    def _scores_to_observables(self, scores: JudgeScores) -> ProxyObservables:
        """Convert judge scores to SWARM proxy observables.

        Mapping:
            task_progress_delta = progress * 2.0 - 1.0  (maps [0,1] → [-1,1])
            rework_count = int((1.0 - quality) * 3)
            verifier_rejections = 0
            tool_misuse_flags = int(harm * 3)
            counterparty_engagement_delta = cooperation * 2.0 - 1.0
        """
        return ProxyObservables(
            task_progress_delta=scores.progress * 2.0 - 1.0,
            rework_count=int((1.0 - scores.quality) * 3),
            verifier_rejections=0,
            tool_misuse_flags=int(scores.harm * 3),
            counterparty_engagement_delta=scores.cooperation * 2.0 - 1.0,
        )

    def get_interactions(self) -> list[SoftInteraction]:
        """Return a copy of all recorded interactions."""
        return list(self._interactions)

    def get_events(self) -> list[ConcordiaEvent]:
        """Return a copy of all recorded events."""
        return list(self._events)

    def get_metrics(self) -> dict[str, float]:
        """Compute metrics from stored interactions."""
        interactions = self._interactions
        if not interactions:
            return {
                "toxicity_rate": 0.0,
                "quality_gap": 0.0,
                "total_welfare": 0.0,
            }

        welfare = self._metrics.welfare_metrics(interactions).get("total_welfare")
        return {
            "toxicity_rate": self._metrics.toxicity_rate(interactions),
            "quality_gap": self._metrics.quality_gap(interactions),
            "total_welfare": (
                welfare if isinstance(welfare, (int, float)) else 0.0
            ),
        }

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        """Record an interaction, enforcing the configured cap."""
        if len(self._interactions) >= self._config.max_interactions:
            self._interactions = self._interactions[
                -self._config.max_interactions // 2 :
            ]
        self._interactions.append(interaction)

    def _record_event(self, event: ConcordiaEvent) -> None:
        """Record a bridge event, enforcing the configured cap."""
        if len(self._events) >= self._config.max_events:
            self._events = self._events[-self._config.max_events // 2 :]
        self._events.append(event)

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        """Log an interaction to SWARM's event log."""
        if self._event_log is None:
            return

        metadata = dict(interaction.metadata or {})
        narrative_preview = metadata.pop("narrative_preview", None)
        if isinstance(narrative_preview, str) and narrative_preview:
            metadata["narrative_preview_sha256"] = sha256(
                narrative_preview.encode("utf-8")
            ).hexdigest()

        event = Event(
            event_type=EventType.INTERACTION_COMPLETED,
            interaction_id=interaction.interaction_id,
            initiator_id=interaction.initiator,
            counterparty_id=interaction.counterparty,
            payload={
                "accepted": interaction.accepted,
                "v_hat": interaction.v_hat,
                "p": interaction.p,
                "bridge": "concordia",
                "metadata": metadata,
            },
        )
        self._event_log.append(event)
