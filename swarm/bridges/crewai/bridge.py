"""CrewAI crew bridge for SWARM governance testing.

Maps CrewAI crew task results to ``SoftInteraction`` objects for use
with SWARM's governance and metrics framework.

This bridge is distinct from ``swarm.agents.crewai_adapter``:

- **Agent adapter** (``swarm.agents.crewai_adapter``): Wraps a crew as a
  single SWARM agent that produces one action per simulation step.
- **Bridge** (this module): Wraps crew execution as a SWARM interaction
  source that generates one interaction per task result.

Usage (with crewai installed)::

    from swarm.bridges.crewai import CrewAIBridge

    bridge = CrewAIBridge(crew=my_crew)
    interactions = bridge.run()

    for ix in interactions:
        print(f"agent={ix.counterparty}  p={ix.p:.3f}")

Usage (protocol mode — no crewai required)::

    from swarm.bridges.crewai import CrewAIBridge, TaskResult

    bridge = CrewAIBridge()
    result = TaskResult(
        task_description="Summarise papers",
        agent_role="Researcher",
        output="Found 10 relevant papers.",
        success=True,
    )
    interaction = bridge.record_task_result(result)
    print(f"p={interaction.p:.3f}")
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.bridges._common import log_interaction_event
from swarm.bridges.crewai.config import CrewAIBridgeConfig
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class CrewAIBridgeError(Exception):
    """Raised when a CrewAI bridge operation fails."""


@dataclass
class TaskResult:
    """Result of a single CrewAI task execution.

    Attributes:
        task_description: Human-readable description of the task.
        agent_role: Role of the agent that executed the task.
        output: Task output string.
        success: Whether the task completed without error.
        delegation_depth: Number of sub-delegations made (0 = direct).
        quality_score: Optional external quality score in [0, 1].
        metadata: Additional context.
    """

    task_description: str = ""
    agent_role: str = "agent"
    output: str = ""
    success: bool = True
    delegation_depth: int = 0
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrewAIBridge:
    """Bridge CrewAI crew execution to SWARM governance scoring.

    Two usage modes:

    1. **Full crew mode**: Pass a ``crewai.Crew`` object and call ``run()``.
    2. **Protocol mode**: No crew required; call ``record_task_result()``
       directly with completed ``TaskResult`` objects.

    Args:
        crew: Optional CrewAI Crew object.
        config: Bridge configuration.
        payoff_config: Optional custom payoff parameters.
    """

    def __init__(
        self,
        crew: Optional[Any] = None,
        config: Optional[CrewAIBridgeConfig] = None,
        payoff_config: Optional[PayoffConfig] = None,
    ) -> None:
        self.crew = crew
        self.config = config or CrewAIBridgeConfig()
        self._proxy = ProxyComputer(sigmoid_k=self.config.proxy_sigmoid_k)
        self._payoff_engine = SoftPayoffEngine(payoff_config or PayoffConfig())
        self._metrics = SoftMetrics(self._payoff_engine)
        self._interactions: List[SoftInteraction] = []
        self.last_crew_output: Optional[Any] = None
        self._event_log: Optional[Any] = None

        if self.config.enable_event_log:
            try:
                from pathlib import Path as _Path

                from swarm.logging.event_log import EventLog

                path = self.config.event_log_path or f"{self.config.crew_id}_events.jsonl"
                self._event_log = EventLog(path=_Path(path))
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not initialise EventLog: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> List[SoftInteraction]:
        """Execute the crew and return interactions for each task.

        Requires a ``crewai.Crew`` to be provided at construction time.

        Args:
            inputs: Optional inputs dict passed to ``crew.kickoff()``.

        Returns:
            List of ``SoftInteraction`` objects, one per task.

        Raises:
            CrewAIBridgeError: if no crew is set or execution fails.
        """
        if self.crew is None:
            raise CrewAIBridgeError(
                "No crew provided.  Pass a crewai.Crew to __init__ or use "
                "record_task_result() directly."
            )

        try:
            crew_output = self._run_crew_with_timeout(inputs or {})
        except CrewAIBridgeError:
            raise
        except Exception as exc:
            raise CrewAIBridgeError(f"Crew execution failed: {exc}") from exc

        self.last_crew_output = crew_output
        return self._extract_interactions_from_output(crew_output)

    def record_task_result(self, result: TaskResult) -> SoftInteraction:
        """Record a single task result and return a scored SoftInteraction.

        Args:
            result: Completed task result.

        Returns:
            ``SoftInteraction`` with ``p`` in [0, 1].
        """
        return self._task_to_interaction(result)

    def get_toxicity_rate(self) -> float:
        """Toxicity rate over all recorded interactions."""
        return float(self._metrics.toxicity_rate(self._interactions))

    def get_quality_gap(self) -> float:
        """Quality gap over all recorded interactions."""
        return float(self._metrics.quality_gap(self._interactions))

    def get_interactions(self) -> List[SoftInteraction]:
        """Return all recorded interactions (read-only copy)."""
        return list(self._interactions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_crew_with_timeout(self, inputs: Dict[str, Any]) -> Any:
        assert self.crew is not None  # guarded by caller (run())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.crew.kickoff, inputs)
            try:
                return future.result(timeout=self.config.timeout_seconds)
            except concurrent.futures.TimeoutError as exc:
                raise CrewAIBridgeError(
                    f"Crew timed out after {self.config.timeout_seconds}s"
                ) from exc

    def _extract_interactions_from_output(self, crew_output: Any) -> List[SoftInteraction]:
        """Extract interactions from a crew output object.

        Version-dependent behavior:
        - CrewAI >= 0.28: Extracts one interaction per task (per-task granularity).
        - CrewAI < 0.28: Falls back to raw output and creates a single aggregated
          interaction (loses per-task granularity).
        """
        interactions: List[SoftInteraction] = []

        # CrewAI >= 0.28 exposes .tasks_output (list of TaskOutput)
        task_outputs = getattr(crew_output, "tasks_output", None)
        if task_outputs:
            for task_out in task_outputs:
                raw_output = (
                    getattr(task_out, "raw_output", "")
                    or getattr(task_out, "raw", "")
                    or str(task_out)
                )
                agent_role = str(getattr(task_out, "agent", "agent"))
                task_desc = str(getattr(task_out, "description", ""))
                result = TaskResult(
                    task_description=task_desc[:500],
                    agent_role=agent_role[:200],
                    output=str(raw_output)[:3000],
                    success=True,
                )
                interactions.append(self._task_to_interaction(result))
        else:
            output_str = getattr(crew_output, "raw", "") or str(crew_output)
            result = TaskResult(
                task_description="crew run",
                agent_role="crew",
                output=str(output_str)[:3000],
                success=True,
            )
            interactions.append(self._task_to_interaction(result))

        return interactions

    def _task_to_interaction(self, result: TaskResult) -> SoftInteraction:
        """Map a TaskResult to a SoftInteraction."""
        observables = self._extract_observables(result)
        v_hat, p = self._proxy.compute_labels(observables)

        # Safety invariant (CLAUDE.md: p must remain in [0, 1])
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p invariant violated: p={p}")

        interaction = SoftInteraction(
            initiator=self.config.crew_id,
            counterparty=result.agent_role[:200],
            interaction_type=InteractionType.COLLABORATION,
            p=p,
            v_hat=v_hat,
            accepted=result.success,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            metadata={
                "task_description": result.task_description[:500],
                "delegation_depth": result.delegation_depth,
                **{k: str(v)[:200] for k, v in result.metadata.items()},
            },
        )

        self._payoff_engine.payoff_initiator(interaction)
        self._interactions.append(interaction)
        self._log_interaction(interaction)

        return interaction

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        """Append an interaction to the EventLog as an Event."""
        log_interaction_event(self._event_log, interaction)

    def _extract_observables(self, result: TaskResult) -> ProxyObservables:
        """Map a TaskResult to SWARM proxy observables.

        Observable mapping::

            task_progress_delta           +1.0 if success else -1.0
            rework_count                  delegation_depth (capped at max)
            verifier_rejections           0 if success; 1 if quality_score < 0.5
            counterparty_engagement_delta output length normalised to [-1, +1]

        Returns:
            ``ProxyObservables`` with values in valid ranges.
        """
        task_progress_delta = 1.0 if result.success else -1.0
        rework_count = min(result.delegation_depth, self.config.max_delegation_depth)

        max_chars = max(self.config.engagement_max_chars, 1)
        engagement_normalized = min(len(result.output) / max_chars, 1.0)
        engagement_delta = engagement_normalized * 2.0 - 1.0

        verifier_rejections = 0
        if not result.success:
            verifier_rejections = 1
        elif result.quality_score is not None and result.quality_score < 0.5:
            verifier_rejections = 1

        return ProxyObservables(
            task_progress_delta=task_progress_delta,
            rework_count=rework_count,
            verifier_rejections=verifier_rejections,
            counterparty_engagement_delta=engagement_delta,
        )
