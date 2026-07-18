"""Mesa ABM model bridge for SWARM governance testing.

Wraps Mesa ``Model.step()`` to produce ``SoftInteraction`` objects from
agent state after each step.  Enables governance mechanism testing on
complex emergent ABM dynamics without modifying existing Mesa models.

Usage::

    import mesa
    from swarm.bridges.mesa import MesaBridge, MesaBridgeConfig

    class MyModel(mesa.Model):
        def __init__(self, n_agents):
            super().__init__()
            self.schedule = mesa.time.RandomActivation(self)
            for i in range(n_agents):
                agent = mesa.Agent(i, self)
                agent.task_progress = 1.0   # optional SWARM attrs
                self.schedule.add(agent)

        def step(self):
            self.schedule.step()

    model = MyModel(n_agents=20)
    bridge = MesaBridge(model=model)

    for step_num in range(100):
        interactions = bridge.step()
        mean_p = sum(ix.p for ix in interactions) / max(len(interactions), 1)
        print(f"step={step_num}  mean_p={mean_p:.3f}")

Protocol mode (no Mesa required)::

    from swarm.bridges.mesa import MesaBridge

    bridge = MesaBridge()
    interactions = bridge.record_agent_states([
        {"agent_id": "a1", "task_progress": 0.9, "rework_count": 0.1},
        {"agent_id": "a2", "task_progress": 0.3, "rework_count": 0.7},
    ])
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from swarm.bridges._common import log_interaction_event
from swarm.bridges.mesa.config import MesaBridgeConfig
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class MesaBridgeError(Exception):
    """Raised when a Mesa bridge operation fails."""


class MesaBridge:
    """Bridge a Mesa ABM model to SWARM governance scoring.

    Each call to ``step()`` advances the Mesa model by one step and
    records a ``SoftInteraction`` for each active agent (up to
    ``config.max_agents_per_step``).

    Protocol mode: if no model is provided, use ``record_agent_states()``
    to feed agent state dicts directly.

    Args:
        model: Optional Mesa ``Model`` instance.
        config: Bridge configuration.
        payoff_config: Optional custom payoff parameters.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[MesaBridgeConfig] = None,
        payoff_config: Optional[PayoffConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or MesaBridgeConfig()
        self._proxy = ProxyComputer(sigmoid_k=self.config.proxy_sigmoid_k)
        self._payoff_engine = SoftPayoffEngine(payoff_config or PayoffConfig())
        self._metrics = SoftMetrics(self._payoff_engine)
        self._interactions: List[SoftInteraction] = []
        self._step_count: int = 0
        self._event_log: Optional[Any] = None

        if self.config.enable_event_log:
            try:
                from pathlib import Path as _Path

                from swarm.logging.event_log import EventLog

                path = self.config.event_log_path or f"{self.config.model_id}_events.jsonl"
                self._event_log = EventLog(path=_Path(path))
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not initialise EventLog: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> List[SoftInteraction]:
        """Advance the Mesa model one step and record interactions.

        Returns:
            List of ``SoftInteraction`` objects recorded in this step.

        Raises:
            MesaBridgeError: if no model is set.
        """
        if self.model is None:
            raise MesaBridgeError(
                "No model provided.  Pass a Mesa Model to __init__ or use "
                "record_agent_states() directly."
            )

        self.model.step()
        self._step_count += 1

        agents = self._get_agents()
        return self.record_agent_states(agents)

    def run(self, n_steps: int) -> List[SoftInteraction]:
        """Run the model for ``n_steps`` and return all interactions.

        Args:
            n_steps: Number of Mesa steps to advance.

        Returns:
            Flat list of all ``SoftInteraction`` objects recorded.
        """
        all_interactions: List[SoftInteraction] = []
        for _ in range(n_steps):
            all_interactions.extend(self.step())
        return all_interactions

    def record_agent_states(self, agents: List[Any]) -> List[SoftInteraction]:
        """Record interactions from a list of agent state dicts or Mesa agents.

        Each element can be:
        - A Mesa ``Agent`` instance (reads attributes via config attr names).
        - A plain ``dict`` with keys matching the attribute names.

        Args:
            agents: List of agent objects or state dicts.

        Returns:
            List of ``SoftInteraction`` objects.
        """
        interactions: List[SoftInteraction] = []
        sampled = agents[: self.config.max_agents_per_step]

        for agent in sampled:
            interaction = self._agent_to_interaction(agent)
            interactions.append(interaction)

        return interactions

    def get_toxicity_rate(self) -> float:
        """Toxicity rate over all recorded interactions."""
        return float(self._metrics.toxicity_rate(self._interactions))

    def get_quality_gap(self) -> float:
        """Quality gap over all recorded interactions."""
        return float(self._metrics.quality_gap(self._interactions))

    def get_interactions(self) -> List[SoftInteraction]:
        """Return all recorded interactions (read-only copy)."""
        return list(self._interactions)

    @property
    def step_count(self) -> int:
        """Number of model steps advanced via this bridge."""
        return self._step_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_agents(self) -> List[Any]:
        """Extract the agent list from a Mesa model."""
        # Mesa >= 2.0: model.agents (AgentSet) or model.schedule.agents
        agents_attr = getattr(self.model, "agents", None)
        if agents_attr is not None:
            try:
                return list(agents_attr)[: self.config.max_agents_per_step]
            except TypeError:
                pass

        schedule = getattr(self.model, "schedule", None)
        if schedule is not None:
            schedule_agents = getattr(schedule, "agents", [])
            return list(schedule_agents)[: self.config.max_agents_per_step]

        logger.warning(
            "Could not extract agents from Mesa model %r.  "
            "Override _get_agents() or use record_agent_states().",
            type(self.model).__name__,
        )
        return []

    def _agent_to_interaction(self, agent: Any) -> SoftInteraction:
        """Map a single Mesa agent to a SoftInteraction."""
        observables = self._extract_observables(agent)
        v_hat, p = self._proxy.compute_labels(observables)

        # Safety invariant
        assert 0.0 <= p <= 1.0, f"p invariant violated: p={p}"

        if isinstance(agent, dict):
            agent_id = str(agent.get("agent_id", f"agent-{id(agent)}"))
        else:
            agent_id = str(getattr(agent, "unique_id", f"agent-{id(agent)}"))

        interaction = SoftInteraction(
            initiator=self.config.model_id,
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            p=p,
            v_hat=v_hat,
            accepted=p >= 0.5,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            metadata={"step": self._step_count},
        )

        self._payoff_engine.payoff_initiator(interaction)
        self._interactions.append(interaction)
        self._log_interaction(interaction)

        return interaction

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        """Append an interaction to the EventLog as an Event."""
        log_interaction_event(self._event_log, interaction)

    def _extract_observables(self, agent: Any) -> ProxyObservables:
        """Extract SWARM proxy observables from a Mesa agent or state dict.

        Attribute mapping (configurable via MesaBridgeConfig):
            task_progress_delta           agent.task_progress -> mapped to [-1,+1]
            rework_count                  agent.rework_count  (int, default 0)
            verifier_rejections           0 (no verifier in Mesa by default)
            counterparty_engagement_delta agent.engagement -> mapped to [-1,+1]

        Falls back to neutral defaults when attributes are absent.

        Returns:
            ``ProxyObservables`` with values in valid ranges.
        """
        def _get_float(attr: str, default: float) -> float:
            if isinstance(agent, dict):
                val = agent.get(attr, default)
            else:
                val = getattr(agent, attr, default)
            try:
                return float(max(0.0, min(1.0, val)))
            except (TypeError, ValueError):
                return default

        def _get_int(attr: str, default: int) -> int:
            if isinstance(agent, dict):
                val = agent.get(attr, default)
            else:
                val = getattr(agent, attr, default)
            try:
                return max(0, int(val))
            except (TypeError, ValueError):
                return default

        # task_progress in [0,1] -> delta in [-1,+1]
        task_progress = _get_float(self.config.agent_task_progress_attr, 1.0)
        task_progress_delta = task_progress * 2.0 - 1.0

        rework_count = _get_int(self.config.agent_rework_count_attr, 0)

        # engagement in [0,1] -> delta in [-1,+1]
        engagement = _get_float(self.config.agent_engagement_attr, 0.5)
        engagement_delta = engagement * 2.0 - 1.0

        return ProxyObservables(
            task_progress_delta=task_progress_delta,
            rework_count=rework_count,
            verifier_rejections=0,
            counterparty_engagement_delta=engagement_delta,
        )
