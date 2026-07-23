"""Middleware protocol for simulation lifecycle hooks.

Middleware classes handle cross-cutting concerns (governance, perturbation,
contract screening, network decay, etc.) that previously lived inline in
the ``Orchestrator``.  Each middleware is registered with the
``MiddlewarePipeline`` and called in registration order at well-defined
lifecycle points.

The separation keeps the orchestrator as a thin loop and makes each
concern independently testable.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _try_call_method(obj: Any, method_name: str, *args: Any, **kwargs: Any) -> bool:
    """Try to call a method on an object, logging failures.

    Args:
        obj: Object to call method on.
        method_name: Name of method to call.
        *args: Positional arguments to pass to method.
        **kwargs: Keyword arguments to pass to method.

    Returns:
        True if call succeeded, False if exception was caught.
    """
    try:
        method = getattr(obj, method_name)
        method(*args, **kwargs)
        return True
    except Exception:
        logger.debug(
            "%s.%s failed",
            type(obj).__name__,
            method_name,
            exc_info=True,
        )
        return False


# -----------------------------------------------------------------------
# Protocol
# -----------------------------------------------------------------------


class Middleware(ABC):
    """Lifecycle hook interface for simulation cross-cutting concerns."""

    @abstractmethod
    def on_epoch_start(self, ctx: "MiddlewareContext") -> None:
        """Called at the start of each epoch."""

    @abstractmethod
    def on_epoch_end(self, ctx: "MiddlewareContext") -> None:
        """Called at the end of each epoch.

        Some implementations may be no-ops if epoch-end work is driven
        by alternative mechanisms (e.g., callbacks).
        """

    @abstractmethod
    def on_step_start(self, ctx: "MiddlewareContext") -> None:
        """Called at the start of each step."""


class MiddlewareContext:
    """Shared mutable bag passed to every middleware hook.

    Holds references to the simulation's live objects so middleware can
    read and mutate state without needing back-references to the
    orchestrator.
    """

    def __init__(
        self,
        state: Any,  # EnvState
        config: Any,  # OrchestratorConfig
        agents: Dict[str, Any],  # agent_id -> BaseAgent
        event_bus: EventBus,
        *,
        network: Any = None,
        governance_engine: Any = None,
        metrics_calculator: Any = None,
        payoff_engine: Any = None,
        handler_registry: Any = None,
        finalizer: Any = None,
    ) -> None:
        self.state = state
        self.config = config
        self.agents = agents
        self.event_bus = event_bus
        self.network = network
        self.governance_engine = governance_engine
        self.metrics_calculator = metrics_calculator
        self.payoff_engine = payoff_engine
        self.handler_registry = handler_registry
        self.finalizer = finalizer

        # Middleware can stash per-epoch/step data here
        self.scratch: Dict[str, Any] = {}


# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------


class MiddlewarePipeline:
    """Ordered collection of middleware.  Runs hooks in registration order."""

    def __init__(self) -> None:
        self._middleware: List[Middleware] = []

    def add(self, mw: Middleware) -> None:
        self._middleware.append(mw)

    def all(self) -> Sequence[Middleware]:
        return self._middleware

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        for mw in self._middleware:
            _try_call_method(mw, "on_epoch_start", ctx)

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        for mw in self._middleware:
            _try_call_method(mw, "on_epoch_end", ctx)

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        for mw in self._middleware:
            _try_call_method(mw, "on_step_start", ctx)


# -----------------------------------------------------------------------
# Concrete middleware
# -----------------------------------------------------------------------


class GovernanceMiddleware(Middleware):
    """Adaptive governance updates and epoch/step governance effects."""

    def __init__(
        self,
        governance_engine: Any,
        adaptive_controller: Any = None,
    ) -> None:
        self._gov = governance_engine
        self._adaptive = adaptive_controller

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        self._update_adaptive(ctx, include_behavioral=False)
        effect = self._gov.apply_epoch_start(ctx.state, ctx.state.current_epoch)
        if ctx.finalizer is not None:
            ctx.finalizer.apply_governance_effect(effect)

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        pass  # governance epoch-end work is driven by callbacks

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        if self._gov.config.adaptive_use_behavioral_features:
            self._update_adaptive(ctx, include_behavioral=True)
        step_effect = self._gov.apply_step(ctx.state, ctx.state.current_step)
        if ctx.finalizer is not None:
            ctx.finalizer.apply_governance_effect(step_effect)

    def _update_adaptive(
        self, ctx: MiddlewareContext, *, include_behavioral: bool
    ) -> None:
        if not self._gov.config.adaptive_governance_enabled:
            return
        from swarm.agents.base import ActionType
        from swarm.forecaster.features import (
            combine_feature_dicts,
            extract_behavioral_features,
            extract_structural_features,
        )
        from swarm.models.agent import AgentType

        agents = list(ctx.agents.values())
        adversarial_count = sum(
            1
            for a in agents
            if a.agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE)
        )
        structural = extract_structural_features(
            horizon_length=ctx.config.steps_per_epoch,
            agent_count=len(agents),
            action_space_size=len(ActionType),
            adversarial_fraction=(
                adversarial_count / len(agents) if agents else 0.0
            ),
        )
        features = structural
        if include_behavioral:
            behavioral = extract_behavioral_features(
                ctx.state.completed_interactions
            )
            features = combine_feature_dicts(structural, behavioral)
        self._gov.update_adaptive_mode(features)


class PerturbationMiddleware(Middleware):
    """Perturbation engine epoch/step triggers and condition checks."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    @property
    def engine(self) -> Any:
        return self._engine

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        self._engine.on_epoch_start(ctx.state.current_epoch)

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        interactions = ctx.state.completed_interactions
        metrics_dict: Dict[str, Any] = {
            "epoch": ctx.state.current_epoch,
            "toxicity_rate": 0.0,
            "quality_gap": 0.0,
        }
        if interactions and ctx.metrics_calculator is not None:
            metrics_dict["toxicity_rate"] = ctx.metrics_calculator.toxicity_rate(
                interactions
            )
            metrics_dict["quality_gap"] = ctx.metrics_calculator.quality_gap(
                interactions
            )
        self._engine.check_condition(metrics_dict)

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        self._engine.on_step_start(
            ctx.state.current_epoch, ctx.state.current_step
        )


class ContractMiddleware(Middleware):
    """Contract market signing, belief updates, and metrics."""

    def __init__(self, contract_market: Any, event_bus: EventBus) -> None:
        self._market = contract_market
        self._emit = event_bus.emit
        self.last_metrics: Any = None

    @property
    def market(self) -> Any:
        return self._market

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        agents = list(ctx.state.agents.values())
        self._market.reset_epoch()
        memberships = self._market.run_signing_stage(
            agents, ctx.state.current_epoch
        )
        self._emit(
            Event(
                event_type=EventType.CONTRACT_SIGNING,
                payload={
                    "memberships": memberships,
                    "pool_composition": self._market.get_pool_composition(),
                },
                epoch=ctx.state.current_epoch,
            )
        )

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        self._market.update_beliefs()
        from swarm.contracts.metrics import compute_contract_metrics

        current_epoch_decisions = [
            d
            for d in self._market.decision_history
            if d.epoch == ctx.state.current_epoch
        ]
        self.last_metrics = compute_contract_metrics(
            current_epoch_decisions,
            self._market.get_contract_interactions(),
            payoff_engine=ctx.payoff_engine,
        )
        self._emit(
            Event(
                event_type=EventType.CONTRACT_METRICS,
                payload=self.last_metrics.to_dict(),
                epoch=ctx.state.current_epoch,
            )
        )

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        pass


class LettaMiddleware(Middleware):
    """Letta (MemGPT) governance block updates."""

    def __init__(self, lifecycle: Any) -> None:
        self._lifecycle = lifecycle

    @property
    def lifecycle(self) -> Any:
        return self._lifecycle

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        interactions = ctx.state.completed_interactions
        metrics: Dict[str, Any] = {}
        if interactions and ctx.metrics_calculator is not None:
            metrics["toxicity_rate"] = ctx.metrics_calculator.toxicity_rate(
                interactions
            )
            metrics["quality_gap"] = ctx.metrics_calculator.quality_gap(
                interactions
            )
        self._lifecycle.update_governance_block(metrics)

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        pass

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        pass


class HandlerLifecycleMiddleware(Middleware):
    """Delegates to all registered handlers' lifecycle hooks."""

    def __init__(self, handler_registry: Any) -> None:
        self._registry = handler_registry

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        for handler in self._registry.all_handlers():
            _try_call_method(handler, "on_epoch_start", ctx.state)

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        for handler in self._registry.all_handlers():
            _try_call_method(handler, "on_epoch_end", ctx.state)

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        for handler in self._registry.all_handlers():
            _try_call_method(handler, "on_step", ctx.state, ctx.state.current_step)


class NetworkDecayMiddleware(Middleware):
    """Network edge decay at epoch end."""

    def __init__(self, network: Any, event_bus: EventBus) -> None:
        self._network = network
        self._emit = event_bus.emit

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        pass

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        pruned = self._network.decay_edges()
        if pruned > 0:
            self._emit(
                Event(
                    event_type=EventType.EPOCH_COMPLETED,
                    payload={"network_edges_pruned": pruned},
                    epoch=ctx.state.current_epoch,
                )
            )

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        pass


class MemoryDecayMiddleware(Middleware):
    """Agent memory decay (rain/river model) at epoch end."""

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        pass

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        for agent in ctx.agents.values():
            if hasattr(agent, "apply_memory_decay"):
                agent.apply_memory_decay(ctx.state.current_epoch)

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        pass


class WorkRegimeAdaptMiddleware(Middleware):
    """Invoke adapt_policy on WorkRegimeAgents at each epoch boundary.

    Computes the population-average payoff and workload pressure from
    simulation state, then calls ``agent.on_epoch_end(...)`` for every
    WorkRegimeAgent in the agent pool.  This wires the drift/grievance
    model into the normal simulation update cycle.
    """

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        pass

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        from swarm.agents.work_regime_agent import WorkRegimeAgent

        wr_agents = [
            agent
            for agent in ctx.agents.values()
            if isinstance(agent, WorkRegimeAgent)
        ]
        if not wr_agents:
            return

        # Peer average payoff — use AgentState.total_payoff values
        agent_states = ctx.state.agents
        payoffs = [
            s.total_payoff for s in agent_states.values()
        ]
        peer_avg = sum(payoffs) / len(payoffs) if payoffs else 0.0

        # Workload pressure — interactions completed this epoch vs bandwidth cap
        bandwidth_cap = getattr(ctx.config, "bandwidth_cap", 0)
        n_interactions = len(ctx.state.completed_interactions)
        if bandwidth_cap and bandwidth_cap > 0:
            workload_pressure = min(1.0, n_interactions / bandwidth_cap)
        else:
            workload_pressure = 0.0

        for agent in wr_agents:
            try:
                agent.on_epoch_end(
                    peer_avg_payoff=peer_avg,
                    workload_pressure=workload_pressure,
                )
            except Exception:
                logger.debug(
                    "WorkRegimeAdaptMiddleware: on_epoch_end failed for %s",
                    agent.agent_id,
                    exc_info=True,
                )

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        pass
