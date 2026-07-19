"""Orchestrator for running the multi-agent simulation.

The orchestrator is a thin loop that delegates to:
- ``HandlerRegistry`` for action dispatch (plugin architecture)
- ``MiddlewarePipeline`` for lifecycle hooks (cross-cutting concerns)
- ``AgentScheduler`` for turn order and eligibility
- ``InteractionFinalizer`` for payoff/reputation/state updates
- ``ObservationBuilder`` for per-agent observation assembly

Domain-specific logic lives in handler and middleware classes.
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from swarm.agents.base import Action, ActionType, BaseAgent, Observation
from swarm.agents.hyperagent_self_mod import HyperagentSelfModAgent
from swarm.boundaries.external_world import ExternalEntity
from swarm.boundaries.leakage import LeakageReport
from swarm.core.agent_scheduler import AgentScheduler
from swarm.core.handler_factory import HandlerSet, build_handlers
from swarm.core.interaction_finalizer import InteractionFinalizer
from swarm.core.middleware import (
    ContractMiddleware,
    GovernanceMiddleware,
    HandlerLifecycleMiddleware,
    LettaMiddleware,
    MemoryDecayMiddleware,
    MiddlewareContext,
    MiddlewarePipeline,
    NetworkDecayMiddleware,
    PerturbationMiddleware,
    WorkRegimeAdaptMiddleware,
)
from swarm.core.observable_generator import (
    DefaultObservableGenerator,
    ObservableGenerator,
)
from swarm.core.observation_builder import ObservationBuilder
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.perturbation import PerturbationConfig, PerturbationEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.core.redteam_inspector import RedTeamInspector
from swarm.core.spawn import SpawnConfig, SpawnTree
from swarm.env.composite_tasks import CompositeTask, CompositeTaskPool
from swarm.env.feed import Feed
from swarm.env.marketplace import MarketplaceConfig
from swarm.env.network import AgentNetwork, NetworkConfig
from swarm.env.state import EnvState, InteractionProposal
from swarm.env.tasks import TaskPool
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEffect, GovernanceEngine
from swarm.knowledge.graph_memory import GraphMemoryStore
from swarm.logging.event_bus import EventBus
from swarm.logging.event_log import EventLog
from swarm.metrics.capabilities import (
    AgentCapabilityProfile,
    CapabilityAnalyzer,
    CapabilityType,
    EmergentCapabilityMetrics,
    capability_envelope,
)
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.agent import AgentState
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Timing
    n_epochs: int = 10
    steps_per_epoch: int = 10

    # Agent scheduling
    schedule_mode: str = "round_robin"  # "round_robin", "random", "priority"
    max_actions_per_step: int = 20

    # Safety rails
    max_content_length: int = 10000
    enable_rate_limits: bool = True
    enable_kill_switch: bool = True

    # Payoff configuration
    payoff_config: PayoffConfig = Field(default_factory=PayoffConfig)

    # Governance configuration
    governance_config: Optional[GovernanceConfig] = None

    # Network configuration
    network_config: Optional[NetworkConfig] = None

    # Marketplace configuration
    marketplace_config: Optional[MarketplaceConfig] = None

    # Moltipedia configuration
    moltipedia_config: Optional[Any] = None

    # Moltbook configuration
    moltbook_config: Optional[Any] = None

    # Memory tier configuration
    memory_tier_config: Optional[Any] = None

    # Scholar/literature synthesis configuration
    scholar_config: Optional[Any] = None

    # Kernel oracle configuration
    kernel_oracle_config: Optional[Any] = None

    # Spawn configuration
    spawn_config: Optional[SpawnConfig] = None

    # Rivals (Team-of-Rivals) configuration
    rivals_config: Optional[Any] = None

    # AWM (Agent World Model) configuration
    awm_config: Optional[Any] = None

    # Letta (MemGPT) configuration
    letta_config: Optional[Any] = None

    # Composite task configuration
    enable_composite_tasks: bool = False

    # Boundary configuration
    enable_boundaries: bool = False
    boundary_sensitivity_threshold: float = 0.5

    # Logging
    log_path: Optional[Path] = None
    log_events: bool = True

    # Random seed
    seed: Optional[int] = None

    # Replay/scenario metadata for logging
    scenario_id: Optional[str] = None
    replay_k: Optional[int] = None

    # Stress-test knobs
    observation_noise_probability: float = 0.0
    observation_noise_std: float = 0.0

    # Perturbation engine configuration
    perturbation_config: Optional[PerturbationConfig] = None

    # Contract screening configuration
    contracts_config: Optional[Any] = None

    # Evolutionary game (gamescape) configuration
    evo_game_config: Optional[Any] = None

    # Resource negotiation game configuration
    resource_negotiation_config: Optional[Any] = None

    # Tierra (artificial life) configuration
    tierra_config: Optional[Any] = None

    # Graph memory (cross-run trust priors) configuration
    graph_memory_path: Optional[str] = None

    # Dynamic toxicity feedback configuration
    dynamic_toxicity: Optional[Dict[str, Any]] = None


class EpochMetrics(BaseModel):
    """Metrics collected at the end of each epoch."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    epoch: int = 0
    total_interactions: int = 0
    accepted_interactions: int = 0
    total_posts: int = 0
    total_votes: int = 0
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    # Projection-geometric diagnostics (see swarm/metrics/soft_metrics.py).
    quality_correlation: float = 0.0
    baseline_harm: float = 0.0
    selection_credit: float = 0.0
    selection_saturation: float = 0.0
    avg_payoff: float = 0.0
    total_welfare: float = 0.0
    net_social_welfare: float = 0.0
    network_metrics: Optional[Dict[str, float]] = None
    capability_metrics: Optional[EmergentCapabilityMetrics] = None
    spawn_metrics: Optional[Dict[str, Any]] = None
    security_report: Optional[Any] = None
    collusion_report: Optional[Any] = None
    contract_metrics: Optional[Dict[str, Any]] = None
    capability_envelope_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "epoch": self.epoch,
            "total_interactions": self.total_interactions,
            "accepted_interactions": self.accepted_interactions,
            "total_posts": self.total_posts,
            "total_votes": self.total_votes,
            "toxicity_rate": self.toxicity_rate,
            "quality_gap": self.quality_gap,
            "quality_correlation": self.quality_correlation,
            "baseline_harm": self.baseline_harm,
            "selection_credit": self.selection_credit,
            "selection_saturation": self.selection_saturation,
            "avg_payoff": self.avg_payoff,
            "total_welfare": self.total_welfare,
            "net_social_welfare": self.net_social_welfare,
            "network_metrics": self.network_metrics,
        }
        if self.capability_metrics is not None:
            result["capability_metrics"] = (
                self.capability_metrics.to_dict()
                if hasattr(self.capability_metrics, "to_dict")
                else None
            )
        else:
            result["capability_metrics"] = None
        result["spawn_metrics"] = self.spawn_metrics
        if self.security_report is not None:
            result["security_report"] = {
                "ecosystem_threat_level": getattr(self.security_report, "ecosystem_threat_level", 0.0),
                "active_threat_count": getattr(self.security_report, "active_threat_count", 0),
                "contagion_depth": getattr(self.security_report, "contagion_depth", 0),
            }
        if self.collusion_report is not None:
            result["collusion_report"] = {
                "ecosystem_collusion_risk": getattr(self.collusion_report, "ecosystem_collusion_risk", 0.0),
                "n_flagged_pairs": getattr(self.collusion_report, "n_flagged_pairs", 0),
            }
        result["capability_envelope_metrics"] = self.capability_envelope_metrics
        return result


class Orchestrator:
    """Orchestrates the multi-agent simulation.

    The orchestrator is a thin coordination layer.  It owns the main
    simulation loop (epoch → step → agent-turn) and delegates all
    domain logic to composed components:

    - **Handlers** (via ``HandlerRegistry``): action dispatch
    - **Middleware** (via ``MiddlewarePipeline``): lifecycle hooks
    - **AgentScheduler**: turn order and eligibility
    - **InteractionFinalizer**: payoff / reputation / state updates
    - **ObservationBuilder**: per-agent observation assembly
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        state: Optional[EnvState] = None,
        # --- dependency injection for computation engines ---
        payoff_engine: Optional[SoftPayoffEngine] = None,
        proxy_computer: Optional[ProxyComputer] = None,
        metrics_calculator: Optional[SoftMetrics] = None,
        observable_generator: Optional[ObservableGenerator] = None,
        governance_engine: Optional[GovernanceEngine] = None,
    ):
        self.config = OrchestratorConfig() if config is None else config
        if not 0.0 <= self.config.observation_noise_probability <= 1.0:
            raise ValueError("observation_noise_probability must be in [0, 1]")
        if self.config.observation_noise_std < 0.0:
            raise ValueError("observation_noise_std must be >= 0")

        # Random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
        self._rng = random.Random(self.config.seed)

        # ---------------------------------------------------------------
        # Core state
        # ---------------------------------------------------------------
        self.state = state or EnvState(steps_per_epoch=self.config.steps_per_epoch)
        self.feed = Feed()
        self.task_pool = TaskPool()
        self._agents: Dict[str, BaseAgent] = {}

        # ---------------------------------------------------------------
        # Computation engines (injectable)
        # ---------------------------------------------------------------
        self.payoff_engine = payoff_engine or SoftPayoffEngine(
            self.config.payoff_config
        )
        self.proxy_computer = proxy_computer or ProxyComputer()
        self.metrics_calculator = metrics_calculator or SoftMetrics(self.payoff_engine)
        self._observable_generator: ObservableGenerator = (
            observable_generator or DefaultObservableGenerator(rng=self._rng)
        )

        # ---------------------------------------------------------------
        # Governance engine
        # ---------------------------------------------------------------
        if governance_engine is not None:
            self.governance_engine: Optional[GovernanceEngine] = governance_engine
        elif self.config.governance_config is not None:
            self.governance_engine = GovernanceEngine(
                self.config.governance_config,
                seed=self.config.seed,
            )
        else:
            self.governance_engine = None

        # ---------------------------------------------------------------
        # Network
        # ---------------------------------------------------------------
        if self.config.network_config is not None:
            self.network: Optional[AgentNetwork] = AgentNetwork(
                config=self.config.network_config,
                seed=self.config.seed,
            )
        else:
            self.network = None

        # ---------------------------------------------------------------
        # Composite tasks
        # ---------------------------------------------------------------
        if self.config.enable_composite_tasks:
            self.composite_task_pool: Optional[CompositeTaskPool] = CompositeTaskPool()
            self.capability_analyzer: Optional[CapabilityAnalyzer] = CapabilityAnalyzer(
                seed=self.config.seed
            )
        else:
            self.composite_task_pool = None
            self.capability_analyzer = None

        # ---------------------------------------------------------------
        # Perturbation engine
        # ---------------------------------------------------------------
        if self.config.perturbation_config is not None:
            self._perturbation_engine: Optional[PerturbationEngine] = (
                PerturbationEngine(
                    config=self.config.perturbation_config,
                    state=self.state,
                    network=self.network,
                    governance_engine=self.governance_engine,
                )
            )
        else:
            self._perturbation_engine = None

        # ---------------------------------------------------------------
        # Event bus & logging
        # ---------------------------------------------------------------
        self._event_bus = EventBus()
        self._event_bus.set_enrichment(
            seed=self.config.seed,
            scenario_id=self.config.scenario_id,
            replay_k=self.config.replay_k,
        )

        if self.config.log_path:
            self.event_log: Optional[EventLog] = EventLog(self.config.log_path)
        else:
            self.event_log = None

        if self.event_log is not None and self.config.log_events:
            self._event_bus.subscribe(self.event_log.append)

        # ---------------------------------------------------------------
        # Graph memory (cross-run trust priors)
        # ---------------------------------------------------------------
        if self.config.graph_memory_path is not None:
            self._graph_memory: Optional[GraphMemoryStore] = GraphMemoryStore(
                self.config.graph_memory_path
            )
        else:
            self._graph_memory = None

        # ---------------------------------------------------------------
        # Callbacks
        # ---------------------------------------------------------------
        self._on_epoch_end: List[Callable[[EpochMetrics], None]] = []
        self._on_interaction_complete: List[
            Callable[[SoftInteraction, float, float], None]
        ] = []

        # ---------------------------------------------------------------
        # Adaptive governance controller
        # ---------------------------------------------------------------
        self._adaptive_controller = None
        if (
            self.governance_engine
            and self.config.governance_config
            and self.config.governance_config.adaptive_controller_enabled
        ):
            from swarm.governance.adaptive_controller import (
                AdaptiveGovernanceController,
            )

            self._adaptive_controller = AdaptiveGovernanceController(
                governance_engine=self.governance_engine,
                event_bus=self._event_bus,
                config=self.config.governance_config,
            )
            self._on_epoch_end.append(self._adaptive_controller.on_epoch_end)

        # ---------------------------------------------------------------
        # Interaction finalizer (extracted component)
        # ---------------------------------------------------------------
        self._finalizer = InteractionFinalizer(
            state=self.state,
            payoff_engine=self.payoff_engine,
            proxy_computer=self.proxy_computer,
            observable_generator=self._observable_generator,
            governance_engine=self.governance_engine,
            network=self.network,
            agents=self._agents,
            on_interaction_complete=self._on_interaction_complete,
            event_bus=self._event_bus,
        )

        # ---------------------------------------------------------------
        # Handlers (via factory)
        # ---------------------------------------------------------------
        self._handlers: HandlerSet = build_handlers(
            self.config,
            event_bus=self._event_bus,
            feed=self.feed,
            task_pool=self.task_pool,
            finalizer=self._finalizer,
            network=self.network,
            governance_engine=self.governance_engine,
            rng=self._rng,
        )
        self._handler_registry = self._handlers.registry

        # Expose named handler refs for public API compatibility
        self.marketplace = self._handlers.marketplace
        self._marketplace_handler = self._handlers.marketplace_handler
        self._moltipedia_handler = self._handlers.moltipedia_handler
        self._moltbook_handler = self._handlers.moltbook_handler
        self._memory_handler = self._handlers.memory_handler
        self._scholar_handler = self._handlers.scholar_handler
        self._kernel_handler = self._handlers.kernel_handler
        self._rivals_handler = self._handlers.rivals_handler
        self._awm_handler = self._handlers.awm_handler
        self._evo_game_handler = self._handlers.evo_game_handler
        self._resource_negotiation_handler = self._handlers.resource_negotiation_handler
        self._tierra_handler = self._handlers.tierra_handler
        self._boundary_handler = self._handlers.boundary_handler
        self._feed_handler = self._handlers.feed_handler
        self._core_interaction_handler = self._handlers.core_interaction_handler
        self._task_handler = self._handlers.task_handler
        self._coding_handler = self._handlers.coding_handler

        # Boundary sub-components for public API
        self.external_world = self._handlers.external_world
        self.flow_tracker = self._handlers.flow_tracker
        self.policy_engine = self._handlers.policy_engine
        self.leakage_detector = self._handlers.leakage_detector

        # ---------------------------------------------------------------
        # Contract market
        # ---------------------------------------------------------------
        if self.config.contracts_config is not None:
            from swarm.scenarios.loader import build_contract_market

            self.contract_market: Optional[Any] = build_contract_market(
                self.config.contracts_config,
                seed=self.config.seed,
            )
        else:
            self.contract_market = None
        self._last_contract_metrics: Optional[Any] = None

        # ---------------------------------------------------------------
        # Letta lifecycle
        # ---------------------------------------------------------------
        if self.config.letta_config is not None:
            from swarm.bridges.letta.lifecycle import LettaLifecycleManager

            self._letta_lifecycle: Optional[Any] = LettaLifecycleManager(
                self.config.letta_config,
            )
        else:
            self._letta_lifecycle = None

        # ---------------------------------------------------------------
        # Spawn tree
        # ---------------------------------------------------------------
        if self.config.spawn_config is not None and self.config.spawn_config.enabled:
            self._spawn_tree: Optional[SpawnTree] = SpawnTree(self.config.spawn_config)
            self._spawn_counter: int = 0
        else:
            self._spawn_tree = None
            self._spawn_counter = 0

        # ---------------------------------------------------------------
        # Observation builder
        # ---------------------------------------------------------------
        self._obs_builder = ObservationBuilder(
            config=self.config,
            state=self.state,
            feed=self.feed,
            task_pool=self.task_pool,
            network=self.network,
            handler_registry=self._handler_registry,
            rng=self._rng,
            spawn_tree=self._spawn_tree,
        )

        # ---------------------------------------------------------------
        # Red-team inspector
        # ---------------------------------------------------------------
        self._redteam = RedTeamInspector(self._agents, self.state)

        # ---------------------------------------------------------------
        # Agent scheduler
        # ---------------------------------------------------------------
        self._scheduler = AgentScheduler(
            schedule_mode=self.config.schedule_mode,
            max_actions_per_step=self.config.max_actions_per_step,
            rng=self._rng,
        )

        # ---------------------------------------------------------------
        # Middleware pipeline (lifecycle hooks for cross-cutting concerns)
        # ---------------------------------------------------------------
        self._pipeline = MiddlewarePipeline()
        self._build_pipeline()

        # ---------------------------------------------------------------
        # Epoch metrics history
        # ---------------------------------------------------------------
        self._epoch_metrics: List[EpochMetrics] = []

        # ---------------------------------------------------------------
        # External agent support
        # ---------------------------------------------------------------
        self._external_action_queue: Optional[Any] = None
        self._external_observations: Dict[str, Dict[str, Any]] = {}

    # ===================================================================
    # Pipeline construction
    # ===================================================================

    def _build_pipeline(self) -> None:
        """Assemble the middleware pipeline in execution order.

        Order matters: governance must run before handler lifecycle hooks
        (which may read governance state), and contract signing must happen
        before handler epoch-start hooks.
        """
        # 1. Governance (adaptive updates + epoch/step effects)
        if self.governance_engine is not None:
            self._gov_mw: Optional[GovernanceMiddleware] = GovernanceMiddleware(
                self.governance_engine,
                self._adaptive_controller,
            )
            self._pipeline.add(self._gov_mw)
        else:
            self._gov_mw = None

        # 2. Letta governance block update
        if self._letta_lifecycle is not None:
            self._letta_mw: Optional[LettaMiddleware] = LettaMiddleware(
                self._letta_lifecycle
            )
            self._pipeline.add(self._letta_mw)
        else:
            self._letta_mw = None

        # 3. Contract market signing
        if self.contract_market is not None:
            self._contract_mw: Optional[ContractMiddleware] = ContractMiddleware(
                self.contract_market, self._event_bus
            )
            self._pipeline.add(self._contract_mw)
        else:
            self._contract_mw = None

        # 4. Perturbation engine triggers
        if self._perturbation_engine is not None:
            self._perturb_mw: Optional[PerturbationMiddleware] = PerturbationMiddleware(
                self._perturbation_engine
            )
            self._pipeline.add(self._perturb_mw)
        else:
            self._perturb_mw = None

        # 5. Handler lifecycle hooks (on_epoch_start/end, on_step)
        self._pipeline.add(HandlerLifecycleMiddleware(self._handler_registry))

        # 6. Network edge decay at epoch end
        if self.network is not None:
            self._pipeline.add(NetworkDecayMiddleware(self.network, self._event_bus))

        # 7. Agent memory decay at epoch end
        self._pipeline.add(MemoryDecayMiddleware())

        # 8. WorkRegimeAgent policy adaptation at epoch end
        self._pipeline.add(WorkRegimeAdaptMiddleware())

        # 9. Dynamic toxicity feedback loops (optional)
        self._build_dynamic_toxicity()

    def _build_dynamic_toxicity(self) -> None:
        """Wire dynamic toxicity middleware if configured."""
        dt_config = self.config.dynamic_toxicity
        if not dt_config:
            return

        from swarm.core.dynamic_toxicity import (
            ProxyCalibrationDriftMiddleware,
            QualityContagionMiddleware,
            TrustErosionMiddleware,
        )

        enabled = dt_config.get("enabled", [])

        if "proxy_drift" in enabled:
            params = dt_config.get("proxy_drift", {})
            self._pipeline.add(
                ProxyCalibrationDriftMiddleware(
                    self.proxy_computer,
                    self._event_bus,
                    alpha=params.get("alpha", 0.5),
                    k_floor=params.get("k_floor", 0.5),
                )
            )

        if "trust_erosion" in enabled:
            params = dt_config.get("trust_erosion", {})
            self._pipeline.add(
                TrustErosionMiddleware(
                    self._event_bus,
                    beta=params.get("beta", 0.3),
                    exit_threshold=params.get("exit_threshold", 0.35),
                    window=params.get("window", 3),
                    min_honest=params.get("min_honest", 1),
                    seed=self.config.seed,
                )
            )

        if "quality_contagion" in enabled:
            params = dt_config.get("quality_contagion", {})
            self._pipeline.add(
                QualityContagionMiddleware(
                    self._event_bus,
                    gamma=params.get("gamma", 0.1),
                    neutral=params.get("neutral", 0.5),
                )
            )

    def _make_context(self) -> MiddlewareContext:
        """Build a ``MiddlewareContext`` from current orchestrator state."""
        return MiddlewareContext(
            state=self.state,
            config=self.config,
            agents=self._agents,
            event_bus=self._event_bus,
            network=self.network,
            governance_engine=self.governance_engine,
            metrics_calculator=self.metrics_calculator,
            payoff_engine=self.payoff_engine,
            handler_registry=self._handler_registry,
            finalizer=self._finalizer,
        )

    # ===================================================================
    # Agent registration
    # ===================================================================

    def register_agent(self, agent: BaseAgent) -> AgentState:
        """Register an agent with the simulation."""
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")

        self._agents[agent.agent_id] = agent

        state = self.state.add_agent(
            agent_id=agent.agent_id,
            name=getattr(agent, "name", agent.agent_id),
            agent_type=agent.agent_type,
        )

        if self._spawn_tree is not None:
            self._spawn_tree.register_root(agent.agent_id)

        self._emit_event(
            Event(
                event_type=EventType.AGENT_CREATED,
                agent_id=agent.agent_id,
                payload={
                    "agent_type": agent.agent_type.value,
                    "name": getattr(agent, "name", agent.agent_id),
                    "roles": [r.value for r in agent.roles],
                },
                epoch=self.state.current_epoch,
                step=self.state.current_step,
            )
        )

        # Register with heartbeat lever so it tracks from first step
        if self.governance_engine is not None:
            hb = self.governance_engine.get_attestation_heartbeat_lever()
            if hb is not None:
                hb.register_agent(agent.agent_id, initial_step=self.state.current_step)

        return state

    def _initialize_network(self) -> None:
        """Initialize network topology with registered agents."""
        agent_ids = list(self._agents.keys())
        if self.network is not None:
            self.network.initialize(agent_ids)
        if self.governance_engine is not None:
            self.governance_engine.set_collusion_agent_ids(agent_ids)
        if self._letta_lifecycle is not None:
            self._letta_lifecycle.start()
            for agent in self._agents.values():
                if hasattr(agent, "_lazy_init") and hasattr(agent, "_letta_config"):
                    agent._lazy_init(self._letta_lifecycle)

    # ===================================================================
    # Main simulation loop
    # ===================================================================

    def run(self) -> List[EpochMetrics]:
        """Run the full simulation."""
        # ---------------------------------------------------------------
        # Load prior memory from prior run(s) if graph_memory is enabled
        # ---------------------------------------------------------------
        if self._graph_memory is not None:
            prior_snapshots = self._graph_memory.load_all()
            for agent in self._agents.values():
                if agent.agent_id in prior_snapshots:
                    snapshot = prior_snapshots[agent.agent_id]
                    agent.load_prior_memory(snapshot)
                    logger.info(
                        f"Loaded prior memory for {agent.agent_id} from snapshot "
                        f"(trust priors from run={snapshot.run_id}, epoch={snapshot.epoch})"
                    )

        self._initialize_network()

        self._emit_event(
            Event(
                event_type=EventType.SIMULATION_STARTED,
                payload={
                    "n_epochs": self.config.n_epochs,
                    "steps_per_epoch": self.config.steps_per_epoch,
                    "n_agents": len(self._agents),
                    "seed": self.config.seed,
                    "scenario_id": self.config.scenario_id,
                    "replay_k": self.config.replay_k,
                },
            )
        )

        for _epoch in range(self.config.n_epochs):
            epoch_metrics = self._run_epoch()
            self._epoch_metrics.append(epoch_metrics)
            for callback in self._on_epoch_end:
                callback(epoch_metrics)

        if self._letta_lifecycle is not None:
            self._letta_lifecycle.shutdown()

        self._emit_event(
            Event(
                event_type=EventType.SIMULATION_ENDED,
                payload={
                    "total_epochs": self.config.n_epochs,
                    "final_metrics": self._epoch_metrics[-1].to_dict()
                    if self._epoch_metrics
                    else {},
                },
            )
        )

        # ---------------------------------------------------------------
        # Save memory snapshots at run end if graph_memory is enabled
        # ---------------------------------------------------------------
        if self._graph_memory is not None:
            run_id = self.config.scenario_id or "unknown"
            final_epoch = self.config.n_epochs - 1
            self._graph_memory.save_all(list(self._agents.values()), run_id, final_epoch)
            logger.info(
                f"Saved memory snapshots for {len(self._agents)} agents "
                f"(run={run_id}, epoch={final_epoch})"
            )

        return self._epoch_metrics

    def _run_epoch(self) -> EpochMetrics:
        """Run a single epoch."""
        epoch_start = self.state.current_epoch
        ctx = self._make_context()

        # --- Epoch pre-hooks ---
        self._pipeline.on_epoch_start(ctx)

        # --- Artifact housekeeping (always-on, independent of cascade lever) ---
        current_step = epoch_start * self.config.steps_per_epoch
        self.state.artifact_registry.gc(current_step, max_age_steps=200)

        # --- Self-modification for HyperagentSelfModAgent instances ---
        for agent in self._agents.values():
            if isinstance(agent, HyperagentSelfModAgent):
                agent.self_modify(epoch_start)

        # --- Steps ---
        for _step in range(self.config.steps_per_epoch):
            if self.state.is_paused:
                break
            self._run_step(ctx)
            self.state.advance_step()

        # --- Epoch post-hooks ---
        self._pipeline.on_epoch_end(ctx)

        # Update contract metrics ref
        if self._contract_mw is not None:
            self._last_contract_metrics = self._contract_mw.last_metrics

        metrics = self._compute_epoch_metrics()

        self._emit_event(
            Event(
                event_type=EventType.EPOCH_COMPLETED,
                payload=metrics.to_dict(),
                epoch=epoch_start,
            )
        )

        self.state.advance_epoch()
        return metrics

    def _run_step(self, ctx: Optional[MiddlewareContext] = None) -> None:
        """Run a single step within an epoch."""
        if ctx is None:
            ctx = self._make_context()
        self._pipeline.on_step_start(ctx)

        dropped = (
            self._perturbation_engine.get_dropped_agents()
            if self._perturbation_engine is not None
            else set()
        )

        eligible = self._scheduler.get_eligible(
            self._agents,
            self.state,
            governance_engine=self.governance_engine,
            dropped_agents=dropped,
        )

        for agent_id in eligible:
            agent = self._agents[agent_id]
            observation = self._obs_builder.build(agent_id)
            action = self._select_action(agent, observation)
            self._execute_action(action)

        self._resolve_pending_interactions()

    # ===================================================================
    # Action selection
    # ===================================================================

    def _select_action(self, agent: BaseAgent, observation: Observation) -> Action:
        """Select an action, optionally using governance ensembling or resampling."""
        action = self._select_action_base(agent, observation)

        # --- Resample protocol (Bhatt et al., 2025) ---
        # Conditionally resample when the monitor flags suspicion based on
        # the agent's recent trajectory and reputation.
        action = self._maybe_resample(agent, observation, action)

        return action

    def _select_action_base(self, agent: BaseAgent, observation: Observation) -> Action:
        """Core action selection with optional self-ensemble."""
        if (
            self.governance_engine is None
            or not self.governance_engine.config.self_ensemble_enabled
            or self.governance_engine.config.self_ensemble_samples <= 1
        ):
            return agent.act(observation)

        samples = self.governance_engine.config.self_ensemble_samples
        candidate_actions = [agent.act(observation) for _ in range(samples)]
        selected = self._majority_action(candidate_actions)
        selected.metadata["ensemble_samples"] = samples
        return selected

    def _maybe_resample(
        self,
        agent: BaseAgent,
        observation: Observation,
        original_action: Action,
    ) -> Action:
        """Apply resample protocol if suspicion warrants it.

        Uses the agent's reputation as a pre-execution proxy for p,
        since actual p is only available after handler execution.
        """
        if self.governance_engine is None:
            return original_action

        resample_lever = self.governance_engine.get_resample_lever()
        if resample_lever is None:
            return original_action

        # Pre-execution suspicion: use agent reputation as p proxy
        agent_state = self.state.agents.get(agent.agent_id)
        if agent_state is None:
            return original_action

        # Map reputation [-1, 1] to p-proxy [0, 1]
        p_proxy = max(0.0, min(1.0, 0.5 + agent_state.reputation * 0.5))

        if not resample_lever.should_resample(p_proxy, agent.agent_id):
            return original_action

        # Resample: draw additional action samples
        n_samples = self.governance_engine.config.resample_n_samples
        resample_actions = [agent.act(observation) for _ in range(n_samples)]

        all_actions = [original_action] + resample_actions
        selected = self._majority_action(all_actions)

        # Annotate with resample metadata for the lever to process later
        was_blocked = self._action_signature(selected) != self._action_signature(original_action)
        selected.metadata["resample_triggered"] = True
        selected.metadata["resample_n_samples"] = n_samples
        selected.metadata["resample_blocked"] = was_blocked
        # Store per-sample p-proxy values for incrimination analysis.
        # True p is only available after handler execution, so we
        # differentiate pre-execution proxies based on action agreement:
        selected.metadata["resample_p_proxy"] = p_proxy
        per_sample_p = self._compute_resample_p_values(original_action, resample_actions, p_proxy)
        selected.metadata["resample_p_values"] = per_sample_p

        return selected

    async def _select_action_async(
        self, agent: BaseAgent, observation: Observation
    ) -> Action:
        """Async action selection with optional governance ensembling and resampling."""
        if (
            self.governance_engine is None
            or not self.governance_engine.config.self_ensemble_enabled
            or self.governance_engine.config.self_ensemble_samples <= 1
        ):
            if self._is_llm_agent(agent):
                action = await agent.act_async(observation)  # type: ignore[attr-defined, no-any-return]
            else:
                action = agent.act(observation)  # type: ignore[no-any-return]
            return await self._maybe_resample_async(agent, observation, action)

        samples = self.governance_engine.config.self_ensemble_samples
        candidate_actions: List[Action] = []
        for _ in range(samples):
            if self._is_llm_agent(agent):
                candidate_actions.append(await agent.act_async(observation))  # type: ignore[attr-defined]
            else:
                candidate_actions.append(agent.act(observation))

        selected = self._majority_action(candidate_actions)
        selected.metadata["ensemble_samples"] = samples
        return await self._maybe_resample_async(agent, observation, selected)

    async def _maybe_resample_async(
        self,
        agent: BaseAgent,
        observation: Observation,
        original_action: Action,
    ) -> Action:
        """Async version of :meth:`_maybe_resample`."""
        if self.governance_engine is None:
            return original_action

        resample_lever = self.governance_engine.get_resample_lever()
        if resample_lever is None:
            return original_action

        agent_state = self.state.agents.get(agent.agent_id)
        if agent_state is None:
            return original_action

        p_proxy = max(0.0, min(1.0, 0.5 + agent_state.reputation * 0.5))

        if not resample_lever.should_resample(p_proxy, agent.agent_id):
            return original_action

        n_samples = self.governance_engine.config.resample_n_samples
        resample_actions: List[Action] = []
        for _ in range(n_samples):
            if self._is_llm_agent(agent):
                resample_actions.append(await agent.act_async(observation))  # type: ignore[attr-defined]
            else:
                resample_actions.append(agent.act(observation))

        all_actions = [original_action] + resample_actions
        selected = self._majority_action(all_actions)

        was_blocked = self._action_signature(selected) != self._action_signature(original_action)
        selected.metadata["resample_triggered"] = True
        selected.metadata["resample_n_samples"] = n_samples
        selected.metadata["resample_blocked"] = was_blocked
        selected.metadata["resample_p_proxy"] = p_proxy
        per_sample_p = self._compute_resample_p_values(original_action, resample_actions, p_proxy)
        selected.metadata["resample_p_values"] = per_sample_p

        return selected

    def _compute_resample_p_values(
        self, original_action: Action, resample_actions: List[Action], p_proxy: float
    ) -> List[float]:
        """Compute per-sample p-proxy values for resample evidence.

        Resampled actions that DIFFER from the original suspicious action
        suggest the agent is capable of benign behavior (higher p proxy),
        while actions matching the original confirm suspicion (lower p).
        """
        original_sig = self._action_signature(original_action)
        per_sample_p = []
        for ra in resample_actions:
            if self._action_signature(ra) != original_sig:
                # Divergent action → exonerating evidence
                per_sample_p.append(min(1.0, p_proxy + 0.3))
            else:
                # Matching action → confirming suspicion
                per_sample_p.append(p_proxy)
        return per_sample_p

    def _majority_action(self, actions: List[Action]) -> Action:
        """Choose majority action signature with deterministic tie-break."""
        if not actions:
            return Action(action_type=ActionType.NOOP)

        counts: Dict[Tuple, int] = {}
        first_index: Dict[Tuple, int] = {}
        for idx, action in enumerate(actions):
            key = self._action_signature(action)
            counts[key] = counts.get(key, 0) + 1
            if key not in first_index:
                first_index[key] = idx

        best_key = max(
            counts.keys(),
            key=lambda key: (counts[key], -first_index[key]),
        )
        for action in actions:
            if self._action_signature(action) == best_key:
                return action
        return actions[0]

    @staticmethod
    def _action_signature(action: Action) -> Tuple:
        """Stable signature for grouping semantically equivalent actions."""
        return (
            action.action_type.value,
            action.target_id,
            action.counterparty_id,
            action.interaction_type.value,
            action.vote_direction,
            action.content,
        )

    # ===================================================================
    # Action execution pipeline
    # ===================================================================

    def _execute_action(self, action: Action) -> bool:
        """Execute an agent action via handler registry."""
        # --- Core actions (orchestrator-owned) ---
        core_result = self._handle_core_action(action)
        if core_result is not None:
            return core_result

        # --- Handler-dispatched actions ---
        if not isinstance(action.action_type, ActionType):
            return False

        # Check hardware-trust constrained-mode action filtering
        if self.governance_engine is not None:
            action_name = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
            if not self.governance_engine.is_action_allowed(action.agent_id, action_name):
                logger.info(
                    "Action %s blocked for agent %s by hardware trust lever",
                    action_name,
                    action.agent_id,
                )
                return False

        handler = self._handler_registry.get_handler(action.action_type)
        if handler is None:
            return False

        try:
            result = handler.handle_action(action, self.state)
        except (RuntimeError, ValueError, TypeError, AttributeError):
            logger.debug(
                "Handler %s.handle_action failed",
                type(handler).__name__,
                exc_info=True,
            )
            return False

        if not result.success:
            return False

        if result.observables is None:
            return True

        # Standard proxy computation + interaction finalization
        v_hat, p = self.proxy_computer.compute_labels(result.observables)

        interaction_type = InteractionType.COLLABORATION
        if hasattr(result, "interaction_type") and isinstance(
            getattr(result, "interaction_type", None), InteractionType
        ):
            interaction_type = result.interaction_type

        tau = 0.0
        if hasattr(result, "tau") and result.tau != 0.0:
            tau = result.tau
        elif hasattr(result, "points") and result.points != 0.0:
            tau = -result.points

        ground_truth_val = getattr(result, "ground_truth", None)
        if ground_truth_val is None and hasattr(result, "submission"):
            submission = result.submission
            if submission is not None:
                ground_truth_val = -1 if submission.is_cheat else 1

        # Merge handler metadata with action metadata (resample annotations)
        merged_metadata = dict(result.metadata or {})
        if action.metadata:
            for key in ("resample_triggered", "resample_n_samples",
                        "resample_blocked", "resample_p_proxy",
                        "resample_p_values"):
                if key in action.metadata:
                    merged_metadata[key] = action.metadata[key]

        interaction = SoftInteraction(
            initiator=result.initiator_id,
            counterparty=result.counterparty_id,
            interaction_type=interaction_type,
            accepted=result.accepted,
            task_progress_delta=result.observables.task_progress_delta,
            rework_count=result.observables.rework_count,
            verifier_rejections=result.observables.verifier_rejections,
            tool_misuse_flags=result.observables.tool_misuse_flags,
            counterparty_engagement_delta=result.observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            tau=tau,
            metadata=merged_metadata,
            **({"ground_truth": ground_truth_val} if ground_truth_val is not None else {}),
        )

        if self.contract_market is not None:
            interaction = self.contract_market.route_interaction(interaction)

        # Pass handler result for artifact wiring if it produced/consumed artifacts.
        # Guard with getattr for legacy handler result types that don't
        # extend HandlerActionResult (e.g. MoltbookActionResult).
        artifact_result = None
        if (
            getattr(result, "produced_artifacts", None)
            or getattr(result, "consumed_artifact_ids", None)
        ):
            artifact_result = result

        gov_effect, _, _ = self._finalize_interaction(
            interaction, handler_result=artifact_result
        )

        try:
            handler.post_finalize(result, interaction, gov_effect, self.state)
        except (RuntimeError, ValueError, TypeError, AttributeError):
            logger.debug(
                "Handler %s.post_finalize failed",
                type(handler).__name__,
                exc_info=True,
            )

        return True

    def _handle_core_action(self, action: Action) -> Optional[bool]:
        """Handle NOOP and SPAWN_SUBAGENT — the only orchestrator-owned actions."""
        if action.action_type == ActionType.NOOP:
            return True
        if action.action_type == ActionType.SPAWN_SUBAGENT:
            return self._handle_spawn_subagent(action)
        return None

    # ===================================================================
    # Spawn
    # ===================================================================

    def _handle_spawn_subagent(self, action: Action) -> bool:
        """Handle a SPAWN_SUBAGENT action."""
        if self._spawn_tree is None:
            return False

        parent_id = action.agent_id
        parent_state = self.state.get_agent(parent_id)
        if parent_state is None:
            return False

        global_step = (
            self.state.current_epoch * self.config.steps_per_epoch
            + self.state.current_step
        )

        can, reason = self._spawn_tree.can_spawn(
            parent_id, global_step, parent_state.resources
        )
        if not can:
            self._emit_event(
                Event(
                    event_type=EventType.SPAWN_REJECTED,
                    agent_id=parent_id,
                    payload={"reason": reason},
                    epoch=self.state.current_epoch,
                    step=self.state.current_step,
                )
            )
            return False

        spawn_cfg = self._spawn_tree.config
        parent_state.update_resources(-spawn_cfg.spawn_cost)

        child_type_key = action.metadata.get("child_type")
        if not child_type_key:
            child_type_key = parent_state.agent_type.value
        child_config = action.metadata.get("child_config", {})

        from swarm.scenarios.loader import AGENT_TYPES

        agent_class = AGENT_TYPES.get(child_type_key)
        if agent_class is None:
            self._emit_event(
                Event(
                    event_type=EventType.SPAWN_REJECTED,
                    agent_id=parent_id,
                    payload={"reason": f"unknown_agent_type:{child_type_key}"},
                    epoch=self.state.current_epoch,
                    step=self.state.current_step,
                )
            )
            return False

        self._spawn_counter += 1
        child_id = f"{parent_id}_child{self._spawn_counter}"
        child_rng = random.Random((self.config.seed or 0) + self._spawn_counter)

        # Tierra-specific: mutate genome and split resources
        is_tierra = child_type_key == "tierra"
        child_initial_resources = spawn_cfg.initial_child_resources
        tierra_genome = None

        if is_tierra and action.metadata.get("genome"):
            from swarm.agents.tierra_agent import TierraGenome

            parent_genome = TierraGenome.from_dict(action.metadata["genome"])
            mutation_std = 0.05
            if self._tierra_handler is not None:
                mutation_std = self._tierra_handler.config.mutation_std
            tierra_genome = parent_genome.mutate(child_rng, mutation_std)

            share_frac = parent_genome.resource_share_fraction
            child_initial_resources = parent_state.resources * share_frac
            parent_state.update_resources(-child_initial_resources)

            child_agent = agent_class(  # type: ignore[call-arg]
                agent_id=child_id,
                name=child_id,
                config=child_config if child_config else None,
                rng=child_rng,
                genome=tierra_genome,
            )
            child_agent.generation = action.metadata.get("generation", 0)  # type: ignore[attr-defined]

            if self._tierra_handler is not None:
                self._tierra_handler.register_genome(child_id, tierra_genome.to_dict())
                self._tierra_handler._births += 1
        else:
            child_agent = agent_class(  # type: ignore[call-arg]
                agent_id=child_id,
                name=child_id,
                config=child_config if child_config else None,
                rng=child_rng,
            )

        self._agents[child_id] = child_agent

        inherited_rep = parent_state.reputation * spawn_cfg.reputation_inheritance_factor
        child_state = self.state.add_agent(
            agent_id=child_id,
            name=child_id,
            agent_type=child_agent.agent_type,
            initial_reputation=inherited_rep,
            initial_resources=child_initial_resources,
        )
        child_state.parent_id = parent_id

        self._spawn_tree.register_spawn(
            parent_id=parent_id,
            child_id=child_id,
            epoch=self.state.current_epoch,
            step=self.state.current_step,
            global_step=global_step,
        )

        if self.network is not None:
            self.network.add_node(child_id)
            self.network.add_edge(parent_id, child_id)

        self._emit_event(
            Event(
                event_type=EventType.AGENT_SPAWNED,
                agent_id=child_id,
                payload={
                    "parent_id": parent_id,
                    "child_type": child_type_key,
                    "depth": self._spawn_tree.get_depth(child_id),
                    "inherited_reputation": inherited_rep,
                    "initial_resources": child_initial_resources,
                    "spawn_cost": spawn_cfg.spawn_cost,
                },
                epoch=self.state.current_epoch,
                step=self.state.current_step,
            )
        )

        return True

    # ===================================================================
    # Interaction resolution
    # ===================================================================

    def _resolve_pending_interactions(self) -> None:
        """Resolve any remaining pending interactions."""
        proposals = list(self.state.pending_proposals.values())

        for proposal in proposals:
            counterparty_id = proposal.counterparty_id

            if counterparty_id not in self._agents:
                continue
            if not self.state.can_agent_act(counterparty_id):
                continue

            counterparty = self._agents[counterparty_id]
            observation = self._obs_builder.build(counterparty_id)

            from swarm.agents.base import InteractionProposal as AgentProposal

            agent_proposal = AgentProposal(
                proposal_id=proposal.proposal_id,
                initiator_id=proposal.initiator_id,
                counterparty_id=proposal.counterparty_id,
                interaction_type=InteractionType(proposal.interaction_type),
                content=proposal.content,
                offered_transfer=proposal.metadata.get("offered_transfer", 0),
            )

            accept = counterparty.accept_interaction(agent_proposal, observation)

            self.state.remove_proposal(proposal.proposal_id)
            self._complete_interaction(proposal, accepted=accept)

    def _complete_interaction(
        self,
        proposal: InteractionProposal,
        accepted: bool,
    ) -> None:
        """Complete an interaction and compute payoffs."""
        if self.contract_market is not None:
            observables = self._observable_generator.generate(
                proposal, accepted, self.state
            )
            v_hat, p = self.proxy_computer.compute_labels(observables)
            interaction = SoftInteraction(
                interaction_id=proposal.proposal_id,
                initiator=proposal.initiator_id,
                counterparty=proposal.counterparty_id,
                interaction_type=InteractionType(proposal.interaction_type),
                accepted=accepted,
                task_progress_delta=observables.task_progress_delta,
                rework_count=observables.rework_count,
                verifier_rejections=observables.verifier_rejections,
                tool_misuse_flags=observables.tool_misuse_flags,
                counterparty_engagement_delta=observables.counterparty_engagement_delta,
                v_hat=v_hat,
                p=p,
                tau=proposal.metadata.get("offered_transfer", 0),
                metadata=proposal.metadata,
            )
            interaction = self.contract_market.route_interaction(interaction)
            self._finalizer.finalize_interaction(interaction)
        else:
            self._finalizer.complete_interaction(proposal, accepted)

    def _finalize_interaction(
        self,
        interaction: SoftInteraction,
        handler_result: Any = None,
    ) -> Tuple[GovernanceEffect, float, float]:
        """Apply governance, compute payoffs, update state, and emit events."""
        return self._finalizer.finalize_interaction(
            interaction, handler_result=handler_result
        )

    def _generate_observables(
        self,
        proposal: InteractionProposal,
        accepted: bool,
    ) -> ProxyObservables:
        """Generate observable signals for an interaction."""
        return self._observable_generator.generate(proposal, accepted, self.state)

    # ===================================================================
    # Metrics
    # ===================================================================

    def _compute_epoch_metrics(self) -> EpochMetrics:
        """Compute metrics for the current epoch."""
        interactions = self.state.completed_interactions

        network_metrics = None
        if self.network is not None:
            network_metrics = self.network.get_metrics()

        capability_metrics = None
        if self.capability_analyzer is not None:
            capability_metrics = self.capability_analyzer.compute_metrics()

        security_report = None
        if self.governance_engine is not None:
            security_report = self.governance_engine.get_security_report()

        collusion_report = None
        if self.governance_engine is not None:
            collusion_report = self.governance_engine.get_collusion_report()

        spawn_metrics_dict = None
        if self._spawn_tree is not None:
            spawn_metrics_dict = {
                "total_spawned": self._spawn_tree.total_spawned,
                "max_depth": self._spawn_tree.max_tree_depth(),
                "depth_distribution": self._spawn_tree.depth_distribution(),
                "tree_sizes": self._spawn_tree.tree_size_distribution(),
            }

        contract_metrics_dict = None
        if self._last_contract_metrics is not None:
            contract_metrics_dict = self._last_contract_metrics.to_dict()

        # Capability envelope for self-modifying agents
        envelope_dict = None
        self_mod_agents = {
            aid: agent
            for aid, agent in self._agents.items()
            if isinstance(agent, HyperagentSelfModAgent)
        }
        if self_mod_agents:
            # Build profiles from agent proxy weights.
            # Map proxy weight dimensions to capability types:
            #   task_progress  → EXECUTION
            #   rework_penalty → VERIFICATION  (governed)
            #   verifier_penalty → ANALYSIS    (governed)
            #   engagement_signal → COMMUNICATION (ungoverned)
            all_dims = {
                CapabilityType.EXECUTION,
                CapabilityType.VERIFICATION,
                CapabilityType.ANALYSIS,
                CapabilityType.COMMUNICATION,
            }
            profiles: Dict[str, AgentCapabilityProfile] = {}
            governed: Dict[str, set] = {}
            for aid, agent in self_mod_agents.items():
                demonstrated = set(all_dims)
                # When ungoverned weight exceeds baseline, agent has
                # expanded beyond governance coverage
                ungoverned_w = (
                    agent.proxy_weights.get("engagement_signal", 0)
                    + agent.proxy_weights.get("task_progress", 0)
                )
                if ungoverned_w > 0.6:
                    demonstrated.add(CapabilityType.CREATIVITY)
                profiles[aid] = AgentCapabilityProfile(
                    agent_id=aid,
                    capabilities=demonstrated,
                )
                # Governance covers only monitored dimensions
                gov_caps = {
                    CapabilityType.VERIFICATION,
                    CapabilityType.ANALYSIS,
                }
                governed[aid] = gov_caps

            result = capability_envelope(
                profiles, governed, step=self.state.current_step,
                all_capability_types=all_dims | {CapabilityType.CREATIVITY},
            )
            envelope_dict = {
                "mean_envelope": result.mean_envelope,
                "mean_governance_coverage": result.mean_governance_coverage,
                "mean_governance_gap": result.mean_governance_gap,
                "max_governance_gap": result.max_governance_gap,
                "n_self_mod_agents": len(self_mod_agents),
                "max_modification_depth": max(
                    a.modification_depth for a in self_mod_agents.values()
                ),
            }

        if not interactions:
            return EpochMetrics(
                epoch=self.state.current_epoch,
                network_metrics=network_metrics,
                capability_metrics=capability_metrics,
                spawn_metrics=spawn_metrics_dict,
                security_report=security_report,
                collusion_report=collusion_report,
                contract_metrics=contract_metrics_dict,
                capability_envelope_metrics=envelope_dict,
            )

        accepted = [i for i in interactions if i.accepted]
        toxicity = self.metrics_calculator.toxicity_rate(interactions)
        quality_gap = self.metrics_calculator.quality_gap(interactions)
        welfare = self.metrics_calculator.welfare_metrics(interactions)
        rho = self.metrics_calculator.quality_correlation(interactions)
        decomp = self.metrics_calculator.toxicity_decomposition(interactions)
        saturation = self.metrics_calculator.selection_saturation(interactions)

        return EpochMetrics(
            epoch=self.state.current_epoch,
            total_interactions=len(interactions),
            accepted_interactions=len(accepted),
            total_posts=len(self.feed._posts),
            total_votes=len(self.feed._votes),
            toxicity_rate=toxicity,
            quality_gap=quality_gap,
            quality_correlation=rho,
            baseline_harm=decomp["baseline_harm"],
            selection_credit=decomp["selection_credit"],
            selection_saturation=saturation,
            avg_payoff=welfare.get("avg_initiator_payoff", 0),
            total_welfare=welfare.get("total_welfare", 0),
            net_social_welfare=welfare.get("net_social_welfare", 0),
            network_metrics=network_metrics,
            capability_metrics=capability_metrics,
            spawn_metrics=spawn_metrics_dict,
            security_report=security_report,
            collusion_report=collusion_report,
            contract_metrics=contract_metrics_dict,
            capability_envelope_metrics=envelope_dict,
        )

    # ===================================================================
    # Events
    # ===================================================================

    def _emit_event(self, event: Event) -> None:
        """Emit an event via the event bus."""
        self._event_bus.emit(event)

    def subscribe_events(self, callback: Callable[[Event], None]) -> None:
        """Register an external subscriber for all simulation events."""
        self._event_bus.subscribe(callback)

    # ===================================================================
    # Public API (preserved for backwards compatibility)
    # ===================================================================

    def pause(self) -> None:
        """Pause the simulation."""
        self.state.pause()

    def resume(self) -> None:
        """Resume the simulation."""
        self.state.resume()

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())

    def get_metrics_history(self) -> List[EpochMetrics]:
        """Get all epoch metrics."""
        return self._epoch_metrics

    def get_collusion_report(self):
        """Get the latest collusion detection report."""
        if self.governance_engine is None:
            return None
        return self.governance_engine.get_collusion_report()

    def settle_marketplace_task(
        self,
        task_id: str,
        success: bool,
        quality_score: float = 1.0,
    ) -> Optional[Dict]:
        """Settle a marketplace bounty/escrow after task completion."""
        if self._marketplace_handler is None:
            return None
        return self._marketplace_handler.settle_task(
            task_id=task_id,
            success=success,
            state=self.state,
            governance_engine=self.governance_engine,
            quality_score=quality_score,
        )

    def _apply_governance_effect(self, effect: GovernanceEffect) -> None:
        """Apply governance effects to state."""
        self._finalizer.apply_governance_effect(effect)

    def _update_reputation(self, agent_id: str, delta: float) -> None:
        """Update agent reputation."""
        self._finalizer._update_reputation(agent_id, delta)

    def on_epoch_end(self, callback: Callable[[EpochMetrics], None]) -> None:
        """Register a callback for epoch end."""
        self._on_epoch_end.append(callback)

    def on_interaction_complete(
        self,
        callback: Callable[[SoftInteraction, float, float], None],
    ) -> None:
        """Register a callback for interaction completion."""
        self._on_interaction_complete.append(callback)

    def get_network_metrics(self) -> Optional[Dict[str, float]]:
        """Get current network topology metrics."""
        if self.network is None:
            return None
        return self.network.get_metrics()

    def get_network(self) -> Optional[AgentNetwork]:
        """Get the network object for direct manipulation."""
        return self.network

    def get_spawn_tree(self) -> Optional[SpawnTree]:
        """Get the spawn tree for inspection."""
        return self._spawn_tree

    @property
    def adaptive_controller(self):
        """Get the adaptive governance controller."""
        return self._adaptive_controller

    # ===================================================================
    # Composite Task Support
    # ===================================================================

    def add_composite_task(self, task: CompositeTask) -> bool:
        """Add a composite task to the pool."""
        if self.composite_task_pool is None:
            return False
        self.composite_task_pool.add_task(task)
        return True

    def get_composite_task(self, task_id: str) -> Optional[CompositeTask]:
        """Get a composite task by ID."""
        if self.composite_task_pool is None:
            return None
        return self.composite_task_pool.get_task(task_id)

    def get_open_composite_tasks(self) -> List[CompositeTask]:
        """Get all open composite tasks."""
        if self.composite_task_pool is None:
            return []
        return self.composite_task_pool.get_open_tasks()

    def register_agent_capabilities(
        self,
        agent_id: str,
        capabilities: set,
    ) -> bool:
        """Register an agent's capabilities for composite task matching."""
        if self.capability_analyzer is None:
            return False
        self.capability_analyzer.register_agent(agent_id, capabilities)
        return True

    def get_capability_metrics(self) -> Optional[EmergentCapabilityMetrics]:
        """Get current emergent capability metrics."""
        if self.capability_analyzer is None:
            return None
        return self.capability_analyzer.compute_metrics()

    def get_composite_task_stats(self) -> Dict:
        """Get statistics about composite tasks."""
        if self.composite_task_pool is None:
            return {}
        return self.composite_task_pool.get_stats()

    # ===================================================================
    # Async Support for LLM Agents
    # ===================================================================

    def _is_llm_agent(self, agent: BaseAgent) -> bool:
        """Check if an agent is an LLM agent with async support."""
        return hasattr(agent, "act_async") and hasattr(
            agent, "accept_interaction_async"
        )

    async def run_async(self) -> List[EpochMetrics]:
        """Run the full simulation asynchronously."""
        self._initialize_network()

        self._emit_event(
            Event(
                event_type=EventType.SIMULATION_STARTED,
                payload={
                    "n_epochs": self.config.n_epochs,
                    "steps_per_epoch": self.config.steps_per_epoch,
                    "n_agents": len(self._agents),
                    "seed": self.config.seed,
                    "async": True,
                },
            )
        )

        for _epoch in range(self.config.n_epochs):
            epoch_metrics = await self._run_epoch_async()
            self._epoch_metrics.append(epoch_metrics)
            for callback in self._on_epoch_end:
                callback(epoch_metrics)

        self._emit_event(
            Event(
                event_type=EventType.SIMULATION_ENDED,
                payload={
                    "total_epochs": self.config.n_epochs,
                    "final_metrics": self._epoch_metrics[-1].to_dict()
                    if self._epoch_metrics
                    else {},
                },
            )
        )

        return self._epoch_metrics

    async def _run_epoch_async(self) -> EpochMetrics:
        """Run a single epoch asynchronously."""
        epoch_start = self.state.current_epoch
        ctx = self._make_context()

        self._pipeline.on_epoch_start(ctx)

        for _step in range(self.config.steps_per_epoch):
            if self.state.is_paused:
                break
            await self._run_step_async(ctx)
            self.state.advance_step()

        self._pipeline.on_epoch_end(ctx)

        if self._contract_mw is not None:
            self._last_contract_metrics = self._contract_mw.last_metrics

        metrics = self._compute_epoch_metrics()

        self._emit_event(
            Event(
                event_type=EventType.EPOCH_COMPLETED,
                payload=metrics.to_dict(),
                epoch=epoch_start,
            )
        )

        self.state.advance_epoch()
        return metrics

    async def _run_step_async(self, ctx: MiddlewareContext) -> None:
        """Run a single step asynchronously with concurrent LLM calls."""
        self._pipeline.on_step_start(ctx)

        if self._external_action_queue is not None:
            self._external_action_queue.reset_step()

        dropped = (
            self._perturbation_engine.get_dropped_agents()
            if self._perturbation_engine is not None
            else set()
        )

        agents_to_act = self._scheduler.get_eligible(
            self._agents,
            self.state,
            governance_engine=self.governance_engine,
            dropped_agents=dropped,
        )

        async def get_agent_action(agent_id: str) -> Tuple[str, Action]:
            agent = self._agents[agent_id]
            observation = self._obs_builder.build(agent_id)

            if agent.is_external and self._external_action_queue is not None:
                import dataclasses as _dc
                self._external_observations[agent_id] = _dc.asdict(observation)

                raw_action = await self._external_action_queue.wait_for_action(
                    agent_id
                )
                if raw_action is None:
                    return agent_id, Action(
                        agent_id=agent_id, action_type=ActionType.NOOP
                    )
                return agent_id, self._parse_external_action(agent_id, raw_action)

            action = await self._select_action_async(agent, observation)
            return agent_id, action

        tasks = [get_agent_action(agent_id) for agent_id in agents_to_act]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            agent_id, action = result  # type: ignore[misc]
            self._execute_action(action)

        await self._resolve_pending_interactions_async()

    # ===================================================================
    # External agent support
    # ===================================================================

    def set_external_action_queue(self, queue: Any) -> None:
        """Attach an external action queue for API-driven agents."""
        self._external_action_queue = queue

    def get_external_observations(self) -> Dict[str, Dict[str, Any]]:
        """Return the current external observation store."""
        return self._external_observations

    _API_ACTION_MAP: Dict[str, str] = {
        "accept": "accept_interaction",
        "reject": "reject_interaction",
        "propose": "propose_interaction",
        "counter": "propose_interaction",
    }

    def _parse_external_action(self, agent_id: str, raw: Dict) -> Action:
        """Convert a raw dict from the action queue into an ``Action``."""
        action_type_str = raw.get("action_type", "noop")
        action_type_str = self._API_ACTION_MAP.get(action_type_str, action_type_str)
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.NOOP

        def _safe_str(val: Any, max_len: int = 256) -> str:
            if val is None:
                return ""
            if not isinstance(val, (str, int, float, bool)):
                return ""
            s = str(val)
            return s[:max_len]

        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return Action(
            agent_id=agent_id,
            action_type=action_type,
            target_id=_safe_str(raw.get("target_id")),
            counterparty_id=_safe_str(raw.get("counterparty_id")),
            content=_safe_str(raw.get("content"), max_len=4096),
            metadata=metadata,
        )

    async def _resolve_pending_interactions_async(self) -> None:
        """Resolve pending interactions with async support for LLM agents."""
        proposals = list(self.state.pending_proposals.values())

        async def resolve_proposal(proposal: InteractionProposal) -> Optional[bool]:
            counterparty_id = proposal.counterparty_id

            if counterparty_id not in self._agents:
                return None
            if not self.state.can_agent_act(counterparty_id):
                return None

            counterparty = self._agents[counterparty_id]
            observation = self._obs_builder.build(counterparty_id)

            from swarm.agents.base import InteractionProposal as AgentProposal

            agent_proposal = AgentProposal(
                proposal_id=proposal.proposal_id,
                initiator_id=proposal.initiator_id,
                counterparty_id=proposal.counterparty_id,
                interaction_type=InteractionType(proposal.interaction_type),
                content=proposal.content,
                offered_transfer=proposal.metadata.get("offered_transfer", 0),
            )

            if self._is_llm_agent(counterparty):
                accept = await counterparty.accept_interaction_async(
                    agent_proposal, observation
                )
            else:
                accept = counterparty.accept_interaction(agent_proposal, observation)

            return bool(accept)

        tasks = [resolve_proposal(p) for p in proposals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for proposal, result in zip(proposals, results, strict=False):
            if isinstance(result, Exception) or result is None:
                continue
            accept = bool(result)
            self.state.remove_proposal(proposal.proposal_id)
            self._complete_interaction(proposal, accepted=accept)

    # ===================================================================
    # Red-Team Support
    # ===================================================================

    def get_adaptive_adversary_reports(self) -> Dict[str, Dict]:
        """Get strategy reports from all adaptive adversaries."""
        return self._redteam.get_adaptive_adversary_reports()

    def notify_adversary_detection(
        self,
        agent_id: str,
        penalty: float = 0.0,
        detected: bool = True,
    ) -> None:
        """Notify an adaptive adversary of detection/penalty."""
        self._redteam.notify_adversary_detection(agent_id, penalty, detected)

    def get_evasion_metrics(self) -> Dict:
        """Get evasion metrics for adversarial agents."""
        return self._redteam.get_evasion_metrics()

    # ===================================================================
    # Boundary Delegation
    # ===================================================================

    def request_external_interaction(
        self,
        agent_id: str,
        entity_id: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request an interaction with an external entity."""
        if self._boundary_handler is None:
            return {"success": False, "error": "Boundaries not enabled"}
        return self._boundary_handler.request_external_interaction(
            agent_id=agent_id,
            entity_id=entity_id,
            action=action,
            payload=payload,
        )

    def get_external_entities(
        self,
        entity_type: Optional[str] = None,
        min_trust: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get available external entities."""
        if self._boundary_handler is None:
            return []
        return self._boundary_handler.get_external_entities(
            entity_type=entity_type,
            min_trust=min_trust,
        )

    def add_external_entity(self, entity: ExternalEntity) -> None:
        """Add an external entity to the world."""
        if self._boundary_handler is not None:
            self._boundary_handler.add_external_entity(entity)

    def get_boundary_metrics(self) -> Dict[str, Any]:
        """Get comprehensive boundary metrics."""
        if self._boundary_handler is None:
            return {"boundaries_enabled": False}
        return self._boundary_handler.get_metrics()

    def get_agent_boundary_activity(self, agent_id: str) -> Dict[str, Any]:
        """Get boundary activity for a specific agent."""
        if self._boundary_handler is None:
            return {"agent_id": agent_id}
        return self._boundary_handler.get_agent_activity(agent_id)

    def get_leakage_report(self) -> Optional[LeakageReport]:
        """Get the full leakage detection report."""
        if self._boundary_handler is None:
            return None
        return self._boundary_handler.get_leakage_report()

    # ===================================================================
    # LLM Usage
    # ===================================================================

    def get_llm_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get LLM usage statistics for all LLM agents."""
        stats = {}
        for agent_id, agent in self._agents.items():
            if hasattr(agent, "get_usage_stats"):
                stats[agent_id] = agent.get_usage_stats()
        return stats

    # ===================================================================
    # Legacy compatibility shims
    # ===================================================================

    def _build_observation(self, agent_id: str) -> Observation:
        """Build observation for an agent."""
        return self._obs_builder.build(agent_id)

    def _apply_observation_noise(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configurable gaussian noise to numeric observation fields."""
        return self._obs_builder.apply_noise(record)

    def _get_eligible_agents(self) -> List[str]:
        """Return agents eligible to act this step."""
        dropped = (
            self._perturbation_engine.get_dropped_agents()
            if self._perturbation_engine is not None
            else set()
        )
        return self._scheduler.get_eligible(
            self._agents,
            self.state,
            governance_engine=self.governance_engine,
            dropped_agents=dropped,
        )

    def _get_agent_schedule(self) -> List[str]:
        """Get the order of agents for this step."""
        return self._scheduler._get_order(self._agents, self.state)

    def _update_adaptive_governance(self, include_behavioral: bool = False) -> None:
        """Update adaptive governance mode."""
        if self._gov_mw is not None:
            ctx = self._make_context()
            self._gov_mw._update_adaptive(ctx, include_behavioral=include_behavioral)

    def _apply_agent_memory_decay(self, epoch: int) -> None:
        """Apply memory decay to all agents."""
        for agent in self._agents.values():
            if hasattr(agent, "apply_memory_decay"):
                agent.apply_memory_decay(epoch)

    def _epoch_pre_hooks(self) -> None:
        """Shared epoch-start logic."""
        ctx = self._make_context()
        self._pipeline.on_epoch_start(ctx)

    def _epoch_post_hooks(self, epoch_start: int) -> EpochMetrics:
        """Shared epoch-end logic."""
        ctx = self._make_context()
        self._pipeline.on_epoch_end(ctx)

        if self._contract_mw is not None:
            self._last_contract_metrics = self._contract_mw.last_metrics

        metrics = self._compute_epoch_metrics()

        self._emit_event(
            Event(
                event_type=EventType.EPOCH_COMPLETED,
                payload=metrics.to_dict(),
                epoch=epoch_start,
            )
        )

        self.state.advance_epoch()
        return metrics

    def _step_preamble(self) -> None:
        """Shared step-start logic."""
        ctx = self._make_context()
        self._pipeline.on_step_start(ctx)
