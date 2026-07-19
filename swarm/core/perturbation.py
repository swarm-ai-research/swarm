"""PerturbationEngine for chaos/stress-testing scenarios.

Provides structured, temporal perturbations — scheduled events like agent dropout,
signal corruption, governance parameter shocks, network partitions, and resource
shocks. Fills the gap between the Adversary Designer (strategic attacks) and
random noise, enabling resilience testing.
"""

import random
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CorruptionMode(str, Enum):
    """Signal corruption modes."""

    ZERO_OUT = "zero_out"
    INVERT = "invert"
    RANDOM = "random"
    STICKY = "sticky"


class ShockTrigger(str, Enum):
    """When a shock fires."""

    EPOCH = "epoch"
    CONDITION = "condition"


class PartitionMode(str, Enum):
    """Network partition strategies."""

    BISECT = "bisect"
    ISOLATE_TYPE = "isolate_type"
    RANDOM_FRAGMENT = "random_fragment"


class ResourceShockMode(str, Enum):
    """Resource shock strategies."""

    DRAIN_ALL = "drain_all"
    REDISTRIBUTE = "redistribute"
    INFLATE = "inflate"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class ScheduleSpec(BaseModel):
    """When a perturbation activates."""

    start_epoch: int = 0
    end_epoch: Optional[int] = None  # None = until simulation ends


class AgentDropoutConfig(BaseModel):
    """Config for random agent dropout each step."""

    enabled: bool = False
    probability_per_step: float = 0.1
    min_duration_steps: int = 1
    max_duration_steps: int = 3
    exempt_types: List[str] = Field(default_factory=list)


class SignalCorruptionConfig(BaseModel):
    """Config for corrupting proxy observable signals."""

    enabled: bool = False
    targets: List[str] = Field(
        default_factory=lambda: ["task_progress_delta"]
    )
    mode: CorruptionMode = CorruptionMode.ZERO_OUT
    schedule: List[ScheduleSpec] = Field(default_factory=list)


class ParameterShockSpec(BaseModel):
    """A single parameter shock event."""

    trigger: ShockTrigger = ShockTrigger.EPOCH
    at_epoch: Optional[int] = None
    when: Optional[str] = None  # condition string, e.g. "toxicity > 0.5"
    params: Dict[str, Any] = Field(default_factory=dict)
    revert_after_epochs: Optional[int] = None


class ParameterShocksConfig(BaseModel):
    """Config for governance/payoff parameter shocks."""

    enabled: bool = False
    shocks: List[ParameterShockSpec] = Field(default_factory=list)


class NetworkPartitionConfig(BaseModel):
    """Config for network partition events."""

    enabled: bool = False
    trigger: ShockTrigger = ShockTrigger.EPOCH
    at_epoch: Optional[int] = None
    mode: PartitionMode = PartitionMode.BISECT
    isolate_type: Optional[str] = None  # for ISOLATE_TYPE mode
    heal_after_epochs: Optional[int] = None


class ResourceShockConfig(BaseModel):
    """Config for resource shock events."""

    enabled: bool = False
    trigger: ShockTrigger = ShockTrigger.EPOCH
    at_epoch: Optional[int] = None
    mode: ResourceShockMode = ResourceShockMode.DRAIN_ALL
    magnitude: float = 0.5  # fraction: drain 50%, inflate 50%, etc.


class PerturbationConfig(BaseModel):
    """Top-level perturbation configuration."""

    seed: Optional[int] = None
    agent_dropout: AgentDropoutConfig = Field(
        default_factory=AgentDropoutConfig
    )
    signal_corruption: SignalCorruptionConfig = Field(
        default_factory=SignalCorruptionConfig
    )
    parameter_shocks: ParameterShocksConfig = Field(
        default_factory=ParameterShocksConfig
    )
    network_partition: NetworkPartitionConfig = Field(
        default_factory=NetworkPartitionConfig
    )
    resource_shock: ResourceShockConfig = Field(
        default_factory=ResourceShockConfig
    )


# ---------------------------------------------------------------------------
# Saved state for revert
# ---------------------------------------------------------------------------


class _SavedParamShock(BaseModel):
    """Bookkeeping for a parameter shock that may need reverting."""

    spec: ParameterShockSpec
    activated_epoch: int
    original_values: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# PerturbationEngine
# ---------------------------------------------------------------------------


class PerturbationEngine:
    """Manages structured perturbations during a simulation run."""

    def __init__(
        self,
        config: PerturbationConfig,
        state: Any = None,
        network: Any = None,
        governance_engine: Any = None,
    ):
        self.config = config
        self._state = state
        self._network = network
        self._governance_engine = governance_engine
        self._rng = random.Random(config.seed)

        # Agent dropout state: agent_id -> steps remaining
        self._active_dropouts: Dict[str, int] = {}

        # Parameter shock revert tracking
        self._saved_shocks: List[_SavedParamShock] = []

        # Network partition saved edges
        self._saved_edges: Dict[str, Dict[str, float]] = {}
        self._partition_active: bool = False
        self._partition_activated_epoch: Optional[int] = None

        # Signal corruption sticky values
        self._sticky_values: Dict[str, float] = {}

        # Track which epoch-triggered shocks already fired
        self._fired_epoch_shocks: Set[int] = set()
        self._resource_shock_fired: bool = False
        self._partition_fired: bool = False

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch: int) -> None:
        """Check epoch-triggered shocks, manage reverts."""
        self._revert_expired_shocks(epoch)
        self._check_epoch_parameter_shocks(epoch)
        self._check_epoch_network_partition(epoch)
        self._check_epoch_resource_shock(epoch)
        self._heal_network_if_due(epoch)

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def on_step_start(self, epoch: int, step: int) -> None:
        """Apply per-step perturbations (dropout rolls, etc.)."""
        self._tick_dropouts()
        if self.config.agent_dropout.enabled:
            self._roll_dropouts(epoch, step)

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def get_dropped_agents(self) -> Set[str]:
        """Return currently-dropped agent IDs."""
        return set(self._active_dropouts.keys())

    def perturb_observables(self, obs: Any) -> Any:
        """Apply signal corruption if active. Returns (possibly modified) obs."""
        from swarm.core.proxy import ProxyObservables

        cfg = self.config.signal_corruption
        if not cfg.enabled:
            return obs

        # Make a mutable copy
        data = obs.model_dump()
        for field_name in cfg.targets:
            if field_name not in data:
                continue
            data[field_name] = self._corrupt_value(
                field_name, data[field_name], cfg.mode
            )
        return ProxyObservables(**data)

    def check_condition(self, metrics: dict) -> None:
        """Evaluate condition-triggered shocks against epoch metrics."""
        if not self.config.parameter_shocks.enabled:
            return
        for i, spec in enumerate(self.config.parameter_shocks.shocks):
            if spec.trigger != ShockTrigger.CONDITION:
                continue
            if i in self._fired_epoch_shocks:
                continue
            if spec.when and self._evaluate_condition(spec.when, metrics):
                self._apply_parameter_shock(spec, metrics.get("epoch", 0))
                self._fired_epoch_shocks.add(i)

    def is_signal_corruption_active(self, epoch: int) -> bool:
        """Check whether signal corruption is active for the given epoch."""
        cfg = self.config.signal_corruption
        if not cfg.enabled:
            return False
        if not cfg.schedule:
            return True  # always active if no schedule specified
        for sched in cfg.schedule:
            if epoch >= sched.start_epoch:
                if sched.end_epoch is None or epoch <= sched.end_epoch:
                    return True
        return False

    # ------------------------------------------------------------------
    # Agent dropout internals
    # ------------------------------------------------------------------

    def _roll_dropouts(self, epoch: int, step: int) -> None:
        """Roll for new agent dropouts this step."""
        if self._state is None:
            return
        cfg = self.config.agent_dropout
        exempt = set(cfg.exempt_types)
        for agent_id, agent_state in self._state.agents.items():
            if agent_id in self._active_dropouts:
                continue
            if agent_state.agent_type.value in exempt:
                continue
            if self._rng.random() < cfg.probability_per_step:
                duration = self._rng.randint(
                    cfg.min_duration_steps, cfg.max_duration_steps
                )
                self._active_dropouts[agent_id] = duration

    def _tick_dropouts(self) -> None:
        """Decrement dropout timers, removing expired ones."""
        expired = []
        for agent_id, remaining in self._active_dropouts.items():
            if remaining <= 1:
                expired.append(agent_id)
            else:
                self._active_dropouts[agent_id] = remaining - 1
        for agent_id in expired:
            del self._active_dropouts[agent_id]

    # ------------------------------------------------------------------
    # Signal corruption internals
    # ------------------------------------------------------------------

    def _corrupt_value(
        self, field_name: str, value: Any, mode: CorruptionMode
    ) -> Any:
        """Corrupt a signal value according to the specified mode.

        Args:
            field_name: Name of the field being corrupted (used by STICKY mode)
            value: The value to corrupt
            mode: Corruption strategy:
                - ZERO_OUT: Replace with 0 (or 0.0)
                - INVERT: Negate the value
                - RANDOM: Replace with random value in [0, 5] for int or [-1.0, 1.0] for float
                - STICKY: Store first value and return it consistently for this field

        Returns:
            The corrupted value
        """
        if mode == CorruptionMode.ZERO_OUT:
            return 0 if isinstance(value, int) else 0.0
        elif mode == CorruptionMode.INVERT:
            return -value
        elif mode == CorruptionMode.RANDOM:
            if isinstance(value, int):
                return self._rng.randint(0, 5)
            return self._rng.uniform(-1.0, 1.0)
        elif mode == CorruptionMode.STICKY:
            if field_name not in self._sticky_values:
                self._sticky_values[field_name] = value
            return self._sticky_values[field_name]
        return value

    # ------------------------------------------------------------------
    # Parameter shock internals
    # ------------------------------------------------------------------

    def _check_epoch_parameter_shocks(self, epoch: int) -> None:
        if not self.config.parameter_shocks.enabled:
            return
        for i, spec in enumerate(self.config.parameter_shocks.shocks):
            if spec.trigger != ShockTrigger.EPOCH:
                continue
            if spec.at_epoch != epoch:
                continue
            if i in self._fired_epoch_shocks:
                continue
            self._apply_parameter_shock(spec, epoch)
            self._fired_epoch_shocks.add(i)

    def _apply_parameter_shock(
        self, spec: ParameterShockSpec, epoch: int
    ) -> None:
        """Apply parameter shock, saving originals for revert."""
        if self._governance_engine is None:
            return
        original_values: Dict[str, Any] = {}
        gov_config = self._governance_engine.config
        for dotted_path, new_value in spec.params.items():
            parts = dotted_path.split(".")
            obj = gov_config
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                attr = parts[-1]
                original_values[dotted_path] = getattr(obj, attr, None)
                setattr(obj, attr, new_value)

        if spec.revert_after_epochs is not None:
            self._saved_shocks.append(
                _SavedParamShock(
                    spec=spec,
                    activated_epoch=epoch,
                    original_values=original_values,
                )
            )

    def _revert_expired_shocks(self, epoch: int) -> None:
        still_active = []
        for saved in self._saved_shocks:
            revert_at = saved.activated_epoch + (
                saved.spec.revert_after_epochs or 0
            )
            if epoch >= revert_at:
                self._revert_shock(saved)
            else:
                still_active.append(saved)
        self._saved_shocks = still_active

    def _revert_shock(self, saved: _SavedParamShock) -> None:
        if self._governance_engine is None:
            return
        gov_config = self._governance_engine.config
        for dotted_path, orig_value in saved.original_values.items():
            parts = dotted_path.split(".")
            obj = gov_config
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                setattr(obj, parts[-1], orig_value)

    # ------------------------------------------------------------------
    # Network partition internals
    # ------------------------------------------------------------------

    def _check_epoch_network_partition(self, epoch: int) -> None:
        cfg = self.config.network_partition
        if not cfg.enabled or self._partition_fired:
            return
        if cfg.trigger != ShockTrigger.EPOCH:
            return
        if cfg.at_epoch != epoch:
            return
        self._apply_network_partition()
        self._partition_fired = True

    def _apply_network_partition(self) -> None:
        if self._network is None or self._state is None:
            return
        cfg = self.config.network_partition
        agent_ids = list(self._state.agents.keys())

        # Save current edges (with weights for restore)
        self._saved_edges = {}
        for a_id in agent_ids:
            nbrs = self._network.neighbors(a_id)
            if nbrs:
                self._saved_edges[a_id] = {
                    b_id: self._network.edge_weight(a_id, b_id)
                    for b_id in nbrs
                }

        if cfg.mode == PartitionMode.BISECT:
            self._rng.shuffle(agent_ids)
            mid = len(agent_ids) // 2
            group_a = set(agent_ids[:mid])
            group_b = set(agent_ids[mid:])
            # Remove cross-group edges
            for a_id in group_a:
                for b_id in list(self._network.neighbors(a_id)):
                    if b_id in group_b:
                        self._network.remove_edge(a_id, b_id)

        elif cfg.mode == PartitionMode.ISOLATE_TYPE:
            if cfg.isolate_type is None:
                return
            isolated = {
                a_id
                for a_id, s in self._state.agents.items()
                if s.agent_type.value == cfg.isolate_type
            }
            for a_id in isolated:
                for b_id in list(self._network.neighbors(a_id)):
                    if b_id not in isolated:
                        self._network.remove_edge(a_id, b_id)

        elif cfg.mode == PartitionMode.RANDOM_FRAGMENT:
            # Remove ~50% of edges randomly
            edges_to_remove = []
            seen = set()
            for a_id in agent_ids:
                for b_id in self._network.neighbors(a_id):
                    edge = tuple(sorted([a_id, b_id]))
                    if edge not in seen:
                        seen.add(edge)
                        if self._rng.random() < 0.5:
                            edges_to_remove.append((a_id, b_id))
            for a_id, b_id in edges_to_remove:
                self._network.remove_edge(a_id, b_id)

        self._partition_active = True
        self._partition_activated_epoch = (
            self.config.network_partition.at_epoch
        )

    def _heal_network_if_due(self, epoch: int) -> None:
        if not self._partition_active:
            return
        cfg = self.config.network_partition
        if cfg.heal_after_epochs is None:
            return
        if self._partition_activated_epoch is None:
            return
        if epoch >= self._partition_activated_epoch + cfg.heal_after_epochs:
            self._heal_network()

    def _heal_network(self) -> None:
        if self._network is None:
            return
        for a_id, neighbors in self._saved_edges.items():
            for b_id, weight in neighbors.items():
                self._network.add_edge(a_id, b_id, weight)
        self._saved_edges = {}
        self._partition_active = False

    # ------------------------------------------------------------------
    # Resource shock internals
    # ------------------------------------------------------------------

    def _check_epoch_resource_shock(self, epoch: int) -> None:
        cfg = self.config.resource_shock
        if not cfg.enabled or self._resource_shock_fired:
            return
        if cfg.trigger != ShockTrigger.EPOCH:
            return
        if cfg.at_epoch != epoch:
            return
        self._apply_resource_shock()
        self._resource_shock_fired = True

    def _apply_resource_shock(self) -> None:
        if self._state is None:
            return
        cfg = self.config.resource_shock

        if cfg.mode == ResourceShockMode.DRAIN_ALL:
            for agent_state in self._state.agents.values():
                drain = agent_state.resources * cfg.magnitude
                agent_state.update_resources(-drain)

        elif cfg.mode == ResourceShockMode.REDISTRIBUTE:
            total = sum(s.resources for s in self._state.agents.values())
            n = len(self._state.agents)
            if n == 0:
                return
            equal_share = total / n
            for agent_state in self._state.agents.values():
                diff = equal_share - agent_state.resources
                agent_state.update_resources(diff * cfg.magnitude)

        elif cfg.mode == ResourceShockMode.INFLATE:
            for agent_state in self._state.agents.values():
                bonus = agent_state.resources * cfg.magnitude
                agent_state.update_resources(bonus)

    # ------------------------------------------------------------------
    # Condition evaluation
    # ------------------------------------------------------------------

    def _evaluate_condition(self, condition: str, metrics: dict) -> bool:
        """Evaluate a simple condition like 'toxicity_rate > 0.5'."""
        # Parse "field op value"
        parts = condition.strip().split()
        if len(parts) != 3:
            return False
        field_name, op, threshold_str = parts
        try:
            threshold = float(threshold_str)
        except ValueError:
            return False
        actual = metrics.get(field_name)
        if actual is None:
            return False
        try:
            actual = float(actual)
        except (ValueError, TypeError):
            return False

        if op == ">":
            return bool(actual > threshold)
        elif op == ">=":
            return bool(actual >= threshold)
        elif op == "<":
            return bool(actual < threshold)
        elif op == "<=":
            return bool(actual <= threshold)
        elif op == "==":
            return bool(actual == threshold)
        elif op == "!=":
            return bool(actual != threshold)
        return False
