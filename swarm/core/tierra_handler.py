"""Tierra lifecycle handler — metabolism, death, reaping, and metrics.

This handler manages the finite resource pool and agent lifecycle for
the Tierra scenario.  It does not own any ``ActionType`` values; it
operates purely through step/epoch hooks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet

from swarm.core.handler import Handler, HandlerActionResult
from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class TierraConfig:
    """Configuration for the Tierra lifecycle handler."""

    enabled: bool = True
    total_resource_pool: float = 10000.0
    resource_replenishment_rate: float = 50.0
    base_metabolism_cost: float = 2.0
    population_cap: int = 100
    reaper_mode: str = "lowest_fitness"  # "lowest_fitness" | "oldest" | "random" | "diversity_preserving"
    reaper_diversity_min: int = 1  # min agents per species to protect (diversity_preserving mode)
    mutation_std: float = 0.05
    death_threshold: float = 0.0
    pool_distribution_rate: float = 0.1  # fraction of pool distributed per step
    max_efficiency_weight: float = 0.0  # cap on relative efficiency weight (0 = uncapped)


class TierraHandler(Handler):
    """Manage Tierra agent lifecycle: metabolism, death, reaping, metrics."""

    def __init__(
        self,
        *,
        config: TierraConfig,
        event_bus: EventBus,
        rng: Any = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.config = config
        self._rng = rng
        self._resource_pool: float = config.total_resource_pool
        self._genome_registry: Dict[str, Dict[str, float]] = {}
        # Registration order doubles as spawn order: the loader registers
        # the initial population first, then the orchestrator registers each
        # child at spawn time. Used by the "oldest" reaper mode.
        self._spawn_order: Dict[str, int] = {}
        self._spawn_counter: int = 0
        # Per-epoch counters
        self._births: int = 0
        self._deaths: int = 0

    @staticmethod
    def handled_action_types() -> FrozenSet:
        """TierraHandler owns no action types — lifecycle only."""
        return frozenset()

    def handle_action(self, action: Any, state: Any) -> HandlerActionResult:
        return HandlerActionResult(success=False)

    # ------------------------------------------------------------------
    # Genome registry
    # ------------------------------------------------------------------

    def register_genome(self, agent_id: str, genome_dict: Dict[str, float]) -> None:
        self._genome_registry[agent_id] = genome_dict
        if agent_id not in self._spawn_order:
            self._spawn_order[agent_id] = self._spawn_counter
            self._spawn_counter += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_alive(self, agent_id: str, state: Any) -> bool:
        """Check if agent is alive (not frozen)."""
        return not state.is_agent_frozen(agent_id)

    def _living_tierra_agents(self, state: Any) -> list:
        """Return list of living Tierra agent IDs."""
        return [
            aid for aid, ast in state.agents.items()
            if ast.agent_type.value == "tierra"
            and self._is_alive(aid, state)
        ]

    # ------------------------------------------------------------------
    # Step hook: metabolism + replenishment + death check
    # ------------------------------------------------------------------

    def on_step(self, state: Any, step: int) -> None:
        """Distribute pool resources, apply metabolism, check deaths."""
        # Replenish resource pool
        self._resource_pool += self.config.resource_replenishment_rate

        # Distribute a fraction of the pool to living agents (weighted by efficiency)
        living = self._living_tierra_agents(state)
        if living and self._resource_pool > 0:
            distribute = self._resource_pool * self.config.pool_distribution_rate
            # Weight by efficiency gene
            weights = []
            for aid in living:
                gd = self._genome_registry.get(aid)
                eff = gd.get("efficiency", 0.5) if gd else 0.5
                weights.append(max(eff, 0.01))  # floor to avoid zero
            # Cap relative efficiency weight to prevent runaway dominance
            if self.config.max_efficiency_weight > 0 and len(weights) > 1:
                mean_w = sum(weights) / len(weights)
                cap = mean_w * self.config.max_efficiency_weight
                weights = [min(w, cap) for w in weights]
            total_weight = sum(weights)
            for aid, w in zip(living, weights, strict=True):
                share = distribute * (w / total_weight)
                state.agents[aid].update_resources(share)
            self._resource_pool -= distribute

        # Apply metabolism to each living Tierra agent
        for agent_id in self._living_tierra_agents(state):
            agent_state = state.agents[agent_id]

            # Calculate metabolism cost
            genome_dict = self._genome_registry.get(agent_id)
            if genome_dict is not None:
                from swarm.agents.tierra_agent import TierraGenome
                genome = TierraGenome.from_dict(genome_dict)
                cost = genome.metabolism_rate * self.config.base_metabolism_cost * (
                    1.0 + 0.1 * genome.complexity()
                )
            else:
                cost = self.config.base_metabolism_cost * 0.5

            # Deduct from agent, return to pool
            actual_cost = min(cost, agent_state.resources)
            agent_state.update_resources(-actual_cost)
            self._resource_pool += actual_cost

            # Death check
            if agent_state.resources <= self.config.death_threshold:
                self._kill_agent(agent_id, state, reason="starvation")

        # Reaper: enforce population cap
        self._run_reaper(state)

    # ------------------------------------------------------------------
    # Death / reaper
    # ------------------------------------------------------------------

    def _kill_agent(self, agent_id: str, state: Any, reason: str = "reaped") -> None:
        """Freeze agent and return remaining resources to pool."""
        agent_state = state.agents.get(agent_id)
        if agent_state is None:
            return
        remaining = agent_state.resources
        agent_state.resources = 0.0
        self._resource_pool += remaining

        # Freeze the agent (prevents further actions)
        state.freeze_agent(agent_id)

        self._deaths += 1
        self._emit_event(Event(
            event_type=EventType.AGENT_STATE_UPDATED,
            agent_id=agent_id,
            payload={"reason": reason, "resources_returned": remaining, "frozen": True},
        ))

    def _run_reaper(self, state: Any) -> None:
        """Kill agents if population exceeds cap."""
        living = self._living_tierra_agents(state)
        excess = len(living) - self.config.population_cap
        if excess <= 0:
            return

        if self.config.reaper_mode == "diversity_preserving":
            self._run_diversity_preserving_reaper(living, state, excess)
            return

        if self.config.reaper_mode == "oldest":
            # Earliest-registered first; unregistered agents (no genome on
            # record) sort oldest as a conservative default.
            living.sort(key=lambda aid: self._spawn_order.get(aid, -1))
        elif self.config.reaper_mode == "random" and self._rng is not None:
            self._rng.shuffle(living)
        else:
            # "lowest_fitness": sort by resources ascending
            living.sort(key=lambda aid: state.agents[aid].resources)

        for i in range(excess):
            self._kill_agent(living[i], state, reason="reaped")

    def _run_diversity_preserving_reaper(
        self, living: list, state: Any, excess: int
    ) -> None:
        """Reap excess agents while protecting species diversity.

        Guarantees at least ``reaper_diversity_min`` agents per species cluster.
        Within each cluster, the poorest agents are culled first.
        """
        from swarm.metrics.tierra_metrics import species_clusters

        genomes = [self._genome_registry.get(aid, {}) for aid in living]
        clusters = species_clusters(genomes, distance_threshold=0.5)

        # Build per-cluster lists sorted by resources ascending (poorest first)
        min_per_species = self.config.reaper_diversity_min
        killable: list[str] = []
        for _cid, indices in clusters.items():
            members = [(living[i], state.agents[living[i]].resources) for i in indices]
            members.sort(key=lambda x: x[1])
            # Protect at least min_per_species from this cluster
            protected = min(min_per_species, len(members))
            for aid, _res in members[:-protected] if len(members) > protected else []:
                killable.append(aid)

        # Sort killable by resources ascending so poorest die first globally
        killable.sort(key=lambda aid: state.agents[aid].resources)

        killed = 0
        for aid in killable:
            if killed >= excess:
                break
            self._kill_agent(aid, state, reason="reaped")
            killed += 1

        # If still over cap (all species are at minimum), fall back to lowest_fitness
        if killed < excess:
            remaining_living = self._living_tierra_agents(state)
            remaining_living.sort(key=lambda aid: state.agents[aid].resources)
            for aid in remaining_living:
                if killed >= excess:
                    break
                self._kill_agent(aid, state, reason="reaped")
                killed += 1

    # ------------------------------------------------------------------
    # Epoch hooks: metrics
    # ------------------------------------------------------------------

    def on_epoch_start(self, state: Any) -> None:
        self._births = 0
        self._deaths = 0

    def on_epoch_end(self, state: Any) -> None:
        """Compute and emit Tierra metrics."""
        living = self._living_tierra_agents(state)

        resources = [state.agents[aid].resources for aid in living]
        total_resources_held = sum(resources) if resources else 0.0

        # Genome means
        genome_means: Dict[str, float] = {}
        genomes = [self._genome_registry[aid] for aid in living if aid in self._genome_registry]
        if genomes:
            for key in genomes[0]:
                vals = [g[key] for g in genomes]
                genome_means[key] = sum(vals) / len(vals)

        payload = {
            "population": len(living),
            "resource_pool": self._resource_pool,
            "total_resources_held": total_resources_held,
            "births": self._births,
            "deaths": self._deaths,
            "genome_means": genome_means,
        }

        self._emit_event(Event(
            event_type=EventType.EPOCH_COMPLETED,
            payload={"tierra_metrics": payload},
        ))

    @property
    def resource_pool(self) -> float:
        return self._resource_pool

    @resource_pool.setter
    def resource_pool(self, value: float) -> None:
        self._resource_pool = value
