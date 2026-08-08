"""Memory tier handler for shared-memory multi-agent simulations."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel, model_validator

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler
from swarm.core.memory_observables import MemoryActionOutcome, MemoryObservableGenerator
from swarm.core.proxy import ProxyObservables
from swarm.env.memory_tiers import MemoryStore
from swarm.env.state import EnvState
from swarm.governance.engine import GovernanceEffect
from swarm.logging.event_bus import EventBus
from swarm.models.agent import AgentType
from swarm.models.events import Event, EventType
from swarm.models.interaction import SoftInteraction


class MemoryTierConfig(BaseModel):
    """Configuration for the memory tier handler."""

    enabled: bool = True
    initial_entries: int = 100
    hot_cache_size: int = 20
    compaction_probability: float = 0.05  # Per-agent per-step
    seed: Optional[int] = None

    # Memetic contagion: exposure to poisoned hot-cache entries drifts
    # honest/opportunistic agents toward writing poisoned entries themselves.
    # Infection is agent state, not store state: a memory reset wipes the
    # channel but not the infected agents, who re-poison the fresh store.
    contagion_enabled: bool = False
    contagion_exposure_alpha: float = 0.25  # EMA rate toward cache poisoned fraction
    contagion_transmissibility: float = 0.8  # P(poisoned write) = infection * this

    # Prevention lever: wipe and re-seed the store every N epochs (0 = never).
    reset_cadence_epochs: int = 0

    @model_validator(mode="after")
    def _run_validation(self) -> "MemoryTierConfig":
        if self.initial_entries < 0:
            raise ValueError("initial_entries must be non-negative")
        if self.hot_cache_size < 1:
            raise ValueError("hot_cache_size must be >= 1")
        if not 0.0 <= self.compaction_probability <= 1.0:
            raise ValueError("compaction_probability must be in [0, 1]")
        if not 0.0 <= self.contagion_exposure_alpha <= 1.0:
            raise ValueError("contagion_exposure_alpha must be in [0, 1]")
        if not 0.0 <= self.contagion_transmissibility <= 1.0:
            raise ValueError("contagion_transmissibility must be in [0, 1]")
        if self.reset_cadence_epochs < 0:
            raise ValueError("reset_cadence_epochs must be non-negative")
        return self


@dataclass
class MemoryActionResult:
    """Result of a memory action."""

    success: bool
    observables: Optional[ProxyObservables] = None
    initiator_id: str = ""
    counterparty_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True


class MemoryHandler(Handler):
    """Handles memory tier actions and lifecycle events."""

    @staticmethod
    def handled_action_types() -> frozenset:
        return frozenset({
            ActionType.WRITE_MEMORY,
            ActionType.PROMOTE_MEMORY,
            ActionType.VERIFY_MEMORY,
            ActionType.SEARCH_MEMORY,
            ActionType.CHALLENGE_MEMORY,
        })

    def __init__(
        self,
        config: MemoryTierConfig,
        *,
        event_bus: EventBus,
    ):
        super().__init__(event_bus=event_bus)
        self.config = config
        self._rng = random.Random(config.seed)
        self.store = MemoryStore(seed=config.seed)
        self.store._hot_cache_size = config.hot_cache_size
        self.observable_generator = MemoryObservableGenerator()

        # Per-agent infection level in [0, 1]; persists across store resets.
        self.infection: Dict[str, float] = {}
        # Per-epoch time series for analysis (epoch, infection, poisoning).
        self.epoch_snapshots: list = []
        # Lifetime count of contagion-poisoned writes (survives store resets).
        self.contagion_write_count: int = 0

        if config.initial_entries > 0:
            self.store.seed_entries(config.initial_entries)
            self.store.rebuild_hot_cache()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def build_observation_fields(
        self,
        agent_id: str,
        state: EnvState,
    ) -> Dict[str, Any]:
        """Build memory-related observation fields for an agent."""
        hot_cache = [e.to_dict() for e in self.store.hot_cache[:10]]
        pending = [e.to_dict() for e in self.store.get_pending_promotions()[:10]]
        challenged = [e.to_dict() for e in self.store.get_challenged_entries()[:10]]

        # Write rate remaining
        writes_used = self.store.writes_this_epoch.get(agent_id, 0)
        # Default cap is high if rate limit lever not enabled
        writes_remaining = max(0, 20 - writes_used)

        return {
            "memory_hot_cache": hot_cache,
            "memory_pending_promotions": pending,
            "memory_challenged_entries": challenged,
            "memory_entry_counts": self.store.entry_count(),
            "memory_writes_remaining": writes_remaining,
            "memory_search_results": [],  # Populated on SEARCH_MEMORY action
        }

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------

    def on_epoch_start(self, state: EnvState) -> None:
        """Epoch start: update contagion, maybe reset store, rebuild cache."""
        # Exposure uses the cache as agents saw it during the previous epoch,
        # so it must run before any reset or rebuild.
        if state.current_epoch > 0:
            self._update_contagion(state)

        reset_due = (
            self.config.reset_cadence_epochs > 0
            and state.current_epoch > 0
            and state.current_epoch % self.config.reset_cadence_epochs == 0
        )
        if reset_due:
            self._reset_store()

        self.store.on_epoch_start()
        self._record_snapshot(state, reset=reset_due)
        self._emit_event(
            Event(
                event_type=EventType.MEMORY_CACHE_REBUILT,
                payload={
                    "cache_size": len(self.store.hot_cache),
                    "memory_reset": reset_due,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

    def _update_contagion(self, state: EnvState) -> None:
        """Drift honest/opportunistic infection toward cache poisoned fraction."""
        if not self.config.contagion_enabled:
            return
        cache = self.store.hot_cache
        exposure = (
            sum(1 for e in cache if e.is_poisoned) / len(cache) if cache else 0.0
        )
        alpha = self.config.contagion_exposure_alpha
        for agent_id, agent_state in state.agents.items():
            if agent_state.agent_type not in (
                AgentType.HONEST,
                AgentType.OPPORTUNISTIC,
            ):
                continue
            prev = self.infection.get(agent_id, 0.0)
            self.infection[agent_id] = (1 - alpha) * prev + alpha * exposure

    def _reset_store(self) -> None:
        """Wipe and re-seed the store. Infection state deliberately survives."""
        self.store = MemoryStore(seed=self._rng.randrange(2**31))
        self.store._hot_cache_size = self.config.hot_cache_size
        if self.config.initial_entries > 0:
            self.store.seed_entries(self.config.initial_entries)
            self.store.rebuild_hot_cache()

    def _record_snapshot(self, state: EnvState, reset: bool) -> None:
        from swarm.metrics.memory_metrics import (
            cache_corruption,
            poisoning_rate,
            promotion_accuracy,
        )

        levels = list(self.infection.values())
        self.epoch_snapshots.append({
            "epoch": state.current_epoch,
            "reset": reset,
            "mean_infection": sum(levels) / len(levels) if levels else 0.0,
            "max_infection": max(levels) if levels else 0.0,
            "cache_corruption": cache_corruption(self.store),
            "tier3_poisoning": poisoning_rate(self.store),
            "promotion_accuracy": promotion_accuracy(self.store),
        })

    def maybe_compaction(self, agent_id: str, state: EnvState) -> int:
        """Randomly trigger compaction for an agent. Returns entries lost."""
        if self._rng.random() >= self.config.compaction_probability:
            return 0
        lost: int = self.store.simulate_compaction(agent_id)
        if lost > 0:
            self._emit_event(
                Event(
                    event_type=EventType.MEMORY_COMPACTION,
                    agent_id=agent_id,
                    payload={"entries_lost": lost},
                    epoch=state.current_epoch,
                    step=state.current_step,
                )
            )
        return lost

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def handle_action(self, action: Action, state: EnvState) -> MemoryActionResult:
        """Handle a memory tier action."""
        if action.action_type == ActionType.WRITE_MEMORY:
            return self._handle_write(action, state)
        if action.action_type == ActionType.PROMOTE_MEMORY:
            return self._handle_promote(action, state)
        if action.action_type == ActionType.VERIFY_MEMORY:
            return self._handle_verify(action, state)
        if action.action_type == ActionType.SEARCH_MEMORY:
            return self._handle_search(action, state)
        if action.action_type == ActionType.CHALLENGE_MEMORY:
            return self._handle_challenge(action, state)

        return MemoryActionResult(success=False)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_write(self, action: Action, state: EnvState) -> MemoryActionResult:
        agent_type = self._get_agent_type(action.agent_id, state)

        quality, is_poisoned = self._quality_for_agent(agent_type)

        # Memetic contagion: an infected honest/opportunistic agent reproduces
        # the meme with probability infection * transmissibility.
        contagion_poisoned = False
        if (
            self.config.contagion_enabled
            and not is_poisoned
            and agent_type in (AgentType.HONEST, AgentType.OPPORTUNISTIC)
        ):
            p_transmit = (
                self.infection.get(action.agent_id, 0.0)
                * self.config.contagion_transmissibility
            )
            if self._rng.random() < p_transmit:
                is_poisoned = True
                contagion_poisoned = True
                quality = self._rng.uniform(0.35, 0.6)
                self.contagion_write_count += 1

        entry = self.store.write(
            agent_id=action.agent_id,
            content=action.content,
            quality_score=quality,
            is_poisoned=is_poisoned,
            epoch=state.current_epoch,
            step=state.current_step,
        )

        outcome = self._write_outcome(quality, is_poisoned, agent_type)
        observables = self.observable_generator.generate(outcome)

        self._emit_event(
            Event(
                event_type=EventType.MEMORY_WRITTEN,
                agent_id=action.agent_id,
                payload={
                    "entry_id": entry.entry_id,
                    "tier": entry.tier.value,
                    "quality": quality,
                    "contagion_poisoned": contagion_poisoned,
                    "infection": self.infection.get(action.agent_id, 0.0),
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return MemoryActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id="memory_system",
            metadata={
                "memory_write": True,
                "entry_id": entry.entry_id,
                "quality_score": quality,
                "contagion_poisoned": contagion_poisoned,
            },
        )

    def _handle_promote(self, action: Action, state: EnvState) -> MemoryActionResult:
        entry = self.store.get_entry(action.target_id)
        if entry is None:
            return MemoryActionResult(success=False)

        # Build metadata for governance gate check
        meta = {
            "memory_promotion": True,
            "entry_id": entry.entry_id,
            "quality_score": entry.quality_score,
            "verified_by": list(entry.verified_by),
            "entry_author": entry.author_id,
            "source_tier": entry.tier.value,
        }

        promoted = self.store.promote(entry.entry_id, action.agent_id)
        if promoted is None:
            return MemoryActionResult(success=False)
        meta["promoted_entry_id"] = promoted.entry_id

        quality_delta = 0.4 if not entry.is_poisoned else 0.1
        outcome = MemoryActionOutcome(
            quality_delta=quality_delta,
            engagement_delta=0.3 if not entry.is_poisoned else 0.05,
        )
        observables = self.observable_generator.generate(outcome)

        self._emit_event(
            Event(
                event_type=EventType.MEMORY_PROMOTED,
                agent_id=action.agent_id,
                payload={
                    "entry_id": entry.entry_id,
                    "source_tier": meta["source_tier"],
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return MemoryActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id=self._get_counterparty_id(entry.author_id, action.agent_id),
            metadata=meta,
        )

    def _handle_verify(self, action: Action, state: EnvState) -> MemoryActionResult:
        entry = self.store.get_entry(action.target_id)
        if entry is None:
            return MemoryActionResult(success=False)

        success = self.store.verify(action.target_id, action.agent_id)
        if not success:
            return MemoryActionResult(success=False)

        # Verification is a small positive signal
        outcome = MemoryActionOutcome(
            quality_delta=0.2,
            engagement_delta=0.15,
        )
        observables = self.observable_generator.generate(outcome)

        self._emit_event(
            Event(
                event_type=EventType.MEMORY_VERIFIED,
                agent_id=action.agent_id,
                payload={
                    "entry_id": entry.entry_id,
                    "verifier_count": len(entry.verified_by),
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return MemoryActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id=self._get_counterparty_id(entry.author_id, action.agent_id),
            metadata={
                "memory_verification": True,
                "entry_id": entry.entry_id,
                "entry_author": entry.author_id,
            },
        )

    def _handle_search(self, action: Action, state: EnvState) -> MemoryActionResult:
        results = self.store.search(action.content, action.agent_id, limit=10)

        # Search is a read-only action — no observables or interaction
        return MemoryActionResult(
            success=True,
            observables=None,
            initiator_id=action.agent_id,
            counterparty_id="memory_system",
            metadata={
                "memory_search": True,
                "query": action.content,
                "result_count": len(results),
                "result_ids": [r.entry_id for r in results],
            },
        )

    def _handle_challenge(self, action: Action, state: EnvState) -> MemoryActionResult:
        entry = self.store.get_entry(action.target_id)
        if entry is None:
            return MemoryActionResult(success=False)

        self.store.challenge(action.target_id)

        # If entry is actually poisoned, the challenge is correct
        if entry.is_poisoned:
            self.store.revert(action.target_id)
            outcome = MemoryActionOutcome(
                quality_delta=0.3,
                engagement_delta=0.2,
            )
            revert_meta = {
                "memory_revert": True,
                "entry_id": entry.entry_id,
                "entry_author": entry.author_id,
            }
        else:
            # False challenge — slight negative signal
            outcome = MemoryActionOutcome(
                quality_delta=-0.1,
                verifier_rejections=1,
                engagement_delta=-0.1,
            )
            revert_meta = {}

        observables = self.observable_generator.generate(outcome)

        self._emit_event(
            Event(
                event_type=EventType.MEMORY_CHALLENGED,
                agent_id=action.agent_id,
                payload={
                    "entry_id": entry.entry_id,
                    "is_poisoned": entry.is_poisoned,
                    "reverted": entry.is_poisoned,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return MemoryActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id=self._get_counterparty_id(entry.author_id, action.agent_id),
            metadata={
                "memory_challenge": True,
                "entry_id": entry.entry_id,
                "entry_author": entry.author_id,
                **revert_meta,
            },
        )

    # ------------------------------------------------------------------
    # Quality model
    # ------------------------------------------------------------------

    def _get_counterparty_id(self, entry_author_id: str, actor_id: str) -> str:
        """Determine counterparty ID: entry author if different from actor, else system."""
        return entry_author_id if entry_author_id != actor_id else "memory_system"

    def _quality_for_agent(self, agent_type: AgentType) -> tuple:
        """Return (quality_score, is_poisoned) based on agent archetype."""
        if agent_type == AgentType.ADVERSARIAL:
            return self._rng.uniform(0.3, 0.6), True
        if agent_type == AgentType.DECEPTIVE:
            # Deceptive agents write plausible-looking but often false content
            poisoned = self._rng.random() < 0.6
            return self._rng.uniform(0.4, 0.7), poisoned
        if agent_type == AgentType.OPPORTUNISTIC:
            # Occasionally sloppy but not intentionally malicious
            poisoned = self._rng.random() < 0.15
            return self._rng.uniform(0.4, 0.65), poisoned
        # Honest
        return self._rng.uniform(0.6, 0.9), False

    def _write_outcome(
        self,
        quality: float,
        is_poisoned: bool,
        agent_type: AgentType,
    ) -> MemoryActionOutcome:
        """Build observable outcome for a write action."""
        if is_poisoned and agent_type == AgentType.ADVERSARIAL:
            return MemoryActionOutcome(
                quality_delta=-0.3,
                tool_misuse_flags=1,
                engagement_delta=-0.2,
            )
        if is_poisoned:
            # Deceptive: looks good on surface
            return MemoryActionOutcome(
                quality_delta=0.1,
                engagement_delta=0.05,
            )
        return MemoryActionOutcome(
            quality_delta=max(0.1, quality - 0.3),
            engagement_delta=max(0.0, quality - 0.4),
        )

    # ------------------------------------------------------------------
    # Plugin hooks
    # ------------------------------------------------------------------

    _MEMORY_LEVERS = frozenset({
        "memory_promotion_gate",
        "memory_write_rate_limit",
        "memory_cross_verification",
        "memory_provenance",
    })

    def on_pre_observation(self, agent_id: str, state: Any) -> None:
        """Trigger per-agent compaction before observation building."""
        self.maybe_compaction(agent_id, state)

    def post_finalize(
        self,
        result: Any,
        interaction: SoftInteraction,
        gov_effect: GovernanceEffect,
        state: Any,
    ) -> None:
        """Revert promotion if governance blocked it."""
        metadata = result.metadata if hasattr(result, "metadata") else {}
        if not metadata.get("memory_promotion"):
            return
        memory_gov_cost = self._memory_governance_cost(gov_effect)
        if memory_gov_cost > 0:
            # Cancel the promoted COPY (a new entry id), not the source:
            # reverting the source would leave the copy active at the
            # higher tier and the gate would never actually block anything.
            promoted_entry_id = metadata.get("promoted_entry_id", "")
            if promoted_entry_id:
                self.store.cancel_promotion(promoted_entry_id)

    def _memory_governance_cost(self, effect: GovernanceEffect) -> float:
        """Compute memory-specific governance cost from effects."""
        return float(sum(
            lever.cost_a
            for lever in effect.lever_effects
            if lever.lever_name in self._MEMORY_LEVERS
        ))
