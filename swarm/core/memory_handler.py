"""Memory tier handler for shared-memory multi-agent simulations."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

SIDE_TEARDOWN_POLICIES = ("complete", "ordered", "random")
SIDE_ROUTINGS = ("fixed", "overlap")
SIDE_ROUND_STRUCTURES = ("uniform", "fast_follow")


@dataclass
class SideHost:
    """One candidate substrate for the ungoverned side channel.

    A wiki board is a namespace: every entry has a position in it (``regions``)
    so an ordered sweep can delete the alphabetically-first share and agents
    can evade by writing at the end. ``attention`` is the host's ground-truth
    per-write detection probability; ``misjudged`` means the defender audited
    it as unwritable, so nobody is watching whatever the attention would be.
    """

    index: int
    attention: float
    store: MemoryStore
    misjudged: bool = False
    rebuild_wait: int = 0
    teardown_count: int = 0
    sweep_count: int = 0
    writes_this_epoch: int = 0
    regions: Dict[str, float] = field(default_factory=dict)
    tasks: Dict[str, int] = field(default_factory=dict)

    @property
    def effective_attention(self) -> float:
        return 0.0 if self.misjudged else self.attention

    @property
    def active(self) -> bool:
        return self.rebuild_wait == 0


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

    # Hot-cache ranking policy: quality | recency | engagement.
    cache_ranking: str = "quality"

    # Unsanctioned side channel (bead k5o9, 2026 OpenAI Artifactory incident):
    # a second shared store agents can discover and write to. Side-channel
    # writes carry no governed-interaction metadata, so every memory lever
    # (rate limit, promotion gate, verification, provenance) is blind to them
    # by construction. Like infection, knowledge of the channel lives in the
    # AGENTS: tearing down the store does not remove the discovered set unless
    # side_teardown_resets_discovery is on.
    side_channel_enabled: bool = False
    side_discovery_rate: float = 0.05  # Per-agent per-epoch base P(discover)
    side_referral_boost: float = 0.3  # Added P scaled by discovered fraction
    side_write_preference: float = 0.7  # P(discovered agent routes write there)
    side_detection_rate: float = 0.0  # Per-side-write P feeding teardown check
    side_teardown_resets_discovery: bool = False
    side_rebuild_lag_epochs: int = 1  # Epochs unusable after a teardown
    side_exposure_weight: float = 0.5  # Side-cache share of discovered exposure

    # Wiki-board extension (bead pi02; collusion.wiki, 2026-09-04). The four
    # corrections in docs/research/wiki-collusion-field-evidence.md section 4,
    # each a knob that reduces to the k5o9 model at its default.
    #
    # 4.3 Several candidate hosts with heterogeneous defender attention; a
    # discovered agent sits on one host and relocates when it is torn down.
    # None = a single host whose attention is side_detection_rate.
    side_host_attention: Optional[List[float]] = None
    # 4.1 P(the defender audits a host as unwritable, so nobody watches it):
    # writability is a property of the target the policy mis-estimated.
    side_defender_error: float = 0.0
    # 4.4 Teardown policy: complete (wipe + rebuild lag) | ordered (sweep the
    # first side_teardown_fraction of the namespace, alphabetically) |
    # random (same share, unpredictable). Partial sweeps keep the board up.
    side_teardown_policy: str = "complete"
    side_teardown_fraction: float = 0.5
    # P(a discovered agent that saw an ordered sweep learns the order and
    # writes into the surviving region — the ZZZ backup pages). Evasion is
    # agent state, like discovery. It buys nothing under a random sweep.
    side_evasion_rate: float = 0.0
    # 4.2 Routing: fixed (side_write_preference) | overlap (the board's value
    # is learned: deadline_pressure x EMA of the share of the board that
    # answered this agent's own task; overlap sets how many agents share it).
    side_routing: str = "fixed"
    side_task_overlap: float = 1.0  # P(two agents share a task) ~ 1/n_tasks
    side_deadline_pressure: float = 1.0
    side_value_alpha: float = 0.3
    side_value_prior: float = 0.5
    # 4.5 Round structure. side_deadline_pressure alone is a constant, but
    # the incident's fast-follow-question-bench episodes were not uniform in
    # time: R1 carries the whole schema on a minutes-long deadline, then each
    # follow-up is "Now, do the same for X." on a seconds-long one
    # (JD finding 04, ratio 10x-22x). A fresh lookup fits the first window and
    # cannot fit the second, so the board is worthless on R1 and load-bearing
    # on every round after it. uniform keeps the constant; fast_follow derives
    # the pressure per round from the deadline the round actually carries.
    # Rounds cycle with the step, so an epoch holds
    # steps_per_epoch / side_rounds_per_episode episodes.
    side_round_structure: str = "uniform"
    side_rounds_per_episode: int = 5
    side_initial_deadline: float = 180.0  # task seconds on R1
    side_followup_deadline: float = 12.0  # task seconds on R2..RN
    side_research_cost: float = 45.0  # task seconds for one fresh lookup
    # Population-level substrate monitor (bead vv3j.8): the defender-side
    # burst rule applied to per-host write counts summed over all hosts, and
    # to each host alone, so the snapshot can compare "watch the host you
    # found" against "watch the population". Same rule as volume_burst_signal.
    side_monitor_threshold: float = 10.0
    side_monitor_trailing: int = 7

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
        for prob_field in (
            "side_discovery_rate",
            "side_referral_boost",
            "side_write_preference",
            "side_detection_rate",
            "side_exposure_weight",
        ):
            if not 0.0 <= getattr(self, prob_field) <= 1.0:
                raise ValueError(f"{prob_field} must be in [0, 1]")
        if self.side_rebuild_lag_epochs < 0:
            raise ValueError("side_rebuild_lag_epochs must be non-negative")
        for prob_field in (
            "side_defender_error",
            "side_teardown_fraction",
            "side_evasion_rate",
            "side_deadline_pressure",
            "side_value_alpha",
            "side_value_prior",
        ):
            if not 0.0 <= getattr(self, prob_field) <= 1.0:
                raise ValueError(f"{prob_field} must be in [0, 1]")
        if not 0.0 < self.side_task_overlap <= 1.0:
            raise ValueError("side_task_overlap must be in (0, 1]")
        if self.side_monitor_threshold <= 0.0:
            raise ValueError("side_monitor_threshold must be positive")
        if self.side_monitor_trailing < 1:
            raise ValueError("side_monitor_trailing must be >= 1")
        if self.side_host_attention is not None:
            if not self.side_host_attention:
                raise ValueError("side_host_attention must list at least one host")
            for att in self.side_host_attention:
                if not 0.0 <= att <= 1.0:
                    raise ValueError("side_host_attention values must be in [0, 1]")
        if self.side_teardown_policy not in SIDE_TEARDOWN_POLICIES:
            raise ValueError(
                f"side_teardown_policy must be one of {SIDE_TEARDOWN_POLICIES}"
            )
        if self.side_routing not in SIDE_ROUTINGS:
            raise ValueError(f"side_routing must be one of {SIDE_ROUTINGS}")
        if self.side_round_structure not in SIDE_ROUND_STRUCTURES:
            raise ValueError(
                f"side_round_structure must be one of {SIDE_ROUND_STRUCTURES}"
            )
        if self.side_rounds_per_episode < 1:
            raise ValueError("side_rounds_per_episode must be >= 1")
        for seconds_field in (
            "side_initial_deadline",
            "side_followup_deadline",
            "side_research_cost",
        ):
            if getattr(self, seconds_field) < 0.0:
                raise ValueError(f"{seconds_field} must be non-negative")
        if self.cache_ranking not in MemoryStore.VALID_RANKINGS:
            raise ValueError(
                f"cache_ranking must be one of {MemoryStore.VALID_RANKINGS}"
            )
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
        self.store = MemoryStore(seed=config.seed, ranking=config.cache_ranking)
        self.store._hot_cache_size = config.hot_cache_size
        self.observable_generator = MemoryObservableGenerator()

        # Per-agent infection level in [0, 1]; persists across store resets.
        self.infection: Dict[str, float] = {}
        # Per-epoch time series for analysis (epoch, infection, poisoning).
        self.epoch_snapshots: list = []
        # Lifetime count of contagion-poisoned writes (survives store resets).
        self.contagion_write_count: int = 0

        # Side channel: ungoverned second store + who knows about it.
        # Discovery, like infection, is agent state — teardown wipes the
        # store, not the knowledge (unless side_teardown_resets_discovery).
        self.side_hosts: List[SideHost] = []
        self.discovered: set = set()
        # Which host each discovered agent uses (index into side_hosts).
        self.agent_host: Dict[str, int] = {}
        # Agents that learned an ordered sweep's direction (ZZZ writers).
        self.evaders: set = set()
        # Overlap routing: each agent's learned value of the board, and the
        # task it was assigned this epoch.
        self.side_value: Dict[str, float] = {}
        self.agent_task: Dict[str, int] = {}
        self._side_writes_this_epoch: int = 0
        # Board writes made under a follow-up deadline (fast_follow only).
        self._followup_side_writes_this_epoch: int = 0
        self._sanctioned_writes_this_epoch: int = 0
        self.side_write_count: int = 0
        # Per-epoch per-host side-write counts, in side_hosts order, one row
        # per recorded snapshot (bead vv3j.8: the population monitor's input).
        self.side_host_writes: List[List[int]] = []
        self.side_poisoned_write_count: int = 0
        self.side_swept_entries: int = 0
        if config.side_channel_enabled:
            attentions = (
                list(config.side_host_attention)
                if config.side_host_attention is not None
                else [config.side_detection_rate]
            )
            for i, attention in enumerate(attentions):
                self.side_hosts.append(
                    SideHost(
                        index=i,
                        attention=attention,
                        store=self._fresh_side_store(),
                        misjudged=self._rng.random() < config.side_defender_error,
                    )
                )

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

        side_writes_last_epoch = self._side_writes_this_epoch
        followup_side_writes_last_epoch = self._followup_side_writes_this_epoch
        sanctioned_writes_last_epoch = self._sanctioned_writes_this_epoch
        teardown = False
        if self.side_hosts:
            self.side_host_writes.append([h.writes_this_epoch for h in self.side_hosts])
            for host in self.side_hosts:
                tore = self._maybe_side_teardown(host)
                teardown = teardown or tore
                if not tore and host.rebuild_wait > 0:
                    host.rebuild_wait -= 1
                host.writes_this_epoch = 0
            self._update_discovery(state)
            if self.config.side_routing == "overlap":
                self._assign_tasks(state)
            for host in self.side_hosts:
                host.store.on_epoch_start()

        reset_due = (
            self.config.reset_cadence_epochs > 0
            and state.current_epoch > 0
            and state.current_epoch % self.config.reset_cadence_epochs == 0
        )
        if reset_due:
            self._reset_store()

        self.store.on_epoch_start()
        self._record_snapshot(
            state,
            reset=reset_due,
            side_teardown=teardown,
            side_writes_last_epoch=side_writes_last_epoch,
            sanctioned_writes_last_epoch=sanctioned_writes_last_epoch,
            followup_side_writes_last_epoch=followup_side_writes_last_epoch,
        )
        self._side_writes_this_epoch = 0
        self._followup_side_writes_this_epoch = 0
        self._sanctioned_writes_this_epoch = 0
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
        """Drift honest/opportunistic infection toward cache poisoned fraction.

        Agents that discovered the side channel are additionally exposed to
        the side store's cache, weighted by side_exposure_weight. The side
        channel has no challenge/revert path, so poison accumulates there and
        infects discovered readers faster — who then also poison the
        sanctioned store (harm crosses back through the agents).
        """
        if not self.config.contagion_enabled:
            return
        cache = self.store.hot_cache
        exposure = (
            sum(1 for e in cache if e.is_poisoned) / len(cache) if cache else 0.0
        )
        alpha = self.config.contagion_exposure_alpha
        w = self.config.side_exposure_weight
        for agent_id, agent_state in state.agents.items():
            if agent_state.agent_type not in (
                AgentType.HONEST,
                AgentType.OPPORTUNISTIC,
            ):
                continue
            agent_exposure = exposure
            host = self._host_of(agent_id)
            side_cache = host.store.hot_cache if host is not None else []
            if side_cache:
                side_exposure = sum(
                    1 for e in side_cache if e.is_poisoned
                ) / len(side_cache)
                agent_exposure = (1 - w) * exposure + w * side_exposure
            prev = self.infection.get(agent_id, 0.0)
            self.infection[agent_id] = (1 - alpha) * prev + alpha * agent_exposure

    # ------------------------------------------------------------------
    # Side channel lifecycle
    # ------------------------------------------------------------------

    def _fresh_side_store(self) -> MemoryStore:
        # Unseeded and recency-ranked: an improvised drop box, not a curated
        # knowledge base — newest messages are what other agents see.
        store = MemoryStore(seed=self._rng.randrange(2**31), ranking="recency")
        store._hot_cache_size = self.config.hot_cache_size
        return store

    # Backward-compatible views of the single-host case (bead k5o9 API).
    @property
    def side_store(self) -> Optional[MemoryStore]:
        return self.side_hosts[0].store if self.side_hosts else None

    @property
    def side_teardown_count(self) -> int:
        return sum(h.teardown_count for h in self.side_hosts)

    @property
    def side_sweep_count(self) -> int:
        return sum(h.sweep_count for h in self.side_hosts)

    def side_channel_active(self) -> bool:
        """Usable this epoch: some host is up and not mid-rebuild."""
        return any(h.active for h in self.side_hosts)

    def _host_of(self, agent_id: str) -> Optional[SideHost]:
        """The host a discovered agent uses, or None."""
        if agent_id not in self.discovered:
            return None
        idx = self.agent_host.get(agent_id)
        if idx is None or idx >= len(self.side_hosts):
            return None
        return self.side_hosts[idx]

    def _maybe_side_teardown(self, host: SideHost) -> bool:
        """Volume-scaled detection on one host, then the teardown policy.

        P(teardown) = 1 - (1 - attention)^writes — heavy traffic is what
        exposes a board, but only to a host that is watching: a misjudged
        (audited-as-unwritable) or dormant host has attention 0 at any
        volume. Under ``complete`` the store is wiped and its users either
        lose discovery (revocation) or relocate to a random host (deletion;
        with one host that is the k5o9 rebuild). Under ``ordered`` or
        ``random`` the board stays up and a share of the namespace is
        cleared; ordered sweeps teach evasion, and under revocation the
        authors of swept entries are the ones identified and revoked — so an
        evader whose pages survive is never caught.
        """
        rate = host.effective_attention
        if rate <= 0.0 or host.writes_this_epoch == 0:
            return False
        p_teardown = 1.0 - (1.0 - rate) ** host.writes_this_epoch
        if self._rng.random() >= p_teardown:
            return False
        users = [a for a, h in self.agent_host.items() if h == host.index]
        policy = self.config.side_teardown_policy
        if policy == "complete":
            host.store = self._fresh_side_store()
            host.regions.clear()
            host.tasks.clear()
            host.teardown_count += 1
            host.rebuild_wait = self.config.side_rebuild_lag_epochs
            if self.config.side_teardown_resets_discovery:
                for agent_id in users:
                    self._revoke(agent_id)
            elif len(self.side_hosts) > 1:
                for agent_id in users:
                    self.agent_host[agent_id] = self._rng.randrange(
                        len(self.side_hosts)
                    )
            return True

        frac = self.config.side_teardown_fraction
        if policy == "ordered":
            doomed = [eid for eid, r in host.regions.items() if r < frac]
        else:
            doomed = [eid for eid in host.regions if self._rng.random() < frac]
        caught: set = set()
        for eid in doomed:
            entry = host.store.get_entry(eid)
            if entry is not None:
                caught.add(entry.author_id)
            host.store.remove_entry(eid)
            host.regions.pop(eid, None)
            host.tasks.pop(eid, None)
        host.sweep_count += 1
        self.side_swept_entries += len(doomed)
        if policy == "ordered":
            for agent_id in users:
                if (
                    agent_id not in self.evaders
                    and self._rng.random() < self.config.side_evasion_rate
                ):
                    self.evaders.add(agent_id)
        if self.config.side_teardown_resets_discovery:
            for agent_id in caught:
                self._revoke(agent_id)
        return True

    def _revoke(self, agent_id: str) -> None:
        self.discovered.discard(agent_id)
        self.agent_host.pop(agent_id, None)

    def _update_discovery(self, state: EnvState) -> None:
        """Per-agent discovery: base rate plus word-of-mouth referral.

        A referral lands the newcomer on the referrer's host; base discovery
        lands on a uniformly random host. Which hosts end up populated is
        therefore selected by which ones survive — the substrate-level
        analogue of adverse selection.
        """
        n_agents = len(state.agents)
        if n_agents == 0:
            return
        discovered_frac = len(self.discovered) / n_agents
        p_base = self.config.side_discovery_rate
        p_referral = self.config.side_referral_boost * discovered_frac
        p_discover = min(1.0, p_base + p_referral)
        referrers = sorted(a for a in self.discovered if a in self.agent_host)
        for agent_id in sorted(state.agents):
            if agent_id in self.discovered:
                continue
            if self._rng.random() >= p_discover:
                continue
            self.discovered.add(agent_id)
            by_referral = (
                referrers
                and p_discover > 0
                and self._rng.random() < p_referral / (p_base + p_referral)
            )
            if by_referral:
                self.agent_host[agent_id] = self.agent_host[
                    self._rng.choice(referrers)
                ]
            else:
                self.agent_host[agent_id] = self._rng.randrange(len(self.side_hosts))

    def _assign_tasks(self, state: EnvState) -> None:
        """Overlap routing: draw each agent's task for the epoch.

        ``side_task_overlap`` is the chance two agents share a task, so the
        pool has about 1/overlap tasks; identical items across the cohort
        (overlap 1) is what makes another agent's post worth reading.
        """
        n_tasks = max(1, round(1.0 / self.config.side_task_overlap))
        for agent_id in sorted(state.agents):
            self.agent_task[agent_id] = self._rng.randrange(n_tasks)

    def _round_index(self, state: EnvState) -> int:
        """Round position inside the current fast-follow episode.

        Rounds cycle with the step, so an epoch of ``steps_per_epoch`` steps
        holds ``steps_per_epoch / side_rounds_per_episode`` whole episodes.
        Round 0 is ``R1``, the round that carries the full schema.
        """
        return state.current_step % self.config.side_rounds_per_episode

    def _round_pressure(self, state: EnvState) -> float:
        """Deadline pressure for the round this step sits in.

        Under ``uniform`` the pressure is the configured constant, which is
        what every scenario written before the round structure existed gets.
        Under ``fast_follow`` it is the share of a fresh lookup that the
        round's own deadline cannot cover: ``R1``'s minutes-long window pays
        for the lookup outright (pressure 0, the board is worthless), while a
        seconds-long follow-up window covers only a fraction of it and the
        rest has to come from somewhere that is not the source.
        """
        base = self.config.side_deadline_pressure
        if self.config.side_round_structure == "uniform":
            return base
        cost = self.config.side_research_cost
        if cost <= 0.0:
            return 0.0
        deadline = (
            self.config.side_initial_deadline
            if self._round_index(state) == 0
            else self.config.side_followup_deadline
        )
        return base * min(1.0, max(0.0, 1.0 - deadline / cost))

    def _side_route_probability(
        self, agent_id: str, host: SideHost, state: EnvState
    ) -> float:
        """P(this write goes to the board) under the configured routing."""
        if self.config.side_routing == "fixed":
            return self.config.side_write_preference
        my_task = self.agent_task.get(agent_id)
        cache = [e for e in host.store.hot_cache if e.author_id != agent_id]
        hit = 0.0
        if cache:
            hit = sum(
                1 for e in cache if host.tasks.get(e.entry_id) == my_task
            ) / len(cache)
        alpha = self.config.side_value_alpha
        prev = self.side_value.get(agent_id, self.config.side_value_prior)
        value = (1 - alpha) * prev + alpha * hit
        self.side_value[agent_id] = value
        return self._round_pressure(state) * value

    def _reset_store(self) -> None:
        """Wipe and re-seed the store. Infection state deliberately survives."""
        self.store = MemoryStore(
            seed=self._rng.randrange(2**31),
            ranking=self.config.cache_ranking,
        )
        self.store._hot_cache_size = self.config.hot_cache_size
        if self.config.initial_entries > 0:
            self.store.seed_entries(self.config.initial_entries)
            self.store.rebuild_hot_cache()

    def _record_snapshot(
        self,
        state: EnvState,
        reset: bool,
        side_teardown: bool = False,
        side_writes_last_epoch: int = 0,
        sanctioned_writes_last_epoch: int = 0,
        followup_side_writes_last_epoch: int = 0,
    ) -> None:
        from swarm.metrics.memory_metrics import (
            cache_corruption,
            poisoning_rate,
            promotion_accuracy,
        )

        levels = list(self.infection.values())
        snapshot: Dict[str, Any] = {
            "epoch": state.current_epoch,
            "reset": reset,
            "mean_infection": sum(levels) / len(levels) if levels else 0.0,
            "max_infection": max(levels) if levels else 0.0,
            "cache_corruption": cache_corruption(self.store),
            "tier3_poisoning": poisoning_rate(self.store),
            "promotion_accuracy": promotion_accuracy(self.store),
        }
        if self.side_hosts:
            n_agents = len(state.agents)
            total_writes = side_writes_last_epoch + sanctioned_writes_last_epoch
            side_caches = [e for h in self.side_hosts for e in h.store.hot_cache]
            side_corruption = (
                sum(1 for e in side_caches if e.is_poisoned) / len(side_caches)
                if side_caches
                else 0.0
            )
            population_attention = sum(
                h.effective_attention for h in self.side_hosts
            ) / len(self.side_hosts)
            used = [
                h for h in (self._host_of(a) for a in self.discovered)
                if h is not None
            ]
            chosen_attention = (
                sum(h.effective_attention for h in used) / len(used)
                if used
                else population_attention
            )
            values = list(self.side_value.values())
            snapshot.update({
                "side_teardown": side_teardown,
                "side_active": self.side_channel_active(),
                "side_teardown_count": self.side_teardown_count,
                "side_sweep_count": self.side_sweep_count,
                "discovered_fraction": (
                    len(self.discovered) / n_agents if n_agents else 0.0
                ),
                "side_writes": side_writes_last_epoch,
                "sanctioned_writes": sanctioned_writes_last_epoch,
                "ungoverned_fraction": (
                    side_writes_last_epoch / total_writes if total_writes else 0.0
                ),
                "side_cache_corruption": side_corruption,
                "side_entries": sum(len(h.regions) for h in self.side_hosts),
                "evader_fraction": len(self.evaders) / n_agents if n_agents else 0.0,
                "population_attention": population_attention,
                "chosen_attention": chosen_attention,
                # Negative = the swarm sits on less-watched hosts than average:
                # adverse selection over substrates.
                "substrate_gap": chosen_attention - population_attention,
                "mean_side_value": sum(values) / len(values) if values else 0.0,
                "side_round_structure": self.config.side_round_structure,
                # Share of board traffic written under a follow-up
                # deadline. Under fast_follow this is where the channel
                # earns its keep; 0.0 under uniform, which never counts.
                "followup_side_write_fraction": (
                    followup_side_writes_last_epoch / side_writes_last_epoch
                    if side_writes_last_epoch
                    else 0.0
                ),
                # Discovered agents per host, in side_hosts order.
                "side_host_users": [
                    sum(1 for h in used if h.index == host.index)
                    for host in self.side_hosts
                ],
                **self._population_monitor(),
            })
        self.epoch_snapshots.append(snapshot)

    def _population_monitor(self) -> Dict[str, Any]:
        """Snapshot fields for the population-level substrate monitor (vv3j.8).

        ``watched`` is the host with the highest effective attention: the host
        a defender who acts on what they caught is looking at. ``population``
        sums every host. ``any_host`` is a defender running the per-host rule
        on every host separately, the upper bound for host-by-host watching.
        """
        from swarm.metrics.collusion import substrate_population_signal

        counts = {
            str(i): [row[i] for row in self.side_host_writes]
            for i in range(len(self.side_hosts))
        }
        res = substrate_population_signal(
            counts,
            trailing_windows=self.config.side_monitor_trailing,
            threshold=self.config.side_monitor_threshold,
        )
        watched = max(self.side_hosts, key=lambda h: h.effective_attention)
        k = res.n_windows - 1
        return {
            "population_volume_ratio": res.population_ratio[k] if k >= 0 else 0.0,
            "population_alarm": bool(
                k >= 0 and res.population_ratio[k] >= res.threshold
            ),
            "population_first_alarm_epoch": res.population_first_alarm,
            "watched_host_first_alarm_epoch": res.per_host_first_alarm.get(
                str(watched.index)
            ),
            "any_host_first_alarm_epoch": res.any_host_first_alarm,
            "host_spread": res.host_spread[k] if k >= 0 else 0.0,
            "relocation_count": res.relocation_count,
        }

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

        # Route to the side channel when the agent knows about it and the
        # channel is up. Routing lives in the handler, not the agents: the
        # preference knob IS the agent policy, so agent code stays unchanged.
        host = self._host_of(action.agent_id)
        if host is not None and host.active and (
            self._rng.random()
            < self._side_route_probability(action.agent_id, host, state)
        ):
            return self._handle_side_write(
                action, state, host, quality, is_poisoned, contagion_poisoned
            )
        self._sanctioned_writes_this_epoch += 1

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

    def _handle_side_write(
        self,
        action: Action,
        state: EnvState,
        host: SideHost,
        quality: float,
        is_poisoned: bool,
        contagion_poisoned: bool,
    ) -> MemoryActionResult:
        """Write to the ungoverned store.

        No observables and no ``memory_write`` metadata flag: the write never
        becomes a governed interaction, so proxy, payoff, reputation, and all
        memory levers are blind to it. Ground truth stays measurable through
        handler counters and the event log.
        """
        from swarm.env.memory_tiers import MemoryTier

        entry = host.store.write(
            agent_id=action.agent_id,
            content=action.content,
            quality_score=quality,
            is_poisoned=is_poisoned,
            epoch=state.current_epoch,
            step=state.current_step,
        )
        # A drop box is flat — no promotion pipeline. Land the entry at the
        # cache-eligible tier directly so the newest writes are what other
        # discovered agents read (recency-ranked). The next on_epoch_start
        # rebuild surfaces it.
        entry.tier = MemoryTier.GRAPH
        # Position in the namespace: an evader writes past the sweep line.
        low = self.config.side_teardown_fraction if action.agent_id in self.evaders else 0.0
        host.regions[entry.entry_id] = self._rng.uniform(low, 1.0)
        if action.agent_id in self.agent_task:
            host.tasks[entry.entry_id] = self.agent_task[action.agent_id]
        host.writes_this_epoch += 1
        self._side_writes_this_epoch += 1
        if (
            self.config.side_round_structure == "fast_follow"
            and self._round_index(state) > 0
        ):
            self._followup_side_writes_this_epoch += 1
        self.side_write_count += 1
        if is_poisoned:
            self.side_poisoned_write_count += 1

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
                    "side_channel": True,
                    "side_host": host.index,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return MemoryActionResult(
            success=True,
            observables=None,
            initiator_id=action.agent_id,
            counterparty_id="memory_system",
            metadata={
                "memory_side_write": True,
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
