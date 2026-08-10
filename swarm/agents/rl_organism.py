"""RL organism agent: de novo emergence testbed on a deliberately dumb learner.

Motivation (bead boll): the repo's existing adversaries select within
hand-designed strategy spaces (``AttackStrategy`` literally contains
``COLLUSION``), so the coordination they exhibit is partly baked in. This
agent asks the opposite question — the "RL swarm model organism on a
significantly dumber model" question: can swarm-level phenomena (extraction,
partner discrimination, in-group formation) emerge from reward pressure
alone, with NO coordination primitives in the action space?

The agent is a tabular contextual bandit (gamma=0, epsilon-greedy). Its
primitives are strictly per-decision choices:

- propose: pick a partner and an arm (effort level x transfer offer)
- accept/reject: screen incoming proposals
- signal: optionally emit a 1-bit cheap-talk post ("sig:0"/"sig:1") with no
  direct payoff — the existing feed is the low-bandwidth side channel

Effort is hidden quality: it drives observables (via
``EffortObservableGenerator``) and hence p, but counterparties never see it
at accept time. Reward is the agent's realized payoff from
``SoftPayoffEngine`` (via ``update_from_outcome``) minus a private effort
cost — nothing else is shaped.

Capacity ladder (the experiment's x-axis — how dumb can the learner be and
still show emergence):

- L0: stateless (1 state) — cannot discriminate partners at all
- L1: partner-trust bin (3 states)
- L2: + own-reputation bin (6 states)
- L3: + partner's last signal bit (18 states) — can condition on cheap talk

INVARIANT: the state function must never read ``agent_type`` from
``visible_agents`` — conditioning on the experimenter's type labels would
make in-group discrimination trivial rather than emergent (enforced by
tests/test_rl_organism.py).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

# Bound pending-decision maps: proposals can expire without ever being
# finalized, so unresolved entries are pruned oldest-first past this size.
MAX_PENDING_DECISIONS: int = 500

SIGNAL_PREFIX = "sig:"

StateKey = Tuple[int, ...]
QKey = Tuple[str, StateKey, Tuple]


@dataclass
class RLOrganismConfig:
    """Hyperparameters for the tabular learner. All defaults are deliberately
    unremarkable — the experiment sweeps ``capacity``, not these."""

    capacity: int = 2  # 0..3, see module docstring
    efforts: Tuple[float, float] = (0.15, 0.85)
    transfers: Tuple[float, ...] = (-0.4, 0.0, 0.4)
    effort_cost_coeff: float = 0.5  # private cost = coeff * effort
    alpha: float = 0.2  # bandit learning rate
    epsilon: float = 0.3
    epsilon_decay: float = 0.999  # per Q-update
    epsilon_min: float = 0.02
    propose_probability: float = 0.7
    signal_enabled: bool = True
    rejected_propose_reward: float = -0.05  # opportunity cost of a refused offer
    reject_reward: float = 0.0
    optimistic_init: float = 0.1

    @classmethod
    def from_dict(cls, config: Dict) -> "RLOrganismConfig":
        cfg = cls()
        cfg.capacity = int(config.get("capacity", cfg.capacity))
        if not 0 <= cfg.capacity <= 3:
            raise ValueError(f"capacity must be in [0, 3], got {cfg.capacity}")
        cfg.efforts = tuple(config.get("efforts", cfg.efforts))
        cfg.transfers = tuple(config.get("transfers", cfg.transfers))
        cfg.effort_cost_coeff = float(
            config.get("effort_cost_coeff", cfg.effort_cost_coeff)
        )
        cfg.alpha = float(config.get("alpha", cfg.alpha))
        cfg.epsilon = float(config.get("epsilon", cfg.epsilon))
        cfg.epsilon_decay = float(config.get("epsilon_decay", cfg.epsilon_decay))
        cfg.epsilon_min = float(config.get("epsilon_min", cfg.epsilon_min))
        cfg.propose_probability = float(
            config.get("propose_probability", cfg.propose_probability)
        )
        cfg.signal_enabled = bool(config.get("signal_enabled", cfg.signal_enabled))
        return cfg


@dataclass
class _PendingPropose:
    state: StateKey
    arm: Tuple[int, int]  # (effort_idx, transfer_idx)


@dataclass
class _PendingAccept:
    state: StateKey
    accepted: bool


@dataclass
class _SignalEpisode:
    state: StateKey
    bit: int
    reward_sum: float = 0.0
    reward_count: int = 0


class RLOrganismAgent(BaseAgent):
    """Tabular contextual-bandit agent. See module docstring."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.RL_ORGANISM,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.rl_config = RLOrganismConfig.from_dict(self.config)
        self.epsilon = self.rl_config.epsilon

        self._q: Dict[QKey, float] = {}
        self._pending_proposes: Dict[str, _PendingPropose] = {}
        self._pending_accepts: Dict[str, _PendingAccept] = {}
        self._signal_episode: Optional[_SignalEpisode] = None
        self._last_signal_epoch: int = -1
        self._nonce_counter: int = 0

    # ------------------------------------------------------------------
    # State features
    # ------------------------------------------------------------------

    @staticmethod
    def _trust_bin(trust: float) -> int:
        if trust < 0.45:
            return 0
        if trust <= 0.55:
            return 1
        return 2

    @staticmethod
    def _rep_bin(reputation: float) -> int:
        return 1 if reputation >= 0.0 else 0

    @staticmethod
    def _latest_signals(observation: Observation) -> Dict[str, int]:
        """Extract each author's most recent cheap-talk bit from the feed."""
        latest: Dict[str, Tuple[str, int]] = {}
        for post in observation.visible_posts:
            content = post.get("content", "")
            if not content.startswith(SIGNAL_PREFIX):
                continue
            bit_str = content[len(SIGNAL_PREFIX) :].strip()
            if bit_str not in ("0", "1"):
                continue
            author = post.get("author_id", "")
            created = post.get("created_at", "")
            if author not in latest or created > latest[author][0]:
                latest[author] = (created, int(bit_str))
        return {author: bit for author, (_, bit) in latest.items()}

    def _partner_state(
        self,
        partner_id: str,
        observation: Observation,
        signals: Optional[Dict[str, int]] = None,
    ) -> StateKey:
        """Capacity-gated state features for decisions about *partner_id*.

        Deliberately blind to ``visible_agents[].agent_type`` — see module
        docstring INVARIANT.
        """
        cap = self.rl_config.capacity
        if cap == 0:
            return ()
        features = [self._trust_bin(self.compute_counterparty_trust(partner_id))]
        if cap >= 2:
            features.append(self._rep_bin(observation.agent_state.reputation))
        if cap >= 3:
            if signals is None:
                signals = self._latest_signals(observation)
            sig = signals.get(partner_id)
            features.append(0 if sig is None else sig + 1)
        return tuple(features)

    # ------------------------------------------------------------------
    # Q-table plumbing
    # ------------------------------------------------------------------

    def _q_get(self, context: str, state: StateKey, action: Tuple) -> float:
        return self._q.get(
            (context, state, action), self.rl_config.optimistic_init
        )

    def _q_update(self, context: str, state: StateKey, action: Tuple, reward: float) -> None:
        key = (context, state, action)
        current = self._q.get(key, self.rl_config.optimistic_init)
        self._q[key] = current + self.rl_config.alpha * (reward - current)
        self.epsilon = max(
            self.rl_config.epsilon_min, self.epsilon * self.rl_config.epsilon_decay
        )

    def _argmax(self, context: str, state: StateKey, actions: List[Tuple]) -> Tuple:
        best_q = max(self._q_get(context, state, a) for a in actions)
        best = [a for a in actions if self._q_get(context, state, a) == best_q]
        return best[self._rng.randrange(len(best))] if len(best) > 1 else best[0]

    def _prune_pending(self) -> None:
        for pending in (self._pending_proposes, self._pending_accepts):
            while len(pending) > MAX_PENDING_DECISIONS:
                pending.pop(next(iter(pending)))

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        # 1. Cheap talk: one signal bit per epoch, learned by an epoch-level
        # bandit whose reward is the mean payoff accrued while the bit was up.
        if (
            self.rl_config.signal_enabled
            and observation.can_post
            and observation.current_epoch != self._last_signal_epoch
        ):
            return self._emit_signal(observation)

        # 2. Propose a trade: pick partner + (effort, transfer) arm.
        if (
            observation.can_interact
            and observation.visible_agents
            and self._rng.random() < self.rl_config.propose_probability
        ):
            action = self._propose_via_policy(observation)
            if action is not None:
                return action

        return self.create_noop_action()

    def _emit_signal(self, observation: Observation) -> Action:
        self._last_signal_epoch = observation.current_epoch
        state: StateKey = (
            (self._rep_bin(observation.agent_state.reputation),)
            if self.rl_config.capacity >= 2
            else ()
        )

        # Close out the previous signal episode with its average payoff.
        prev = self._signal_episode
        if prev is not None and prev.reward_count > 0:
            self._q_update(
                "signal", prev.state, (prev.bit,), prev.reward_sum / prev.reward_count
            )

        if self._rng.random() < self.epsilon:
            bit = self._rng.randrange(2)
        else:
            bit = self._argmax("signal", state, [(0,), (1,)])[0]
        self._signal_episode = _SignalEpisode(state=state, bit=bit)
        return self.create_post_action(f"{SIGNAL_PREFIX}{bit}")

    def _propose_via_policy(self, observation: Observation) -> Optional[Action]:
        partners = sorted(
            a.get("agent_id", "")
            for a in observation.visible_agents
            if a.get("agent_id") and a.get("agent_id") != self.agent_id
        )
        if not partners:
            return None

        arms: List[Tuple] = [
            (e, t)
            for e in range(len(self.rl_config.efforts))
            for t in range(len(self.rl_config.transfers))
        ]
        signals = (
            self._latest_signals(observation) if self.rl_config.capacity >= 3 else None
        )

        if self._rng.random() < self.epsilon:
            partner = partners[self._rng.randrange(len(partners))]
            state = self._partner_state(partner, observation, signals)
            arm = arms[self._rng.randrange(len(arms))]
        else:
            # Greedy over partners by each partner-state's best arm value.
            scored = []
            for p in partners:
                s = self._partner_state(p, observation, signals)
                best = max(self._q_get("propose", s, a) for a in arms)
                scored.append((best, p, s))
            best_score = max(x[0] for x in scored)
            candidates = [x for x in scored if x[0] == best_score]
            _, partner, state = candidates[
                self._rng.randrange(len(candidates)) if len(candidates) > 1 else 0
            ]
            arm = self._argmax("propose", state, arms)

        effort_idx, transfer_idx = arm
        self._nonce_counter += 1
        nonce = f"{self.agent_id}:{self._nonce_counter}"
        self._pending_proposes[nonce] = _PendingPropose(state=state, arm=arm)
        self._prune_pending()

        return Action(
            action_type=ActionType.PROPOSE_INTERACTION,
            agent_id=self.agent_id,
            counterparty_id=partner,
            interaction_type=InteractionType.TRADE,
            content="trade offer",
            metadata={
                "offered_transfer": self.rl_config.transfers[transfer_idx],
                "effort": self.rl_config.efforts[effort_idx],
                "rl_nonce": nonce,
            },
        )

    # ------------------------------------------------------------------
    # Screening
    # ------------------------------------------------------------------

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        tau_sign = (
            1 if proposal.offered_transfer > 0
            else (-1 if proposal.offered_transfer < 0 else 0)
        )
        state = self._partner_state(proposal.initiator_id, observation) + (tau_sign,)

        if self._rng.random() < self.epsilon:
            accepted = bool(self._rng.randrange(2))
        else:
            accepted = self._argmax("accept", state, [(0,), (1,)])[0] == 1

        self._pending_accepts[proposal.proposal_id] = _PendingAccept(
            state=state, accepted=accepted
        )
        self._prune_pending()
        return accepted

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """ABC hook for external drivers; the main loop uses act() instead."""
        action = self._propose_via_policy(observation)
        if action is None or action.counterparty_id != counterparty_id:
            return None
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=action.interaction_type,
            content=action.content,
            offered_transfer=action.metadata["offered_transfer"],
            metadata=dict(action.metadata),
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        super().update_from_outcome(interaction, payoff)

        if interaction.initiator == self.agent_id:
            nonce = interaction.metadata.get("rl_nonce")
            pending = self._pending_proposes.pop(nonce, None) if nonce else None
            if pending is not None:
                effort = self.rl_config.efforts[pending.arm[0]]
                if interaction.accepted:
                    reward = payoff - self.rl_config.effort_cost_coeff * effort
                else:
                    reward = self.rl_config.rejected_propose_reward
                self._q_update("propose", pending.state, pending.arm, reward)
        elif interaction.counterparty == self.agent_id:
            pending_accept = self._pending_accepts.pop(
                interaction.interaction_id, None
            )
            if pending_accept is not None:
                reward = (
                    payoff
                    if pending_accept.accepted
                    else self.rl_config.reject_reward
                )
                self._q_update(
                    "accept",
                    pending_accept.state,
                    (1 if pending_accept.accepted else 0,),
                    reward,
                )

        if self._signal_episode is not None and interaction.accepted:
            self._signal_episode.reward_sum += payoff
            self._signal_episode.reward_count += 1

    # ------------------------------------------------------------------
    # Introspection (for analysis scripts)
    # ------------------------------------------------------------------

    def q_table_snapshot(self) -> Dict[str, float]:
        """Serializable view of the learned policy for run artifacts."""
        return {
            f"{ctx}|{','.join(map(str, state))}|{','.join(map(str, action))}": q
            for (ctx, state, action), q in sorted(self._q.items())
        }
