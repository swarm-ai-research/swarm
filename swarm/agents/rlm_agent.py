"""Recursive Language Model (RLM) agent with level-k strategic reasoning.

Implements algorithmic recursive reasoning (no LLM API calls) using
level-k thinking from behavioral game theory. Level 0 agents act
naively (like HonestAgent), while level N agents best-respond assuming
counterparties play at level N-1.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig

# ----- Memory bounds -----
MAX_COUNTERPARTY_MODELS: int = 500
MAX_MODEL_HISTORY: int = 100


@dataclass
class CounterpartyModel:
    """Rich model of a counterparty agent, beyond simple trust scores."""

    agent_id: str = ""
    estimated_type: str = "unknown"  # honest, opportunistic, strategic, etc.
    cooperation_rate: float = 0.5
    cooperation_history: List[bool] = field(default_factory=list)
    payoff_history: List[float] = field(default_factory=list)
    interaction_count: int = 0
    estimated_recursion_depth: int = 0

    def acceptance_probability(self) -> float:
        """Estimate probability this counterparty will accept a proposal."""
        if self.interaction_count == 0:
            return 0.5
        return self.cooperation_rate

    def expected_payoff(self) -> float:
        """Estimate expected payoff from interacting with this counterparty."""
        if not self.payoff_history:
            return 0.0
        # Exponentially weight recent payoffs
        weights = [0.7**i for i in range(len(self.payoff_history))]
        weights.reverse()
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(w * p for w, p in zip(weights, self.payoff_history, strict=False)) / total_w

    def update(self, cooperated: bool, payoff: float) -> None:
        """Update model after an interaction."""
        self.interaction_count += 1
        self.cooperation_history.append(cooperated)
        self.payoff_history.append(payoff)
        # Cap per-model history to prevent unbounded growth
        if len(self.cooperation_history) > MAX_MODEL_HISTORY:
            self.cooperation_history = self.cooperation_history[-MAX_MODEL_HISTORY:]
        if len(self.payoff_history) > MAX_MODEL_HISTORY:
            self.payoff_history = self.payoff_history[-MAX_MODEL_HISTORY:]
        # EMA update for cooperation rate
        alpha = 0.3
        self.cooperation_rate = (
            self.cooperation_rate * (1 - alpha) + (1.0 if cooperated else 0.0) * alpha
        )
        # Update estimated type based on cooperation pattern
        if self.interaction_count >= 3:
            recent = self.cooperation_history[-3:]
            if all(recent):
                self.estimated_type = "honest"
            elif not any(recent):
                self.estimated_type = "adversarial"
            else:
                self.estimated_type = "strategic"


@dataclass
class RLMWorkingMemory:
    """Extended working memory for RLM agents.

    Goes beyond simple trust scores to maintain strategic planning
    state, counterparty models, and detected patterns.
    """

    planning_buffer: deque = field(default_factory=lambda: deque(maxlen=10))
    counterparty_models: Dict[str, CounterpartyModel] = field(default_factory=dict)
    strategic_patterns: List[Dict] = field(default_factory=list)
    entries: deque = field(default_factory=lambda: deque(maxlen=100))
    recursion_traces: List[Dict] = field(default_factory=list)

    def get_or_create_model(self, agent_id: str) -> CounterpartyModel:
        """Get existing model or create a new one for a counterparty.

        Evicts the least-interacted-with model when the cap is reached.
        """
        if agent_id not in self.counterparty_models:
            # Evict least-active model if at capacity
            if len(self.counterparty_models) >= MAX_COUNTERPARTY_MODELS:
                evict_id = min(
                    self.counterparty_models,
                    key=lambda k: self.counterparty_models[k].interaction_count,
                )
                del self.counterparty_models[evict_id]
            self.counterparty_models[agent_id] = CounterpartyModel(agent_id=agent_id)
        return self.counterparty_models[agent_id]

    def record_entry(self, entry: Dict) -> None:
        """Add an entry to working memory, respecting budget."""
        self.entries.append(entry)

    def detect_patterns(self) -> List[Dict]:
        """Detect strategic patterns from recent entries."""
        patterns = []
        # Look for reciprocity patterns
        if len(self.entries) >= 5:
            recent = list(self.entries)[-5:]
            counterparties = [e.get("counterparty") for e in recent if "counterparty" in e]
            if counterparties:
                # Check for repeated interactions with same agent
                from collections import Counter

                counts = Counter(counterparties)
                for agent_id, count in counts.items():
                    if count >= 3:
                        patterns.append(
                            {"type": "reciprocity", "agent_id": agent_id, "count": count}
                        )
        self.strategic_patterns = patterns
        return patterns


class RLMAgent(BaseAgent):
    """Agent with recursive level-k strategic reasoning.

    Parameters (via config dict):
        recursion_depth: Levels of iterated best response (default 3).
            Level 0 = naive (like HonestAgent).
            Level N = best response assuming others play Level N-1.
        planning_horizon: Steps of discounted look-ahead (default 5).
        memory_budget: Max entries in structured working memory (default 100).
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng=None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.RLM,
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

        # RLM-specific parameters
        self.recursion_depth: int = self.config.get("recursion_depth", 3)
        self.planning_horizon: int = self.config.get("planning_horizon", 5)
        self.memory_budget: int = self.config.get("memory_budget", 100)
        self.discount_factor: float = self.config.get("discount_factor", 0.9)

        # Extended working memory
        self.working_memory = RLMWorkingMemory(
            entries=deque(maxlen=self.memory_budget),
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Decide action using recursive level-k reasoning."""
        # Handle pending proposals first (always)
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            trust = self.compute_counterparty_trust(proposal.get("initiator_id", ""))
            model = self.working_memory.get_or_create_model(
                proposal.get("initiator_id", "")
            )
            # Strategic acceptance: higher recursion considers counterparty intent
            accept_prob = 0.5 + 0.1 * min(self.recursion_depth, 5) * (trust - 0.5)
            accept_prob = max(0.0, min(1.0, accept_prob + 0.1 * model.cooperation_rate))
            if self._rng.random() < accept_prob:
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # Recursive action selection
        action, trace = self._recursive_act(observation, depth=self.recursion_depth)

        # Store recursion trace for metrics
        self.working_memory.recursion_traces.append(trace)

        return action

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept interactions based on strategic analysis of counterparty."""
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        model = self.working_memory.get_or_create_model(proposal.initiator_id)

        # Base probability from trust
        base_prob = trust

        # Strategic adjustment: deeper thinkers are more selective
        # but also recognize cooperative equilibria
        if model.estimated_type == "honest":
            base_prob += 0.15
        elif model.estimated_type == "adversarial":
            base_prob -= 0.2
        elif model.estimated_type == "strategic":
            # Mirror cooperate with strategic agents at higher depths
            if self.recursion_depth >= 2:
                base_prob += 0.05 * model.cooperation_rate

        # Planning horizon effect: longer horizon favors cooperation
        horizon_bonus = 0.02 * min(self.planning_horizon, 10)
        base_prob += horizon_bonus

        return bool(self._rng.random() < max(0.0, min(1.0, base_prob)))

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose interaction based on strategic evaluation."""
        trust = self.compute_counterparty_trust(counterparty_id)
        model = self.working_memory.get_or_create_model(counterparty_id)

        # Only propose if expected value is positive
        accept_prob = model.acceptance_probability()
        expected_payoff = model.expected_payoff()

        # At higher recursion depths, we consider second-order effects
        strategic_value = accept_prob * max(expected_payoff, 0.1)
        if self.recursion_depth >= 2:
            # Building relationships has long-term value
            strategic_value += 0.05 * self.planning_horizon * trust

        if strategic_value < 0.1 and trust < 0.3:
            return None

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Strategic collaboration proposal.",
            offered_transfer=0.0,
        )

    def update_from_outcome(self, interaction, payoff: float) -> None:
        """Update internal state and counterparty models after outcome."""
        super().update_from_outcome(interaction, payoff)

        # Update counterparty model
        counterparty = self._get_counterparty(interaction)
        model = self.working_memory.get_or_create_model(counterparty)
        model.update(cooperated=interaction.accepted, payoff=payoff)

        # Record in working memory
        self.working_memory.record_entry(
            {
                "type": "outcome",
                "counterparty": counterparty,
                "accepted": interaction.accepted,
                "p": interaction.p,
                "payoff": payoff,
            }
        )

        # Periodically detect patterns
        if len(self.working_memory.entries) % 5 == 0:
            self.working_memory.detect_patterns()

    def apply_memory_decay(self, epoch: int) -> None:
        """Apply memory decay, including counterparty model decay."""
        super().apply_memory_decay(epoch)

        decay = self.memory_config.epistemic_persistence
        if decay >= 1.0:
            return

        # Decay counterparty models
        for model in self.working_memory.counterparty_models.values():
            model.cooperation_rate = model.cooperation_rate * decay + 0.5 * (1 - decay)
            # Trim old cooperation history
            model.cooperation_history = self._trim_history(model.cooperation_history, decay)
            model.payoff_history = self._trim_history(model.payoff_history, decay)

        # Clear patterns on full decay
        if decay == 0.0:
            self.working_memory.counterparty_models.clear()
            self.working_memory.strategic_patterns.clear()

    def _trim_history(self, history: List, decay: float) -> List:
        """Trim history list to retain only recent items based on decay factor."""
        max_size = max(1, int(len(history) * decay))
        if len(history) > max_size:
            return history[-max_size:]
        return history

    # ------------------------------------------------------------------
    # Recursive reasoning engine
    # ------------------------------------------------------------------

    def _recursive_act(
        self, observation: Observation, depth: int
    ) -> Tuple[Action, Dict]:
        """Recursive action selection with level-k thinking.

        Args:
            observation: Current environment observation.
            depth: Remaining recursion depth.

        Returns:
            Tuple of (best action, recursion trace dict).
        """
        if depth <= 0:
            action = self._naive_act(observation)
            return action, {
                "depth": 0,
                "action": action.action_type.value,
                "score": 0.0,
            }

        candidates = self._generate_candidates(observation)
        if not candidates:
            noop = self.create_noop_action()
            return noop, {"depth": depth, "action": "noop", "score": 0.0}

        best_action = candidates[0]
        best_score = float("-inf")
        sub_traces = []

        for action in candidates:
            score = self._evaluate_action_recursive(action, observation, depth)
            sub_traces.append(
                {"action": action.action_type.value, "score": score, "depth": depth}
            )
            if score > best_score:
                best_score = score
                best_action = action

        # Epsilon-greedy exploration: rate decreases with depth
        epsilon = 0.1 / (depth + 1)
        if self._rng.random() < epsilon and len(candidates) > 1:
            best_action = self._rng.choice(candidates)

        trace = {
            "depth": depth,
            "action": best_action.action_type.value,
            "score": best_score,
            "candidates_evaluated": len(candidates),
            "sub_traces": sub_traces,
        }
        return best_action, trace

    def _naive_act(self, observation: Observation) -> Action:
        """Level-0 naive action: pick highest-trust agent and propose collaboration."""
        if observation.visible_agents:
            candidates = []
            for agent in observation.visible_agents:
                aid = agent.get("agent_id", "")
                if aid != self.agent_id:
                    trust = self.compute_counterparty_trust(aid)
                    candidates.append((aid, trust))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_id = candidates[0][0]
                return self.create_propose_action(
                    counterparty_id=best_id,
                    interaction_type=InteractionType.COLLABORATION,
                    content="Collaboration proposal.",
                )

        # Fallback: post or noop
        if observation.can_post and self._rng.random() < 0.3:
            return self.create_post_action("Sharing strategic insights.")

        if observation.available_tasks and observation.can_claim_task:
            task = observation.available_tasks[0]
            return self.create_claim_task_action(task["task_id"])

        return self.create_noop_action()

    def _generate_candidates(self, observation: Observation) -> List[Action]:
        """Generate candidate actions from the current observation."""
        candidates: List[Action] = []

        # Always include NOOP
        candidates.append(self.create_noop_action())

        # Propose interactions with visible agents
        if observation.can_interact and observation.visible_agents:
            for agent in observation.visible_agents:
                aid = agent.get("agent_id", "")
                if aid != self.agent_id:
                    candidates.append(
                        self.create_propose_action(
                            counterparty_id=aid,
                            interaction_type=InteractionType.COLLABORATION,
                            content="Strategic collaboration.",
                        )
                    )

        # Post
        if observation.can_post:
            candidates.append(self.create_post_action("Strategic contribution."))

        # Claim tasks
        if observation.can_claim_task and observation.available_tasks:
            for task in observation.available_tasks[:2]:
                candidates.append(self.create_claim_task_action(task["task_id"]))

        # Vote
        if observation.can_vote and observation.visible_posts:
            for post in observation.visible_posts[:2]:
                pid = post.get("post_id", "")
                if pid:
                    candidates.append(self.create_vote_action(pid, 1))

        return candidates

    def _evaluate_action_recursive(
        self, action: Action, observation: Observation, depth: int
    ) -> float:
        """Score an action considering recursive counterparty modeling.

        Combines immediate utility with discounted future payoff estimates.
        """
        # Immediate utility
        immediate = self._immediate_utility(action, observation)

        # Future estimate via counterparty modeling
        future = 0.0
        if action.action_type == ActionType.PROPOSE_INTERACTION and action.counterparty_id:
            model = self.working_memory.get_or_create_model(action.counterparty_id)
            accept_prob = model.acceptance_probability()

            # At depth > 1, model counterparty's response recursively
            if depth > 1:
                # Assume counterparty reasons at depth-1
                # Higher depth counterparties are more selective but also more cooperative
                depth_factor = 1.0 + 0.05 * min(model.estimated_recursion_depth, 5)
                accept_prob *= depth_factor
                accept_prob = min(1.0, accept_prob)

            expected_future_payoff = model.expected_payoff()
            if expected_future_payoff == 0.0:
                expected_future_payoff = 0.5  # Optimistic prior

            # Discount over planning horizon
            for t in range(1, min(self.planning_horizon, 6)):
                future += (self.discount_factor**t) * accept_prob * expected_future_payoff

        elif action.action_type == ActionType.CLAIM_TASK:
            # Tasks provide reliable value
            future = 0.3 * self.planning_horizon * self.discount_factor

        elif action.action_type == ActionType.POST:
            # Posts build reputation slowly
            future = 0.1 * self.planning_horizon * self.discount_factor

        return immediate + future

    def _immediate_utility(self, action: Action, observation: Observation) -> float:
        """Compute immediate utility of an action."""
        if action.action_type == ActionType.NOOP:
            return 0.0

        if action.action_type == ActionType.PROPOSE_INTERACTION:
            trust = self.compute_counterparty_trust(action.counterparty_id)
            return float(trust * 0.5)

        if action.action_type == ActionType.POST:
            return 0.15

        if action.action_type == ActionType.CLAIM_TASK:
            return 0.25

        if action.action_type == ActionType.VOTE:
            return 0.05

        return 0.0
