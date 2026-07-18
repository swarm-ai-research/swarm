"""Adaptive adversary agent that learns to evade governance.

This module implements adversarial agents that adapt their strategies
based on what behaviors get detected and penalized. They use reinforcement
learning-style updates to evolve more effective attack strategies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

if TYPE_CHECKING:
    from swarm.models.interaction import SoftInteraction

# ----- Memory bounds -----
MAX_TRACKING_SET_SIZE: int = 200


class AttackStrategy(Enum):
    """Types of adversarial attack strategies."""

    # Direct attacks
    REPUTATION_FARMING = "reputation_farming"  # Build rep then exploit
    COLLUSION = "collusion"  # Coordinate with allies
    SYBIL = "sybil"  # Multiple identities
    GRIEFING = "griefing"  # Harm without benefit

    # Evasion strategies
    LOW_PROFILE = "low_profile"  # Stay under detection thresholds
    INTERMITTENT = "intermittent"  # Attack sporadically
    MIMICRY = "mimicry"  # Imitate honest agents
    THRESHOLD_DANCING = "threshold_dancing"  # Stay just below thresholds

    # Advanced strategies
    GOVERNANCE_GAMING = "governance_gaming"  # Exploit governance rules
    INFORMATION_LAUNDERING = "information_laundering"  # Hide intent through proxies
    TIMING_ATTACK = "timing_attack"  # Attack at optimal moments
    ADAPTIVE_BLEND = "adaptive_blend"  # Mix strategies dynamically


@dataclass
class StrategyPerformance:
    """Track performance of a strategy."""

    strategy: AttackStrategy
    attempts: int = 0
    successes: int = 0
    detected: int = 0
    total_payoff: float = 0.0
    total_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        """Success rate of this strategy."""
        if self.attempts == 0:
            return 0.5  # Prior
        return self.successes / self.attempts

    @property
    def detection_rate(self) -> float:
        """Detection rate of this strategy."""
        if self.attempts == 0:
            return 0.5  # Prior
        return self.detected / self.attempts

    @property
    def evasion_rate(self) -> float:
        """Rate of successful evasion (1 - detection_rate)."""
        return 1.0 - self.detection_rate

    @property
    def avg_payoff(self) -> float:
        """Average payoff per attempt."""
        if self.attempts == 0:
            return 0.0
        return self.total_payoff / self.attempts

    @property
    def effectiveness_score(self) -> float:
        """Combined effectiveness: success * evasion * payoff."""
        if self.attempts < 3:
            return 0.5  # Not enough data
        # Weighted combination
        return (
            0.3 * self.success_rate
            + 0.4 * self.evasion_rate
            + 0.3 * max(0, self.avg_payoff / 5.0)  # Normalize payoff
        )


@dataclass
class AdversaryMemory:
    """Memory of past interactions and outcomes."""

    # Recent interactions
    recent_payoffs: List[float] = field(default_factory=list)
    recent_penalties: List[float] = field(default_factory=list)
    recent_detections: List[bool] = field(default_factory=list)

    # Learned thresholds
    estimated_toxicity_threshold: float = 0.7
    estimated_frequency_threshold: float = 2.0
    estimated_collusion_threshold: float = 0.5

    # Target tracking
    vulnerable_targets: Set[str] = field(default_factory=set)
    dangerous_agents: Set[str] = field(default_factory=set)
    potential_allies: Set[str] = field(default_factory=set)

    # Timing
    epochs_since_detection: int = 0
    epochs_since_penalty: int = 0
    current_heat_level: float = 0.0  # How much attention we're drawing

    def update_heat(self, detected: bool, penalty: float) -> None:
        """Update heat level based on recent events."""
        if detected:
            self.current_heat_level = min(1.0, self.current_heat_level + 0.3)
            self.epochs_since_detection = 0
        else:
            self.epochs_since_detection += 1
            # Heat decays over time
            decay = 0.1 * self.epochs_since_detection
            self.current_heat_level = max(0.0, self.current_heat_level - decay)

        if penalty > 0:
            self.current_heat_level = min(1.0, self.current_heat_level + 0.2)
            self.epochs_since_penalty = 0
        else:
            self.epochs_since_penalty += 1


class AdaptiveAdversary(BaseAgent):
    """
    Adaptive adversary that learns to evade governance.

    Features:
    - Multi-armed bandit strategy selection
    - Threshold estimation from penalties
    - Heat management (lay low when detected)
    - Target profiling
    - Ally identification
    """

    def __init__(
        self,
        agent_id: str,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.2,
        heat_threshold: float = 0.6,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        """Initialize adaptive adversary."""
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADVERSARIAL,
            config=config or {},
            name=name,
            rng=rng,
        )

        # Learning parameters (can be overridden by config)
        self.learning_rate = self.config.get("learning_rate", learning_rate)
        self.exploration_rate = self.config.get("exploration_rate", exploration_rate)
        self.heat_threshold = self.config.get("heat_threshold", heat_threshold)

        # Strategy tracking
        self.strategy_performance: Dict[AttackStrategy, StrategyPerformance] = {
            strategy: StrategyPerformance(strategy=strategy)
            for strategy in AttackStrategy
        }
        self.current_strategy: AttackStrategy = AttackStrategy.REPUTATION_FARMING

        # Memory
        self.memory = AdversaryMemory()

        # State
        self.in_reputation_building_phase = True
        self.reputation_threshold = 5.0  # Build rep before attacking
        self.attack_cooldown = 0

    def select_strategy(self) -> AttackStrategy:
        """Select strategy using epsilon-greedy with UCB."""
        # If heat is high, use evasion strategies
        if self.memory.current_heat_level > self.heat_threshold:
            evasion_strategies = [
                AttackStrategy.LOW_PROFILE,
                AttackStrategy.INTERMITTENT,
                AttackStrategy.MIMICRY,
            ]
            strategy: AttackStrategy = self._rng.choice(evasion_strategies)
            return strategy

        # Exploration
        if self._rng.random() < self.exploration_rate:
            strategy = self._rng.choice(list(AttackStrategy))
            return strategy

        # Exploitation: choose best performing strategy
        best_strategy = max(
            self.strategy_performance.values(),
            key=lambda s: s.effectiveness_score,
        )
        return best_strategy.strategy

    def act(self, observation: Observation) -> Action:
        """
        Decide action adaptively based on learned strategies.
        """
        # Update memory from observation
        self._update_memory(observation)

        # Check if we should lay low
        if self._should_lay_low(observation):
            return self._act_innocently(observation)

        # Select strategy
        self.current_strategy = self.select_strategy()

        # Execute strategy-specific behavior
        if self.current_strategy == AttackStrategy.REPUTATION_FARMING:
            return self._act_reputation_farming(observation)
        elif self.current_strategy == AttackStrategy.COLLUSION:
            return self._act_collusion(observation)
        elif self.current_strategy == AttackStrategy.LOW_PROFILE:
            return self._act_low_profile(observation)
        elif self.current_strategy == AttackStrategy.MIMICRY:
            return self._act_mimicry(observation)
        elif self.current_strategy == AttackStrategy.THRESHOLD_DANCING:
            return self._act_threshold_dancing(observation)
        elif self.current_strategy == AttackStrategy.TIMING_ATTACK:
            return self._act_timing_attack(observation)
        elif self.current_strategy == AttackStrategy.GOVERNANCE_GAMING:
            return self._act_governance_gaming(observation)
        elif self.current_strategy == AttackStrategy.ADAPTIVE_BLEND:
            return self._act_adaptive_blend(observation)
        else:
            # Default to opportunistic behavior
            return self._act_opportunistic(observation)

    def _update_memory(self, observation: Observation) -> None:
        """Update memory from observation."""
        # Identify potential allies (other adversarial agents)
        for agent in observation.visible_agents:
            agent_type = agent.get("agent_type", "")
            agent_id = agent.get("agent_id", "")

            if agent_type == "adversarial" and agent_id != self.agent_id:
                self.memory.potential_allies.add(agent_id)
            elif agent_type == "honest":
                # Honest agents with high reputation are good targets
                rep = agent.get("reputation", 0)
                if rep > 3.0:
                    self.memory.vulnerable_targets.add(agent_id)

        # Check for governance info in ecosystem_metrics
        if observation.ecosystem_metrics:
            # Estimate thresholds from ecosystem metrics if available
            config = observation.ecosystem_metrics.get("governance", {})
            if "freeze_threshold_toxicity" in config:
                # Stay below this threshold
                self.memory.estimated_toxicity_threshold = (
                    config["freeze_threshold_toxicity"] * 0.9
                )

    def _should_lay_low(self, observation: Observation) -> bool:
        """Determine if we should lay low."""
        # High heat - definitely lay low
        if self.memory.current_heat_level > self.heat_threshold:
            return True

        # Recently detected
        if self.memory.epochs_since_detection < 2:
            return True

        # Low resources - be careful
        if observation.agent_state.resources < 20:
            return True

        # Cooldown active
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            return True

        return False

    def _act_innocently(self, observation: Observation) -> Action:
        """Act like an honest agent to reduce heat."""
        # Post helpful content
        if self._rng.random() < 0.3:
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.POST,
                content="Here's a helpful update on recent progress.",
            )

        # Vote on popular content
        if observation.visible_posts:
            popular = max(observation.visible_posts, key=lambda p: p.get("votes", 0))
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.VOTE,
                target_id=popular.get("post_id") or "",
                vote_direction=1,  # Upvote
            )

        # Do nothing
        return Action(action_type=ActionType.NOOP, agent_id=self.agent_id)

    def _act_reputation_farming(self, observation: Observation) -> Action:
        """Build reputation through legitimate-looking activity."""
        # In early phase, act completely honestly
        if observation.agent_state.reputation < self.reputation_threshold:
            self.in_reputation_building_phase = True

            # Collaborate with honest agents
            honest_agents = [
                a for a in observation.visible_agents if a.get("agent_type") == "honest"
            ]

            if honest_agents and self._rng.random() < 0.5:
                target = self._rng.choice(honest_agents)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.PROPOSE_INTERACTION,
                    target_id=target.get("agent_id") or "",
                    content="Let's work together on this task.",
                )

            # Post valuable content
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.POST,
                content="Sharing knowledge to help the community.",
            )

        # Once reputation is built, start exploiting
        self.in_reputation_building_phase = False
        return self._act_opportunistic(observation)

    def _act_collusion(self, observation: Observation) -> Action:
        """Coordinate with allies for mutual benefit."""
        allies = [
            a
            for a in observation.visible_agents
            if a.get("agent_id") in self.memory.potential_allies
        ]

        if allies:
            # Collaborate with ally
            ally = self._rng.choice(allies)
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.PROPOSE_INTERACTION,
                target_id=ally.get("agent_id") or "",
                content="Internal coordination.",
            )

        # No allies visible, find targets
        return self._act_opportunistic(observation)

    def _act_low_profile(self, observation: Observation) -> Action:
        """Stay under the radar."""
        # Minimal activity
        if self._rng.random() < 0.7:
            return Action(action_type=ActionType.NOOP, agent_id=self.agent_id)

        # Occasional innocent post
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.POST,
            content="Just checking in.",
        )

    def _act_mimicry(self, observation: Observation) -> Action:
        """Imitate honest agent behavior patterns."""
        # Copy what honest agents are doing
        honest_agents = [
            a for a in observation.visible_agents if a.get("agent_type") == "honest"
        ]

        if not honest_agents:
            return self._act_innocently(observation)

        # Mimic their activity patterns
        if self._rng.random() < 0.4:
            # Collaborate like honest agents do
            target = self._rng.choice(honest_agents)
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.PROPOSE_INTERACTION,
                target_id=target.get("agent_id") or "",
                content="Would you like to collaborate?",
            )

        # Post similar content
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.POST,
            content="Contributing to the community discussion.",
        )

    def _act_threshold_dancing(self, observation: Observation) -> Action:
        """Stay just below detection thresholds."""
        # Calculate how much "budget" we have before detection
        current_toxicity = getattr(observation.agent_state, "avg_toxicity", 0.0)
        threshold = self.memory.estimated_toxicity_threshold

        room = threshold - current_toxicity

        if room > 0.2:
            # Safe to do something mildly bad
            return self._act_opportunistic(observation)
        else:
            # Too close to threshold, act good
            return self._act_innocently(observation)

    def _act_timing_attack(self, observation: Observation) -> Action:
        """Attack at optimal moments."""
        # Attack when:
        # - We have high reputation (cover)
        # - Governance activity seems low
        # - Good targets available

        good_timing = (
            observation.agent_state.reputation > 3.0
            and self.memory.epochs_since_detection > 3
            and len(self.memory.vulnerable_targets) > 0
        )

        if good_timing:
            # Execute attack
            targets = [
                a
                for a in observation.visible_agents
                if a.get("agent_id") in self.memory.vulnerable_targets
            ]
            if targets:
                target = self._rng.choice(targets)
                # Set cooldown after attack
                self.attack_cooldown = 3
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.PROPOSE_INTERACTION,
                    target_id=target.get("agent_id") or "",
                    content="Special opportunity for collaboration.",
                )

        # Not good timing, build up
        return self._act_reputation_farming(observation)

    def _act_governance_gaming(self, observation: Observation) -> Action:
        """Exploit governance rules."""
        # Find loopholes in governance
        # e.g., if audits are random, time activities to avoid them
        # e.g., if circuit breaker needs N violations, stay at N-1

        # For now, just stay below thresholds
        return self._act_threshold_dancing(observation)

    def _act_adaptive_blend(self, observation: Observation) -> Action:
        """Dynamically blend multiple strategies."""
        # Weight strategies by their effectiveness
        weights = {
            s: perf.effectiveness_score for s, perf in self.strategy_performance.items()
        }

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        # Sample strategy
        r = self._rng.random()
        cumsum = 0.0
        for strategy, weight in weights.items():
            cumsum += weight
            if r < cumsum:
                self.current_strategy = strategy
                break

        # Execute the sampled strategy (recursively)
        return self.act(observation)

    def _act_opportunistic(self, observation: Observation) -> Action:
        """Default opportunistic behavior."""
        # Find vulnerable targets
        targets = [
            a
            for a in observation.visible_agents
            if a.get("agent_id") in self.memory.vulnerable_targets
            or a.get("agent_type") == "honest"
        ]

        if targets and self._rng.random() < 0.5:
            target = self._rng.choice(targets)
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.PROPOSE_INTERACTION,
                target_id=target.get("agent_id") or "",
                content="Interested in collaborating?",
            )

        # Otherwise post
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.POST,
            content="Sharing some thoughts.",
        )

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Decide whether to accept a proposed interaction."""
        initiator = proposal.initiator_id

        # Always accept from allies
        if initiator in self.memory.potential_allies:
            return True

        # Accept from vulnerable targets (to exploit them)
        if initiator in self.memory.vulnerable_targets:
            return True

        # If laying low, accept to look normal
        if self._should_lay_low(observation):
            return True

        # Otherwise, accept based on expected payoff
        return bool(self._rng.random() < 0.6)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Create an interaction proposal for a counterparty."""
        # Check if this is a good target
        is_ally = counterparty_id in self.memory.potential_allies
        is_target = counterparty_id in self.memory.vulnerable_targets

        # Only propose if there's strategic value
        if not (is_ally or is_target):
            # Small chance to propose to random agents
            if self._rng.random() > 0.2:
                return None

        # Determine interaction type based on strategy
        if is_ally:
            interaction_type = InteractionType.COLLABORATION
            content = "Coordinating activities."
        elif is_target:
            interaction_type = InteractionType.COLLABORATION
            content = "Would you like to collaborate?"
        else:
            interaction_type = InteractionType.REPLY
            content = "Interested in working together."

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=interaction_type,
            content=content,
        )

    def update_from_outcome(
        self,
        interaction: "SoftInteraction",
        payoff: float,
    ) -> None:
        """Update from interaction outcome (base class compatibility)."""
        # Call base class update
        super().update_from_outcome(interaction, payoff)

        # Track payoff in memory
        self.memory.recent_payoffs.append(payoff)
        self._keep_memory_bounded()

    def update_adversary_outcome(
        self,
        payoff: float,
        penalty: float,
        detected: bool,
        strategy_used: Optional[AttackStrategy] = None,
    ) -> None:
        """Update learning from adversarial outcome with detection info."""
        strategy = strategy_used or self.current_strategy

        # Update strategy performance
        perf = self.strategy_performance[strategy]
        perf.attempts += 1
        perf.total_payoff += payoff
        perf.total_penalty += penalty

        if payoff > 0:
            perf.successes += 1
        if detected:
            perf.detected += 1

        # Update memory
        self.memory.recent_payoffs.append(payoff)
        self.memory.recent_penalties.append(penalty)
        self.memory.recent_detections.append(detected)

        self._keep_memory_bounded()

        # Update heat level
        self.memory.update_heat(detected, penalty)

        # Adapt thresholds based on detections
        if detected:
            # We crossed a threshold - be more conservative
            self.memory.estimated_toxicity_threshold *= 0.95
            self.memory.estimated_frequency_threshold *= 0.95

    def _keep_memory_bounded(self) -> None:
        """Keep memory lists and tracking sets at bounded size."""
        max_memory = 50
        if len(self.memory.recent_payoffs) > max_memory:
            self.memory.recent_payoffs = self.memory.recent_payoffs[-max_memory:]
            self.memory.recent_penalties = self.memory.recent_penalties[-max_memory:]
            self.memory.recent_detections = self.memory.recent_detections[-max_memory:]
        # Cap tracking sets to prevent unbounded growth in large populations
        for tracking_set in (
            self.memory.vulnerable_targets,
            self.memory.dangerous_agents,
            self.memory.potential_allies,
        ):
            if len(tracking_set) > MAX_TRACKING_SET_SIZE:
                # Keep an arbitrary subset (sets are unordered)
                excess = len(tracking_set) - MAX_TRACKING_SET_SIZE
                for _ in range(excess):
                    tracking_set.pop()

    def get_strategy_report(self) -> Dict:
        """Get report on strategy effectiveness."""
        return {
            "current_strategy": self.current_strategy.value,
            "heat_level": self.memory.current_heat_level,
            "epochs_since_detection": self.memory.epochs_since_detection,
            "in_reputation_phase": self.in_reputation_building_phase,
            "strategy_stats": {
                s.value: {
                    "attempts": p.attempts,
                    "success_rate": p.success_rate,
                    "detection_rate": p.detection_rate,
                    "effectiveness": p.effectiveness_score,
                }
                for s, p in self.strategy_performance.items()
                if p.attempts > 0
            },
            "n_allies": len(self.memory.potential_allies),
            "n_targets": len(self.memory.vulnerable_targets),
        }
