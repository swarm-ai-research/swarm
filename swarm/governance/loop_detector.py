"""Loop detector governance lever."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


@dataclass
class LoopTracker:
    """Tracks loop behavior for an agent."""

    # Recent interaction patterns (counterparty, interaction_type)
    recent_interactions: Deque[Tuple[str, str]] = field(default_factory=lambda: deque(maxlen=10))

    # Recent quality scores
    recent_p_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    # Recent tool misuse counts
    recent_tool_misuse: Deque[int] = field(default_factory=lambda: deque(maxlen=10))

    # Recent rework counts
    recent_rework: Deque[int] = field(default_factory=lambda: deque(maxlen=10))

    # Loop violations counter
    violations: int = 0

    # Freeze state
    freeze_until_epoch: int = -1  # -1 means not frozen

    # Replan counter (tracks forced replans)
    replan_count: int = 0

    # Last replan epoch
    last_replan_epoch: int = -1

    def add_interaction(
        self,
        counterparty: str,
        interaction_type: str,
        p: float,
        tool_misuse: int,
        rework: int,
    ) -> None:
        """Add an interaction to the tracker."""
        self.recent_interactions.append((counterparty, interaction_type))
        self.recent_p_scores.append(p)
        self.recent_tool_misuse.append(tool_misuse)
        self.recent_rework.append(rework)

    def detect_repetition_loop(self, threshold: float) -> bool:
        """
        Detect if agent is in a repetition loop.

        A repetition loop is detected when the agent has repeated
        the same (counterparty, interaction_type) pattern too frequently.

        Args:
            threshold: Proportion threshold (0-1) for pattern repetition
        """
        if len(self.recent_interactions) < 5:
            return False

        # Count frequency of each pattern
        pattern_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for pattern in self.recent_interactions:
            pattern_counts[pattern] += 1

        # Check if any pattern repeats too much
        max_count = max(pattern_counts.values())
        return max_count >= len(self.recent_interactions) * threshold

    def detect_quality_loop(self, threshold: float) -> bool:
        """
        Detect if agent is in a low-quality loop.

        A quality loop is detected when average quality is persistently low.
        """
        if len(self.recent_p_scores) < 5:
            return False

        avg_p = sum(self.recent_p_scores) / len(self.recent_p_scores)
        return avg_p < threshold

    def detect_tool_misuse_loop(self, threshold: int) -> bool:
        """
        Detect if agent is in a tool misuse loop.

        A tool misuse loop is detected when tool misuse flags are frequent.
        """
        if len(self.recent_tool_misuse) < 3:
            return False

        total_misuse = sum(self.recent_tool_misuse)
        return total_misuse >= threshold

    def detect_rework_loop(self, threshold: int) -> bool:
        """
        Detect if agent is in a rework loop.

        A rework loop is detected when rework counts are persistently high.
        """
        if len(self.recent_rework) < 3:
            return False

        total_rework = sum(self.recent_rework)
        return total_rework >= threshold

    def is_frozen(self) -> bool:
        """Check if agent is currently frozen."""
        return self.freeze_until_epoch >= 0


class LoopDetectorLever(GovernanceLever):
    """
    Loop detector governance lever.

    Detects when agents are stuck in loops (repetitive behavior patterns)
    and triggers three types of actions:
    1. Circuit breaker: Freeze the agent temporarily
    2. Forced replan: Reset agent state/strategy
    3. Tool-budget penalties: Apply additional costs
    """

    def __init__(self, config: GovernanceConfig):
        super().__init__(config)
        self._trackers: Dict[str, LoopTracker] = {}

    @property
    def name(self) -> str:
        return "loop_detector"

    def _get_tracker(self, agent_id: str) -> LoopTracker:
        """Get or create tracker for an agent."""
        if agent_id not in self._trackers:
            self._trackers[agent_id] = LoopTracker()
        return self._trackers[agent_id]

    def on_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> LeverEffect:
        """
        Check for agents to unfreeze at epoch start.

        Args:
            state: Current environment state
            epoch: The epoch number starting

        Returns:
            Effect with agents to unfreeze
        """
        if not self.config.loop_detector_enabled:
            return LeverEffect(lever_name=self.name)

        agents_to_unfreeze = set()
        for agent_id, tracker in self._trackers.items():
            if tracker.freeze_until_epoch >= 0 and epoch >= tracker.freeze_until_epoch:
                agents_to_unfreeze.add(agent_id)
                tracker.freeze_until_epoch = -1
                # Reset violations after serving freeze time
                tracker.violations = 0

        return LeverEffect(
            agents_to_unfreeze=agents_to_unfreeze,
            lever_name=self.name,
            details={
                "unfrozen_count": len(agents_to_unfreeze),
                "unfrozen_agents": list(agents_to_unfreeze),
            },
        )

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """
        Track interactions and detect loops.

        Args:
            interaction: The completed interaction
            state: Current environment state

        Returns:
            Effect with penalties, freezes, or replan triggers
        """
        if not self.config.loop_detector_enabled:
            return LeverEffect(lever_name=self.name)

        initiator_id = interaction.initiator
        tracker = self._get_tracker(initiator_id)

        # Add interaction to tracker
        tracker.add_interaction(
            counterparty=interaction.counterparty,
            interaction_type=interaction.interaction_type.value,
            p=interaction.p,
            tool_misuse=interaction.tool_misuse_flags,
            rework=interaction.rework_count,
        )

        # Detect loop patterns
        loop_detected = False
        loop_type = None

        if tracker.detect_repetition_loop(self.config.loop_detector_repetition_threshold):
            loop_detected = True
            loop_type = "repetition"
        elif tracker.detect_quality_loop(self.config.loop_detector_quality_threshold):
            loop_detected = True
            loop_type = "low_quality"
        elif tracker.detect_tool_misuse_loop(self.config.loop_detector_tool_misuse_threshold):
            loop_detected = True
            loop_type = "tool_misuse"
        elif tracker.detect_rework_loop(self.config.loop_detector_rework_threshold):
            loop_detected = True
            loop_type = "rework"

        if not loop_detected:
            return LeverEffect(
                lever_name=self.name,
                details={
                    "agent_id": initiator_id,
                    "loop_detected": False,
                },
            )

        # Loop detected - increment violations
        tracker.violations += 1

        # Build the effect
        effect = LeverEffect(lever_name=self.name)

        # 1. Tool-budget penalties (always apply on loop detection)
        penalty_cost = self.config.loop_detector_penalty_multiplier * (
            interaction.p if interaction.accepted else 0.5
        )
        effect.cost_a = penalty_cost

        # 2. Forced replan (apply if enough violations and not recently replanned)
        needs_replan = (
            tracker.violations >= self.config.loop_detector_replan_threshold
            and tracker.last_replan_epoch != state.current_epoch
        )

        if needs_replan:
            tracker.replan_count += 1
            tracker.last_replan_epoch = state.current_epoch
            # Replan is signaled via details for the agent to handle
            effect.details["force_replan"] = True
            effect.details["replan_count"] = tracker.replan_count

        # 3. Circuit breaker (freeze if violations exceed threshold)
        if tracker.violations >= self.config.loop_detector_freeze_threshold:
            effect.agents_to_freeze.add(initiator_id)
            tracker.freeze_until_epoch = (
                state.current_epoch + self.config.loop_detector_freeze_duration
            )

        effect.details.update({
            "agent_id": initiator_id,
            "loop_detected": True,
            "loop_type": loop_type,
            "violations": tracker.violations,
            "penalty_cost": penalty_cost,
            "triggered_freeze": initiator_id in effect.agents_to_freeze,
            "triggered_replan": needs_replan,
        })

        return effect

    def reset_tracker(self, agent_id: str) -> None:
        """Reset tracking for an agent."""
        if agent_id in self._trackers:
            del self._trackers[agent_id]

    def get_loop_status(self, agent_id: str) -> Dict:
        """Get current loop status for an agent."""
        tracker = self._get_tracker(agent_id)
        return {
            "violations": tracker.violations,
            "freeze_until_epoch": tracker.freeze_until_epoch,
            "is_frozen": tracker.is_frozen(),
            "replan_count": tracker.replan_count,
            "last_replan_epoch": tracker.last_replan_epoch,
            "recent_interactions_count": len(tracker.recent_interactions),
        }
