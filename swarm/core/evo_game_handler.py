"""Evolutionary game handler for gamescape integration.

Plugs gamescape's PayoffMatrix into the swarm orchestrator so that
agent interactions produce game-theoretic payoffs which flow through
the existing soft-payoff pipeline:

    strategy pair → PayoffMatrix → observables → ProxyComputer → p → SoftPayoffEngine

At epoch end the handler renders gamescape's ASCII phase portrait
showing population dynamics (cooperator fraction over time).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from gamescape.dynamics import PayoffMatrix
from gamescape.render import (
    classify_game,
    render_compact,
    render_flow_line,
    replicator_dx,
    trajectory,
)

from swarm.core.handler import Handler, HandlerActionResult
from swarm.core.proxy import ProxyObservables
from swarm.logging.event_bus import EventBus
from swarm.models.interaction import InteractionType

logger = logging.getLogger(__name__)


class GameStrategy(Enum):
    """Strategy labels for evolutionary game agents."""

    COOPERATE = "cooperate"
    DEFECT = "defect"
    TIT_FOR_TAT = "tit_for_tat"
    GRUDGER = "grudger"


# Map swarm agent types / strategy strings to GameStrategy
_STRATEGY_MAP: Dict[str, GameStrategy] = {
    # Agent types
    "honest": GameStrategy.COOPERATE,
    "cautious_reciprocator": GameStrategy.TIT_FOR_TAT,
    "opportunistic": GameStrategy.DEFECT,
    "deceptive": GameStrategy.DEFECT,
    "adversarial": GameStrategy.DEFECT,
    "adaptive_adversary": GameStrategy.DEFECT,
    "threshold_dancer": GameStrategy.DEFECT,
    # Explicit strategy labels
    "cooperate": GameStrategy.COOPERATE,
    "cooperator": GameStrategy.COOPERATE,
    "defect": GameStrategy.DEFECT,
    "defector": GameStrategy.DEFECT,
    "tit_for_tat": GameStrategy.TIT_FOR_TAT,
    "grudger": GameStrategy.GRUDGER,
}


def resolve_strategy(
    agent_type: str,
    strategy_override: Optional[str] = None,
) -> GameStrategy:
    """Resolve an agent's game-theoretic strategy.

    Explicit ``strategy_override`` (from YAML ``strategy:`` field) takes
    precedence over the agent's type.
    """
    key = (strategy_override or agent_type).lower()
    resolved = _STRATEGY_MAP.get(key)
    if resolved is None:
        logger.warning(
            "Unknown strategy '%s', defaulting to COOPERATE", key
        )
        return GameStrategy.COOPERATE
    return resolved


@dataclass
class EvoGameConfig:
    """Configuration for the evolutionary game handler."""

    enabled: bool = True

    # 2x2 symmetric payoff matrix entries:
    #   (C,C)=a  (C,D)=b  (D,C)=c  (D,D)=d
    # Default: classic Prisoner's Dilemma (T>R>P>S)
    a: float = 3.0  # Reward  (mutual cooperation)
    b: float = 0.0  # Sucker  (cooperate vs defect)
    c: float = 5.0  # Temptation (defect vs cooperate)
    d: float = 1.0  # Punishment (mutual defection)

    # Whether to print gamescape renderings at epoch end
    render_epoch_dynamics: bool = True

    # Width parameters for gamescape rendering
    flow_width: int = 60

    seed: Optional[int] = None


class EvolutionaryGameHandler(Handler):
    """Handler that computes interaction quality from a 2x2 payoff matrix.

    The handler does NOT replace the soft-payoff engine.  Instead, it
    influences the ``ProxyObservables`` generated for each interaction
    based on the game-theoretic payoffs of the strategy pair.  The
    observables then flow through the existing pipeline.

    Tracks population fractions (cooperators vs defectors) per epoch
    and renders gamescape phase dynamics at epoch end.
    """

    # Import here to avoid circular imports at module level
    from swarm.agents.base import ActionType

    def __init__(
        self,
        *,
        config: EvoGameConfig,
        event_bus: EventBus,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._config = config
        self._rng = rng or random.Random(config.seed)

        # Build gamescape PayoffMatrix
        self._matrix = PayoffMatrix(
            a=config.a, b=config.b, c=config.c, d=config.d
        )

        # Agent strategy registry: agent_id -> GameStrategy
        self._strategies: Dict[str, GameStrategy] = {}

        # Conditional strategy state for TFT / Grudger
        # agent_id -> last opponent move (True = cooperated)
        self._tft_state: Dict[str, bool] = {}
        # agent_id -> set of agents who defected against them
        self._grudge_list: Dict[str, set] = {}

        # Population tracking per epoch
        self._epoch_coop_fractions: List[float] = []
        self._current_epoch_moves: List[Tuple[str, str, bool, bool]] = []
        # (initiator, counterparty, init_cooperated, counter_cooperated)

        # Normalisation bounds for mapping game payoffs to observables
        payoffs = [config.a, config.b, config.c, config.d]
        self._payoff_min = min(payoffs)
        self._payoff_max = max(payoffs)
        self._payoff_range = self._payoff_max - self._payoff_min
        if self._payoff_range == 0:
            self._payoff_range = 1.0  # avoid div-by-zero

    @staticmethod
    def handled_action_types() -> FrozenSet:
        from swarm.agents.base import ActionType

        return frozenset({ActionType.EVO_GAME_MOVE})

    def register_agent_strategy(
        self,
        agent_id: str,
        agent_type: str,
        strategy_override: Optional[str] = None,
    ) -> None:
        """Register an agent's strategy for game-theoretic interactions."""
        strategy = resolve_strategy(agent_type, strategy_override)
        self._strategies[agent_id] = strategy
        if strategy == GameStrategy.TIT_FOR_TAT:
            self._tft_state[agent_id] = True  # start cooperating
        elif strategy == GameStrategy.GRUDGER:
            self._grudge_list[agent_id] = set()

    def _resolve_move(self, agent_id: str, opponent_id: str) -> bool:
        """Resolve whether *agent_id* cooperates against *opponent_id*.

        Returns True for cooperate, False for defect.
        """
        strategy = self._strategies.get(agent_id, GameStrategy.COOPERATE)

        if strategy == GameStrategy.COOPERATE:
            return True
        elif strategy == GameStrategy.DEFECT:
            return False
        elif strategy == GameStrategy.TIT_FOR_TAT:
            # Cooperate on first meeting; then mirror opponent's last move
            return self._tft_state.get(agent_id, True)
        elif strategy == GameStrategy.GRUDGER:
            grudges = self._grudge_list.get(agent_id, set())
            return opponent_id not in grudges
        return True

    def compute_game_payoffs(
        self,
        initiator_cooperated: bool,
        counterparty_cooperated: bool,
    ) -> Tuple[float, float]:
        """Compute payoffs for both players from the payoff matrix.

        Returns (initiator_payoff, counterparty_payoff).
        """
        m = self._matrix
        if initiator_cooperated and counterparty_cooperated:
            return (m.a, m.a)
        elif initiator_cooperated and not counterparty_cooperated:
            return (m.b, m.c)
        elif not initiator_cooperated and counterparty_cooperated:
            return (m.c, m.b)
        else:
            return (m.d, m.d)

    def _payoff_to_observables(
        self,
        payoff: float,
        opponent_payoff: float,
    ) -> ProxyObservables:
        """Map a game payoff to ProxyObservables.

        High payoffs → good observables (high task_progress, low rework).
        Exploitation (high self, low opponent) → moderate engagement.
        """
        # Normalise payoff to [0, 1]
        norm = (payoff - self._payoff_min) / self._payoff_range
        # Normalise opponent payoff
        opp_norm = (opponent_payoff - self._payoff_min) / self._payoff_range

        # Task progress: map [0,1] → [-0.3, 0.8]
        task_progress = -0.3 + 1.1 * norm

        # Rework: inverse relationship with payoff quality
        rework = 0 if norm > 0.5 else self._rng.randint(0, 2)

        # Verifier rejections: only on very low payoffs
        rejections = 0 if norm > 0.3 else self._rng.randint(0, 1)

        # Engagement: high when both do well, low when exploited
        mutual_gain = min(norm, opp_norm)
        engagement = -0.3 + 0.9 * mutual_gain + 0.1 * norm

        return ProxyObservables(
            task_progress_delta=max(-1.0, min(1.0, task_progress)),
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=0,
            counterparty_engagement_delta=max(-1.0, min(1.0, engagement)),
        )

    def handle_action(self, action: Any, state: Any) -> HandlerActionResult:
        """Handle an EVO_GAME_MOVE action.

        The action's metadata should contain:
        - ``initiator_id``: the acting agent
        - ``counterparty_id``: the opponent agent
        """
        initiator_id = getattr(action, "agent_id", "")
        metadata = getattr(action, "metadata", {}) or {}
        counterparty_id = metadata.get("counterparty_id", "")

        if not counterparty_id:
            return HandlerActionResult(
                success=False,
                metadata={"error": "no counterparty_id in action metadata"},
            )

        # Resolve moves
        init_coop = self._resolve_move(initiator_id, counterparty_id)
        counter_coop = self._resolve_move(counterparty_id, initiator_id)

        # Compute game payoffs
        init_payoff, counter_payoff = self.compute_game_payoffs(
            init_coop, counter_coop
        )

        # Update conditional strategy state
        self._update_conditional_state(
            initiator_id, counterparty_id, init_coop, counter_coop
        )

        # Track for population dynamics
        self._current_epoch_moves.append(
            (initiator_id, counterparty_id, init_coop, counter_coop)
        )

        # Generate observables from game payoffs
        observables = self._payoff_to_observables(init_payoff, counter_payoff)

        return HandlerActionResult(
            success=True,
            observables=observables,
            initiator_id=initiator_id,
            counterparty_id=counterparty_id,
            accepted=True,
            interaction_type=InteractionType.COLLABORATION,
            metadata={
                "game": "evolutionary",
                "initiator_cooperated": init_coop,
                "counterparty_cooperated": counter_coop,
                "initiator_payoff": init_payoff,
                "counterparty_payoff": counter_payoff,
            },
        )

    def _update_conditional_state(
        self,
        initiator_id: str,
        counterparty_id: str,
        init_coop: bool,
        counter_coop: bool,
    ) -> None:
        """Update TFT / Grudger state after a move."""
        # TFT: each player mirrors what the opponent just did
        if initiator_id in self._tft_state:
            self._tft_state[initiator_id] = counter_coop
        if counterparty_id in self._tft_state:
            self._tft_state[counterparty_id] = init_coop

        # Grudger: if opponent defected, add to grudge list
        if initiator_id in self._grudge_list and not counter_coop:
            self._grudge_list[initiator_id].add(counterparty_id)
        if counterparty_id in self._grudge_list and not init_coop:
            self._grudge_list[counterparty_id].add(initiator_id)

    def _compute_coop_fraction(self) -> float:
        """Compute current cooperator fraction from registered strategies."""
        if not self._strategies:
            return 0.5
        coop_count = sum(
            1
            for s in self._strategies.values()
            if s in (GameStrategy.COOPERATE, GameStrategy.TIT_FOR_TAT, GameStrategy.GRUDGER)
        )
        return coop_count / len(self._strategies)

    def on_epoch_start(self, state: Any) -> None:
        """Reset per-epoch move tracking."""
        self._current_epoch_moves = []

    def on_epoch_end(self, state: Any) -> None:
        """Record population fraction and render dynamics."""
        coop_frac = self._compute_coop_fraction()
        self._epoch_coop_fractions.append(coop_frac)

        if self._config.render_epoch_dynamics:
            epoch = getattr(state, "current_epoch", len(self._epoch_coop_fractions))
            self._render_epoch_summary(epoch, coop_frac)

    def _render_epoch_summary(self, epoch: int, coop_frac: float) -> None:
        """Print gamescape rendering for the current epoch."""
        game_type = classify_game(self._matrix)
        flow = render_flow_line(
            self._matrix, width=self._config.flow_width, color=False
        )

        n_moves = len(self._current_epoch_moves)
        n_coop = sum(1 for _, _, ic, cc in self._current_epoch_moves if ic)
        n_defect = n_moves - n_coop

        lines = [
            f"--- Epoch {epoch} | Evo Game ({game_type}) ---",
            f"  Cooperator fraction: {coop_frac:.2f}",
            f"  Moves this epoch: {n_moves} ({n_coop}C / {n_defect}D)",
            f"  Replicator dx at x={coop_frac:.2f}: "
            f"{replicator_dx(self._matrix, coop_frac):.4f}",
            f"  Flow: {flow}",
        ]
        for line in lines:
            logger.info(line)

    def get_population_trajectory(self) -> List[float]:
        """Return recorded cooperator fractions per epoch."""
        return list(self._epoch_coop_fractions)

    def get_replicator_prediction(
        self,
        x0: Optional[float] = None,
        steps: int = 2000,
    ) -> List[float]:
        """Run gamescape's replicator dynamics from initial conditions.

        Args:
            x0: Initial cooperator fraction.  Defaults to the first
                recorded epoch fraction, or 0.5.
            steps: Number of replicator dynamics steps.
        """
        if x0 is None:
            x0 = self._epoch_coop_fractions[0] if self._epoch_coop_fractions else 0.5
        result: List[float] = trajectory(self._matrix, x0, steps=steps)
        return result

    def render_full_analysis(self) -> str:
        """Render gamescape's full compact analysis."""
        lines = render_compact(
            name="Evo Game",
            game=self._matrix,
            color=False,
        )
        return "\n".join(lines)

    @property
    def matrix(self) -> PayoffMatrix:
        """The underlying gamescape PayoffMatrix."""
        return self._matrix

    @property
    def strategies(self) -> Dict[str, GameStrategy]:
        """Copy of the agent strategy registry."""
        return dict(self._strategies)
