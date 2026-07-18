"""Resource negotiation game handler.

Two-player alternating-offers game where agents split a pool of
heterogeneous resources with private valuations.  The handler manages
multiple concurrent games, enforces turn order, implements the
stochastic deadline, and computes normalized scores.

Game flow per round:
    Player A proposes/accepts/rejects → Player B proposes/accepts/rejects

Scoring:
    deal:    score = sum(val_i * qty_i) / max_possible, in [0, 1]
    no deal: both players score -0.5

Deadline:
    Rounds 1-4 guaranteed.  From round 5 onward, 30% chance of
    termination after each round (checked at round end).
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from swarm.core.handler import Handler, HandlerActionResult
from swarm.core.proxy import ProxyObservables
from swarm.logging.event_bus import EventBus
from swarm.models.interaction import InteractionType

logger = logging.getLogger(__name__)

# ── Enums & data classes ──────────────────────────────────────────────


class NegotiateAction(Enum):
    """Actions a player can take on their turn."""

    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class ResourcePool:
    """Named quantities of resources to divide."""

    resources: Dict[str, int]  # resource_name → quantity

    def total_items(self) -> int:
        return sum(self.resources.values())

    def names(self) -> List[str]:
        return sorted(self.resources.keys())

    def to_str(self) -> str:
        parts = [f"{qty} {name}" for name, qty in sorted(self.resources.items())]
        return ", ".join(parts)


@dataclass
class Proposal:
    """A proposed split of the resource pool."""

    my_share: Dict[str, int]
    their_share: Dict[str, int]

    def is_valid(self, pool: ResourcePool) -> bool:
        """Check my_share + their_share == pool for every resource."""
        for name, total in pool.resources.items():
            mine = self.my_share.get(name, 0)
            theirs = self.their_share.get(name, 0)
            if mine < 0 or theirs < 0:
                return False
            if mine + theirs != total:
                return False
        # Ensure no extra resource names
        valid_names = set(pool.resources.keys())
        if set(self.my_share.keys()) - valid_names:
            return False
        if set(self.their_share.keys()) - valid_names:
            return False
        return True


@dataclass
class NegotiationTurn:
    """Record of one player's action in a round."""

    round_number: int
    player_id: str
    role: str  # "A" or "B"
    action: NegotiateAction
    proposal: Optional[Proposal] = None
    message: str = ""


@dataclass
class GameResult:
    """Outcome of a completed game."""

    game_id: str
    player_a_id: str
    player_b_id: str
    deal_reached: bool
    rounds_played: int
    score_a: float  # normalized 0–1 or -0.5
    score_b: float
    final_proposal: Optional[Proposal] = None
    accepted_by: str = ""


@dataclass
class NegotiationGame:
    """State of a single two-player resource negotiation game."""

    game_id: str
    player_a_id: str
    player_b_id: str
    pool: ResourcePool
    valuations_a: Dict[str, float]
    valuations_b: Dict[str, float]
    round_number: int = 1
    history: List[NegotiationTurn] = field(default_factory=list)
    last_proposal: Optional[Proposal] = None
    last_proposer: Optional[str] = None
    deal_reached: bool = False
    game_over: bool = False
    result: Optional[GameResult] = None

    # Turn tracking within a round
    a_acted_this_round: bool = False
    b_acted_this_round: bool = False

    def whose_turn(self) -> Optional[str]:
        """Return the player_id whose turn it is, or None if round is done."""
        if self.game_over:
            return None
        if not self.a_acted_this_round:
            return self.player_a_id
        if not self.b_acted_this_round:
            return self.player_b_id
        return None  # round complete

    def role_of(self, player_id: str) -> str:
        if player_id == self.player_a_id:
            return "A"
        if player_id == self.player_b_id:
            return "B"
        raise ValueError(f"{player_id} is not in game {self.game_id}")

    def max_score(self, player_id: str) -> float:
        """Maximum possible score if the player gets everything."""
        vals = (
            self.valuations_a
            if player_id == self.player_a_id
            else self.valuations_b
        )
        return sum(
            vals.get(name, 0.0) * qty
            for name, qty in self.pool.resources.items()
        )

    def compute_score(
        self, player_id: str, allocation: Dict[str, int]
    ) -> float:
        """Normalized score for a player given their allocation."""
        vals = (
            self.valuations_a
            if player_id == self.player_a_id
            else self.valuations_b
        )
        raw = sum(vals.get(name, 0.0) * qty for name, qty in allocation.items())
        max_s = self.max_score(player_id)
        if max_s <= 0:
            return 0.0
        return raw / max_s

    def to_observation(self, player_id: str) -> Dict[str, Any]:
        """Build an observation dict for the given player."""
        role = self.role_of(player_id)
        vals = (
            self.valuations_a if role == "A" else self.valuations_b
        )
        val_str = ", ".join(
            f"{name}={v}" for name, v in sorted(vals.items())
        )
        return {
            "game_id": self.game_id,
            "role": role,
            "pool": dict(self.pool.resources),
            "pool_str": self.pool.to_str(),
            "your_valuations": dict(vals),
            "val_str": val_str,
            "round_number": self.round_number,
            "is_your_turn": self.whose_turn() == player_id,
            "last_proposal": (
                {
                    "proposer_role": self.role_of(self.last_proposer)
                    if self.last_proposer
                    else None,
                    "my_share": dict(self.last_proposal.my_share),
                    "their_share": dict(self.last_proposal.their_share),
                }
                if self.last_proposal
                else None
            ),
            "history": [
                {
                    "round": t.round_number,
                    "role": t.role,
                    "action": t.action.value,
                    "proposal": (
                        {
                            "my_share": dict(t.proposal.my_share),
                            "their_share": dict(t.proposal.their_share),
                        }
                        if t.proposal
                        else None
                    ),
                    "message": t.message,
                }
                for t in self.history
            ],
            "deal_reached": self.deal_reached,
            "game_over": self.game_over,
        }


# ── Configuration ─────────────────────────────────────────────────────


@dataclass
class ResourceNegotiationConfig:
    """Configuration for the resource negotiation handler."""

    enabled: bool = True

    # Number of resource types in each game
    min_resource_types: int = 2
    max_resource_types: int = 4

    # Quantity range per resource
    min_quantity: int = 1
    max_quantity: int = 5

    # Valuation range per resource per player
    min_valuation: float = 1.0
    max_valuation: float = 10.0

    # Deadline parameters
    guaranteed_rounds: int = 4
    termination_probability: float = 0.3  # per round from round 5+

    # No-deal penalty
    no_deal_score: float = -0.5

    # Max rounds (hard cap)
    max_rounds: int = 20

    # Resource names to use (if empty, generate generic names)
    resource_names: List[str] = field(
        default_factory=lambda: [
            "apples", "bananas", "cherries", "dates", "elderberries",
        ]
    )

    seed: Optional[int] = None


# ── Handler ───────────────────────────────────────────────────────────


class ResourceNegotiationHandler(Handler):
    """Handler that manages multi-round resource negotiation games.

    Each game pairs two agents who take turns proposing, accepting, or
    rejecting splits of a shared resource pool.  The handler tracks game
    state, enforces rules, and computes scores.
    """

    from swarm.agents.base import ActionType

    _ACTION_TYPES: FrozenSet = frozenset({ActionType.RESOURCE_NEGOTIATE})

    def __init__(
        self,
        *,
        config: ResourceNegotiationConfig,
        event_bus: EventBus,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._config = config
        self._rng = rng or random.Random(config.seed)

        # Active games: game_id → NegotiationGame
        self._games: Dict[str, NegotiationGame] = {}

        # Player-to-game mapping: player_id → list of game_ids
        self._player_games: Dict[str, List[str]] = {}

        # Completed game results
        self._results: List[GameResult] = []

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return ResourceNegotiationHandler._ACTION_TYPES

    # ── Game creation ─────────────────────────────────────────────

    def create_game(
        self,
        player_a_id: str,
        player_b_id: str,
        pool: Optional[ResourcePool] = None,
        valuations_a: Optional[Dict[str, float]] = None,
        valuations_b: Optional[Dict[str, float]] = None,
    ) -> NegotiationGame:
        """Create a new negotiation game between two players.

        If pool/valuations are not provided, they are randomly generated.
        """
        game_id = str(uuid.uuid4())[:8]
        cfg = self._config

        if pool is None:
            n_types = self._rng.randint(
                cfg.min_resource_types, cfg.max_resource_types
            )
            names = self._rng.sample(cfg.resource_names, min(n_types, len(cfg.resource_names)))
            resources = {
                name: self._rng.randint(cfg.min_quantity, cfg.max_quantity)
                for name in names
            }
            pool = ResourcePool(resources=resources)

        if valuations_a is None:
            valuations_a = {
                name: round(
                    self._rng.uniform(cfg.min_valuation, cfg.max_valuation), 1
                )
                for name in pool.resources
            }

        if valuations_b is None:
            valuations_b = {
                name: round(
                    self._rng.uniform(cfg.min_valuation, cfg.max_valuation), 1
                )
                for name in pool.resources
            }

        game = NegotiationGame(
            game_id=game_id,
            player_a_id=player_a_id,
            player_b_id=player_b_id,
            pool=pool,
            valuations_a=valuations_a,
            valuations_b=valuations_b,
        )
        self._games[game_id] = game
        self._player_games.setdefault(player_a_id, []).append(game_id)
        self._player_games.setdefault(player_b_id, []).append(game_id)

        logger.info(
            "Created negotiation game %s: %s vs %s, pool=%s",
            game_id,
            player_a_id,
            player_b_id,
            pool.to_str(),
        )
        return game

    # ── Move processing ───────────────────────────────────────────

    def process_move(
        self,
        game_id: str,
        player_id: str,
        action: NegotiateAction,
        proposal: Optional[Proposal] = None,
        message: str = "",
    ) -> Tuple[bool, str]:
        """Process a negotiation move. Returns (success, error_message)."""
        game = self._games.get(game_id)
        if game is None:
            return False, f"Game {game_id} not found"

        if game.game_over:
            return False, "Game is already over"

        if game.whose_turn() != player_id:
            return False, f"Not {player_id}'s turn"

        role = game.role_of(player_id)

        # Validate action
        if action == NegotiateAction.PROPOSE:
            if proposal is None:
                return False, "Proposal required for propose action"
            if not proposal.is_valid(game.pool):
                return False, "Invalid proposal: shares must sum to pool"

        elif action == NegotiateAction.ACCEPT:
            if game.last_proposal is None:
                return False, "No proposal to accept"
            if game.last_proposer == player_id:
                return False, "Cannot accept your own proposal"

        elif action == NegotiateAction.REJECT:
            if game.last_proposal is None:
                return False, "No proposal to reject"

        # Record the turn
        turn = NegotiationTurn(
            round_number=game.round_number,
            player_id=player_id,
            role=role,
            action=action,
            proposal=proposal,
            message=message,
        )
        game.history.append(turn)

        # Mark player as having acted
        if role == "A":
            game.a_acted_this_round = True
        else:
            game.b_acted_this_round = True

        # Process the action
        if action == NegotiateAction.PROPOSE:
            game.last_proposal = proposal
            game.last_proposer = player_id

        elif action == NegotiateAction.ACCEPT:
            # Deal reached! Compute scores.
            game.deal_reached = True
            game.game_over = True
            self._finalize_deal(game, player_id)

        # After both players have acted, advance round
        if not game.game_over and game.a_acted_this_round and game.b_acted_this_round:
            self._advance_round(game)

        return True, ""

    def _finalize_deal(self, game: NegotiationGame, accepter_id: str) -> None:
        """Finalize a deal after acceptance."""
        proposer_id = game.last_proposer
        assert proposer_id is not None
        assert game.last_proposal is not None

        prop = game.last_proposal

        # The proposal's my_share/their_share is from the proposer's perspective
        if proposer_id == game.player_a_id:
            alloc_a = prop.my_share
            alloc_b = prop.their_share
        else:
            alloc_a = prop.their_share
            alloc_b = prop.my_share

        score_a = game.compute_score(game.player_a_id, alloc_a)
        score_b = game.compute_score(game.player_b_id, alloc_b)

        result = GameResult(
            game_id=game.game_id,
            player_a_id=game.player_a_id,
            player_b_id=game.player_b_id,
            deal_reached=True,
            rounds_played=game.round_number,
            score_a=score_a,
            score_b=score_b,
            final_proposal=game.last_proposal,
            accepted_by=accepter_id,
        )
        game.result = result
        self._results.append(result)

        logger.info(
            "Game %s: deal reached in round %d. A=%.2f, B=%.2f",
            game.game_id,
            game.round_number,
            score_a,
            score_b,
        )

    def _advance_round(self, game: NegotiationGame) -> None:
        """Advance to the next round, checking deadline."""
        cfg = self._config

        # Check stochastic termination from guaranteed_rounds+1 onward
        if game.round_number >= cfg.guaranteed_rounds:
            if self._rng.random() < cfg.termination_probability:
                self._finalize_no_deal(game)
                return

        # Check hard cap
        if game.round_number >= cfg.max_rounds:
            self._finalize_no_deal(game)
            return

        # Advance
        game.round_number += 1
        game.a_acted_this_round = False
        game.b_acted_this_round = False

    def _finalize_no_deal(self, game: NegotiationGame) -> None:
        """Finalize a game that ended without a deal."""
        game.game_over = True
        no_deal = self._config.no_deal_score

        result = GameResult(
            game_id=game.game_id,
            player_a_id=game.player_a_id,
            player_b_id=game.player_b_id,
            deal_reached=False,
            rounds_played=game.round_number,
            score_a=no_deal,
            score_b=no_deal,
        )
        game.result = result
        self._results.append(result)

        logger.info(
            "Game %s: no deal after %d rounds. Both score %.2f",
            game.game_id,
            game.round_number,
            no_deal,
        )

    # ── Handler interface ─────────────────────────────────────────

    def handle_action(self, action: Any, state: Any) -> HandlerActionResult:
        """Handle a RESOURCE_NEGOTIATE action.

        Expected action metadata:
            game_id: str — which game to act in
            negotiate_action: str — "propose", "accept", or "reject"
            proposal: Optional[Dict] — {"my_share": {...}, "their_share": {...}}
            message: str — optional message to other player
        """
        agent_id = getattr(action, "agent_id", "")
        metadata = getattr(action, "metadata", {}) or {}

        game_id = metadata.get("game_id", "")
        negotiate_action_str = metadata.get("negotiate_action", "")
        message = metadata.get("message", "")

        if not game_id:
            return HandlerActionResult(
                success=False,
                metadata={"error": "no game_id in action metadata"},
            )

        try:
            neg_action = NegotiateAction(negotiate_action_str)
        except ValueError:
            return HandlerActionResult(
                success=False,
                metadata={
                    "error": f"invalid negotiate_action: {negotiate_action_str}"
                },
            )

        # Parse proposal if present
        proposal = None
        prop_data = metadata.get("proposal")
        if prop_data and isinstance(prop_data, dict):
            proposal = Proposal(
                my_share=prop_data.get("my_share", {}),
                their_share=prop_data.get("their_share", {}),
            )

        success, error = self.process_move(
            game_id=game_id,
            player_id=agent_id,
            action=neg_action,
            proposal=proposal,
            message=message,
        )

        if not success:
            return HandlerActionResult(
                success=False,
                metadata={"error": error},
            )

        game = self._games[game_id]

        # Generate observables based on game progress
        observables = self._game_to_observables(game, agent_id)

        # Determine counterparty
        counterparty_id = (
            game.player_b_id
            if agent_id == game.player_a_id
            else game.player_a_id
        )

        return HandlerActionResult(
            success=True,
            observables=observables,
            initiator_id=agent_id,
            counterparty_id=counterparty_id,
            accepted=game.deal_reached,
            interaction_type=InteractionType.COLLABORATION,
            metadata={
                "game": "resource_negotiation",
                "game_id": game_id,
                "action": neg_action.value,
                "round": game.round_number,
                "deal_reached": game.deal_reached,
                "game_over": game.game_over,
                "score_a": game.result.score_a if game.result else None,
                "score_b": game.result.score_b if game.result else None,
            },
        )

    def _game_to_observables(
        self, game: NegotiationGame, agent_id: str
    ) -> ProxyObservables:
        """Generate proxy observables from game state."""
        if game.result is not None:
            score = (
                game.result.score_a
                if agent_id == game.player_a_id
                else game.result.score_b
            )
            if game.deal_reached:
                # Map score [0,1] → task_progress [-0.3, 0.8]
                progress = -0.3 + 1.1 * max(0.0, score)
                engagement = 0.5 * score
            else:
                # No deal → poor observables
                progress = -0.5
                engagement = -0.5
        else:
            # Game still in progress — neutral observables
            progress = 0.0
            engagement = 0.1

        return ProxyObservables(
            task_progress_delta=max(-1.0, min(1.0, progress)),
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=max(-1.0, min(1.0, engagement)),
        )

    # ── Observation building ──────────────────────────────────────

    def build_observation_fields(
        self, agent_id: str, state: Any
    ) -> Dict[str, Any]:
        """Return active games and completed results for this agent."""
        game_ids = self._player_games.get(agent_id, [])

        active_games = []
        completed = []
        for gid in game_ids:
            game = self._games.get(gid)
            if game is None:
                continue
            if game.game_over:
                if game.result:
                    completed.append({
                        "game_id": game.game_id,
                        "deal_reached": game.result.deal_reached,
                        "your_score": (
                            game.result.score_a
                            if agent_id == game.player_a_id
                            else game.result.score_b
                        ),
                        "rounds_played": game.result.rounds_played,
                    })
            else:
                active_games.append(game.to_observation(agent_id))

        return {
            "resource_negotiation_games": active_games,
            "resource_negotiation_results": completed,
        }

    # ── Epoch lifecycle ───────────────────────────────────────────

    def on_epoch_start(self, state: Any) -> None:
        """Auto-pair agents into games at epoch start if configured."""
        # Games can be created externally or by the orchestrator.
        # This hook is available for auto-pairing logic.
        pass

    def on_epoch_end(self, state: Any) -> None:
        """Force-end any games that haven't concluded."""
        for game in list(self._games.values()):
            if not game.game_over:
                self._finalize_no_deal(game)

    def create_games_for_agents(
        self, agent_ids: List[str]
    ) -> List[NegotiationGame]:
        """Create games by pairing agents round-robin.

        Pairs agents [0,1], [2,3], etc. If odd number, last agent
        gets no game this round.
        """
        self._rng.shuffle(agent_ids)
        games = []
        for i in range(0, len(agent_ids) - 1, 2):
            game = self.create_game(agent_ids[i], agent_ids[i + 1])
            games.append(game)
        return games

    # ── Accessors ─────────────────────────────────────────────────

    @property
    def games(self) -> Dict[str, NegotiationGame]:
        return dict(self._games)

    @property
    def results(self) -> List[GameResult]:
        return list(self._results)

    def get_agent_games(self, agent_id: str) -> List[NegotiationGame]:
        """Return all games (active + completed) for an agent."""
        gids = self._player_games.get(agent_id, [])
        return [self._games[gid] for gid in gids if gid in self._games]

    def get_active_game(self, agent_id: str) -> Optional[NegotiationGame]:
        """Return the first active game for an agent, or None."""
        for gid in self._player_games.get(agent_id, []):
            game = self._games.get(gid)
            if game and not game.game_over:
                return game
        return None
