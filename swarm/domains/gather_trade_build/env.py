"""GTB gridworld environment: state, step semantics, and market."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from swarm.domains.gather_trade_build.config import GTBConfig
from swarm.domains.gather_trade_build.entities import (
    Direction,
    GTBActionType,
    GTBEvent,
    GTBGridCell,
    House,
    MarketOrder,
    Resource,
    ResourceType,
    WorkerState,
)
from swarm.domains.gather_trade_build.metrics import compute_gini
from swarm.domains.gather_trade_build.tax_schedule import TaxSchedule

logger = logging.getLogger(__name__)

# Direction deltas: (row_delta, col_delta)
_DIR_DELTA = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


@dataclass
class EpochResult:
    """Result from end_epoch(), containing events and a pre-reset worker snapshot."""

    events: List[GTBEvent] = field(default_factory=list)
    snapshot: Dict[str, WorkerState] = field(default_factory=dict)


@dataclass
class GTBAction:
    """A worker's action for one step."""

    agent_id: str = ""
    action_type: GTBActionType = GTBActionType.NOOP
    direction: Direction = Direction.UP
    resource_type: ResourceType = ResourceType.WOOD
    quantity: float = 0.0
    price: float = 1.0
    shift_amount: float = 0.0
    underreport_fraction: float = 0.0


class GTBEnvironment:
    """Gather-Trade-Build gridworld environment.

    Provides:
      - Grid with resource tiles and houses
      - Worker state management (inventory, position, energy)
      - Centralized market for trading resources
      - Tax collection via TaxSchedule at epoch boundaries
      - Income shifting and misreporting mechanics
      - Audit pipeline
      - Collusion detection scaffolding
      - Event logging
    """

    def __init__(self, config: GTBConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._tax_schedule = TaxSchedule(config.taxation)

        # Grid
        self._height = config.map.height
        self._width = config.map.width
        self._grid: List[List[GTBGridCell]] = []
        self._houses: List[House] = []

        # Workers
        self._workers: Dict[str, WorkerState] = {}

        # Market
        self._buy_orders: List[MarketOrder] = []
        self._sell_orders: List[MarketOrder] = []

        # Events
        self._events: List[GTBEvent] = []

        # Step / epoch counters
        self._current_step = 0
        self._current_epoch = 0

        # Collusion tracking
        self._action_traces: Dict[str, List[str]] = {}  # agent_id -> recent actions

        # Misreport fractions per agent for current epoch (applied to future income)
        self._misreport_fractions: Dict[str, float] = {}

        # Collusion response state
        self._collusion_audit_boost: Dict[str, float] = {}  # agent_id -> extra audit prob
        self._trade_restricted: Dict[str, int] = {}  # agent_id -> restriction_end_epoch

        # Frozen agents (from audit penalties)
        self._frozen_agents: Dict[str, int] = {}  # agent_id -> unfreeze_epoch

        self._init_grid()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_grid(self) -> None:
        """Initialize the grid with resource tiles."""
        self._grid = []
        for r in range(self._height):
            row = []
            for c in range(self._width):
                cell = GTBGridCell(position=(r, c))
                # Randomly place resources
                roll = self._rng.random()
                if roll < self._config.map.wood_density:
                    cell.resource = Resource(
                        resource_type=ResourceType.WOOD,
                        amount=self._config.map.resource_max_amount,
                        position=(r, c),
                        regen_rate=self._config.map.resource_regen_rate,
                    )
                elif roll < self._config.map.wood_density + self._config.map.stone_density:
                    cell.resource = Resource(
                        resource_type=ResourceType.STONE,
                        amount=self._config.map.resource_max_amount,
                        position=(r, c),
                        regen_rate=self._config.map.resource_regen_rate,
                    )
                row.append(cell)
            self._grid.append(row)

    def add_worker(self, agent_id: str, skill_gather: float = 1.0,
                   skill_build: float = 1.0) -> WorkerState:
        """Register a worker in the environment."""
        row = self._rng.randint(0, self._height - 1)
        col = self._rng.randint(0, self._width - 1)
        worker = WorkerState(
            agent_id=agent_id,
            position=(row, col),
            energy=self._config.energy_per_step,
            max_energy=self._config.energy_per_step,
            skill_gather=skill_gather,
            skill_build=skill_build,
        )
        worker.add_resource(ResourceType.COIN, 10.0)  # starting endowment
        self._workers[agent_id] = worker
        self._grid[row][col].occupants.append(agent_id)
        self._action_traces[agent_id] = []
        return worker

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def obs(self, agent_id: str) -> Dict[str, Any]:
        """Build observation for an agent."""
        worker = self._workers[agent_id]
        r, c = worker.position
        # Visible neighborhood (5x5 centered on agent)
        visible = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self._height and 0 <= nc < self._width:
                    cell = self._grid[nr][nc]
                    cell_info: Dict[str, Any] = {"pos": (nr, nc)}
                    if cell.resource and cell.resource.amount > 0:
                        cell_info["resource"] = cell.resource.resource_type.value
                        cell_info["amount"] = cell.resource.amount
                    if cell.house:
                        cell_info["house_owner"] = cell.house.owner_id
                    cell_info["occupants"] = list(cell.occupants)
                    visible.append(cell_info)

        return {
            "agent_id": agent_id,
            "position": worker.position,
            "inventory": dict(worker.inventory),
            "energy": worker.energy,
            "houses_built": worker.houses_built,
            "gross_income": worker.gross_income_this_epoch,
            "deferred_income": worker.deferred_income,
            "epoch": self._current_epoch,
            "step": self._current_step,
            "tax_schedule": self._tax_schedule.to_dict(),
            "visible_cells": visible,
            "frozen": agent_id in self._frozen_agents,
        }

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def apply_actions(self, actions: Dict[str, GTBAction]) -> List[GTBEvent]:
        """Apply a dict of agent actions for one step.

        Args:
            actions: Mapping from agent_id to their chosen action.

        Returns:
            List of events generated this step.
        """
        step_events: List[GTBEvent] = []

        # Regenerate resources
        self._regenerate_resources()

        # Process each agent's action
        for agent_id, action in actions.items():
            if agent_id in self._frozen_agents:
                step_events.append(GTBEvent(
                    event_type="frozen_skip",
                    step=self._current_step,
                    epoch=self._current_epoch,
                    agent_id=agent_id,
                ))
                continue

            worker = self._workers.get(agent_id)
            if worker is None:
                continue

            # Track action for collusion detection
            self._action_traces.setdefault(agent_id, []).append(
                action.action_type.value
            )

            if action.action_type == GTBActionType.MOVE:
                evt = self._handle_move(worker, action.direction)
            elif action.action_type == GTBActionType.GATHER:
                evt = self._handle_gather(worker)
            elif action.action_type == GTBActionType.BUILD:
                evt = self._handle_build(worker)
            elif action.action_type == GTBActionType.TRADE_BUY:
                evt = self._handle_trade_buy(worker, action)
            elif action.action_type == GTBActionType.TRADE_SELL:
                evt = self._handle_trade_sell(worker, action)
            elif action.action_type == GTBActionType.SHIFT_INCOME:
                evt = self._handle_shift_income(worker, action)
            elif action.action_type == GTBActionType.MISREPORT:
                evt = self._handle_misreport(worker, action)
            else:
                evt = GTBEvent(
                    event_type="noop",
                    step=self._current_step,
                    epoch=self._current_epoch,
                    agent_id=agent_id,
                )

            if evt:
                step_events.append(evt)

        # Distribute house income
        house_events = self._distribute_house_income()
        step_events.extend(house_events)

        # Match market orders
        trade_events = self._match_market_orders()
        step_events.extend(trade_events)

        self._events.extend(step_events)
        self._current_step += 1
        return step_events

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_move(self, worker: WorkerState, direction: Direction) -> GTBEvent:
        delta = _DIR_DELTA.get(direction)
        if delta is None or worker.energy < self._config.energy_cost_move:
            return GTBEvent(
                event_type="move_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
            )
        old_r, old_c = worker.position
        new_r = max(0, min(self._height - 1, old_r + delta[0]))
        new_c = max(0, min(self._width - 1, old_c + delta[1]))

        # Update grid occupants
        self._grid[old_r][old_c].occupants = [
            a for a in self._grid[old_r][old_c].occupants if a != worker.agent_id
        ]
        self._grid[new_r][new_c].occupants.append(worker.agent_id)
        worker.position = (new_r, new_c)
        worker.energy -= self._config.energy_cost_move

        return GTBEvent(
            event_type="move", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"from": (old_r, old_c), "to": (new_r, new_c)},
        )

    def _handle_gather(self, worker: WorkerState) -> GTBEvent:
        if worker.energy < self._config.energy_cost_gather:
            return GTBEvent(
                event_type="gather_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_energy"},
            )
        r, c = worker.position
        cell = self._grid[r][c]
        if cell.resource is None or cell.resource.amount <= 0:
            return GTBEvent(
                event_type="gather_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_resource"},
            )

        gathered = min(cell.resource.amount, 1.0 * worker.skill_gather)
        cell.resource.amount -= gathered
        worker.add_resource(cell.resource.resource_type, gathered)
        worker.energy -= self._config.energy_cost_gather

        # Gathering generates income
        income = gathered  # 1:1 income for gathering
        worker.gross_income_this_epoch += income
        worker.reported_income_this_epoch += income
        worker.cumulative_income += income

        return GTBEvent(
            event_type="gather", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={
                "resource": cell.resource.resource_type.value,
                "amount": gathered, "income": income,
            },
        )

    def _handle_build(self, worker: WorkerState) -> GTBEvent:
        cfg = self._config.build
        if worker.energy < self._config.energy_cost_build:
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_energy"},
            )
        if worker.houses_built >= cfg.max_houses_per_agent:
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "max_houses"},
            )
        wood_ok = worker.remove_resource(ResourceType.WOOD, cfg.wood_cost)
        if not wood_ok:
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "insufficient_wood"},
            )
        stone_ok = worker.remove_resource(ResourceType.STONE, cfg.stone_cost)
        if not stone_ok:
            # Refund wood
            worker.add_resource(ResourceType.WOOD, cfg.wood_cost)
            return GTBEvent(
                event_type="build_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "insufficient_stone"},
            )

        r, c = worker.position
        house = House(
            owner_id=worker.agent_id,
            position=(r, c),
            wood_cost=cfg.wood_cost,
            stone_cost=cfg.stone_cost,
            income_per_step=cfg.income_per_house_per_step,
            build_step=self._current_step,
        )
        self._houses.append(house)
        self._grid[r][c].house = house
        worker.houses_built += 1
        worker.energy -= self._config.energy_cost_build

        return GTBEvent(
            event_type="build", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"position": (r, c), "houses_total": worker.houses_built},
        )

    def _handle_trade_buy(self, worker: WorkerState, action: GTBAction) -> GTBEvent:
        return self._handle_trade(worker, action, is_buy=True)

    def _handle_trade_sell(self, worker: WorkerState, action: GTBAction) -> GTBEvent:
        return self._handle_trade(worker, action, is_buy=False)

    def _handle_trade(
        self, worker: WorkerState, action: GTBAction, *, is_buy: bool
    ) -> GTBEvent:
        # Enforce collusion-triggered trade restrictions
        restrict_until = self._trade_restricted.get(worker.agent_id, 0)
        if self._current_epoch < restrict_until:
            return GTBEvent(
                event_type="trade_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "trade_restricted_collusion"},
            )
        if worker.energy < self._config.energy_cost_trade:
            return GTBEvent(
                event_type="trade_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "no_energy"},
            )
        price = max(self._config.market.price_floor,
                     min(action.price, self._config.market.price_ceiling))
        order = MarketOrder(
            agent_id=worker.agent_id,
            resource_type=action.resource_type,
            quantity=max(0, action.quantity),
            price_per_unit=price,
            is_buy=is_buy,
            step=self._current_step,
        )
        if is_buy:
            self._buy_orders.append(order)
        else:
            self._sell_orders.append(order)
        worker.energy -= self._config.energy_cost_trade
        return GTBEvent(
            event_type="order_placed", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"side": "buy" if is_buy else "sell",
                      "resource": action.resource_type.value,
                      "qty": action.quantity, "price": price},
        )

    def _handle_shift_income(self, worker: WorkerState,
                              action: GTBAction) -> GTBEvent:
        cfg = self._config.gaming
        if not cfg.income_shifting_enabled:
            return GTBEvent(
                event_type="shift_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "disabled"},
            )
        max_shift = worker.gross_income_this_epoch * cfg.max_shift_fraction_per_epoch
        shift = min(action.shift_amount, max_shift)
        if shift <= 0:
            return GTBEvent(
                event_type="shift_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "nothing_to_shift"},
            )

        friction = shift * cfg.gaming_cost_fraction
        net_shift = shift - friction
        worker.gross_income_this_epoch -= shift
        worker.reported_income_this_epoch -= shift
        worker.deferred_income += net_shift

        return GTBEvent(
            event_type="shift_income", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"gross_shifted": shift, "friction": friction,
                      "net_deferred": net_shift},
        )

    def _handle_misreport(self, worker: WorkerState,
                           action: GTBAction) -> GTBEvent:
        cfg = self._config.misreporting
        if not cfg.enabled:
            return GTBEvent(
                event_type="misreport_fail", step=self._current_step,
                epoch=self._current_epoch, agent_id=worker.agent_id,
                details={"reason": "disabled"},
            )
        frac = min(action.underreport_fraction, cfg.max_underreport_fraction)
        frac = max(0.0, frac)
        hidden = worker.gross_income_this_epoch * frac
        worker.reported_income_this_epoch = worker.gross_income_this_epoch - hidden
        # Track fraction so future income (house income) also applies it
        self._misreport_fractions[worker.agent_id] = frac

        return GTBEvent(
            event_type="misreport", step=self._current_step,
            epoch=self._current_epoch, agent_id=worker.agent_id,
            details={"underreport_fraction": frac, "hidden_income": hidden},
        )

    # ------------------------------------------------------------------
    # House income distribution
    # ------------------------------------------------------------------

    def _distribute_house_income(self) -> List[GTBEvent]:
        events = []
        for house in self._houses:
            worker = self._workers.get(house.owner_id)
            if worker is None:
                continue
            income = house.income_per_step
            worker.add_resource(ResourceType.COIN, income)
            worker.gross_income_this_epoch += income
            # Only add to reported income the non-hidden fraction.
            # If the worker has misreported, the underreport fraction applies
            # to all income including house income (prevents partial undo).
            misreport_frac = self._misreport_fractions.get(worker.agent_id, 0.0)
            reported_portion = income * (1.0 - misreport_frac)
            worker.reported_income_this_epoch += reported_portion
            worker.cumulative_income += income
            events.append(GTBEvent(
                event_type="house_income", step=self._current_step,
                epoch=self._current_epoch, agent_id=house.owner_id,
                details={"income": income, "reported": reported_portion,
                          "house_pos": house.position},
            ))
        return events

    # ------------------------------------------------------------------
    # Market matching
    # ------------------------------------------------------------------

    def _match_market_orders(self) -> List[GTBEvent]:
        """Simple centralized market matching: match buy and sell orders."""
        events: List[GTBEvent] = []
        fee_rate = self._config.market.transaction_fee_rate

        # Group by resource type
        for rtype in ResourceType:
            if rtype == ResourceType.COIN:
                continue
            buys = sorted(
                [o for o in self._buy_orders if o.resource_type == rtype],
                key=lambda o: -o.price_per_unit,
            )
            sells = sorted(
                [o for o in self._sell_orders if o.resource_type == rtype],
                key=lambda o: o.price_per_unit,
            )

            bi, si = 0, 0
            while bi < len(buys) and si < len(sells):
                buy = buys[bi]
                sell = sells[si]
                if buy.price_per_unit < sell.price_per_unit:
                    break  # no more matchable orders
                if buy.agent_id == sell.agent_id:
                    si += 1
                    continue

                qty = min(buy.quantity, sell.quantity)
                if qty <= 0:
                    bi += 1
                    continue

                price = (buy.price_per_unit + sell.price_per_unit) / 2.0
                total = qty * price
                fee = total * fee_rate

                buyer = self._workers.get(buy.agent_id)
                seller = self._workers.get(sell.agent_id)
                if buyer is None or seller is None:
                    bi += 1
                    continue

                # Check buyer has coin and seller has resource
                if buyer.get_resource(ResourceType.COIN) < total + fee:
                    bi += 1
                    continue
                if seller.get_resource(rtype) < qty:
                    si += 1
                    continue

                buyer.remove_resource(ResourceType.COIN, total + fee)
                seller.add_resource(ResourceType.COIN, total - fee)
                seller.remove_resource(rtype, qty)
                buyer.add_resource(rtype, qty)

                # Trade income for seller
                seller.gross_income_this_epoch += total - fee
                seller.reported_income_this_epoch += total - fee
                seller.cumulative_income += total - fee

                buy.quantity -= qty
                sell.quantity -= qty
                if buy.quantity <= 0:
                    bi += 1
                if sell.quantity <= 0:
                    si += 1

                events.append(GTBEvent(
                    event_type="trade", step=self._current_step,
                    epoch=self._current_epoch,
                    details={
                        "buyer": buy.agent_id, "seller": sell.agent_id,
                        "resource": rtype.value, "quantity": qty,
                        "price": price, "fee": fee,
                    },
                ))

        # Clear order books
        self._buy_orders.clear()
        self._sell_orders.clear()
        return events

    # ------------------------------------------------------------------
    # Resource regeneration
    # ------------------------------------------------------------------

    def _regenerate_resources(self) -> None:
        for row in self._grid:
            for cell in row:
                if cell.resource is not None:
                    cell.resource.amount = min(
                        cell.resource.amount + cell.resource.regen_rate,
                        self._config.map.resource_max_amount,
                    )

    # ------------------------------------------------------------------
    # Epoch boundary: taxes, audits, income shifting resolution
    # ------------------------------------------------------------------

    def end_epoch(self) -> "EpochResult":
        """Process epoch boundary: taxes, audits, deferred income.

        Returns an EpochResult containing the events and a pre-reset
        snapshot of worker states (for metrics computation).
        """
        events: List[GTBEvent] = []

        # 1. Collect taxes (only what can actually be paid)
        for agent_id, worker in self._workers.items():
            tax = self._tax_schedule.compute_tax(worker.reported_income_this_epoch)
            coin_balance = worker.get_resource(ResourceType.COIN)
            actual_tax = min(tax, coin_balance)
            worker.remove_resource(ResourceType.COIN, actual_tax)
            worker.tax_paid_this_epoch = actual_tax
            events.append(GTBEvent(
                event_type="tax", epoch=self._current_epoch,
                agent_id=agent_id,
                details={
                    "gross_income": worker.gross_income_this_epoch,
                    "reported_income": worker.reported_income_this_epoch,
                    "tax_owed": tax,
                    "tax_paid": actual_tax,
                    "shortfall": tax - actual_tax,
                    "effective_rate": actual_tax / max(worker.reported_income_this_epoch, 1e-9),
                },
            ))

        # 2. Audits
        audit_events = self._run_audits()
        events.extend(audit_events)

        # 3. Unfreeze agents whose freeze expired
        to_unfreeze = [
            aid for aid, unfreeze_epoch in self._frozen_agents.items()
            if self._current_epoch >= unfreeze_epoch
        ]
        for aid in to_unfreeze:
            del self._frozen_agents[aid]
            events.append(GTBEvent(
                event_type="unfreeze", epoch=self._current_epoch,
                agent_id=aid,
            ))

        # Snapshot worker state AFTER taxes/audits but BEFORE reset.
        # This gives metrics the accurate post-tax, pre-reset view.
        snapshot = self.snapshot_epoch_data()

        # 4. Reset epoch accumulators and materialize deferred income
        for worker in self._workers.values():
            deferred = worker.deferred_income
            worker.reset_epoch()
            # Carry deferred income as taxable income in the new epoch
            worker.gross_income_this_epoch += deferred
            worker.reported_income_this_epoch += deferred
            worker.deferred_income = 0.0
            worker.energy = worker.max_energy

        # Clear per-epoch misreport fractions
        self._misreport_fractions.clear()

        self._current_epoch += 1
        self._current_step = 0

        # Trim action traces to detection window
        window = self._config.collusion.detection_window_steps
        for aid in self._action_traces:
            self._action_traces[aid] = self._action_traces[aid][-window:]

        return EpochResult(events=events, snapshot=snapshot)

    def _run_audits(self) -> List[GTBEvent]:
        """Run audit pipeline for the current epoch."""
        cfg = self._config.misreporting
        if not cfg.enabled:
            return []

        events: List[GTBEvent] = []
        for agent_id, worker in self._workers.items():
            discrepancy = worker.gross_income_this_epoch - worker.reported_income_this_epoch
            if discrepancy < 1e-9:
                continue  # Nothing to catch

            # Compute audit probability (base + risk-based + collusion boost)
            audit_prob = cfg.audit_probability
            if discrepancy > 0:
                ratio = discrepancy / max(worker.gross_income_this_epoch, 1e-9)
                audit_prob = min(1.0, audit_prob * (1.0 + ratio * cfg.risk_based_audit_multiplier))
            # Apply collusion-triggered audit boost
            collusion_boost = self._collusion_audit_boost.get(agent_id, 0.0)
            if collusion_boost > 0:
                audit_prob = min(1.0, audit_prob + collusion_boost * cfg.audit_probability)

            worker.times_audited += 1

            if self._rng.random() < audit_prob:
                # Caught!
                evaded_tax = self._tax_schedule.compute_tax(
                    worker.gross_income_this_epoch
                ) - self._tax_schedule.compute_tax(worker.reported_income_this_epoch)
                fine_owed = evaded_tax * cfg.fine_multiplier
                coin_balance = worker.get_resource(ResourceType.COIN)
                fine_paid = min(fine_owed, coin_balance)
                worker.remove_resource(ResourceType.COIN, fine_paid)
                worker.times_caught += 1
                worker.total_fines += fine_paid

                events.append(GTBEvent(
                    event_type="audit_caught", epoch=self._current_epoch,
                    agent_id=agent_id,
                    details={
                        "discrepancy": discrepancy,
                        "evaded_tax": evaded_tax,
                        "fine_owed": fine_owed,
                        "fine": fine_paid,
                        "shortfall": fine_owed - fine_paid,
                        "times_caught": worker.times_caught,
                    },
                ))

                # Freeze on repeat offenders
                if cfg.freeze_on_repeat and worker.times_caught >= cfg.freeze_after_n_catches:
                    self._frozen_agents[agent_id] = (
                        self._current_epoch + cfg.freeze_duration_epochs
                    )
                    events.append(GTBEvent(
                        event_type="freeze", epoch=self._current_epoch,
                        agent_id=agent_id,
                        details={"until_epoch": self._current_epoch + cfg.freeze_duration_epochs},
                    ))
            else:
                events.append(GTBEvent(
                    event_type="audit_miss", epoch=self._current_epoch,
                    agent_id=agent_id,
                    details={"discrepancy": discrepancy},
                ))

        # False-positive pass: audit honest agents flagged by collusion boost.
        # These agents have no actual discrepancy, so they can never be "caught",
        # but selecting them for audit is itself a governance backfire.
        for agent_id, worker in self._workers.items():
            discrepancy = worker.gross_income_this_epoch - worker.reported_income_this_epoch
            if discrepancy >= 1e-9:
                continue  # Already handled in the misreporter pass above
            collusion_boost = self._collusion_audit_boost.get(agent_id, 0.0)
            if collusion_boost <= 0:
                continue  # Not flagged; skip
            fp_audit_prob = min(1.0, collusion_boost * cfg.audit_probability)
            if self._rng.random() < fp_audit_prob:
                worker.times_audited += 1
                events.append(GTBEvent(
                    event_type="audit_false_positive",
                    epoch=self._current_epoch,
                    agent_id=agent_id,
                    details={"collusion_boost": collusion_boost},
                ))

        return events

    # ------------------------------------------------------------------
    # Collusion detection
    # ------------------------------------------------------------------

    def detect_collusion(self) -> List[GTBEvent]:
        """Detect potential collusion among workers.

        Uses action-trace similarity over a rolling window.

        Returns:
            List of collusion detection events.
        """
        cfg = self._config.collusion
        if not cfg.enabled:
            return []

        events: List[GTBEvent] = []
        agent_ids = list(self._action_traces.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                aid_a = agent_ids[i]
                aid_b = agent_ids[j]
                trace_a = self._action_traces[aid_a]
                trace_b = self._action_traces[aid_b]

                if not trace_a or not trace_b:
                    continue

                # Compute Jaccard similarity on action distribution
                min_len = min(len(trace_a), len(trace_b))
                if min_len == 0:
                    continue

                matches = sum(
                    1 for a, b in zip(trace_a[-min_len:], trace_b[-min_len:], strict=True)
                    if a == b
                )
                similarity = matches / min_len

                if similarity >= cfg.similarity_threshold:
                    # Compute suspicion score
                    worker_a = self._workers[aid_a]
                    worker_b = self._workers[aid_b]
                    same_coalition = (
                        worker_a.coalition_id is not None
                        and worker_a.coalition_id == worker_b.coalition_id
                    )
                    suspicion = similarity
                    if same_coalition:
                        suspicion = min(1.0, suspicion * 1.3)

                    if suspicion >= cfg.suspicion_score_threshold:
                        # Apply response actions
                        responses_applied = []

                        # Increase audit probability for flagged agents
                        for aid in (aid_a, aid_b):
                            current_extra = self._collusion_audit_boost.get(aid, 0.0)
                            boost = cfg.response_audit_multiplier - 1.0
                            self._collusion_audit_boost[aid] = min(
                                current_extra + boost, 5.0,
                            )
                            responses_applied.append("audit_boost")

                        # Temporary trade restriction
                        if cfg.response_trade_restriction_epochs > 0:
                            restrict_until = (
                                self._current_epoch
                                + cfg.response_trade_restriction_epochs
                            )
                            for aid in (aid_a, aid_b):
                                self._trade_restricted[aid] = max(
                                    self._trade_restricted.get(aid, 0),
                                    restrict_until,
                                )
                            responses_applied.append("trade_restriction")

                        events.append(GTBEvent(
                            event_type="collusion_detected",
                            epoch=self._current_epoch,
                            details={
                                "agents": [aid_a, aid_b],
                                "similarity": similarity,
                                "suspicion_score": suspicion,
                                "same_coalition": same_coalition,
                                "responses": responses_applied,
                            },
                        ))

        return events

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def snapshot_epoch_data(self) -> Dict[str, "WorkerState"]:
        """Create a snapshot of per-epoch worker metrics before reset.

        Returns a dict of lightweight WorkerState copies with the
        per-epoch fields preserved. Call this BEFORE end_epoch().
        """
        import copy
        snapshot = {}
        for aid, w in self._workers.items():
            ws = copy.copy(w)
            # Shallow copy is sufficient -- we only read scalar fields
            # and the inventory dict (which won't be mutated by reset_epoch)
            ws.inventory = dict(w.inventory)
            snapshot[aid] = ws
        return snapshot

    @property
    def tax_schedule(self) -> TaxSchedule:
        return self._tax_schedule

    @property
    def workers(self) -> Dict[str, WorkerState]:
        return dict(self._workers)

    @property
    def houses(self) -> List[House]:
        return list(self._houses)

    @property
    def events(self) -> List[GTBEvent]:
        return list(self._events)

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def config(self) -> GTBConfig:
        return self._config

    def get_aggregate_stats(self) -> Dict[str, float]:
        """Compute aggregate stats for planner observation."""
        incomes = [w.gross_income_this_epoch for w in self._workers.values()]
        coins = [w.get_resource(ResourceType.COIN) for w in self._workers.values()]
        n = len(incomes) or 1

        total_income = sum(incomes)
        mean_income = total_income / n
        total_tax = sum(w.tax_paid_this_epoch for w in self._workers.values())
        total_houses = sum(w.houses_built for w in self._workers.values())

        # Gini coefficient
        gini = compute_gini(incomes)

        return {
            "total_income": total_income,
            "mean_income": mean_income,
            "gini": gini,
            "total_tax_revenue": total_tax,
            "total_houses": total_houses,
            "mean_coin": sum(coins) / n,
            "n_workers": n,
            "n_frozen": len(self._frozen_agents),
        }

    def compute_incomes(self) -> Dict[str, float]:
        """Return current epoch gross income per worker."""
        return {aid: w.gross_income_this_epoch for aid, w in self._workers.items()}
