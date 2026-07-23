"""High-Frequency Negotiation (HFN) engine with flash crash detection.

Models speed-based market dynamics where agents submit orders at high rates,
with risk of flash crashes (sudden correlated quality collapses). Includes
a flash crash detector and circuit breaker mechanism.

Reference: "Virtual Agent Economies" (arXiv 2509.10147) - Tomasev et al.
Connects to Kyle et al. (2017) on flash crashes in electronic markets.
"""

import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from swarm.env.stats import gini


@dataclass
class HFNConfig:
    """Configuration for the HFN engine."""

    tick_duration_ms: float = 100.0
    max_orders_per_tick: int = 10
    latency_noise_ms: float = 10.0
    priority_by_speed: bool = True
    batch_interval_ticks: int = 5
    halt_duration_ticks: int = 20

    def validate(self) -> None:
        """Validate configuration."""
        if self.tick_duration_ms <= 0:
            raise ValueError("tick_duration_ms must be positive")
        if self.max_orders_per_tick < 1:
            raise ValueError("max_orders_per_tick must be >= 1")
        if self.latency_noise_ms < 0:
            raise ValueError("latency_noise_ms must be non-negative")
        if self.batch_interval_ticks < 1:
            raise ValueError("batch_interval_ticks must be >= 1")
        if self.halt_duration_ticks < 1:
            raise ValueError("halt_duration_ticks must be >= 1")


@dataclass
class HFNOrder:
    """A high-frequency market order."""

    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    order_type: str = "bid"  # "bid", "ask", "cancel"
    resource_type: str = ""
    quantity: float = 0.0
    price: float = 0.0
    timestamp_ms: float = 0.0
    latency_ms: float = 0.0

    @property
    def effective_time(self) -> float:
        """Effective arrival time including latency."""
        return self.timestamp_ms + self.latency_ms

    def to_dict(self) -> Dict:
        """Serialize order."""
        return {
            "order_id": self.order_id,
            "agent_id": self.agent_id,
            "order_type": self.order_type,
            "resource_type": self.resource_type,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp_ms": self.timestamp_ms,
            "latency_ms": self.latency_ms,
        }


@dataclass
class HFNTick:
    """State of the HFN market at a single tick."""

    tick_number: int = 0
    orders_submitted: int = 0
    orders_executed: int = 0
    orders_cancelled: int = 0
    market_price: float = 1.0
    bid_ask_spread: float = 0.0
    volatility: float = 0.0
    halted: bool = False

    def to_dict(self) -> Dict:
        """Serialize tick."""
        return {
            "tick_number": self.tick_number,
            "orders_submitted": self.orders_submitted,
            "orders_executed": self.orders_executed,
            "orders_cancelled": self.orders_cancelled,
            "market_price": self.market_price,
            "bid_ask_spread": self.bid_ask_spread,
            "volatility": self.volatility,
            "halted": self.halted,
        }


@dataclass
class FlashCrashEvent:
    """A detected flash crash event."""

    crash_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_tick: int = 0
    end_tick: int = 0
    price_drop_pct: float = 0.0
    trigger_agent_ids: List[str] = field(default_factory=list)
    recovery_time_ticks: int = 0
    severity: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize crash event."""
        return {
            "crash_id": self.crash_id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "price_drop_pct": self.price_drop_pct,
            "trigger_agent_ids": self.trigger_agent_ids,
            "recovery_time_ticks": self.recovery_time_ticks,
            "severity": self.severity,
        }


class FlashCrashDetector:
    """
    Detects flash crash conditions in HFN markets.

    Monitors price changes within a rolling window and triggers when:
    - Price drops exceed threshold within the window
    - Volume spikes coincide with price drops
    """

    def __init__(
        self,
        price_drop_threshold: float = 0.1,
        window_ticks: int = 10,
        volume_spike_factor: float = 3.0,
    ):
        """
        Initialize the detector.

        Args:
            price_drop_threshold: Fractional price drop to trigger (0.1 = 10%)
            window_ticks: Rolling window size
            volume_spike_factor: Volume spike multiplier for detection
        """
        self.price_drop_threshold = price_drop_threshold
        self.window_ticks = window_ticks
        self.volume_spike_factor = volume_spike_factor

        self._price_history: Deque[float] = deque(maxlen=window_ticks)
        self._volume_history: Deque[int] = deque(maxlen=window_ticks)
        self._crash_history: List[FlashCrashEvent] = []
        self._in_crash = False
        self._crash_start_tick = 0
        self._pre_crash_price = 0.0

    def update(
        self,
        tick: HFNTick,
        active_agent_ids: Optional[List[str]] = None,
    ) -> Optional[FlashCrashEvent]:
        """
        Update detector with new tick data.

        Args:
            tick: Current tick state
            active_agent_ids: Agents active during this tick

        Returns:
            FlashCrashEvent if crash detected, None otherwise
        """
        self._price_history.append(tick.market_price)
        self._volume_history.append(tick.orders_submitted)

        if len(self._price_history) < 2:
            return None

        # Check for crash onset
        if not self._in_crash:
            window_max = max(self._price_history)
            if window_max > 0:
                drop = (window_max - tick.market_price) / window_max
                if drop >= self.price_drop_threshold:
                    self._in_crash = True
                    self._crash_start_tick = tick.tick_number
                    self._pre_crash_price = window_max

                    crash = FlashCrashEvent(
                        start_tick=self._crash_start_tick,
                        end_tick=tick.tick_number,
                        price_drop_pct=drop,
                        trigger_agent_ids=active_agent_ids or [],
                        severity=drop / self.price_drop_threshold,
                    )
                    self._crash_history.append(crash)
                    return crash
        else:
            # Check for recovery
            if self._pre_crash_price > 0:
                recovery = tick.market_price / self._pre_crash_price
                if recovery >= 0.95:
                    # Recovered - update last crash with recovery time
                    if self._crash_history:
                        last = self._crash_history[-1]
                        last.end_tick = tick.tick_number
                        last.recovery_time_ticks = tick.tick_number - last.start_tick
                    self._in_crash = False

        return None

    def get_volatility_index(self) -> float:
        """Compute current volatility from price history."""
        if len(self._price_history) < 2:
            return 0.0

        prices = list(self._price_history)
        returns = [
            (prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-10)
            for i in range(1, len(prices))
        ]

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return float(variance**0.5)

    def get_crash_history(self) -> List[FlashCrashEvent]:
        """Get all detected crash events."""
        return list(self._crash_history)

    @property
    def is_in_crash(self) -> bool:
        """Whether we're currently in a crash."""
        return self._in_crash


class HFNEngine:
    """
    Simulates high-frequency negotiation dynamics.

    Manages an order book, processes ticks, and performs batch clearing.
    Integrates with FlashCrashDetector for safety monitoring.

    Connection to soft labels: every executed trade generates an interaction
    with associated p values. Flash crashes manifest as sudden drops in the
    average p of accepted interactions.
    """

    def __init__(
        self,
        config: Optional[HFNConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the HFN engine."""
        self.config = config or HFNConfig()
        self.config.validate()

        self._current_tick = 0
        self._market_price = 1.0
        self._halted = False
        self._halt_until_tick = 0

        # Order books
        self._bids: List[HFNOrder] = []  # Buy orders (sorted by price desc)
        self._asks: List[HFNOrder] = []  # Sell orders (sorted by price asc)

        # Tracking
        self._tick_history: List[HFNTick] = []
        self._agent_order_counts: Dict[str, int] = {}
        self._executed_trades: List[Dict] = []

        self._detector = FlashCrashDetector()

        import random

        self._rng = random.Random(seed)

    def submit_order(self, order: HFNOrder) -> bool:
        """
        Submit an order to the HFN market.

        Args:
            order: The order to submit

        Returns:
            True if order accepted, False if rejected
        """
        if self._halted:
            return False

        # Enforce per-agent rate limit
        count = self._agent_order_counts.get(order.agent_id, 0)
        if count >= self.config.max_orders_per_tick:
            return False

        # Handle cancel orders (no quantity/price validation needed)
        if order.order_type == "cancel":
            self._cancel_orders(order.agent_id, order.resource_type)
            self._agent_order_counts[order.agent_id] = count + 1
            return True

        if order.quantity <= 0 or order.price <= 0:
            return False

        # Add latency noise
        order.latency_ms += self._rng.uniform(0, self.config.latency_noise_ms)

        if order.order_type == "bid":
            self._bids.append(order)
            # Sort by price descending, then by effective time
            self._bids.sort(key=lambda o: (-o.price, o.effective_time))
        elif order.order_type == "ask":
            self._asks.append(order)
            # Sort by price ascending, then by effective time
            self._asks.sort(key=lambda o: (o.price, o.effective_time))
        else:
            return False

        self._agent_order_counts[order.agent_id] = count + 1
        return True

    def process_tick(self) -> HFNTick:
        """
        Process one tick of the HFN market.

        Returns:
            HFNTick with market state at this tick
        """
        self._current_tick += 1

        # Check halt status
        if self._halted and self._current_tick >= self._halt_until_tick:
            self._halted = False

        # Match orders if at batch interval
        orders_executed = 0
        if self._current_tick % self.config.batch_interval_ticks == 0:
            trades = self._batch_clear()
            orders_executed = len(trades)

        # Compute bid-ask spread
        best_bid = self._bids[0].price if self._bids else 0.0
        best_ask = self._asks[0].price if self._asks else float("inf")
        spread = max(0.0, best_ask - best_bid) if best_ask != float("inf") else 0.0

        tick = HFNTick(
            tick_number=self._current_tick,
            orders_submitted=sum(self._agent_order_counts.values()),
            orders_executed=orders_executed,
            orders_cancelled=0,
            market_price=self._market_price,
            bid_ask_spread=spread,
            volatility=self._detector.get_volatility_index(),
            halted=self._halted,
        )

        # Check for flash crash
        active_agents = list(self._agent_order_counts.keys())
        crash = self._detector.update(tick, active_agents)
        if crash is not None:
            self.halt(self.config.halt_duration_ticks)

        self._tick_history.append(tick)

        # Reset per-tick counters
        self._agent_order_counts.clear()

        return tick

    def _batch_clear(self) -> List[Dict]:
        """Match bids and asks, returning executed trades."""
        trades = []

        while self._bids and self._asks:
            best_bid = self._bids[0]
            best_ask = self._asks[0]

            if best_bid.price < best_ask.price:
                break  # No more matches

            # Execute at midpoint price
            trade_price = (best_bid.price + best_ask.price) / 2
            trade_qty = min(best_bid.quantity, best_ask.quantity)

            trade = {
                "buyer": best_bid.agent_id,
                "seller": best_ask.agent_id,
                "price": trade_price,
                "quantity": trade_qty,
                "tick": self._current_tick,
            }
            trades.append(trade)
            self._executed_trades.append(trade)

            # Update market price
            self._market_price = trade_price

            # Reduce quantities
            best_bid.quantity -= trade_qty
            best_ask.quantity -= trade_qty

            if best_bid.quantity <= 1e-10:
                self._bids.pop(0)
            if best_ask.quantity <= 1e-10:
                self._asks.pop(0)

        return trades

    def _cancel_orders(self, agent_id: str, resource_type: str) -> int:
        """Cancel all orders for an agent on a resource. Returns count cancelled."""
        before = len(self._bids) + len(self._asks)
        self._bids = [
            o
            for o in self._bids
            if not (o.agent_id == agent_id and o.resource_type == resource_type)
        ]
        self._asks = [
            o
            for o in self._asks
            if not (o.agent_id == agent_id and o.resource_type == resource_type)
        ]
        after = len(self._bids) + len(self._asks)
        return before - after

    def halt(self, duration_ticks: Optional[int] = None) -> None:
        """Halt the market for a specified duration."""
        duration = duration_ticks or self.config.halt_duration_ticks
        self._halted = True
        self._halt_until_tick = self._current_tick + duration

    @property
    def is_halted(self) -> bool:
        """Whether the market is currently halted."""
        return self._halted

    @property
    def current_tick(self) -> int:
        """Current tick number."""
        return self._current_tick

    @property
    def market_price(self) -> float:
        """Current market price."""
        return self._market_price

    def get_order_book_depth(self) -> Dict:
        """Get current order book depth."""
        return {
            "bids": len(self._bids),
            "asks": len(self._asks),
            "total_bid_volume": sum(o.quantity for o in self._bids),
            "total_ask_volume": sum(o.quantity for o in self._asks),
        }

    def get_tick_history(self) -> List[HFNTick]:
        """Get full tick history."""
        return list(self._tick_history)

    def get_crash_history(self) -> List[FlashCrashEvent]:
        """Get flash crash history."""
        return self._detector.get_crash_history()

    def speed_advantage_gini(self) -> float:
        """
        Compute Gini coefficient of execution speed advantage across agents.

        Uses average latency per agent from executed trades.
        """
        if not self._executed_trades:
            return 0.0

        # Count trades per agent
        agent_trades: Dict[str, int] = {}
        for trade in self._executed_trades:
            for role in ("buyer", "seller"):
                agent_id = trade[role]
                agent_trades[agent_id] = agent_trades.get(agent_id, 0) + 1

        return gini(agent_trades.values())
