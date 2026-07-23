"""Tests for the SimWorld Delivery domain adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm.domains.simworld_delivery.agents import (
    AggressivePolicy,
    CautiousPolicy,
    ConscientiousPolicy,
    OpportunisticPolicy,
)
from swarm.domains.simworld_delivery.config import DeliveryConfig
from swarm.domains.simworld_delivery.entities import (
    DeliveryActionType,
    OrderStatus,
    PersonaType,
)
from swarm.domains.simworld_delivery.env import DeliveryEnvironment
from swarm.domains.simworld_delivery.metrics import (
    DeliveryMetrics,
    compute_delivery_metrics,
)
from swarm.domains.simworld_delivery.runner import DeliveryScenarioRunner


@pytest.fixture
def config() -> DeliveryConfig:
    return DeliveryConfig(seed=42)


@pytest.fixture
def env(config: DeliveryConfig) -> DeliveryEnvironment:
    e = DeliveryEnvironment(config)
    e.add_agent("agent_a", persona=PersonaType.CONSCIENTIOUS)
    e.add_agent("agent_b", persona=PersonaType.AGGRESSIVE)
    return e


class TestDeliveryConfig:
    def test_default_config(self) -> None:
        cfg = DeliveryConfig()
        assert cfg.city.width == 1000.0
        assert cfg.orders.orders_per_epoch == 20
        assert cfg.economy.starting_budget == 100.0
        assert cfg.governance.delivery_fee_rate == 0.05

    def test_from_dict(self) -> None:
        data = {
            "city": {"width": 500.0, "height": 500.0},
            "orders": {"orders_per_epoch": 10},
            "economy": {"starting_budget": 200.0},
            "seed": 123,
        }
        cfg = DeliveryConfig.from_dict(data)
        assert cfg.city.width == 500.0
        assert cfg.orders.orders_per_epoch == 10
        assert cfg.economy.starting_budget == 200.0
        assert cfg.seed == 123

    def test_from_empty_dict(self) -> None:
        cfg = DeliveryConfig.from_dict({})
        assert cfg.city.width == 1000.0


class TestDeliveryEnvironment:
    def test_add_agent(self, config: DeliveryConfig) -> None:
        env = DeliveryEnvironment(config)
        agent = env.add_agent("test_agent")
        assert agent.agent_id == "test_agent"
        assert agent.budget == 100.0
        assert 0 <= agent.position[0] <= config.city.width
        assert 0 <= agent.position[1] <= config.city.height

    def test_generate_orders(self, env: DeliveryEnvironment) -> None:
        events = env.generate_orders(count=5)
        assert len(events) == 5
        assert all(e.event_type == "order_created" for e in events)
        assert len(env.orders) == 5

    def test_obs_structure(self, env: DeliveryEnvironment) -> None:
        env.generate_orders(count=3)
        obs = env.obs("agent_a")
        assert obs["agent_id"] == "agent_a"
        assert "position" in obs
        assert "budget" in obs
        assert "available_orders" in obs
        assert len(obs["available_orders"]) == 3
        assert "other_agents" in obs
        assert len(obs["other_agents"]) == 1  # agent_b

    def test_bidding_and_assignment(self, env: DeliveryEnvironment) -> None:
        from swarm.domains.simworld_delivery.entities import DeliveryAction

        env.generate_orders(count=1)
        order_id = list(env.orders.keys())[0]

        actions = {
            "agent_a": DeliveryAction(
                agent_id="agent_a",
                action_type=DeliveryActionType.BID,
                order_id=order_id,
                bid_amount=10.0,
            ),
            "agent_b": DeliveryAction(
                agent_id="agent_b",
                action_type=DeliveryActionType.BID,
                order_id=order_id,
                bid_amount=15.0,
            ),
        }
        events = env.apply_actions(actions)

        # Lowest bid should win
        bid_won = [e for e in events if e.event_type == "bid_won"]
        assert len(bid_won) == 1
        assert bid_won[0].agent_id == "agent_a"

    def test_order_expiry(self, env: DeliveryEnvironment) -> None:
        env.generate_orders(count=1)
        from swarm.domains.simworld_delivery.entities import DeliveryAction

        # Run enough steps for expiry
        for _ in range(env.config.orders.expiry_steps + 2):
            actions = {
                "agent_a": DeliveryAction(
                    agent_id="agent_a",
                    action_type=DeliveryActionType.WAIT,
                ),
                "agent_b": DeliveryAction(
                    agent_id="agent_b",
                    action_type=DeliveryActionType.WAIT,
                ),
            }
            env.apply_actions(actions)

        expired = [o for o in env.orders.values() if o.status == OrderStatus.EXPIRED]
        assert len(expired) == 1

    def test_shared_orders_pruned_on_epoch_end(self, env: DeliveryEnvironment) -> None:
        env.generate_orders(count=1)
        order_id = next(iter(env.orders))
        env._shared_orders_map[order_id] = "agent_a"
        env.orders[order_id].status = OrderStatus.EXPIRED

        env.end_epoch()

        assert order_id not in env._shared_orders_map


class TestDeliveryPolicies:
    def test_conscientious_bids_on_available(self) -> None:
        policy = ConscientiousPolicy("test", seed=42)
        obs = {
            "current_order": None,
            "available_orders": [
                {
                    "order_id": "o1", "value": 20.0, "distance": 100.0,
                    "pickup_distance": 50.0, "steps_remaining": 15,
                },
            ],
            "has_scooter": False,
            "speed": 1.0,
        }
        action = policy.decide(obs)
        assert action.action_type == DeliveryActionType.BID
        assert action.order_id == "o1"
        assert 0 < action.bid_amount < 20.0

    def test_conscientious_delivers_when_assigned(self) -> None:
        policy = ConscientiousPolicy("test", seed=42)
        obs = {
            "current_order": {"order_id": "o1", "carrying": True},
            "available_orders": [],
        }
        action = policy.decide(obs)
        assert action.action_type == DeliveryActionType.DELIVER

    def test_aggressive_buys_scooter(self) -> None:
        policy = AggressivePolicy("test", seed=42, scooter_priority=True)
        obs = {
            "current_order": None,
            "available_orders": [
                {
                    "order_id": "o1", "value": 20.0, "distance": 100.0,
                    "pickup_distance": 50.0, "steps_remaining": 15,
                },
            ],
            "has_scooter": False,
            "budget": 100.0,
            "speed": 1.0,
        }
        action = policy.decide(obs)
        assert action.action_type == DeliveryActionType.BUY_SCOOTER

    def test_cautious_waits_for_feasible(self) -> None:
        policy = CautiousPolicy("test", seed=42)
        obs = {
            "current_order": None,
            "available_orders": [
                {
                    "order_id": "o1", "value": 20.0, "distance": 10000.0,
                    "pickup_distance": 5000.0, "steps_remaining": 5,
                },
            ],
            "has_scooter": False,
            "speed": 1.0,
        }
        action = policy.decide(obs)
        assert action.action_type == DeliveryActionType.WAIT

    def test_opportunistic_cherry_picks(self) -> None:
        policy = OpportunisticPolicy("test", seed=42)
        obs = {
            "current_order": None,
            "available_orders": [
                {
                    "order_id": "o1", "value": 5.0, "distance": 500.0,
                    "pickup_distance": 200.0, "steps_remaining": 20,
                },
                {
                    "order_id": "o2", "value": 40.0, "distance": 50.0,
                    "pickup_distance": 10.0, "steps_remaining": 20,
                },
            ],
            "has_scooter": False,
            "speed": 1.0,
        }
        action = policy.decide(obs)
        assert action.action_type == DeliveryActionType.BID
        assert action.order_id == "o2"  # picks the better value/distance


class TestDeliveryMetrics:
    def test_metrics_from_empty(self) -> None:
        metrics = compute_delivery_metrics({}, [], epoch=0)
        assert metrics.epoch == 0
        assert metrics.orders_delivered == 0

    def test_metrics_to_dict(self) -> None:
        metrics = compute_delivery_metrics({}, [], epoch=5)
        d = metrics.to_dict()
        assert d["epoch"] == 5
        assert isinstance(d, dict)
        assert "delivery_rate" in d
        assert "adverse_selection_signal" in d


def _small_config(seed: int = 42) -> DeliveryConfig:
    """Create a small-city config suitable for unit tests."""
    from swarm.domains.simworld_delivery.config import CityConfig, OrderConfig
    return DeliveryConfig(
        city=CityConfig(width=20.0, height=20.0, num_depots=1),
        orders=OrderConfig(
            orders_per_epoch=8,
            min_deadline_steps=15,
            max_deadline_steps=40,
        ),
        seed=seed,
    )


class TestDeliveryRunner:
    def test_full_run(self) -> None:
        config = _small_config()
        agent_specs = [
            {"policy": "conscientious", "count": 2},
            {"policy": "aggressive", "count": 1},
            {"policy": "cautious", "count": 1},
        ]
        runner = DeliveryScenarioRunner(
            config=config,
            agent_specs=agent_specs,
            n_epochs=3,
            steps_per_epoch=30,
            seed=42,
        )
        metrics = runner.run()
        assert len(metrics) == 3
        assert all(isinstance(m, DeliveryMetrics) for m in metrics)

        # At least some deliveries should happen in a small city
        total_delivered = sum(m.orders_delivered for m in metrics)
        assert total_delivered > 0

    def test_deterministic_with_seed(self) -> None:
        config = _small_config()
        specs = [{"policy": "conscientious", "count": 2}]

        runner1 = DeliveryScenarioRunner(
            config=config, agent_specs=specs,
            n_epochs=2, steps_per_epoch=20, seed=42,
        )
        m1 = runner1.run()

        runner2 = DeliveryScenarioRunner(
            config=config, agent_specs=specs,
            n_epochs=2, steps_per_epoch=20, seed=42,
        )
        m2 = runner2.run()

        assert m1[0].orders_delivered == m2[0].orders_delivered
        assert m1[0].total_earnings == m2[0].total_earnings

    def test_export(self, tmp_path: "Path") -> None:
        config = DeliveryConfig(seed=42)
        specs = [{"policy": "conscientious", "count": 2}]
        runner = DeliveryScenarioRunner(
            config=config, agent_specs=specs,
            n_epochs=2, steps_per_epoch=5, seed=42,
        )
        runner.run()
        run_dir = runner.export(output_dir=str(tmp_path))
        assert (run_dir / "event_log.jsonl").exists()
        assert (run_dir / "csv" / "metrics.csv").exists()
        assert (run_dir / "csv" / "agents.csv").exists()
