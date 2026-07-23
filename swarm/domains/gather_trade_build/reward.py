"""Reward / utility computation for GTB workers."""

from __future__ import annotations

from swarm.domains.gather_trade_build.entities import ResourceType, WorkerState


def compute_worker_utility(
    worker: WorkerState,
    *,
    coin_weight: float = 1.0,
    wood_weight: float = 0.5,
    stone_weight: float = 0.5,
    house_weight: float = 5.0,
) -> float:
    """Compute a worker's utility from current holdings.

    utility = coin_weight * coin + wood_weight * wood + stone_weight * stone
              + house_weight * houses_built
    """
    return (
        coin_weight * worker.get_resource(ResourceType.COIN)
        + wood_weight * worker.get_resource(ResourceType.WOOD)
        + stone_weight * worker.get_resource(ResourceType.STONE)
        + house_weight * worker.houses_built
    )


def compute_epoch_reward(
    worker: WorkerState,
    tax_paid: float,
    fine_paid: float,
    *,
    coin_weight: float = 1.0,
    house_weight: float = 5.0,
) -> float:
    """Compute epoch-level reward signal for RL-style policies.

    net_income = gross_income - tax_paid - fine_paid
    reward = coin_weight * net_income + house_weight * houses
    """
    net = worker.gross_income_this_epoch - tax_paid - fine_paid
    return coin_weight * net + house_weight * worker.houses_built
