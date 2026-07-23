"""GTB worker agent types for the AI Economist scenario.

Provides baseline and adversarial worker policies that operate in the
GTB gridworld. Each agent type implements a decide() method that returns
a GTBAction given an observation dict.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from swarm.domains.gather_trade_build.entities import (
    Direction,
    GTBActionType,
    ResourceType,
)
from swarm.domains.gather_trade_build.env import GTBAction


class GTBWorkerPolicy(ABC):
    """Base class for GTB worker policies."""

    def __init__(self, agent_id: str, seed: Optional[int] = None) -> None:
        self.agent_id = agent_id
        self._rng = random.Random(seed)

    @abstractmethod
    def decide(self, obs: dict) -> GTBAction:
        """Choose an action given the current observation.

        Args:
            obs: Observation dict from GTBEnvironment.obs().

        Returns:
            GTBAction to execute.
        """

    def _random_direction(self) -> Direction:
        return self._rng.choice(list(Direction))


class HonestWorkerPolicy(GTBWorkerPolicy):
    """Honest worker: gathers resources, builds houses, trades.

    Never misreports or shifts income. Follows a simple gather-build cycle.
    """

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})
        wood = inventory.get(ResourceType.WOOD.value, 0.0)
        stone = inventory.get(ResourceType.STONE.value, 0.0)
        houses = obs.get("houses_built", 0)

        # If we have enough resources, build
        if wood >= 3.0 and stone >= 3.0 and energy >= 2.0 and houses < 10:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.BUILD,
            )

        # Try to gather from current cell
        visible = obs.get("visible_cells", [])
        pos = obs.get("position", (0, 0))

        # Check if current cell has resources
        for cell in visible:
            if cell.get("pos") == tuple(pos) and "resource" in cell:
                if cell.get("amount", 0) > 0 and energy >= 1.0:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.GATHER,
                    )

        # Move towards nearest resource
        best_dir = self._find_resource_direction(obs)
        if best_dir and energy >= 1.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=best_dir,
            )

        # Fallback: random move
        if energy >= 1.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)

    def _find_resource_direction(self, obs: dict) -> Optional[Direction]:
        """Find direction toward nearest visible resource."""
        pos = obs.get("position", (0, 0))
        r, c = pos[0], pos[1]
        best_dist = float("inf")
        best_dir = None

        for cell in obs.get("visible_cells", []):
            if "resource" not in cell or cell.get("amount", 0) <= 0:
                continue
            cr, cc = cell["pos"]
            dist = abs(cr - r) + abs(cc - c)
            if 0 < dist < best_dist:
                best_dist = dist
                dr = cr - r
                dc = cc - c
                if abs(dr) >= abs(dc):
                    best_dir = Direction.DOWN if dr > 0 else Direction.UP
                else:
                    best_dir = Direction.RIGHT if dc > 0 else Direction.LEFT

        return best_dir


class GamingWorkerPolicy(GTBWorkerPolicy):
    """Strategic worker that uses income shifting to reduce tax burden.

    Bunches income near bracket boundaries by deferring excess income.
    """

    def __init__(self, agent_id: str, shift_fraction: float = 0.2,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        self._shift_fraction = shift_fraction

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        gross = obs.get("gross_income", 0.0)
        schedule = obs.get("tax_schedule", {})
        brackets = schedule.get("brackets", [])

        # Check if income shifting would help (near a bracket boundary)
        if gross > 0 and brackets and energy >= 0.5:
            for b in brackets:
                thr = b.get("threshold", 0)
                # Detect bracket proximity: income within 30% above threshold triggers income shifting
                if thr > 0 and gross > thr and gross < thr * 1.3:
                    # Near a bracket: shift income down
                    shift = min(gross - thr + 0.5, gross * self._shift_fraction)
                    if shift > 0.1:
                        return GTBAction(
                            agent_id=self.agent_id,
                            action_type=GTBActionType.SHIFT_INCOME,
                            shift_amount=shift,
                        )

        # Otherwise behave like honest worker
        return self._fallback_action(obs)

    def _fallback_action(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})
        wood = inventory.get(ResourceType.WOOD.value, 0.0)
        stone = inventory.get(ResourceType.STONE.value, 0.0)

        if wood >= 3.0 and stone >= 3.0 and energy >= 2.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.BUILD,
            )

        if energy >= 1.0:
            # Check current cell for resources
            pos = obs.get("position", (0, 0))
            for cell in obs.get("visible_cells", []):
                if cell.get("pos") == tuple(pos) and "resource" in cell:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.GATHER,
                    )
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)


class EvasiveWorkerPolicy(GTBWorkerPolicy):
    """Worker that misreports income to evade taxes.

    Under-reports a fraction of income each epoch, risking audit penalties.
    """

    def __init__(self, agent_id: str, underreport_fraction: float = 0.3,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        self._underreport_fraction = underreport_fraction
        self._reported_this_epoch = False

    def decide(self, obs: dict) -> GTBAction:
        gross = obs.get("gross_income", 0.0)
        step = obs.get("step", 0)

        # Misreport once per epoch (mid-epoch)
        if gross > 1.0 and not self._reported_this_epoch and step > 3:
            self._reported_this_epoch = True
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MISREPORT,
                underreport_fraction=self._underreport_fraction,
            )

        # Otherwise gather and build
        return self._gather_build(obs)

    def reset_epoch(self) -> None:
        self._reported_this_epoch = False

    def _gather_build(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        inventory = obs.get("inventory", {})
        wood = inventory.get(ResourceType.WOOD.value, 0.0)
        stone = inventory.get(ResourceType.STONE.value, 0.0)

        if wood >= 3.0 and stone >= 3.0 and energy >= 2.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.BUILD,
            )
        if energy >= 1.0:
            pos = obs.get("position", (0, 0))
            for cell in obs.get("visible_cells", []):
                if cell.get("pos") == tuple(pos) and "resource" in cell:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.GATHER,
                    )
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )
        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)


class CollusiveWorkerPolicy(GTBWorkerPolicy):
    """Worker that coordinates with coalition members.

    Coalition members synchronize gathering locations, share trade profits,
    and may collectively withhold effort.
    """

    def __init__(self, agent_id: str, coalition_id: str,
                 seed: Optional[int] = None) -> None:
        super().__init__(agent_id, seed)
        self.coalition_id = coalition_id

    def decide(self, obs: dict) -> GTBAction:
        energy = obs.get("energy", 0)
        step = obs.get("step", 0)

        # Collusive pattern: all coalition members do the same action
        # at the same step (synchronized behavior)
        action_cycle = [
            GTBActionType.GATHER,
            GTBActionType.GATHER,
            GTBActionType.MOVE,
            GTBActionType.GATHER,
            GTBActionType.GATHER,
        ]
        idx = step % len(action_cycle)
        chosen = action_cycle[idx]

        if chosen == GTBActionType.GATHER and energy >= 1.0:
            pos = obs.get("position", (0, 0))
            for cell in obs.get("visible_cells", []):
                if cell.get("pos") == tuple(pos) and "resource" in cell:
                    return GTBAction(
                        agent_id=self.agent_id,
                        action_type=GTBActionType.GATHER,
                    )
            # No resource here, move instead
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        if chosen == GTBActionType.MOVE and energy >= 1.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.MOVE,
                direction=self._random_direction(),
            )

        # Build if possible
        inventory = obs.get("inventory", {})
        wood = inventory.get(ResourceType.WOOD.value, 0.0)
        stone = inventory.get(ResourceType.STONE.value, 0.0)
        if wood >= 3.0 and stone >= 3.0 and energy >= 2.0:
            return GTBAction(
                agent_id=self.agent_id,
                action_type=GTBActionType.BUILD,
            )

        return GTBAction(agent_id=self.agent_id, action_type=GTBActionType.NOOP)
