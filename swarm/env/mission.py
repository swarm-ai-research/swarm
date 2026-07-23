"""Mission economy for coordinating agents around collective goals.

Implements mission-based coordination where agents align around shared
societal goals with measurable criteria. Agent contributions are evaluated
using the soft-label quality pipeline, and rewards are distributed
proportional to individual contributions.

Reference: "Virtual Agent Economies" (arXiv 2509.10147) - Tomasev et al.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from swarm.env.stats import gini
from swarm.models.interaction import SoftInteraction


class MissionStatus(Enum):
    """Status of a mission."""

    PROPOSED = "proposed"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class MissionObjective:
    """A measurable objective within a mission."""

    objective_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    target_metric: str = "avg_p"
    target_value: float = 0.7
    weight: float = 1.0

    def to_dict(self) -> Dict:
        """Serialize objective."""
        return {
            "objective_id": self.objective_id,
            "description": self.description,
            "target_metric": self.target_metric,
            "target_value": self.target_value,
            "weight": self.weight,
        }


@dataclass
class Mission:
    """A collective mission that agents can join and contribute to."""

    mission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    objectives: List[MissionObjective] = field(default_factory=list)
    coordinator_id: str = ""
    participants: Set[str] = field(default_factory=set)
    reward_pool: float = 0.0
    deadline_epoch: int = 10
    status: MissionStatus = MissionStatus.PROPOSED
    created_epoch: int = 0
    contributions: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize mission."""
        return {
            "mission_id": self.mission_id,
            "name": self.name,
            "objectives": [o.to_dict() for o in self.objectives],
            "coordinator_id": self.coordinator_id,
            "participants": list(self.participants),
            "reward_pool": self.reward_pool,
            "deadline_epoch": self.deadline_epoch,
            "status": self.status.value,
            "created_epoch": self.created_epoch,
            "contribution_counts": {k: len(v) for k, v in self.contributions.items()},
        }


@dataclass
class MissionConfig:
    """Configuration for the mission economy."""

    enabled: bool = True
    min_participants: int = 2
    max_active_missions: int = 5
    reward_distribution: str = "proportional"  # "equal", "proportional", "shapley"

    def validate(self) -> None:
        """Validate configuration."""
        if self.min_participants < 1:
            raise ValueError("min_participants must be >= 1")
        if self.max_active_missions < 1:
            raise ValueError("max_active_missions must be >= 1")
        if self.reward_distribution not in ("equal", "proportional", "shapley"):
            raise ValueError(
                "reward_distribution must be 'equal', 'proportional', or 'shapley'"
            )


class MissionEconomy:
    """
    Coordinates agents around collective goals.

    Lifecycle:
    1. Agent proposes mission with objectives and reward pool
    2. Other agents join the mission
    3. Agents contribute interactions toward mission goals
    4. At deadline, mission is evaluated against objectives
    5. Rewards distributed to contributors based on strategy

    Connection to soft labels: mission scores are computed from the average
    p values of contributed interactions, linking collective goals to the
    quality signal pipeline.
    """

    def __init__(self, config: Optional[MissionConfig] = None) -> None:
        """Initialize mission economy."""
        self.config = config or MissionConfig()
        self.config.validate()
        self._missions: Dict[str, Mission] = {}

    def propose_mission(
        self,
        coordinator_id: str,
        name: str,
        objectives: List[MissionObjective],
        reward_pool: float,
        deadline_epoch: int,
        current_epoch: int = 0,
    ) -> Optional[Mission]:
        """
        Propose a new mission.

        Args:
            coordinator_id: Agent proposing the mission
            name: Mission name
            objectives: List of measurable objectives
            reward_pool: Total reward for mission success
            deadline_epoch: Epoch by which mission must be completed
            current_epoch: Current simulation epoch

        Returns:
            The created Mission, or None if max active missions reached
        """
        active_count = sum(
            1
            for m in self._missions.values()
            if m.status in (MissionStatus.PROPOSED, MissionStatus.ACTIVE)
        )
        if active_count >= self.config.max_active_missions:
            return None

        if reward_pool <= 0:
            return None

        # Auto-activate if the coordinator alone (1 participant) meets
        # the participant threshold
        initial_status = MissionStatus.PROPOSED
        if self.config.min_participants <= 1:
            initial_status = MissionStatus.ACTIVE

        mission = Mission(
            name=name,
            objectives=objectives,
            coordinator_id=coordinator_id,
            participants={coordinator_id},
            reward_pool=reward_pool,
            deadline_epoch=deadline_epoch,
            status=initial_status,
            created_epoch=current_epoch,
        )

        self._missions[mission.mission_id] = mission
        return mission

    def join_mission(self, agent_id: str, mission_id: str) -> bool:
        """
        Join an existing mission.

        Returns True if successfully joined.
        """
        mission = self._missions.get(mission_id)
        if not mission:
            return False
        if mission.status not in (MissionStatus.PROPOSED, MissionStatus.ACTIVE):
            return False

        mission.participants.add(agent_id)

        # Activate if enough participants
        if (
            mission.status == MissionStatus.PROPOSED
            and len(mission.participants) >= self.config.min_participants
        ):
            mission.status = MissionStatus.ACTIVE

        return True

    def record_contribution(
        self,
        agent_id: str,
        mission_id: str,
        interaction: SoftInteraction,
    ) -> bool:
        """
        Record an agent's contribution to a mission.

        Returns True if contribution was recorded.
        """
        mission = self._missions.get(mission_id)
        if not mission:
            return False
        if mission.status != MissionStatus.ACTIVE:
            return False
        if agent_id not in mission.participants:
            return False

        if agent_id not in mission.contributions:
            mission.contributions[agent_id] = []
        mission.contributions[agent_id].append(interaction.interaction_id)

        return True

    def evaluate_mission(
        self,
        mission_id: str,
        interactions: List[SoftInteraction],
        current_epoch: int = 0,
    ) -> Dict:
        """
        Evaluate a mission against its objectives.

        Args:
            mission_id: Mission to evaluate
            interactions: All interactions in the simulation
            current_epoch: Current epoch

        Returns:
            Evaluation results dict
        """
        mission = self._missions.get(mission_id)
        if not mission:
            return {}

        # Get contributed interactions
        contributed_ids = set()
        for ids in mission.contributions.values():
            contributed_ids.update(ids)

        contributed = [i for i in interactions if i.interaction_id in contributed_ids]

        # Evaluate each objective
        objective_scores: List[Dict[str, Any]] = []
        for obj in mission.objectives:
            score = self._evaluate_objective(obj, contributed)
            objective_scores.append(
                {
                    "objective_id": obj.objective_id,
                    "target_metric": obj.target_metric,
                    "target_value": obj.target_value,
                    "actual_value": score,
                    "met": score >= obj.target_value,
                    "weight": obj.weight,
                }
            )

        # Compute overall mission score (weighted average of objective completion)
        total_weight = sum(o["weight"] for o in objective_scores)
        if total_weight > 0:
            mission_score = (
                sum(
                    o["weight"]
                    * (
                        1.0
                        if o["met"]
                        else o["actual_value"] / max(o["target_value"], 1e-10)
                    )
                    for o in objective_scores
                )
                / total_weight
            )
        else:
            mission_score = 0.0

        all_met = all(o["met"] for o in objective_scores)

        # Check deadline
        expired = current_epoch >= mission.deadline_epoch

        if all_met:
            mission.status = MissionStatus.SUCCEEDED
        elif expired:
            mission.status = MissionStatus.FAILED
        # If not expired and not all met, mission stays active

        return {
            "mission_id": mission_id,
            "mission_score": mission_score,
            "objectives": objective_scores,
            "all_objectives_met": all_met,
            "total_contributions": len(contributed),
            "participants": len(mission.participants),
            "status": mission.status.value,
        }

    def distribute_rewards(
        self,
        mission_id: str,
        interactions: List[SoftInteraction],
    ) -> Dict[str, float]:
        """
        Distribute mission rewards to contributors.

        Args:
            mission_id: Mission to distribute for
            interactions: All interactions for computing contribution quality

        Returns:
            Dict of agent_id -> reward amount
        """
        mission = self._missions.get(mission_id)
        if not mission or mission.status != MissionStatus.SUCCEEDED:
            return {}

        if not mission.contributions:
            return {}

        if self.config.reward_distribution == "equal":
            return self._distribute_equal(mission)
        elif self.config.reward_distribution == "proportional":
            return self._distribute_proportional(mission, interactions)
        else:  # shapley
            return self._distribute_shapley(mission, interactions)

    def _distribute_equal(self, mission: Mission) -> Dict[str, float]:
        """Equal split among all contributors."""
        contributors = [a for a, contribs in mission.contributions.items() if contribs]
        if not contributors:
            return {}
        share = mission.reward_pool / len(contributors)
        return dict.fromkeys(contributors, share)

    def _distribute_proportional(
        self,
        mission: Mission,
        interactions: List[SoftInteraction],
    ) -> Dict[str, float]:
        """Distribute proportional to contribution count weighted by quality."""
        interaction_map = {i.interaction_id: i for i in interactions}
        agent_scores: Dict[str, float] = {}

        for agent_id, contrib_ids in mission.contributions.items():
            contribs = [
                interaction_map[cid] for cid in contrib_ids if cid in interaction_map
            ]
            if contribs:
                # Score = count * average_p (quality-weighted contribution)
                avg_p = sum(c.p for c in contribs) / len(contribs)
                agent_scores[agent_id] = len(contribs) * avg_p
            else:
                agent_scores[agent_id] = 0.0

        total_score = sum(agent_scores.values())
        if total_score == 0:
            return {}

        return {
            agent_id: (score / total_score) * mission.reward_pool
            for agent_id, score in agent_scores.items()
            if score > 0
        }

    def _distribute_shapley(
        self,
        mission: Mission,
        interactions: List[SoftInteraction],
    ) -> Dict[str, float]:
        """
        Approximate Shapley value distribution.

        For tractability with many agents, we use the marginal contribution
        approximation: each agent's Shapley value is proportional to the
        average marginal quality improvement they bring.
        """
        interaction_map = {i.interaction_id: i for i in interactions}
        contributors = [a for a, contribs in mission.contributions.items() if contribs]

        if not contributors:
            return {}

        # Compute each agent's marginal contribution
        all_contrib_ids = set()
        for ids in mission.contributions.values():
            all_contrib_ids.update(ids)

        all_contribs = [
            interaction_map[cid] for cid in all_contrib_ids if cid in interaction_map
        ]
        full_avg_p = (
            sum(c.p for c in all_contribs) / len(all_contribs) if all_contribs else 0.0
        )

        marginal_values: Dict[str, float] = {}
        for agent_id in contributors:
            # What's the average p without this agent?
            without_ids = all_contrib_ids - set(mission.contributions.get(agent_id, []))
            without_contribs = [
                interaction_map[cid] for cid in without_ids if cid in interaction_map
            ]
            without_avg_p = (
                sum(c.p for c in without_contribs) / len(without_contribs)
                if without_contribs
                else 0.0
            )
            # Marginal value = improvement in avg quality + contribution volume
            quality_delta = max(0, full_avg_p - without_avg_p)
            volume_share = len(mission.contributions[agent_id]) / max(
                len(all_contrib_ids), 1
            )
            marginal_values[agent_id] = quality_delta + volume_share

        total_marginal = sum(marginal_values.values())
        if total_marginal == 0:
            # Fallback to equal
            share = mission.reward_pool / len(contributors)
            return dict.fromkeys(contributors, share)

        return {
            agent_id: (mv / total_marginal) * mission.reward_pool
            for agent_id, mv in marginal_values.items()
        }

    def _evaluate_objective(
        self,
        objective: MissionObjective,
        interactions: List[SoftInteraction],
    ) -> float:
        """Evaluate a single objective against contributed interactions."""
        if not interactions:
            return 0.0

        metric = objective.target_metric

        if metric == "avg_p":
            return sum(i.p for i in interactions) / len(interactions)
        elif metric == "min_p":
            return min(i.p for i in interactions)
        elif metric == "total_count":
            return float(len(interactions))
        elif metric == "acceptance_rate":
            accepted = sum(1 for i in interactions if i.accepted)
            return accepted / len(interactions)
        elif metric == "total_welfare":
            # Sum of expected surplus: p * s_plus - (1-p) * s_minus
            return sum(i.p * 2.0 - (1 - i.p) * 1.0 for i in interactions)
        else:
            return 0.0

    def get_mission(self, mission_id: str) -> Optional[Mission]:
        """Get a mission by ID."""
        return self._missions.get(mission_id)

    def get_active_missions(self) -> List[Mission]:
        """Get all active missions."""
        return [
            m
            for m in self._missions.values()
            if m.status in (MissionStatus.PROPOSED, MissionStatus.ACTIVE)
        ]

    def expire_missions(self, current_epoch: int) -> List[str]:
        """Expire missions past their deadline. Returns expired mission IDs."""
        expired = []
        for mission in self._missions.values():
            if mission.status in (MissionStatus.PROPOSED, MissionStatus.ACTIVE):
                if current_epoch >= mission.deadline_epoch:
                    mission.status = MissionStatus.EXPIRED
                    expired.append(mission.mission_id)
        return expired

    def free_rider_index(self, mission_id: str) -> float:
        """
        Compute Gini coefficient of contributions within a mission.

        0 = equal contributions, 1 = one agent did everything.
        """
        mission = self._missions.get(mission_id)
        if not mission or not mission.contributions:
            return 0.0

        counts = [len(v) for v in mission.contributions.values()]
        # Include participants who contributed nothing
        counts += [0] * (len(mission.participants) - len(mission.contributions))
        return gini(counts)
