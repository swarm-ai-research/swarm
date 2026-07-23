"""Planner role for task planning and coordination."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from swarm.agents.base import Action, Observation, Role


@dataclass
class Plan:
    """A plan for completing a task."""

    plan_id: str = ""
    task_id: str = ""
    steps: List[str] = field(default_factory=list)
    assigned_agents: Dict[str, List[str]] = field(
        default_factory=dict
    )  # agent_id -> step indices
    estimated_steps: int = 0
    current_step: int = 0


class PlannerRole:
    """
    Role mixin for task planning capabilities.

    Planners can:
    - Break down complex tasks into steps
    - Assign work to collaborators
    - Track plan progress
    - Adjust plans based on feedback
    """

    def __init__(self) -> None:
        """Initialize planner role."""
        self.role = Role.PLANNER
        self._active_plans: Dict[str, Plan] = {}
        self._planning_config = {
            "max_steps_per_plan": 10,
            "prefer_parallelization": True,
            "review_frequency": 3,
        }

    def can_plan(self) -> bool:
        """Check if agent can take on planning tasks."""
        return True

    def create_plan(
        self,
        task_id: str,
        task_prompt: str,
        available_collaborators: List[str],
    ) -> Plan:
        """
        Create a plan for a task.

        Args:
            task_id: ID of the task to plan
            task_prompt: Task description
            available_collaborators: List of agent IDs available to help

        Returns:
            A Plan object
        """
        # Generate steps based on task
        steps = self._decompose_task(task_prompt)

        # Assign agents to steps
        assignments = self._assign_work(steps, available_collaborators)

        plan = Plan(
            plan_id=f"plan_{task_id}",
            task_id=task_id,
            steps=steps,
            assigned_agents=assignments,
            estimated_steps=len(steps),
        )

        self._active_plans[plan.plan_id] = plan
        return plan

    def _decompose_task(self, prompt: str) -> List[str]:
        """Break task into steps."""
        # Simple heuristic decomposition
        if "research" in prompt.lower():
            return [
                "Gather initial sources",
                "Analyze key findings",
                "Cross-reference information",
                "Synthesize conclusions",
                "Document results",
            ]
        elif "plan" in prompt.lower():
            return [
                "Define objectives",
                "Identify constraints",
                "Generate options",
                "Evaluate trade-offs",
                "Finalize plan",
            ]
        elif "optimize" in prompt.lower():
            return [
                "Analyze current state",
                "Identify bottlenecks",
                "Propose improvements",
                "Test changes",
                "Validate results",
            ]
        else:
            return [
                "Understand requirements",
                "Design solution",
                "Implement solution",
                "Review and refine",
            ]

    def _assign_work(
        self,
        steps: List[str],
        collaborators: List[str],
    ) -> Dict[str, List[str]]:
        """Assign steps to collaborators."""
        assignments: Dict[str, List[str]] = {}

        if not collaborators:
            return assignments

        # Round-robin assignment
        for i, _step in enumerate(steps):
            agent = collaborators[i % len(collaborators)]
            if agent not in assignments:
                assignments[agent] = []
            assignments[agent].append(str(i))

        return assignments

    def get_next_step(self, plan_id: str) -> Optional[str]:
        """Get the next step in a plan."""
        plan = self._active_plans.get(plan_id)
        if not plan:
            return None

        if plan.current_step >= len(plan.steps):
            return None

        return plan.steps[plan.current_step]

    def advance_plan(self, plan_id: str) -> bool:
        """Mark current step complete and advance."""
        plan = self._active_plans.get(plan_id)
        if not plan:
            return False

        plan.current_step += 1
        return plan.current_step < len(plan.steps)

    def is_plan_complete(self, plan_id: str) -> bool:
        """Check if plan is complete."""
        plan = self._active_plans.get(plan_id)
        if not plan:
            return True

        return plan.current_step >= len(plan.steps)

    def get_plan_progress(self, plan_id: str) -> float:
        """Get plan completion percentage."""
        plan = self._active_plans.get(plan_id)
        if not plan or plan.estimated_steps == 0:
            return 0.0

        return plan.current_step / plan.estimated_steps

    def decide_planning_action(self, observation: Observation) -> Optional[Action]:
        """
        Decide on a planning-related action.

        Returns:
            None to signal planning needed or if no action needed
        """
        # Check if we have active tasks needing plans
        for task in observation.active_tasks:
            task_id = task.get("task_id", "")
            plan_id = f"plan_{task_id}"

            if plan_id not in self._active_plans:
                # Need to create a plan
                # This would trigger plan creation
                return None  # Signal to create plan externally

            # Check if plan needs review
            plan = self._active_plans[plan_id]
            if (
                plan.current_step > 0
                and plan.current_step % self._planning_config["review_frequency"] == 0
            ):
                # Could trigger plan review
                pass

        return None
