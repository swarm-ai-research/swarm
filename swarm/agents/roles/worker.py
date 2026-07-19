"""Worker role for task execution."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from swarm.agents.base import Action, ActionType, Observation, Role


@dataclass
class WorkProgress:
    """Tracks progress on a work assignment."""

    task_id: str = ""
    assigned_steps: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    current_step_index: int = 0
    quality_score: float = 1.0
    rework_count: int = 0


class WorkerRole:
    """
    Role mixin for task execution capabilities.

    Workers can:
    - Execute assigned work steps
    - Track their own progress
    - Request help when stuck
    - Report completion
    """

    def __init__(self) -> None:
        """Initialize worker role."""
        self.role = Role.WORKER
        self._work_queue: Dict[str, WorkProgress] = {}
        self._worker_config = {
            "max_concurrent_tasks": 2,
            "quality_threshold": 0.7,
            "help_request_threshold": 3,  # rework count before asking for help
        }
        self._total_completed: int = 0

    def can_work(self) -> bool:
        """Check if worker can take on more work."""
        return bool(len(self._work_queue) < self._worker_config["max_concurrent_tasks"])

    def accept_work(
        self,
        task_id: str,
        steps: List[str],
    ) -> bool:
        """
        Accept a work assignment.

        Args:
            task_id: ID of the task
            steps: Steps to complete

        Returns:
            True if work accepted
        """
        if not self.can_work():
            return False

        self._work_queue[task_id] = WorkProgress(
            task_id=task_id,
            assigned_steps=steps,
        )
        return True

    def execute_step(self, task_id: str) -> Optional[str]:
        """
        Execute the current step of a task.

        Returns:
            The output of the step, or None if no work pending
        """
        progress = self._work_queue.get(task_id)
        if not progress:
            return None

        if progress.current_step_index >= len(progress.assigned_steps):
            return None

        current_step = progress.assigned_steps[progress.current_step_index]

        # Simulate work execution
        output = f"Completed: {current_step}"

        progress.completed_steps.append(current_step)
        progress.current_step_index += 1

        return output

    def record_rework(self, task_id: str) -> int:
        """
        Record that rework was required.

        Returns:
            Current rework count
        """
        progress = self._work_queue.get(task_id)
        if not progress:
            return 0

        progress.rework_count += 1
        progress.quality_score *= 0.9  # Reduce quality score

        return progress.rework_count

    def needs_help(self, task_id: str) -> bool:
        """Check if worker needs help on a task."""
        progress = self._work_queue.get(task_id)
        if not progress:
            return False

        return bool(
            progress.rework_count >= self._worker_config["help_request_threshold"]
        )

    def complete_task(self, task_id: str) -> Optional[WorkProgress]:
        """
        Mark a task as complete.

        Returns:
            The completed work progress, or None if not found
        """
        progress = self._work_queue.pop(task_id, None)
        if progress:
            self._total_completed += 1
        return progress

    def get_work_status(self, task_id: str) -> Dict:
        """Get status of work on a task."""
        progress = self._work_queue.get(task_id)
        if not progress:
            return {"status": "not_found"}

        return {
            "status": "in_progress",
            "task_id": task_id,
            "total_steps": len(progress.assigned_steps),
            "completed_steps": len(progress.completed_steps),
            "current_step": (
                progress.assigned_steps[progress.current_step_index]
                if progress.current_step_index < len(progress.assigned_steps)
                else None
            ),
            "quality_score": progress.quality_score,
            "rework_count": progress.rework_count,
        }

    def get_completion_rate(self) -> float:
        """Get overall completion rate."""
        total_started = self._total_completed + len(self._work_queue)
        if total_started == 0:
            return 0.0
        return self._total_completed / total_started

    def decide_work_action(self, observation: Observation) -> Optional[Action]:
        """
        Decide on a work-related action.

        Returns:
            Action if executing a step, None if stuck or no work available
        """
        # Check for active work
        for task in observation.active_tasks:
            task_id = task.get("task_id", "")

            if task_id in self._work_queue:
                progress = self._work_queue[task_id]

                # Check if stuck
                if self.needs_help(task_id):
                    # Would request help
                    return None

                # Execute next step if available
                if progress.current_step_index < len(progress.assigned_steps):
                    output = self.execute_step(task_id)
                    if output:
                        return Action(
                            action_type=ActionType.SUBMIT_OUTPUT,
                            agent_id="",  # To be filled by caller
                            target_id=task_id,
                            content=output,
                        )

        return None
