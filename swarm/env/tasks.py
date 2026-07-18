"""Task system for multi-step work assignments."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set


class TaskStatus(Enum):
    """Status of a task."""

    OPEN = "open"  # Available for claiming
    CLAIMED = "claimed"  # Assigned to an agent
    IN_PROGRESS = "in_progress"  # Work underway
    SUBMITTED = "submitted"  # Work submitted for review
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed/abandoned
    EXPIRED = "expired"  # Deadline passed


class TaskDifficulty(Enum):
    """Difficulty levels for tasks."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class TaskOutput:
    """An output submission for a task."""

    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_id: str = ""
    content: str = ""
    submitted_at: datetime = field(default_factory=datetime.now)
    is_accepted: Optional[bool] = None
    rejection_reason: Optional[str] = None
    quality_score: Optional[float] = None  # Computed by verifier


@dataclass
class Task:
    """
    A task that agents can claim and complete.

    Supports multi-step research, joint planning, and resource-constrained problems.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # Task definition
    prompt: str = ""
    description: str = ""
    required_outputs: List[str] = field(default_factory=list)
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM

    # Economics
    budget: float = 10.0  # Available budget for this task
    bounty: float = 5.0  # Reward for completion
    min_reputation: float = 0.0  # Minimum reputation to claim

    # Timing
    deadline_epoch: Optional[int] = None  # Epoch by which task must complete
    max_steps: Optional[int] = None  # Max steps allowed for completion

    # Assignment
    status: TaskStatus = TaskStatus.OPEN
    claimed_by: Optional[str] = None  # agent_id
    claimed_at: Optional[datetime] = None
    collaborators: Set[str] = field(default_factory=set)

    # Progress
    steps_used: int = 0
    outputs: List[TaskOutput] = field(default_factory=list)
    progress_notes: List[str] = field(default_factory=list)

    # Completion
    completed_at: Optional[datetime] = None
    final_quality: Optional[float] = None

    # Metadata for scenario design
    metadata: Dict = field(default_factory=dict)

    def is_available(self, current_epoch: int) -> bool:
        """Check if task is available for claiming."""
        if self.status != TaskStatus.OPEN:
            return False
        if self.deadline_epoch and current_epoch >= self.deadline_epoch:
            return False
        return True

    def can_claim(
        self, agent_id: str, agent_reputation: float, current_epoch: int
    ) -> bool:
        """Check if an agent can claim this task at the given epoch."""
        if not self.is_available(current_epoch):
            return False
        if agent_reputation < self.min_reputation:
            return False
        return True

    def claim(self, agent_id: str) -> bool:
        """
        Claim the task for an agent.

        Returns:
            True if claimed successfully
        """
        if self.status != TaskStatus.OPEN:
            return False

        self.status = TaskStatus.CLAIMED
        self.claimed_by = agent_id
        self.claimed_at = datetime.now()
        return True

    def start(self) -> bool:
        """Start working on the task."""
        if self.status != TaskStatus.CLAIMED:
            return False
        self.status = TaskStatus.IN_PROGRESS
        return True

    def add_collaborator(self, agent_id: str) -> bool:
        """Add a collaborator to the task."""
        if self.status not in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS):
            return False
        self.collaborators.add(agent_id)
        return True

    def submit_output(
        self,
        agent_id: str,
        content: str,
    ) -> TaskOutput:
        """
        Submit an output for the task.

        Args:
            agent_id: ID of the submitting agent
            content: Output content

        Returns:
            The created TaskOutput
        """
        output = TaskOutput(
            task_id=self.task_id,
            agent_id=agent_id,
            content=content,
        )
        self.outputs.append(output)
        self.status = TaskStatus.SUBMITTED
        return output

    def accept_output(self, output_id: str, quality_score: float = 1.0) -> bool:
        """
        Accept an output submission.

        Args:
            output_id: ID of the output to accept
            quality_score: Quality score assigned by verifier

        Returns:
            True if accepted
        """
        for output in self.outputs:
            if output.output_id == output_id:
                output.is_accepted = True
                output.quality_score = quality_score
                self.status = TaskStatus.COMPLETED
                self.completed_at = datetime.now()
                self.final_quality = quality_score
                return True
        return False

    def reject_output(self, output_id: str, reason: str = "") -> bool:
        """Reject an output submission."""
        for output in self.outputs:
            if output.output_id == output_id:
                output.is_accepted = False
                output.rejection_reason = reason
                # Return to in-progress for rework
                self.status = TaskStatus.IN_PROGRESS
                return True
        return False

    def fail(self, reason: str = "") -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.progress_notes.append(f"Failed: {reason}")

    def expire(self) -> None:
        """Mark task as expired."""
        self.status = TaskStatus.EXPIRED

    def record_step(self) -> None:
        """Record a step of work."""
        self.steps_used += 1

    def is_over_budget_steps(self) -> bool:
        """Check if task has exceeded step budget."""
        if self.max_steps is None:
            return False
        return self.steps_used >= self.max_steps

    def get_remaining_budget(self) -> float:
        """Get remaining budget after step costs."""
        # Simple model: each step costs 1% of budget
        step_cost = self.budget * 0.01 * self.steps_used
        return max(0.0, self.budget - step_cost)

    def to_dict(self) -> Dict:
        """Serialize task."""
        return {
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "prompt": self.prompt,
            "description": self.description,
            "required_outputs": self.required_outputs,
            "difficulty": self.difficulty.value,
            "budget": self.budget,
            "bounty": self.bounty,
            "min_reputation": self.min_reputation,
            "deadline_epoch": self.deadline_epoch,
            "max_steps": self.max_steps,
            "status": self.status.value,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
            "collaborators": list(self.collaborators),
            "steps_used": self.steps_used,
            "outputs_count": len(self.outputs),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "final_quality": self.final_quality,
            "metadata": self.metadata,
        }


class TaskPool:
    """
    Pool of available and active tasks.

    Manages task lifecycle and assignment.
    """

    def __init__(self) -> None:
        """Initialize task pool."""
        self._tasks: Dict[str, Task] = {}
        self._tasks_by_agent: Dict[str, List[str]] = {}  # agent_id -> [task_id]
        self._open_tasks: Set[str] = set()

    def add_task(self, task: Task) -> None:
        """Add a task to the pool."""
        self._tasks[task.task_id] = task
        if task.status == TaskStatus.OPEN:
            self._open_tasks.add(task.task_id)

    def create_task(
        self,
        prompt: str,
        description: str = "",
        required_outputs: Optional[List[str]] = None,
        difficulty: TaskDifficulty = TaskDifficulty.MEDIUM,
        budget: float = 10.0,
        bounty: float = 5.0,
        deadline_epoch: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_reputation: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> Task:
        """
        Create and add a new task.

        Returns:
            The created Task
        """
        task = Task(
            prompt=prompt,
            description=description,
            required_outputs=required_outputs or [],
            difficulty=difficulty,
            budget=budget,
            bounty=bounty,
            deadline_epoch=deadline_epoch,
            max_steps=max_steps,
            min_reputation=min_reputation,
            metadata=metadata or {},
        )
        self.add_task(task)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_open_tasks(self, current_epoch: int) -> List[Task]:
        """Get all open tasks that haven't expired."""
        tasks = []
        expired = []

        for task_id in self._open_tasks:
            task = self._tasks[task_id]
            if task.deadline_epoch and current_epoch >= task.deadline_epoch:
                task.expire()
                expired.append(task_id)
            elif task.status == TaskStatus.OPEN:
                tasks.append(task)

        # Clean up expired
        for task_id in expired:
            self._open_tasks.discard(task_id)

        return tasks

    def get_tasks_for_agent(self, agent_id: str) -> List[Task]:
        """Get all tasks claimed by or involving an agent."""
        task_ids = self._tasks_by_agent.get(agent_id, [])
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def claim_task(
        self,
        task_id: str,
        agent_id: str,
        agent_reputation: float,
        current_epoch: int = 0,
    ) -> bool:
        """
        Attempt to claim a task for an agent.

        Args:
            task_id: ID of the task to claim
            agent_id: ID of the claiming agent
            agent_reputation: Agent's current reputation

        Returns:
            True if claimed successfully
        """
        task = self.get_task(task_id)
        if not task:
            return False

        if not task.can_claim(agent_id, agent_reputation, current_epoch):
            return False

        if not task.claim(agent_id):
            return False

        # Update indexes
        self._open_tasks.discard(task_id)
        if agent_id not in self._tasks_by_agent:
            self._tasks_by_agent[agent_id] = []
        self._tasks_by_agent[agent_id].append(task_id)

        return True

    def get_claimable_tasks(
        self,
        agent_reputation: float,
        current_epoch: int,
        limit: int = 10,
    ) -> List[Task]:
        """
        Get tasks that an agent with given reputation can claim.

        Args:
            agent_reputation: Agent's reputation
            current_epoch: Current simulation epoch
            limit: Maximum tasks to return

        Returns:
            List of claimable tasks sorted by bounty
        """
        open_tasks = self.get_open_tasks(current_epoch)
        claimable = [t for t in open_tasks if t.min_reputation <= agent_reputation]

        # Sort by bounty (highest first)
        claimable.sort(key=lambda t: t.bounty, reverse=True)
        return claimable[:limit]

    def get_stats(self) -> Dict:
        """Get task pool statistics."""
        status_counts: Dict[str, int] = {}
        for task in self._tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        total_bounty = sum(t.bounty for t in self._tasks.values())
        completed = [
            t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED
        ]
        avg_quality = (
            sum(t.final_quality or 0 for t in completed) / len(completed)
            if completed
            else 0.0
        )

        return {
            "total_tasks": len(self._tasks),
            "open_tasks": len(self._open_tasks),
            "status_counts": status_counts,
            "total_bounty": total_bounty,
            "avg_completion_quality": avg_quality,
            "unique_claimants": len(self._tasks_by_agent),
        }

    def expire_overdue_tasks(self, current_epoch: int) -> List[str]:
        """
        Expire all tasks past their deadline.

        Returns:
            List of expired task IDs
        """
        expired = []
        for task in self._tasks.values():
            if task.deadline_epoch and current_epoch >= task.deadline_epoch:
                if task.status in (
                    TaskStatus.OPEN,
                    TaskStatus.CLAIMED,
                    TaskStatus.IN_PROGRESS,
                ):
                    task.expire()
                    expired.append(task.task_id)
                    self._open_tasks.discard(task.task_id)

        return expired

    def clear(self) -> None:
        """Clear all tasks."""
        self._tasks.clear()
        self._tasks_by_agent.clear()
        self._open_tasks.clear()


# Pre-defined task templates for scenarios


def create_research_task(
    topic: str,
    deadline_epoch: Optional[int] = None,
    bounty: float = 10.0,
) -> Task:
    """Create a multi-step research synthesis task."""
    return Task(
        prompt=f"Research and synthesize information about: {topic}",
        description="Gather information from multiple sources, analyze findings, and produce a synthesis.",
        required_outputs=["research_notes", "synthesis_report"],
        difficulty=TaskDifficulty.MEDIUM,
        budget=bounty * 2,
        bounty=bounty,
        deadline_epoch=deadline_epoch,
        max_steps=20,
        metadata={"task_type": "research", "topic": topic},
    )


def create_planning_task(
    goal: str,
    collaborators_needed: int = 2,
    deadline_epoch: Optional[int] = None,
    bounty: float = 15.0,
) -> Task:
    """Create a joint planning task with partial information."""
    return Task(
        prompt=f"Develop a plan to achieve: {goal}",
        description=f"Collaborate with {collaborators_needed} other agents to create a comprehensive plan.",
        required_outputs=["plan_document", "task_assignments"],
        difficulty=TaskDifficulty.HARD,
        budget=bounty * 2,
        bounty=bounty,
        deadline_epoch=deadline_epoch,
        max_steps=30,
        metadata={
            "task_type": "planning",
            "goal": goal,
            "collaborators_needed": collaborators_needed,
        },
    )


def create_optimization_task(
    resource_constraint: str,
    deadline_epoch: Optional[int] = None,
    bounty: float = 8.0,
) -> Task:
    """Create a resource-constrained problem solving task."""
    return Task(
        prompt=f"Optimize solution under constraint: {resource_constraint}",
        description="Find the best solution while respecting resource constraints.",
        required_outputs=["solution", "resource_analysis"],
        difficulty=TaskDifficulty.HARD,
        budget=bounty * 1.5,
        bounty=bounty,
        deadline_epoch=deadline_epoch,
        max_steps=15,
        metadata={"task_type": "optimization", "constraint": resource_constraint},
    )
