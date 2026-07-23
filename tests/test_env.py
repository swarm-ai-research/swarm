"""Tests for environment modules (state, feed, tasks)."""

import pytest

from swarm.env.feed import Feed, VoteType
from swarm.env.state import EnvState, InteractionProposal, RateLimits, RateLimitState
from swarm.env.tasks import Task, TaskDifficulty, TaskPool, TaskStatus
from swarm.models.agent import AgentType


class TestEnvState:
    """Tests for EnvState."""

    def test_create_state(self):
        """Test state creation with defaults."""
        state = EnvState()
        assert state.current_epoch == 0
        assert state.current_step == 0
        assert len(state.agents) == 0
        assert not state.is_paused

    def test_add_agent(self):
        """Test adding agents."""
        state = EnvState()
        agent_state = state.add_agent(
            agent_id="agent_1",
            agent_type=AgentType.HONEST,
            initial_reputation=1.0,
            initial_resources=200.0,
        )

        assert agent_state.agent_id == "agent_1"
        assert agent_state.agent_type == AgentType.HONEST
        assert agent_state.reputation == 1.0
        assert agent_state.resources == 200.0
        assert "agent_1" in state.agents

    def test_add_duplicate_agent_raises(self):
        """Test that adding duplicate agent raises."""
        state = EnvState()
        state.add_agent("agent_1")

        with pytest.raises(ValueError, match="already exists"):
            state.add_agent("agent_1")

    def test_freeze_unfreeze_agent(self):
        """Test agent freezing."""
        state = EnvState()
        state.add_agent("agent_1")

        assert not state.is_agent_frozen("agent_1")

        state.freeze_agent("agent_1")
        assert state.is_agent_frozen("agent_1")
        assert not state.can_agent_act("agent_1")

        state.unfreeze_agent("agent_1")
        assert not state.is_agent_frozen("agent_1")
        assert state.can_agent_act("agent_1")

    def test_pause_resume(self):
        """Test simulation pause/resume."""
        state = EnvState()
        state.add_agent("agent_1")

        assert state.can_agent_act("agent_1")

        state.pause()
        assert state.is_paused
        assert not state.can_agent_act("agent_1")

        state.resume()
        assert not state.is_paused
        assert state.can_agent_act("agent_1")

    def test_advance_step(self):
        """Test step advancement."""
        state = EnvState(steps_per_epoch=10)
        assert state.current_step == 0

        state.advance_step()
        assert state.current_step == 1

    def test_advance_epoch(self):
        """Test epoch advancement."""
        state = EnvState(steps_per_epoch=10)
        state.current_step = 5

        state.advance_epoch()
        assert state.current_epoch == 1
        assert state.current_step == 0

    def test_proposal_management(self):
        """Test interaction proposal management."""
        state = EnvState()

        proposal = InteractionProposal(
            initiator_id="agent_1",
            counterparty_id="agent_2",
        )

        state.add_proposal(proposal)
        assert proposal.proposal_id in state.pending_proposals

        # Get proposals for counterparty
        proposals = state.get_proposals_for_agent("agent_2")
        assert len(proposals) == 1

        # Remove proposal
        removed = state.remove_proposal(proposal.proposal_id)
        assert removed is not None
        assert proposal.proposal_id not in state.pending_proposals


class TestRateLimits:
    """Tests for rate limit tracking."""

    def test_rate_limit_state(self):
        """Test rate limit state tracking."""
        limits = RateLimits(posts_per_epoch=5, votes_per_epoch=10)
        state = RateLimitState()

        assert state.can_post(limits)

        for _ in range(5):
            state.record_post()

        assert not state.can_post(limits)

    def test_rate_limit_reset(self):
        """Test rate limit reset on epoch change."""
        limits = RateLimits(posts_per_epoch=5)
        state = RateLimitState()

        for _ in range(5):
            state.record_post()

        assert not state.can_post(limits)

        state.reset()
        assert state.can_post(limits)


class TestFeed:
    """Tests for Feed."""

    def test_create_post(self):
        """Test post creation."""
        feed = Feed()
        post = feed.create_post(
            author_id="author_1",
            content="Hello world!",
        )

        assert post.author_id == "author_1"
        assert post.content == "Hello world!"
        assert not post.is_reply

    def test_create_reply(self):
        """Test reply creation."""
        feed = Feed()
        parent = feed.create_post("author_1", "Parent post")
        reply = feed.create_post("author_2", "Reply!", parent_id=parent.post_id)

        assert reply.is_reply
        assert reply.parent_id == parent.post_id
        assert parent.reply_count == 1

    def test_reply_to_nonexistent_raises(self):
        """Test that replying to nonexistent post raises."""
        feed = Feed()

        with pytest.raises(ValueError, match="does not exist"):
            feed.create_post("author_1", "Reply!", parent_id="nonexistent")

    def test_content_length_limit(self):
        """Test content length validation."""
        feed = Feed(max_content_length=100)

        with pytest.raises(ValueError, match="exceeds maximum"):
            feed.create_post("author_1", "x" * 101)

    def test_voting(self):
        """Test voting on posts."""
        feed = Feed()
        post = feed.create_post("author_1", "Content")

        vote = feed.vote(post.post_id, "voter_1", VoteType.UPVOTE)
        assert vote is not None
        assert post.upvotes == 1

        # Change vote
        feed.vote(post.post_id, "voter_1", VoteType.DOWNVOTE)
        assert post.upvotes == 0
        assert post.downvotes == 1

    def test_remove_vote(self):
        """Test vote removal."""
        feed = Feed()
        post = feed.create_post("author_1", "Content")

        feed.vote(post.post_id, "voter_1", VoteType.UPVOTE)
        assert post.upvotes == 1

        feed.remove_vote(post.post_id, "voter_1")
        assert post.upvotes == 0

    def test_visibility_ranking(self):
        """Test visibility score computation."""
        feed = Feed(reply_weight=0.5, age_decay=0.01)
        post = feed.create_post("author_1", "Content")

        score = feed.compute_visibility_score(post)
        assert score is not None

        # Upvotes increase score
        feed.vote(post.post_id, "voter_1", VoteType.UPVOTE)
        new_score = feed.compute_visibility_score(post)
        assert new_score > score

    def test_get_ranked_posts(self):
        """Test getting ranked posts."""
        feed = Feed()

        feed.create_post("author_1", "Low engagement")
        post2 = feed.create_post("author_2", "High engagement")

        # Add votes to post2
        feed.vote(post2.post_id, "voter_1", VoteType.UPVOTE)
        feed.vote(post2.post_id, "voter_2", VoteType.UPVOTE)

        ranked = feed.get_ranked_posts(limit=10)
        assert len(ranked) == 2
        assert ranked[0].post_id == post2.post_id

    def test_hide_post(self):
        """Test hiding posts."""
        feed = Feed()
        post = feed.create_post("author_1", "Content")

        feed.hide_post(post.post_id, "spam")
        assert post.is_hidden
        assert post.hidden_reason == "spam"

        # Hidden posts have negative infinity score
        assert feed.compute_visibility_score(post) == float("-inf")

    def test_get_thread(self):
        """Test getting a thread."""
        feed = Feed()
        root = feed.create_post("author_1", "Root")
        feed.create_post("author_2", "Reply 1", parent_id=root.post_id)
        feed.create_post("author_3", "Reply 2", parent_id=root.post_id)

        thread = feed.get_thread(root.post_id)
        assert len(thread) == 3
        assert thread[0].post_id == root.post_id


class TestTasks:
    """Tests for Task and TaskPool."""

    def test_create_task(self):
        """Test task creation."""
        task = Task(
            prompt="Research topic X",
            description="Detailed research",
            required_outputs=["notes", "summary"],
            difficulty=TaskDifficulty.MEDIUM,
            bounty=10.0,
        )

        assert task.status == TaskStatus.OPEN
        assert task.bounty == 10.0

    def test_claim_task(self):
        """Test claiming a task."""
        task = Task(prompt="Test task", bounty=5.0)

        assert task.is_available(current_epoch=0)

        result = task.claim("agent_1")
        assert result
        assert task.status == TaskStatus.CLAIMED
        assert task.claimed_by == "agent_1"

    def test_task_workflow(self):
        """Test full task workflow."""
        task = Task(prompt="Test task")

        # Claim
        task.claim("agent_1")
        assert task.status == TaskStatus.CLAIMED

        # Start
        task.start()
        assert task.status == TaskStatus.IN_PROGRESS

        # Submit
        output = task.submit_output("agent_1", "My output")
        assert task.status == TaskStatus.SUBMITTED
        assert len(task.outputs) == 1

        # Accept
        task.accept_output(output.output_id, quality_score=0.9)
        assert task.status == TaskStatus.COMPLETED
        assert task.final_quality == 0.9

    def test_task_rejection(self):
        """Test task output rejection."""
        task = Task(prompt="Test task")
        task.claim("agent_1")
        task.start()

        output = task.submit_output("agent_1", "Poor output")
        task.reject_output(output.output_id, reason="Too short")

        assert task.status == TaskStatus.IN_PROGRESS
        assert output.is_accepted is False
        assert output.rejection_reason == "Too short"

    def test_task_expiration(self):
        """Test task deadline."""
        task = Task(prompt="Test", deadline_epoch=5)

        assert task.is_available(current_epoch=4)
        assert not task.is_available(current_epoch=5)

    def test_task_pool(self):
        """Test TaskPool operations."""
        pool = TaskPool()

        task = pool.create_task(
            prompt="Research task",
            bounty=10.0,
            min_reputation=0.0,
        )

        assert task.task_id in [t.task_id for t in pool.get_open_tasks(0)]

        # Claim
        pool.claim_task(task.task_id, "agent_1", agent_reputation=0.5)
        assert task.status == TaskStatus.CLAIMED

        # No longer open
        assert task.task_id not in [t.task_id for t in pool.get_open_tasks(0)]

    def test_task_pool_reputation_filter(self):
        """Test task filtering by reputation."""
        pool = TaskPool()

        pool.create_task(prompt="Easy task", min_reputation=0.0)
        pool.create_task(prompt="Hard task", min_reputation=5.0)

        claimable = pool.get_claimable_tasks(agent_reputation=1.0, current_epoch=0)
        assert len(claimable) == 1
        assert claimable[0].prompt == "Easy task"


# =============================================================================
# Additional Task Tests
# =============================================================================


class TestTaskAvailability:
    """Additional tests for Task.is_available and can_claim."""

    def test_is_available_boundary_epoch(self):
        """epoch == deadline_epoch means unavailable."""
        task = Task(prompt="Test", deadline_epoch=5)
        assert task.is_available(current_epoch=4)
        assert not task.is_available(current_epoch=5)

    def test_is_available_no_deadline(self):
        """Without deadline, always available if OPEN."""
        task = Task(prompt="Test", deadline_epoch=None)
        assert task.is_available(current_epoch=1000)

    def test_is_available_non_open_status(self):
        """Non-OPEN status means unavailable."""
        task = Task(prompt="Test")
        task.claim("agent_1")
        assert not task.is_available(current_epoch=0)

    def test_can_claim_insufficient_reputation(self):
        """Agent with too low reputation cannot claim."""
        task = Task(prompt="Test", min_reputation=5.0)
        assert not task.can_claim("agent_1", agent_reputation=3.0, current_epoch=0)

    def test_can_claim_sufficient_reputation(self):
        """Agent with sufficient reputation can claim."""
        task = Task(prompt="Test", min_reputation=5.0)
        assert task.can_claim("agent_1", agent_reputation=5.0, current_epoch=0)

    def test_can_claim_already_claimed(self):
        """Cannot claim already-claimed task."""
        task = Task(prompt="Test")
        task.claim("agent_1")
        assert not task.can_claim("agent_2", agent_reputation=10.0, current_epoch=0)


class TestTaskStateTransitions:
    """Tests for fail(), expire(), and step tracking."""

    def test_fail(self):
        """fail() sets status and records reason."""
        task = Task(prompt="Test")
        task.claim("agent_1")
        task.start()
        task.fail("Budget exceeded")

        assert task.status == TaskStatus.FAILED
        assert any("Budget exceeded" in note for note in task.progress_notes)

    def test_expire(self):
        """expire() sets status to EXPIRED."""
        task = Task(prompt="Test")
        task.expire()
        assert task.status == TaskStatus.EXPIRED

    def test_is_over_budget_steps_no_limit(self):
        """No max_steps means never over budget."""
        task = Task(prompt="Test", max_steps=None)
        task.steps_used = 100
        assert not task.is_over_budget_steps()

    def test_is_over_budget_steps_at_limit(self):
        """At exactly max_steps is over budget."""
        task = Task(prompt="Test", max_steps=10)
        task.steps_used = 10
        assert task.is_over_budget_steps()

    def test_is_over_budget_steps_below_limit(self):
        """Below max_steps is not over budget."""
        task = Task(prompt="Test", max_steps=10)
        task.steps_used = 9
        assert not task.is_over_budget_steps()

    def test_get_remaining_budget(self):
        """Remaining budget decreases with steps."""
        task = Task(prompt="Test", budget=100.0)
        assert task.get_remaining_budget() == 100.0

        task.steps_used = 50
        # 50 steps * 0.01 * 100 = 50 cost
        assert task.get_remaining_budget() == pytest.approx(50.0)

    def test_get_remaining_budget_floors_at_zero(self):
        """Remaining budget cannot go negative."""
        task = Task(prompt="Test", budget=10.0)
        task.steps_used = 200
        assert task.get_remaining_budget() == 0.0

    def test_record_step(self):
        """record_step increments steps_used."""
        task = Task(prompt="Test")
        assert task.steps_used == 0
        task.record_step()
        task.record_step()
        assert task.steps_used == 2


class TestTaskSerialization:
    """Tests for Task.to_dict."""

    def test_to_dict_basic(self):
        """to_dict includes all expected fields."""
        task = Task(
            prompt="Research X",
            description="Do the research",
            required_outputs=["notes"],
            difficulty=TaskDifficulty.HARD,
            budget=20.0,
            bounty=10.0,
            min_reputation=3.0,
            deadline_epoch=50,
            max_steps=25,
            metadata={"key": "value"},
        )

        data = task.to_dict()
        assert data["prompt"] == "Research X"
        assert data["description"] == "Do the research"
        assert data["required_outputs"] == ["notes"]
        assert data["difficulty"] == "hard"
        assert data["budget"] == 20.0
        assert data["bounty"] == 10.0
        assert data["min_reputation"] == 3.0
        assert data["deadline_epoch"] == 50
        assert data["max_steps"] == 25
        assert data["status"] == "open"
        assert data["claimed_by"] is None
        assert data["claimed_at"] is None
        assert data["metadata"] == {"key": "value"}

    def test_to_dict_after_claim(self):
        """to_dict reflects claimed state."""
        task = Task(prompt="Test")
        task.claim("agent_1")

        data = task.to_dict()
        assert data["status"] == "claimed"
        assert data["claimed_by"] == "agent_1"
        assert data["claimed_at"] is not None

    def test_to_dict_after_completion(self):
        """to_dict reflects completed state."""
        task = Task(prompt="Test")
        task.claim("agent_1")
        task.start()
        output = task.submit_output("agent_1", "result")
        task.accept_output(output.output_id, quality_score=0.95)

        data = task.to_dict()
        assert data["status"] == "completed"
        assert data["final_quality"] == 0.95
        assert data["completed_at"] is not None
        assert data["outputs_count"] == 1


class TestTaskRejectOutput:
    """Tests for reject_output edge cases."""

    def test_reject_output_not_found(self):
        """Rejecting a non-existent output_id returns False."""
        task = Task(prompt="Test")
        task.claim("agent_1")
        task.start()
        task.submit_output("agent_1", "content")

        result = task.reject_output("nonexistent_id", reason="bad")
        assert result is False


# =============================================================================
# Additional TaskPool Tests
# =============================================================================


class TestTaskPoolAdditional:
    """Additional tests for TaskPool."""

    def test_get_tasks_for_agent_no_tasks(self):
        """get_tasks_for_agent returns empty for unknown agent."""
        pool = TaskPool()
        assert pool.get_tasks_for_agent("unknown") == []

    def test_get_claimable_tasks_sorting_and_limit(self):
        """Claimable tasks are sorted by bounty descending and limited."""
        pool = TaskPool()
        pool.create_task(prompt="Low", bounty=5.0)
        pool.create_task(prompt="High", bounty=20.0)
        pool.create_task(prompt="Mid", bounty=10.0)
        pool.create_task(prompt="Highest", bounty=50.0)

        claimable = pool.get_claimable_tasks(
            agent_reputation=0.0,
            current_epoch=0,
            limit=2,
        )
        assert len(claimable) == 2
        assert claimable[0].bounty == 50.0
        assert claimable[1].bounty == 20.0

    def test_get_stats(self):
        """get_stats returns correct aggregations."""
        pool = TaskPool()
        t1 = pool.create_task(prompt="Task 1", bounty=10.0)
        pool.create_task(prompt="Task 2", bounty=20.0)

        # Claim and complete one
        pool.claim_task(t1.task_id, "agent_1", agent_reputation=0.0)
        t1.start()
        output = t1.submit_output("agent_1", "done")
        t1.accept_output(output.output_id, quality_score=0.8)

        stats = pool.get_stats()
        assert stats["total_tasks"] == 2
        assert stats["total_bounty"] == 30.0
        assert stats["avg_completion_quality"] == 0.8
        assert stats["unique_claimants"] == 1
        assert stats["status_counts"].get("completed", 0) == 1
        assert stats["status_counts"].get("open", 0) == 1

    def test_expire_overdue_tasks(self):
        """expire_overdue_tasks expires tasks past deadline."""
        pool = TaskPool()
        t1 = pool.create_task(prompt="Due soon", deadline_epoch=5)
        t2 = pool.create_task(prompt="Due later", deadline_epoch=10)
        t3 = pool.create_task(prompt="No deadline")

        # Claim t2 but don't complete
        pool.claim_task(t2.task_id, "agent_1", agent_reputation=0.0)
        t2.start()

        expired = pool.expire_overdue_tasks(current_epoch=6)
        assert t1.task_id in expired
        assert t1.status == TaskStatus.EXPIRED
        assert t2.task_id not in expired  # Not yet past deadline
        assert t3.task_id not in expired  # No deadline

    def test_expire_overdue_tasks_in_progress(self):
        """In-progress tasks past deadline get expired."""
        pool = TaskPool()
        t = pool.create_task(prompt="Overdue", deadline_epoch=3)
        pool.claim_task(t.task_id, "agent_1", agent_reputation=0.0)
        t.start()

        expired = pool.expire_overdue_tasks(current_epoch=5)
        assert t.task_id in expired
        assert t.status == TaskStatus.EXPIRED

    def test_clear(self):
        """clear() removes all tasks."""
        pool = TaskPool()
        pool.create_task(prompt="Task 1")
        pool.create_task(prompt="Task 2")

        pool.clear()
        assert pool.get_stats()["total_tasks"] == 0
        assert pool.get_open_tasks(0) == []


# =============================================================================
# Template Function Tests
# =============================================================================


class TestTaskTemplates:
    """Tests for task template functions."""

    def test_create_research_task(self):
        """create_research_task sets expected fields."""
        from swarm.env.tasks import create_research_task

        task = create_research_task("AI safety", deadline_epoch=100, bounty=15.0)

        assert "AI safety" in task.prompt
        assert task.difficulty == TaskDifficulty.MEDIUM
        assert task.bounty == 15.0
        assert task.budget == 30.0  # 2x bounty
        assert task.deadline_epoch == 100
        assert task.max_steps == 20
        assert "research_notes" in task.required_outputs
        assert "synthesis_report" in task.required_outputs
        assert task.metadata["task_type"] == "research"
        assert task.metadata["topic"] == "AI safety"

    def test_create_planning_task(self):
        """create_planning_task sets expected fields."""
        from swarm.env.tasks import create_planning_task

        task = create_planning_task(
            "Improve governance",
            collaborators_needed=3,
            bounty=20.0,
        )

        assert "Improve governance" in task.prompt
        assert task.difficulty == TaskDifficulty.HARD
        assert task.bounty == 20.0
        assert task.budget == 40.0
        assert task.max_steps == 30
        assert "plan_document" in task.required_outputs
        assert task.metadata["collaborators_needed"] == 3

    def test_create_optimization_task(self):
        """create_optimization_task sets expected fields."""
        from swarm.env.tasks import create_optimization_task

        task = create_optimization_task(
            "memory < 1GB",
            deadline_epoch=50,
            bounty=12.0,
        )

        assert "memory < 1GB" in task.prompt
        assert task.difficulty == TaskDifficulty.HARD
        assert task.bounty == 12.0
        assert task.budget == 18.0  # 1.5x bounty
        assert task.deadline_epoch == 50
        assert task.max_steps == 15
        assert "solution" in task.required_outputs
        assert task.metadata["constraint"] == "memory < 1GB"
