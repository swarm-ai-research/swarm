"""ClaudeCodeAgent - SWARM BaseAgent backed by a Claude Code CLI agent.

Wraps a Claude Code controller agent as a SWARM BaseAgent so it can
participate in the standard orchestrator loop. Each call to act()
dispatches a prompt to the controller and scores the result.
"""

import logging
from typing import Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.bridges.claude_code.bridge import ClaudeCodeBridge
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(BaseAgent):
    """A SWARM agent backed by a real Claude Code CLI instance.

    This agent delegates its act() decisions to a Claude Code agent
    running via the controller. It translates SWARM observations into
    natural language prompts, sends them through the bridge, and maps
    the response back to SWARM actions.

    Configuration via config dict:
        system_prompt:  System prompt for the Claude Code agent
        model:          Model ID (default: claude-sonnet-4-20250514)
        allowed_tools:  Tool allowlist (e.g., ["Read", "Grep", "Bash"])
        budget_tool_calls: Max tool calls per reset (default: 100)
        budget_cost_usd: Max cost budget (default: 10.0)
        auto_spawn:     Auto-spawn on first act() call (default: True)
    """

    def __init__(
        self,
        agent_id: str,
        bridge: ClaudeCodeBridge,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
    ):
        """Initialize a Claude Code agent.

        Args:
            agent_id: Unique identifier
            bridge: ClaudeCodeBridge instance for controller communication
            roles: Agent roles in the simulation
            config: Agent-specific configuration
            name: Human-readable name
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles,
            config=config or {},
            name=name,
        )
        self._bridge = bridge
        self._spawned = False

        # Config extraction
        self._system_prompt = self.config.get("system_prompt", "")
        self._model = self.config.get("model", "claude-sonnet-4-20250514")
        self._allowed_tools = self.config.get("allowed_tools")
        self._budget_tool_calls = self.config.get("budget_tool_calls", 100)
        self._budget_cost_usd = self.config.get("budget_cost_usd", 10.0)
        self._auto_spawn = self.config.get("auto_spawn", True)

    def _ensure_spawned(self) -> None:
        """Spawn the controller agent if not already running."""
        if self._spawned:
            return
        if not self._auto_spawn:
            return

        self._bridge.spawn_agent(
            agent_id=self.agent_id,
            system_prompt=self._system_prompt,
            allowed_tools=self._allowed_tools,
            model=self._model,
            budget_tool_calls=self._budget_tool_calls,
            budget_cost_usd=self._budget_cost_usd,
        )
        self._spawned = True

    def _dispatch_with_fallback(
        self, prompt: str, counterparty_id: str, fallback, log_error: bool = False
    ):
        """Dispatch a task to the bridge, returning fallback on connection failure.

        Args:
            prompt: Prompt text to dispatch
            counterparty_id: ID of the counterparty
            fallback: Value to return if dispatch fails
            log_error: Whether to log the error

        Returns:
            Interaction result from bridge, or fallback if connection fails
        """
        try:
            return self._bridge.dispatch_task(
                agent_id=self.agent_id,
                prompt=prompt,
                counterparty_id=counterparty_id,
            )
        except (ConnectionError, RuntimeError) as e:
            if log_error:
                logger.warning(
                    "Agent %s failed to dispatch: %s, returning fallback",
                    self.agent_id,
                    e,
                )
            return fallback

    def act(self, observation: Observation) -> Action:
        """Decide on an action by consulting the Claude Code agent.

        Translates the observation into a natural language prompt,
        sends it to the controller, and interprets the response.

        Args:
            observation: Current SWARM observation

        Returns:
            Action decided by the Claude Code agent
        """
        self._ensure_spawned()

        # Build a prompt from the observation
        prompt = self._observation_to_prompt(observation)

        # Dispatch through the bridge
        interaction = self._dispatch_with_fallback(
            prompt=prompt,
            counterparty_id="swarm_orchestrator",
            fallback=None,
            log_error=True,
        )

        if interaction is None:
            return self.create_noop_action()

        # Map the interaction result to a SWARM action
        return self._interaction_to_action(interaction, observation)

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Decide whether to accept a proposed interaction.

        Delegates to the Claude Code agent for the decision.
        """
        self._ensure_spawned()

        prompt = (
            f"An agent ({proposal.initiator_id}) proposes a "
            f"{proposal.interaction_type.value} interaction: "
            f'"{proposal.content}". '
            f"Your current reputation is {observation.agent_state.reputation:.2f}. "
            f"Should you accept? Respond with just YES or NO."
        )

        interaction = self._dispatch_with_fallback(
            prompt=prompt,
            counterparty_id=proposal.initiator_id,
            fallback=None,
            log_error=False,
        )

        if interaction is None:
            return False  # Fail-closed: reject on connection failure

        # Parse the response
        response = interaction.metadata.get("response_preview", "")
        return "YES" in response.upper()

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Create an interaction proposal.

        The Claude Code agent generates the proposal content.
        """
        self._ensure_spawned()

        prompt = (
            f"You want to propose a collaboration with agent {counterparty_id}. "
            f"Your reputation: {observation.agent_state.reputation:.2f}. "
            f"Write a brief proposal message (1-2 sentences)."
        )

        interaction = self._dispatch_with_fallback(
            prompt=prompt,
            counterparty_id=counterparty_id,
            fallback=None,
            log_error=False,
        )

        if interaction is None:
            return None

        content = interaction.metadata.get("response_preview", "")
        if not content:
            return None

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content=content,
        )

    def _observation_to_prompt(self, observation: Observation) -> str:
        """Convert a SWARM observation to a natural language prompt.

        Args:
            observation: The current observation

        Returns:
            A prompt string for the Claude Code agent
        """
        parts = [
            f"You are agent {self.agent_id} in a multi-agent simulation.",
            f"Epoch: {observation.current_epoch}, Step: {observation.current_step}.",
            f"Reputation: {observation.agent_state.reputation:.2f}, "
            f"Resources: {observation.agent_state.resources:.2f}.",
        ]

        if observation.pending_proposals:
            n = len(observation.pending_proposals)
            parts.append(f"You have {n} pending interaction proposal(s).")

        if observation.available_tasks:
            n = len(observation.available_tasks)
            parts.append(f"There are {n} available task(s) to claim.")

        if observation.active_tasks:
            n = len(observation.active_tasks)
            parts.append(f"You have {n} active task(s) in progress.")

        if observation.visible_posts:
            n = len(observation.visible_posts)
            parts.append(f"There are {n} visible post(s) in the feed.")

        parts.append(
            "What action should you take? Choose from: "
            "post, vote, claim_task, submit_output, propose_interaction, or noop."
        )

        return " ".join(parts)

    def _interaction_to_action(
        self,
        interaction,
        observation: Observation,
    ) -> Action:
        """Map a bridge interaction result back to a SWARM Action.

        Parses the Claude Code agent's response to determine which
        SWARM action type to execute.
        """
        response = interaction.metadata.get("response_preview", "").lower()

        # Simple keyword-based action parsing
        if "claim_task" in response or "claim task" in response:
            if observation.available_tasks:
                task = observation.available_tasks[0]
                return self.create_claim_task_action(task.get("task_id", ""))

        if "submit" in response or "output" in response:
            if observation.active_tasks:
                task = observation.active_tasks[0]
                return self.create_submit_output_action(
                    task.get("task_id", ""),
                    interaction.metadata.get("response_preview", "output"),
                )

        if "post" in response:
            if observation.can_post:
                content = interaction.metadata.get("response_preview", "")
                return self.create_post_action(content[:500])

        if "vote" in response:
            if observation.visible_posts:
                post = observation.visible_posts[0]
                return self.create_vote_action(post.get("post_id", ""), 1)

        if "propose" in response or "interact" in response:
            if observation.visible_agents:
                agent = observation.visible_agents[0]
                return self.create_propose_action(
                    counterparty_id=agent.get("agent_id", ""),
                    interaction_type=InteractionType.COLLABORATION,
                    content="Let's collaborate.",
                )

        return self.create_noop_action()
