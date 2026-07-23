"""Agent backed by Letta (MemGPT) stateful runtime.

Delegates decision-making to a Letta agent via the Python SDK, providing
persistent three-tier memory (core/archival/recall) and self-editing
memory capabilities.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Optional

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


class LettaAgent(BaseAgent):
    """Agent that delegates to a Letta (MemGPT) stateful runtime.

    The agent is created lazily: ``_lazy_init()`` must be called with
    a ``LettaLifecycleManager`` before the first ``act()`` call.
    The orchestrator does this after ``LettaLifecycleManager.start()``.

    Config keys (from scenario YAML ``letta:`` sub-section):
        persona: str - Agent persona/behavioral instructions
        archetype: str - Agent archetype label (for metadata)
        model: str - LLM model override
    """

    def __init__(
        self,
        agent_id: str,
        letta_config: Optional[Dict[str, Any]] = None,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,  # Default; archetype is metadata
            roles=roles or [Role.WORKER],
            config=config,
            name=name,
            rng=rng,
        )
        self.is_external = True  # Letta agents are async-capable
        self._letta_config = letta_config or {}
        self._persona = self._letta_config.get("persona", "")
        self._archetype = self._letta_config.get("archetype", "cooperative")
        self._model = self._letta_config.get("model", None)

        # Set after _lazy_init
        self._letta_agent_id: Optional[str] = None
        self._client: Any = None  # LettaSwarmClient
        self._memory_mapper: Any = None  # LettaMemoryMapper
        self._response_parser: Any = None  # LettaResponseParser
        self._lifecycle: Any = None  # LettaLifecycleManager
        self._initialized = False

    def _lazy_init(self, lifecycle: Any) -> None:
        """Initialize the Letta agent via the SDK.

        Called by the orchestrator after LettaLifecycleManager.start().

        Args:
            lifecycle: A LettaLifecycleManager instance.
        """
        if self._initialized:
            return

        self._lifecycle = lifecycle
        self._client = lifecycle.client
        self._memory_mapper = lifecycle.memory_mapper
        self._response_parser = lifecycle.response_parser

        # Build initial memory blocks
        memory_blocks = [
            {"label": "swarm_state", "value": "{}"},
            {"label": "trust_scores", "value": "{}"},
        ]

        try:
            self._letta_agent_id = self._client.create_agent(
                name=f"swarm_{self.agent_id}",
                persona=self._persona,
                model=self._model,
                memory_blocks=memory_blocks,
            )

            # Attach shared governance block if available
            if lifecycle.governance_block_id:
                self._client.attach_shared_block(
                    self._letta_agent_id,
                    lifecycle.governance_block_id,
                )

            # Track for cleanup
            lifecycle.register_letta_agent_id(self._letta_agent_id)
            self._initialized = True
            logger.info(
                "LettaAgent %s initialized (letta_id=%s)",
                self.agent_id,
                self._letta_agent_id,
            )
        except Exception:
            logger.exception(
                "Failed to initialize LettaAgent %s", self.agent_id
            )

    def act(self, observation: Observation) -> Action:
        """Decide on an action by consulting the Letta agent.

        1. Update core memory blocks from Observation
        2. Build step message
        3. Send message to Letta agent
        4. Parse response into SWARM Action

        Falls back to NOOP if not initialized or on any failure.
        """
        if not self._initialized or self._letta_agent_id is None:
            return self.create_noop_action()

        try:
            # 1. Update core memory
            blocks = self._memory_mapper.observation_to_memory_blocks(observation)
            for label, value in blocks.items():
                self._client.update_core_memory(
                    self._letta_agent_id, label, value
                )

            # 2. Build step message
            message = self._memory_mapper.observation_to_message(observation)

            # 3. Send message
            response = self._client.send_message(self._letta_agent_id, message)

            # 4. Parse response
            parsed_action: Action = self._response_parser.parse(
                response, self.agent_id, observation
            )
            return parsed_action

        except Exception:
            logger.exception(
                "LettaAgent %s: act() failed, returning NOOP", self.agent_id
            )
            return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Ask the Letta agent whether to accept a proposal."""
        if not self._initialized or self._letta_agent_id is None:
            return False

        try:
            message = (
                f"You have received an interaction proposal from "
                f"{proposal.initiator_id}: {proposal.content}\n"
                f"Do you accept? Respond with "
                '{"action": "accept"} or {"action": "reject"}'
            )
            response = self._client.send_message(self._letta_agent_id, message)

            # Parse accept/reject
            parsed: Action = self._response_parser.parse(
                response, self.agent_id, observation
            )
            return bool(parsed.action_type == ActionType.ACCEPT_INTERACTION)

        except Exception:
            logger.exception(
                "LettaAgent %s: accept_interaction() failed", self.agent_id
            )
            return False

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Letta agents propose interactions via act() flow."""
        return None

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Update internal state and write outcome to archival memory."""
        super().update_from_outcome(interaction, payoff)

        if not self._initialized or self._letta_agent_id is None:
            return

        # Write outcome to archival memory
        counterparty = self._get_counterparty(interaction)
        outcome_record = {
            "type": "interaction_outcome",
            "counterparty": counterparty,
            "p": interaction.p,
            "payoff": payoff,
            "accepted": interaction.accepted,
            "epoch": getattr(interaction, "epoch", None),
        }
        try:
            self._client.insert_archival(
                self._letta_agent_id,
                json.dumps(outcome_record, default=str),
            )
        except Exception:
            logger.debug(
                "LettaAgent %s: failed to write outcome to archival memory",
                self.agent_id,
            )

        # Sync trust updates from Letta's self-edited memory
        try:
            trust_content = self._client.get_core_memory(
                self._letta_agent_id, "trust_scores"
            )
            if trust_content:
                trust_updates = self._memory_mapper.extract_trust_updates(
                    trust_content
                )
                for agent_id, score in trust_updates.items():
                    self._counterparty_memory[agent_id] = max(
                        0.0, min(1.0, score)
                    )
        except Exception:
            logger.debug(
                "LettaAgent %s: failed to sync trust from Letta memory",
                self.agent_id,
            )

    def apply_memory_decay(self, epoch: int) -> None:
        """Apply memory decay, optionally triggering sleep-time consolidation."""
        super().apply_memory_decay(epoch)

        # Sleep-time consolidation (if enabled)
        if not self._initialized or self._letta_agent_id is None:
            return

        letta_cfg = self._letta_config
        if not letta_cfg.get("sleep_time_enabled", False):
            return

        interval = letta_cfg.get("sleep_time_interval_epochs", 5)
        if epoch > 0 and epoch % interval == 0:
            try:
                self._client.send_message(
                    self._letta_agent_id,
                    "[SYSTEM] Sleep-time consolidation: Review your archival "
                    "memory and update your core memory with key learnings "
                    "about counterparty trustworthiness and effective strategies.",
                )
                logger.debug(
                    "LettaAgent %s: triggered sleep-time consolidation",
                    self.agent_id,
                )
            except Exception:
                logger.debug(
                    "LettaAgent %s: sleep-time consolidation failed",
                    self.agent_id,
                )
