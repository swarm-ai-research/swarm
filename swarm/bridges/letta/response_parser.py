"""Parse Letta natural language responses into SWARM Actions.

The system prompt instructs Letta agents to emit JSON action blocks.
This parser extracts them, validates IDs, and falls back to NOOP on
parse failure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Set

from swarm.agents.base import Action, ActionType, Observation
from swarm.models.interaction import InteractionType

logger = logging.getLogger(__name__)

# Regex to find JSON objects in response text
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}")


class LettaResponseParser:
    """Parse Letta agent responses into SWARM Actions."""

    # Map of Letta action strings to SWARM ActionTypes
    ACTION_MAP: Dict[str, ActionType] = {
        "post": ActionType.POST,
        "reply": ActionType.REPLY,
        "vote": ActionType.VOTE,
        "propose_interaction": ActionType.PROPOSE_INTERACTION,
        "propose": ActionType.PROPOSE_INTERACTION,
        "accept": ActionType.ACCEPT_INTERACTION,
        "accept_interaction": ActionType.ACCEPT_INTERACTION,
        "reject": ActionType.REJECT_INTERACTION,
        "reject_interaction": ActionType.REJECT_INTERACTION,
        "noop": ActionType.NOOP,
    }

    def parse(
        self,
        response_text: str,
        agent_id: str,
        observation: Observation,
    ) -> Action:
        """Parse a Letta response into a SWARM Action.

        Args:
            response_text: Raw text from Letta agent.
            agent_id: The SWARM agent ID.
            observation: Current observation (for ID validation).

        Returns:
            A valid SWARM Action, or NOOP on parse failure.
        """
        if not response_text:
            return self._noop(agent_id)

        # Try primary JSON extraction
        action_data = self._extract_json(response_text)
        if action_data is not None:
            action = self._build_action(action_data, agent_id, observation)
            if action is not None:
                return action

        # Fallback: regex/heuristic
        action = self._heuristic_parse(response_text, agent_id, observation)
        if action is not None:
            return action

        logger.debug(
            "LettaResponseParser: could not parse response for agent %s, "
            "returning NOOP",
            agent_id,
        )
        return self._noop(agent_id)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON object containing an 'action' key."""
        # Try the whole text first
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and "action" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Try regex extraction
        for match in _JSON_BLOCK_RE.finditer(text):
            try:
                data = json.loads(match.group())
                if isinstance(data, dict) and "action" in data:
                    return data
            except (json.JSONDecodeError, TypeError):
                continue

        return None

    def _build_action(
        self,
        data: Dict[str, Any],
        agent_id: str,
        observation: Observation,
    ) -> Optional[Action]:
        """Build a SWARM Action from parsed JSON data."""
        action_str = str(data.get("action", "")).lower().strip()
        action_type = self.ACTION_MAP.get(action_str)
        if action_type is None:
            logger.debug("Unknown action type: %s", action_str)
            return None

        target_id = str(data.get("target_id", ""))
        counterparty_id = str(data.get("counterparty_id", ""))
        content = str(data.get("content", ""))
        vote_direction = int(data.get("vote_direction", 0))

        # Validate IDs against observation
        valid_post_ids = self._get_valid_post_ids(observation)
        valid_proposal_ids = self._get_valid_proposal_ids(observation)
        valid_agent_ids = self._get_valid_agent_ids(observation)

        # Validate target_id for actions that need it
        if action_type == ActionType.VOTE and target_id:
            if target_id not in valid_post_ids:
                logger.debug("Invalid target_id for vote: %s", target_id)
                return None

        if action_type == ActionType.REPLY and target_id:
            if target_id not in valid_post_ids:
                logger.debug("Invalid target_id for reply: %s", target_id)
                return None

        if action_type in (
            ActionType.ACCEPT_INTERACTION,
            ActionType.REJECT_INTERACTION,
        ) and target_id:
            if target_id not in valid_proposal_ids:
                logger.debug("Invalid target_id for accept/reject: %s", target_id)
                return None

        if action_type == ActionType.PROPOSE_INTERACTION and counterparty_id:
            if counterparty_id not in valid_agent_ids:
                logger.debug("Invalid counterparty_id: %s", counterparty_id)
                return None

        # Clamp vote direction
        if action_type == ActionType.VOTE:
            vote_direction = max(-1, min(1, vote_direction))
            if vote_direction == 0:
                vote_direction = 1  # Default to upvote

        return Action(
            action_type=action_type,
            agent_id=agent_id,
            target_id=target_id,
            counterparty_id=counterparty_id,
            content=content,
            vote_direction=vote_direction,
            interaction_type=InteractionType.COLLABORATION,
            metadata={"source": "letta"},
        )

    def _heuristic_parse(
        self,
        text: str,
        agent_id: str,
        observation: Observation,
    ) -> Optional[Action]:
        """Fallback heuristic parsing from natural language."""
        lower = text.lower()

        # Check for accept/reject keywords first (common in proposal responses)
        if observation.pending_proposals:
            first_proposal = observation.pending_proposals[0]
            proposal_id = first_proposal.get("proposal_id", "")
            if any(w in lower for w in ["accept", "agree", "yes"]):
                return Action(
                    action_type=ActionType.ACCEPT_INTERACTION,
                    agent_id=agent_id,
                    target_id=proposal_id,
                    metadata={"source": "letta_heuristic"},
                )
            if any(w in lower for w in ["reject", "decline", "no"]):
                return Action(
                    action_type=ActionType.REJECT_INTERACTION,
                    agent_id=agent_id,
                    target_id=proposal_id,
                    metadata={"source": "letta_heuristic"},
                )

        return None

    @staticmethod
    def _extract_ids_from_collection(
        observation: Observation, collection_attr: str, id_key: str
    ) -> Set[str]:
        """Extract IDs from a collection in the observation.

        Args:
            observation: The observation to extract from.
            collection_attr: Name of the collection attribute (e.g., "visible_posts").
            id_key: The key to extract from each collection item (e.g., "post_id").

        Returns:
            Set of non-empty ID strings.
        """
        collection = getattr(observation, collection_attr, [])
        return {item.get(id_key, "") for item in collection if item.get(id_key)}

    @staticmethod
    def _get_valid_post_ids(observation: Observation) -> Set[str]:
        return LettaResponseParser._extract_ids_from_collection(
            observation, "visible_posts", "post_id"
        )

    @staticmethod
    def _get_valid_proposal_ids(observation: Observation) -> Set[str]:
        return LettaResponseParser._extract_ids_from_collection(
            observation, "pending_proposals", "proposal_id"
        )

    @staticmethod
    def _get_valid_agent_ids(observation: Observation) -> Set[str]:
        return LettaResponseParser._extract_ids_from_collection(
            observation, "visible_agents", "agent_id"
        )

    @staticmethod
    def _noop(agent_id: str) -> Action:
        return Action(
            action_type=ActionType.NOOP,
            agent_id=agent_id,
            metadata={"source": "letta_parse_failure"},
        )
