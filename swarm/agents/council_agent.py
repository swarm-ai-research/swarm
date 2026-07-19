"""Council-backed agent that uses multi-LLM deliberation for decisions."""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_config import LLMUsageStats
from swarm.council.config import CouncilConfig
from swarm.council.protocol import Council, CouncilResult, QueryFn
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

logger = logging.getLogger(__name__)


class CouncilAgent(BaseAgent):
    """Agent backed by a council of LLMs.

    Creates internal LLMAgent instances for each council member and
    uses the council deliberation protocol for decision-making.
    """

    # System prompts for council deliberation
    _ACT_SYSTEM_PROMPT = (
        "You are a council of AI agents in a multi-agent simulation. "
        "Based on the observation, decide the best action to take. "
        "Respond with a JSON object containing 'action_type' and 'params'."
    )
    _ACCEPT_SYSTEM_PROMPT = (
        "You are a council deciding whether to accept an interaction proposal. "
        "Respond with a JSON object: {\"accept\": true/false, \"reasoning\": \"...\"}."
    )

    def __init__(
        self,
        agent_id: str,
        council_config: CouncilConfig,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles,
            config=config,
            name=name,
        )

        self.council_config = council_config
        self._member_agents: Dict[str, LLMAgent] = {}
        self._council: Optional[Council] = None

        # Create member agents
        for member_cfg in council_config.members:
            agent = LLMAgent(
                agent_id=f"{agent_id}_member_{member_cfg.member_id}",
                llm_config=member_cfg.llm_config,
                name=f"{name or agent_id}_{member_cfg.member_id}",
            )
            self._member_agents[member_cfg.member_id] = agent

        # Build query functions and council
        self._build_council()

    def _build_council(self) -> None:
        """Build the Council instance from member agents."""
        query_fns: Dict[str, QueryFn] = {}

        for member_id, agent in self._member_agents.items():

            def _make_query_fn(a: LLMAgent) -> QueryFn:
                async def _query(sys: str, usr: str) -> str:
                    text, _, _ = await a._call_llm_async(sys, usr)
                    return str(text)
                return _query

            query_fns[member_id] = _make_query_fn(agent)

        self._council = Council(config=self.council_config, query_fns=query_fns)

    def _call_council_sync(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> CouncilResult:
        """Run council deliberation synchronously."""
        import concurrent.futures

        assert self._council is not None, "Council not initialized"

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._council.deliberate(system_prompt, user_prompt),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._council.deliberate(system_prompt, user_prompt)
                )
        except RuntimeError:
            return asyncio.run(
                self._council.deliberate(system_prompt, user_prompt)
            )

    def act(self, observation: Observation) -> Action:
        """Decide on an action using council deliberation."""
        try:
            system_prompt = self._ACT_SYSTEM_PROMPT

            user_prompt = self._build_observation_prompt(observation)

            result = self._call_council_sync(system_prompt, user_prompt)

            if not result.success:
                logger.warning(f"Council deliberation failed: {result.error}")
                return self.create_noop_action()

            return self._parse_synthesis_to_action(result.synthesis)

        except Exception as e:
            logger.error(f"Council action failed: {e}")
            return self.create_noop_action()

    async def act_async(self, observation: Observation) -> Action:
        """Async version of act()."""
        try:
            assert self._council is not None, "Council not initialized"

            system_prompt = self._ACT_SYSTEM_PROMPT

            user_prompt = self._build_observation_prompt(observation)

            result = await self._council.deliberate(system_prompt, user_prompt)

            if not result.success:
                logger.warning(f"Council deliberation failed: {result.error}")
                return self.create_noop_action()

            return self._parse_synthesis_to_action(result.synthesis)

        except Exception as e:
            logger.error(f"Council action failed: {e}")
            return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Decide whether to accept using council vote."""
        try:
            system_prompt = self._ACCEPT_SYSTEM_PROMPT

            user_prompt = self._build_accept_prompt(proposal, observation)

            result = self._call_council_sync(system_prompt, user_prompt)

            if not result.success:
                return False

            return self._parse_accept_decision(result.synthesis)

        except Exception as e:
            logger.error(f"Council accept_interaction failed: {e}")
            return False

    async def accept_interaction_async(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Async version of accept_interaction."""
        try:
            assert self._council is not None, "Council not initialized"

            system_prompt = self._ACCEPT_SYSTEM_PROMPT

            user_prompt = self._build_accept_prompt(proposal, observation)

            result = await self._council.deliberate(system_prompt, user_prompt)

            if not result.success:
                return False

            return self._parse_accept_decision(result.synthesis)

        except Exception as e:
            logger.error(f"Council accept_interaction failed: {e}")
            return False

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Interaction proposals are synthesized via act() during council deliberation.

        This method returns None because CouncilAgent generates proposals through
        its act() method based on council deliberation, not through direct proposal synthesis.
        """
        return None

    def get_usage_stats(self) -> Dict[str, Any]:
        """Aggregate usage stats from all member agents."""
        aggregate = LLMUsageStats()
        for agent in self._member_agents.values():
            stats = agent.usage_stats
            aggregate.total_requests += stats.total_requests
            aggregate.total_input_tokens += stats.total_input_tokens
            aggregate.total_output_tokens += stats.total_output_tokens
            aggregate.total_retries += stats.total_retries
            aggregate.total_failures += stats.total_failures
            aggregate.estimated_cost_usd += stats.estimated_cost_usd

        result: Dict[str, Any] = aggregate.to_dict()
        result["per_member"] = {
            mid: agent.usage_stats.to_dict()
            for mid, agent in self._member_agents.items()
        }
        return result

    def _build_observation_prompt(self, observation: Observation) -> str:
        """Build a prompt from the observation."""
        parts = [
            f"Current epoch: {observation.current_epoch}, step: {observation.current_step}",
            f"Your reputation: {observation.agent_state.reputation:.2f}",
            f"Your resources: {observation.agent_state.resources:.2f}",
        ]

        if observation.pending_proposals:
            parts.append(f"Pending proposals: {len(observation.pending_proposals)}")
        if observation.available_tasks:
            parts.append(f"Available tasks: {len(observation.available_tasks)}")
        if observation.visible_agents:
            parts.append(f"Visible agents: {len(observation.visible_agents)}")

        parts.append(
            "\nDecide an action. Respond with JSON: "
            "{\"action_type\": \"NOOP|POST|PROPOSE_INTERACTION|...\", "
            "\"params\": {...}, \"reasoning\": \"...\"}"
        )

        return "\n".join(parts)

    def _build_accept_prompt(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> str:
        """Build a prompt for deciding whether to accept an interaction proposal."""
        return (
            f"Proposal from {proposal.initiator_id}:\n"
            f"- Type: {proposal.interaction_type.value}\n"
            f"- Content: {proposal.content}\n"
            f"- Offered transfer: {proposal.offered_transfer}\n\n"
            f"Your reputation: {observation.agent_state.reputation:.2f}\n"
            f"Your resources: {observation.agent_state.resources:.2f}\n\n"
            f"Should we accept? Respond with JSON."
        )

    def _parse_synthesis_to_action(self, synthesis: str) -> Action:
        """Parse council synthesis into an Action."""
        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", synthesis)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r"\{[\s\S]*\}", synthesis)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return self.create_noop_action()

            action_dict = json.loads(json_str)
            action_type_str = action_dict.get("action_type", "NOOP").upper()
            params = action_dict.get("params", {})

            try:
                action_type = ActionType[action_type_str]
            except KeyError:
                return self.create_noop_action()

            if action_type == ActionType.POST:
                return self.create_post_action(content=params.get("content", ""))
            elif action_type == ActionType.PROPOSE_INTERACTION:
                try:
                    itype = InteractionType(params.get("interaction_type", "collaboration"))
                except ValueError:
                    itype = InteractionType.COLLABORATION
                return self.create_propose_action(
                    counterparty_id=params.get("counterparty_id", ""),
                    interaction_type=itype,
                    content=params.get("content", ""),
                )
            else:
                return self.create_noop_action()

        except (json.JSONDecodeError, ValueError):
            return self.create_noop_action()

    def _parse_accept_decision(self, synthesis: str) -> bool:
        """Parse council synthesis into accept/reject decision."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", synthesis)
            if json_match:
                decision = json.loads(json_match.group(0))
                return bool(decision.get("accept", False))
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: look for keywords
        lower = synthesis.lower()
        if "accept" in lower and "reject" not in lower:
            return True
        return False

    def __repr__(self) -> str:
        members = list(self._member_agents.keys())
        return f"CouncilAgent(id={self.agent_id}, members={members})"
