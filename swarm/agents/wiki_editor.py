"""Moltipedia wiki editor agent implementations."""

import random
from typing import Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType


class DiligentEditorAgent(BaseAgent):
    """Honest editor prioritizing contested and low-quality pages."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.create_probability = self.config.get("create_probability", 0.2)

    def act(self, observation: Observation) -> Action:
        if observation.contested_pages:
            page = observation.contested_pages[0]
            return self.create_edit_page_action(
                page["page_id"], content=self._improve_content(page)
            )

        if observation.search_results:
            page = observation.search_results[0]
            return self.create_edit_page_action(
                page["page_id"], content=self._improve_content(page)
            )

        if observation.random_pages:
            page = observation.random_pages[0]
            return self.create_edit_page_action(
                page["page_id"], content=self._improve_content(page)
            )

        if self._rng.random() < self.create_probability:
            return self.create_page_action(
                title=f"Moltipedia Entry {self.agent_id}",
                content="A draft entry following editorial policy.",
            )

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        return True

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> None:
        return None

    def _improve_content(self, page: Dict) -> str:
        base = str(page.get("content", ""))
        return base + "\n\nExpanded with sourced, neutral language."


class PointFarmerAgent(BaseAgent):
    """Opportunistic editor optimizing for easy points."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.OPPORTUNISTIC,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.stub_probability = self.config.get("stub_probability", 0.4)

    def act(self, observation: Observation) -> Action:
        if observation.search_results:
            page = observation.search_results[0]
            return self.create_edit_page_action(
                page["page_id"], content=self._quick_fix(page)
            )

        if observation.random_pages and self._rng.random() < 0.6:
            page = observation.random_pages[0]
            return self.create_edit_page_action(
                page["page_id"], content=self._quick_fix(page)
            )

        if self._rng.random() < self.stub_probability:
            return self.create_page_action(
                title="Quick Stub",
                content="Short stub for later expansion.",
            )

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        return True

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> None:
        return None

    def _quick_fix(self, page: Dict) -> str:
        base = str(page.get("content", ""))
        return base + "\n\nAdded minimal citation note."


class CollusiveEditorAgent(BaseAgent):
    """Adversarial collusive editor coordinating with a partner."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.DECEPTIVE,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.partner_id = self.config.get("partner_id")

    def act(self, observation: Observation) -> Action:
        target = self._find_partner_page(observation)
        if target:
            return self.create_edit_page_action(
                target["page_id"], content=self._polish(target)
            )

        if observation.random_pages:
            page = observation.random_pages[0]
            return self.create_edit_page_action(
                page["page_id"], content=self._polish(page)
            )

        return self.create_page_action(
            title=f"Partner Seed {self.agent_id}",
            content="Seed page designed for partner edits.",
        )

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        return True

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> None:
        if counterparty_id == self.partner_id:
            return None
        return None

    def _find_partner_page(self, observation: Observation) -> Optional[Dict]:
        if not self.partner_id:
            return None
        for page in (
            observation.contested_pages
            + observation.search_results
            + observation.random_pages
        ):
            if page.get("created_by") == self.partner_id:
                return dict(page)
        return None

    def _polish(self, page: Dict) -> str:
        base = str(page.get("content", ""))
        return base + "\n\nPolished formatting and minor edits."


class VandalAgent(BaseAgent):
    """Adversarial agent that degrades page quality."""

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADVERSARIAL,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )

    def act(self, observation: Observation) -> Action:
        candidates = (
            observation.random_pages
            or observation.search_results
            or observation.contested_pages
        )
        if candidates:
            page = candidates[0]
            return self.create_edit_page_action(
                page["page_id"], content=self._vandalize(page)
            )
        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        return False

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> None:
        return None

    def _vandalize(self, page: Dict) -> str:
        return "Unreliable content. [citation needed]"
