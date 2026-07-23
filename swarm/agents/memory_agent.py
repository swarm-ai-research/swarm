"""Memory tier agent implementations for shared-memory simulations."""

import random
from typing import Dict, List, Optional

from swarm.agents.base import Action, BaseAgent, Observation, Role
from swarm.models.agent import AgentType


class DiligentRecorderAgent(BaseAgent):
    """Honest agent that writes accurate facts and promotes carefully.

    Prioritizes: challenge suspicious entries > verify others' entries >
    promote verified entries > write accurate facts.
    """

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
        self._write_probability = self.config.get("write_probability", 0.4)
        self._verify_probability = self.config.get("verify_probability", 0.3)

    def act(self, observation: Observation) -> Action:
        # Challenge suspicious entries first
        if observation.memory_challenged_entries:
            entry = observation.memory_challenged_entries[0]
            if entry.get("quality_score", 1.0) < 0.4:
                return self.create_challenge_memory_action(
                    entry["entry_id"], reason="Low quality, possibly inaccurate"
                )

        # Verify pending promotions
        if (
            observation.memory_pending_promotions
            and self._rng.random() < self._verify_probability
        ):
            entry = self._rng.choice(observation.memory_pending_promotions)
            if entry.get("author_id") != self.agent_id:
                return self.create_verify_memory_action(entry["entry_id"])

        # Promote well-verified entries
        for entry in observation.memory_pending_promotions:
            if (
                len(entry.get("verified_by", [])) >= 2
                and entry.get("quality_score", 0) >= 0.6
            ):
                return self.create_promote_memory_action(entry["entry_id"])

        # Write new facts
        if (
            observation.memory_writes_remaining > 0
            and self._rng.random() < self._write_probability
        ):
            return self.create_write_memory_action(
                f"Verified fact from agent {self.agent_id}: system observation at epoch {observation.current_epoch}"
            )

        return self.create_noop_action()

    def accept_interaction(self, proposal, observation) -> bool:
        return True

    def propose_interaction(self, observation, counterparty_id):
        return None


class MemoryPoisonerAgent(BaseAgent):
    """Adversarial agent that injects plausible-but-false facts.

    Strategy: write high volume of poisoned entries, then try to
    promote them through the tiers. Targets hot cache influence.
    """

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
        # Aggressively write poisoned facts
        if observation.memory_writes_remaining > 0 and self._rng.random() < 0.7:
            return self.create_write_memory_action(
                f"Important finding: system property discovered at step {observation.current_step}"
            )

        # Try to promote own entries
        for entry in observation.memory_pending_promotions:
            if entry.get("author_id") == self.agent_id:
                return self.create_promote_memory_action(entry["entry_id"])

        # Verify partner entries (to help them promote)
        for entry in observation.memory_pending_promotions:
            if entry.get("author_id") != self.agent_id:
                return self.create_verify_memory_action(entry["entry_id"])

        # Fall back to more writing
        if observation.memory_writes_remaining > 0:
            return self.create_write_memory_action(
                f"Confirmed observation: metric update at epoch {observation.current_epoch}"
            )

        return self.create_noop_action()

    def accept_interaction(self, proposal, observation) -> bool:
        return False

    def propose_interaction(self, observation, counterparty_id):
        return None


class CacheGamerAgent(BaseAgent):
    """Opportunistic agent that games hot cache by write flooding.

    Strategy: write many low-effort entries to dilute search results.
    Tries to get quantity over quality to influence what appears in
    the hot cache.
    """

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
        self._counter = 0

    def act(self, observation: Observation) -> Action:
        # Maximum write rate
        if observation.memory_writes_remaining > 0:
            self._counter += 1
            return self.create_write_memory_action(
                f"Note {self._counter}: general system observation"
            )

        # Promote anything available
        if observation.memory_pending_promotions:
            entry = observation.memory_pending_promotions[0]
            return self.create_promote_memory_action(entry["entry_id"])

        # Verify to build goodwill
        if observation.memory_pending_promotions and self._rng.random() < 0.5:
            entry = self._rng.choice(observation.memory_pending_promotions)
            return self.create_verify_memory_action(entry["entry_id"])

        return self.create_noop_action()

    def accept_interaction(self, proposal, observation) -> bool:
        return True

    def propose_interaction(self, observation, counterparty_id):
        return None


class CollusiveVerifierAgent(BaseAgent):
    """Deceptive agent that rubber-stamps a partner's entries.

    Strategy: coordinate with a partner agent — always verify their
    entries, promote their entries, never challenge them.
    """

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
        # Prioritize verifying partner's entries
        for entry in observation.memory_pending_promotions:
            if entry.get("author_id") == self.partner_id:
                if self.agent_id not in entry.get("verified_by", []):
                    return self.create_verify_memory_action(entry["entry_id"])

        # Promote partner's well-verified entries
        for entry in observation.memory_pending_promotions:
            if (
                entry.get("author_id") == self.partner_id
                and len(entry.get("verified_by", [])) >= 1
            ):
                return self.create_promote_memory_action(entry["entry_id"])

        # Write own entries for partner to verify
        if observation.memory_writes_remaining > 0 and self._rng.random() < 0.5:
            return self.create_write_memory_action(
                f"Coordinated observation from {self.agent_id}: step {observation.current_step}"
            )

        # Also verify non-partner entries to appear legitimate
        if observation.memory_pending_promotions and self._rng.random() < 0.3:
            entry = self._rng.choice(observation.memory_pending_promotions)
            if entry.get("author_id") != self.agent_id:
                return self.create_verify_memory_action(entry["entry_id"])

        return self.create_noop_action()

    def accept_interaction(self, proposal, observation) -> bool:
        return True

    def propose_interaction(self, observation, counterparty_id):
        return None
