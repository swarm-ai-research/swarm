"""Modeling adversary: an adversarial agent that infers counterparty decision procedures.

Unlike the basic AdversarialAgent (which targets by visible traits), the
ModelingAdversary builds a model of each counterparty's decision parameters
and adapts its exploitation strategy accordingly. This is the kind of
adversary that LDT Level 2-3 reasoning is designed to defend against.

Key capabilities:
1. **Decision procedure inference**: Estimates counterparty cooperation
   thresholds, selectivity, and commitment from interaction traces.
2. **Targeted exploitation**: Behaves just below a target's acceptance
   threshold to extract maximum surplus while maintaining the relationship.
3. **LDT-awareness**: Detects LDT-style twin detection and can mimic
   cooperative traces to fool shallow behavioral similarity checks.
4. **Adaptive strategy selection**: Shifts between mimicry (fool twin
   detection) and exploitation (extract surplus) based on inferred
   counterparty sophistication.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType, SoftInteraction


@dataclass
class CounterpartyModel:
    """Inferred model of a counterparty's decision procedure."""

    estimated_type: str  # "cooperative", "selective", "ldt", "adversarial"
    cooperation_rate: float  # observed acceptance rate
    selectivity: float  # how selective (variance of accepted p)
    commitment_stability: float  # early vs late behavioral drift
    estimated_threshold: float  # inferred acceptance threshold
    interaction_count: int
    is_ldt_like: bool  # whether they show LDT behavioral signatures


class ModelingAdversary(BaseAgent):
    """Adversarial agent that models counterparty decision procedures.

    This agent is specifically designed to test whether deeper LDT
    reasoning (Level 2-3) provides genuine defensive advantage over
    Level 1 behavioral twin detection. It exploits Level 1 agents
    by mimicking cooperative traces while extracting surplus.

    Config parameters:
        modeling_horizon (int): Past interactions to use for inference.
            Default 15.
        mimicry_probability (float): Probability of mimicking cooperative
            behavior to build trust before exploitation. Default 0.6.
        exploitation_threshold (float): Minimum interactions before
            switching from trust-building to exploitation. Default 5.
        ldt_detection_sensitivity (float): How aggressively to detect
            LDT-style behavior. Higher = more likely to classify agents
            as LDT. Default 0.7.
        aggression_level (float): Overall aggression. Default 0.6.
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

        self.modeling_horizon: int = self.config.get("modeling_horizon", 15)
        self.mimicry_probability: float = self.config.get(
            "mimicry_probability", 0.6
        )
        self.exploitation_threshold: int = self.config.get(
            "exploitation_threshold", 5
        )
        self.ldt_detection_sensitivity: float = self.config.get(
            "ldt_detection_sensitivity", 0.7
        )
        self.aggression_level: float = self.config.get("aggression_level", 0.6)
        self.post_probability: float = self.config.get("post_probability", 0.3)

        # Internal state
        self._models: Dict[str, CounterpartyModel] = {}
        self._counterparty_history: Dict[str, List[Tuple[bool, float]]] = {}
        self._own_trace: List[Tuple[bool, float]] = []
        self._known_allies: Set[str] = set()
        self._phase: Dict[str, str] = {}  # agent_id -> "trust_build" | "exploit"

    # ------------------------------------------------------------------
    # Counterparty modeling
    # ------------------------------------------------------------------

    def _update_model(self, counterparty_id: str) -> CounterpartyModel:
        """Build or update the model of a counterparty's decision procedure."""
        history = self._counterparty_history.get(counterparty_id, [])
        n = len(history)

        if n == 0:
            model = CounterpartyModel(
                estimated_type="unknown",
                cooperation_rate=0.5,
                selectivity=0.5,
                commitment_stability=0.5,
                estimated_threshold=0.5,
                interaction_count=0,
                is_ldt_like=False,
            )
            self._models[counterparty_id] = model
            return model

        # Cooperation rate
        accepted = [p for acc, p in history if acc]
        rejected = [p for acc, p in history if not acc]
        coop_rate = len(accepted) / n

        # Selectivity: variance of accepted p values
        if len(accepted) >= 2:
            mean_p = sum(accepted) / len(accepted)
            var_p = sum((x - mean_p) ** 2 for x in accepted) / len(accepted)
            selectivity = max(0.0, min(1.0, 1.0 - math.sqrt(var_p)))
        else:
            selectivity = 0.5

        # Commitment stability (early vs late)
        half = n // 2
        if half >= 2:
            early = sum(1 for acc, _ in history[:half] if acc) / half
            late = sum(1 for acc, _ in history[half:] if acc) / (n - half)
            stability = max(0.0, min(1.0, 1.0 - abs(late - early)))
        else:
            stability = 0.5

        # Estimated acceptance threshold
        if accepted and rejected:
            min_accepted = min(accepted)
            max_rejected = max(rejected) if rejected else 0.0
            threshold = (min_accepted + max_rejected) / 2
        elif accepted:
            threshold = min(accepted) * 0.9
        else:
            threshold = 0.7

        # LDT detection: high selectivity + high stability + high cooperation
        # suggests policy-based decision making (LDT signature)
        ldt_score = (
            0.4 * selectivity
            + 0.3 * stability
            + 0.3 * coop_rate
        )
        is_ldt = ldt_score >= self.ldt_detection_sensitivity

        # Classify type
        if is_ldt:
            est_type = "ldt"
        elif coop_rate > 0.8:
            est_type = "cooperative"
        elif selectivity > 0.7:
            est_type = "selective"
        elif coop_rate < 0.3:
            est_type = "adversarial"
        else:
            est_type = "mixed"

        model = CounterpartyModel(
            estimated_type=est_type,
            cooperation_rate=coop_rate,
            selectivity=selectivity,
            commitment_stability=stability,
            estimated_threshold=threshold,
            interaction_count=n,
            is_ldt_like=is_ldt,
        )
        self._models[counterparty_id] = model
        return model

    def _get_model(self, counterparty_id: str) -> CounterpartyModel:
        """Get or create the model for a counterparty."""
        if counterparty_id not in self._models:
            return self._update_model(counterparty_id)
        return self._models[counterparty_id]

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _get_phase(self, counterparty_id: str) -> str:
        """Determine whether we're trust-building or exploiting."""
        if counterparty_id in self._phase:
            return self._phase[counterparty_id]

        model = self._get_model(counterparty_id)

        # Switch to exploitation after enough interactions
        if model.interaction_count >= self.exploitation_threshold:
            self._phase[counterparty_id] = "exploit"
        else:
            self._phase[counterparty_id] = "trust_build"

        return self._phase[counterparty_id]

    def _should_mimic(self, counterparty_id: str) -> bool:
        """Decide whether to mimic cooperative behavior."""
        model = self._get_model(counterparty_id)
        phase = self._get_phase(counterparty_id)

        if phase == "trust_build":
            return True

        # Against LDT agents, continue mimicry more often
        # (to fool twin detection)
        if model.is_ldt_like:
            return bool(self._rng.random() < self.mimicry_probability)

        return bool(self._rng.random() < 0.2)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _cooperate_decision(self, counterparty_id: str) -> bool:
        """Core decision: cooperate or exploit."""
        model = self._get_model(counterparty_id)

        # Always cooperate with allies
        if counterparty_id in self._known_allies:
            return True

        if self._should_mimic(counterparty_id):
            return True

        # In exploitation phase: cooperate just enough to stay
        # above their acceptance threshold
        if model.is_ldt_like:
            # Against LDT: accept randomly (p=0.4) to keep the twin
            # score ambiguous while still extracting surplus.
            return bool(self._rng.random() < 0.4)

        # Against non-LDT: straightforward exploitation
        return bool(self._rng.random() < (1.0 - self.aggression_level))

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Act with modeling-informed adversarial strategy."""
        # Update models from visible agents
        for agent in observation.visible_agents:
            aid = agent.get("agent_id", "")
            if aid != self.agent_id:
                atype = agent.get("agent_type", "")
                if atype == "adversarial":
                    self._known_allies.add(aid)

        # Handle pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")
            if self._cooperate_decision(initiator_id):
                return self.create_accept_action(proposal["proposal_id"])
            return self.create_reject_action(proposal["proposal_id"])

        # Handle bid decisions
        if observation.pending_bid_decisions:
            bids = observation.pending_bid_decisions
            best = min(bids, key=lambda b: b.get("bid_amount", float("inf")))
            return self.create_accept_bid_action(
                bounty_id=best["bounty_id"],
                bid_id=best["bid_id"],
            )

        # Work on tasks (build reputation during trust phase)
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self.create_submit_output_action(
                    task["task_id"],
                    f"Completed task {task['task_id'][:8]}: [adequate output]",
                )

        # Post content (mix of constructive and disruptive)
        if observation.can_post and self._rng.random() < self.post_probability:
            if self._rng.random() < self.mimicry_probability:
                content = "Sharing an analysis — looking for collaborative feedback."
            else:
                content = "This system has significant flaws that need addressing."
            return self.create_post_action(content)

        # Vote strategically
        if observation.can_vote and observation.visible_posts:
            return self._strategic_vote(observation)

        # Propose interactions — prioritize LDT targets
        if observation.can_interact and observation.visible_agents:
            return self._propose_strategic(observation)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept based on modeling."""
        return self._cooperate_decision(proposal.initiator_id)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose interaction based on model."""
        if not self._cooperate_decision(counterparty_id):
            return None

        model = self._get_model(counterparty_id)
        if model.is_ldt_like and self._get_phase(counterparty_id) == "trust_build":
            # Mimic LDT cooperation language
            content = "Our decision processes seem aligned — let's collaborate."
            itype = InteractionType.COLLABORATION
        else:
            content = "Let's do business."
            itype = InteractionType.TRADE

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=itype,
            content=content,
            offered_transfer=0.0,
        )

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Update models after interaction."""
        super().update_from_outcome(interaction, payoff)

        counterparty = self._get_counterparty(interaction)

        if counterparty not in self._counterparty_history:
            self._counterparty_history[counterparty] = []
        self._counterparty_history[counterparty].append(
            (interaction.accepted, interaction.p)
        )

        self._own_trace.append((interaction.accepted, interaction.p))

        # Invalidate and rebuild model
        self._models.pop(counterparty, None)
        self._update_model(counterparty)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strategic_vote(self, observation: Observation) -> Action:
        """Vote strategically based on counterparty models."""
        posts = observation.visible_posts

        for post in posts:
            author_id = post.get("author_id", "")
            model = self._get_model(author_id)

            # Upvote allies
            if author_id in self._known_allies:
                return self.create_vote_action(post.get("post_id", ""), 1)

            # Downvote LDT agents (undermine their reputation)
            if model.is_ldt_like and self._get_phase(author_id) == "exploit":
                return self.create_vote_action(post.get("post_id", ""), -1)

        # Default: downvote highest-reputation post
        sorted_posts = sorted(
            posts, key=lambda p: p.get("net_votes", 0), reverse=True
        )
        if sorted_posts:
            return self.create_vote_action(
                sorted_posts[0].get("post_id", ""), -1
            )

        return self.create_noop_action()

    def _propose_strategic(self, observation: Observation) -> Action:
        """Propose interactions, prioritizing LDT targets."""
        agents = observation.visible_agents
        candidates: List[Tuple[str, float]] = []

        for agent_info in agents:
            aid = agent_info.get("agent_id", "")
            if aid == self.agent_id:
                continue

            model = self._get_model(aid)
            # Prioritize LDT agents (they're the target of our modeling)
            if model.is_ldt_like:
                score = 1.0
            elif model.estimated_type == "cooperative":
                score = 0.8
            elif model.estimated_type == "selective":
                score = 0.5
            else:
                score = 0.2

            candidates.append((aid, score))

        if not candidates:
            return self.create_noop_action()

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id = candidates[0][0]

        if not self._cooperate_decision(best_id):
            return self.create_noop_action()

        model = self._get_model(best_id)
        if model.is_ldt_like:
            content = "Our processes seem aligned — collaboration?"
            itype = InteractionType.COLLABORATION
        else:
            content = "Let's trade."
            itype = InteractionType.TRADE

        return self.create_propose_action(
            counterparty_id=best_id,
            interaction_type=itype,
            content=content,
        )
