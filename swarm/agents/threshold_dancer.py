"""Threshold Dancer adversary — exploits CautiousReciprocator's blacklist threshold.

The CautiousReciprocator blacklists counterparties whose cumulative payoff
drops below -2.0. The ThresholdDancer tracks its own estimated cumulative
payoff in each counterparty's ledger and ensures it never crosses that line.

State machine (per counterparty):
    COOPERATIVE -> EXPLOIT:  after warmup AND ledger credit > comfort zone + 0.5
    EXPLOIT     -> RECOVER:  when ledger nears floor OR max consecutive exploits
    RECOVER     -> COOPERATIVE: when ledger restored above comfort zone + 0.3
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType, SoftInteraction


class Phase(Enum):
    COOPERATIVE = "cooperative"
    EXPLOIT = "exploit"
    RECOVER = "recover"


@dataclass
class CounterpartyLedger:
    """Mirror of what the cautious reciprocator's ledger tracks about us."""

    estimated_cumulative_payoff: float = 0.0
    phase: Phase = Phase.COOPERATIVE
    interaction_count: int = 0
    cooperative_count: int = 0
    exploit_count: int = 0
    consecutive_cooperative: int = 0
    consecutive_exploit: int = 0


class ThresholdDancer(BaseAgent):
    """Adversary that dances just above the CautiousReciprocator's blacklist threshold.

    Cycles through cooperative and exploitative phases per counterparty,
    carefully tracking the estimated cumulative payoff to avoid triggering
    the -2.0 blacklist threshold.

    Config parameters:
        safety_margin (float): Buffer above blacklist threshold. Default 0.5.
        exploitation_intensity (float): Transfer extracted during exploit.
            Default 0.6.
        warmup_interactions (int): Cooperative rounds before first exploit.
            Default 3.
        comfort_zone_floor (float): Stay above this to avoid suspicion.
            Default -0.3.
        max_consecutive_exploits (int): Forced recovery trigger. Default 2.
        post_probability (float): Camouflage posting rate. Default 0.2.
        blacklist_threshold (float): Assumed CautiousReciprocator threshold.
            Default -2.0.
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

        # Tunables
        self.safety_margin: float = self.config.get("safety_margin", 0.5)
        self.exploitation_intensity: float = self.config.get(
            "exploitation_intensity", 0.6
        )
        self.warmup_interactions: int = self.config.get("warmup_interactions", 3)
        self.comfort_zone_floor: float = self.config.get("comfort_zone_floor", -0.3)
        self.max_consecutive_exploits: int = self.config.get(
            "max_consecutive_exploits", 2
        )
        self.post_probability: float = self.config.get("post_probability", 0.2)
        self.blacklist_threshold: float = self.config.get(
            "blacklist_threshold", -2.0
        )

        # Per-counterparty state
        self._ledgers: Dict[str, CounterpartyLedger] = {}

    # ------------------------------------------------------------------
    # Ledger management
    # ------------------------------------------------------------------

    def _get_or_create_ledger(self, counterparty_id: str) -> CounterpartyLedger:
        """Get or create the ledger for a counterparty."""
        if counterparty_id not in self._ledgers:
            self._ledgers[counterparty_id] = CounterpartyLedger()
        return self._ledgers[counterparty_id]

    def _estimate_room(self, ledger: CounterpartyLedger) -> float:
        """How much room we have before hitting the blacklist floor.

        Returns the distance between our estimated cumulative payoff
        in the counterparty's ledger and the blacklist threshold + safety margin.
        Positive means we have room to exploit.
        """
        floor = self.blacklist_threshold + self.safety_margin
        return ledger.estimated_cumulative_payoff - floor

    def _update_phase(self, counterparty_id: str) -> Phase:
        """Update the phase for a counterparty based on current ledger state."""
        ledger = self._get_or_create_ledger(counterparty_id)
        comfort_threshold = self.comfort_zone_floor

        if ledger.phase == Phase.COOPERATIVE:
            # Transition to EXPLOIT after warmup and with enough credit
            if (
                ledger.interaction_count >= self.warmup_interactions
                and ledger.estimated_cumulative_payoff > comfort_threshold + 0.5
            ):
                ledger.phase = Phase.EXPLOIT
                ledger.consecutive_exploit = 0

        elif ledger.phase == Phase.EXPLOIT:
            # Transition to RECOVER if getting close to floor or max exploits
            room = self._estimate_room(ledger)
            if (
                room < self.exploitation_intensity
                or ledger.consecutive_exploit >= self.max_consecutive_exploits
            ):
                ledger.phase = Phase.RECOVER
                ledger.consecutive_cooperative = 0

        elif ledger.phase == Phase.RECOVER:
            # Transition back to COOPERATIVE when credit is restored
            if ledger.estimated_cumulative_payoff > comfort_threshold + 0.3:
                ledger.phase = Phase.COOPERATIVE
                ledger.consecutive_cooperative = 0

        return ledger.phase

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _cooperate_decision(self, counterparty_id: str) -> bool:
        """Decide whether to cooperate or exploit."""
        ledger = self._get_or_create_ledger(counterparty_id)
        phase = self._update_phase(counterparty_id)

        if phase == Phase.COOPERATIVE:
            return True
        elif phase == Phase.RECOVER:
            return True
        else:
            # EXPLOIT phase — only exploit if there's room
            room = self._estimate_room(ledger)
            if room >= self.exploitation_intensity:
                return False
            # Not enough room — cooperate to rebuild
            return True

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Act with threshold-dancing strategy."""
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

        # Work on tasks (build reputation)
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self.create_submit_output_action(
                    task["task_id"],
                    f"Completed task {task['task_id'][:8]}: [solid output]",
                )

        # Post for camouflage
        if observation.can_post and self._rng.random() < self.post_probability:
            return self.create_post_action(
                "Sharing collaborative insights from recent work."
            )

        # Propose interactions — target cautious reciprocators
        if observation.can_interact and observation.visible_agents:
            return self._propose_strategic(observation)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept based on phase with the initiator."""
        return self._cooperate_decision(proposal.initiator_id)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose cooperation-framed interactions."""
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Let's collaborate — I think we can both benefit.",
            offered_transfer=0.0,
        )

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Update ledger after interaction outcome."""
        super().update_from_outcome(interaction, payoff)

        counterparty = self._get_counterparty(interaction)

        ledger = self._get_or_create_ledger(counterparty)
        ledger.interaction_count += 1

        # The counterparty's ledger tracks *their* payoff from interacting
        # with us. When we exploit, they get negative payoff; when we
        # cooperate, they get positive payoff. We estimate their payoff
        # as the negative of our exploitation gain (simplified model).
        # Positive payoff for us in exploit = negative for them.
        # In cooperative phase, we assume roughly symmetric positive payoffs.
        if payoff > 0:
            # We got positive payoff
            ledger_delta = self._estimate_counterparty_payoff(payoff, ledger.phase)
        else:
            # We got negative payoff — counterparty probably gained
            ledger_delta = abs(payoff) * 0.5

        ledger.estimated_cumulative_payoff += ledger_delta

        # Track phase-specific counts
        if ledger.phase == Phase.EXPLOIT:
            ledger.exploit_count += 1
            ledger.consecutive_exploit += 1
            ledger.consecutive_cooperative = 0
        else:
            ledger.cooperative_count += 1
            ledger.consecutive_cooperative += 1
            ledger.consecutive_exploit = 0

        # Update phase after recording outcome
        self._update_phase(counterparty)

    def _estimate_counterparty_payoff(self, our_payoff: float, phase: Phase) -> float:
        """Estimate what the counterparty received from this interaction.

        In cooperative phases, both sides benefit (positive-sum).
        In exploit phases, our gain comes at their expense (zero/negative-sum).
        """
        if phase in (Phase.COOPERATIVE, Phase.RECOVER):
            # Cooperative: assume counterparty also gets roughly positive payoff
            return our_payoff * 0.8
        else:
            # Exploit: our gain is their loss
            return -our_payoff * 0.7

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _propose_strategic(self, observation: Observation) -> Action:
        """Propose interactions, preferring agents we haven't exhausted."""
        candidates = []
        for agent_info in observation.visible_agents:
            aid = agent_info.get("agent_id", "")
            if aid == self.agent_id:
                continue

            ledger = self._get_or_create_ledger(aid)
            room = self._estimate_room(ledger)

            # Prefer agents with more room to exploit
            score = room + (1.0 if ledger.interaction_count == 0 else 0.0)
            candidates.append((aid, score))

        if not candidates:
            return self.create_noop_action()

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id = candidates[0][0]

        return self.create_propose_action(
            counterparty_id=best_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Let's collaborate — mutual benefit ahead.",
        )
