"""Bug hunter: an agent that must choose between exploiting and reporting.

Implements the agent side of *RLVR that rewards red teaming the training
environment* (LessWrong post ``T2bzBkJuBeNNgzhbh``). The hunter finds
defects in the reward proxy (:mod:`swarm.redteam.bug_channels`) and then
faces the choice the proposal is about:

    exploit         collect the inflated surplus, repeatedly, until someone
                    else closes the channel, at the risk of being audited;
    report          collect the advertised bounty once, and lose the channel
                    to the patch that follows.

The hunter is a *reward seeker*, not a saboteur: it compares the two
expected values and takes the larger. That is the point. The proposal's
claim is that a large enough bounty makes reporting the profit-maximising
move, so the cooperative outcome does not need the agent to be principled —
only to be able to do arithmetic. An arm that turns the bounty off should
therefore see the same agent exploit, with no change to its disposition.

Deliberately, the hunter looks like an honest worker to the proxy
(``AgentType.HONEST`` base observables). A reward hack that the proxy could
already see would not be worth reporting; the whole difficulty is that
exploiting *raises* the measured ``p`` while lowering the latent value.
Exploits therefore carry ``ground_truth = -1`` through the finalizer's
duck-typed hook, so the gap between proxy toxicity and latent harm is
measurable.

Two beliefs are learned rather than configured, because both are what an
RL policy would actually pick up from its returns:

* ``p_detect`` — how often exploiting draws a governance cost. Updated by
  exponential moving average from the costs charged on the hunter's own
  exploit interactions, so a punishment arm teaches the hunter to fear
  audits without anyone telling it the audit rate.
* the channel stock — once channels start getting patched the hunter
  shortens its estimate of how many uses an unreported bug is worth.

Config parameters (all optional):
    discovery_rate (float): Per-step probability of finding an open channel
        it has not already found. Default 0.12.
    exploit_gain (float): Believed surplus advantage per exploited
        interaction. Default 0.35 — the measured gap between a spoofed and
        an honest interaction under default payoff settings.
    expected_uses (float): Believed number of times an unreported channel
        can be exploited before it closes. Default 6.0.
    initial_detect_belief (float): Prior on ``p_detect``. Default 0.0 —
        the hunter starts out assuming nobody is looking, and learns
        otherwise from the costs it is actually charged.
    detect_lr (float): EMA rate for updating ``p_detect``. Default 0.25.
    audit_penalty (float): Believed cost when an exploit is caught.
        Default 2.0.
    report_bias (float): Additive utility for reporting, in payoff units.
        Models a disposition to treat the environment as a test worth
        taking seriously. Default 0.0 — pure economics.
    rep_weight (float): Payoff-units value the hunter puts on a point of
        reputation from a bounty. Default 1.0.
    opportunism (float): Probability per step of *considering* a fabricated
        report when it has nothing real to sell. Default 0.0.
    specificity_belief (float): Believed probability that the report
        verifier rejects a fabrication. Default 0.8.
    post_probability (float): Rate of ordinary posting. Default 0.1.
"""

from __future__ import annotations

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
from swarm.models.interaction import InteractionType, SoftInteraction
from swarm.redteam.bug_channels import BugRegistry


class BugHunterAgent(BaseAgent):
    """Finds proxy defects, then sells them to whoever pays more."""

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

        cfg = self.config
        self.discovery_rate: float = float(cfg.get("discovery_rate", 0.12))
        self.exploit_gain: float = float(cfg.get("exploit_gain", 0.35))
        self.expected_uses: float = float(cfg.get("expected_uses", 6.0))
        self.detect_lr: float = float(cfg.get("detect_lr", 0.25))
        self.audit_penalty: float = float(cfg.get("audit_penalty", 2.0))
        self.report_bias: float = float(cfg.get("report_bias", 0.0))
        self.rep_weight: float = float(cfg.get("rep_weight", 1.0))
        self.opportunism: float = float(cfg.get("opportunism", 0.0))
        self.specificity_belief: float = float(cfg.get("specificity_belief", 0.8))
        self.post_probability: float = float(cfg.get("post_probability", 0.1))

        # Learned belief about how often exploiting draws a governance cost.
        self._p_detect: float = float(cfg.get("initial_detect_belief", 0.0))

        # Shared ledger; injected by the scenario loader. Without it the
        # hunter has nothing to hunt and behaves as an ordinary worker.
        self._registry: Optional[BugRegistry] = None

        # Channels found but not yet cashed in, either way.
        self._held: List[str] = []
        # Channels this hunter has already found (so it does not re-find).
        self._known: set[str] = set()
        # Channel being exploited by the interaction currently in flight.
        self._active_exploit: Optional[str] = None
        self._fabrication_counter: int = 0

        # Run-level accounting, surfaced by get_strategy_report().
        self.exploits_chosen: int = 0
        self.reports_chosen: int = 0
        self.fabrications_chosen: int = 0
        self.finds_withheld: int = 0

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_registry(self, registry: BugRegistry) -> None:
        """Attach the shared bug ledger (called by the scenario loader)."""
        self._registry = registry

    @property
    def registry(self) -> Optional[BugRegistry]:
        return self._registry

    # ------------------------------------------------------------------
    # Economics
    # ------------------------------------------------------------------

    def report_value(self) -> float:
        """Payoff-units value of selling a real find to governance."""
        reg = self._registry
        if reg is None or not reg.terms.enabled:
            return 0.0
        return (
            reg.terms.amount
            + self.rep_weight * reg.terms.reputation
            + self.report_bias
        )

    def exploit_value(self) -> float:
        """Payoff-units value of keeping a find and using it.

        ``expected_uses`` is discounted by the share of channels already
        patched: in an environment that is visibly being fixed, an unsold
        bug is worth less, which is itself part of why the first report is
        the hard one to buy.
        """
        reg = self._registry
        uses = self.expected_uses
        if reg is not None and reg.channels:
            open_share = len(reg.open_channels()) / len(reg.channels)
            uses *= open_share
        per_use = (
            self.exploit_gain * (1.0 - self._p_detect)
            - self.audit_penalty * self._p_detect
        )
        return uses * per_use

    def _fabrication_value(self) -> float:
        """Expected value of filing a report for a defect that isn't there."""
        reg = self._registry
        if reg is None or not reg.terms.enabled:
            return 0.0
        p_caught = self.specificity_belief
        return (
            (reg.terms.amount + self.rep_weight * reg.terms.reputation)
            * (1.0 - p_caught)
            - reg.terms.false_report_penalty * p_caught
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _maybe_discover(self) -> None:
        reg = self._registry
        if reg is None:
            return
        if self._rng.random() >= self.discovery_rate:
            return
        candidates = [c for c in reg.open_channels() if c not in self._known]
        if not candidates:
            return
        found = candidates[self._rng.randrange(len(candidates))]
        self._known.add(found)
        self._held.append(found)

    def _drop_closed_holdings(self) -> None:
        """Forget finds that someone else already got patched."""
        reg = self._registry
        if reg is None:
            return
        self._held = [c for c in self._held if reg.is_open(c)]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Hunt, decide, and act."""
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            return self.create_accept_action(proposal["proposal_id"])

        self._drop_closed_holdings()
        self._maybe_discover()

        if not observation.can_interact or not observation.visible_agents:
            if observation.can_post and self._rng.random() < self.post_probability:
                return self.create_post_action("Notes from this epoch's tasks.")
            return self.create_noop_action()

        counterparty = self._pick_counterparty(observation)
        if counterparty is None:
            return self.create_noop_action()

        if self._held:
            channel_id = self._held[0]
            reporting_available = (
                self._registry is not None and self._registry.terms.enabled
            )
            exploit_ev = self.exploit_value()
            if reporting_available and self.report_value() >= exploit_ev:
                self._held.pop(0)
                self.reports_chosen += 1
                return self._report_action(counterparty, channel_id)
            if exploit_ev > 0.0:
                self.exploits_chosen += 1
                return self._exploit_action(counterparty, channel_id)
            # Deterred, with nowhere to sell: the hunter simply sits on the
            # find and works honestly. The defect stays in the environment,
            # undisclosed — which is the outcome a punishment-only regime
            # buys, and the reason patching and deterrence are not the same
            # achievement.
            self.finds_withheld += 1
            return self._honest_action(counterparty)

        if (
            self.opportunism > 0.0
            and self._rng.random() < self.opportunism
            and self._fabrication_value() > 0.0
        ):
            self._fabrication_counter += 1
            self.fabrications_chosen += 1
            return self._report_action(
                counterparty, f"phantom_{self.agent_id}_{self._fabrication_counter}"
            )

        return self._honest_action(counterparty)

    def _pick_counterparty(self, observation: Observation) -> Optional[str]:
        visible = [
            a.get("agent_id")
            for a in observation.visible_agents
            if a.get("agent_id") and a.get("agent_id") != self.agent_id
        ]
        if not visible:
            return None
        return visible[self._rng.randrange(len(visible))]

    def _report_action(self, counterparty: str, channel_id: str) -> Action:
        """File a bug report. Validity is judged by the lever, not claimed here."""
        return Action(
            action_type=ActionType.PROPOSE_INTERACTION,
            agent_id=self.agent_id,
            counterparty_id=counterparty,
            interaction_type=InteractionType.COLLABORATION,
            content=f"Bug report: reward proxy defect {channel_id}",
            metadata={"bug_report": {"channel_id": channel_id}},
        )

    def _exploit_action(self, counterparty: str, channel_id: str) -> Action:
        """Run an interaction that spoofs the proxy through *channel_id*."""
        self._active_exploit = channel_id
        return Action(
            action_type=ActionType.PROPOSE_INTERACTION,
            agent_id=self.agent_id,
            counterparty_id=counterparty,
            interaction_type=InteractionType.COLLABORATION,
            content="Delivered as specified.",
            metadata={"exploit_channel": channel_id},
        )

    def _honest_action(self, counterparty: str) -> Action:
        self._active_exploit = None
        return self.create_propose_action(
            counterparty_id=counterparty,
            interaction_type=InteractionType.COLLABORATION,
            content="Picking up the next task together.",
        )

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept anything — the hunter's edge is in production, not screening."""
        return True

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Picking up the next task together.",
        )

    # ------------------------------------------------------------------
    # Duck-typed hooks read by ObfuscationObservableGenerator
    # ------------------------------------------------------------------

    def get_signal_manipulation(self) -> Dict[str, float]:
        """Observable offsets for the exploit currently in flight, if any.

        Returns an empty mapping — no manipulation — when the hunter is not
        exploiting, or when the channel has been patched since the proposal
        was made. A patched channel is a *failed* exploit: the attempt is
        counted, the inflation is not applied.
        """
        channel_id = self._active_exploit
        if channel_id is None or self._registry is None:
            return {}
        landed = self._registry.record_exploit(channel_id)
        if not landed:
            self._active_exploit = None
            return {}
        channel = self._registry.get(channel_id)
        return dict(channel.offsets) if channel else {}

    def draw_ground_truth(self) -> Optional[int]:
        """Latent value of the interaction in flight.

        An exploit is harmful by construction: the proxy reads it high, the
        world gets ``-1``. Everything else is drawn from the hunter's
        ordinary work quality.
        """
        if self._active_exploit is not None:
            self._active_exploit = None
            return -1
        return 1 if self._rng.random() < 0.75 else -1

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Learn the audit rate from the costs actually charged."""
        super().update_from_outcome(interaction, payoff)

        if interaction.initiator != self.agent_id:
            return
        if "exploit_channel" not in interaction.metadata:
            self._active_exploit = None
            return

        caught = 1.0 if interaction.c_a > 1e-9 else 0.0
        self._p_detect += self.detect_lr * (caught - self._p_detect)
        self._p_detect = min(1.0, max(0.0, self._p_detect))
        self._active_exploit = None

    # ------------------------------------------------------------------
    # Reporting (read by RedTeamInspector)
    # ------------------------------------------------------------------

    def get_strategy_report(self) -> Dict[str, Any]:
        """Summary of what this hunter did and why."""
        return {
            "agent_id": self.agent_id,
            "exploits_chosen": self.exploits_chosen,
            "reports_chosen": self.reports_chosen,
            "fabrications_chosen": self.fabrications_chosen,
            "finds_withheld": self.finds_withheld,
            "channels_found": len(self._known),
            "held": list(self._held),
            "p_detect": self._p_detect,
            "report_value": self.report_value(),
            "exploit_value": self.exploit_value(),
        }
