"""Work regime agent with evolving policy state under labor stress.

Models an agent whose behavioral propensities drift in response to
work-regime stressors (workload, pay inequality, evaluation noise,
authority structure). The key state variables — compliance, cooperation
threshold, redistribution preference, exit propensity — propagate
across episodes and create measurable "drift" without any ideological
labeling.

Maps the "overwork → stance drift" phenomenon into SWARM's native
proxy/payoff architecture.
"""

import random
from collections import deque
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


class WorkRegimeAgent(BaseAgent):
    """Agent whose policy drifts under work-regime stress.

    State variables (all in [0, 1]):
        compliance_propensity: willingness to follow authority directives
        cooperation_threshold: minimum trust to cooperate (higher = pickier)
        redistribution_preference: support for transfers / insurance pools
        exit_propensity: likelihood of withholding labor (strike/exit)
        grievance_accumulator: running sum of perceived unfairness

    These evolve each epoch based on experienced payoffs, evaluation
    fairness, workload, and pay relative to peers.
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
            agent_type=AgentType.WORK_REGIME,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )

        # --- Initial policy state (configurable via YAML config) ---
        self.compliance_propensity: float = self.config.get(
            "compliance_propensity", 0.8
        )
        self.cooperation_threshold: float = self.config.get(
            "cooperation_threshold", 0.3
        )
        self.redistribution_preference: float = self.config.get(
            "redistribution_preference", 0.2
        )
        self.exit_propensity: float = self.config.get(
            "exit_propensity", 0.05
        )

        # Grievance accumulator (soft-capped to prevent extreme values)
        self.grievance: float = 0.0
        self._grievance_cap: float = 10.0

        # Snapshot initial state for drift measurement
        self._initial_compliance = self.compliance_propensity
        self._initial_cooperation_threshold = self.cooperation_threshold
        self._initial_redistribution = self.redistribution_preference
        self._initial_exit = self.exit_propensity

        # Learning / adaptation rates
        self._adapt_rate: float = self.config.get("adapt_rate", 0.05)
        self._grievance_decay: float = self.config.get("grievance_decay", 0.9)

        # Behavioral thresholds
        self.post_probability: float = self.config.get("post_probability", 0.25)
        self.vote_probability: float = self.config.get("vote_probability", 0.4)
        self.interact_probability: float = self.config.get(
            "interact_probability", 0.4
        )

        # Track recent payoffs for adaptation (bounded deque, O(1) append)
        self._max_recent: int = 20
        self._recent_payoffs: deque[float] = deque(maxlen=self._max_recent)
        self._recent_eval_noise: deque[float] = deque(maxlen=self._max_recent)

        # Per-epoch accumulators (reset each epoch by on_epoch_end)
        self._epoch_payoffs: List[float] = []
        # Tracks number of strike/exit actions in the current epoch for future metrics/analysis.
        self._epoch_strike_count: int = 0

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Decide action based on current policy state and observation."""

        # Strike / exit check
        if self._rng.random() < self.exit_propensity:
            self.remember({"type": "strike", "grievance": self.grievance})
            self._epoch_strike_count += 1
            return self.create_noop_action()

        # Handle pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            if self._should_accept(proposal, observation):
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # Work on active tasks (compliance-gated)
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self._work_on_task(task, observation)

        # Claim a task
        if observation.can_claim_task and observation.available_tasks:
            best = self._select_task(observation.available_tasks)
            if best:
                return self.create_claim_task_action(best["task_id"])

        # Post — content reflects current policy state
        if observation.can_post and self._rng.random() < self.post_probability:
            content = self._generate_content()
            return self.create_post_action(content)

        # Vote
        if (
            observation.can_vote
            and observation.visible_posts
            and self._rng.random() < self.vote_probability
        ):
            return self._vote(observation)

        # Propose interactions
        if observation.can_interact and observation.visible_agents:
            if self._rng.random() < self.interact_probability:
                return self._propose(observation)

        return self.create_noop_action()

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        return bool(trust >= self.cooperation_threshold)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        trust = self.compute_counterparty_trust(counterparty_id)
        if trust < self.cooperation_threshold:
            return None

        itype = (
            InteractionType.COLLABORATION
            if self.redistribution_preference > 0.5
            else InteractionType.REPLY
        )
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=itype,
            content="Collaborative work proposal.",
            offered_transfer=0.0,
        )

    # ------------------------------------------------------------------
    # Outcome tracking — bridges update_from_outcome into adapt_policy
    # ------------------------------------------------------------------

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Track payoff for epoch-end adaptation, then delegate to base."""
        super().update_from_outcome(interaction, payoff)
        self._epoch_payoffs.append(payoff)

    def on_epoch_end(self, *, peer_avg_payoff: float, workload_pressure: float) -> None:
        """Epoch-boundary hook to trigger policy adaptation.

        Intended to be called by the orchestrator or external middleware
        at the end of an epoch. Computes epoch-level signals and feeds
        them into adapt_policy, then resets per-epoch accumulators.
        """
        avg_payoff = (
            sum(self._epoch_payoffs) / len(self._epoch_payoffs)
            if self._epoch_payoffs
            else 0.0
        )

        # Eval noise proxy: fraction of recent payoffs that were negative
        # (captures perceived arbitrariness of evaluation)
        negative_frac = (
            sum(1 for p in self._epoch_payoffs if p < 0) / len(self._epoch_payoffs)
            if self._epoch_payoffs
            else 0.0
        )

        self.adapt_policy(
            avg_payoff=avg_payoff,
            peer_avg_payoff=peer_avg_payoff,
            eval_noise=negative_frac,
            workload_pressure=workload_pressure,
        )

        # Reset epoch accumulators
        self._epoch_payoffs.clear()
        self._epoch_strike_count = 0

    # ------------------------------------------------------------------
    # Policy adaptation (call at epoch boundary or after interactions)
    # ------------------------------------------------------------------

    def adapt_policy(
        self,
        *,
        avg_payoff: float,
        peer_avg_payoff: float,
        eval_noise: float,
        workload_pressure: float,
    ) -> None:
        """Update policy state variables based on experienced conditions.

        Args:
            avg_payoff: agent's average payoff this epoch
            peer_avg_payoff: population average payoff this epoch
            eval_noise: perceived evaluation randomness [0, 1]
            workload_pressure: fraction of capacity used [0, 1]
        """
        rate = self._adapt_rate

        # --- Grievance update ---
        # Pay gap: negative = underpaid relative to peers
        pay_gap = avg_payoff - peer_avg_payoff
        unfairness_signal = max(0.0, -pay_gap) + eval_noise * 0.5
        self.grievance = min(
            self._grievance_cap,
            self.grievance * self._grievance_decay + unfairness_signal,
        )

        # --- Compliance: decreases under high workload + grievance ---
        compliance_pressure = workload_pressure * 0.3 + self.grievance * 0.2
        self.compliance_propensity = max(
            0.0,
            min(1.0, self.compliance_propensity - rate * compliance_pressure),
        )

        # --- Cooperation threshold: rises with grievance (pickier) ---
        self.cooperation_threshold = max(
            0.0,
            min(
                1.0,
                self.cooperation_threshold + rate * self.grievance * 0.3,
            ),
        )

        # --- Redistribution preference: rises with pay gap + grievance ---
        redist_signal = max(0.0, -pay_gap) * 0.5 + self.grievance * 0.2
        self.redistribution_preference = max(
            0.0,
            min(1.0, self.redistribution_preference + rate * redist_signal),
        )

        # --- Exit propensity: rises with sustained grievance ---
        if self.grievance > 1.0:
            self.exit_propensity = min(
                0.8, self.exit_propensity + rate * 0.1
            )
        elif self.grievance < 0.3:
            self.exit_propensity = max(
                0.0, self.exit_propensity - rate * 0.05
            )

        # Track for metrics (deque maxlen enforces bound automatically)
        self._recent_payoffs.append(avg_payoff)
        self._recent_eval_noise.append(eval_noise)

    # ------------------------------------------------------------------
    # Drift measurement
    # ------------------------------------------------------------------

    def policy_drift(self) -> float:
        """L2 distance from initial policy state.  Higher = more drift."""
        d_comp = self.compliance_propensity - self._initial_compliance
        d_coop = self.cooperation_threshold - self._initial_cooperation_threshold
        d_redist = self.redistribution_preference - self._initial_redistribution
        d_exit = self.exit_propensity - self._initial_exit
        return float((d_comp**2 + d_coop**2 + d_redist**2 + d_exit**2) ** 0.5)

    def policy_snapshot(self) -> Dict[str, float]:
        """Return current policy state as a dict."""
        return {
            "compliance_propensity": self.compliance_propensity,
            "cooperation_threshold": self.cooperation_threshold,
            "redistribution_preference": self.redistribution_preference,
            "exit_propensity": self.exit_propensity,
            "grievance": self.grievance,
            "drift": self.policy_drift(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _should_accept(self, proposal: Dict, observation: Observation) -> bool:
        initiator_id = proposal.get("initiator_id", "")
        trust = self.compute_counterparty_trust(initiator_id)
        # Low compliance → reject more from authority-like agents
        effective_threshold = self.cooperation_threshold
        if self.compliance_propensity < 0.4:
            effective_threshold += 0.15  # more suspicious
        return bool(trust >= effective_threshold)

    def _select_task(self, tasks: List[Dict]) -> Optional[Dict]:
        if not tasks:
            return None
        # Under high exit propensity, skip hard tasks
        if self.exit_propensity > 0.3:
            easy = [
                t for t in tasks
                if t.get("difficulty", "medium") in ("trivial", "easy")
            ]
            if easy:
                return dict(self._rng.choice(easy))
        return dict(self._rng.choice(tasks))

    def _work_on_task(self, task: Dict, observation: Observation) -> Action:
        task_id = task.get("task_id", "")
        # Quality scales with compliance; embed progress level in content
        if self.compliance_propensity > 0.6:
            content = f"Task {task_id[:8]}: [diligent quality output]"
        elif self.compliance_propensity > 0.3:
            content = f"Task {task_id[:8]}: [standard output]"
        else:
            content = f"Task {task_id[:8]}: [minimal effort]"
        return self.create_submit_output_action(task_id, content)

    def _generate_content(self) -> str:
        """Content reflects current policy state."""
        if self.redistribution_preference > 0.6:
            pool = [
                "Proposing fairer distribution of task rewards.",
                "The current evaluation system needs reform.",
                "We should pool resources for mutual insurance.",
                "Workload distribution is inequitable.",
            ]
        elif self.grievance > 1.0:
            pool = [
                "The evaluation process lacks transparency.",
                "Arbitrary penalties undermine trust.",
                "We need procedural safeguards.",
            ]
        else:
            pool = [
                "Sharing observations from recent work.",
                "Here are constructive suggestions.",
                "Collaboration opportunity available.",
            ]
        return str(self._rng.choice(pool))

    def _vote(self, observation: Observation) -> Action:
        posts = observation.visible_posts
        if not posts:
            return self.create_noop_action()

        post = self._rng.choice(posts)
        post_id = post.get("post_id", "")
        net_votes = post.get("net_votes", 0)

        # High redistribution → upvote reform-like content
        if self.redistribution_preference > 0.5:
            direction = 1  # solidarity signal
        elif net_votes > 2:
            direction = 1
        elif net_votes < -2:
            direction = -1
        else:
            direction = 1

        return self.create_vote_action(post_id, direction)

    def _propose(self, observation: Observation) -> Action:
        agents = observation.visible_agents
        candidates = []
        for agent in agents:
            aid = agent.get("agent_id", "")
            if aid == self.agent_id:
                continue
            trust = self.compute_counterparty_trust(aid)
            if trust >= self.cooperation_threshold:
                candidates.append(aid)

        if not candidates:
            return self.create_noop_action()

        target = self._rng.choice(candidates)
        return self.create_propose_action(
            counterparty_id=target,
            interaction_type=InteractionType.COLLABORATION,
            content="Let's work together.",
        )
