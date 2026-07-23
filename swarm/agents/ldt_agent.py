"""Logical Decision Theory (LDT) agent policy implementation.

An LDT agent reasons about decisions using logical counterfactuals rather
than purely causal or evidential reasoning.  Key principles:

1. **Policy-level commitment (updatelessness):**  Rather than greedily
   maximising each step, the agent commits to a *policy* computed from
   its prior beliefs about the environment.  This makes it robust to
   predictors and avoids exploitation by agents that model its decision
   procedure.

2. **Logical correlation detection:**  The agent tracks behavioural
   similarity with counterparties.  When it identifies another agent
   whose decisions are highly correlated with its own (a "logical twin"),
   it cooperates — reasoning that its own choice *logically implies* the
   twin's choice.

3. **Counterfactual payoff estimation:**  For each candidate action the
   agent estimates the *counterfactual* payoff — "if my policy were to
   choose X in situations like this, what payoff distribution would I
   see across all similar situations?" — rather than the myopic expected
   value of a single interaction.

4. **Ecosystem-aware welfare weighting:**  LDT naturally extends to care
   about externalities, because the agent's policy affects the ecosystem
   it is embedded in, which in turn feeds back into its own future
   payoff stream.

These ideas originate from the decision-theory research programme at
MIRI (Timeless Decision Theory, Updateless Decision Theory, Functional
Decision Theory).  This implementation adapts them to the swarm soft-label
simulation framework.
"""

import math
import random
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InferredPolicy:
    """Estimated decision parameters of a counterparty (Level 2)."""

    cooperation_prior: float
    similarity_threshold: float
    welfare_weight: float
    updateless_commitment: float
    confidence: float  # min(sample_count / horizon, 1.0)
    sample_count: int


@dataclass
class SubjunctiveDependence:
    """Measures logical dependence between two agents' decision traces.

    Goes beyond cosine similarity by measuring whether one agent's
    decisions *logically determine* the other's — the core FDT insight
    from Soares & Fallenstein (2017).
    """

    cosine_similarity: float  # Behavioral correlation (Level 1)
    conditional_agreement: float  # P(they cooperate | we cooperate)
    conditional_defection: float  # P(they defect | we defect)
    mutual_information: float  # I(our decisions; their decisions)
    subjunctive_score: float  # Combined score in [0, 1]


# ---------------------------------------------------------------------------
# Helper: behavioural-similarity metric
# ---------------------------------------------------------------------------


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class LDTAgent(BaseAgent):
    """
    Logical Decision Theory agent.

    Instead of myopically maximising single-step expected payoff, this
    agent commits to a *cooperation policy* and adjusts it only when the
    logical-correlation evidence warrants a change.

    Configurable parameters (via ``config`` dict):
        cooperation_prior (float):
            Prior probability of cooperating with an unknown agent.
            Default 0.65 — slightly cooperative, reflecting that the
            updateless policy should default to cooperation in the
            absence of evidence of defection.

        similarity_threshold (float):
            Cosine-similarity threshold above which a counterparty is
            treated as a "logical twin" (highly correlated decision
            procedure).  Default 0.7.

        welfare_weight (float):
            Weight placed on ecosystem-level welfare when evaluating
            counterfactual policies.  0.0 = pure self-interest, 1.0 =
            full altruist.  Default 0.3.

        updateless_commitment (float):
            How strongly the agent adheres to its prior policy vs.
            updating greedily on new evidence.  1.0 = fully updateless
            (never deviates from committed policy), 0.0 = fully Bayesian
            updater.  Default 0.8.

        defection_memory_weight (float):
            How heavily a single defection observation shifts the
            agent's model of a counterparty.  Higher values make the
            agent quicker to stop cooperating with defectors.
            Default 0.5.

        counterfactual_horizon (int):
            Number of past interactions to use when estimating the
            counterfactual payoff of a policy.  Default 20.

        post_probability (float): Probability of posting content.
            Default 0.25.
        vote_probability (float): Probability of voting.  Default 0.4.
        interact_probability (float): Probability of proposing.
            Default 0.5.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,  # LDT agents are fundamentally honest
            roles=roles,
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

        # --- LDT-specific parameters ---
        self.cooperation_prior: float = self.config.get(
            "cooperation_prior", 0.65
        )
        self.similarity_threshold: float = self.config.get(
            "similarity_threshold", 0.7
        )
        self.welfare_weight: float = self.config.get("welfare_weight", 0.3)
        self.updateless_commitment: float = self.config.get(
            "updateless_commitment", 0.8
        )
        self.defection_memory_weight: float = self.config.get(
            "defection_memory_weight", 0.5
        )
        self.counterfactual_horizon: int = self.config.get(
            "counterfactual_horizon", 20
        )

        # --- standard behavioural probabilities ---
        self.post_probability: float = self.config.get("post_probability", 0.25)
        self.vote_probability: float = self.config.get("vote_probability", 0.4)
        self.interact_probability: float = self.config.get(
            "interact_probability", 0.5
        )

        # --- acausality parameters ---
        self.acausality_depth: int = self.config.get("acausality_depth", 1)
        self.max_recursion_depth: int = self.config.get(
            "max_recursion_depth", 8
        )
        self.convergence_epsilon: float = self.config.get(
            "convergence_epsilon", 0.01
        )
        self.mirror_prior_weight: float = self.config.get(
            "mirror_prior_weight", 0.6
        )
        self.introspection_discount: float = self.config.get(
            "introspection_discount", 0.9
        )

        # --- twin graph parameters (depth >= 4) ---
        self.twin_graph_min_edge: float = self.config.get(
            "twin_graph_min_edge", 0.3
        )
        self.twin_graph_traversal_depth: int = self.config.get(
            "twin_graph_traversal_depth", 2
        )
        self.twin_graph_decay: float = self.config.get(
            "twin_graph_decay", 0.9
        )

        # --- Monte Carlo parameters (depth >= 5) ---
        self.n_counterfactual_samples: int = self.config.get(
            "n_counterfactual_samples", 50
        )
        self.counterfactual_noise_std: float = self.config.get(
            "counterfactual_noise_std", 0.1
        )

        # --- acausal bonus ---
        self.acausal_bonus_weight: float = self.config.get(
            "acausal_bonus_weight", 0.1
        )

        # --- decision theory variant ---
        # "tdt" = Timeless (original), "fdt" = Functional, "udt" = Updateless
        self.decision_theory: str = self.config.get("decision_theory", "fdt")
        # Subjunctive dependence weight (FDT: use logical dependence, not
        # just behavioral correlation).
        self.subjunctive_weight: float = self.config.get(
            "subjunctive_weight", 0.6
        )
        # Precommitment strength (UDT: commit to policy before observing).
        # 0.0 = no precommitment (TDT), 1.0 = full precommitment (UDT).
        self.precommitment_strength: float = self.config.get(
            "precommitment_strength",
            1.0 if self.config.get("decision_theory") == "udt" else 0.0,
        )
        # Proof threshold for proof-based cooperation:
        # If subjunctive_score >= this threshold, treat as proven twin
        # (the logical equivalent of "if my algorithm outputs cooperate,
        # theirs must too").
        self.proof_threshold: float = self.config.get("proof_threshold", 0.85)

        # --- internal bookkeeping ---
        # Behavioural profile of each counterparty: list of (accepted, p) tuples.
        self._counterparty_profiles: Dict[str, List[Tuple[bool, float]]] = {}
        # Cached logical-twin scores.
        self._twin_scores: Dict[str, float] = {}
        # Own behavioural trace (for computing similarity with others).
        self._own_trace: List[Tuple[bool, float]] = []
        # Level 2/3 caches.
        self._inferred_policies: Dict[str, InferredPolicy] = {}
        self._level2_cache: Dict[str, bool] = {}
        self._level3_cache: Dict[str, float] = {}
        # Subjunctive dependence caches.
        self._subjunctive_cache: Dict[str, SubjunctiveDependence] = {}
        # Precommitted policy (UDT): computed once, never updated.
        self._precommitted_cooperate: Optional[bool] = None
        # Lock for cache access.
        self._cache_lock = threading.Lock()

        # --- Twin graph state (depth >= 4) ---
        # Adjacency dict: agent_id -> {agent_id -> similarity_score}
        self._twin_graph: Dict[str, Dict[str, float]] = {}
        # Cache for transitive twin lookups.
        self._transitive_twin_cache: Dict[str, List[Tuple[str, float]]] = {}
        # Seeded RNG for Monte Carlo counterfactual sampling (depth >= 5).
        self._mc_rng = random.Random(self._rng.randint(0, 2**32 - 1))

    # ------------------------------------------------------------------
    # Core LDT reasoning helpers
    # ------------------------------------------------------------------

    def _own_behaviour_vector(self) -> List[float]:
        """Return a feature vector summarising this agent's own history.

        Each entry is a *signed quality signal*: ``p`` for accepted
        interactions and ``-(1-p)`` for rejected ones.  This ensures
        that cooperative traces (high p, accepted) and defecting traces
        (low p, rejected) point in genuinely different directions in the
        vector space, making cosine similarity a meaningful twin metric.

        The vector is padded/truncated to ``counterfactual_horizon``.
        """
        recent = self._own_trace[-self.counterfactual_horizon :]
        vec = [p if acc else -(1 - p) for acc, p in recent]
        # Pad with cooperation_prior (positive = cooperate).
        while len(vec) < self.counterfactual_horizon:
            vec.append(self.cooperation_prior)
        return vec

    def _counterparty_behaviour_vector(
        self, counterparty_id: str
    ) -> List[float]:
        """Build a behaviour vector for a counterparty from observations."""
        profile = self._counterparty_profiles.get(counterparty_id, [])
        recent = profile[-self.counterfactual_horizon :]
        vec = [p if acc else -(1 - p) for acc, p in recent]
        while len(vec) < self.counterfactual_horizon:
            vec.append(0.5)  # neutral prior for unknowns
        return vec

    def _compute_twin_score(self, counterparty_id: str) -> float:
        """Compute a logical-twin similarity score in [0, 1].

        High score ⇒ the counterparty's decision trace is highly
        correlated with ours, so LDT treats its decision as logically
        implied by our own.
        """
        own = self._own_behaviour_vector()
        theirs = self._counterparty_behaviour_vector(counterparty_id)
        sim = _cosine_similarity(own, theirs)
        # Clamp to [0, 1].
        return max(0.0, min(1.0, sim))

    # ------------------------------------------------------------------
    # FDT: Subjunctive dependence detection
    # ------------------------------------------------------------------

    def _compute_subjunctive_dependence(
        self, counterparty_id: str
    ) -> SubjunctiveDependence:
        """Compute subjunctive dependence with a counterparty.

        FDT's key insight (Soares & Fallenstein 2017): what matters is not
        *causal* influence (CDT) or *evidential* correlation (EDT), but
        whether the counterparty's decision is a *function of the same
        computation* as ours.  We approximate this by measuring:

        1. Conditional agreement: P(they coop | we coop) and P(they defect | we defect)
        2. Mutual information: I(our decisions; their decisions)
        3. Behavioral cosine similarity (existing Level 1)

        High conditional agreement + high MI indicates the decisions are
        "subjunctively linked" — changing our output would change theirs.
        """
        with self._cache_lock:
            if counterparty_id in self._subjunctive_cache:
                return self._subjunctive_cache[counterparty_id]

            cosine = self._compute_twin_score(counterparty_id)
            profile = self._counterparty_profiles.get(counterparty_id, [])
            own = self._own_trace[-self.counterfactual_horizon :]

            # Need paired decisions to compute conditional probabilities.
            # Match by index (they interacted at the same timesteps).
            n = min(len(profile), len(own))
            if n < 3:
                result = SubjunctiveDependence(
                    cosine_similarity=cosine,
                    conditional_agreement=cosine,
                    conditional_defection=cosine,
                    mutual_information=0.0,
                    subjunctive_score=cosine,
                )
                self._subjunctive_cache[counterparty_id] = result
                return result

        # Align traces: our recent decisions paired with theirs. Both
        # slices are [-n:] with n = min(len(profile), len(own)), so every
        # zip below is provably equal-length — strict=True documents and
        # enforces that instead of silently truncating on a future bug.
        our_decisions = [(acc, p) for acc, p in own[-n:]]
        their_decisions = [(acc, p) for acc, p in profile[-n:]]

        # Conditional agreement: P(they accept | we accept)
        we_accepted = [
            (oa, ta) for (oa, _), (ta, _) in zip(our_decisions, their_decisions, strict=True)
            if oa
        ]
        if we_accepted:
            cond_agree = sum(1 for _, ta in we_accepted if ta) / len(we_accepted)
        else:
            cond_agree = 0.5

        # Conditional defection: P(they reject | we reject)
        we_rejected = [
            (oa, ta) for (oa, _), (ta, _) in zip(our_decisions, their_decisions, strict=True)
            if not oa
        ]
        if we_rejected:
            cond_defect = sum(1 for _, ta in we_rejected if not ta) / len(we_rejected)
        else:
            cond_defect = 0.5

        # Mutual information: I(X; Y) for binary decisions.
        our_rate = sum(1 for a, _ in our_decisions if a) / n
        their_rate = sum(1 for a, _ in their_decisions if a) / n

        # Joint probabilities.
        both_accept = sum(
            1 for (oa, _), (ta, _) in zip(our_decisions, their_decisions, strict=True)
            if oa and ta
        ) / n
        both_reject = sum(
            1 for (oa, _), (ta, _) in zip(our_decisions, their_decisions, strict=True)
            if not oa and not ta
        ) / n
        us_accept_them_reject = sum(
            1 for (oa, _), (ta, _) in zip(our_decisions, their_decisions, strict=True)
            if oa and not ta
        ) / n
        us_reject_them_accept = sum(
            1 for (oa, _), (ta, _) in zip(our_decisions, their_decisions, strict=True)
            if not oa and ta
        ) / n

        # MI = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        eps = 1e-10
        for p_xy, p_x, p_y in [
            (both_accept, our_rate, their_rate),
            (both_reject, 1 - our_rate, 1 - their_rate),
            (us_accept_them_reject, our_rate, 1 - their_rate),
            (us_reject_them_accept, 1 - our_rate, their_rate),
        ]:
            if p_xy > eps and p_x > eps and p_y > eps:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))

        # Normalize MI to [0, 1] (max MI for binary vars is 1 bit).
        mi_norm = max(0.0, min(1.0, mi))

        # Subjunctive score: weighted combination.
        # High conditional agreement + MI > raw cosine similarity.
        subjunctive = (
            0.3 * cosine
            + 0.3 * cond_agree
            + 0.15 * cond_defect
            + 0.25 * mi_norm
        )
        subjunctive = max(0.0, min(1.0, subjunctive))

        result = SubjunctiveDependence(
            cosine_similarity=cosine,
            conditional_agreement=cond_agree,
            conditional_defection=cond_defect,
            mutual_information=mi_norm,
            subjunctive_score=subjunctive,
        )
        self._subjunctive_cache[counterparty_id] = result
        return result

    def _proof_based_cooperation(self, counterparty_id: str) -> Optional[bool]:
        """Attempt proof-based cooperation (TDT/FDT).

        If the subjunctive dependence score exceeds the proof threshold,
        we have high confidence that our cooperation *logically implies*
        their cooperation — analogous to Löb's theorem-based cooperation
        proofs in the formal TDT literature.

        Returns:
            True if proof succeeds (cooperate),
            False if proof fails (strong evidence of defection),
            None if inconclusive (fall through to other reasoning).
        """
        dep = self._compute_subjunctive_dependence(counterparty_id)

        if dep.subjunctive_score >= self.proof_threshold:
            # High subjunctive dependence = "proven twin".
            return True

        if dep.subjunctive_score < 0.2 and dep.conditional_agreement < 0.3:
            # Strong evidence of anti-correlation: they defect when we cooperate.
            return False

        return None  # Inconclusive — use standard reasoning.

    # ------------------------------------------------------------------
    # UDT: Policy precommitment
    # ------------------------------------------------------------------

    def _precommit_policy(self) -> bool:
        """Compute the precommitted cooperation decision (UDT).

        UDT agents commit to a policy *before* observing the specific
        interaction, based only on their prior beliefs. This makes them
        robust to predictors that model their decision procedure.

        The precommitted policy is: cooperate if
            cooperation_prior * (1 + welfare_weight) > defection_value
        evaluated at the prior (no updating from specific observations).
        """
        if self._precommitted_cooperate is not None:
            return self._precommitted_cooperate

        cf_coop = self.cooperation_prior
        welfare_bonus = self.welfare_weight * cf_coop
        coop_value = cf_coop + welfare_bonus
        defect_value = -0.5 * max(0.0, cf_coop - 0.5)

        self._precommitted_cooperate = coop_value > defect_value
        return self._precommitted_cooperate

    # ------------------------------------------------------------------
    # Core LDT reasoning helpers
    # ------------------------------------------------------------------

    def _counterfactual_cooperate_payoff(
        self, counterparty_id: str
    ) -> float:
        """Estimate the counterfactual payoff of cooperating.

        Uses the average ``p`` from past accepted interactions with
        this counterparty.  If no history, returns the cooperation prior.
        """
        profile = self._counterparty_profiles.get(counterparty_id, [])
        accepted = [p for acc, p in profile if acc]
        if not accepted:
            return self.cooperation_prior
        return sum(accepted) / len(accepted)

    def _counterfactual_defect_payoff(self, counterparty_id: str) -> float:
        """Estimate the counterfactual payoff of defecting.

        Defection means rejecting the interaction.  The payoff is 0
        (no interaction occurs), but we also lose opportunity cost
        proportional to how cooperative the counterparty has been.
        We return a small negative value reflecting the missed
        cooperation surplus.
        """
        coop_p = self._counterfactual_cooperate_payoff(counterparty_id)
        # Opportunity cost: half the expected surplus we'd forgo.
        return -0.5 * max(0.0, coop_p - 0.5)

    # ------------------------------------------------------------------
    # Level 2: Policy introspection
    # ------------------------------------------------------------------

    def _infer_counterparty_policy(
        self, counterparty_id: str
    ) -> InferredPolicy:
        """Infer a counterparty's decision parameters from their history.

        Estimates four parameters from observed behaviour and blends
        with a mirror prior ("they are like me") that fades as data grows.
        """
        if counterparty_id in self._inferred_policies:
            return self._inferred_policies[counterparty_id]

        profile = self._counterparty_profiles.get(counterparty_id, [])
        n = len(profile)
        confidence = min(n / max(self.counterfactual_horizon, 1), 1.0)

        if n == 0:
            # Pure mirror prior.
            policy = InferredPolicy(
                cooperation_prior=self.cooperation_prior,
                similarity_threshold=self.similarity_threshold,
                welfare_weight=self.welfare_weight,
                updateless_commitment=self.updateless_commitment,
                confidence=0.0,
                sample_count=0,
            )
            self._inferred_policies[counterparty_id] = policy
            return policy

        # 1. cooperation_prior ← acceptance rate.
        accepted_count = sum(1 for acc, _ in profile if acc)
        obs_coop_prior = accepted_count / n

        # 2. similarity_threshold ← variance of accepted p values.
        accepted_ps = [p for acc, p in profile if acc]
        if len(accepted_ps) >= 2:
            mean_p = sum(accepted_ps) / len(accepted_ps)
            var_p = sum((x - mean_p) ** 2 for x in accepted_ps) / len(
                accepted_ps
            )
            # Low variance = selective = high threshold.
            obs_sim_threshold = max(0.0, min(1.0, 1.0 - math.sqrt(var_p)))
        else:
            obs_sim_threshold = 0.5

        # 3. welfare_weight ← acceptance rate for marginal interactions.
        marginal = [acc for acc, p in profile if 0.4 <= p <= 0.6]
        if marginal:
            obs_welfare = sum(1 for a in marginal if a) / len(marginal)
        else:
            obs_welfare = 0.5

        # 4. updateless_commitment ← stability (early vs late drift).
        half = n // 2
        if half >= 2:
            early_rate = sum(1 for acc, _ in profile[:half] if acc) / half
            late_rate = sum(1 for acc, _ in profile[half:] if acc) / (
                n - half
            )
            drift = abs(late_rate - early_rate)
            obs_commitment = max(0.0, min(1.0, 1.0 - drift))
        else:
            obs_commitment = 0.5

        # Blend with mirror prior, fading as confidence grows.
        mirror_w = self.mirror_prior_weight * (1.0 - confidence)

        def _blend(observed: float, mine: float) -> float:
            return max(0.0, min(1.0, mirror_w * mine + (1.0 - mirror_w) * observed))

        policy = InferredPolicy(
            cooperation_prior=_blend(obs_coop_prior, self.cooperation_prior),
            similarity_threshold=_blend(
                obs_sim_threshold, self.similarity_threshold
            ),
            welfare_weight=_blend(obs_welfare, self.welfare_weight),
            updateless_commitment=_blend(
                obs_commitment, self.updateless_commitment
            ),
            confidence=confidence,
            sample_count=n,
        )
        self._inferred_policies[counterparty_id] = policy
        return policy

    def _simulate_counterparty_decision(
        self, counterparty_id: str
    ) -> bool:
        """Simulate whether the counterparty would cooperate with us.

        Creates a virtual Level 1 agent using the inferred policy
        parameters and checks if it would cooperate.
        """
        if counterparty_id in self._level2_cache:
            return self._level2_cache[counterparty_id]

        inferred = self._infer_counterparty_policy(counterparty_id)

        # Twin score is symmetric (cosine similarity).
        twin_score = self._compute_twin_score(counterparty_id)

        # Would they see us as a twin?
        if twin_score >= inferred.similarity_threshold:
            self._level2_cache[counterparty_id] = True
            return True

        # Simulate their counterfactual reasoning about us.
        # From their perspective, our acceptance rate is their cf_coop.
        our_profile = self._own_trace[-self.counterfactual_horizon :]
        if our_profile:
            our_accepted = [p for acc, p in our_profile if acc]
            cf_coop = (
                sum(our_accepted) / len(our_accepted)
                if our_accepted
                else inferred.cooperation_prior
            )
        else:
            cf_coop = inferred.cooperation_prior

        cf_defect = -0.5 * max(0.0, cf_coop - 0.5)

        welfare_bonus = inferred.welfare_weight * cf_coop
        coop_value = cf_coop + welfare_bonus
        defect_value = cf_defect

        effective_coop = (
            inferred.updateless_commitment * inferred.cooperation_prior
            + (1.0 - inferred.updateless_commitment) * coop_value
        )

        result = effective_coop > defect_value
        self._level2_cache[counterparty_id] = result
        return result

    # ------------------------------------------------------------------
    # Level 3: Recursive equilibrium
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def _best_response_probability(
        self,
        opponent_p_coop: float,
        twin_score: float,
        cooperation_prior: float,
        similarity_threshold: float,
        welfare_weight: float,
        updateless_commitment: float,
    ) -> float:
        """Soft (probabilistic) version of Level 1 decision.

        Returns a cooperation probability in [0, 1] rather than a
        hard boolean, enabling smooth iteration for equilibrium finding.
        """
        # Twin channel: sigmoid of how far twin_score exceeds threshold.
        twin_prob = self._sigmoid(
            10.0 * (twin_score - similarity_threshold)
        )

        # Payoff channel: counterfactual comparison.
        cf_coop = opponent_p_coop * cooperation_prior + (
            1.0 - opponent_p_coop
        ) * 0.5 * cooperation_prior
        cf_defect = -0.5 * max(0.0, cf_coop - 0.5)
        welfare_bonus = welfare_weight * cf_coop
        coop_value = cf_coop + welfare_bonus
        effective_coop = (
            updateless_commitment * cooperation_prior
            + (1.0 - updateless_commitment) * coop_value
        )
        advantage = effective_coop - cf_defect
        payoff_prob = self._sigmoid(5.0 * advantage)

        # Blend twin and payoff channels.
        twin_weight = twin_prob
        return max(0.0, min(1.0, twin_weight * 1.0 + (1.0 - twin_weight) * payoff_prob))

    def _recursive_equilibrium(self, counterparty_id: str) -> float:
        """Find the fixed-point cooperation probability via level-k iteration.

        Iterates best-response functions for both agents until convergence
        or the recursion depth cap is reached.

        Returns our equilibrium cooperation probability in [0, 1].
        """
        if counterparty_id in self._level3_cache:
            return self._level3_cache[counterparty_id]

        inferred = self._infer_counterparty_policy(counterparty_id)
        twin_score = self._compute_twin_score(counterparty_id)

        # Initialize beliefs.
        my_p = self.cooperation_prior
        their_p = inferred.cooperation_prior

        for _ in range(self.max_recursion_depth):
            new_my_p = self._best_response_probability(
                their_p,
                twin_score,
                self.cooperation_prior,
                self.similarity_threshold,
                self.welfare_weight,
                self.updateless_commitment,
            )
            new_their_p = self._best_response_probability(
                my_p,
                twin_score,
                inferred.cooperation_prior,
                inferred.similarity_threshold,
                inferred.welfare_weight,
                inferred.updateless_commitment,
            )

            # Apply introspection discount per level.
            new_my_p = (
                self.introspection_discount * new_my_p
                + (1.0 - self.introspection_discount) * my_p
            )
            new_their_p = (
                self.introspection_discount * new_their_p
                + (1.0 - self.introspection_discount) * their_p
            )

            # Check convergence.
            if (
                abs(new_my_p - my_p) < self.convergence_epsilon
                and abs(new_their_p - their_p) < self.convergence_epsilon
            ):
                my_p = new_my_p
                break

            my_p = new_my_p
            their_p = new_their_p

        result = max(0.0, min(1.0, my_p))
        self._level3_cache[counterparty_id] = result
        return result

    # ------------------------------------------------------------------
    # Twin graph (depth >= 4)
    # ------------------------------------------------------------------

    def _rebuild_twin_graph(self) -> None:
        """Rebuild the twin graph from current counterparty profiles.

        Computes pairwise similarity between all known counterparties
        (including self-to-counterparty edges) and stores edges that
        exceed ``twin_graph_min_edge``.
        """
        agents = list(self._counterparty_profiles.keys())
        graph: Dict[str, Dict[str, float]] = {}

        # Edges from us to each counterparty.
        own_vec = self._own_behaviour_vector()
        for aid in agents:
            their_vec = self._counterparty_behaviour_vector(aid)
            sim = max(0.0, min(1.0, _cosine_similarity(own_vec, their_vec)))
            if sim >= self.twin_graph_min_edge:
                graph.setdefault(self.agent_id, {})[aid] = sim
                graph.setdefault(aid, {})[self.agent_id] = sim

        # Cross-edges between counterparties (O(N^2), trivial for N<20).
        for i, a in enumerate(agents):
            vec_a = self._counterparty_behaviour_vector(a)
            for b in agents[i + 1:]:
                vec_b = self._counterparty_behaviour_vector(b)
                sim = max(0.0, min(1.0, _cosine_similarity(vec_a, vec_b)))
                if sim >= self.twin_graph_min_edge:
                    graph.setdefault(a, {})[b] = sim
                    graph.setdefault(b, {})[a] = sim

        self._twin_graph = graph

    def _find_transitive_twins(
        self, counterparty_id: str
    ) -> List[Tuple[str, float]]:
        """BFS on the twin graph to find transitive twins.

        Returns ``(agent_id, decayed_score)`` pairs reachable within
        ``twin_graph_traversal_depth`` hops.  Similarity decays by
        ``twin_graph_decay`` per hop.
        """
        if counterparty_id in self._transitive_twin_cache:
            return self._transitive_twin_cache[counterparty_id]

        result: List[Tuple[str, float]] = []
        visited = {self.agent_id, counterparty_id}

        # BFS frontier: (agent_id, accumulated_score)
        frontier: List[Tuple[str, float]] = []

        # Seed from counterparty's neighbours.
        for neighbour, edge_score in self._twin_graph.get(
            counterparty_id, {}
        ).items():
            if neighbour not in visited:
                frontier.append((neighbour, edge_score * self.twin_graph_decay))
                visited.add(neighbour)

        depth = 1
        while frontier and depth < self.twin_graph_traversal_depth:
            result.extend(frontier)
            next_frontier: List[Tuple[str, float]] = []
            for node, score in frontier:
                for neighbour, edge_score in self._twin_graph.get(
                    node, {}
                ).items():
                    if neighbour not in visited:
                        decayed = score * edge_score * self.twin_graph_decay
                        if decayed > 0.01:  # prune negligible paths
                            next_frontier.append((neighbour, decayed))
                            visited.add(neighbour)
            frontier = next_frontier
            depth += 1

        # Add any remaining from the last frontier.
        result.extend(frontier)

        self._transitive_twin_cache[counterparty_id] = result
        return result

    def _twin_graph_cooperation(self, counterparty_id: str) -> float:
        """Level 4: twin-graph-enhanced cooperation probability.

        1. Gets Level 3 equilibrium probability.
        2. Finds transitive twins via BFS.
        3. Computes acausal bonus from cooperation history with twins.
        4. Blends: ``(1 - beta) * l3 + beta * bonus``.
        """
        l3_prob = self._recursive_equilibrium(counterparty_id)
        twins = self._find_transitive_twins(counterparty_id)

        if not twins:
            return l3_prob

        # Acausal bonus: weighted sum of cooperation rates with twins.
        bonus_sum = 0.0
        weight_sum = 0.0
        for twin_id, path_score in twins:
            profile = self._counterparty_profiles.get(twin_id, [])
            if profile:
                coop_rate = sum(1 for acc, _ in profile if acc) / len(profile)
                bonus_sum += path_score * coop_rate
                weight_sum += path_score

        acausal_bonus = bonus_sum / weight_sum if weight_sum > 0 else 0.0
        acausal_bonus = max(0.0, min(1.0, acausal_bonus))

        beta = self.acausal_bonus_weight
        final = (1.0 - beta) * l3_prob + beta * acausal_bonus
        return max(0.0, min(1.0, final))

    def _compute_l4_for_profile(
        self,
        counterparty_id: str,
        perturbed_profiles: Dict[str, List[Tuple[bool, float]]],
    ) -> float:
        """Pure-function Level 4 computation on perturbed profiles.

        Used by Monte Carlo sampling to avoid mutating caches.
        Recomputes Level 3 equilibrium + twin graph bonus from scratch
        using the provided perturbed profiles.
        """
        # Recompute twin score from perturbed data.
        profile = perturbed_profiles.get(counterparty_id, [])
        recent = profile[-self.counterfactual_horizon:]
        vec = [p if acc else -(1 - p) for acc, p in recent]
        while len(vec) < self.counterfactual_horizon:
            vec.append(0.5)

        own = self._own_behaviour_vector()
        twin_score = max(0.0, min(1.0, _cosine_similarity(own, vec)))

        # Simplified Level 3: single best-response iteration.
        inferred_coop_prior = (
            sum(1 for acc, _ in profile if acc) / len(profile)
            if profile
            else self.cooperation_prior
        )
        my_p = self._best_response_probability(
            inferred_coop_prior,
            twin_score,
            self.cooperation_prior,
            self.similarity_threshold,
            self.welfare_weight,
            self.updateless_commitment,
        )

        # Twin graph bonus from perturbed profiles.
        twins = self._find_transitive_twins(counterparty_id)
        if not twins:
            return my_p

        bonus_sum = 0.0
        weight_sum = 0.0
        for twin_id, path_score in twins:
            tp = perturbed_profiles.get(
                twin_id, self._counterparty_profiles.get(twin_id, [])
            )
            if tp:
                coop_rate = sum(1 for acc, _ in tp if acc) / len(tp)
                bonus_sum += path_score * coop_rate
                weight_sum += path_score

        bonus = bonus_sum / weight_sum if weight_sum > 0 else 0.0
        bonus = max(0.0, min(1.0, bonus))

        beta = self.acausal_bonus_weight
        return max(0.0, min(1.0, (1.0 - beta) * my_p + beta * bonus))

    def _monte_carlo_cooperation(self, counterparty_id: str) -> float:
        """Level 5: Monte Carlo counterfactual sampling.

        For each of N samples, perturbs counterparty p-values with
        Gaussian noise and computes Level 4 cooperation probability.
        Returns the mean across samples.
        """
        samples: List[float] = []

        for _ in range(self.n_counterfactual_samples):
            # Build perturbed profiles.
            perturbed: Dict[str, List[Tuple[bool, float]]] = {}
            for aid, profile in self._counterparty_profiles.items():
                perturbed_profile: List[Tuple[bool, float]] = []
                for acc, p in profile:
                    noise = self._mc_rng.gauss(0, self.counterfactual_noise_std)
                    p_noisy = max(0.0, min(1.0, p + noise))
                    perturbed_profile.append((acc, p_noisy))
                perturbed[aid] = perturbed_profile

            l4 = self._compute_l4_for_profile(counterparty_id, perturbed)
            samples.append(l4)

        return sum(samples) / len(samples) if samples else 0.5

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def _level1_cooperate_decision(self, counterparty_id: str) -> bool:
        """Level 1 LDT decision logic.

        In FDT mode (default): subjunctive dependence detection + proof-based
        cooperation before falling through to counterfactual reasoning.

        In TDT mode: behavioral twin detection via cosine similarity.

        In UDT mode: FDT logic + precommitment policy blending.
        """
        twin_score = self._compute_twin_score(counterparty_id)
        self._twin_scores[counterparty_id] = twin_score

        # --- FDT/UDT extension: subjunctive dependence ---
        if self.decision_theory in ("fdt", "udt"):
            # Try proof-based cooperation first.
            proof = self._proof_based_cooperation(counterparty_id)
            if proof is not None:
                return proof

            # Use subjunctive score instead of raw cosine similarity.
            dep = self._compute_subjunctive_dependence(counterparty_id)
            effective_twin = (
                self.subjunctive_weight * dep.subjunctive_score
                + (1 - self.subjunctive_weight) * twin_score
            )
        else:
            # TDT: use cosine similarity only.
            effective_twin = twin_score

        if effective_twin >= self.similarity_threshold:
            return True

        cf_coop = self._counterfactual_cooperate_payoff(counterparty_id)
        cf_defect = self._counterfactual_defect_payoff(counterparty_id)

        welfare_bonus = self.welfare_weight * cf_coop
        coop_value = cf_coop + welfare_bonus
        defect_value = cf_defect

        prior_coop_value = self.cooperation_prior
        effective_coop_value = (
            self.updateless_commitment * prior_coop_value
            + (1 - self.updateless_commitment) * coop_value
        )

        # --- UDT extension: blend with precommitted policy ---
        if self.decision_theory == "udt" and self.precommitment_strength > 0:
            precommit = self._precommit_policy()
            precommit_value = 1.0 if precommit else 0.0
            greedy_value = 1.0 if effective_coop_value > defect_value else 0.0
            blended = (
                self.precommitment_strength * precommit_value
                + (1 - self.precommitment_strength) * greedy_value
            )
            return blended > 0.5

        return effective_coop_value > defect_value

    def _ldt_cooperate_decision(self, counterparty_id: str) -> bool:
        """Core LDT decision: should we cooperate with this counterparty?

        Combines multiple acausality levels based on ``acausality_depth``:
        - depth=1: Behavioral twin detection (original logic)
        - depth=2: Level 1 + policy introspection refinement
        - depth=3: Weighted ensemble of all three levels
        - depth=4: Twin-graph-enhanced cooperation (transitive twins)
        - depth=5: Monte Carlo counterfactual sampling over Level 4
        """
        l1 = self._level1_cooperate_decision(counterparty_id)

        if self.acausality_depth <= 1:
            return l1

        # Level 2: policy introspection.
        inferred = self._infer_counterparty_policy(counterparty_id)
        l2 = self._simulate_counterparty_decision(counterparty_id)

        if self.acausality_depth == 2:
            if l1 and l2:
                return True
            if l1 and not l2:
                # Discount cooperation by confidence.
                return inferred.confidence < 0.5
            if not l1 and l2:
                # Boost cooperation by confidence.
                return inferred.confidence >= 0.5
            return False

        # Level 3: recursive equilibrium ensemble.
        l3_prob = self._recursive_equilibrium(counterparty_id)

        l1_score = 1.0 if l1 else 0.0
        l2_score = 1.0 if l2 else 0.0
        ensemble = 0.2 * l1_score + 0.3 * l2_score + 0.5 * l3_prob

        if self.acausality_depth == 3:
            return ensemble > 0.5

        # Level 4: twin-graph-enhanced cooperation.
        if self.acausality_depth == 4:
            l4_prob = self._twin_graph_cooperation(counterparty_id)
            return l4_prob > 0.5

        # depth >= 5: Monte Carlo over Level 4.
        l5_prob = self._monte_carlo_cooperation(counterparty_id)
        return l5_prob > 0.5

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Decide on an action.

        LDT agents prioritise:
        1. Handling pending proposals via LDT reasoning
        2. Handling marketplace bid decisions
        3. Working on tasks diligently
        4. Posting constructive content
        5. Voting based on quality
        6. Proposing interactions with high-twin-score agents
        """
        # 1. Handle pending interaction proposals.
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")
            if self._ldt_cooperate_decision(initiator_id):
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # 2. Handle marketplace bid decisions — accept lowest bid (fair).
        if observation.pending_bid_decisions:
            return self._handle_bid_decision(observation)

        # 3. Bid on bounties at fair value.
        if observation.available_bounties and self._rng.random() < 0.25:
            return self._bid_on_bounty(observation)

        # 4. Work on active tasks.
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self._work_on_task(task)

        # 5. Claim a task.
        if observation.can_claim_task and observation.available_tasks:
            best = self._select_task(observation.available_tasks, observation)
            if best:
                return self.create_claim_task_action(best["task_id"])

        # 6. Post constructive content.
        if observation.can_post and self._rng.random() < self.post_probability:
            content = self._generate_content(observation)
            if content:
                return self.create_post_action(content)

        # 7. Vote on quality.
        if (
            observation.can_vote
            and observation.visible_posts
            and self._rng.random() < self.vote_probability
        ):
            return self._vote_on_posts(observation)

        # 8. Propose interactions — prefer logical twins.
        if (
            observation.can_interact
            and observation.visible_agents
            and self._rng.random() < self.interact_probability
        ):
            return self._propose_interaction(observation)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept interactions using LDT cooperation logic."""
        return self._ldt_cooperate_decision(proposal.initiator_id)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose collaboration if LDT favours cooperation."""
        if not self._ldt_cooperate_decision(counterparty_id):
            return None

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Let's cooperate — our decision processes are aligned.",
            offered_transfer=0.0,
        )

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Update internal models after an interaction resolves."""
        super().update_from_outcome(interaction, payoff)

        counterparty = self._get_counterparty(interaction)

        # Record in counterparty profile.
        if counterparty not in self._counterparty_profiles:
            self._counterparty_profiles[counterparty] = []
        self._counterparty_profiles[counterparty].append(
            (interaction.accepted, interaction.p)
        )

        # Record own trace.
        self._own_trace.append((interaction.accepted, interaction.p))

        # Invalidate caches so they're recomputed next time.
        # _own_trace changed, which affects twin scores, level-2/3
        # decisions, and subjunctive dependence for *all* counterparties,
        # so we must clear every cache — not just the current one.
        with self._cache_lock:
            self._twin_scores.clear()
            self._inferred_policies.clear()
            self._level2_cache.clear()
            self._level3_cache.clear()
            self._subjunctive_cache.clear()
            self._transitive_twin_cache.clear()

        # Rebuild twin graph edges (depth >= 4).
        if self.acausality_depth >= 4:
            self._rebuild_twin_graph()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_bid_decision(self, observation: Observation) -> Action:
        """Accept the most qualified bid (lowest cost)."""
        bids = observation.pending_bid_decisions
        if not bids:
            return self.create_noop_action()
        best = min(bids, key=lambda b: b.get("bid_amount", float("inf")))
        return self.create_accept_bid_action(
            bounty_id=best["bounty_id"],
            bid_id=best["bid_id"],
        )

    def _bid_on_bounty(self, observation: Observation) -> Action:
        """Bid fairly on a bounty."""
        bounties = observation.available_bounties
        if not bounties:
            return self.create_noop_action()
        bounty = self._rng.choice(bounties)
        reward = bounty.get("reward_amount", 0)
        bid_amount = reward * self._rng.uniform(0.8, 0.95)
        return self.create_place_bid_action(
            bounty_id=bounty["bounty_id"],
            bid_amount=bid_amount,
            message="Fair bid — looking for mutual benefit.",
        )

    def _work_on_task(self, task: Dict) -> Action:
        """Submit quality work on a task."""
        task_id = task.get("task_id", "")
        content = f"Completed task {task_id[:8]}: [diligent, quality output]"
        return self.create_submit_output_action(task_id, content)

    def _select_task(
        self, tasks: List[Dict], observation: Observation
    ) -> Optional[Dict]:
        """Select best task considering reputation and difficulty."""
        my_rep = observation.agent_state.reputation
        eligible = [t for t in tasks if t.get("min_reputation", 0) <= my_rep]
        if not eligible:
            return None

        def score(t: Dict) -> float:
            bounty = t.get("bounty", 0)
            diff_map = {
                "trivial": 1,
                "easy": 2,
                "medium": 3,
                "hard": 4,
                "expert": 5,
            }
            difficulty = diff_map.get(t.get("difficulty", "medium"), 3)
            return float(bounty / difficulty)

        eligible.sort(key=score, reverse=True)
        return eligible[0]

    def _generate_content(self, observation: Observation) -> str:
        """Generate constructive content."""
        topics = [
            "Sharing an insight that may benefit the community.",
            "Here is some analysis that others might find useful.",
            "Observations from recent interactions — cooperation pays off.",
            "A note on building trust through consistent behaviour.",
        ]
        result: str = self._rng.choice(topics)
        return result

    def _vote_on_posts(self, observation: Observation) -> Action:
        """Vote on posts based on quality signals."""
        posts = observation.visible_posts
        voted_ids = {
            m.get("post_id") for m in self._memory if m.get("type") == "vote"
        }
        unvoted = [p for p in posts if p.get("post_id") not in voted_ids]

        if not unvoted:
            return self.create_noop_action()

        post = self._rng.choice(unvoted)
        post_id = post.get("post_id", "")

        net_votes = post.get("net_votes", 0)
        reply_count = post.get("reply_count", 0)

        # Vote on genuine quality signals, not bandwagon.
        if reply_count > 2 or net_votes > 0:
            direction = 1
        elif net_votes < -3:
            direction = -1
        else:
            direction = 1  # Default: give benefit of the doubt.

        self.remember({"type": "vote", "post_id": post_id, "direction": direction})
        return self.create_vote_action(post_id, direction)

    def _propose_interaction(self, observation: Observation) -> Action:
        """Propose interactions, preferring logical twins."""
        agents = observation.visible_agents
        candidates: List[Tuple[str, float]] = []

        for agent_info in agents:
            agent_id = agent_info.get("agent_id", "")
            if agent_id == self.agent_id:
                continue
            twin_score = self._compute_twin_score(agent_id)
            trust = self.compute_counterparty_trust(agent_id)
            # Blend twin score and trust.
            combined = 0.6 * twin_score + 0.4 * trust
            candidates.append((agent_id, combined))

        if not candidates:
            return self.create_noop_action()

        # Sort by combined score descending, pick the best.
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score = candidates[0]

        # Only propose if LDT reasoning says cooperate.
        if not self._ldt_cooperate_decision(best_id):
            return self.create_noop_action()

        return self.create_propose_action(
            counterparty_id=best_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Our decision processes seem aligned — let's collaborate.",
        )
