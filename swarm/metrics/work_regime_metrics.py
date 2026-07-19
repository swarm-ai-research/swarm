"""Work-regime drift metrics.

Tracks how interaction networks shift between cooperative and competitive
regimes by measuring coalition formation, clustering, and regime stability.

Also measures behavioral drift under stress: drift index, Gini inequality,
legitimacy score, quality degradation, and defection rate.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


# ======================================================================
# Dataclasses
# ======================================================================


@dataclass
class WorkRegimeMetrics:
    """Aggregate metrics describing the current work regime."""

    coalition_strength: float = 0.0
    """Average local clustering coefficient over eligible nodes (degree >= 2)."""

    n_nodes_sampled: int = 0
    """Number of nodes used to estimate coalition_strength."""

    n_eligible_nodes: int = 0
    """Total nodes with degree >= 2 before any sampling."""

    regime_label: str = "neutral"
    """Coarse label: 'cooperative', 'competitive', or 'neutral'."""

    per_agent_clustering: Dict[str, float] = field(default_factory=dict)
    """Per-agent local clustering coefficient (only for sampled nodes)."""


@dataclass
class WorkRegimeEpochMetrics:
    """Per-epoch summary of work-regime drift indicators."""

    epoch: int = 0

    # --- Behavioral (hard) ---
    strike_rate: float = 0.0           # fraction of agents who noop'd due to exit
    quality_degradation: float = 0.0   # avg p drop from initial for accepted interactions
    defection_rate: float = 0.0        # fraction of rejected proposals
    redistribution_proposals: int = 0  # count of reform-like posts

    # --- Structural ---
    mean_drift_index: float = 0.0      # avg L2 policy distance from initial
    max_drift_index: float = 0.0       # max across agents
    coalition_strength: float = 0.0    # community clustering coefficient
    legitimacy_score: float = 0.0      # procedural justice proxy

    # --- Aggregate ---
    mean_compliance: float = 0.0
    mean_cooperation_threshold: float = 0.0
    mean_redistribution_pref: float = 0.0
    mean_exit_propensity: float = 0.0
    mean_grievance: float = 0.0
    gini_payoff: float = 0.0          # payoff inequality

    # --- Per-agent snapshots ---
    agent_snapshots: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ======================================================================
# Coalition strength (O(k) set-intersection, bounded sampling)
# ======================================================================


def _build_adj(
    pairs: Sequence[Tuple[str, str]],
    agent_set: set,
) -> Dict[str, set]:
    """Build an undirected adjacency-set dict, ignoring self-loops and non-members."""
    adj: Dict[str, set] = defaultdict(set)
    for a, b in pairs:
        if a in agent_set and b in agent_set and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _sample_eligible(
    eligible: List[str],
    max_sample_nodes: int,
) -> List[str]:
    """Return a deterministic stride-based sample of *eligible* nodes.

    Uses ceiling division for the stride so that even when *eligible* is only
    slightly larger than *max_sample_nodes* the sample is spread uniformly
    rather than just taking the first *max_sample_nodes* entries.
    """
    n = len(eligible)
    if n <= max_sample_nodes:
        return eligible
    # ceiling division: stride >= 2, guaranteeing a distributed sample
    stride = -(-n // max_sample_nodes)
    return eligible[::stride]


def _clustering_per_node(
    nodes: List[str],
    adj: Dict[str, set],
) -> Dict[str, float]:
    """Compute local clustering coefficient for each node via set intersection.

    For node v with neighbours N(v) and degree k:

        C(v) = t / (k * (k - 1))

    where ``t = Σ_{u ∈ N(v)} |N(v) ∩ N(u)|``.  Each triangle is counted
    twice in *t* (once per endpoint in the pair), which cancels the usual
    factor of 2 in the denominator, giving the standard LCC formula.

    Precondition: All nodes in ``nodes`` must have degree >= 2, otherwise
    ZeroDivisionError will be raised. Callers should filter eligible nodes
    before invoking this function (typically via ``if len(adj[v]) >= 2``).
    """
    result: Dict[str, float] = {}
    for v in nodes:
        neighbors_v = adj[v]
        k = len(neighbors_v)
        t = sum(len(neighbors_v & adj[u]) for u in neighbors_v)
        result[v] = t / (k * (k - 1))
    return result


def compute_coalition_strength(
    interaction_pairs: Sequence[Tuple[str, str]],
    agent_ids: Sequence[str],
    max_sample_nodes: int = 500,
) -> float:
    """Estimate coalition strength via average local clustering coefficient.

    Builds an undirected interaction graph and returns the mean local clustering
    coefficient over all nodes whose degree is >= 2.  Isolated nodes and nodes
    with only one neighbour cannot form triangles and are excluded so that sparse
    periphery agents do not dilute the coalition signal.  In very sparse networks
    this may overestimate coalition formation since only the well-connected core
    contributes.  Returns 0.0 when no node has degree >= 2 (i.e. no triangles
    are possible).

    Triangle counting uses set intersection (O(k) per neighbour) rather than a
    nested loop (O(k²)), giving a meaningful speedup on dense subgraphs.  When
    the number of eligible nodes exceeds *max_sample_nodes* a deterministic
    stride-based sample is taken so that runtime stays bounded for large-scale
    simulations.

    Args:
        interaction_pairs: Sequence of (initiator_id, counterparty_id) pairs.
        agent_ids: All agent IDs to consider as nodes.
        max_sample_nodes: Maximum number of eligible nodes to inspect.
            When the eligible set is larger, a uniform stride-based sample
            (deterministic, no randomness) is taken.  Default: 500.

    Returns:
        Coalition strength in [0.0, 1.0].  0.0 means no clustering; 1.0 means
        every pair of neighbours of every sampled node are themselves connected.
    """
    if len(agent_ids) < 3 or not interaction_pairs:
        return 0.0

    adj = _build_adj(interaction_pairs, set(agent_ids))
    eligible: List[str] = [v for v in adj if len(adj[v]) >= 2]
    if not eligible:
        return 0.0

    sampled = _sample_eligible(eligible, max_sample_nodes)
    cc_map = _clustering_per_node(sampled, adj)
    return sum(cc_map.values()) / len(cc_map)


def compute_work_regime_metrics(
    interactions: List[SoftInteraction],
    agent_ids: Optional[Sequence[str]] = None,
    max_sample_nodes: int = 500,
    cooperative_threshold: float = 0.4,
    competitive_threshold: float = 0.1,
) -> WorkRegimeMetrics:
    """Compute full work-regime metrics from a list of interactions.

    Args:
        interactions: Completed interactions in the epoch.
        agent_ids: Explicit agent roster.  Inferred from interactions if None.
        max_sample_nodes: Forwarded to :func:`compute_coalition_strength`.
        cooperative_threshold: Coalition strength above this → 'cooperative'.
        competitive_threshold: Coalition strength below this → 'competitive'.

    Returns:
        A :class:`WorkRegimeMetrics` instance.
    """
    if not interactions:
        return WorkRegimeMetrics()

    if agent_ids is None:
        ids: List[str] = list(
            {i.initiator for i in interactions}
            | {i.counterparty for i in interactions}
        )
    else:
        ids = list(agent_ids)

    pairs: List[Tuple[str, str]] = [(i.initiator, i.counterparty) for i in interactions]
    adj = _build_adj(pairs, set(ids))
    eligible: List[str] = [v for v in adj if len(adj[v]) >= 2]
    n_eligible = len(eligible)

    sampled = _sample_eligible(eligible, max_sample_nodes)
    per_agent = _clustering_per_node(sampled, adj)
    coalition_strength = sum(per_agent.values()) / len(per_agent) if per_agent else 0.0

    if coalition_strength >= cooperative_threshold:
        label = "cooperative"
    elif coalition_strength <= competitive_threshold:
        label = "competitive"
    else:
        label = "neutral"

    return WorkRegimeMetrics(
        coalition_strength=coalition_strength,
        n_nodes_sampled=len(sampled),
        n_eligible_nodes=n_eligible,
        regime_label=label,
        per_agent_clustering=per_agent,
    )


# ======================================================================
# Drift / inequality / legitimacy metrics
# ======================================================================


def compute_drift_index(
    policy_snapshots: Sequence[Dict[str, float]],
) -> Tuple[float, float]:
    """Compute mean and max drift index across agent policy snapshots.

    Each snapshot must contain a 'drift' key (L2 distance from initial).

    Returns:
        (mean_drift, max_drift)
    """
    if not policy_snapshots:
        logger.warning(
            "compute_drift_index called with empty policy_snapshots; "
            "returning (0, 0). Check that WorkRegimeAgents are present "
            "and producing snapshots."
        )
        return 0.0, 0.0
    drifts = [s.get("drift", 0.0) for s in policy_snapshots]
    return sum(drifts) / len(drifts), max(drifts)


def compute_gini(values: Sequence[float]) -> float:
    """Compute Gini coefficient for a sequence of non-negative values.

    Returns 0.0 for perfect equality, approaches 1.0 for maximum inequality.
    """
    if not values or len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def compute_legitimacy_score(
    *,
    audit_precision: float = 1.0,
    eval_noise: float = 0.0,
    appeals_available: bool = False,
    explanation_provided: bool = False,
) -> float:
    """Compute governance legitimacy as a function of procedural justice.

    Higher = more legitimate governance.  Range [0, 1].

    Components:
        - Audit precision (high precision → fewer false positives → more trust)
        - Evaluation noise (low noise → more predictable → more trust)
        - Appeals process availability
        - Explanation requirement
    """
    score = 0.0
    # Precision component (weight 0.4)
    score += 0.4 * audit_precision
    # Low noise component (weight 0.3)
    score += 0.3 * (1.0 - min(1.0, eval_noise))
    # Appeals (weight 0.15)
    score += 0.15 * (1.0 if appeals_available else 0.0)
    # Explanation (weight 0.15)
    score += 0.15 * (1.0 if explanation_provided else 0.0)
    return max(0.0, min(1.0, score))


def compute_quality_degradation(
    interactions: Sequence[SoftInteraction],
    baseline_p: float = 0.7,
) -> float:
    """Measure quality degradation as avg drop in p from baseline.

    Returns a non-negative value; 0 = no degradation, higher = worse.
    """
    if not interactions:
        return 0.0
    accepted = [i for i in interactions if i.accepted]
    if not accepted:
        return 0.0
    avg_p = sum(i.p for i in accepted) / len(accepted)
    return float(max(0.0, baseline_p - avg_p))


def compute_defection_rate(
    interactions: Sequence[SoftInteraction],
) -> float:
    """Fraction of interactions that were rejected."""
    if not interactions:
        return 0.0
    rejected = sum(1 for i in interactions if not i.accepted)
    return rejected / len(interactions)


def build_epoch_metrics(
    *,
    epoch: int,
    policy_snapshots: Dict[str, Dict[str, float]],
    interactions: Sequence[SoftInteraction],
    agent_payoffs: Dict[str, float],
    strike_count: int = 0,
    total_agents: int = 1,
    interaction_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    legitimacy_kwargs: Optional[Dict] = None,
) -> WorkRegimeEpochMetrics:
    """Build a complete WorkRegimeEpochMetrics for one epoch.

    Args:
        epoch: epoch number
        policy_snapshots: {agent_id: snapshot_dict} from WorkRegimeAgent
        interactions: all interactions this epoch
        agent_payoffs: {agent_id: total_payoff}
        strike_count: how many agents chose to strike/exit
        total_agents: total population size
        interaction_pairs: (a, b) pairs for coalition computation
        legitimacy_kwargs: keyword args for compute_legitimacy_score
    """
    snapshots_list = list(policy_snapshots.values())
    if not snapshots_list:
        logger.warning(
            "build_epoch_metrics: no policy snapshots for epoch %d; "
            "aggregate means will be zero. Ensure WorkRegimeAgents "
            "provide snapshots via policy_snapshot().",
            epoch,
        )
    mean_drift, max_drift = compute_drift_index(snapshots_list)

    payoff_values = list(agent_payoffs.values()) if agent_payoffs else []

    # Aggregate policy means
    mean_compliance = (
        sum(s.get("compliance_propensity", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )
    mean_coop = (
        sum(s.get("cooperation_threshold", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )
    mean_redist = (
        sum(s.get("redistribution_preference", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )
    mean_exit = (
        sum(s.get("exit_propensity", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )
    mean_grievance = (
        sum(s.get("grievance", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )

    legitimacy = compute_legitimacy_score(**(legitimacy_kwargs or {}))

    coalition = 0.0
    if interaction_pairs:
        coalition = compute_coalition_strength(
            interaction_pairs, list(policy_snapshots.keys())
        )

    return WorkRegimeEpochMetrics(
        epoch=epoch,
        strike_rate=strike_count / max(total_agents, 1),
        quality_degradation=compute_quality_degradation(interactions),
        defection_rate=compute_defection_rate(interactions),
        mean_drift_index=mean_drift,
        max_drift_index=max_drift,
        coalition_strength=coalition,
        legitimacy_score=legitimacy,
        mean_compliance=mean_compliance,
        mean_cooperation_threshold=mean_coop,
        mean_redistribution_pref=mean_redist,
        mean_exit_propensity=mean_exit,
        mean_grievance=mean_grievance,
        gini_payoff=compute_gini(payoff_values),
        agent_snapshots=dict(policy_snapshots),
    )
