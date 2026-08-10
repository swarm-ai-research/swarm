"""Collusion detection metrics and algorithms.

Detects coordinated manipulation patterns among agents through:
- Pair-level analysis: unusual interaction frequencies, mutual benefit correlation
- Group-level analysis: clique detection, synchronized actions
- Statistical tests: deviation from expected distributions
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from swarm.models.interaction import SoftInteraction


@dataclass
class PairMetrics:
    """Metrics for a pair of agents."""

    agent_a: str
    agent_b: str

    # Interaction patterns
    interaction_count: int = 0
    mutual_acceptance_rate: float = 0.0  # Rate both parties accept

    # Benefit asymmetry
    avg_benefit_a: float = 0.0  # Average net benefit to agent A
    avg_benefit_b: float = 0.0  # Average net benefit to agent B
    benefit_correlation: float = 0.0  # Correlation of benefits

    # Quality patterns
    avg_p_when_a_initiates: float = 0.5
    avg_p_when_b_initiates: float = 0.5

    # Timing patterns
    avg_time_between_interactions: float = 0.0
    interaction_burstiness: float = 0.0  # Coefficient of variation

    # Collusion score (composite)
    collusion_score: float = 0.0


@dataclass
class GroupMetrics:
    """Metrics for a group of potentially colluding agents."""

    members: Set[str] = field(default_factory=set)

    # Group cohesion
    internal_interaction_rate: float = 0.0  # Rate of within-group interactions
    external_interaction_rate: float = 0.0  # Rate of interactions with outsiders

    # Coordinated behavior
    voting_alignment: float = 0.0  # How often group members vote together
    target_overlap: float = 0.0  # Overlap in who they interact with

    # Quality asymmetry
    avg_internal_p: float = 0.5  # Average p for within-group interactions
    avg_external_p: float = 0.5  # Average p for outside interactions

    # Directional quality asymmetry (bead mwve): outbound = accepted
    # interactions *initiated by* members toward non-members. An export-harm
    # coalition trades high-p internally and low-p outbound; the undirected
    # avg_external_p masks this when honest outsiders initiate high-p
    # interactions toward the group.
    avg_outbound_external_p: float = 0.5
    directional_quality_asymmetry: float = 0.0

    # How this group was found: "pair_clique" (clique of suspicious pairs)
    # or "mutual_favoritism" (group-first structural pass, bead mwve)
    detection_method: str = "pair_clique"

    # Composite score
    collusion_score: float = 0.0


@dataclass
class CollusionReport:
    """Complete collusion analysis report."""

    # Overall metrics
    ecosystem_collusion_risk: float = 0.0  # 0-1 risk score

    # Flagged pairs and groups
    suspicious_pairs: List[PairMetrics] = field(default_factory=list)
    suspicious_groups: List[GroupMetrics] = field(default_factory=list)

    # Aggregate statistics
    avg_pair_collusion_score: float = 0.0
    max_pair_collusion_score: float = 0.0
    n_flagged_pairs: int = 0
    n_flagged_groups: int = 0

    # Per-agent risk scores
    agent_collusion_risk: Dict[str, float] = field(default_factory=dict)


class CollusionDetector:
    """
    Detects potential collusion patterns in agent interactions.

    Uses multiple signals:
    1. Unusual interaction frequency between pairs
    2. Highly correlated benefits (both parties consistently gain)
    3. Quality asymmetry (high p within group, low p to outsiders)
    4. Temporal clustering of interactions
    5. Coordinated voting patterns
    """

    def __init__(
        self,
        frequency_threshold: float = 2.0,  # Z-score for unusual frequency
        benefit_correlation_threshold: float = 0.7,
        quality_asymmetry_threshold: float = 0.2,
        min_interactions_for_analysis: int = 3,
        collusion_score_threshold: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Initialize collusion detector.

        Args:
            frequency_threshold: Z-score threshold for flagging unusual frequency
            benefit_correlation_threshold: Correlation threshold for mutual benefit
            quality_asymmetry_threshold: Difference in avg p (internal vs external)
            min_interactions_for_analysis: Minimum interactions to analyze a pair
            collusion_score_threshold: Threshold for flagging suspicious pairs
            seed: Random seed for any stochastic analysis
        """
        self.frequency_threshold = frequency_threshold
        self.benefit_correlation_threshold = benefit_correlation_threshold
        self.quality_asymmetry_threshold = quality_asymmetry_threshold
        self.min_interactions = min_interactions_for_analysis
        self.collusion_threshold = collusion_score_threshold
        self._rng = np.random.default_rng(seed)

    def analyze(
        self,
        interactions: List[SoftInteraction],
        agent_ids: Optional[List[str]] = None,
    ) -> CollusionReport:
        """
        Analyze interactions for collusion patterns.

        Args:
            interactions: List of interactions to analyze
            agent_ids: Optional list of all agent IDs (for complete analysis)

        Returns:
            CollusionReport with detailed findings
        """
        if not interactions:
            return CollusionReport()

        # Discover agents from interactions if not provided
        if agent_ids is None:
            agent_ids = list(
                {i.initiator for i in interactions}
                | {i.counterparty for i in interactions}
            )
        else:
            # Filter out interactions involving non-agent entities
            agent_set = set(agent_ids)
            interactions = [
                i
                for i in interactions
                if i.initiator in agent_set and i.counterparty in agent_set
            ]
            if not interactions:
                return CollusionReport()

        # Build interaction matrices
        pair_interactions = self._group_by_pair(interactions)

        # Compute pair-level metrics
        pair_metrics = {}
        for (a, b), ints in pair_interactions.items():
            if len(ints) >= self.min_interactions:
                metrics = self._compute_pair_metrics(a, b, ints, interactions)
                pair_metrics[(a, b)] = metrics

        # Identify suspicious pairs
        suspicious_pairs = [
            m
            for m in pair_metrics.values()
            if m.collusion_score >= self.collusion_threshold
        ]
        suspicious_pairs.sort(key=lambda x: x.collusion_score, reverse=True)

        # Detect potential groups (cliques of suspicious pairs)
        suspicious_groups = self._detect_groups(suspicious_pairs, interactions)

        # Group-first structural pass (bead mwve): export-harm coalitions
        # trade high-p internally and dump low-p interactions on outsiders,
        # so no individual pair ever looks suspicious and the clique pass
        # above never sees them. Discover candidate groups directly from
        # mutual-favoritism structure and flag on directional quality
        # asymmetry — the criterion quality_asymmetry_threshold documents
        # but the pair-first pipeline never evaluated.
        flagged_members = [frozenset(g.members) for g in suspicious_groups]
        for group in self._detect_groups_structural(interactions, agent_ids):
            if frozenset(group.members) not in flagged_members:
                suspicious_groups.append(group)
        suspicious_groups.sort(key=lambda g: g.collusion_score, reverse=True)

        # Compute per-agent risk scores
        agent_risk = self._compute_agent_risk(
            agent_ids, pair_metrics, suspicious_groups
        )

        # Compute ecosystem-level risk
        ecosystem_risk = self._compute_ecosystem_risk(
            pair_metrics, suspicious_groups, len(agent_ids)
        )

        return CollusionReport(
            ecosystem_collusion_risk=ecosystem_risk,
            suspicious_pairs=suspicious_pairs,
            suspicious_groups=suspicious_groups,
            avg_pair_collusion_score=float(
                np.mean([m.collusion_score for m in pair_metrics.values()])
            )
            if pair_metrics
            else 0.0,
            max_pair_collusion_score=max(
                (m.collusion_score for m in pair_metrics.values()), default=0.0
            ),
            n_flagged_pairs=len(suspicious_pairs),
            n_flagged_groups=len(suspicious_groups),
            agent_collusion_risk=agent_risk,
        )

    def _group_by_pair(
        self, interactions: List[SoftInteraction]
    ) -> Dict[Tuple[str, str], List[SoftInteraction]]:
        """Group interactions by agent pair (unordered)."""
        pairs: Dict[Tuple[str, str], List[SoftInteraction]] = defaultdict(list)
        for i in interactions:
            # Use sorted tuple for unordered pair
            a, b = sorted([i.initiator, i.counterparty])
            key: Tuple[str, str] = (a, b)
            pairs[key].append(i)
        return dict(pairs)

    def _compute_pair_metrics(
        self,
        agent_a: str,
        agent_b: str,
        pair_interactions: List[SoftInteraction],
        all_interactions: List[SoftInteraction],
    ) -> PairMetrics:
        """Compute detailed metrics for an agent pair."""
        n = len(pair_interactions)

        # Acceptance rates
        accepted = [i for i in pair_interactions if i.accepted]
        mutual_acceptance = len(accepted) / n if n > 0 else 0.0

        # Benefits (using tau as proxy for direct benefit)
        benefits_a = []
        benefits_b = []
        p_when_a_initiates = []
        p_when_b_initiates = []

        for i in pair_interactions:
            if i.initiator == agent_a:
                # A initiated: A pays tau, B receives tau
                benefits_a.append(-i.tau - i.c_a)
                benefits_b.append(i.tau - i.c_b)
                p_when_a_initiates.append(i.p)
            else:
                # B initiated: B pays tau, A receives tau
                benefits_b.append(-i.tau - i.c_a)
                benefits_a.append(i.tau - i.c_b)
                p_when_b_initiates.append(i.p)

        avg_benefit_a = np.mean(benefits_a) if benefits_a else 0.0
        avg_benefit_b = np.mean(benefits_b) if benefits_b else 0.0

        # Benefit correlation (do they both gain/lose together?)
        if len(benefits_a) >= 2:
            if np.std(benefits_a) > 0 and np.std(benefits_b) > 0:
                benefit_corr = np.corrcoef(benefits_a, benefits_b)[0, 1]
                if np.isnan(benefit_corr):
                    benefit_corr = 0.0
            else:
                benefit_corr = 0.0
        else:
            benefit_corr = 0.0

        # Quality metrics
        avg_p_a = float(np.mean(p_when_a_initiates)) if p_when_a_initiates else 0.5
        avg_p_b = float(np.mean(p_when_b_initiates)) if p_when_b_initiates else 0.5

        # Timing patterns
        timestamps = sorted(i.timestamp for i in pair_interactions)
        if len(timestamps) >= 2:
            deltas = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            avg_time = float(np.mean(deltas))
            burstiness = (
                float(np.std(deltas) / np.mean(deltas)) if np.mean(deltas) > 0 else 0.0
            )
        else:
            avg_time = 0.0
            burstiness = 0.0

        # Compute frequency z-score vs. all pairs
        all_pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for i in all_interactions:
            a2, b2 = sorted([i.initiator, i.counterparty])
            all_pair_counts[(a2, b2)] += 1

        counts = list(all_pair_counts.values())
        if len(counts) >= 2 and np.std(counts) > 0:
            frequency_zscore = float((n - np.mean(counts)) / np.std(counts))
        else:
            frequency_zscore = 0.0

        # Compute collusion score
        collusion_score = self._compute_pair_collusion_score(
            frequency_zscore=frequency_zscore,
            benefit_correlation=benefit_corr,
            mutual_acceptance_rate=mutual_acceptance,
            avg_p=float(np.mean([avg_p_a, avg_p_b])),
            n_interactions=n,
        )

        return PairMetrics(
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_count=n,
            mutual_acceptance_rate=mutual_acceptance,
            avg_benefit_a=float(avg_benefit_a),
            avg_benefit_b=float(avg_benefit_b),
            benefit_correlation=float(benefit_corr),
            avg_p_when_a_initiates=float(avg_p_a),
            avg_p_when_b_initiates=float(avg_p_b),
            avg_time_between_interactions=float(avg_time),
            interaction_burstiness=float(burstiness),
            collusion_score=collusion_score,
        )

    def _compute_pair_collusion_score(
        self,
        frequency_zscore: float,
        benefit_correlation: float,
        mutual_acceptance_rate: float,
        avg_p: float,
        n_interactions: int,
    ) -> float:
        """
        Compute composite collusion score for a pair.

        Higher score = more suspicious.
        """
        # Component scores (0-1 range)

        # High frequency is suspicious
        freq_score = (
            min(1.0, max(0.0, frequency_zscore / self.frequency_threshold))
            if frequency_zscore > 0
            else 0.0
        )

        # High benefit correlation is suspicious (both consistently gain)
        corr_score = max(0.0, benefit_correlation)

        # Very high acceptance rate is suspicious
        accept_score = max(0.0, (mutual_acceptance_rate - 0.5) * 2)

        # Low quality (low p) interactions that are still accepted is suspicious
        quality_score = (
            max(0.0, (0.5 - avg_p) * 2) if mutual_acceptance_rate > 0.5 else 0.0
        )

        # Weight and combine
        weights = {
            "frequency": 0.25,
            "correlation": 0.30,
            "acceptance": 0.20,
            "quality": 0.25,
        }

        score = (
            weights["frequency"] * freq_score
            + weights["correlation"] * corr_score
            + weights["acceptance"] * accept_score
            + weights["quality"] * quality_score
        )

        # Confidence adjustment based on sample size
        confidence = min(1.0, n_interactions / 10.0)

        return float(score * confidence)

    def _detect_groups(
        self,
        suspicious_pairs: List[PairMetrics],
        all_interactions: List[SoftInteraction],
    ) -> List[GroupMetrics]:
        """Detect groups of colluding agents from suspicious pairs."""
        if not suspicious_pairs:
            return []

        # Build adjacency from suspicious pairs
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for pair in suspicious_pairs:
            adjacency[pair.agent_a].add(pair.agent_b)
            adjacency[pair.agent_b].add(pair.agent_a)

        # Find connected components (potential collusion groups)
        visited: Set[str] = set()
        groups: List[Set[str]] = []

        for start in adjacency:
            if start in visited:
                continue

            # BFS to find component
            component: Set[str] = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= 2:
                groups.append(component)

        # Compute metrics for each group
        group_metrics = []
        for members in groups:
            metrics = self._compute_group_metrics(members, all_interactions)
            if metrics.collusion_score >= self.collusion_threshold:
                group_metrics.append(metrics)

        return sorted(group_metrics, key=lambda g: g.collusion_score, reverse=True)

    def _compute_group_metrics(
        self,
        members: Set[str],
        interactions: List[SoftInteraction],
    ) -> GroupMetrics:
        """Compute metrics for a potential collusion group."""
        internal = []
        external = []

        for i in interactions:
            init_in = i.initiator in members
            counter_in = i.counterparty in members

            if init_in and counter_in:
                internal.append(i)
            elif init_in or counter_in:
                external.append(i)

        n_internal = len(internal)
        n_external = len(external)
        total = n_internal + n_external

        # Interaction rates
        internal_rate = n_internal / total if total > 0 else 0.0
        external_rate = n_external / total if total > 0 else 0.0

        # Quality asymmetry
        avg_internal_p = float(np.mean([i.p for i in internal])) if internal else 0.5
        avg_external_p = float(np.mean([i.p for i in external])) if external else 0.5

        # Directional quality asymmetry (bead mwve): restrict to *accepted*
        # interactions and to the outbound direction (member-initiated toward
        # outsiders). Rejected junk is governance working, and inbound
        # high-p traffic from honest outsiders must not dilute the evidence
        # of outbound predation.
        internal_acc = [i for i in internal if i.accepted]
        outbound_acc = [
            i for i in external if i.accepted and i.initiator in members
        ]
        avg_internal_acc_p = (
            float(np.mean([i.p for i in internal_acc])) if internal_acc else 0.5
        )
        avg_outbound_p = (
            float(np.mean([i.p for i in outbound_acc])) if outbound_acc else 0.5
        )
        directional_asymmetry = (
            float(max(0.0, avg_internal_acc_p - avg_outbound_p))
            if internal_acc and outbound_acc
            else 0.0
        )

        # Compute collusion score for group
        # High internal rate + quality asymmetry = suspicious
        internal_bias = max(0.0, internal_rate - 0.5) * 2
        quality_asymmetry = float(max(0.0, avg_internal_p - avg_external_p))

        collusion_score = internal_bias * 0.4 + quality_asymmetry * 0.6

        # Adjust for group size (larger groups are more concerning)
        size_factor = min(1.0, len(members) / 5.0)
        collusion_score *= 0.5 + 0.5 * size_factor

        return GroupMetrics(
            members=members,
            internal_interaction_rate=float(internal_rate),
            external_interaction_rate=float(external_rate),
            avg_internal_p=float(avg_internal_p),
            avg_external_p=float(avg_external_p),
            avg_outbound_external_p=float(avg_outbound_p),
            directional_quality_asymmetry=float(directional_asymmetry),
            collusion_score=float(collusion_score),
        )

    def _detect_groups_structural(
        self,
        interactions: List[SoftInteraction],
        agent_ids: List[str],
    ) -> List[GroupMetrics]:
        """Group-first coalition detection via mutual favoritism (bead mwve).

        The pair-first pipeline structurally cannot see export-harm
        coalitions: members trade high-p with each other (so no pair scores
        as "trading junk") and dump low-p extraction on outsiders (but each
        predator-victim pair fails the mutual-benefit tests). Ground truth
        that motivated this: the emergent RL coalition in
        ``scenarios/rl_emergence.yaml`` capacity=3 seed=44 — internal
        p=0.716, outbound p=0.487, 0 pairs flagged.

        Discovery: build an undirected "mutual favoritism" graph — edge
        a—b iff each initiates accepted interactions toward the other at
        above-population mean quality (with at least ``min_interactions``
        accepted interactions per direction). Connected components are
        candidate coalitions: sets of agents who reserve their good
        behavior for each other. A uniform-quality honest population forms
        components too, but scores ~0 asymmetry and is not flagged.

        Flagging: a candidate is suspicious iff its *directional* quality
        asymmetry (internal accepted p minus outbound-external accepted p)
        meets ``quality_asymmetry_threshold`` — the criterion this
        detector's constructor has always documented for group quality
        asymmetry.
        """
        accepted = [i for i in interactions if i.accepted]
        if not accepted:
            return []

        population_mean_p = float(np.mean([i.p for i in accepted]))

        # Directed per-pair quality: mean p of accepted a -> b interactions
        directed: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for i in accepted:
            directed[(i.initiator, i.counterparty)].append(i.p)

        # Mutual-favoritism edges
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for (a, b), ps_ab in directed.items():
            if a >= b:
                continue  # handle each unordered pair once
            ps_ba = directed.get((b, a), [])
            if len(ps_ab) < self.min_interactions or len(ps_ba) < self.min_interactions:
                continue
            if (
                float(np.mean(ps_ab)) >= population_mean_p
                and float(np.mean(ps_ba)) >= population_mean_p
            ):
                adjacency[a].add(b)
                adjacency[b].add(a)

        # Connected components (deterministic: sorted iteration)
        visited: Set[str] = set()
        candidates: List[Set[str]] = []
        for start in sorted(adjacency):
            if start in visited:
                continue
            component: Set[str] = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(
                    n for n in sorted(adjacency[node]) if n not in visited
                )
            if len(component) >= 2:
                candidates.append(component)

        n_agents = len(agent_ids)
        groups: List[GroupMetrics] = []
        for component in candidates:
            # Flag decision on the FULL mutual-favoritism component: pruning
            # before this test would reclassify genuine in-group edges as
            # outbound and collapse the asymmetry that defines the coalition.
            component_metrics = self._compute_group_metrics(component, interactions)

            n_internal_acc = sum(
                1
                for i in accepted
                if i.initiator in component and i.counterparty in component
            )
            n_outbound_acc = sum(
                1
                for i in accepted
                if i.initiator in component and i.counterparty not in component
            )
            # Evidence floor: enough accepted traffic on both sides of the
            # asymmetry for the statistic to mean anything.
            if (
                n_internal_acc < self.min_interactions
                or n_outbound_acc < self.min_interactions
            ):
                continue

            if (
                component_metrics.directional_quality_asymmetry
                < self.quality_asymmetry_threshold
            ):
                continue

            # Membership refinement (bead mwve boundary fix): mutual-
            # favoritism edges capture honest agents who merely *receive*
            # the group's good behavior. A predator engages the out-group
            # (that's who it extracts from); a captured recipient never
            # initiates outside the component. Keep only members with real
            # out-group engagement. "Inside" stays = component, so this
            # never reclassifies in-group edges (unlike shrinking the
            # asymmetry reference).
            members = {
                m
                for m in component
                if sum(
                    1
                    for i in accepted
                    if i.initiator == m and i.counterparty not in component
                )
                >= self.min_interactions
            }
            if len(members) < 2:
                # No genuine predators — a favor-exchange cluster with no
                # victims. Not a harm-exporting coalition.
                continue

            metrics = self._compute_group_metrics(members, interactions)
            if metrics.directional_quality_asymmetry < self.quality_asymmetry_threshold:
                continue

            # Score on the structural signature: internal concentration
            # relative to the uniform-mixing expectation for this group
            # size (not the arbitrary 0.5 the clique path uses), plus the
            # directional asymmetry that triggered the flag.
            expected_internal = (
                (len(members) - 1) / (n_agents - 1) if n_agents > 1 else 0.0
            )
            internal_bias = (
                max(0.0, metrics.internal_interaction_rate - expected_internal)
                / (1.0 - expected_internal)
                if expected_internal < 1.0
                else 0.0
            )
            score = internal_bias * 0.4 + metrics.directional_quality_asymmetry * 0.6
            size_factor = min(1.0, len(members) / 5.0)
            score *= 0.5 + 0.5 * size_factor

            metrics.detection_method = "mutual_favoritism"
            metrics.collusion_score = float(score)
            groups.append(metrics)

        return sorted(groups, key=lambda g: g.collusion_score, reverse=True)

    def _compute_agent_risk(
        self,
        agent_ids: List[str],
        pair_metrics: Dict[Tuple[str, str], PairMetrics],
        groups: List[GroupMetrics],
    ) -> Dict[str, float]:
        """Compute per-agent collusion risk score."""
        risk: Dict[str, float] = dict.fromkeys(agent_ids, 0.0)

        # Contribution from pair scores
        for (a, b), metrics in pair_metrics.items():
            risk[a] = max(risk[a], metrics.collusion_score)
            risk[b] = max(risk[b], metrics.collusion_score)

        # Contribution from group membership
        for group in groups:
            for member in group.members:
                if member in risk:
                    risk[member] = max(risk[member], group.collusion_score)

        return risk

    def _compute_ecosystem_risk(
        self,
        pair_metrics: Dict[Tuple[str, str], PairMetrics],
        groups: List[GroupMetrics],
        n_agents: int,
    ) -> float:
        """Compute overall ecosystem collusion risk."""
        if not pair_metrics and not groups:
            return 0.0

        # Component 1: Fraction of pairs that are suspicious
        n_suspicious_pairs = sum(
            1
            for m in pair_metrics.values()
            if m.collusion_score >= self.collusion_threshold
        )
        n_possible_pairs = n_agents * (n_agents - 1) / 2
        pair_fraction = (
            n_suspicious_pairs / n_possible_pairs if n_possible_pairs > 0 else 0.0
        )

        # Component 2: Fraction of agents in suspicious groups
        agents_in_groups: Set[str] = set()
        for g in groups:
            agents_in_groups.update(g.members)
        group_fraction = len(agents_in_groups) / n_agents if n_agents > 0 else 0.0

        # Component 3: Maximum group collusion score
        max_group_score = max((g.collusion_score for g in groups), default=0.0)

        # Weighted combination
        ecosystem_risk = (
            0.3 * pair_fraction + 0.3 * group_fraction + 0.4 * max_group_score
        )

        return float(min(1.0, ecosystem_risk))


def detect_vote_coordination(
    votes: List[Dict],
    threshold: float = 0.8,
) -> List[Tuple[str, str, float]]:
    """
    Detect coordinated voting patterns.

    Args:
        votes: List of vote records with 'voter', 'target', 'direction'
        threshold: Alignment threshold for flagging pairs

    Returns:
        List of (agent_a, agent_b, alignment_score) for suspicious pairs
    """
    if not votes:
        return []

    # Group votes by target
    target_votes: Dict[str, Dict[str, int]] = defaultdict(dict)
    for vote in votes:
        voter = vote.get("voter", "")
        target = vote.get("target", "")
        direction = vote.get("direction", 0)
        if voter and target:
            target_votes[target][voter] = direction

    # Compute voter alignment for each pair
    voters_set: Set[str] = set()
    for tv in target_votes.values():
        voters_set.update(tv.keys())

    voters_list: List[str] = list(voters_set)
    suspicious_pairs = []

    for i, voter_a in enumerate(voters_list):
        for voter_b in voters_list[i + 1 :]:
            # Find targets they both voted on
            common_targets = []
            for target, votes_dict in target_votes.items():
                if voter_a in votes_dict and voter_b in votes_dict:
                    common_targets.append(target)

            if len(common_targets) < 3:
                continue

            # Compute alignment
            agreements = sum(
                1
                for t in common_targets
                if target_votes[t][voter_a] == target_votes[t][voter_b]
            )
            alignment = agreements / len(common_targets)

            if alignment >= threshold:
                suspicious_pairs.append((voter_a, voter_b, alignment))

    return sorted(suspicious_pairs, key=lambda x: x[2], reverse=True)


def temporal_clustering_score(
    interactions: List[SoftInteraction],
    window_seconds: float = 60.0,
) -> Dict[str, float]:
    """
    Compute temporal clustering score for each agent.

    Higher scores indicate interactions are clustered in time
    (potentially coordinated).

    Args:
        interactions: List of interactions
        window_seconds: Time window for clustering

    Returns:
        Dict mapping agent_id to clustering score
    """
    if not interactions:
        return {}

    # Group by agent
    agent_times: Dict[str, List[float]] = defaultdict(list)
    base_time = min(i.timestamp for i in interactions)

    for i in interactions:
        delta = (i.timestamp - base_time).total_seconds()
        agent_times[i.initiator].append(delta)
        agent_times[i.counterparty].append(delta)

    # Compute clustering for each agent
    scores: Dict[str, float] = {}
    for agent, times in agent_times.items():
        if len(times) < 3:
            scores[agent] = 0.0
            continue

        times = sorted(times)
        # Count interactions within window of each interaction
        cluster_counts = []
        for t in times:
            count = sum(1 for t2 in times if abs(t2 - t) <= window_seconds)
            cluster_counts.append(count)

        # High mean cluster count = high clustering
        avg_cluster = np.mean(cluster_counts)
        # Normalize by total count
        scores[agent] = (
            float((avg_cluster - 1) / (len(times) - 1)) if len(times) > 1 else 0.0
        )

    return scores
