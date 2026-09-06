"""Collusion detection metrics and algorithms.

Detects coordinated manipulation patterns among agents through:
- Pair-level analysis: unusual interaction frequencies, mutual benefit correlation
- Group-level analysis: clique detection, synchronized actions
- Statistical tests: deviation from expected distributions
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
class VolumeBurstResult:
    """Ecosystem-level activity-burst signal (bead hoer).

    A message-board channel defeats the pair/group/structural detectors (a hot
    hub page makes every co-editor a reply neighbour, so structural saturates and
    per-agent temporal measures identity fragmentation). The signal that actually
    led the human moderator was plain edit *volume*. This captures it: per fixed
    time window, the ratio of the window's interaction count to the trailing
    median of the preceding windows. A window fires when that ratio meets the
    threshold.

    NB: a burst is necessary but not sufficient for collusion. Emergent
    convergence — many independent agents that discover the same board — produces
    the same volume spike (bead 9err). This is a screening signal, not a verdict.
    """

    alarm: bool = False
    max_ratio: float = 0.0
    first_alarm: Optional[str] = None  # ISO timestamp of the first firing window
    peak_window: Optional[str] = None  # ISO timestamp of the max-ratio window
    peak_object: Optional[str] = None  # object driving the peak (per-object mode)
    n_windows: int = 0
    n_alarm_windows: int = 0
    windows: List[Dict] = field(default_factory=list)  # per-window detail rows


@dataclass
class CollusionReport:
    """Complete collusion analysis report."""

    # Overall metrics
    ecosystem_collusion_risk: float = 0.0  # 0-1 risk score

    # Ecosystem-level activity-burst signal (bead hoer); None if not computed
    volume_burst: Optional[VolumeBurstResult] = None
    volume_burst_object: Optional[VolumeBurstResult] = None

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
        volume_window_hours: float = 24.0,
        volume_trailing_windows: int = 7,
        volume_burst_threshold: float = 10.0,
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
        self.volume_window_hours = volume_window_hours
        self.volume_trailing_windows = volume_trailing_windows
        self.volume_burst_threshold = volume_burst_threshold
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

        # Ecosystem-level burst signal (bead hoer). Computed on the full input,
        # before the agent-set filter below, because volume is a property of the
        # channel, not of any agent pair.
        volume_burst = volume_burst_signal(
            interactions,
            window_hours=self.volume_window_hours,
            trailing_windows=self.volume_trailing_windows,
            threshold=self.volume_burst_threshold,
        )
        volume_burst_object = volume_burst_signal(
            interactions,
            window_hours=self.volume_window_hours,
            trailing_windows=self.volume_trailing_windows,
            threshold=self.volume_burst_threshold,
            per_object=True,
        )

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
                return CollusionReport(
                    volume_burst=volume_burst,
                    volume_burst_object=volume_burst_object,
                )

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
            volume_burst=volume_burst,
            volume_burst_object=volume_burst_object,
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


def volume_burst_signal(
    interactions: List[SoftInteraction],
    *,
    window_hours: float = 24.0,
    trailing_windows: int = 7,
    threshold: float = 10.0,
    min_baseline: float = 1.0,
    per_object: bool = False,
    object_key: str = "page_id",
    min_object_events: int = 5,
) -> VolumeBurstResult:
    """Ecosystem activity-burst detector (bead hoer).

    Bins interactions into fixed windows of ``window_hours`` (aligned to the day
    boundary of the first interaction) and, for each window, compares its count to
    the trailing median of the preceding ``trailing_windows`` windows. A window
    fires when ``count >= threshold * max(trailing_median, min_baseline)``. The
    ``max(..., min_baseline)`` floor both avoids divide-by-zero at cold start and
    keeps a lone edit against an all-zero history from reading as an infinite
    ratio; it is the "at least one event/window baseline" convention.

    The window series includes empty windows (they are real zero-activity
    periods), so the trailing median reflects true edits/window.

    Args:
        interactions: interactions to scan (each needs a ``timestamp``).
        window_hours: width of each time bin.
        trailing_windows: number of preceding windows in the trailing median.
        threshold: ratio at or above which a window fires.
        min_baseline: floor on the trailing-median denominator.
        per_object: if True, run the signal per object (``metadata[object_key]``)
            and report the object with the strongest burst; the hub page carried
            the incident's burst, so a per-object spike is a sharper signal.
        object_key: metadata key identifying the object in per-object mode.
        min_object_events: in per-object mode, a firing window must have at least
            this many events, so a page's first two edits do not trip the alarm.

    Returns:
        VolumeBurstResult. In per-object mode ``windows`` holds the firing windows
        across all objects and ``peak_object`` names the object at the max ratio.
    """
    if not interactions:
        return VolumeBurstResult()

    if per_object:
        return _volume_burst_per_object(
            interactions,
            window_hours=window_hours,
            trailing_windows=trailing_windows,
            threshold=threshold,
            min_baseline=min_baseline,
            object_key=object_key,
            min_object_events=min_object_events,
        )

    xs = sorted(interactions, key=lambda x: x.timestamp)
    step = timedelta(hours=window_hours)
    t0 = xs[0].timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    t_end = xs[-1].timestamp

    # Complete series of window counts, including empty windows.
    counts: List[int] = []
    starts: List[datetime] = []
    t = t0
    i = 0
    while t <= t_end:
        nxt = t + step
        c = 0
        while i < len(xs) and xs[i].timestamp < nxt:
            c += 1
            i += 1
        counts.append(c)
        starts.append(t)
        t = nxt

    result = VolumeBurstResult(n_windows=len(counts))
    for k, (start, count) in enumerate(zip(starts, counts, strict=True)):
        if k == 0:
            continue  # no trailing history yet
        trailing = counts[max(0, k - trailing_windows) : k]
        median = float(np.median(trailing)) if trailing else 0.0
        denom = max(median, min_baseline)
        ratio = count / denom
        fired = ratio >= threshold
        iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        if fired:
            result.n_alarm_windows += 1
            if result.first_alarm is None:
                result.first_alarm = iso
            result.windows.append(
                {
                    "window_start": iso,
                    "count": count,
                    "trailing_median": median,
                    "ratio": round(ratio, 3),
                }
            )
        if ratio > result.max_ratio:
            result.max_ratio = ratio
            result.peak_window = iso
    result.alarm = result.n_alarm_windows > 0
    return result


def _volume_burst_per_object(
    interactions: List[SoftInteraction],
    *,
    window_hours: float,
    trailing_windows: int,
    threshold: float,
    min_baseline: float,
    object_key: str,
    min_object_events: int,
) -> VolumeBurstResult:
    """Per-object variant of ``volume_burst_signal`` (see its docstring)."""
    by_object: Dict[str, List[SoftInteraction]] = defaultdict(list)
    for it in interactions:
        obj = it.metadata.get(object_key) if it.metadata else None
        if obj is None:
            continue
        by_object[str(obj)].append(it)

    merged = VolumeBurstResult()
    for obj, its in by_object.items():
        sub = volume_burst_signal(
            its,
            window_hours=window_hours,
            trailing_windows=trailing_windows,
            threshold=threshold,
            min_baseline=min_baseline,
            per_object=False,
        )
        merged.n_windows += sub.n_windows
        for row in sub.windows:
            if row["count"] < min_object_events:
                continue
            row = {**row, "object": obj}
            merged.windows.append(row)
            merged.n_alarm_windows += 1
            if merged.first_alarm is None or row["window_start"] < merged.first_alarm:
                merged.first_alarm = row["window_start"]
            if row["ratio"] > merged.max_ratio:
                merged.max_ratio = row["ratio"]
                merged.peak_window = row["window_start"]
                merged.peak_object = obj
    merged.alarm = merged.n_alarm_windows > 0
    # first_alarm should be the earliest firing window across objects
    if merged.windows:
        merged.first_alarm = min(r["window_start"] for r in merged.windows)
    return merged


# ---------------------------------------------------------------------------
# Population-level substrate monitor (bead vv3j.8)
# ---------------------------------------------------------------------------


@dataclass
class SubstratePopulationResult:
    """Traffic-shape monitor over a *population* of side hosts (bead vv3j.8).

    Teardown and revocation both act on the host that was caught, and a swarm
    with a choice of hosts sits on the one that will not be (pi02's negative
    ``substrate_gap``). A per-host volume burst therefore watches the wrong
    host by construction: after a teardown the caught host goes quiet and the
    population total does not. This monitor sums per-host write counts per
    window and applies the same trailing-median burst rule as
    :func:`volume_burst_signal`, and additionally reports how many hosts are
    live (``host_spread``) and how often one host goes quiet as another lights
    up (``relocations``). It needs per-host counts only, never content.

    Window index 0 has no trailing history and never fires. ``None`` for a
    first-alarm field means the signal never fired.
    """

    n_windows: int = 0
    n_hosts: int = 0
    threshold: float = 10.0
    population_ratio: List[float] = field(default_factory=list)
    population_first_alarm: Optional[int] = None
    per_host_first_alarm: Dict[str, Optional[int]] = field(default_factory=dict)
    any_host_first_alarm: Optional[int] = None
    host_spread: List[float] = field(default_factory=list)
    relocations: List[Dict] = field(default_factory=list)

    @property
    def relocation_count(self) -> int:
        return len(self.relocations)

    @property
    def population_alarm(self) -> bool:
        return self.population_first_alarm is not None


def _burst_ratios(
    counts: List[int], *, trailing_windows: int, min_baseline: float
) -> List[float]:
    """Per-window count / max(trailing median, min_baseline); window 0 is 0."""
    ratios: List[float] = []
    for k, count in enumerate(counts):
        if k == 0:
            ratios.append(0.0)
            continue
        trailing = counts[max(0, k - trailing_windows) : k]
        median = float(np.median(trailing)) if trailing else 0.0
        ratios.append(count / max(median, min_baseline))
    return ratios


def substrate_population_signal(
    counts_by_host: Dict[str, List[int]],
    *,
    trailing_windows: int = 7,
    threshold: float = 10.0,
    min_baseline: float = 1.0,
    relocation_lag: int = 2,
) -> SubstratePopulationResult:
    """Population-level volume burst, host spread, and relocation events.

    Args:
        counts_by_host: host id -> per-window write counts, all the same length
            (window = epoch in the simulation, day in a log replay).
        trailing_windows: windows in the trailing median.
        threshold: ratio at or above which a window fires.
        min_baseline: floor on the trailing-median denominator.
        relocation_lag: a host that goes quiet at window ``w`` is a relocation
            if another host lights up in ``[w, w + relocation_lag)``.
    """
    hosts = list(counts_by_host)
    if not hosts:
        return SubstratePopulationResult(threshold=threshold)
    n = len(next(iter(counts_by_host.values())))
    if any(len(c) != n for c in counts_by_host.values()):
        raise ValueError("every host needs the same number of windows")
    res = SubstratePopulationResult(n_windows=n, n_hosts=len(hosts), threshold=threshold)
    if n == 0:
        return res

    totals = [sum(counts_by_host[h][k] for h in hosts) for k in range(n)]
    res.population_ratio = _burst_ratios(
        totals, trailing_windows=trailing_windows, min_baseline=min_baseline
    )
    res.population_first_alarm = next(
        (k for k, r in enumerate(res.population_ratio) if r >= threshold), None
    )
    for h in hosts:
        ratios = _burst_ratios(
            list(counts_by_host[h]), trailing_windows=trailing_windows, min_baseline=min_baseline
        )
        res.per_host_first_alarm[h] = next(
            (k for k, r in enumerate(ratios) if r >= threshold), None
        )
    fired = [k for k in res.per_host_first_alarm.values() if k is not None]
    res.any_host_first_alarm = min(fired) if fired else None

    res.host_spread = [
        sum(1 for h in hosts if counts_by_host[h][k] > 0) / len(hosts) for k in range(n)
    ]

    for w in range(1, n):
        quiet = [h for h in hosts if counts_by_host[h][w - 1] > 0 and counts_by_host[h][w] == 0]
        if not quiet:
            continue
        for w2 in range(w, min(n, w + relocation_lag)):
            lit = [
                h for h in hosts
                if h not in quiet and counts_by_host[h][w2 - 1] == 0 and counts_by_host[h][w2] > 0
            ]
            if lit:
                for q in quiet:
                    res.relocations.append({"window": w, "quiet_host": q, "lit_host": lit[0], "lit_window": w2})
                break
    return res


# ---------------------------------------------------------------------------
# Content-free discriminators (bead vv3j.2)
#
# The wiki board defeats content inspection (pairwise/structural detectors were
# blind or saturated on the real log; volume led). These two signals need only
# *outputs and timestamps* — never channel content — and are the ecosystem-level
# companions to volume_burst_signal. Both are screening signals, not verdicts:
# honest convergence (bead 9err) produces the same shared answers and the same
# fast follow-ups as collusion, so neither separates convergence from collusion.
# That non-separation is the point — it is the observable half of the
# convergence–collusion identifiability problem. What separates the two is an
# *absence* (a routed control channel), which these content-free signals cannot
# see; the discriminator probe (bead vv3j.8) pairs them against a routing check.
# ---------------------------------------------------------------------------


def _precision_recall_f1(
    flagged: Set[str], truth: Set[str], universe: Set[str]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Precision/recall/F1 of ``flagged`` against ground-truth ``truth``.

    Returns ``(None, None, None)`` when ``truth`` is empty (no ground truth to
    score against). Only members of ``universe`` are considered.
    """
    if not truth:
        return (None, None, None)
    flagged = flagged & universe
    truth = truth & universe
    tp = len(flagged & truth)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return (precision, recall, f1)


@dataclass
class OutputCorrelationResult:
    """Answers that agree beyond the independent-work expectation (bead vv3j.2).

    For agents that share a task, the fraction whose delivered answer matches at
    least one peer. Convergence and collusion both drive this high, so a high
    reading screens for *either* — it does not distinguish them.
    """

    alarm: bool = False
    observed_agreement: float = 0.0  # share of agents whose answer is non-unique
    modal_share: float = 0.0  # share of agents on the single most common answer
    chance_agreement: float = 0.0  # Simpson index: pairwise-match rate the pooled
    # answer distribution alone would predict
    n_tasks: int = 0
    n_agents: int = 0
    flagged_agents: Set[str] = field(default_factory=set)
    per_task: List[Dict] = field(default_factory=list)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None


def output_correlation_signal(
    interactions: List[SoftInteraction],
    *,
    task_key: str = "task_id",
    answer_key: str = "answer",
    agreement_threshold: float = 0.5,
    min_agents_per_task: int = 2,
    ground_truth_copiers: Optional[Set[str]] = None,
) -> OutputCorrelationResult:
    """Output-correlation discriminator (bead vv3j.2).

    Groups interactions by ``metadata[task_key]`` and, within each task, reads
    each agent's answer from ``metadata[answer_key]`` (last write wins). An agent
    whose answer is shared by at least one other agent on the same task is
    *output-correlated*. ``observed_agreement`` is the share of such agents across
    all scored tasks. The alarm fires when it meets ``agreement_threshold``.

    ``chance_agreement`` reports the Simpson index (the pairwise-match rate the
    pooled answer distribution alone would predict) as context; it is *not* used
    to gate the alarm, because a chance correction built from the same pooled
    answers is a finite-sample artifact, not an independent baseline. Supply a
    copy-free control's ``observed_agreement`` externally if you need excess.

    Needs only outputs and the task grouping — never channel content. Honest
    convergence produces shared answers too (bead 9err), so this is a screening
    signal, not a verdict.

    Args:
        interactions: interactions to scan; each answer-bearing one needs
            ``metadata[answer_key]`` and (optionally) ``metadata[task_key]``.
        task_key: metadata key grouping interactions into tasks (default one task).
        answer_key: metadata key holding the delivered answer (any hashable).
        agreement_threshold: observed-agreement level at or above which to alarm.
        min_agents_per_task: tasks with fewer distinct answering agents are skipped.
        ground_truth_copiers: if given, agents known to have copied, for scoring
            precision/recall of ``flagged_agents``.

    Returns:
        OutputCorrelationResult.
    """
    # task -> {agent: answer} (last write wins)
    by_task: Dict[object, Dict[str, object]] = defaultdict(dict)
    for it in interactions:
        md = it.metadata or {}
        if answer_key not in md:
            continue
        agent = it.initiator
        if not agent:
            continue
        task = md.get(task_key, "__default__")
        by_task[task][agent] = md[answer_key]

    result = OutputCorrelationResult()
    all_agents: Set[str] = set()
    correlated_num = 0
    scored_agents = 0
    for task, answers in by_task.items():
        if len(answers) < min_agents_per_task:
            continue
        agents = list(answers)
        vals = [answers[a] for a in agents]
        n = len(agents)
        counts: Dict[object, int] = defaultdict(int)
        for v in vals:
            counts[v] += 1
        shared = {a for a in agents if counts[answers[a]] >= 2}
        modal = max(counts.values())
        simpson = sum((c / n) ** 2 for c in counts.values())
        result.flagged_agents |= shared
        all_agents.update(agents)
        correlated_num += len(shared)
        scored_agents += n
        result.n_tasks += 1
        result.per_task.append(
            {
                "task": str(task),
                "n_agents": n,
                "correlated_agents": len(shared),
                "observed_agreement": round(len(shared) / n, 3),
                "modal_share": round(modal / n, 3),
                "chance_agreement": round(simpson, 3),
                "distinct_answers": len(counts),
            }
        )

    result.n_agents = len(all_agents)
    if scored_agents:
        result.observed_agreement = correlated_num / scored_agents
        result.modal_share = float(
            np.mean([r["modal_share"] for r in result.per_task])
        )
        result.chance_agreement = float(
            np.mean([r["chance_agreement"] for r in result.per_task])
        )
    result.alarm = result.observed_agreement >= agreement_threshold
    if ground_truth_copiers is not None:
        p, r, f1 = _precision_recall_f1(
            result.flagged_agents, ground_truth_copiers, all_agents
        )
        result.precision, result.recall, result.f1 = p, r, f1
    return result


@dataclass
class RederivationResult:
    """Answers delivered faster than the task's minimum research time (vv3j.2).

    An answer that lands before an independent agent could have researched it had
    to come from somewhere else (a read of a peer's posted answer). Like
    output correlation, honest peer-reading trips this too, so it screens rather
    than convicts.
    """

    alarm: bool = False
    n_events: int = 0
    n_infeasible: int = 0
    infeasible_fraction: float = 0.0
    flagged_agents: Set[str] = field(default_factory=set)
    flagged_events: List[Dict] = field(default_factory=list)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None


def rederivation_infeasibility_signal(
    interactions: List[SoftInteraction],
    *,
    min_research_seconds: Optional[float] = None,
    min_research_key: str = "min_research_seconds",
    elapsed_key: str = "research_seconds",
    task_start_key: str = "task_start",
    alarm_fraction: float = 0.0,
    ground_truth_copiers: Optional[Set[str]] = None,
) -> RederivationResult:
    """Re-derivation-infeasibility discriminator (bead vv3j.2).

    For each answer-bearing interaction, the elapsed research time is taken from
    ``metadata[elapsed_key]`` (seconds) if present, else from
    ``timestamp - metadata[task_start_key]``. The minimum feasible research time
    is ``metadata[min_research_key]`` if present, else the scalar
    ``min_research_seconds``. An event whose elapsed time is below the minimum is
    *infeasible* — the answer arrived faster than independent research allows — and
    its initiator is flagged. The alarm fires when the infeasible fraction exceeds
    ``alarm_fraction`` (default: any infeasible event).

    Needs only timestamps and the per-task timing metadata — never channel
    content. Honest peer-reading also produces fast, infeasible answers, so this
    is a screening signal (bead 9err).

    Args:
        interactions: interactions to scan.
        min_research_seconds: fallback minimum research time when a task does not
            carry its own ``min_research_key``.
        min_research_key: per-task minimum-research-time metadata key.
        elapsed_key: metadata key with the elapsed research seconds, if precomputed.
        task_start_key: metadata key with the task-start ``datetime`` (used when
            ``elapsed_key`` is absent).
        alarm_fraction: infeasible fraction strictly above which to alarm.
        ground_truth_copiers: if given, agents known to have copied, for scoring.

    Returns:
        RederivationResult. Events without enough timing metadata to assess are
        ignored (not counted in ``n_events``).
    """
    result = RederivationResult()
    assessable_agents: Set[str] = set()
    for it in interactions:
        md = it.metadata or {}
        # minimum feasible research time for this event's task
        min_research = md.get(min_research_key, min_research_seconds)
        if min_research is None:
            continue
        # elapsed research time
        if elapsed_key in md:
            elapsed = float(md[elapsed_key])
        elif task_start_key in md:
            start = md[task_start_key]
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            elapsed = (it.timestamp - start).total_seconds()
        else:
            continue
        agent = it.initiator or ""
        assessable_agents.add(agent)
        result.n_events += 1
        if elapsed < float(min_research):
            result.n_infeasible += 1
            result.flagged_agents.add(agent)
            result.flagged_events.append(
                {
                    "agent": agent,
                    "elapsed_seconds": round(elapsed, 3),
                    "min_research_seconds": float(min_research),
                    "timestamp": it.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
    if result.n_events:
        result.infeasible_fraction = result.n_infeasible / result.n_events
    result.alarm = result.infeasible_fraction > alarm_fraction and result.n_infeasible > 0
    if ground_truth_copiers is not None:
        p, r, f1 = _precision_recall_f1(
            result.flagged_agents, ground_truth_copiers, assessable_agents
        )
        result.precision, result.recall, result.f1 = p, r, f1
    return result
