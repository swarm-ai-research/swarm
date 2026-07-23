"""Module C: Matching Market (Labor/Dating/Housing).

Two-sided matching with preference elicitation and mechanism choice.
Implements Deferred Acceptance, recommender baseline, and hybrid.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from swarm.csm.types import (
    MatchCandidate,
    MatchOutcome,
    PreferenceDimensionality,
    PreferenceModel,
)

# ---------------------------------------------------------------------------
# Matching mechanisms
# ---------------------------------------------------------------------------

class MatchingMechanism:
    """Base class for matching mechanisms."""

    def match(
        self,
        proposers: List[MatchCandidate],
        receivers: List[MatchCandidate],
        rng: Optional[np.random.Generator] = None,
    ) -> MatchOutcome:
        raise NotImplementedError


class DeferredAcceptance(MatchingMechanism):
    """Gale-Shapley Deferred Acceptance algorithm.

    Strategy-proof for the proposing side.
    """

    def match(
        self,
        proposers: List[MatchCandidate],
        receivers: List[MatchCandidate],
        rng: Optional[np.random.Generator] = None,
    ) -> MatchOutcome:
        if rng is None:
            rng = np.random.default_rng()

        n_proposers = len(proposers)
        n_receivers = len(receivers)

        if n_proposers == 0 or n_receivers == 0:
            return MatchOutcome()

        # Build preference rankings:
        # proposer_prefs[i] = list of receiver indices in preference order
        # receiver_prefs[j] = dict mapping proposer index -> rank
        proposer_prefs = self._rank_others(proposers, receivers, rng)
        receiver_ranks = self._rank_as_dict(receivers, proposers, rng)

        # DA state
        free_proposers = list(range(n_proposers))
        next_proposal = [0] * n_proposers  # Next receiver to propose to
        # receiver -> currently held proposer (or None)
        current_match: Dict[int, Optional[int]] = dict.fromkeys(range(n_receivers))

        while free_proposers:
            p = free_proposers.pop(0)

            if next_proposal[p] >= n_receivers:
                continue  # Exhausted all options

            r = proposer_prefs[p][next_proposal[p]]
            next_proposal[p] += 1

            current_holder = current_match[r]
            if current_holder is None:
                current_match[r] = p
            else:
                # Receiver compares current holder vs new proposer
                rank_new = receiver_ranks[r].get(p, n_proposers)
                rank_old = receiver_ranks[r].get(current_holder, n_proposers)
                if rank_new < rank_old:
                    current_match[r] = p
                    free_proposers.append(current_holder)
                else:
                    free_proposers.append(p)

        # Build matches
        matches = []
        for r_idx, p_idx in current_match.items():
            if p_idx is not None:
                matches.append((proposers[p_idx].candidate_id,
                                receivers[r_idx].candidate_id))

        # Compute welfare and stability
        welfare_p, welfare_r = self._compute_welfare(
            matches, proposers, receivers
        )
        stability = self._compute_stability(
            matches, proposers, receivers, proposer_prefs, receiver_ranks
        )
        congestion = n_proposers / max(n_receivers, 1)

        return MatchOutcome(
            matches=matches,
            stability_rate=stability,
            welfare_proposers=welfare_p,
            welfare_receivers=welfare_r,
            congestion_index=congestion,
        )

    def _compute_sorted_scores(
        self,
        rankers: List[MatchCandidate],
        targets: List[MatchCandidate],
        rng: np.random.Generator,
    ) -> List[List[tuple]]:
        """Compute sorted (index, score) tuples for each ranker's evaluation of targets."""
        rankings = []
        for ranker in rankers:
            scores = []
            for j, target in enumerate(targets):
                u = ranker.preferences.utility(target.attributes, 0.0)
                # Add noise
                if ranker.preferences.noise_std > 0:
                    u += float(rng.normal(0, ranker.preferences.noise_std))
                scores.append((j, u))
            scores.sort(key=lambda x: x[1], reverse=True)
            rankings.append(scores)
        return rankings

    def _rank_others(
        self,
        rankers: List[MatchCandidate],
        targets: List[MatchCandidate],
        rng: np.random.Generator,
    ) -> List[List[int]]:
        """Each ranker scores targets by preference utility, returns sorted indices."""
        sorted_scores = self._compute_sorted_scores(rankers, targets, rng)
        return [[idx for idx, _ in scores] for scores in sorted_scores]

    def _rank_as_dict(
        self,
        rankers: List[MatchCandidate],
        targets: List[MatchCandidate],
        rng: np.random.Generator,
    ) -> List[Dict[int, int]]:
        """Each ranker returns dict mapping target index -> rank (0 = best)."""
        sorted_scores = self._compute_sorted_scores(rankers, targets, rng)
        return [{idx: rank for rank, (idx, _) in enumerate(scores)} for scores in sorted_scores]

    def _compute_welfare(
        self,
        matches: List[tuple],
        proposers: List[MatchCandidate],
        receivers: List[MatchCandidate],
    ) -> Tuple[float, float]:
        """Compute total welfare for each side."""
        p_map = {c.candidate_id: c for c in proposers}
        r_map = {c.candidate_id: c for c in receivers}

        w_p = 0.0
        w_r = 0.0
        for p_id, r_id in matches:
            p_cand = p_map.get(p_id)
            r_cand = r_map.get(r_id)
            if p_cand and r_cand:
                w_p += p_cand.preferences.utility(r_cand.attributes, 0.0)
                w_r += r_cand.preferences.utility(p_cand.attributes, 0.0)

        return w_p, w_r

    def _compute_stability(
        self,
        matches: List[tuple],
        proposers: List[MatchCandidate],
        receivers: List[MatchCandidate],
        proposer_prefs: List[List[int]],
        receiver_ranks: List[Dict[int, int]],
    ) -> float:
        """Compute stability rate (1 - fraction of blocking pairs)."""
        n_proposers = len(proposers)
        n_receivers = len(receivers)

        if not matches:
            return 1.0

        # Build current match mappings
        p_id_to_idx = {c.candidate_id: i for i, c in enumerate(proposers)}
        r_id_to_idx = {c.candidate_id: i for i, c in enumerate(receivers)}

        p_matched_to: Dict[int, int] = {}  # proposer_idx -> receiver_idx
        r_matched_to: Dict[int, int] = {}  # receiver_idx -> proposer_idx

        for p_id, r_id in matches:
            pi = p_id_to_idx.get(p_id)
            ri = r_id_to_idx.get(r_id)
            if pi is not None and ri is not None:
                p_matched_to[pi] = ri
                r_matched_to[ri] = pi

        # Count blocking pairs
        blocking = 0
        total_pairs = 0

        for pi in range(n_proposers):
            current_r = p_matched_to.get(pi)
            pref_list = proposer_prefs[pi]

            for ri in pref_list:
                total_pairs += 1
                if current_r is not None and ri == current_r:
                    break  # All subsequent are less preferred

                # pi prefers ri over current match
                # Does ri prefer pi over their current match?
                current_p_for_r = r_matched_to.get(ri)
                rank_of_pi = receiver_ranks[ri].get(pi, n_proposers)

                if current_p_for_r is None:
                    # ri is unmatched, this is a blocking pair
                    blocking += 1
                else:
                    rank_of_current = receiver_ranks[ri].get(
                        current_p_for_r, n_proposers
                    )
                    if rank_of_pi < rank_of_current:
                        blocking += 1

        max_possible = n_proposers * n_receivers
        if max_possible == 0:
            return 1.0
        return 1.0 - (blocking / max_possible)


class RecommenderBaseline(MatchingMechanism):
    """Simple recommender baseline (platform ranking).

    Platform ranks matches by a noisy quality score.
    """

    def __init__(self, noise_std: float = 0.3):
        self.noise_std = noise_std

    def match(
        self,
        proposers: List[MatchCandidate],
        receivers: List[MatchCandidate],
        rng: Optional[np.random.Generator] = None,
    ) -> MatchOutcome:
        if rng is None:
            rng = np.random.default_rng()

        if not proposers or not receivers:
            return MatchOutcome()

        # Platform scores each (proposer, receiver) pair
        scores: List[Tuple[int, int, float]] = []
        for i, p in enumerate(proposers):
            for j, r in enumerate(receivers):
                # Noisy sum of mutual utility
                u_p = p.preferences.utility(r.attributes, 0.0)
                u_r = r.preferences.utility(p.attributes, 0.0)
                score = u_p + u_r + float(rng.normal(0, self.noise_std))
                scores.append((i, j, score))

        # Greedy assignment by score
        scores.sort(key=lambda x: x[2], reverse=True)
        matched_p: Set[int] = set()
        matched_r: Set[int] = set()
        matches = []

        for pi, ri, _ in scores:
            if pi in matched_p or ri in matched_r:
                continue
            matches.append((proposers[pi].candidate_id,
                            receivers[ri].candidate_id))
            matched_p.add(pi)
            matched_r.add(ri)

        # Welfare
        p_map = {c.candidate_id: c for c in proposers}
        r_map = {c.candidate_id: c for c in receivers}
        w_p = sum(
            p_map[pid].preferences.utility(r_map[rid].attributes, 0.0)
            for pid, rid in matches
        )
        w_r = sum(
            r_map[rid].preferences.utility(p_map[pid].attributes, 0.0)
            for pid, rid in matches
        )

        congestion = len(proposers) / max(len(receivers), 1)

        return MatchOutcome(
            matches=matches,
            stability_rate=0.0,  # Not computed for recommender
            welfare_proposers=w_p,
            welfare_receivers=w_r,
            congestion_index=congestion,
        )


class HybridMechanism(MatchingMechanism):
    """Hybrid: recommender pre-filters, then DA on top-k.

    Reduces congestion by limiting proposal set.
    """

    def __init__(self, top_k: int = 5, recommender_noise: float = 0.2):
        self.top_k = top_k
        self.recommender_noise = recommender_noise

    def match(
        self,
        proposers: List[MatchCandidate],
        receivers: List[MatchCandidate],
        rng: Optional[np.random.Generator] = None,
    ) -> MatchOutcome:
        if rng is None:
            rng = np.random.default_rng()

        if not proposers or not receivers:
            return MatchOutcome()

        # Phase 1: Recommender pre-filters for each proposer
        # Each proposer gets a shortlist of top-k receivers
        shortlists: List[List[int]] = []
        for prop in proposers:
            scores = []
            for j, recv in enumerate(receivers):
                u = prop.preferences.utility(recv.attributes, 0.0)
                u += float(rng.normal(0, self.recommender_noise))
                scores.append((j, u))
            scores.sort(key=lambda x: x[1], reverse=True)
            shortlists.append([idx for idx, _ in scores[: self.top_k]])

        # Phase 2: DA on shortlists
        n_proposers = len(proposers)
        n_receivers = len(receivers)

        # Receiver rankings (over all proposers)
        receiver_ranks: List[Dict[int, int]] = []
        for recv in receivers:
            scores = []
            for j, prop in enumerate(proposers):
                u = recv.preferences.utility(prop.attributes, 0.0)
                if recv.preferences.noise_std > 0:
                    u += float(rng.normal(0, recv.preferences.noise_std))
                scores.append((j, u))
            scores.sort(key=lambda x: x[1], reverse=True)
            receiver_ranks.append(
                {idx: rank for rank, (idx, _) in enumerate(scores)}
            )

        # DA with limited proposal lists
        free_proposers = list(range(n_proposers))
        next_proposal = [0] * n_proposers
        current_match: Dict[int, Optional[int]] = dict.fromkeys(range(n_receivers))

        while free_proposers:
            p = free_proposers.pop(0)
            shortlist = shortlists[p]

            if next_proposal[p] >= len(shortlist):
                continue

            r = shortlist[next_proposal[p]]
            next_proposal[p] += 1

            current_holder = current_match[r]
            if current_holder is None:
                current_match[r] = p
            else:
                rank_new = receiver_ranks[r].get(p, n_proposers)
                rank_old = receiver_ranks[r].get(current_holder, n_proposers)
                if rank_new < rank_old:
                    current_match[r] = p
                    free_proposers.append(current_holder)
                else:
                    free_proposers.append(p)

        matches = []
        for r_idx, p_idx in current_match.items():
            if p_idx is not None:
                matches.append((proposers[p_idx].candidate_id,
                                receivers[r_idx].candidate_id))

        # Welfare
        p_map = {c.candidate_id: c for c in proposers}
        r_map = {c.candidate_id: c for c in receivers}
        w_p = sum(
            p_map[pid].preferences.utility(r_map[rid].attributes, 0.0)
            for pid, rid in matches
        )
        w_r = sum(
            r_map[rid].preferences.utility(p_map[pid].attributes, 0.0)
            for pid, rid in matches
        )

        congestion = len(proposers) / max(len(receivers), 1)

        return MatchOutcome(
            matches=matches,
            stability_rate=0.0,  # Hybrid: stability not guaranteed
            welfare_proposers=w_p,
            welfare_receivers=w_r,
            congestion_index=congestion,
        )


# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------

def generate_candidates(
    n: int,
    side: str = "proposer",
    preference_dim: PreferenceDimensionality = PreferenceDimensionality.LOW,
    rng: Optional[np.random.Generator] = None,
) -> List[MatchCandidate]:
    """Generate candidates for one side of the matching market.

    Args:
        n: Number of candidates.
        side: "proposer" or "receiver".
        preference_dim: Preference complexity.
        rng: Random generator.

    Returns:
        List of MatchCandidate.
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates = []
    for i in range(n):
        if preference_dim == PreferenceDimensionality.LOW:
            attrs = {"skill": float(rng.beta(2, 2)),
                     "experience": float(rng.beta(2, 2))}
            weights = {
                "skill": float(rng.uniform(0.5, 2.0)),
                "experience": float(rng.uniform(0.3, 1.5)),
            }
            noise = 0.0
        else:
            attr_names = [
                "skill", "experience", "communication",
                "creativity", "reliability", "location",
                "culture_fit", "salary_expectation",
            ]
            attrs = {name: float(rng.beta(2, 2)) for name in attr_names}
            weights = {name: float(rng.uniform(0.1, 2.0)) for name in attr_names}
            noise = 0.1

        candidates.append(MatchCandidate(
            candidate_id=f"{side}_{i}",
            side=side,
            attributes=attrs,
            preferences=PreferenceModel(weights=weights, noise_std=noise),
        ))

    return candidates


# ---------------------------------------------------------------------------
# Congestion measurement
# ---------------------------------------------------------------------------

def measure_congestion(
    proposers: List[MatchCandidate],
    receivers: List[MatchCandidate],
    proposal_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Measure congestion in the matching market.

    Args:
        proposers: Proposing-side candidates.
        receivers: Receiving-side candidates.
        proposal_counts: Optional dict of receiver_id -> number of proposals received.

    Returns:
        Dict with congestion metrics.
    """
    ratio = len(proposers) / max(len(receivers), 1)

    if proposal_counts:
        counts = list(proposal_counts.values())
        mean_proposals = sum(counts) / max(len(counts), 1)
        max_proposals = max(counts) if counts else 0
        variance = (
            sum((c - mean_proposals) ** 2 for c in counts) / max(len(counts), 1)
        )
    else:
        mean_proposals = ratio
        max_proposals = 0
        variance = 0.0

    return {
        "proposer_receiver_ratio": ratio,
        "mean_proposals_per_receiver": mean_proposals,
        "max_proposals_per_receiver": max_proposals,
        "proposal_variance": variance,
    }
