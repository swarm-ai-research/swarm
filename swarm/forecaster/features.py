"""Feature extraction for incoherence risk forecasting."""

from typing import Dict, Iterable

from swarm.models.interaction import SoftInteraction


def extract_structural_features(
    horizon_length: int,
    agent_count: int,
    action_space_size: int,
    adversarial_fraction: float,
) -> Dict[str, float]:
    """Build policy-stable structural feature vector."""
    return {
        "horizon_length": float(horizon_length),
        "agent_count": float(agent_count),
        "action_space_size": float(action_space_size),
        "adversarial_fraction": float(adversarial_fraction),
    }


def extract_behavioral_features(
    interactions: Iterable[SoftInteraction],
) -> Dict[str, float]:
    """Build within-episode behavioral feature vector.

    Computes uncertainty as 1.0 - abs(2 * p - 1.0) for each interaction.
    Precondition: SoftInteraction.p must be in [0.0, 1.0] per the type contract.
    """
    rows = list(interactions)
    if not rows:
        return {
            "behavioral_interaction_count": 0.0,
            "behavioral_acceptance_rate": 0.0,
            "behavioral_uncertainty_mean": 0.0,
            "behavioral_uncertainty_max": 0.0,
        }

    accepted = [row for row in rows if row.accepted]
    uncertainties = [1.0 - abs(2 * row.p - 1.0) for row in rows]
    return {
        "behavioral_interaction_count": float(len(rows)),
        "behavioral_acceptance_rate": float(len(accepted) / len(rows)),
        "behavioral_uncertainty_mean": float(sum(uncertainties) / len(uncertainties)),
        "behavioral_uncertainty_max": float(max(uncertainties)),
    }


def combine_feature_dicts(*feature_maps: Dict[str, float]) -> Dict[str, float]:
    """Merge feature dictionaries in order (later values overwrite earlier)."""
    combined: Dict[str, float] = {}
    for feature_map in feature_maps:
        combined.update(feature_map)
    return combined
