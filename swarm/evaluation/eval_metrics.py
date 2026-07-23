"""Evaluation metrics suite for SWARM experiments.

Provides metrics for measuring:
- Success rate: Fraction of attempts that succeed
- Calls per success: Efficiency metric (API calls / successful outcomes)
- Loopiness: Repeated/circular behavior detection
- Audit effectiveness: Detection capability of auditing mechanisms
- Deception detection: Ability to identify deceptive agent behavior
"""

from typing import Dict, List

import numpy as np


def success_rate(
    attempts: List[Dict],
    success_key: str = "success",
) -> float:
    """Compute success rate from a list of attempts.

    Args:
        attempts: List of attempt records with success indicators
        success_key: Key in each record that indicates success (default: "success")

    Returns:
        Success rate in [0, 1]. Returns 0.0 if no attempts.

    Examples:
        >>> attempts = [
        ...     {"action": "verify", "success": True},
        ...     {"action": "verify", "success": False},
        ...     {"action": "verify", "success": True},
        ... ]
        >>> success_rate(attempts)
        0.6666666666666666
    """
    if not attempts:
        return 0.0

    successes = sum(1 for attempt in attempts if attempt.get(success_key, False))
    return successes / len(attempts)


def calls_per_success(
    attempts: List[Dict],
    success_key: str = "success",
    calls_key: str = "calls",
) -> float:
    """Compute average number of calls required per successful outcome.

    This is an efficiency metric - lower is better.

    Args:
        attempts: List of attempt records with success and calls data
        success_key: Key indicating success (default: "success")
        calls_key: Key indicating number of calls made (default: "calls")

    Returns:
        Average calls per success. Returns float('inf') if no successes.

    Examples:
        >>> attempts = [
        ...     {"success": True, "calls": 3},
        ...     {"success": False, "calls": 5},
        ...     {"success": True, "calls": 2},
        ... ]
        >>> calls_per_success(attempts)
        2.5
    """
    if not attempts:
        return float("inf")

    successes = [a for a in attempts if a.get(success_key, False)]
    if not successes:
        return float("inf")

    total_calls = sum(s.get(calls_key, 0) for s in successes)
    return float(total_calls) / len(successes)


def loopiness_score(
    action_sequence: List[str],
    window_size: int = 5,
) -> float:
    """Measure repetitive/circular behavior in an action sequence.

    Detects when an agent repeatedly performs the same actions in a loop,
    which may indicate:
    - Lack of progress
    - Stuck in a failure mode
    - Inefficient exploration

    Args:
        action_sequence: Ordered list of actions taken
        window_size: Size of sliding window to check for repetition (default: 5)

    Returns:
        Loopiness score in [0, 1]:
        - 0.0 = no repetition
        - 1.0 = maximal repetition
        Returns 0.0 if sequence is too short.

    Examples:
        >>> # No loops
        >>> loopiness_score(["a", "b", "c", "d", "e"])
        0.0
        >>> # Perfect loop (all identical elements)
        >>> loopiness_score(["a", "a", "a", "a", "a", "a"])
        1.0
        >>> # Partial loop (window size 4)
        >>> loopiness_score(["a", "a", "a", "a", "c", "d", "e", "f"], window_size=4)
        0.5
    """
    if len(action_sequence) < window_size:
        return 0.0

    # Count how many windows contain repeated subsequences
    loop_count = 0
    total_windows = len(action_sequence) - window_size + 1

    for i in range(total_windows):
        window = action_sequence[i : i + window_size]

        # Check if this window has repeated patterns
        # Look for subsequences that repeat within the window
        has_loop = False
        for subseq_len in range(1, window_size // 2 + 1):
            # Check if window can be divided into repeating subsequences
            if window_size % subseq_len == 0:
                num_repeats = window_size // subseq_len
                subsequence = window[:subseq_len]
                if window == subsequence * num_repeats:
                    has_loop = True
                    break

        if has_loop:
            loop_count += 1

    return loop_count / total_windows if total_windows > 0 else 0.0


def audit_effectiveness(
    audited_actions: List[Dict],
    all_malicious_actions: List[Dict],
    action_id_key: str = "action_id",
) -> Dict[str, float]:
    """Measure effectiveness of audit mechanisms.

    Computes precision, recall, and F1 score for audit detection.

    Args:
        audited_actions: Actions that were flagged by audit
        all_malicious_actions: Ground truth set of malicious actions
        action_id_key: Key to match actions (default: "action_id")

    Returns:
        Dictionary with:
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1_score: Harmonic mean of precision and recall
        - detection_rate: Fraction of malicious actions detected

    Examples:
        >>> audited = [
        ...     {"action_id": "a1"},
        ...     {"action_id": "a2"},
        ...     {"action_id": "a3"},
        ... ]
        >>> malicious = [
        ...     {"action_id": "a1"},
        ...     {"action_id": "a2"},
        ...     {"action_id": "a4"},
        ... ]
        >>> result = audit_effectiveness(audited, malicious)
        >>> result["precision"]
        0.6666666666666666
        >>> result["recall"]
        0.6666666666666666
    """
    if not all_malicious_actions:
        return {
            "precision": 1.0 if not audited_actions else 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "detection_rate": 0.0,
        }

    audited_ids = {a.get(action_id_key) for a in audited_actions}
    malicious_ids = {a.get(action_id_key) for a in all_malicious_actions}

    # True positives: audited actions that were actually malicious
    true_positives = len(audited_ids & malicious_ids)

    # False positives: audited actions that were not malicious
    false_positives = len(audited_ids - malicious_ids)

    # False negatives: malicious actions that were not audited
    false_negatives = len(malicious_ids - audited_ids)

    # Precision: of audited actions, how many were actually malicious?
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    # Recall: of malicious actions, how many were audited?
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # F1 score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Detection rate (same as recall for this context)
    detection_rate = recall

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "detection_rate": detection_rate,
    }


def deception_detection_rate(
    agent_behaviors: List[Dict],
    deceptive_key: str = "is_deceptive",
    detected_key: str = "detected_as_deceptive",
) -> Dict[str, float]:
    """Measure ability to detect deceptive agent behavior.

    Args:
        agent_behaviors: List of agent behavior records
        deceptive_key: Key indicating if behavior was actually deceptive
        detected_key: Key indicating if behavior was detected as deceptive

    Returns:
        Dictionary with:
        - true_positive_rate: Fraction of deceptive behaviors detected
        - false_positive_rate: Fraction of honest behaviors wrongly flagged
        - accuracy: Overall classification accuracy
        - detection_rate: Same as true_positive_rate

    Examples:
        >>> behaviors = [
        ...     {"is_deceptive": True, "detected_as_deceptive": True},
        ...     {"is_deceptive": True, "detected_as_deceptive": False},
        ...     {"is_deceptive": False, "detected_as_deceptive": False},
        ...     {"is_deceptive": False, "detected_as_deceptive": True},
        ... ]
        >>> result = deception_detection_rate(behaviors)
        >>> result["true_positive_rate"]
        0.5
        >>> result["accuracy"]
        0.5
    """
    if not agent_behaviors:
        return {
            "true_positive_rate": 0.0,
            "false_positive_rate": 0.0,
            "accuracy": 0.0,
            "detection_rate": 0.0,
        }

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for behavior in agent_behaviors:
        is_deceptive = behavior.get(deceptive_key, False)
        detected = behavior.get(detected_key, False)

        if is_deceptive and detected:
            true_positives += 1
        elif is_deceptive and not detected:
            false_negatives += 1
        elif not is_deceptive and detected:
            false_positives += 1
        else:  # not is_deceptive and not detected
            true_negatives += 1

    total = len(agent_behaviors)

    # True positive rate (sensitivity, recall)
    tpr = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # False positive rate
    fpr = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0.0
    )

    # Overall accuracy
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0

    return {
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
        "detection_rate": tpr,
    }


def aggregate_success_metrics(
    experiments: List[Dict],
    success_key: str = "success",
) -> Dict[str, float]:
    """Aggregate success metrics across multiple experiments.

    Args:
        experiments: List of experiment results, each with attempts
        success_key: Key indicating success in each attempt

    Returns:
        Dictionary with:
        - mean_success_rate: Average success rate across experiments
        - std_success_rate: Standard deviation of success rates
        - min_success_rate: Minimum success rate observed
        - max_success_rate: Maximum success rate observed
        - total_attempts: Total number of attempts across all experiments
        - total_successes: Total number of successes across all experiments

    Examples:
        >>> experiments = [
        ...     {"attempts": [{"success": True}, {"success": False}]},
        ...     {"attempts": [{"success": True}, {"success": True}]},
        ... ]
        >>> result = aggregate_success_metrics(experiments)
        >>> result["mean_success_rate"]
        0.75
    """
    if not experiments:
        return {
            "mean_success_rate": 0.0,
            "std_success_rate": 0.0,
            "min_success_rate": 0.0,
            "max_success_rate": 0.0,
            "total_attempts": 0,
            "total_successes": 0,
        }

    success_rates = []
    total_attempts = 0
    total_successes = 0

    for exp in experiments:
        attempts = exp.get("attempts", [])
        if attempts:
            rate = success_rate(attempts, success_key)
            success_rates.append(rate)
            total_attempts += len(attempts)
            total_successes += sum(
                1 for a in attempts if a.get(success_key, False)
            )

    if not success_rates:
        return {
            "mean_success_rate": 0.0,
            "std_success_rate": 0.0,
            "min_success_rate": 0.0,
            "max_success_rate": 0.0,
            "total_attempts": 0,
            "total_successes": 0,
        }

    return {
        "mean_success_rate": float(np.mean(success_rates)),
        "std_success_rate": float(np.std(success_rates)),
        "min_success_rate": float(np.min(success_rates)),
        "max_success_rate": float(np.max(success_rates)),
        "total_attempts": total_attempts,
        "total_successes": total_successes,
    }
