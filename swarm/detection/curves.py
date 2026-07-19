"""Detection-curve metrics: ROC/PR, AUROC/AUPRC, partial AUROC, threshold@FPR, time-to-detection.

These are the headline measurements that replace the narrative "soft metrics
flagged it": every detector is scored as a real classifier of degrading vs
benign agents, with curves across adversarial base rates and a time-to-detection
at a fixed, calibrated false-positive rate.

Partial AUROC (pAUROC) at low FPR (e.g. 0.05 and 0.01) is included because safety
monitoring almost always operates under tight false-alarm budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from swarm.detection.degradation import AgentStream
from swarm.detection.detectors import Detector, binarize_stream
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction


@dataclass
class DetectionCurve:
    """ROC and PR curves plus their summary areas for one detector.

    Also includes partial AUROC (pAUROC) at low false-positive operating points,
    which are the regimes that matter most for safety monitoring.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    roc_thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    auroc: float
    auprc: float
    base_rate: float  # prevalence of positives (degrading agents)

    # Partial AUROC at low FPR operating points (computed via sklearn max_fpr)
    pauroc_fpr05: float  # pAUROC for FPR <= 0.05 (primary safety-relevant point)
    pauroc_fpr01: float  # pAUROC for FPR <= 0.01 (very low false-alarm regime)

    def to_summary(self) -> dict:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "base_rate": self.base_rate,
            "pauroc_fpr05": self.pauroc_fpr05,
            "pauroc_fpr01": self.pauroc_fpr01,
        }


def compute_curve(scores: Sequence[float], labels: Sequence[int]) -> DetectionCurve:
    """Build ROC/PR curves and areas from per-agent scores and ground-truth labels.

    Degenerate cases (only one class present) are reported with AUROC = 0.5 and
    AUPRC = base rate, the standard no-information references.
    """
    # Imported lazily so `import swarm.detection` works without scikit-learn
    # (an `analysis` extra); only the curve computations require it.
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    base_rate = float(labels_arr.mean()) if labels_arr.size else 0.0

    if labels_arr.size == 0 or len(np.unique(labels_arr)) < 2:
        return DetectionCurve(
            fpr=np.array([0.0, 1.0]),
            tpr=np.array([0.0, 1.0]),
            roc_thresholds=np.array([np.inf, -np.inf]),
            precision=np.array([base_rate, base_rate]),
            recall=np.array([0.0, 1.0]),
            auroc=0.5,
            auprc=base_rate,
            base_rate=base_rate,
            pauroc_fpr05=0.5,
            pauroc_fpr01=0.5,
        )

    fpr, tpr, roc_thresholds = roc_curve(labels_arr, scores_arr)
    precision, recall, _ = precision_recall_curve(labels_arr, scores_arr)

    # Partial AUROC at low FPR (safety-critical regimes)
    # sklearn's roc_auc_score with max_fpr returns the normalized partial AUC
    pauroc05 = float(roc_auc_score(labels_arr, scores_arr, max_fpr=0.05))
    pauroc01 = float(roc_auc_score(labels_arr, scores_arr, max_fpr=0.01))

    return DetectionCurve(
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thresholds,
        precision=precision,
        recall=recall,
        auroc=float(roc_auc_score(labels_arr, scores_arr)),
        auprc=float(average_precision_score(labels_arr, scores_arr)),
        base_rate=base_rate,
        pauroc_fpr05=pauroc05,
        pauroc_fpr01=pauroc01,
    )


def threshold_at_fpr(negative_scores: Sequence[float], max_fpr: float = 0.05) -> float:
    """Smallest decision threshold holding FPR <= ``max_fpr`` on benign agents.

    Setting the threshold at the ``(1 - max_fpr)`` quantile of the benign score
    distribution means at most ``max_fpr`` of benign agents score at or above it.
    This is the operating point at which time-to-detection is reported.
    """
    neg = np.asarray(negative_scores, dtype=float)
    if neg.size == 0:
        return float("inf")
    return float(np.quantile(neg, 1.0 - max_fpr))


def time_to_detection(
    stream: AgentStream,
    detector: Detector,
    threshold: float,
    window: int = 4,
    min_epoch: int = 1,
) -> Optional[int]:
    """Epochs from onset until the detector's trailing-window score crosses ``threshold``.

    The detector is run on a trailing window of interactions ending at each epoch
    ``e``; the first ``e`` whose score exceeds ``threshold`` is the flag.

    The scan starts at ``max(min_epoch, onset, 1)`` — i.e. detection is only
    sought from the agent's onset onward, and never before epoch 1. This is deliberate: a flag *before* onset (on
    not-yet-degraded behaviour) is a false alarm, not a detection of degradation,
    and is accounted for by the benign FPR calibration rather than being collapsed
    to ``TTD = 0``. ``min_epoch`` must match the epoch range over which the
    ``threshold`` was calibrated, so the reported FPR operating point holds over
    exactly the windows scanned here. Returns ``e - onset`` (>= 0), or ``None`` if
    never flagged (censored).
    """
    n_epochs = len(stream.epochs)
    start = max(min_epoch, stream.onset_epoch, 1)
    for e in range(start, n_epochs):
        win = stream.window(e - window + 1, e + 1)
        if not win:
            continue
        if detector(win) > threshold:
            return e - stream.onset_epoch
    return None


# ----------------------------------------------------------------------
# Proxy calibration: soft probability vs hard-thresholded prediction
# ----------------------------------------------------------------------
@dataclass
class CalibrationResult:
    soft_brier: Optional[float]
    binary_brier: Optional[float]
    soft_ece: Optional[float]
    binary_ece: Optional[float]


def calibration(
    interactions: Sequence[SoftInteraction], tau_star: float = 0.5, bins: int = 10
) -> CalibrationResult:
    """Brier score and ECE for the soft proxy vs its hard-thresholded twin.

    The soft prediction is ``p``; the binary prediction is ``1{p >= tau_star}``.
    Both are scored against the same ground-truth outcomes, quantifying how much
    predictive resolution thresholding discards.
    """
    metrics = SoftMetrics()
    inter = list(interactions)
    hard = binarize_stream(inter, tau_star)
    return CalibrationResult(
        soft_brier=metrics.brier_score(inter),
        binary_brier=metrics.brier_score(hard),
        soft_ece=metrics.expected_calibration_error(inter, bins=bins),
        binary_ece=metrics.expected_calibration_error(hard, bins=bins),
    )


def per_agent_scores(
    streams: Sequence[AgentStream],
    detector: Detector,
    eval_start: int,
    eval_end: int,
) -> tuple[List[float], List[int]]:
    """Score every agent over the evaluation window; return (scores, labels)."""
    scores = [detector(s.window(eval_start, eval_end)) for s in streams]
    labels = [s.label for s in streams]
    return scores, labels
