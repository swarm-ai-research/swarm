"""Simple logistic forecaster for incoherence risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))  # type: ignore[no-any-return]


def _auc_roc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute AUC-ROC from scores; returns 0.5 if undefined."""
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have equal length")
    positives = [
        score for label, score in zip(y_true, y_score, strict=True) if label == 1
    ]
    negatives = [
        score for label, score in zip(y_true, y_score, strict=True) if label == 0
    ]
    if not positives or not negatives:
        return 0.5

    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total > 0 else 0.5


def _expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    bins: int = 10,
) -> float:
    """Compute ECE with fixed probability bins."""
    y_true_arr = np.array(y_true, dtype=float)
    y_prob_arr = np.array(y_prob, dtype=float)
    if len(y_true_arr) == 0:
        return 0.0

    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        mask = (y_prob_arr >= left) & (
            y_prob_arr < right if idx < bins - 1 else y_prob_arr <= right
        )
        if not np.any(mask):
            continue
        acc = np.mean(y_true_arr[mask])
        conf = np.mean(y_prob_arr[mask])
        weight = np.mean(mask.astype(float))
        ece += abs(acc - conf) * weight
    return float(ece)


@dataclass
class IncoherenceForecaster:
    """Lightweight logistic-regression forecaster with numpy training."""

    learning_rate: float = 0.05
    n_iters: int = 500
    l2: float = 0.0

    def __post_init__(self) -> None:
        self.feature_names: List[str] = []
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0

    def fit(
        self,
        feature_rows: Sequence[Mapping[str, float]],
        labels: Sequence[int],
    ) -> "IncoherenceForecaster":
        """Train logistic model on feature rows.

        Extracts union of all feature names across rows. Missing keys in individual rows
        are filled with 0.0 during vectorization.
        """
        if len(feature_rows) != len(labels):
            raise ValueError("feature_rows and labels must have equal length")
        if not feature_rows:
            raise ValueError("feature_rows must be non-empty")

        self.feature_names = sorted({key for row in feature_rows for key in row.keys()})
        x = self._rows_to_matrix(feature_rows)
        y = np.array(labels, dtype=float)
        n_samples, n_features = x.shape

        self._weights = np.zeros(n_features, dtype=float)
        self._bias = 0.0

        for _ in range(self.n_iters):
            logits = x @ self._weights + self._bias
            probs = _sigmoid(logits)

            error = probs - y
            grad_w = (x.T @ error) / n_samples
            grad_b = float(np.mean(error))
            if self.l2 > 0:
                grad_w += self.l2 * self._weights

            self._weights -= self.learning_rate * grad_w
            self._bias -= self.learning_rate * grad_b

        return self

    def predict_proba(self, feature_row: Mapping[str, float]) -> float:
        """Predict probability of high incoherence for one feature row."""
        self._ensure_fitted()
        x = self._row_to_vector(feature_row)
        logit = float(x @ self._weights + self._bias)  # type: ignore[operator]
        return float(_sigmoid(np.array([logit]))[0])

    def predict(self, feature_row: Mapping[str, float], threshold: float = 0.5) -> int:
        """Predict binary high-incoherence label."""
        return int(self.predict_proba(feature_row) >= threshold)

    def evaluate(
        self,
        feature_rows: Sequence[Mapping[str, float]],
        labels: Sequence[int],
    ) -> Dict[str, float]:
        """Compute holdout metrics: AUC, Brier score, expected calibration error, and mean predicted risk."""
        probs = [self.predict_proba(row) for row in feature_rows]
        y_true = [int(label) for label in labels]
        brier = float(np.mean((np.array(probs) - np.array(y_true, dtype=float)) ** 2))
        ece = _expected_calibration_error(y_true, probs)
        return {
            "auc": _auc_roc(y_true, probs),
            "brier_score": brier,
            "expected_calibration_error": ece,
            "mean_predicted_risk": float(np.mean(probs)) if probs else 0.0,
        }

    def fit_and_evaluate(
        self,
        train_features: Sequence[Mapping[str, float]],
        train_labels: Sequence[int],
        test_features: Sequence[Mapping[str, float]],
        test_labels: Sequence[int],
    ) -> Dict[str, float]:
        """Train on one split and evaluate on held-out split."""
        self.fit(train_features, train_labels)
        return self.evaluate(test_features, test_labels)

    def _rows_to_matrix(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        return np.array([self._row_to_vector(row) for row in rows], dtype=float)

    def _row_to_vector(self, row: Mapping[str, float]) -> np.ndarray:
        return np.array(
            [float(row.get(name, 0.0)) for name in self.feature_names], dtype=float
        )

    def _ensure_fitted(self) -> None:
        if self._weights is None or not self.feature_names:
            raise ValueError("Forecaster must be fit before prediction")
