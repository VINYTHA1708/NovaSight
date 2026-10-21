"""NovaSight baseline open-set detectors (Part 6.2).

Baselines (all use the same splits; thresholds set on CAL at 1-alpha known acceptance):
  - closed_set: no rejection, always predicts argmax
  - msp: max softmax probability (from B1 head or linear probe)
  - max_logit: max raw logit
  - energy: log-sum-exp of logits (Liu et al. 2020)
  - cosine_centroid: per-class cosine sim, threshold at quantile
  - knn: k-nearest-neighbour (k=5) distance to training set
  - mahalanobis: Mahalanobis without conformal (uses same Gaussian)
  - novasight: our method (min inflated Mahalanobis + conformal)

Linear probe [APPROVED-POLICY]: multinomial logistic regression on
  standardised frozen features, C=1.0, fixed solver/seed, no tuning
  beyond train pool.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import get_config

logger = logging.getLogger(__name__)


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
) -> LogisticRegression:
    """Train a linear probe on training features.

    [APPROVED-POLICY]: C=1.0, solver=lbfgs, multi_class=multinomial, seed=42.
    Training pool only; never tuned on any other pool.

    Args:
        train_features: (N, D) feature array (standardised).
        train_labels: (N,) integer label array.

    Returns:
        Fitted LogisticRegression.
    """
    cfg = get_config()
    lp = cfg["approved_policy"]["linear_probe"]
    probe = LogisticRegression(
        C=lp["C"],
        solver=lp["solver"],
        multi_class=lp["multi_class"],
        max_iter=1000,
        random_state=lp["seed"],
    )
    probe.fit(train_features, train_labels)
    logger.info(
        "Linear probe trained: n=%d, C=%.1f, solver=%s, seed=%d",
        len(train_features), lp["C"], lp["solver"], lp["seed"],
    )
    return probe


@dataclass
class BaselineScores:
    """Scores for a single baseline method.

    Attributes:
        method: Baseline name.
        scores: (N,) score array (higher = more known / less novel).
        labels: (N,) true label array.
        is_known: (N,) boolean array (True if known class).
        threshold: Threshold set on CAL (at 1-alpha known acceptance).
        predictions: (N,) predicted label array (int).
    """

    method: str
    scores: np.ndarray
    labels: np.ndarray
    is_known: np.ndarray
    threshold: float
    predictions: np.ndarray


class ClosedSetBaseline:
    """Always predicts the class with highest score; never rejects."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    def score(self, logits: np.ndarray) -> np.ndarray:
        """Return max logit as a score (higher = more confident)."""
        return logits.max(axis=1)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return logits.argmax(axis=1)


class MSPBaseline:
    """Maximum softmax probability baseline."""

    def score(self, logits: np.ndarray) -> np.ndarray:
        """Return max softmax probability."""
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        return probs.max(axis=1)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return logits.argmax(axis=1)


class MaxLogitBaseline:
    """Max-logit baseline."""

    def score(self, logits: np.ndarray) -> np.ndarray:
        return logits.max(axis=1)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return logits.argmax(axis=1)


class EnergyBaseline:
    """Energy score baseline (Liu et al. 2020)."""

    def score(self, logits: np.ndarray) -> np.ndarray:
        """Return log-sum-exp (higher = more known)."""
        return np.log(np.exp(logits).sum(axis=1))

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return logits.argmax(axis=1)


class CosineCentroidBaseline:
    """Per-class cosine similarity, threshold at per-class quantile."""

    def __init__(self, class_centroids: np.ndarray) -> None:
        """Args:
            class_centroids: (K, D) array of class centroids.
        """
        norms = np.linalg.norm(class_centroids, axis=1, keepdims=True)
        self.centroids = class_centroids / np.where(norms == 0, 1, norms)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return max cosine similarity across classes."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        f_norm = features / np.where(norms == 0, 1, norms)
        sims = f_norm @ self.centroids.T   # (N, K)
        return sims.max(axis=1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        f_norm = features / np.where(norms == 0, 1, norms)
        sims = f_norm @ self.centroids.T
        return sims.argmax(axis=1)


class KNNBaseline:
    """k-nearest-neighbour distance baseline (k=5)."""

    def __init__(self, train_features: np.ndarray, k: int = 5) -> None:
        """Args:
            train_features: (N, D) training feature array.
            k: Number of nearest neighbours.
        """
        self.train_features = train_features
        self.k = k

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return negative mean k-NN distance (higher = more known)."""
        N = len(features)
        scores = np.zeros(N)
        for i in range(N):
            diffs = self.train_features - features[i]
            dists = np.sum(diffs ** 2, axis=1)
            knn_dists = np.sort(dists)[:self.k]
            scores[i] = -knn_dists.mean()  # negated so higher = more known
        return scores


class MahalanobisBaseline:
    """Mahalanobis distance without conformal (uses same GaussianModel)."""

    def __init__(self, gaussian_model: Any) -> None:
        """Args:
            gaussian_model: Fitted GaussianModel instance.
        """
        self.model = gaussian_model

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return negative novelty score (higher = more known)."""
        return -self.model.batch_novelty_scores(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        preds = []
        for i in range(len(features)):
            cls, _ = self.model.closed_set_label(features[i])
            preds.append(self.model.class_names.index(cls))
        return np.array(preds)


def set_threshold_on_cal(
    scores: np.ndarray,
    is_known: np.ndarray,
    alpha: float,
) -> float:
    """Set a rejection threshold on CAL at (1 - alpha) known acceptance.

    Threshold = alpha-quantile of known-class cal scores.
    Known images below threshold would be rejected.

    Args:
        scores: (N,) score array.
        is_known: (N,) boolean mask for known-class images.
        alpha: Significance level.

    Returns:
        Threshold value.
    """
    known_scores = scores[is_known]
    # Reject below threshold -> set threshold at alpha-quantile
    threshold = float(np.quantile(known_scores, alpha))
    return threshold
