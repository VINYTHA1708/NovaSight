"""NovaSight class-conditional Gaussian model with shared covariance (Part 4.4).

Model:
  f | y=c ~ N(mu_c, Sigma)  [WORKING ASSUMPTION; not a fact about NEU]
  Sigma estimated via Ledoit-Wolf shrinkage on residuals.

Scores:
  d_c(f) = (f - mu_c)^T A (f - mu_c)           [Mahalanobis]
  dt_c(f) = d_c(f) / (1 + 1/n_c)               [inflated]
  N(f) = min_c dt_c(f)                          [novelty score, APPROVED-POLICY]
  c_hat = argmin_c dt_c(f)                      [closed-set prediction]

Mixture variant (APPROVED-POLICY):
  k-means sub-means per class (k in {1,2,3}), sharing A.
  class score = min over sub-means, each inflated by (1 + 1/n_sub).
  k=1 is the plain Gaussian.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class GaussianModel:
    """Fitted class-conditional Gaussian model.

    Attributes:
        class_names: Ordered list of known class names.
        class_means: List of sub-mean arrays per class (each shape (k_c, D)).
        n_c: Number of training examples per class.
        n_sub: Number of training examples per sub-mean.
        A: Precision matrix (D, D).
        condition_number: Condition number of Sigma_hat (for diagnostics).
        shrinkage_intensity: Ledoit-Wolf shrinkage parameter.
        k: Mixture components per class (1 = plain Gaussian).
        feature_dim: Feature dimensionality D.
        train_feature_mean: Per-block means for z-scoring (L234 only).
        train_feature_std: Per-block stds for z-scoring (L234 only).
    """

    class_names: list[str]
    class_means: list[np.ndarray]       # len=K; each (k_c, D)
    n_c: list[int]                      # per-class train count
    n_sub: list[list[int]]              # per-class, per-sub-mean count
    A: np.ndarray                       # (D, D) precision matrix
    condition_number: float
    shrinkage_intensity: float
    k: int                              # mixture components
    feature_dim: int
    train_feature_mean: np.ndarray | None = None  # (D,) for z-scoring
    train_feature_std: np.ndarray | None = None   # (D,)

    def inflated_score(self, f: np.ndarray, c_idx: int) -> float:
        """Compute inflated Mahalanobis score for class c_idx.

        Score = min over sub-means of dt_sub.
        dt_sub = d_sub / (1 + 1/n_sub).

        Args:
            f: Feature vector (D,).
            c_idx: Class index.

        Returns:
            Minimum inflated Mahalanobis score across sub-means.
        """
        scores = []
        means = self.class_means[c_idx]   # (k_c, D)
        for j, mu in enumerate(means):
            n_sub_j = self.n_sub[c_idx][j]
            diff = f - mu
            d = float(diff @ self.A @ diff)
            dt = d / (1.0 + 1.0 / n_sub_j)
            scores.append(dt)
        return min(scores)

    def novelty_score(self, f: np.ndarray) -> float:
        """Compute the primary novelty score N(f) = min_c dt_c(f).

        Args:
            f: Feature vector (D,).

        Returns:
            Minimum inflated Mahalanobis score across all known classes.
        """
        return min(self.inflated_score(f, c) for c in range(len(self.class_names)))

    def closed_set_label(self, f: np.ndarray) -> tuple[str, float]:
        """Return the closed-set label and its score.

        Args:
            f: Feature vector (D,).

        Returns:
            Tuple of (class_name, inflated_score).
        """
        scores = [self.inflated_score(f, c) for c in range(len(self.class_names))]
        best_idx = int(np.argmin(scores))
        return self.class_names[best_idx], scores[best_idx]

    def all_inflated_scores(self, f: np.ndarray) -> np.ndarray:
        """Return inflated scores for all known classes.

        Args:
            f: Feature vector (D,).

        Returns:
            Array of shape (K,) with inflated scores.
        """
        return np.array([self.inflated_score(f, c) for c in range(len(self.class_names))])

    def batch_novelty_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute novelty scores for a batch.

        Args:
            features: (N, D) feature matrix.

        Returns:
            (N,) array of novelty scores.
        """
        N = features.shape[0]
        scores = np.zeros(N)
        for i in range(N):
            scores[i] = self.novelty_score(features[i])
        return scores

    def batch_all_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute all class inflated scores for a batch.

        Args:
            features: (N, D) feature matrix.

        Returns:
            (N, K) array of inflated scores.
        """
        K = len(self.class_names)
        N = features.shape[0]
        result = np.zeros((N, K))
        for i in range(N):
            result[i] = self.all_inflated_scores(features[i])
        return result


def fit_gaussian(
    train_features: dict[str, np.ndarray],
    k: int = 1,
    kmeans_seed: int = 42,
) -> GaussianModel:
    """Fit the class-conditional Gaussian model on training features.

    Args:
        train_features: Dict of class_name -> (n_c, D) feature array.
        k: Number of mixture components per class (1 = plain Gaussian).
        kmeans_seed: Seed for k-means sub-mean estimation.

    Returns:
        Fitted GaussianModel.

    Raises:
        ValueError: If Sigma_hat is not positive definite.
    """
    class_names = sorted(train_features.keys())
    K = len(class_names)

    # Stack all features to get residuals
    all_features = np.vstack([train_features[c] for c in class_names])
    all_labels = np.concatenate([
        np.full(len(train_features[c]), i)
        for i, c in enumerate(class_names)
    ])
    D = all_features.shape[1]

    # Compute class means
    class_means_raw: list[np.ndarray] = []
    for c in class_names:
        class_means_raw.append(np.mean(train_features[c], axis=0))

    # Compute residuals
    residuals = all_features - np.array(class_means_raw)[all_labels]

    # Ledoit-Wolf shrinkage on residuals
    lw = LedoitWolf(assume_centered=True)
    lw.fit(residuals)
    Sigma_hat = lw.covariance_
    shrinkage = float(lw.shrinkage_)
    logger.info("Ledoit-Wolf shrinkage intensity: %.4f", shrinkage)

    # Assert positive definiteness
    eigenvalues = np.linalg.eigvalsh(Sigma_hat)
    min_eig = float(eigenvalues.min())
    if min_eig <= 0:
        raise ValueError(
            f"Sigma_hat is not positive definite (min eigenvalue = {min_eig:.6e}). "
            "Increase shrinkage or check feature dimensionality."
        )

    # Condition number
    cond = float(eigenvalues.max() / eigenvalues.min())
    logger.info("Sigma_hat condition number: %.2e", cond)

    # Precision matrix
    A = np.linalg.inv(Sigma_hat)

    # Mixture sub-means via k-means (APPROVED-POLICY: seeded)
    class_means_list: list[np.ndarray] = []
    n_sub_list: list[list[int]] = []
    n_c_list: list[int] = []

    for c in class_names:
        X_c = train_features[c]   # (n_c, D)
        n_c = len(X_c)
        n_c_list.append(n_c)

        if k == 1:
            class_means_list.append(np.mean(X_c, axis=0).reshape(1, -1))
            n_sub_list.append([n_c])
        else:
            k_actual = min(k, n_c)  # can't have more centroids than samples
            km = KMeans(n_clusters=k_actual, random_state=kmeans_seed, n_init=10)
            km.fit(X_c)
            labels = km.labels_
            centroids = np.zeros((k_actual, D))
            n_subs = []
            for j in range(k_actual):
                mask = labels == j
                n_j = int(mask.sum())
                centroids[j] = X_c[mask].mean(axis=0) if n_j > 0 else km.cluster_centers_[j]
                n_subs.append(max(n_j, 1))
            class_means_list.append(centroids)
            n_sub_list.append(n_subs)

    feature_dim = D
    model = GaussianModel(
        class_names=class_names,
        class_means=class_means_list,
        n_c=n_c_list,
        n_sub=n_sub_list,
        A=A,
        condition_number=cond,
        shrinkage_intensity=shrinkage,
        k=k,
        feature_dim=feature_dim,
    )
    logger.info(
        "GaussianModel fitted: K=%d classes, D=%d dims, k=%d mixtures",
        K, D, k
    )
    return model


def apply_zscore(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply z-scoring to features (for L234 multi-block features).

    Args:
        features: (N, D) feature matrix.
        mean: (D,) mean vector (from TRAIN only).
        std: (D,) std vector (from TRAIN only). Zeros replaced with 1.

    Returns:
        Z-scored feature matrix (N, D).
    """
    safe_std = np.where(std == 0, 1.0, std)
    return (features - mean) / safe_std


def compute_zscore_stats(
    train_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for z-scoring from training features.

    Args:
        train_features: (N, D) array.

    Returns:
        Tuple of (mean, std) arrays of shape (D,).
    """
    return train_features.mean(axis=0), train_features.std(axis=0)
