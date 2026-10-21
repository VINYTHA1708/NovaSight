"""NovaSight explainability module (Part 4.9, optional GradCAM).

Produces:
  1. Nearest-neighbour explanation: top-3 training images closest to query
     in embedding space per known class, for embedding display in the UI.
  2. Score breakdown: per-class inflated Mahalanobis scores with bar chart.

GradCAM is optionally available for B1 (ResNetWithHead) using
torchvision's gradient utilities.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def nearest_neighbours(
    query_feat: np.ndarray,
    train_feats: dict[str, np.ndarray],
    top_k: int = 3,
) -> dict[str, list[int]]:
    """Find nearest training neighbours per class.

    Args:
        query_feat: (D,) query feature vector.
        train_feats: Dict of class_name -> (n, D) training features.
        top_k: Number of neighbours to return per class.

    Returns:
        Dict of class_name -> list of top_k indices (in class array).
    """
    result: dict[str, list[int]] = {}
    for cname, feats in train_feats.items():
        diffs = feats - query_feat    # (n, D)
        dists = np.sum(diffs ** 2, axis=1)
        knn = int(min(top_k, len(feats)))
        indices = list(np.argsort(dists)[:knn])
        result[cname] = indices
    return result


def score_breakdown(
    query_feat: np.ndarray,
    gaussian,
    conformal=None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute per-class score breakdown for display.

    Args:
        query_feat: (D,) feature vector.
        gaussian: GaussianModel.
        conformal: Optional ConformalScores for p-values.
        alpha: Significance level for prediction set.

    Returns:
        Dict with per-class scores, prediction set, and novelty score.
    """
    K = len(gaussian.class_names)
    scores = {}
    for c_idx, cname in enumerate(gaussian.class_names):
        dt = gaussian.inflated_score(query_feat, c_idx)
        scores[cname] = dt

    novelty = min(scores.values())

    p_vals: dict[str, float] | None = None
    pred_set: list[str] | None = None
    if conformal is not None:
        p_vals = conformal.all_p_values(scores)
        pred_set = [c for c, p in p_vals.items() if p > alpha]

    return {
        "per_class_dt": scores,
        "novelty_score": novelty,
        "p_values": p_vals,
        "prediction_set": pred_set,
        "alpha": alpha,
    }
