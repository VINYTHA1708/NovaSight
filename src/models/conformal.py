"""NovaSight conformal decision layer (Part 4.5).

Mondrian (class-conditional) conformal prediction:
  S_c = {dt_c(x_i) : x_i in CAL, y_i = c}   (40 per class)
  p_c(x) = (1 + #{s in S_c : s >= dt_c(x)}) / (|S_c| + 1)
  Gamma_alpha(x) = {c : p_c(x) > alpha}

Output rules:
  empty set  -> "unknown, send to operator"
  {c}        -> class c
  2+ classes -> "ambiguous between ..."

INVARIANT: adding/removing an enrolled class never changes known class
p-values (bit-identical). Unit-tested.

S_pool = union of all S_c;  tau_hat = mean(S_pool)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.gaussian import GaussianModel

logger = logging.getLogger(__name__)


@dataclass
class ConformalScores:
    """Calibration scores for Mondrian conformal prediction.

    Attributes:
        class_scores: Dict of class_name -> sorted array of cal inflated scores.
        S_pool: Concatenation of all class scores.
        tau_hat: Mean of S_pool.
        alpha_default: Default significance level.
    """

    class_scores: dict[str, np.ndarray]   # class_name -> (n_cal,) sorted descending
    S_pool: np.ndarray                    # (K * n_cal,)
    tau_hat: float
    alpha_default: float = 0.05
    alpha_grid: list[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.20]
    )

    def p_value(self, score: float, class_name: str) -> float:
        """Compute the conformal p-value for a given inflated score.

        p_c(x) = (1 + #{s in S_c : s >= dt_c(x)}) / (|S_c| + 1)

        Args:
            score: Inflated Mahalanobis score dt_c(x).
            class_name: Class name.

        Returns:
            Conformal p-value in (0, 1].
        """
        S_c = self.class_scores[class_name]
        n = len(S_c)
        count = int(np.sum(S_c >= score))
        return (1 + count) / (n + 1)

    def all_p_values(self, scores: dict[str, float]) -> dict[str, float]:
        """Compute p-values for all known classes.

        Args:
            scores: Dict of class_name -> inflated score.

        Returns:
            Dict of class_name -> p-value.
        """
        return {c: self.p_value(scores[c], c) for c in scores}

    def prediction_set(
        self,
        scores: dict[str, float],
        alpha: float | None = None,
    ) -> tuple[set[str], dict[str, float]]:
        """Return the conformal prediction set and p-values.

        Args:
            scores: Dict of class_name -> inflated score (known classes only).
            alpha: Significance level. Defaults to self.alpha_default.

        Returns:
            Tuple of (prediction_set, p_values_dict).
        """
        if alpha is None:
            alpha = self.alpha_default
        p_vals = self.all_p_values(scores)
        pred_set = {c for c, p in p_vals.items() if p > alpha}
        return pred_set, p_vals

    def p_value_from_pool(self, score: float) -> float:
        """Compute a p-value against the pooled calibration scores S_pool.

        Used for enrollment consistency and overlap checks.

        Args:
            score: A score to compare against S_pool.

        Returns:
            p-value = (1 + #{s in S_pool : s >= score}) / (|S_pool| + 1).
        """
        n = len(self.S_pool)
        count = int(np.sum(self.S_pool >= score))
        return (1 + count) / (n + 1)


def build_conformal_scores(
    gaussian: GaussianModel,
    cal_features: dict[str, np.ndarray],
    alpha_default: float = 0.05,
    alpha_grid: list[float] | None = None,
) -> ConformalScores:
    """Compute class-conditional conformal calibration scores.

    Uses ONLY the cal pool (never train or test).

    Args:
        gaussian: Fitted GaussianModel.
        cal_features: Dict of class_name -> (n_cal, D) feature array.
        alpha_default: Default significance level.
        alpha_grid: Full grid of alpha values.

    Returns:
        ConformalScores object.
    """
    if alpha_grid is None:
        alpha_grid = [0.01, 0.02, 0.05, 0.10, 0.20]

    class_scores: dict[str, np.ndarray] = {}
    for c_idx, class_name in enumerate(gaussian.class_names):
        if class_name not in cal_features:
            raise ValueError(f"No cal features for class: {class_name}")
        feats = cal_features[class_name]   # (n_cal, D)
        scores = np.array([
            gaussian.inflated_score(feats[i], c_idx)
            for i in range(len(feats))
        ])
        class_scores[class_name] = scores
        logger.debug("Cal scores for %s: n=%d, min=%.2f, max=%.2f",
                     class_name, len(scores), scores.min(), scores.max())

    S_pool = np.concatenate(list(class_scores.values()))
    tau_hat = float(S_pool.mean())
    logger.info(
        "Conformal calibration: K=%d classes, |S_pool|=%d, tau_hat=%.4f",
        len(class_scores), len(S_pool), tau_hat,
    )

    return ConformalScores(
        class_scores=class_scores,
        S_pool=S_pool,
        tau_hat=tau_hat,
        alpha_default=alpha_default,
        alpha_grid=alpha_grid,
    )


def make_decision(
    pred_set: set[str],
    c_hat: str,
    enrolled_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Apply the output rule to a prediction set.

    Rules:
      empty  -> "unknown"
      {c}    -> class c
      {2+}   -> "ambiguous"

    Args:
        pred_set: Conformal prediction set.
        c_hat: Closed-set label (argmin inflated score).
        enrolled_classes: Names of enrolled classes to include in display.

    Returns:
        Decision dict with keys: verdict, class_name, ambiguous_classes.
    """
    if len(pred_set) == 0:
        return {
            "verdict": "unknown",
            "class_name": None,
            "ambiguous_classes": [],
            "closed_set_label": c_hat,
            "message": "Unknown defect — send to operator",
        }
    elif len(pred_set) == 1:
        cls = next(iter(pred_set))
        return {
            "verdict": "known",
            "class_name": cls,
            "ambiguous_classes": [],
            "closed_set_label": c_hat,
            "message": cls,
        }
    else:
        classes = sorted(pred_set)
        return {
            "verdict": "ambiguous",
            "class_name": None,
            "ambiguous_classes": classes,
            "closed_set_label": c_hat,
            "message": f"Ambiguous between: {', '.join(classes)}",
        }
