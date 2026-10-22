"""Numpy/float64 scoring engine for inflated Mahalanobis and Mondrian conformal prediction.

Reproduces gaussian.GaussianModel, conformal.ConformalScores, and make_decision
identically in float64 without requiring torch, torchvision, or scikit-learn.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.runtime.artifact import RuntimeArtifact


def make_decision(
    pred_set: set[str],
    c_hat: str,
    enrolled_classes: Sequence[str] = (),
) -> dict[str, Any]:
    """Translate prediction set and closed-set argmin into a decision verdict.

    Args:
        pred_set: Set of class names in the conformal prediction set.
        c_hat: Closed-set argmin predicted label.
        enrolled_classes: Optional sequence of enrolled class names.

    Returns:
        Decision dictionary with keys: verdict, class_name, ambiguous_classes,
        closed_set_label, message.
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


class RuntimeScorer:
    """Computes inflated Mahalanobis distances and conformal prediction sets in float64."""

    def __init__(self, artifact: RuntimeArtifact) -> None:
        """Initialize with verified RuntimeArtifact."""
        self.artifact = artifact
        self.class_names = list(artifact.class_names)
        self.A = artifact.A.astype(np.float64)
        self.class_means = [m.astype(np.float64) for m in artifact.class_means]
        self.n_c = list(artifact.n_c)
        self.n_sub = [list(ns) for ns in artifact.n_sub]
        self.class_scores = {k: v.astype(np.float64) for k, v in artifact.class_scores.items()}
        self.S_pool = artifact.S_pool.astype(np.float64)
        self.alpha_default = artifact.alpha_default

    def inflated_score(self, f: np.ndarray, c_idx: int) -> float:
        """Compute inflated Mahalanobis score dt_c(f) for known class c_idx.

        dt_c(f) = min_j (f - mu_{c,j})^T A (f - mu_{c,j}) / (1 + 1 / n_{c,j})

        Args:
            f: Feature vector of shape (D,).
            c_idx: Known class index.

        Returns:
            Scalar inflated Mahalanobis distance.
        """
        f64 = np.asarray(f, dtype=np.float64).ravel()
        means = self.class_means[c_idx]
        scores = []
        for j, mu in enumerate(means):
            n_sub_j = self.n_sub[c_idx][j]
            diff = f64 - mu
            # Quadratic form diff^T A diff
            d = float(diff @ self.A @ diff)
            dt = d / (1.0 + 1.0 / n_sub_j)
            scores.append(dt)
        return float(min(scores))

    def per_class_scores(self, f: np.ndarray) -> dict[str, float]:
        """Compute inflated scores for all known classes.

        Args:
            f: Feature vector of shape (D,).

        Returns:
            Dict of class_name -> inflated score dt_c(f).
        """
        return {c: self.inflated_score(f, i) for i, c in enumerate(self.class_names)}

    def novelty_score(self, f: np.ndarray) -> float:
        """Primary novelty score N(f) = min_c dt_c(f) over known classes."""
        return float(min(self.inflated_score(f, i) for i in range(len(self.class_names))))

    def closed_set_label(self, f: np.ndarray) -> tuple[str, float]:
        """Return the closed-set label and its inflated score."""
        scores = self.per_class_scores(f)
        best_class = min(scores.keys(), key=lambda c: scores[c])
        return best_class, float(scores[best_class])

    def conformal_p_value(self, score: float, class_name: str) -> float:
        """Compute class-conditional conformal p-value.

        p_c = (1 + #{s in S_c : s >= score}) / (|S_c| + 1)
        """
        S_c = self.class_scores[class_name]
        count = int(np.sum(S_c >= score))
        return float((1.0 + count) / (len(S_c) + 1.0))

    def p_value_from_pool(self, score: float) -> float:
        """Compute conformal p-value against pooled calibration scores S_pool.

        p_pool = (1 + #{s in S_pool : s >= score}) / (|S_pool| + 1)
        """
        count = int(np.sum(self.S_pool >= score))
        return float((1.0 + count) / (len(self.S_pool) + 1.0))

    def all_p_values(self, scores: dict[str, float]) -> dict[str, float]:
        """Compute Mondrian p-values for a dict of scores."""
        return {c: self.conformal_p_value(scores[c], c) for c in scores}

    def predict(
        self,
        f: np.ndarray,
        alpha: float | None = None,
        enrolled_classes: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Full prediction pipeline: scores, p-values, prediction set, verdict.

        Args:
            f: Extracted feature vector (D,).
            alpha: Conformal significance level (defaults to artifact alpha).
            enrolled_classes: Optional dict of newly enrolled classes:
                name -> {"mean": (D,), "n": int}

        Returns:
            Dict containing:
                verdict: "known" | "ambiguous" | "unknown"
                class_name: predicted class or None
                prediction_set: list of classes in prediction set
                closed_set_label: closest class
                novelty_score: min known distance
                per_class_dt: dict of class -> inflated distance
                p_values: dict of class -> conformal p-value
                alpha: significance level used
                decision: full decision dict
        """
        f64 = np.asarray(f, dtype=np.float64).ravel()
        if alpha is None:
            alpha = self.alpha_default

        # 1. Known classes
        dt_scores = self.per_class_scores(f64)
        c_hat, _ = self.closed_set_label(f64)
        p_vals = self.all_p_values(dt_scores)

        # 2. Add enrolled classes if present
        if enrolled_classes:
            for enc_name, enc_info in enrolled_classes.items():
                mean_vec = np.asarray(enc_info["mean"], dtype=np.float64)
                n_enc = int(enc_info.get("n", 5))
                diff = f64 - mean_vec
                d = float(diff @ self.A @ diff)
                dt_enc = d / (1.0 + 1.0 / n_enc)
                dt_scores[enc_name] = dt_enc
                # Enrolled class uses pooled calibration p-value
                p_vals[enc_name] = self.p_value_from_pool(dt_enc)

        # 3. Prediction set: {c : p_c > alpha}
        pred_set = {c for c, p in p_vals.items() if p > alpha}

        # 4. Final decision
        enrolled_names = list(enrolled_classes.keys()) if enrolled_classes else []
        decision = make_decision(pred_set, c_hat, enrolled_names)

        novelty = float(min(self.inflated_score(f64, i) for i in range(len(self.class_names))))

        return {
            "verdict": decision["verdict"],
            "class_name": decision["class_name"],
            "prediction_set": sorted(pred_set),
            "closed_set_label": c_hat,
            "novelty_score": novelty,
            "per_class_dt": dt_scores,
            "p_values": p_vals,
            "alpha": float(alpha),
            "decision": decision,
        }
