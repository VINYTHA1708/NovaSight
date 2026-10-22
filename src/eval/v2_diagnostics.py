"""Stage V2 diagnostics on selection pools only (Protocol v1.1 Task E10, D.5).

Computes on sel_cal / sel_eval features:
  - Per-class score-scale ratios with bootstrap confidence intervals.
  - Effective dimension: tr(Sigma)^2 / tr(Sigma^2).
  - Skewness and kurtosis diagnostics.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_effective_dimension(cov_matrix: np.ndarray) -> float:
    """Compute effective dimension: tr(Sigma)^2 / tr(Sigma^2)."""
    tr1 = float(np.trace(cov_matrix))
    tr2 = float(np.trace(cov_matrix @ cov_matrix))
    if tr2 <= 0:
        return 0.0
    return (tr1 ** 2) / tr2


def compute_score_scale_ratios(
    class_features: dict[str, np.ndarray],
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute per-class score scale ratios with bootstrap intervals on sel data only."""
    rng = np.random.default_rng(seed)
    results = {}

    all_scales = {}
    for c, feats in class_features.items():
        mu = feats.mean(axis=0)
        dists = np.linalg.norm(feats - mu, axis=1)
        all_scales[c] = float(np.mean(dists))

    macro_scale = float(np.mean(list(all_scales.values())))

    for c, feats in class_features.items():
        n = len(feats)
        boot_ratios = []
        for _ in range(bootstrap_samples):
            idx = rng.choice(n, size=n, replace=True)
            sample = feats[idx]
            mu_s = sample.mean(axis=0)
            dists_s = np.linalg.norm(sample - mu_s, axis=1)
            scale_s = float(np.mean(dists_s))
            boot_ratios.append(scale_s / max(macro_scale, 1e-6))

        point_ratio = all_scales[c] / max(macro_scale, 1e-6)
        ci_lo = float(np.percentile(boot_ratios, 2.5))
        ci_hi = float(np.percentile(boot_ratios, 97.5))

        results[c] = {
            "scale": all_scales[c],
            "scale_ratio": point_ratio,
            "bootstrap_95_ci": [ci_lo, ci_hi],
        }

    return results
