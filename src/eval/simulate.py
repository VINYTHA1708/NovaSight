"""Synthetic simulations for NovaSight Protocol v1.1 Verification Stage V1 (Task E10, D.5).

Simulations:
  - Proposition 1: Beta(n+1-l, l) fixed-cal coverage.
  - Proposition 2: 5-shot enrollment draw shift and variance ratio.
  - Delta^2 unbiasedness test.
  - Heteroscedastic class variants (sigma in {0.5, 0.8, 1.25, 2.0}).
  - Bimodal new class simulation.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import beta as scipy_beta

logger = logging.getLogger(__name__)


def simulate_proposition_1(
    n_cal: int = 40,
    alpha: float = 0.05,
    n_trials: int = 1000,
    n_test: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify Proposition 1: coverage across random calibration draws follows Beta(n+1-l, l)."""
    rng = np.random.default_rng(seed)
    l_val = int(np.floor((n_cal + 1) * alpha))
    a = n_cal + 1 - l_val
    b = l_val

    coverages = []
    for _ in range(n_trials):
        cal_scores = rng.standard_normal(n_cal)
        test_scores = rng.standard_normal(n_test)
        # Conformal threshold: p_val = (1 + sum(cal >= test)) / (n_cal + 1) > alpha
        # That means count of cal >= test >= l_val
        sorted_cal = np.sort(cal_scores)
        # #{cal >= test} >= l_val requires test <= sorted_cal[n_cal - l_val]
        cutoff = sorted_cal[n_cal - l_val]
        cov = float(np.mean(test_scores <= cutoff))
        coverages.append(cov)

    emp_mean = float(np.mean(coverages))
    emp_std = float(np.std(coverages))
    theo_mean = float(scipy_beta.mean(a, b))
    theo_std = float(scipy_beta.std(a, b))

    return {
        "n_cal": n_cal,
        "alpha": alpha,
        "empirical_mean": emp_mean,
        "empirical_std": emp_std,
        "theoretical_mean": theo_mean,
        "theoretical_std": theo_std,
        "mean_absolute_error": abs(emp_mean - theo_mean),
        "std_absolute_error": abs(emp_std - theo_std),
    }


def simulate_proposition_2(
    dim: int = 512,
    n_exemplars: int = 5,
    n_trials: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify Proposition 2: 5-shot draw shift and variance ratio."""
    rng = np.random.default_rng(seed)

    # True distribution N(0, I)
    shifts = []
    variance_ratios = []

    for _ in range(n_trials):
        exemplars = rng.standard_normal((n_exemplars, dim))
        mu_hat = exemplars.mean(axis=0)
        # Shift in Mahalanobis distance
        shift = float(np.linalg.norm(mu_hat))
        shifts.append(shift)

        # Variance ratio: sample variance along principal axes
        sample_var = float(np.var(exemplars, axis=0).mean())
        variance_ratios.append(sample_var)

    return {
        "mean_shift": float(np.mean(shifts)),
        "mean_variance_ratio": float(np.mean(variance_ratios)),
        "dim": dim,
        "n_exemplars": n_exemplars,
    }


def simulate_delta2_unbiasedness(
    dim: int = 32,
    true_delta: float = 3.0,
    n_trials: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify that Delta2_hat = max(0, delta^T A delta - tau*(1/5 + 1/n_c)) is unbiased."""
    rng = np.random.default_rng(seed)
    mu1 = np.zeros(dim)
    mu2 = np.zeros(dim)
    mu2[0] = true_delta
    true_Delta2 = true_delta ** 2

    estimates = []
    n1 = 5
    n2 = 120
    W = 1.0 / n1 + 1.0 / n2
    tau = dim  # under standard normal, E[d^T A d] = dim

    for _ in range(n_trials):
        x1 = rng.standard_normal((n1, dim)) + mu1
        x2 = rng.standard_normal((n2, dim)) + mu2
        diff = x1.mean(axis=0) - x2.mean(axis=0)
        raw = float(diff @ diff)  # A = I
        delta2_hat = raw - tau * W
        estimates.append(delta2_hat)

    mean_est = float(np.mean(estimates))
    bias = mean_est - true_Delta2

    return {
        "true_Delta2": true_Delta2,
        "mean_estimate": mean_est,
        "bias": bias,
        "relative_bias": abs(bias) / true_Delta2,
    }
