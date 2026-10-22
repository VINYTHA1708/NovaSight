"""NovaSight standardized metrics and evaluation functions (Protocol v1.1 Task E6, Part B).

Replaces clashing v1.0 terms with standardized terminology:
  - empty_rate: P(Gamma_alpha == empty)
  - ambiguous_rate: P(|Gamma_alpha| >= 2)
  - known_FRR: P(c not in Gamma_alpha | Y = c) on known test images
  - unk_rejection: P(Gamma_alpha == empty | unknown image)
  - operator_routing_rate: empty_rate + ambiguous_rate
  - new_class_coverage: P(p_new > alpha | new class)
  - new_class_recall: P(c_hat == new_class | new class)
  - new_class_FRR: 1 - new_class_coverage

Confusion matrix layout:
  Columns: [Class_1, ..., Class_K, 'ambiguous', 'unknown']
  Rows: [Class_1, ..., Class_K, 'unknown']
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import beta as scipy_beta
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

AMBIGUOUS_LABEL = "ambiguous"
UNKNOWN_LABEL = "unknown"


def compute_decision_rates(
    prediction_sets: list[set[str]],
    true_labels: list[str],
    is_known: list[bool],
) -> dict[str, float]:
    """Compute standardized Protocol v1.1 decision rates (Part B)."""
    n_total = len(prediction_sets)
    if n_total == 0:
        return {}

    n_empty = sum(1 for s in prediction_sets if len(s) == 0)
    n_ambig = sum(1 for s in prediction_sets if len(s) >= 2)
    n_single = sum(1 for s in prediction_sets if len(s) == 1)

    empty_rate = n_empty / n_total
    ambiguous_rate = n_ambig / n_total
    routing_rate = (n_empty + n_ambig) / n_total

    # Known class FRR: true_label not in Gamma
    known_indices = [i for i, k in enumerate(is_known) if k]
    if known_indices:
        known_not_covered = sum(1 for i in known_indices if true_labels[i] not in prediction_sets[i])
        known_frr = known_not_covered / len(known_indices)
    else:
        known_frr = float("nan")

    # Unknown rejection: Gamma is empty
    unk_indices = [i for i, k in enumerate(is_known) if not k]
    if unk_indices:
        unk_rejected = sum(1 for i in unk_indices if len(prediction_sets[i]) == 0)
        unk_rejection = unk_rejected / len(unk_indices)
    else:
        unk_rejection = float("nan")

    return {
        "empty_rate": float(empty_rate),
        "ambiguous_rate": float(ambiguous_rate),
        "operator_routing_rate": float(routing_rate),
        "known_FRR": float(known_frr),
        "unk_rejection": float(unk_rejection),
    }


def confusion_matrix_v11(
    true_labels: list[str],
    prediction_sets: list[set[str]],
    closed_set_preds: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute K+1 x K+2 confusion matrix with separate 'ambiguous' and 'unknown' columns."""
    row_names = class_names + [UNKNOWN_LABEL]
    col_names = class_names + [AMBIGUOUS_LABEL, UNKNOWN_LABEL]

    row_map = {r: i for i, r in enumerate(row_names)}
    col_map = {c: i for i, c in enumerate(col_names)}

    matrix = np.zeros((len(row_names), len(col_names)), dtype=int)

    for true, p_set, c_hat in zip(true_labels, prediction_sets, closed_set_preds):
        r_idx = row_map.get(true, row_map[UNKNOWN_LABEL])
        if len(p_set) == 0:
            c_idx = col_map[UNKNOWN_LABEL]
        elif len(p_set) >= 2:
            c_idx = col_map[AMBIGUOUS_LABEL]
        else:
            single = next(iter(p_set))
            c_idx = col_map.get(single, col_map[AMBIGUOUS_LABEL])
        matrix[r_idx, c_idx] += 1

    # Format Markdown
    header = "| True \\ Pred | " + " | ".join(col_names) + " |"
    sep = "|" + "---|" * (len(col_names) + 1)
    rows = [header, sep]
    for i, rname in enumerate(row_names):
        vals = " | ".join(str(matrix[i, j]) for j in range(len(col_names)))
        rows.append(f"| **{rname}** | {vals} |")

    return {
        "matrix": matrix.tolist(),
        "row_names": row_names,
        "col_names": col_names,
        "markdown_table": "\n".join(rows),
    }


def compute_bootstrap_ci(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute 1,000-resample image bootstrap CI (point, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    point = float(metric_fn(known_scores, unknown_scores))

    boot_stats = []
    n_k = len(known_scores)
    n_u = len(unknown_scores)
    if n_k == 0 or n_u == 0:
        return point, float("nan"), float("nan")

    for _ in range(n_resamples):
        k_sample = rng.choice(known_scores, size=n_k, replace=True)
        u_sample = rng.choice(unknown_scores, size=n_u, replace=True)
        stat = metric_fn(k_sample, u_sample)
        if not np.isnan(stat):
            boot_stats.append(stat)

    if not boot_stats:
        return point, float("nan"), float("nan")

    ci_lo = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, ci_lo, ci_hi


def auroc_metric(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Compute AUROC where higher score means more unknown."""
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores, unknown_scores])
    try:
        return float(roc_auc_score(y_true, y_scores))
    except Exception:
        return float("nan")


def aupr_metric(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Compute AUPR (average precision) for novelty detection."""
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores, unknown_scores])
    try:
        return float(average_precision_score(y_true, y_scores))
    except Exception:
        return float("nan")


def fpr95_metric(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Compute FPR at 95% TPR (unknown detection)."""
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores, unknown_scores])
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        # Find smallest fpr where tpr >= 0.95
        valid = [f for f, t in zip(fpr, tpr) if t >= 0.95]
        return float(min(valid)) if valid else float("nan")
    except Exception:
        return float("nan")


def paired_mcnemar_test(
    preds_before: list[str],
    preds_after: list[str],
    targets: list[str],
) -> dict[str, float]:
    """Compute descriptive paired McNemar test comparing classification correctness before/after enrollment."""
    # b: correct before, incorrect after
    # c: incorrect before, correct after
    b = 0
    c = 0
    for pb, pa, y in zip(preds_before, preds_after, targets):
        cb = (pb == y)
        ca = (pa == y)
        if cb and not ca:
            b += 1
        elif not cb and ca:
            c += 1

    # Continuity-corrected chi-square: (|b - c| - 1)^2 / (b + c)
    if b + c > 0:
        stat = float((abs(b - c) - 1.0) ** 2 / (b + c))
        # 1-df chi2 p-value
        from scipy.stats import chi2
        p_val = float(1.0 - chi2.cdf(stat, df=1))
    else:
        stat = 0.0
        p_val = 1.0

    return {
        "mcnemar_stat": stat,
        "mcnemar_p_value": p_val,
        "b_correct_to_wrong": b,
        "c_wrong_to_correct": c,
    }


def compute_beta_spread(n: int = 40, alpha: float = 0.05) -> dict[str, float]:
    """Compute theoretical Beta(n+1-l, l) spread for fixed-CAL coverage (G1)."""
    l_val = int(np.floor((n + 1) * alpha))
    a = n + 1 - l_val
    b = l_val
    mean_val = float(scipy_beta.mean(a, b))
    std_val = float(scipy_beta.std(a, b))
    q025 = float(scipy_beta.ppf(0.025, a, b))
    q975 = float(scipy_beta.ppf(0.975, a, b))
    return {
        "beta_mean": mean_val,
        "beta_std": std_val,
        "beta_ci_95": [q025, q975],
    }
