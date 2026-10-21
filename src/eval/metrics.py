"""NovaSight evaluation metrics (Part 6.3).

Functions:
  - confusion_matrix_with_unknown: K+1 x K+1 matrix including "unknown" column/row
  - auroc_and_fpr95: AUROC, FPR@95%TPR for any baseline
  - fdr_frr: FDR (False Discovery Rate) and FRR (False Rejection Rate)
  - bootstrap_ci: 1000-sample stratified bootstrap 95% CI
  - summary_table: collect all metrics for a single run
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "unknown"


def confusion_matrix_with_unknown(
    true_labels: list[str],
    pred_labels: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute K+1 x K+1 confusion matrix including the 'unknown' class.

    Rows = true class (including 'unknown' for held-out images).
    Columns = predicted class (including 'unknown' for rejected images).

    Args:
        true_labels: True class names; use UNKNOWN_LABEL for held-out images.
        pred_labels: Predicted class names; use UNKNOWN_LABEL for rejected images.
        class_names: Ordered list of known class names (no 'unknown').

    Returns:
        Dict with:
          - matrix: (K+1, K+1) numpy array
          - row_names: List of row names (true labels)
          - col_names: List of col names (predicted labels)
          - markdown_table: Human-readable markdown table
    """
    all_labels = class_names + [UNKNOWN_LABEL]
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)
    cm = np.zeros((n, n), dtype=int)

    for true, pred in zip(true_labels, pred_labels):
        r = label_to_idx.get(true, n - 1)  # unknown -> last row
        c = label_to_idx.get(pred, n - 1)  # unknown -> last col
        cm[r, c] += 1

    # Markdown table
    col_header = "| True \\ Pred | " + " | ".join(all_labels) + " |"
    sep = "|" + "---|" * (n + 1)
    rows = [col_header, sep]
    for i, rname in enumerate(all_labels):
        row_vals = " | ".join(str(cm[i, j]) for j in range(n))
        rows.append(f"| **{rname}** | {row_vals} |")
    md_table = "\n".join(rows)

    return {
        "matrix": cm,
        "row_names": all_labels,
        "col_names": all_labels,
        "markdown_table": md_table,
    }


def auroc_and_fpr95(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
) -> dict[str, float]:
    """Compute AUROC and FPR@95%TPR.

    Convention: unknown images should have HIGHER novelty scores.

    Args:
        known_scores: (n_k,) novelty scores for known-class images.
        unknown_scores: (n_u,) novelty scores for unknown-class images.

    Returns:
        Dict with AUROC and FPR95.
    """
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return {"AUROC": float("nan"), "FPR95": float("nan")}

    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])

    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = float("nan")

    # FPR@95%TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = float("nan")
    for f, t in zip(fpr, tpr):
        if t >= 0.95:
            fpr95 = float(f)
            break

    return {"AUROC": auroc, "FPR95": fpr95}


def frr_and_fdr(
    verdicts: list[str],
    true_labels: list[str],
    class_names: list[str],
) -> dict[str, float]:
    """Compute FRR and FDR for a set of decisions.

    FRR (False Rejection Rate): fraction of known images marked "unknown".
    FDR (False Discovery Rate): fraction of accepted images that are actually unknown.

    Args:
        verdicts: List of verdict strings ("known", "unknown", "ambiguous").
        true_labels: True labels (known class name or UNKNOWN_LABEL).
        class_names: Known class names.

    Returns:
        Dict with FRR and FDR.
    """
    known_mask = np.array([l in class_names for l in true_labels])
    rejected_mask = np.array([v == "unknown" for v in verdicts])

    n_known = int(known_mask.sum())
    n_unk = int((~known_mask).sum())

    # FRR: known images rejected / total known
    frr = float((known_mask & rejected_mask).sum()) / max(n_known, 1)

    # FDR: unknown images accepted / total accepted
    accepted_mask = ~rejected_mask
    n_accepted = int(accepted_mask.sum())
    fdr = float((~known_mask & accepted_mask).sum()) / max(n_accepted, 1)

    return {"FRR": frr, "FDR": fdr, "n_known": n_known, "n_unknown": n_unk}


def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic.

    Args:
        values: Input data array.
        stat_fn: Function that takes an array and returns a scalar.
        n_boot: Number of bootstrap iterations.
        alpha: Confidence level (produces (1-alpha) CI).
        seed: Random seed.

    Returns:
        (lower, upper) confidence interval.
    """
    rng = np.random.default_rng(seed)
    boot_stats = np.zeros(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = stat_fn(values[idx])
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def per_class_accuracy(
    true_labels: list[str],
    pred_labels: list[str],
    class_names: list[str],
) -> dict[str, float]:
    """Compute per-class accuracy.

    Args:
        true_labels: True class names.
        pred_labels: Predicted class names.
        class_names: Known class names.

    Returns:
        Dict of class_name -> accuracy.
    """
    result = {}
    for c in class_names:
        mask = [l == c for l in true_labels]
        n = sum(mask)
        if n == 0:
            result[c] = float("nan")
            continue
        correct = sum(1 for t, p, m in zip(true_labels, pred_labels, mask)
                      if m and p == c)
        result[c] = correct / n
    return result




def collect_run_metrics(
    gaussian,
    conformal,
    test_features: dict[str, np.ndarray],
    unk_features: np.ndarray,
    alpha: float = 0.05,
    enrollment_recalls: list[float] | None = None,
) -> dict[str, Any]:
    """Collect all T1-T5 metrics for one run.

    T1: Closed-set accuracy = argmin(inflated_score) prediction (never unknown).
        This is the standard K-class accuracy for the K-class subproblem.
    CM: Uses conformal verdict; ambiguous -> unknown column in the CM.
    T3: FRR = fraction of known images NOT uniquely identified (ambiguous + empty).

    Args:
        gaussian: GaussianModel.
        conformal: ConformalScores.
        test_features: Dict of class_name -> (n, D) known-class test features.
        unk_features: (n_u, D) held-out class test features.
        alpha: Significance level.
        enrollment_recalls: List of per-draw H1 recall values (for T5).

    Returns:
        Dict of all metrics.
    """
    from src.models.conformal import make_decision

    # Known test: compute T1 (closed-set) and conformal verdicts separately
    t1_true, t1_pred = [], []
    cm_true, cm_pred = [], []
    all_verdict = []
    known_novelty_scores = []

    for cname, feats in test_features.items():
        for f in feats:
            # T1: argmin closed-set label (always a class)
            all_scores_arr = gaussian.all_inflated_scores(f)
            c_hat = gaussian.class_names[int(np.argmin(all_scores_arr))]
            t1_true.append(cname)
            t1_pred.append(c_hat)

            # Conformal decision for CM and rejection rates
            scores = {c: gaussian.inflated_score(f, ci)
                      for ci, c in enumerate(gaussian.class_names)}
            pred_set, p_vals = conformal.prediction_set(scores, alpha)
            dec = make_decision(pred_set, c_hat)
            all_verdict.append(dec["verdict"])

            # CM predicted label: only "known" singleton counts; else "unknown"
            if dec["verdict"] == "known":
                cm_pred.append(dec["class_name"])
            else:
                cm_pred.append(UNKNOWN_LABEL)
            cm_true.append(cname)
            known_novelty_scores.append(gaussian.novelty_score(f))

    known_novelty_scores = np.array(known_novelty_scores)
    t1_acc = float(sum(1 for t, p in zip(t1_true, t1_pred) if t == p) / max(len(t1_true), 1))

    # Unknown: decisions for CM and rejection
    unk_true_labels = [UNKNOWN_LABEL] * len(unk_features)
    unk_pred = []
    unk_novelty_scores = gaussian.batch_novelty_scores(unk_features)
    for f in unk_features:
        scores = {c: gaussian.inflated_score(f, ci)
                  for ci, c in enumerate(gaussian.class_names)}
        pred_set, _ = conformal.prediction_set(scores, alpha)
        c_hat, _ = gaussian.closed_set_label(f)
        dec = make_decision(pred_set, c_hat)
        unk_pred.append(dec["class_name"] if dec["verdict"] == "known" else UNKNOWN_LABEL)

    # Confusion matrix
    cm_result = confusion_matrix_with_unknown(
        cm_true + unk_true_labels,
        cm_pred + unk_pred,
        gaussian.class_names,
    )

    # T2: AUROC, FPR@95
    auc_results = auroc_and_fpr95(known_novelty_scores, unk_novelty_scores)

    # T3: FRR = not uniquely identified; FDR = unknown accepted as known
    frr = float(sum(1 for v in all_verdict if v != "known") / max(len(all_verdict), 1))
    n_unk_accepted = sum(1 for p in unk_pred if p != UNKNOWN_LABEL)
    n_known_accepted = sum(1 for p in cm_pred if p != UNKNOWN_LABEL)
    fdr = n_unk_accepted / max(n_known_accepted + n_unk_accepted, 1)
    unk_rejection = sum(1 for p in unk_pred if p == UNKNOWN_LABEL) / max(len(unk_pred), 1)

    return {
        "T1_accuracy": t1_acc,
        "T2_AUROC": auc_results["AUROC"],
        "T2_FPR95": auc_results["FPR95"],
        "T3_FRR": frr,
        "T3_FDR": fdr,
        "T3_unk_rejection": unk_rejection,
        "T5_mean_recall_H1": float(np.mean(enrollment_recalls)) if enrollment_recalls else None,
        "confusion_matrix": cm_result,
        "alpha": alpha,
    }
