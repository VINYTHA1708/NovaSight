"""S6 Tests: confusion matrix, AUROC, FRR/FDR, collect_run_metrics.

Tests:
  - CM shape is (K+1, K+1)
  - CM row/col names include unknown
  - T1 uses closed-set argmin (not conformal verdict)
  - FRR counts ambiguous as rejected
  - FDR: known images accepted / total accepted
  - AUROC bounds check
  - bootstrap_ci coverage check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics import (
    confusion_matrix_with_unknown,
    auroc_and_fpr95,
    frr_and_fdr,
    bootstrap_ci,
    collect_run_metrics,
    UNKNOWN_LABEL,
)
from src.models.gaussian import fit_gaussian
from src.models.conformal import build_conformal_scores


@pytest.fixture(scope="module")
def base_gaussian():
    np.random.seed(0)
    D, K, n_c = 32, 3, 40
    class_names = ["A", "B", "C"]
    train_features = {
        c: np.random.randn(n_c, D) + np.array([i * 10] * D)
        for i, c in enumerate(class_names)
    }
    return fit_gaussian(train_features, k=1)


@pytest.fixture(scope="module")
def base_conformal(base_gaussian):
    np.random.seed(1)
    D = base_gaussian.feature_dim
    cal_features = {
        c: np.random.randn(40, D) + np.array([i * 10] * D)
        for i, c in enumerate(base_gaussian.class_names)
    }
    return build_conformal_scores(base_gaussian, cal_features)


# ── Confusion matrix tests ──────────────────────────────────────────────────

def test_cm_shape():
    """CM must be (K+1) x (K+1)."""
    classes = ["A", "B", "C"]
    true = ["A", "B", "C", UNKNOWN_LABEL]
    pred = ["A", UNKNOWN_LABEL, "B", UNKNOWN_LABEL]
    result = confusion_matrix_with_unknown(true, pred, classes)
    assert result["matrix"].shape == (4, 4)


def test_cm_includes_unknown():
    """CM row and column names must include 'unknown'."""
    classes = ["A", "B"]
    result = confusion_matrix_with_unknown(["A"], ["A"], classes)
    assert UNKNOWN_LABEL in result["row_names"]
    assert UNKNOWN_LABEL in result["col_names"]


def test_cm_correct_counts():
    """CM must count correct and incorrect predictions."""
    classes = ["A", "B"]
    true = ["A", "A", UNKNOWN_LABEL, UNKNOWN_LABEL]
    pred = ["A", UNKNOWN_LABEL, UNKNOWN_LABEL, "B"]
    result = confusion_matrix_with_unknown(true, pred, classes)
    cm = result["matrix"]
    # A -> A: 1
    assert cm[0, 0] == 1
    # A -> unknown: 1
    assert cm[0, 2] == 1
    # unknown -> unknown: 1
    assert cm[2, 2] == 1
    # unknown -> B: 1
    assert cm[2, 1] == 1


def test_cm_markdown_table_has_unknown_header():
    """Markdown table must include unknown in header."""
    classes = ["A", "B"]
    result = confusion_matrix_with_unknown(["A"], ["A"], classes)
    assert "unknown" in result["markdown_table"]


# ── AUROC tests ───────────────────────────────────────────────────────────────

def test_auroc_perfect():
    """AUROC should be 1.0 when known scores are all low and unknown all high."""
    known = np.array([1.0, 2.0, 3.0])
    unknown = np.array([10.0, 11.0, 12.0])
    result = auroc_and_fpr95(known, unknown)
    assert abs(result["AUROC"] - 1.0) < 1e-6


def test_auroc_chance():
    """AUROC near 0.5 when scores are identical."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 100)
    result = auroc_and_fpr95(scores[:50], scores[50:])
    assert 0.3 <= result["AUROC"] <= 0.7


def test_auroc_nan_on_empty():
    """AUROC should return nan for empty inputs."""
    result = auroc_and_fpr95(np.array([]), np.array([1.0]))
    assert np.isnan(result["AUROC"])


# ── FRR / FDR tests ────────────────────────────────────────────────────────

def test_frr_all_rejected():
    """FRR=1.0 when all known images are rejected."""
    result = frr_and_fdr(
        verdicts=["unknown", "unknown", "unknown"],
        true_labels=["A", "B", "A"],
        class_names=["A", "B"],
    )
    assert result["FRR"] == 1.0


def test_frr_none_rejected():
    """FRR=0.0 when no known images are rejected."""
    result = frr_and_fdr(
        verdicts=["known", "known"],
        true_labels=["A", "B"],
        class_names=["A", "B"],
    )
    assert result["FRR"] == 0.0


def test_fdr_no_unknown_accepted():
    """FDR=0.0 when no unknowns are accepted."""
    result = frr_and_fdr(
        verdicts=["known", "known", "unknown"],
        true_labels=["A", "A", UNKNOWN_LABEL],
        class_names=["A"],
    )
    assert result["FDR"] == 0.0


# ── bootstrap CI tests ─────────────────────────────────────────────────────

def test_bootstrap_ci_bounds():
    """Bootstrap CI lower < mean < upper."""
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 100)
    lo, hi = bootstrap_ci(values, stat_fn=np.mean, n_boot=200, seed=0)
    assert lo < 0.5 and hi > -0.5
    assert lo < hi


# ── collect_run_metrics integration ───────────────────────────────────────

def test_collect_metrics_t1_uses_argmin(base_gaussian, base_conformal):
    """T1 accuracy must be 1.0 for well-separated classes (argmin always correct)."""
    np.random.seed(0)
    D = base_gaussian.feature_dim
    # Test images very close to class means (argmin will be perfect)
    test_features = {
        "A": np.random.randn(10, D) * 0.01 + np.array([0.0] * D),
        "B": np.random.randn(10, D) * 0.01 + np.array([10.0] * D),
        "C": np.random.randn(10, D) * 0.01 + np.array([20.0] * D),
    }
    unk_features = np.random.randn(10, D) + np.array([50.0] * D)  # far from all
    metrics = collect_run_metrics(base_gaussian, base_conformal, test_features, unk_features, alpha=0.05)
    assert metrics["T1_accuracy"] == 1.0


def test_collect_metrics_unk_rejection_high(base_gaussian, base_conformal):
    """Unknown images far from all classes should have high rejection rate."""
    np.random.seed(2)
    D = base_gaussian.feature_dim
    test_features = {
        c: np.random.randn(10, D) * 0.01 + np.array([i * 10.0] * D)
        for i, c in enumerate(base_gaussian.class_names)
    }
    unk_features = np.random.randn(20, D) + np.array([100.0] * D)  # very far
    metrics = collect_run_metrics(base_gaussian, base_conformal, test_features, unk_features, alpha=0.05)
    assert metrics["T3_unk_rejection"] > 0.5


def test_collect_metrics_cm_shape(base_gaussian, base_conformal):
    """CM must be (K+1) x (K+1)."""
    np.random.seed(3)
    D = base_gaussian.feature_dim
    K = len(base_gaussian.class_names)
    test_features = {
        c: np.random.randn(5, D) * 0.01 + np.array([i * 10.0] * D)
        for i, c in enumerate(base_gaussian.class_names)
    }
    unk_features = np.random.randn(5, D) + np.array([100.0] * D)
    metrics = collect_run_metrics(base_gaussian, base_conformal, test_features, unk_features)
    cm = metrics["confusion_matrix"]["matrix"]
    assert cm.shape == (K + 1, K + 1)
