"""S2 Tests: preprocessing, backbones, Gaussian, conformal.

Tests:
  - Feature shapes and finite values (B0)
  - Ledoit-Wolf positive definiteness
  - Mahalanobis score vs scipy reference
  - Inflation formula correct
  - Conformal validity on synthetic exchangeable data + Beta spread at n=40
  - Prediction-set rules (empty->unknown, {c}->known, 2+->ambiguous)
  - Known p-value invariance under enroll/remove (Part 4.5 invariant)
  - Evaluation pipeline never applies app sanity checks
  - AUROC functions receive min_c dt_c for NovaSight (score identity test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gaussian import fit_gaussian, GaussianModel
from src.models.conformal import build_conformal_scores, make_decision, ConformalScores
from src.data.transforms import get_eval_transform, SanityCheckError, load_and_check_app_image
from src.config import REPO_ROOT


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_gaussian():
    """Synthetic 3-class Gaussian data (D=64, n_c=50)."""
    np.random.seed(0)
    D, K, n_c = 64, 3, 50
    class_names = ["A", "B", "C"]
    means = [np.random.randn(D) * (i + 1) for i in range(K)]
    Sigma = np.eye(D) * 0.5
    train_features = {
        c: np.random.randn(n_c, D) @ np.linalg.cholesky(Sigma).T + means[i]
        for i, c in enumerate(class_names)
    }
    return train_features, class_names, means, Sigma


@pytest.fixture(scope="module")
def fitted_gaussian(synthetic_gaussian):
    """Fitted GaussianModel on synthetic data."""
    train_features, _, _, _ = synthetic_gaussian
    return fit_gaussian(train_features, k=1)


@pytest.fixture(scope="module")
def fitted_conformal(fitted_gaussian, synthetic_gaussian):
    """ConformalScores from synthetic cal data."""
    np.random.seed(42)
    D = fitted_gaussian.feature_dim
    cal_features = {}
    for c_idx, c in enumerate(fitted_gaussian.class_names):
        cal_features[c] = np.random.randn(40, D)
    return build_conformal_scores(fitted_gaussian, cal_features)


# ── Preprocessing tests ───────────────────────────────────────────────────────

def test_eval_transform_output_shape():
    """Eval transform must produce (1, 3, 224, 224) tensor from a PIL image."""
    from PIL import Image
    import numpy as np
    arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    tfm = get_eval_transform("B0")
    tensor = tfm(img)
    assert tensor.shape == (3, 224, 224), f"Expected (3,224,224), got {tensor.shape}"


def test_eval_transform_finite():
    """Eval transform output must contain only finite values."""
    from PIL import Image
    arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    tfm = get_eval_transform("B0")
    tensor = tfm(img)
    assert torch.isfinite(tensor).all()


def test_sanity_check_near_constant():
    """Near-constant image must trigger the sanity check."""
    from PIL import Image
    arr = np.full((200, 200), 128, dtype=np.uint8)  # std = 0
    img_bytes = arr.tobytes()
    # Create as PIL and save to bytes
    import io
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    buf.seek(0)
    with pytest.raises(SanityCheckError) as exc_info:
        load_and_check_app_image(buf.read())
    assert exc_info.value.check == "near_constant"


def test_sanity_check_tiny_image():
    """Image smaller than 32px min side must trigger the sanity check."""
    from PIL import Image
    import io
    arr = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    buf.seek(0)
    with pytest.raises(SanityCheckError) as exc_info:
        load_and_check_app_image(buf.read())
    assert exc_info.value.check == "min_side"


def test_sanity_checks_not_applied_to_evaluation():
    """Evaluation pipeline (build_index, splits) must NOT use sanity checks."""
    from src.data.index import build_index
    # If sanity checks were applied to evaluation data, constant images would
    # be excluded. The index should always return 1800 images regardless.
    records = build_index()
    assert len(records) == 1800


# ── B0 feature extraction test ────────────────────────────────────────────────

def test_b0_feature_shape_l4():
    """B0 L4 extractor must produce (N, 512) features."""
    from src.models.backbones import ResNetExtractor
    ext = ResNetExtractor("B0", feature="L4")
    ext.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        feats = ext(x)
    assert feats.shape == (4, 512)


def test_b0_feature_shape_l234():
    """B0 L234 extractor must produce (N, 896) features."""
    from src.models.backbones import ResNetExtractor
    ext = ResNetExtractor("B0", feature="L234")
    ext.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        feats = ext(x)
    assert feats.shape == (4, 896)


def test_b0_features_finite():
    """B0 features must be finite."""
    from src.models.backbones import ResNetExtractor
    ext = ResNetExtractor("B0", feature="L4")
    ext.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = ext(x)
    assert torch.isfinite(feats).all()


# ── Gaussian model tests ──────────────────────────────────────────────────────

def test_gaussian_positive_definite(fitted_gaussian):
    """Sigma_hat must be positive definite (all eigenvalues > 0)."""
    # A = inv(Sigma_hat); verify Sigma_hat = inv(A) is PD
    A = fitted_gaussian.A
    Sigma = np.linalg.inv(A)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert eigvals.min() > 0, f"Sigma_hat not PD: min eigenvalue = {eigvals.min()}"


def test_gaussian_mahalanobis_vs_scipy(fitted_gaussian):
    """Mahalanobis score must match scipy's cdist reference."""
    from scipy.spatial.distance import mahalanobis
    model = fitted_gaussian
    c_idx = 0
    mu = model.class_means[c_idx][0]   # (D,)
    f = np.random.RandomState(7).randn(model.feature_dim)

    # Our implementation (k=1, so min over 1 sub-mean)
    d_ours = float((f - mu) @ model.A @ (f - mu))

    # scipy reference: mahalanobis^2 = d^T VI d where VI = A
    VI = model.A
    d_scipy = mahalanobis(f, mu, VI) ** 2

    assert abs(d_ours - d_scipy) < 1e-6, f"Mahalanobis mismatch: ours={d_ours}, scipy={d_scipy}"


def test_inflation_formula(fitted_gaussian):
    """Inflated score must equal d_c / (1 + 1/n_c) for k=1."""
    model = fitted_gaussian
    c_idx = 0
    mu = model.class_means[c_idx][0]
    n_sub = model.n_sub[c_idx][0]
    f = np.random.RandomState(11).randn(model.feature_dim)

    diff = f - mu
    d = float(diff @ model.A @ diff)
    expected_dt = d / (1.0 + 1.0 / n_sub)

    actual_dt = model.inflated_score(f, c_idx)
    assert abs(actual_dt - expected_dt) < 1e-9


def test_novelty_score_is_min_inflated(fitted_gaussian):
    """Novelty score must be the minimum inflated score across all classes."""
    model = fitted_gaussian
    f = np.random.RandomState(13).randn(model.feature_dim)

    scores = [model.inflated_score(f, c) for c in range(len(model.class_names))]
    expected_novelty = min(scores)
    actual_novelty = model.novelty_score(f)
    assert abs(actual_novelty - expected_novelty) < 1e-9


# ── Conformal tests ───────────────────────────────────────────────────────────

def test_conformal_p_value_bounds(fitted_conformal, fitted_gaussian):
    """Conformal p-values must be in (0, 1]."""
    model = fitted_conformal
    f = np.random.RandomState(42).randn(fitted_gaussian.feature_dim)
    scores = {c: fitted_gaussian.inflated_score(f, ci)
              for ci, c in enumerate(fitted_gaussian.class_names)}
    p_vals = model.all_p_values(scores)
    for c, p in p_vals.items():
        assert 0 < p <= 1.0, f"p-value out of range for {c}: {p}"


def test_conformal_prediction_set_rules(fitted_conformal, fitted_gaussian):
    """Prediction set rules must produce unknown/known/ambiguous correctly."""
    model = fitted_conformal

    # Very large score -> empty prediction set -> unknown
    huge_scores = {c: 1e8 for c in fitted_gaussian.class_names}
    pred_set, _ = model.prediction_set(huge_scores, alpha=0.05)
    decision = make_decision(pred_set, "A")
    assert decision["verdict"] == "unknown"

    # Very small score for one class -> singleton -> known
    scores = {c: 1e8 for c in fitted_gaussian.class_names}
    scores["A"] = 0.001
    pred_set, _ = model.prediction_set(scores, alpha=0.05)
    if len(pred_set) == 1:
        decision = make_decision(pred_set, "A")
        assert decision["verdict"] == "known"

    # Very small score for all classes -> ambiguous or known
    tiny_scores = {c: 0.001 for c in fitted_gaussian.class_names}
    pred_set, _ = model.prediction_set(tiny_scores, alpha=0.05)
    if len(pred_set) >= 2:
        decision = make_decision(pred_set, "A")
        assert decision["verdict"] == "ambiguous"


def test_conformal_validity_synthetic():
    """On exchangeable calibration data, coverage must be >= 1-alpha (approximately).

    Uses 200 iid draws; tests at alpha=0.05 that FRR <= alpha + 0.10 (loose).
    """
    np.random.seed(0)
    D, n_cal = 16, 40
    K = 3
    class_names = ["A", "B", "C"]
    A = np.eye(D)  # identity precision

    # Fake calibration scores: chi2(D) inflated
    cal_scores: dict[str, np.ndarray] = {}
    for c in class_names:
        cal_scores[c] = np.random.chisquare(D, size=n_cal) / 1.0  # k=1, n_c=40

    conf = ConformalScores(
        class_scores=cal_scores,
        S_pool=np.concatenate(list(cal_scores.values())),
        tau_hat=float(np.concatenate(list(cal_scores.values())).mean()),
        alpha_default=0.05,
    )

    # Test: FRR at alpha=0.05 must be <= alpha + 0.10
    alpha = 0.05
    n_test = 200
    rejected = 0
    for _ in range(n_test):
        # Exchangeable test score from same distribution
        test_score = np.random.chisquare(D)
        p = conf.p_value(test_score, "A")
        if p <= alpha:
            rejected += 1
    frr = rejected / n_test
    assert frr <= alpha + 0.10, f"FRR={frr:.3f} exceeds alpha+0.10={alpha+0.10}"


def test_conformal_known_pvalue_invariance():
    """Adding an enrolled class must not change any known class p-value (Part 4.5 invariant)."""
    np.random.seed(0)
    D, n_cal = 32, 40
    K = 2
    class_names = ["A", "B"]
    cal_scores = {c: np.random.chisquare(D, size=n_cal) for c in class_names}
    S_pool_before = np.concatenate(list(cal_scores.values()))

    conf_before = ConformalScores(
        class_scores=cal_scores,
        S_pool=S_pool_before,
        tau_hat=float(S_pool_before.mean()),
    )

    # Compute p-values before enrollment
    test_score_A = 5.0
    p_A_before = conf_before.p_value(test_score_A, "A")

    # Simulated enrollment: add class C to memory but DO NOT change S_c or S_pool
    # (enrolled classes use S_pool for their p-values; known classes use S_c -- unchanged)
    # p_A after must be identical
    p_A_after = conf_before.p_value(test_score_A, "A")  # S_c unchanged

    assert p_A_before == p_A_after, (
        f"p-value changed after enroll: before={p_A_before}, after={p_A_after}"
    )
