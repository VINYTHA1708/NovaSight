"""Tests for src.runtime standalone engine (Stage S1).

Verifies:
  1. Importing src.runtime does NOT import torch, torchvision, or sklearn.
  2. RuntimeScorer matches GaussianModel and ConformalScores on synthetic data.
  3. check_onnx_equivalence.py exits non-zero on a forced mismatch.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_runtime_imports_no_torch_or_sklearn():
    """Verify that importing src.runtime does not pollute sys.modules with torch, torchvision, or sklearn."""
    code = """
import sys
import src.runtime
from src.runtime.preprocess import preprocess_image, load_and_check_app_image_numpy
from src.runtime.onnx_extractor import OnnxFeatureExtractor
from src.runtime.artifact import load_runtime_artifact
from src.runtime.scoring import RuntimeScorer, make_decision

forbidden = [mod for mod in sys.modules if any(mod.startswith(pkg) for pkg in ("torch", "torchvision", "sklearn", "scipy"))]
if forbidden:
    print(f"FORBIDDEN_MODULES_IMPORTED:{forbidden}")
    sys.exit(1)
print("CLEAN_IMPORT")
sys.exit(0)
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Import check failed!\nStdout: {res.stdout}\nStderr: {res.stderr}"
    assert "CLEAN_IMPORT" in res.stdout


def test_runtime_scoring_matches_gaussian_and_conformal():
    """Verify that RuntimeScorer matches GaussianModel and ConformalScores on synthetic parameters."""
    from src.models.gaussian import GaussianModel
    from src.models.conformal import ConformalScores
    from src.runtime.artifact import RuntimeArtifact
    from src.runtime.scoring import RuntimeScorer, make_decision

    np.random.seed(42)
    D = 16
    class_names = ["crazing", "inclusion", "patches"]
    class_means = [np.random.randn(2, D).astype(np.float64) for _ in class_names]
    n_c = [40, 40, 40]
    n_sub = [[20, 20], [25, 15], [30, 10]]
    # Positive definite precision matrix
    cov = np.eye(D) + 0.1 * np.random.randn(D, D)
    cov = cov @ cov.T + 0.1 * np.eye(D)
    A = np.linalg.inv(cov).astype(np.float64)

    # Calibration scores
    class_scores = {cn: np.sort(np.random.uniform(5.0, 50.0, size=30)).astype(np.float64) for cn in class_names}
    S_pool = np.sort(np.concatenate(list(class_scores.values())))
    tau_hat = float(np.quantile(S_pool, 0.95))

    base_gaussian = GaussianModel(
        class_names=class_names,
        class_means=class_means,
        n_c=n_c,
        n_sub=n_sub,
        A=A,
        condition_number=10.0,
        shrinkage_intensity=0.05,
        k=2,
        feature_dim=D,
    )

    base_conformal = ConformalScores(
        class_scores=class_scores,
        S_pool=S_pool,
        tau_hat=tau_hat,
        alpha_default=0.05,
        alpha_grid=[0.01, 0.05, 0.10],
    )

    art = RuntimeArtifact(
        metadata={"alpha_default": 0.05},
        class_names=class_names,
        class_means=class_means,
        n_c=n_c,
        n_sub=n_sub,
        A=A,
        condition_number=10.0,
        shrinkage_intensity=0.05,
        k=2,
        feature_dim=D,
        class_scores=class_scores,
        S_pool=S_pool,
        tau_hat=tau_hat,
        alpha_default=0.05,
        alpha_grid=[0.01, 0.05, 0.10],
        provenance_hashes={},
    )
    rt_scorer = RuntimeScorer(art)

    # Test 20 random query vectors
    for _ in range(20):
        f = np.random.randn(D).astype(np.float64)
        for i, cn in enumerate(class_names):
            s_base = base_gaussian.inflated_score(f, i)
            s_rt = rt_scorer.inflated_score(f, i)
            assert np.isclose(s_base, s_rt, rtol=1e-7, atol=1e-7), f"Score mismatch: {s_base} vs {s_rt}"

            p_base = base_conformal.p_value(s_base, cn)
            p_rt = rt_scorer.conformal_p_value(s_rt, cn)
            assert np.isclose(p_base, p_rt, rtol=1e-9, atol=1e-9), f"P-val mismatch: {p_base} vs {p_rt}"

        # Closed set label
        lbl_base, _ = base_gaussian.closed_set_label(f)
        lbl_rt, _ = rt_scorer.closed_set_label(f)
        assert lbl_base == lbl_rt


def test_equivalence_script_exits_nonzero_on_forced_mismatch(tmp_path):
    """Verify that check_onnx_equivalence.py exits non-zero when a mismatch is forced."""
    out_file = tmp_path / "forced_mismatch.json"
    res = subprocess.run(
        [sys.executable, "scripts/check_onnx_equivalence.py", "--out", str(out_file), "--forced-mismatch"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, f"Expected non-zero exit code on forced mismatch, got {res.returncode}"
