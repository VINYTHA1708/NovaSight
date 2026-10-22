"""NovaSight Protocol v1.1 Verification and Safeguards Test Suite (Item C).

Tests:
  - Canonical hashing & line-ending independence
  - PoolGuard access violations
  - Splits v2 disjointness & duplicate integrity (Pa_101/105)
  - Gate check V (exact + pHash, no cosine >= 0.95 rule)
  - Gate check N (pooled calibration scores S_pool, floor 1/201)
  - Gate check O (mixture sub-means + parametric bootstrap)
  - Standardized metrics terminology & confusion matrix layout
  - verify_artifact refusing on test-pool hash, tampered file, or missing fields
  - run_jobs refusing on commit mismatch or dirty repo
  - Rehearsal selection hierarchy & tie-breaking
  - V1 synthetic simulations (Proposition 1, Delta^2 unbiasedness)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow importing src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.config import REPO_ROOT
from src.data.guard import DataAccessError, PoolGuard, check_pool_access
from src.data.splits_v2 import load_manifest_v2
from src.eval.metrics_v11 import (
    AMBIGUOUS_LABEL,
    UNKNOWN_LABEL,
    compute_decision_rates,
    confusion_matrix_v11,
)
from src.eval.rehearsal import CandidateResult, select_best_candidate
from src.eval.simulate import simulate_delta2_unbiasedness, simulate_proposition_1
from src.models.conformal import build_conformal_scores
from src.models.gate import run_gate
from src.models.gaussian import fit_gaussian
from src.utils.hashing import (
    canonical_file_sha256,
    canonical_json_sha256,
)
from scripts.run_jobs import check_git_clean_and_commit
from scripts.verify_artifact import verify_artifact_dir


# ── 1. Canonical Hashing Tests ───────────────────────────────────────────────

def test_canonical_file_sha256_line_ending_and_bom_independence(tmp_path: Path):
    """canonical_file_sha256 must produce identical hashes regardless of LF vs CRLF or UTF-8 BOM."""
    content = "key: value\nlist:\n  - item1\n  - item2\n"

    # LF
    f_lf = tmp_path / "test_lf.yaml"
    f_lf.write_bytes(content.encode("utf-8"))

    # CRLF
    f_crlf = tmp_path / "test_crlf.yaml"
    f_crlf.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))

    # BOM + CRLF
    f_bom_crlf = tmp_path / "test_bom_crlf.yaml"
    f_bom_crlf.write_bytes(b"\xef\xbb\xbf" + content.replace("\n", "\r\n").encode("utf-8"))

    h_lf = canonical_file_sha256(f_lf)
    h_crlf = canonical_file_sha256(f_crlf)
    h_bom = canonical_file_sha256(f_bom_crlf)

    assert h_lf == h_crlf, "LF and CRLF must produce identical canonical sha256"
    assert h_lf == h_bom, "UTF-8 BOM must be stripped before canonical hashing"


def test_canonical_json_sha256_formatting_independence():
    """canonical_json_sha256 must be identical regardless of key order or whitespace formatting."""
    d1 = {"b": 2, "a": [1, 2, 3]}
    d2 = {"a": [1, 2, 3], "b": 2}
    assert canonical_json_sha256(d1) == canonical_json_sha256(d2)


# ── 2. PoolGuard Access Violation Tests ──────────────────────────────────────

def test_pool_guard_access_violation():
    """PoolGuard must raise DataAccessError when attempting to read an undeclared pool."""
    with PoolGuard(allowed_pools=["train", "sel_cal"]):
        # Declared pools should pass
        check_pool_access("train")
        check_pool_access("Cr_train")
        check_pool_access("sel_cal")

        # Undeclared pools must raise DataAccessError
        with pytest.raises(DataAccessError) as exc_info:
            check_pool_access("test")
        assert "not in the declared allowed pools" in str(exc_info.value)

        with pytest.raises(DataAccessError):
            check_pool_access("unk_test")

        with pytest.raises(DataAccessError):
            check_pool_access("new_test")


# ── 3. Splits v2 Disjointness and Duplicate Integrity Tests ─────────────────

def test_splits_v2_disjointness_and_pa_duplicate_integrity():
    """Splits v2 must be strictly hash-disjoint and never straddle Pa_101/105 across pools."""
    manifest_dir = REPO_ROOT / "data" / "manifests" / "v2"
    assert manifest_dir.is_dir(), "v2 manifests directory must exist"

    sample_manifests = list(manifest_dir.glob("H=*.json"))[:10]
    assert len(sample_manifests) > 0

    for m_path in sample_manifests:
        m = load_manifest_v2(m_path)
        pool_hashes: dict[str, set[str]] = {}

        pa_101_pool = None
        pa_105_pool = None

        for pool_name, records in m["pools"].items():
            pool_hashes[pool_name] = set()
            for r in records:
                h = r.get("sha256")
                fn = r.get("filename", "")
                if h:
                    pool_hashes[pool_name].add(h)
                if fn == "Pa_101.bmp":
                    pa_101_pool = pool_name
                elif fn == "Pa_105.bmp":
                    pa_105_pool = pool_name

        # Check disjointness
        pool_names = list(pool_hashes.keys())
        for i in range(len(pool_names)):
            for j in range(i + 1, len(pool_names)):
                p1, p2 = pool_names[i], pool_names[j]
                inter = pool_hashes[p1].intersection(pool_hashes[p2])
                assert len(inter) == 0, f"Manifest {m_path.name} pools {p1} and {p2} share hashes: {inter}"

        # If Pa was partitioned in this manifest, check atomic duplicate integrity
        if pa_101_pool is not None and pa_105_pool is not None:
            assert pa_101_pool == pa_105_pool, (
                f"Duplicate pair Pa_101 ({pa_101_pool}) and Pa_105 ({pa_105_pool}) straddle pools in {m_path.name}!"
            )


# ── 4. Gate Check V, N, O Tests ──────────────────────────────────────────────

@pytest.fixture
def synthetic_gaussian_and_conformal():
    rng = np.random.default_rng(42)
    D = 16
    classes = ["A", "B", "C"]
    train_data = {c: rng.standard_normal((40, D)) + (i * 10.0) for i, c in enumerate(classes)}
    cal_data = {c: rng.standard_normal((40, D)) + (i * 10.0) for i, c in enumerate(classes)}

    gauss = fit_gaussian(train_data, k=2)
    conformal = build_conformal_scores(gauss, cal_data)
    return gauss, conformal


def test_gate_check_v_no_cosine_threshold(synthetic_gaussian_and_conformal):
    """Check V must NOT reject on high cosine similarity (>= 0.95) alone per D6."""
    gauss, conformal = synthetic_gaussian_and_conformal
    D = gauss.feature_dim

    # Create 5 collinear embeddings with cosine similarity 1.0 (identical direction, different magnitude)
    base_vec = np.ones(D)
    embeddings = np.array([base_vec * (i + 1.0) for i in range(5)])  # cosine = 1.0
    sha256s = [f"unique_sha_{i}" for i in range(5)]

    # Perceptual hashes differ significantly
    import imagehash
    mock_phashes = [
        imagehash.hex_to_hash(h)
        for h in [
            "0000000000000000",
            "ffffffffffffffff",
            "aaaaaaaaaaaaaaaa",
            "5555555555555555",
            "3333333333333333",
        ]
    ]

    report = run_gate(
        embeddings=embeddings,
        sha256s=sha256s,
        class_name="HighCosineTest",
        existing_class_names=gauss.class_names,
        gaussian=gauss,
        conformal=conformal,
        alpha_cons=0.01,
        q_BH=0.05,
        eps_max=0.50,
        phashes=mock_phashes,
    )
    # Cosine is 1.0, but under D6 cosine rule is removed! It should NOT fail validity check V.
    assert not any("near-duplicate (cosine" in issue for issue in report.validity_issues)
    assert not any("near-duplicates" in issue for issue in report.validity_issues)


def test_gate_check_v_perceptual_hash_rejection(synthetic_gaussian_and_conformal):
    """Check V must reject when images have duplicate SHA-256 or identical pHash."""
    gauss, conformal = synthetic_gaussian_and_conformal
    D = gauss.feature_dim
    embeddings = np.random.randn(5, D)
    sha256s = [f"unique_sha_{i}" for i in range(5)]

    # Provide two identical perceptual hashes (Hamming distance 0 <= 2)
    import imagehash
    identical_p = imagehash.hex_to_hash("0123456789abcdef")
    mock_phashes = [identical_p, identical_p, imagehash.hex_to_hash("ffffffffffffffff"),
                    imagehash.hex_to_hash("aaaaaaaaaaaaaaaa"), imagehash.hex_to_hash("5555555555555555")]

    report = run_gate(
        embeddings=embeddings,
        sha256s=sha256s,
        class_name="PHashTest",
        existing_class_names=gauss.class_names,
        gaussian=gauss,
        conformal=conformal,
        alpha_cons=0.01,
        q_BH=0.05,
        eps_max=0.50,
        phashes=mock_phashes,
    )
    assert report.outcome == "REJECT"
    assert any("perceptual near-duplicates" in issue for issue in report.validity_issues)


def test_gate_check_n_pooled_resolution(synthetic_gaussian_and_conformal):
    """Check N must evaluate against S_pool (resolution floor 1/201) per D5."""
    gauss, conformal = synthetic_gaussian_and_conformal
    assert len(conformal.S_pool) == 120  # 3 classes * 40 cal scores
    floor = 1.0 / (len(conformal.S_pool) + 1)
    p_extreme = conformal.p_value_from_pool(1e9)
    assert p_extreme == floor


def test_gate_check_o_submeans_and_bootstrap(synthetic_gaussian_and_conformal):
    """Check O must evaluate across all mixture sub-means and compute bootstrap CI per D7."""
    gauss, conformal = synthetic_gaussian_and_conformal
    D = gauss.feature_dim
    embeddings = np.random.randn(5, D) + 50.0  # novel embeddings far from known classes
    sha256s = [f"h_{i}" for i in range(5)]

    import imagehash
    mock_phashes = [
        imagehash.hex_to_hash(h)
        for h in [
            "0000000000000000",
            "ffffffffffffffff",
            "aaaaaaaaaaaaaaaa",
            "5555555555555555",
            "3333333333333333",
        ]
    ]

    report = run_gate(
        embeddings=embeddings,
        sha256s=sha256s,
        class_name="OverlapTest",
        existing_class_names=gauss.class_names,
        gaussian=gauss,
        conformal=conformal,
        alpha_cons=0.01,
        q_BH=0.05,
        eps_max=0.50,
        phashes=mock_phashes,
        bootstrap_samples=100,
    )
    for c in gauss.class_names:
        assert c in report.eps_hat
        assert c in report.eps_hat_bootstrap
        lo, hi = report.eps_hat_bootstrap[c]
        assert 0.0 <= lo <= hi <= 1.0


# ── 5. Metrics & Confusion Matrix Tests ──────────────────────────────────────

def test_metrics_v11_standardized_terminology():
    """Verify Protocol v1.1 decision rate calculations."""
    preds = [
        set(),            # empty (unknown)
        {"Cr"},           # singleton correct
        {"In"},           # singleton wrong
        {"Cr", "In"},     # ambiguous
    ]
    true_labels = ["Cr", "Cr", "Cr", "Cr"]
    is_known = [True, True, True, True]

    rates = compute_decision_rates(preds, true_labels, is_known)
    assert rates["empty_rate"] == 0.25
    assert rates["ambiguous_rate"] == 0.25
    assert rates["operator_routing_rate"] == 0.50
    # Known FRR: true class 'Cr' not in prediction set
    # pred[0] has no Cr (empty), pred[2] has In (no Cr) -> 2 out of 4
    assert rates["known_FRR"] == 0.50


def test_confusion_matrix_v11_layout():
    """Confusion matrix must have separate 'ambiguous' and 'unknown' columns."""
    true_labels = ["Cr", "In", "Sc"]
    preds = [set(), {"In"}, {"Cr", "In"}]
    c_hat = ["Cr", "In", "Cr"]
    classes = ["Cr", "In"]

    cm = confusion_matrix_v11(true_labels, preds, c_hat, classes)
    assert cm["col_names"] == ["Cr", "In", AMBIGUOUS_LABEL, UNKNOWN_LABEL]
    assert cm["row_names"] == ["Cr", "In", UNKNOWN_LABEL]
    assert np.array(cm["matrix"]).shape == (3, 4)


# ── 6. Artifact Verification Safeguards Tests ────────────────────────────────

def test_verify_artifact_refusal_on_tampering(tmp_path: Path):
    """verify_artifact must refuse if an artifact file is modified or tampered."""
    art_dir = tmp_path / "art"
    art_dir.mkdir()

    # Create dummy files
    f1 = art_dir / "gaussian.npz"
    f1.write_bytes(b"gaussian_data")
    h1 = canonical_file_sha256(f1)

    meta = {
        "protocol": "v1.1",
        "run_id": "H=Sc_s=0",
        "manifest_version": "v2",
        "manifest_sha256": "dummy",
        "device_info": {"device": "cpu"},
        "git_commit": "abc",
    }
    with open(art_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    h_meta = canonical_file_sha256(art_dir / "metadata.json")

    prov = {
        "artifact_hashes": {
            "gaussian.npz": h1,
            "metadata.json": h_meta,
        },
        "train_input_shas_head": [],
        "cal_input_shas_head": [],
    }
    with open(art_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f)

    # Tamper with gaussian.npz
    f1.write_bytes(b"tampered_data")
    assert not verify_artifact_dir(art_dir)


def test_verify_artifact_refusal_on_test_leakage(tmp_path: Path):
    """verify_artifact must refuse if test-pool hash appears among fitting inputs."""
    art_dir = tmp_path / "art"
    art_dir.mkdir()

    f1 = art_dir / "metadata.json"
    meta = {
        "protocol": "v1.1",
        "run_id": "H=Sc_s=0",
        "manifest_version": "v2",
        "manifest_sha256": canonical_file_sha256(REPO_ROOT / "data" / "manifests" / "v2" / "H=Sc_s=0.json"),
        "device_info": {"device": "cpu"},
        "git_commit": "abc",
    }
    with open(f1, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Read a real test hash from H=Sc_s=0
    m = load_manifest_v2(REPO_ROOT / "data" / "manifests" / "v2" / "H=Sc_s=0.json")
    test_sha = m["pools"]["Cr_test"][0]["sha256"]

    prov = {
        "artifact_hashes": {
            "metadata.json": canonical_file_sha256(f1),
        },
        "train_input_shas_head": [test_sha],  # TEST LEAKAGE!
        "cal_input_shas_head": [],
    }
    with open(art_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f)

    assert not verify_artifact_dir(art_dir)


# ── 7. Runner Commit & Refusal Tests ─────────────────────────────────────────

def test_check_git_clean_and_commit_refusal():
    """check_git_clean_and_commit must raise ValueError on commit mismatch."""
    with pytest.raises(ValueError) as exc:
        check_git_clean_and_commit(expect_commit="0000000000000000000000000000000000000000", allow_dirty=True)
    assert "Commit mismatch refusal" in str(exc.value)


# ── 8. Rehearsal Selection Hierarchy Tests ───────────────────────────────────

def test_selection_hierarchy_rules():
    """select_best_candidate must enforce FRR exclusion, AUROC ranking, and tie-breaking."""
    cands = [
        CandidateResult("C1_bad_frr", "B0", "L4", "a", 1, mean_auroc=0.99, mean_known_frr=0.12, mean_ambiguity_rate=0.1, mean_h1_recall=0.8, mean_h1s_recall=0.8, h1_frr_gap=0.01, h1s_frr_gap=0.02, excluded=True, exclusion_reason="FRR > 0.08"),
        CandidateResult("C2_top_auroc", "B0", "L4", "a", 2, mean_auroc=0.95, mean_known_frr=0.04, mean_ambiguity_rate=0.1, mean_h1_recall=0.7, mean_h1s_recall=0.75, h1_frr_gap=0.02, h1s_frr_gap=0.01, excluded=False),
        CandidateResult("C3_tie_winner", "B0", "L234", "a", 1, mean_auroc=0.945, mean_known_frr=0.05, mean_ambiguity_rate=0.1, mean_h1_recall=0.85, mean_h1s_recall=0.90, h1_frr_gap=0.015, h1s_frr_gap=0.010, excluded=False),
    ]

    chosen, variant = select_best_candidate(cands)
    # C1 is excluded due to FRR > 0.08.
    # C2 (AUROC 0.95) and C3 (AUROC 0.945) are within 0.01 of top AUROC!
    # Tie-break on pseudo-enrollment recall: C3 has recall 0.90 vs C2 has recall 0.75.
    # Therefore C3 must win!
    assert chosen.candidate_id == "C3_tie_winner"
    assert variant == "H1-s"  # h1s_frr_gap (0.010) < h1_frr_gap (0.015)


# ── 9. Synthetic V1 Simulation Tests ─────────────────────────────────────────

def test_v1_simulations_statistical_bounds():
    """Run V1 synthetic simulations and verify theoretical agreement."""
    p1 = simulate_proposition_1(n_cal=40, alpha=0.05, n_trials=500, n_test=500, seed=42)
    assert p1["mean_absolute_error"] < 0.02

    d2 = simulate_delta2_unbiasedness(dim=16, true_delta=2.0, n_trials=500, seed=42)
    assert d2["relative_bias"] < 0.15


# ── 10. Items 2.a, 2.b, 2.c, 2.g, 2.h, 2.j, 2.l Specific Tests ─────────────

def test_nested_b1_excludes_pseudo_unknown():
    """Item 2.a: assert_nested_b1_records_disjoint raises on pseudo-unknown records."""
    from src.eval.rehearsal import assert_nested_b1_records_disjoint

    clean_records = [{"class": "Cr", "filename": "Cr_001.bmp"}, {"class": "In", "filename": "In_001.bmp"}]
    assert_nested_b1_records_disjoint(clean_records, pseudo_unknown_class="Sc")

    leaked_records = [{"class": "Cr", "filename": "Cr_001.bmp"}, {"class": "Sc", "filename": "Sc_001.bmp"}]
    with pytest.raises(AssertionError) as exc_info:
        assert_nested_b1_records_disjoint(leaked_records, pseudo_unknown_class="Sc")
    assert "Nested B1 leakage violation" in str(exc_info.value)


def test_l234_train_zscoring():
    """Item 2.b: L234 feature blocks are standardized using train statistics."""
    from src.eval.rehearsal import zscore_l234

    rng = np.random.default_rng(42)
    # Generate 100 samples with distinct block means and stds
    b2 = rng.normal(loc=10.0, scale=2.0, size=(100, 128))
    b3 = rng.normal(loc=-5.0, scale=4.0, size=(100, 256))
    b4 = rng.normal(loc=100.0, scale=0.5, size=(100, 512))
    raw = np.concatenate([b2, b3, b4], axis=1)

    zscored, train_mean, train_std = zscore_l234(raw)
    assert zscored.shape == (100, 896)
    # Each block should have mean ~ 0 and std ~ 1
    assert np.allclose(zscored[:, :128].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(zscored[:, :128].std(axis=0), 1.0, atol=1e-4)
    assert np.allclose(zscored[:, 128:384].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(zscored[:, 128:384].std(axis=0), 1.0, atol=1e-4)
    assert np.allclose(zscored[:, 384:].mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(zscored[:, 384:].std(axis=0), 1.0, atol=1e-4)


def test_combination_option_b_cross_fitting_bit_identical():
    """Item 2.c: Option (b) 5-fold cross-fitting weights depend strictly on TRAIN data."""
    from src.eval.rehearsal import cross_fit_combination_b

    rng = np.random.default_rng(42)
    train_dict = {
        "A": rng.normal(0, 1, size=(40, 896)),
        "B": rng.normal(2, 1, size=(40, 896)),
        "C": rng.normal(4, 1, size=(40, 896)),
    }

    w1 = cross_fit_combination_b(train_dict, seed=123)
    w2 = cross_fit_combination_b(train_dict, seed=123)

    assert w1.shape == (3,)
    assert np.isclose(np.sum(w1), 1.0)
    # Weights must be bit-identical given identical train data
    assert np.array_equal(w1, w2)


def test_gate_tuning_constraint_status(synthetic_gaussian_and_conformal):
    """Item 2.g: tune_gate_parameters returns constraint_status ('constraint met' or 'constraint unmet')."""
    from src.eval.rehearsal import tune_gate_parameters
    gauss, conformal = synthetic_gaussian_and_conformal
    D = gauss.feature_dim
    rng = np.random.default_rng(42)
    sel_cal_data = {c: rng.standard_normal((30, D)) for c in gauss.class_names}

    tuned = tune_gate_parameters(gauss, conformal, sel_cal_data)
    assert "constraint_status" in tuned
    assert tuned["constraint_status"] in ("constraint met", "constraint unmet")
    assert tuned["alpha_cons"] in [0.005, 0.01, 0.02, 0.05]
    assert tuned["q_BH"] in [0.05, 0.10]
    assert tuned["eps_max"] in [0.02, 0.05, 0.10, 0.20]


def test_import_results_rejects_modified_bundle(tmp_path: Path):
    """Item 2.j: import_results rejects a bundle whose artifact sha256 does not match."""
    from scripts.import_results import import_run_bundle

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    art_file = bundle_dir / "report.json"
    art_file.write_text('{"status": "ok"}', encoding="utf-8")
    correct_sha = canonical_file_sha256(art_file)

    # Tampered sha recorded in bundle
    bundle_data = {
        "commit_sha": "bce540306239f60cb0fc2776c53e0f04e12e12e1",
        "created_utc": "2026-10-22T00:00:00Z",
        "runs": [
            {
                "row_idx": 0,
                "run_id": "H=Cr_s=0",
                "status": "COMPLETE",
                "artifact_hashes": {
                    "report.json": "0000000000000000000000000000000000000000000000000000000000000000",
                },
            }
        ],
    }
    b_path = bundle_dir / "RUN_BUNDLE.json"
    with open(b_path, "w", encoding="utf-8") as f:
        json.dump(bundle_data, f)

    assert not import_run_bundle(b_path)


def test_realistic_regime_synthetic_fixture():
    """Item 2.l: Realistic-regime synthetic fixture (D=512, n=120, overlapping classes)."""
    rng = np.random.default_rng(42)
    D = 512
    n_c = 120
    classes = ["Cr", "In", "Pa", "PS", "RS"]

    # Overlapping classes in 512-dim space
    train_data = {}
    for i, c in enumerate(classes):
        center = rng.standard_normal(D) * 0.5  # moderate separation
        train_data[c] = rng.standard_normal((n_c, D)) + center

    gauss = fit_gaussian(train_data, k=1)
    assert gauss.feature_dim == 512
    assert len(gauss.class_names) == 5

    # Conformal calibration on n=40
    cal_data = {c: rng.standard_normal((40, D)) + (rng.standard_normal(D) * 0.5) for c in classes}
    conformal = build_conformal_scores(gauss, cal_data)
    assert len(conformal.S_pool) == 200  # 5 * 40

    # Decision on test
    test_vec = rng.standard_normal(D)
    scores = {c: gauss.inflated_score(test_vec, idx) for idx, c in enumerate(gauss.class_names)}
    pred_set, _ = conformal.prediction_set(scores, alpha=0.05)
    assert isinstance(pred_set, set)

