"""Verification script for ONNX equivalence against PyTorch model and precomputed caches (Stage S1).

Evaluates the 250 allowed images:
  - 200 known sel_eval images (40 each: Cr, In, Pa, PS, RS)
  - 50 held-out Sc images from held_enroll_pool
(NEVER accesses test, unk_test, new_test, or cal pools per APPROVAL_TEST_POOL_REREPORT=NO).

Verifies:
  a. Preprocessing: max |numpy tensor - torchvision tensor| <= 1e-6
  b. Features: max abs <= 1e-4 and max relative diff ONNX vs live PyTorch extractor
  c. Features: max abs diff ONNX vs cached feature_cache arrays <= 1e-4
  d. Inflated scores: max relative diff runtime vs GaussianModel <= 1e-6
  e. p-values: number of images where any class p-value differs == 0
  f. Prediction sets, verdicts, and closed-set labels mismatches == 0
  g. Timing: median ONNX CPU seconds per image (time.process_time), 1 thread

Writes results to reports/app_mvp/onnx_equivalence.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config
from src.data.transforms import get_eval_transform
from src.models.backbones import ResNetExtractor
from src.models.gaussian import GaussianModel
from src.models.conformal import ConformalScores
from src.models.conformal import make_decision as base_make_decision
from src.runtime.preprocess import preprocess_image
from src.runtime.onnx_extractor import OnnxFeatureExtractor
from src.runtime.artifact import load_runtime_artifact
from src.runtime.scoring import RuntimeScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_allowed_manifest_images() -> list[dict[str, Any]]:
    """Load metadata for the 250 allowed images from H=Sc_s=0 manifest.

    Pools:
      Cr_sel_eval (40), In_sel_eval (40), Pa_sel_eval (40),
      PS_sel_eval (40), RS_sel_eval (40), held_enroll_pool (50).
    """
    manifest_path = REPO_ROOT / "data" / "manifests" / "H=Sc_s=0.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        mf = json.load(f)

    allowed_pools = [
        ("Cr_sel_eval", "Cr", "sel_eval"),
        ("In_sel_eval", "In", "sel_eval"),
        ("Pa_sel_eval", "Pa", "sel_eval"),
        ("PS_sel_eval", "PS", "sel_eval"),
        ("RS_sel_eval", "RS", "sel_eval"),
        ("held_enroll_pool", "Sc", "enroll_pool"),
    ]

    records = []
    for pool_key, prefix, pool_type in allowed_pools:
        items = mf["pools"][pool_key]
        for idx_in_pool, item in enumerate(items):
            records.append({
                "pool_key": pool_key,
                "prefix": prefix,
                "pool_type": pool_type,
                "idx_in_pool": idx_in_pool,
                "filename": item["filename"],
                "path": item["path"],
                "sha256": item["sha256"],
            })

    assert len(records) == 250, f"Expected 250 allowed images, got {len(records)}"
    return records


def run_equivalence_check(
    onnx_path: str | Path = "models/onnx/resnet18_b0_l4.onnx",
    output_json: str | Path = "reports/app_mvp/onnx_equivalence.json",
    forced_mismatch: bool = False,
) -> bool:
    """Run full ONNX equivalence verification against PyTorch and precomputed caches."""
    onnx_path = Path(onnx_path)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing models and runtime scorers...")
    # 1. PyTorch extractor and eval transform
    pt_extractor = ResNetExtractor("B0", "L4")
    pt_extractor.eval()
    tv_transform = get_eval_transform("B0")

    # 2. Base PyTorch/scipy Gaussian & Conformal models from frozen artifact
    art_dir = REPO_ROOT / "artifacts" / "base"
    meta_path = art_dir / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    g_data = np.load(art_dir / "gaussian.npz", allow_pickle=True)
    class_names = [str(c) for c in g_data["class_names"]]
    m_data = np.load(art_dir / "class_means.npz")
    class_means = [m_data[f"means_{i}"] for i in range(len(class_names))]
    n_sub = [list(m_data[f"n_sub_{i}"]) for i in range(len(class_names))]

    base_gaussian = GaussianModel(
        class_names=class_names,
        class_means=class_means,
        n_c=list(g_data["n_c"]),
        n_sub=n_sub,
        A=g_data["A"],
        condition_number=float(g_data["condition_number"]),
        shrinkage_intensity=float(g_data["shrinkage_intensity"]),
        k=int(g_data["k"]),
        feature_dim=int(g_data["feature_dim"]),
    )

    c_data = np.load(art_dir / "conformal.npz")
    base_conformal = ConformalScores(
        class_scores={cn: c_data[f"scores_{cn}"] for cn in class_names},
        S_pool=c_data["S_pool"],
        tau_hat=float(c_data["tau_hat"][0]),
        alpha_default=meta.get("alpha_default", 0.05),
        alpha_grid=meta.get("alpha_grid", [0.01, 0.02, 0.05, 0.10, 0.20]),
    )

    # 3. Pure numpy runtime artifact and scorer
    runtime_artifact = load_runtime_artifact(art_dir, REPO_ROOT / "artifacts" / "provenance.json")
    runtime_scorer = RuntimeScorer(runtime_artifact)

    # 4. ONNX feature extractor (1 thread for deterministic timing)
    onnx_extractor = OnnxFeatureExtractor(onnx_path, num_threads=1)

    # 5. Pre-load cache arrays for comparison
    cache_root = REPO_ROOT / "data" / "processed" / "feature_cache"
    caches = {
        "Cr_sel_eval": np.load(cache_root / "B0_L4_H=Sc_s=0_Cr_sel_eval.npz")["features"],
        "In_sel_eval": np.load(cache_root / "B0_L4_H=Sc_s=0_In_sel_eval.npz")["features"],
        "Pa_sel_eval": np.load(cache_root / "B0_L4_H=Sc_s=0_Pa_sel_eval.npz")["features"],
        "PS_sel_eval": np.load(cache_root / "B0_L4_H=Sc_s=0_PS_sel_eval.npz")["features"],
        "RS_sel_eval": np.load(cache_root / "B0_L4_H=Sc_s=0_RS_sel_eval.npz")["features"],
        "held_enroll_pool": np.load(cache_root / "B0_L4_H=Sc_s=0_Sc_enroll_pool.npz")["features"],
    }

    records = load_allowed_manifest_images()
    logger.info("Evaluating %d allowed images...", len(records))

    max_preprocess_diff = 0.0
    max_feat_abs_diff_pt = 0.0
    max_feat_rel_diff_pt = 0.0
    max_feat_abs_diff_cache = 0.0
    max_score_rel_diff = 0.0
    p_value_diff_count = 0
    pred_set_mismatches = []
    verdict_mismatches = []
    closed_set_mismatches = []
    cpu_timings = []

    for i, rec in enumerate(records):
        img_path = REPO_ROOT / rec["path"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image file missing: {img_path}")

        img = Image.open(img_path)

        # a. Preprocessing comparison
        tv_tensor = tv_transform(img).unsqueeze(0).numpy() # (1, 3, 224, 224)
        np_tensor = preprocess_image(img)                  # (1, 3, 224, 224)
        prep_diff = float(np.max(np.abs(tv_tensor - np_tensor)))
        if prep_diff > max_preprocess_diff:
            max_preprocess_diff = prep_diff

        # b. ONNX inference & timing
        t0 = time.process_time()
        onnx_feat = onnx_extractor.extract(np_tensor)  # (512,) float64
        t1 = time.process_time()
        cpu_timings.append(t1 - t0)

        # PyTorch live feature
        with torch.no_grad():
            pt_feat = pt_extractor(torch.from_numpy(tv_tensor)).squeeze(0).numpy().astype(np.float64)

        feat_diff = np.abs(onnx_feat - pt_feat)
        abs_diff_pt = float(np.max(feat_diff))
        rel_diff_pt = float(np.max(feat_diff / (np.abs(pt_feat) + 1e-12)))
        if abs_diff_pt > max_feat_abs_diff_pt:
            max_feat_abs_diff_pt = abs_diff_pt
        if rel_diff_pt > max_feat_rel_diff_pt:
            max_feat_rel_diff_pt = rel_diff_pt

        # c. Compare against precomputed cache
        cached_feat = caches[rec["pool_key"]][rec["idx_in_pool"]].astype(np.float64)
        cache_diff = float(np.max(np.abs(onnx_feat - cached_feat)))
        if cache_diff > max_feat_abs_diff_cache:
            max_feat_abs_diff_cache = cache_diff

        # d. Inflated score comparison: RuntimeScorer vs GaussianModel on the same ONNX feature
        base_scores = {c: float(base_gaussian.inflated_score(onnx_feat, ci))
                       for ci, c in enumerate(base_gaussian.class_names)}
        base_c_hat, _ = base_gaussian.closed_set_label(onnx_feat)
        base_p_vals = base_conformal.all_p_values(base_scores)
        base_pred_set, _ = base_conformal.prediction_set(base_scores, alpha=0.05)
        base_decision = base_make_decision(base_pred_set, base_c_hat)

        # Runtime scorer
        rt_result = runtime_scorer.predict(onnx_feat, alpha=0.05)
        rt_scores = rt_result["per_class_dt"]
        rt_c_hat = rt_result["closed_set_label"]
        rt_p_vals = rt_result["p_values"]
        rt_pred_set = set(rt_result["prediction_set"])
        rt_verdict = rt_result["verdict"]

        if forced_mismatch and i == 0:
            rt_verdict = "forced_mismatch_verdict"

        # Compare inflated scores relative difference (runtime vs GaussianModel)
        for c in class_names:
            bs = base_scores[c]
            rs = rt_scores[c]
            s_rel = abs(bs - rs) / (abs(bs) + 1e-12)
            if s_rel > max_score_rel_diff:
                max_score_rel_diff = s_rel

            # e. p-value check
            bp = base_p_vals[c]
            rp = rt_p_vals[c]
            if abs(bp - rp) > 1e-9:
                p_value_diff_count += 1
                logger.warning("P-value mismatch for %s on class %s: base=%.6f, rt=%.6f",
                               rec["filename"], c, bp, rp)

        # f. Prediction sets, verdicts, and closed_set_label
        if base_pred_set != rt_pred_set:
            pred_set_mismatches.append(rec["filename"])
        if base_decision["verdict"] != rt_verdict:
            verdict_mismatches.append(rec["filename"])
        if base_c_hat != rt_c_hat:
            closed_set_mismatches.append(rec["filename"])

    median_cpu_sec = float(np.median(cpu_timings))

    # Pass / Fail criteria
    pass_a = (max_preprocess_diff <= 1e-6)
    pass_b = (max_feat_abs_diff_pt <= 1e-4)
    pass_c = (max_feat_abs_diff_cache <= 1e-4)
    pass_d = (max_score_rel_diff <= 1e-6)
    pass_e = (p_value_diff_count == 0)
    pass_f = (len(pred_set_mismatches) == 0 and len(verdict_mismatches) == 0 and len(closed_set_mismatches) == 0)
    all_passed = pass_a and pass_b and pass_c and pass_d and pass_e and pass_f

    results = {
        "n_evaluated": len(records),
        "onnx_path": str(onnx_path),
        "passed": all_passed,
        "checks": {
            "a_preprocessing_max_diff": {
                "value": max_preprocess_diff,
                "threshold": 1e-6,
                "passed": pass_a,
            },
            "b_features_onnx_vs_pytorch": {
                "max_abs_diff": max_feat_abs_diff_pt,
                "max_rel_diff": max_feat_rel_diff_pt,
                "abs_threshold": 1e-4,
                "passed": pass_b,
            },
            "c_features_onnx_vs_cache": {
                "max_abs_diff": max_feat_abs_diff_cache,
                "threshold": 1e-4,
                "passed": pass_c,
            },
            "d_inflated_scores_max_rel_diff": {
                "value": max_score_rel_diff,
                "threshold": 1e-6,
                "passed": pass_d,
            },
            "e_p_value_diff_count": {
                "count": p_value_diff_count,
                "expected": 0,
                "passed": pass_e,
            },
            "f_decision_mismatches": {
                "prediction_set_mismatches": pred_set_mismatches,
                "verdict_mismatches": verdict_mismatches,
                "closed_set_mismatches": closed_set_mismatches,
                "total_mismatch_images": len(set(pred_set_mismatches + verdict_mismatches + closed_set_mismatches)),
                "expected": 0,
                "passed": pass_f,
            },
            "g_timing_cpu_seconds": {
                "median_cpu_sec": median_cpu_sec,
                "threads": 1,
            },
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("       NOVASIGHT ONNX RUNTIME EQUIVALENCE REPORT (STAGE S1)       ")
    print("=" * 70)
    print(f"Evaluated Images       : {len(records)} (200 sel_eval + 50 held_enroll_pool)")
    print(f"a. Preprocessing Diff  : {max_preprocess_diff:.2e}  (Threshold <= 1e-6) -> {'PASS' if pass_a else 'FAIL'}")
    print(f"b. Live Extractor Diff : abs={max_feat_abs_diff_pt:.2e}, rel={max_feat_rel_diff_pt:.2e} (Threshold <= 1e-4) -> {'PASS' if pass_b else 'FAIL'}")
    print(f"c. Feature Cache Diff  : {max_feat_abs_diff_cache:.2e}  (Threshold <= 1e-4) -> {'PASS' if pass_c else 'FAIL'}")
    print(f"d. Inflated Score Diff : {max_score_rel_diff:.2e}  (Threshold <= 1e-6) -> {'PASS' if pass_d else 'FAIL'}")
    print(f"e. P-value Mismatches  : {p_value_diff_count}       (Threshold == 0)    -> {'PASS' if pass_e else 'FAIL'}")
    print(f"f. Verdict Mismatches  : {results['checks']['f_decision_mismatches']['total_mismatch_images']}       (Threshold == 0)    -> {'PASS' if pass_f else 'FAIL'}")
    print(f"g. Median CPU Latency  : {median_cpu_sec * 1000.0:.2f} ms/image (1 thread)")
    print("-" * 70)
    print(f"OVERALL RESULT         : {'PASS' if all_passed else 'FAIL'}")
    print(f"Saved Report           : {output_json}")
    print("=" * 70 + "\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Check ONNX runtime equivalence against PyTorch.")
    parser.add_argument("--onnx-path", default="models/onnx/resnet18_b0_l4.onnx", help="Path to ONNX model.")
    parser.add_argument("--out", default="reports/app_mvp/onnx_equivalence.json", help="Path to output JSON report.")
    parser.add_argument("--forced-mismatch", action="store_true", help="Force a mismatch for test verification.")
    args = parser.parse_args()

    passed = run_equivalence_check(
        onnx_path=args.onnx_path,
        output_json=args.out,
        forced_mismatch=args.forced_mismatch,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
