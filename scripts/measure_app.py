"""Measure app.py memory footprint and verify zero torch/sklearn leaks (Stage S2).

Simulates:
  - 20 inspect calls on sel_eval images
  - 1 5-shot enrollment of Sc (from held_enroll_pool)
  - 5 inspect calls on post-enrollment images (sel_eval + Sc)
Measures process RSS at each step via psutil.Process().memory_info().rss.
Saves profile to reports/app_mvp/app_memory_profile.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psutil
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROCESS = psutil.Process()


def get_rss_mb() -> float:
    """Return current process Resident Set Size in megabytes."""
    return float(PROCESS.memory_info().rss / (1024.0 * 1024.0))


def assert_no_leaks(step_name: str) -> None:
    """Assert that torch, torchvision, and sklearn are absent from sys.modules."""
    assert "torch" not in sys.modules, f"torch leaked at step {step_name}"
    assert "torchvision" not in sys.modules, f"torchvision leaked at step {step_name}"
    assert "sklearn" not in sys.modules, f"sklearn leaked at step {step_name}"


def main() -> None:
    rss_series: list[dict[str, object]] = []
    step_idx = 0

    # 1. Initial RSS before app import
    pre_import_rss = get_rss_mb()
    rss_series.append({"step": step_idx, "label": "pre_import", "rss_mb": round(pre_import_rss, 2)})
    step_idx += 1

    # 2. Import app
    import app
    base_rss_mb = get_rss_mb()
    assert_no_leaks("post_app_import")
    rss_series.append({"step": step_idx, "label": "post_app_import", "rss_mb": round(base_rss_mb, 2)})
    step_idx += 1

    post_model_load_rss_mb = base_rss_mb

    # Load manifests for allowed pools ONLY (sel_eval + held_enroll_pool)
    manifest_path = REPO_ROOT / "data" / "manifests" / "H=Sc_s=0.json"
    with open(manifest_path, encoding="utf-8") as f:
        mf = json.load(f)

    # Gather 20 sel_eval images (4 each from Cr, In, Pa, PS, RS)
    sel_eval_images = []
    for c in ["Cr", "In", "Pa", "PS", "RS"]:
        pool_key = f"{c}_sel_eval"
        for item in mf["pools"][pool_key][:4]:
            sel_eval_images.append(REPO_ROOT / item["path"])

    assert len(sel_eval_images) == 20, f"Expected 20 sel_eval images, got {len(sel_eval_images)}"

    # 3. Simulate 20 inspections
    for i, img_path in enumerate(sel_eval_images):
        pil_img = Image.open(img_path).convert("RGB")
        arr = np.array(pil_img, dtype=np.uint8)
        verdict, chart, details_json, flag_op = app.inspect_image(arr, 0.05)
        assert not verdict.startswith("ERROR:"), f"Inspection error: {verdict}"
        assert_no_leaks(f"inspect_{i+1}")
        current_rss = get_rss_mb()
        rss_series.append({"step": step_idx, "label": f"inspect_{i+1}", "rss_mb": round(current_rss, 2)})
        step_idx += 1

    # 4. Simulate 1 5-shot enrollment of Sc (from held_enroll_pool)
    enroll_items = mf["pools"]["held_enroll_pool"][:5]
    enroll_images = []
    for item in enroll_items:
        p = REPO_ROOT / item["path"]
        pil_img = Image.open(p).convert("RGB")
        enroll_images.append(np.array(pil_img, dtype=np.uint8))

    status, gate_text = app.enroll_new_class("Sc_scratches", *enroll_images, operator_note="Operator verified scratch defect pattern.")
    assert "enrolled" in status.lower(), f"Enrollment failed: {status}"
    assert_no_leaks("post_enrollment")

    post_enroll_rss_mb = get_rss_mb()
    rss_series.append({"step": step_idx, "label": "enroll_Sc_scratches", "rss_mb": round(post_enroll_rss_mb, 2)})
    step_idx += 1

    # 5. Simulate 5 post-enrollment inspections (4 sel_eval + 1 Sc from held_enroll_pool)
    post_images = [sel_eval_images[0], sel_eval_images[5], sel_eval_images[10], sel_eval_images[15]]
    sc_query_path = REPO_ROOT / mf["pools"]["held_enroll_pool"][10]["path"]
    post_images.append(sc_query_path)

    for j, p in enumerate(post_images):
        pil_img = Image.open(p).convert("RGB")
        arr = np.array(pil_img, dtype=np.uint8)
        verdict, chart, details_json, flag_op = app.inspect_image(arr, 0.05)
        assert not verdict.startswith("ERROR:"), f"Post-enroll inspection error: {verdict}"
        assert_no_leaks(f"post_inspect_{j+1}")
        current_rss = get_rss_mb()
        rss_series.append({"step": step_idx, "label": f"post_inspect_{j+1}", "rss_mb": round(current_rss, 2)})
        step_idx += 1

    peak_rss_mb = max(s["rss_mb"] for s in rss_series)

    # 6. Save memory profile
    out_dir = REPO_ROOT / "reports" / "app_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "app_memory_profile.json"

    report = {
        "base_rss_mb": round(base_rss_mb, 2),
        "post_model_load_rss_mb": round(post_model_load_rss_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "post_enroll_rss_mb": round(post_enroll_rss_mb, 2),
        "memory_headroom_512_mb": round(512.0 - peak_rss_mb, 2),
        "torch_leaked": "torch" in sys.modules,
        "torchvision_leaked": "torchvision" in sys.modules,
        "sklearn_leaked": "sklearn" in sys.modules,
        "rss_series": rss_series,
    }

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("NovaSight App Memory Profiler (Stage S2)")
    print("=" * 60)
    print(f"Base RSS (post import):        {report['base_rss_mb']:.2f} MB")
    print(f"Post Model Load RSS:           {report['post_model_load_rss_mb']:.2f} MB")
    print(f"Post 5-Shot Enroll RSS:        {report['post_enroll_rss_mb']:.2f} MB")
    print(f"Peak RSS:                      {report['peak_rss_mb']:.2f} MB")
    print(f"Headroom under 512 MB:         {report['memory_headroom_512_mb']:.2f} MB")
    print(f"Torch / Sklearn leaked:        {report['torch_leaked'] or report['sklearn_leaked']}")
    print(f"Profile saved to:              {profile_path}")
    print("=" * 60)

    if peak_rss_mb > 420.0:
        print(f"CRITICAL: peak_rss_mb ({peak_rss_mb:.2f} MB) exceeds limit (420.0 MB)!")
        sys.exit(1)
    else:
        print("ACCEPTANCE PASSED: Peak RSS <= 420.0 MB and zero leaks.")
        sys.exit(0)


if __name__ == "__main__":
    main()
