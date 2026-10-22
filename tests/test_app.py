"""Tests for app.py: inspect_image, safety gate policy, and post-enrollment inspection (Stage S3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import app
from src.config import REPO_ROOT


@pytest.fixture(autouse=True)
def clean_app_events(tmp_path, monkeypatch):
    """Isolate events log and database storage to a temporary directory."""
    temp_events = tmp_path / "events.jsonl"
    temp_db = tmp_path / "test_app.db"
    from src.app.storage import SQLiteStorage
    store = SQLiteStorage(temp_db)
    monkeypatch.setattr(app, "EVENTS_PATH", temp_events)
    monkeypatch.setattr(app, "STORAGE", store)
    yield


def _load_5_sc_images() -> list[np.ndarray]:
    """Helper to load 5 images from held_enroll_pool."""
    manifest_path = REPO_ROOT / "data" / "manifests" / "H=Sc_s=0.json"
    with open(manifest_path, encoding="utf-8") as f:
        mf = json.load(f)

    pool_items = mf["pools"]["held_enroll_pool"][:5]
    images = []
    for item in pool_items:
        img_path = REPO_ROOT / item["path"]
        assert img_path.exists(), f"Image not found at {img_path}"
        images.append(np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8))
    return images


def test_inspect_real_neu_image():
    """A real NEU image as a numpy array passes inspect_image and returns a valid verdict."""
    manifest_path = REPO_ROOT / "data" / "manifests" / "H=Sc_s=0.json"
    with open(manifest_path, encoding="utf-8") as f:
        mf = json.load(f)

    rel_path = mf["pools"]["Cr_sel_eval"][0]["path"]
    img_path = REPO_ROOT / rel_path
    assert img_path.exists(), f"Image not found at {img_path}"

    img_pil = Image.open(img_path).convert("RGB")
    arr = np.array(img_pil, dtype=np.uint8)

    verdict, chart, details_json, flag_op = app.inspect_image(arr, 0.05)

    assert not verdict.startswith("⚠ Rejected:"), f"inspect_image rejected input: {verdict}"
    assert not verdict.startswith("ERROR:"), f"inspect_image returned error: {verdict}"
    assert isinstance(flag_op, bool)

    details = json.loads(details_json)
    assert "prediction_set" in details
    assert "p_values" in details
    assert "novelty_score" in details


def test_reject_safety_rule_blocks_invalid_enrollments():
    """Safety rule blocks: fewer than 5 images, empty name, existing name, and duplicate images."""
    images = _load_5_sc_images()

    # 1. Fewer than 5 images
    status, _ = app.enroll_new_class("test_cls", images[0], images[1], images[2], images[3], None)
    assert "blocked" in status.lower()
    mem = app.get_memory()
    assert "test_cls" not in mem.all_class_names

    # 2. Empty class name
    status, _ = app.enroll_new_class("   ", *images)
    assert "blocked" in status.lower()

    # 3. Existing base class name
    status, _ = app.enroll_new_class("crazing", *images)
    assert "blocked" in status.lower()
    assert "already exists" in status.lower()

    # 4. Duplicate images (same sha256 triggers Gate REJECT)
    dup_images = [images[0]] * 5
    status, gate_text = app.enroll_new_class("dup_cls", *dup_images)
    assert "blocked" in status.lower()
    assert "REJECT" in gate_text
    mem = app.get_memory()
    assert "dup_cls" not in mem.all_class_names


def test_review_safety_rule_requires_operator_justification_note():
    """When gate outcome is REVIEW, note < 10 chars blocks; note >= 10 chars enrolls and records note."""
    images = _load_5_sc_images()
    note_short = "too short"
    note_valid = "Confirmed distinct longitudinal scratches from rolling line 2"

    # Attempt with short note
    status_short, gate_text = app.enroll_new_class(
        "Sc_defect_review", *images, operator_note=note_short
    )
    if "REVIEW" in gate_text:
        assert "required" in status_short.lower()
        mem = app.get_memory()
        assert "Sc_defect_review" not in mem.all_class_names

        # Now enroll with valid note >= 10 chars
        status_ok, gate_text_ok = app.enroll_new_class(
            "Sc_defect_review", *images, operator_note=note_valid
        )
        assert "enrolled" in status_ok.lower()
        mem = app.get_memory()
        assert "Sc_defect_review" in mem.all_class_names
        # Check that note is recorded in gate_report
        assert mem.enrolled["Sc_defect_review"].gate_report.get("operator_note") == note_valid
    else:
        # If gate was ACCEPT, it should have enrolled
        assert "enrolled" in status_short.lower()


def test_sc_enrollment_and_post_enrollment_recognition():
    """Enroll 5 Sc images with review note, then inspect an Sc query image -> recognized as Sc."""
    images = _load_5_sc_images()
    review_note = "Confirmed distinct scratches defect on production line A"

    status, gate_text = app.enroll_new_class("Sc_scratches", *images, operator_note=review_note)
    assert "enrolled" in status.lower(), f"Enrollment failed: {status}"

    mem = app.get_memory()
    assert "Sc_scratches" in mem.all_class_names

    # Inspect query Sc image
    query_path = REPO_ROOT / "examples" / "Sc_192.bmp"
    if not query_path.exists():
        manifest_path = REPO_ROOT / "data" / "manifests" / "H=Sc_s=0.json"
        with open(manifest_path, encoding="utf-8") as f:
            mf = json.load(f)
        query_path = REPO_ROOT / mf["pools"]["held_enroll_pool"][10]["path"]

    img_pil = Image.open(query_path).convert("RGB")
    arr = np.array(img_pil, dtype=np.uint8)

    verdict, chart, details_json, flag_op = app.inspect_image(arr, 0.05)
    details = json.loads(details_json)

    # Conformal prediction set or closed-set label contains Sc_scratches
    recognized = (
        "Sc_scratches" in details.get("prediction_set", [])
        or details.get("closed_set_label") == "Sc_scratches"
        or "Sc_scratches" in verdict
    )
    assert recognized, f"Sc_scratches not recognized in inspection! Verdict: {verdict}\nDetails: {details}"
