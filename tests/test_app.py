"""Tests for app.py: inspect_image and enroll_new_class."""
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
    """Isolate events log to a temporary directory."""
    temp_events = tmp_path / "events.jsonl"
    monkeypatch.setattr(app, "EVENTS_PATH", temp_events)
    yield


def test_inspect_real_neu_image():
    """A real NEU image as a numpy array passes inspect_image and returns a valid verdict."""
    manifest_path = REPO_ROOT / "data" / "manifests" / "H=Sc_s=0.json"
    with open(manifest_path, encoding="utf-8") as f:
        mf = json.load(f)

    # Use first image from Cr_sel_eval
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


def test_enroll_pool_images_enroll():
    """5 images from held_enroll_pool enroll successfully into the memory."""
    manifest_path = REPO_ROOT / "data" / "manifests" / "H=Sc_s=0.json"
    with open(manifest_path, encoding="utf-8") as f:
        mf = json.load(f)

    pool_items = mf["pools"]["held_enroll_pool"][:5]
    assert len(pool_items) == 5, f"Expected 5 items, got {len(pool_items)}"

    images = []
    for item in pool_items:
        img_path = REPO_ROOT / item["path"]
        assert img_path.exists(), f"Image not found at {img_path}"
        images.append(np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8))

    status, gate_text = app.enroll_new_class("Sc_defect", *images)

    assert "enrolled" in status.lower(), f"Enrollment did not report success: {status}"
    assert "Sc_defect" in status
    assert len(gate_text) > 0

    mem = app.get_memory()
    assert "Sc_defect" in mem.all_class_names
