"""Unit tests for src/app/stream.py (Stage S6)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.app.storage import SQLiteStorage
from src.app.stream import ProductionStream
from src.config import REPO_ROOT
from src.runtime.artifact import load_runtime_artifact
from src.runtime.onnx_extractor import OnnxFeatureExtractor
from src.runtime.scoring import RuntimeScorer


@pytest.fixture
def stream_setup(tmp_path):
    """Fixture providing isolated SQLiteStorage and ProductionStream."""
    db_file = tmp_path / "test_stream.db"
    store = SQLiteStorage(db_file)
    art = load_runtime_artifact()
    scorer = RuntimeScorer(art)
    ext = OnnxFeatureExtractor(REPO_ROOT / "models" / "onnx" / "resnet18_b0_l4.onnx")

    def get_mem():
        return store.rebuild_memory(scorer, "H1")

    stream = ProductionStream(
        storage=store,
        scorer=scorer,
        extractor=ext,
        get_memory_fn=get_mem,
        speed_fps=10.0,   # Fast rate for unit testing
    )
    yield stream, store
    stream.stop()


def test_stream_state_machine(stream_setup):
    """Start stream -> produce 10 inspections -> pause -> confirm 10 in DB -> resume -> produce 5 more -> stop."""
    stream, store = stream_setup

    # Synchronously process 10 items
    for _ in range(10):
        rec = stream.process_one_image()
        assert "verdict" in rec
        assert "latency_ms" in rec

    stream.pause()
    assert stream.is_paused is True

    # Verify 10 records in DB
    stats = store.get_inspection_stats()
    assert stats["total_inspections"] == 10

    # Resume and produce 5 more
    stream.resume()
    assert stream.is_paused is False

    for _ in range(5):
        stream.process_one_image()

    stats2 = store.get_inspection_stats()
    assert stats2["total_inspections"] == 15

    stream.stop()
    assert stream.is_running is False


def test_defect_burst_injection(stream_setup):
    """Test defect burst injection queue."""
    stream, store = stream_setup

    injected = stream.inject_defect_burst(count=5)
    assert injected == 5

    # Process all 5 burst items
    for _ in range(5):
        rec = stream.process_one_image()
        assert rec["image_filename"].startswith("Sc_")

    stats = store.get_inspection_stats()
    assert stats["total_inspections"] == 5
