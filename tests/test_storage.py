"""Unit tests for src/app/storage.py (Stage S4)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.app.storage import SQLiteStorage, PostgresStorage, get_storage
from src.runtime.artifact import load_runtime_artifact
from src.runtime.memory import EnrolledClass, RuntimeMemory
from src.runtime.scoring import RuntimeScorer


@pytest.fixture
def sqlite_store(tmp_path):
    """Instantiate an isolated SQLiteStorage in a temp directory."""
    db_file = tmp_path / "test_novasight.db"
    return SQLiteStorage(db_file)


@pytest.fixture
def sample_enrolled_class():
    """Create a sample EnrolledClass object."""
    return EnrolledClass(
        name="test_defect",
        mu=np.ones(512, dtype=np.float64) * 0.5,
        n=5,
        embeddings=[np.ones(512, dtype=np.float64) * 0.5 for _ in range(5)],
        sha256s=["sha1", "sha2", "sha3", "sha4", "sha5"],
        gate_report={"outcome": "ACCEPT", "notes": "All checks passed."},
        inheritance="H1",
        r_hat=1.0,
        version=1,
        confirmed_count=2,
        enrolled_at="2026-10-22T12:00:00Z",
    )


def test_sqlite_class_crud(sqlite_store, sample_enrolled_class):
    """Test saving, retrieving, listing, and deleting an enrolled class."""
    assert sqlite_store.list_classes() == []

    sqlite_store.save_class(sample_enrolled_class)
    retrieved = sqlite_store.get_class("test_defect")
    assert retrieved is not None
    assert retrieved.name == "test_defect"
    assert retrieved.n == 5
    assert np.allclose(retrieved.mu, sample_enrolled_class.mu)
    assert retrieved.sha256s == sample_enrolled_class.sha256s

    classes = sqlite_store.list_classes()
    assert len(classes) == 1
    assert classes[0].name == "test_defect"

    # Delete
    deleted = sqlite_store.delete_class("test_defect")
    assert deleted is True
    assert sqlite_store.get_class("test_defect") is None
    assert sqlite_store.list_classes() == []


def test_schema_migration_idempotence(sqlite_store):
    """Calling init_schema multiple times causes no errors and preserves data."""
    sqlite_store.init_schema()
    sqlite_store.init_schema()

    ev_id = sqlite_store.record_event({"type": "test_event", "val": 123})
    assert ev_id > 0

    sqlite_store.init_schema()
    events = sqlite_store.list_events()
    assert len(events) == 1
    assert events[0]["val"] == 123


def test_inspections_and_review_workflow(sqlite_store):
    """Test recording inspections, listing unreviewed queue, and updating reviews."""
    ins_id = sqlite_store.record_inspection({
        "image_sha256": "abc123sha",
        "verdict": "unknown",
        "class_name": None,
        "novelty_score": 1500.5,
        "prediction_set": [],
        "details": {"test": True},
        "flagged_for_op": True,
        "latency_ms": 750.0,
    })
    assert ins_id > 0

    # Unreviewed queue contains this item
    unreviewed = sqlite_store.list_inspections(unreviewed_only=True)
    assert len(unreviewed) == 1
    assert unreviewed[0]["id"] == ins_id
    assert unreviewed[0]["flagged_for_op"] is True
    assert unreviewed[0]["reviewed"] is False

    # Review inspection
    updated = sqlite_store.update_inspection_review(
        inspection_id=ins_id,
        review_label="scratches",
        review_note="Confirmed scratch defect near weld zone.",
    )
    assert updated is True

    # Queue should now be empty
    assert len(sqlite_store.list_inspections(unreviewed_only=True)) == 0

    # Stats reflect review
    stats = sqlite_store.get_inspection_stats()
    assert stats["total_inspections"] == 1
    assert stats["flagged_count"] == 1
    assert stats["reviewed_count"] == 1
    assert stats["unreviewed_queue_count"] == 0
    assert stats["verdict_distribution"]["unknown"] == 1


def test_rebuild_memory_from_db(sqlite_store, sample_enrolled_class):
    """Rebuilding memory from the database populates RuntimeMemory."""
    sqlite_store.save_class(sample_enrolled_class)

    art = load_runtime_artifact()
    scorer = RuntimeScorer(art)

    memory = sqlite_store.rebuild_memory(scorer, "H1")
    assert "test_defect" in memory.all_class_names
    assert "test_defect" in memory.enrolled
    assert memory.enrolled["test_defect"].n == 5


def test_export_events_jsonl_and_csv(sqlite_store):
    """Test JSONL and CSV event exports."""
    sqlite_store.record_event({
        "uuid": "u-1",
        "timestamp": "2026-10-22T10:00:00Z",
        "version": 0,
        "type": "enroll",
        "class_name": "c1",
        "operator_id": "op1",
        "operator_note": "Justification A",
    })
    sqlite_store.record_event({
        "uuid": "u-2",
        "timestamp": "2026-10-22T11:00:00Z",
        "version": 1,
        "type": "enroll",
        "class_name": "c2",
        "operator_id": "op2",
        "operator_note": "Justification B",
    })

    jsonl_str = sqlite_store.export_events_jsonl()
    lines = jsonl_str.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["class_name"] == "c1"

    csv_str = sqlite_store.export_events_csv()
    assert "uuid,timestamp,version,type,class_name,operator_id,operator_note" in csv_str
    assert "Justification A" in csv_str
    assert "c2" in csv_str


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set; skipping live Postgres test")
def test_postgres_storage_live():
    """Live Postgres test executed when DATABASE_URL is available in environment."""
    url = os.environ["DATABASE_URL"]
    pg_store = PostgresStorage(url)
    assert "Postgres" in pg_store.get_provider_name()

    cls = EnrolledClass(
        name="pg_test_defect",
        mu=np.zeros(512, dtype=np.float64),
        n=5,
        embeddings=[],
        sha256s=["s1", "s2"],
        gate_report={"outcome": "ACCEPT"},
        inheritance="H1",
        r_hat=1.0,
        version=0,
        confirmed_count=0,
        enrolled_at="2026-10-22T00:00:00Z",
    )
    pg_store.save_class(cls)
    retrieved = pg_store.get_class("pg_test_defect")
    assert retrieved is not None
    assert retrieved.name == "pg_test_defect"
    pg_store.delete_class("pg_test_defect")
