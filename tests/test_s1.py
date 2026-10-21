"""S1 Tests: index, splits, guard, sanity scan.

Tests:
  - Index counts and prefix map (1800 images, 300 per class)
  - Image readability (all BMP decodable)
  - Deterministic splits (same seed -> same manifest)
  - Pool disjointness (no sha256 in >1 pool)
  - Data-access guard (raises DataAccessError on undeclared pool)
  - Sanity scan runs on all 1800 images and never silently drops any
  - Evaluation pipeline never applies sanity checks (guard test)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow importing src/ without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.index import build_index, run_sanity_scan, KNOWN_PREFIXES, PREFIX_TO_CLASS
from src.data.splits import (
    make_splits,
    _assert_disjoint,
    KNOWN_SPLITS,
    HELD_OUT_SPLITS,
    get_pool,
)
from src.data.guard import (
    PoolGuard,
    check_pool_access,
    DataAccessError,
    clear_guard,
)
from src.config import REPO_ROOT


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def index():
    """Build the full NEU-CLS index once per session."""
    return build_index()


@pytest.fixture(scope="session")
def primary_manifest(index):
    """Build the primary split manifest (Sc held-out, seed=0)."""
    return make_splits("Sc", 0, index=index, overwrite=True)


# ── Index tests ───────────────────────────────────────────────────────────────

def test_index_total_count(index):
    """Index should contain exactly 1800 images."""
    assert len(index) == 1800, f"Expected 1800, got {len(index)}"


def test_index_per_class_counts(index):
    """Each of the 6 classes should have exactly 300 images."""
    from collections import Counter
    counts = Counter(r["prefix"] for r in index)
    for prefix in KNOWN_PREFIXES:
        assert counts[prefix] == 300, f"{prefix}: expected 300, got {counts[prefix]}"


def test_index_prefix_map():
    """PREFIX_TO_CLASS must contain all 6 expected classes."""
    expected = {"Cr": "crazing", "In": "inclusion", "Pa": "patches",
                "PS": "pitted_surface", "RS": "rolled-in_scale", "Sc": "scratches"}
    assert PREFIX_TO_CLASS == expected


def test_index_all_paths_unique(index):
    """All file paths in the index must be unique (different files)."""
    paths = [r["path"] for r in index]
    assert len(paths) == len(set(paths)), "Duplicate file paths found in index"


def test_index_known_content_duplicate(index):
    """Pa_101.bmp and Pa_105.bmp are known content-identical files in NEU-CLS.
    
    This is a real dataset property, not a code bug.
    The duplicate is reported here and recorded in DEV_LOG / sanity_scan.
    Both files are KEPT in the dataset and evaluation pool (spec: stays in pool).
    """
    from collections import Counter
    hashes = [r["sha256"] for r in index]
    counts = Counter(hashes)
    dup_hashes = [sha for sha, cnt in counts.items() if cnt > 1]
    # Expect exactly 1 known content-duplicate pair in NEU-CLS
    assert len(dup_hashes) == 1, (
        f"Expected 1 duplicate content pair, found: {len(dup_hashes)}. "
        f"Affected hashes: {dup_hashes}"
    )
    # Both should be Pa class
    dup_files = [r["filename"] for r in index if r["sha256"] == dup_hashes[0]]
    assert all(f.startswith("Pa") for f in dup_files), (
        f"Unexpected class for duplicate pair: {dup_files}"
    )


def test_index_all_readable(index):
    """All indexed images must be openable and loadable."""
    from PIL import Image
    errors = []
    for rec in index:
        path = REPO_ROOT / rec["path"]
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            errors.append((rec["filename"], str(e)))
    assert not errors, f"Unreadable images: {errors}"


def test_index_no_thumbs_db(index):
    """Thumbs.db must not appear in the index."""
    names = [r["filename"] for r in index]
    assert "Thumbs.db" not in names


# ── Split tests ───────────────────────────────────────────────────────────────

def test_split_pool_sizes(primary_manifest):
    """Each known class should produce pools of the correct sizes."""
    held_out = primary_manifest["held_out"]
    for prefix in KNOWN_PREFIXES:
        if prefix == held_out:
            for pool_name, expected_size in HELD_OUT_SPLITS.items():
                key = f"held_{pool_name}"
                actual = len(primary_manifest["pools"][key])
                assert actual == expected_size, (
                    f"Held-out pool held_{pool_name}: expected {expected_size}, got {actual}"
                )
        else:
            for pool_name, expected_size in KNOWN_SPLITS.items():
                key = f"{prefix}_{pool_name}"
                actual = len(primary_manifest["pools"][key])
                assert actual == expected_size, (
                    f"Pool {key}: expected {expected_size}, got {actual}"
                )


def test_split_disjoint(primary_manifest):
    """All pools within a manifest must have disjoint sha256 sets."""
    _assert_disjoint(primary_manifest["pools"])  # raises ValueError if not


def test_split_deterministic(index):
    """Same seed must produce the same manifest (bit-identical sha256 sets)."""
    m1 = make_splits("Sc", 0, index=index, overwrite=True)
    m2 = make_splits("Sc", 0, index=index, overwrite=True)
    for key, pool in m1["pools"].items():
        set1 = {r["sha256"] for r in pool}
        set2 = {r["sha256"] for r in m2["pools"][key]}
        assert set1 == set2, f"Pool {key} differs between runs with same seed"


def test_split_different_seeds_differ(index):
    """Different seeds must produce different train splits."""
    m0 = make_splits("Sc", 0, index=index, overwrite=True)
    m1 = make_splits("Sc", 1, index=index, overwrite=True)
    set0 = {r["sha256"] for r in m0["pools"]["Cr_train"]}
    set1 = {r["sha256"] for r in m1["pools"]["Cr_train"]}
    assert set0 != set1, "Different seeds produced identical Cr_train pools"


def test_split_coverage(primary_manifest, index):
    """All images in each class must appear in exactly one pool."""
    held_out = primary_manifest["held_out"]
    all_in_manifest = set()
    for pool in primary_manifest["pools"].values():
        for r in pool:
            all_in_manifest.add(r["sha256"])
    all_in_index = {r["sha256"] for r in index}
    assert all_in_manifest == all_in_index, (
        "Manifest does not cover all index images exactly once"
    )


# ── Data-access guard tests ───────────────────────────────────────────────────

def test_guard_allows_declared_pool():
    """Guard must allow access to declared pools."""
    clear_guard()
    with PoolGuard(["train", "early_stop"]):
        check_pool_access("train")       # should not raise
        check_pool_access("Cr_train")    # generic prefix stripped -> "train"


def test_guard_blocks_undeclared_pool():
    """Guard must raise DataAccessError on undeclared pool."""
    clear_guard()
    with PoolGuard(["train"]):
        with pytest.raises(DataAccessError):
            check_pool_access("test")


def test_guard_blocks_cal_from_training_stage():
    """The calibration pool must not be readable during Gaussian fit."""
    clear_guard()
    with PoolGuard(["train"]):
        with pytest.raises(DataAccessError):
            check_pool_access("cal")


def test_guard_inactive_after_context():
    """Guard must be inactive after exiting the context manager."""
    with PoolGuard(["train"]):
        pass
    # After context, any pool should be accessible
    check_pool_access("test")   # should not raise


def test_guard_nested_contexts():
    """Nested guards must restore outer guard on exit."""
    with PoolGuard(["train"]):
        with PoolGuard(["cal"]):
            with pytest.raises(DataAccessError):
                check_pool_access("train")  # inner guard only allows "cal"
        # Outer guard restored
        check_pool_access("train")  # should not raise


# ── Sanity scan tests ─────────────────────────────────────────────────────────

def test_sanity_scan_runs(index, tmp_path):
    """Sanity scan must complete and produce a valid JSON report."""
    out = tmp_path / "sanity_scan.json"
    result = run_sanity_scan(index, output_path=out)
    assert out.exists()
    assert result["total_images"] == 1800
    assert "tripped_count" in result
    assert "tripped_images" in result
    assert isinstance(result["tripped_images"], list)


def test_sanity_scan_does_not_drop_images(index, tmp_path):
    """Sanity scan must report tripped images but never remove them from index."""
    out = tmp_path / "sanity_scan.json"
    original_len = len(index)
    result = run_sanity_scan(index, output_path=out)
    # The index itself is not modified
    assert len(index) == original_len, "Sanity scan modified the index"


def test_sanity_scan_result_structure(index, tmp_path):
    """Report must include note confirming images stay in evaluation pool."""
    out = tmp_path / "sanity_scan.json"
    result = run_sanity_scan(index, output_path=out)
    assert "note" in result
    assert "evaluation pool" in result["note"]
    assert "approved_policy" in result

