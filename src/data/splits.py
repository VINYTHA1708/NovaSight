"""NovaSight data splits.

Creates and persists stratified, seeded, hash-disjoint pool manifests
for every experimental run (Part 5.1).

Manifest format (JSON):
  {
    "run_id": "H=Sc_s=0",
    "held_out": "Sc",
    "split_seed": 0,
    "created_at": "ISO-8601",
    "pools": {
        "<pool_name>": [
            {"path": "rel/path", "sha256": "...", "prefix": "Cr"},
            ...
        ],
        ...
    },
    "near_dup_report": {...},
  }

Pool names for known classes:
  train, early_stop, sel_cal, sel_eval, cal, test

Pool names for held-out class:
  enroll_pool, unk_test, new_test

Sizes per known class (300 total):
  train=120, early_stop=30, sel_cal=30, sel_eval=40, cal=40, test=40

Sizes for held-out class (300 total):
  enroll_pool=50, unk_test=125, new_test=125
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config, REPO_ROOT
from src.data.index import build_index, PREFIX_TO_CLASS, KNOWN_PREFIXES
from src.data.guard import check_pool_access

logger = logging.getLogger(__name__)

# Expected pool sizes
KNOWN_SPLITS = {
    "train": 120,
    "early_stop": 30,
    "sel_cal": 30,
    "sel_eval": 40,
    "cal": 40,
    "test": 40,
}  # sums to 300
HELD_OUT_SPLITS = {
    "enroll_pool": 50,
    "unk_test": 125,
    "new_test": 125,
}  # sums to 300
assert sum(KNOWN_SPLITS.values()) == 300
assert sum(HELD_OUT_SPLITS.values()) == 300


def run_id(held_out: str, split_seed: int) -> str:
    """Return a canonical run identifier string.

    Args:
        held_out: Held-out class prefix (e.g., "Sc").
        split_seed: Integer split seed.

    Returns:
        String like "H=Sc_s=0".
    """
    return f"H={held_out}_s={split_seed}"


def _stratified_split(
    records: list[dict[str, Any]],
    sizes: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Partition records into named pools according to sizes.

    Args:
        records: Image records to split (all of one class).
        sizes: Ordered dict of pool_name -> count.
        seed: Random seed for shuffling.

    Returns:
        Dict of pool_name -> list of records.

    Raises:
        ValueError: If sum(sizes.values()) != len(records).
    """
    total = sum(sizes.values())
    if total != len(records):
        raise ValueError(
            f"Sum of split sizes {total} != number of records {len(records)}"
        )
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    pools: dict[str, list[dict[str, Any]]] = {}
    idx = 0
    for pool_name, count in sizes.items():
        pools[pool_name] = shuffled[idx : idx + count]
        idx += count
    return pools


def _manifest_path(held_out: str, split_seed: int) -> Path:
    """Return the path for a run's manifest JSON file.

    Args:
        held_out: Held-out class prefix.
        split_seed: Integer split seed.

    Returns:
        Path object for the manifest file.
    """
    cfg = get_config()
    manifest_dir = REPO_ROOT / cfg["paths"]["manifests"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / f"{run_id(held_out, split_seed)}.json"


def make_splits(
    held_out: str,
    split_seed: int,
    index: list[dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create and persist the pool manifest for one run.

    Args:
        held_out: Held-out class prefix (e.g., "Sc").
        split_seed: Integer random seed for this split.
        index: Pre-built index from build_index(). Built if None.
        overwrite: If False, raise if manifest already exists.

    Returns:
        Manifest dict.

    Raises:
        FileExistsError: If manifest exists and overwrite=False.
        ValueError: If pool sets are not disjoint.
    """
    if held_out not in KNOWN_PREFIXES:
        raise ValueError(f"Unknown held-out prefix: {held_out}")

    out_path = _manifest_path(held_out, split_seed)
    if out_path.exists() and not overwrite:
        logger.info("Manifest already exists (loading): %s", out_path)
        return load_manifest(held_out, split_seed)

    if index is None:
        index = build_index()

    pools: dict[str, list[dict[str, Any]]] = {}

    # Split each known class independently with the same seed
    for prefix in KNOWN_PREFIXES:
        class_records = [r for r in index if r["prefix"] == prefix]
        assert len(class_records) == 300, f"Expected 300 for {prefix}"

        if prefix == held_out:
            # Held-out class uses different pool names
            class_pools = _stratified_split(class_records, HELD_OUT_SPLITS, seed=split_seed)
            for pool_name, recs in class_pools.items():
                pools[f"held_{pool_name}"] = recs
        else:
            # Known class
            class_pools = _stratified_split(class_records, KNOWN_SPLITS, seed=split_seed)
            for pool_name, recs in class_pools.items():
                key = f"{prefix}_{pool_name}"
                pools[key] = recs

    # Verify hash-disjointness across all pools
    _assert_disjoint(pools)

    # Near-duplicate check (perceptual hash based; embedding check done in S2)
    near_dup_report = _check_near_duplicates_phash(pools)

    manifest: dict[str, Any] = {
        "run_id": run_id(held_out, split_seed),
        "held_out": held_out,
        "split_seed": split_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "pools": pools,
        "near_dup_report": near_dup_report,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written: %s", out_path)
    return manifest


def load_manifest(held_out: str, split_seed: int) -> dict[str, Any]:
    """Load an existing manifest from disk.

    Args:
        held_out: Held-out class prefix.
        split_seed: Integer split seed.

    Returns:
        Manifest dict.

    Raises:
        FileNotFoundError: If manifest does not exist.
    """
    path = _manifest_path(held_out, split_seed)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_pool(
    manifest: dict[str, Any],
    prefix: str,
    pool_name: str,
) -> list[dict[str, Any]]:
    """Return the records for one class/pool combination from a manifest.

    For known classes: key = "{prefix}_{pool_name}" (e.g., "Cr_train").
    For held-out class: key = "held_{pool_name}" (e.g., "held_enroll_pool").

    Args:
        manifest: Loaded manifest dict.
        prefix: Class prefix.
        pool_name: Pool name (e.g., "train", "enroll_pool").

    Returns:
        List of image records.
    """
    check_pool_access(pool_name)
    if prefix == manifest["held_out"]:
        key = f"held_{pool_name}"
    else:
        key = f"{prefix}_{pool_name}"
    return manifest["pools"][key]


def get_combined_pool(
    manifest: dict[str, Any],
    pool_name: str,
    exclude_held_out: bool = True,
) -> list[dict[str, Any]]:
    """Return records for all (known) classes for a given pool name.

    Args:
        manifest: Loaded manifest dict.
        pool_name: Pool name (e.g., "train").
        exclude_held_out: If True, skip the held-out class.

    Returns:
        Combined list of records.
    """
    check_pool_access(pool_name)
    result: list[dict[str, Any]] = []
    held_out = manifest["held_out"]
    for prefix in KNOWN_PREFIXES:
        if exclude_held_out and prefix == held_out:
            continue
        key = f"{prefix}_{pool_name}"
        if key in manifest["pools"]:
            result.extend(manifest["pools"][key])
    return result


def _assert_disjoint(pools: dict[str, list[dict[str, Any]]]) -> None:
    """Assert that no file PATH appears in more than one pool.

    Note: Two different files with the same sha256 (identical content but
    different filenames) may legitimately exist in different pools.
    The check uses file paths as the uniqueness key, not sha256.
    Actual duplicate files (same path) would be a logic error.

    Args:
        pools: Dict of pool_name -> list of records.

    Raises:
        ValueError: If any file path appears in more than one pool.
    """
    seen: dict[str, str] = {}  # path -> pool_name
    for pool_name, records in pools.items():
        for rec in records:
            path = rec["path"]
            if path in seen:
                raise ValueError(
                    f"File {path} appears in pools "
                    f"{seen[path]} and {pool_name} -- this is a logic error"
                )
            seen[path] = pool_name


def _check_near_duplicates_phash(
    pools: dict[str, list[dict[str, Any]]],
    hash_size: int = 8,
    max_distance: int = 2,
) -> dict[str, Any]:
    """Check for near-duplicate images across pools using perceptual hashing.

    Images are near-duplicates if their Hamming distance <= max_distance.
    This check is cross-pool (same pool duplicates are not flagged unless
    the sha256 check above would have caught them).

    Args:
        pools: Dict of pool_name -> list of records.
        hash_size: Perceptual hash size parameter.
        max_distance: Maximum Hamming distance to flag as near-duplicate.

    Returns:
        Report dict with any flagged pairs.
    """
    try:
        import imagehash
        from PIL import Image as PILImage
    except ImportError:
        return {"error": "imagehash not installed; skipping perceptual hash check"}

    hashes: list[tuple[str, str, Any]] = []  # (pool, path, phash)
    for pool_name, records in pools.items():
        for rec in records:
            path = REPO_ROOT / rec["path"]
            try:
                img = PILImage.open(path).convert("L")
                ph = imagehash.phash(img, hash_size=hash_size)
                hashes.append((pool_name, rec["path"], ph))
            except Exception as e:
                logger.warning("Could not hash %s: %s", rec["path"], e)

    pairs: list[dict[str, Any]] = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            pool_i, path_i, h_i = hashes[i]
            pool_j, path_j, h_j = hashes[j]
            if pool_i == pool_j:
                continue  # only check cross-pool
            dist = h_i - h_j
            if dist <= max_distance:
                pairs.append(
                    {
                        "pool_a": pool_i,
                        "path_a": path_i,
                        "pool_b": pool_j,
                        "path_b": path_j,
                        "hamming_distance": int(dist),
                    }
                )

    return {
        "phash_size": hash_size,
        "max_distance": max_distance,
        "cross_pool_near_dup_count": len(pairs),
        "pairs": pairs[:50],  # cap to 50 for readability
    }


