"""NovaSight Protocol v1.1 Dataset Partitioning (Splits v2).

Enforces:
1. Duplicate-group-awareness (e.g., Pa_101.bmp and Pa_105.bmp stay in the same pool).
2. Strict hash-disjointness across all pools.
3. Deterministic seeding (split_seed).
4. Full support for single-held-out (E1-E3) and ordered-pair (E4) manifests.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.hashing import sha256_file

logger = logging.getLogger(__name__)

KNOWN_PREFIXES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
PREFIX_TO_CLASS = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches",
}

# Pool partition sizes
KNOWN_POOL_SIZES = {
    "train": 120,
    "early_stop": 30,
    "sel_cal": 30,
    "sel_eval": 40,
    "cal": 40,
    "test": 40,
}
HELD_POOL_SIZES = {
    "enroll_pool": 50,
    "unk_test": 125,
    "new_test": 125,
}


def load_dataset_records(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Scan raw directory and group image records by prefix.

    Args:
        raw_dir: Path to directory containing NEU-CLS images.

    Returns:
        Dict of prefix -> list of image records.
    """
    records_by_prefix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_files = sorted(list(raw_dir.glob("*.bmp")) + list(raw_dir.rglob("*.bmp")))
    seen_files = set()

    for p in image_files:
        if p.name in seen_files:
            continue
        seen_files.add(p.name)
        prefix = p.name.split("_")[0]
        if prefix not in PREFIX_TO_CLASS:
            continue
        rel_path = f"data/raw/NEU-CLS/{p.name}"
        records_by_prefix[prefix].append({
            "path": rel_path,
            "filename": p.name,
            "prefix": prefix,
            "class_name": PREFIX_TO_CLASS[prefix],
            "sha256": sha256_file(p),
        })

    for prefix in KNOWN_PREFIXES:
        recs = records_by_prefix.get(prefix, [])
        if len(recs) != 300:
            raise ValueError(f"Prefix {prefix} expected 300 images, found {len(recs)}")

    return records_by_prefix


def partition_prefix_records(
    records: list[dict[str, Any]],
    pool_sizes: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Partition records into pools respecting duplicate groups and target sizes.

    Args:
        records: 300 records for one class.
        pool_sizes: Dict of pool_name -> target size.
        seed: Deterministic random seed.

    Returns:
        Dict of pool_name -> list of records.
    """
    # Group records by SHA-256
    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        hash_groups[r["sha256"]].append(r)

    # Sort groups deterministically by hash
    sorted_hashes = sorted(hash_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(sorted_hashes)

    # Separate multi-item groups and singleton groups
    multi_groups = [h for h in sorted_hashes if len(hash_groups[h]) > 1]
    single_groups = [h for h in sorted_hashes if len(hash_groups[h]) == 1]

    pools: dict[str, list[dict[str, Any]]] = {p: [] for p in pool_sizes}
    remaining_capacity = dict(pool_sizes)

    # Place multi-item groups first into pools that have enough capacity
    for h in multi_groups:
        items = hash_groups[h]
        group_size = len(items)
        # Find first pool that can accommodate this entire group
        placed = False
        for p in pool_sizes:
            if remaining_capacity[p] >= group_size:
                pools[p].extend(items)
                remaining_capacity[p] -= group_size
                placed = True
                break
        if not placed:
            raise RuntimeError(f"Could not place duplicate group of size {group_size} into any pool.")

    # Place singletons into remaining capacities
    single_idx = 0
    for p, target_size in pool_sizes.items():
        needed = remaining_capacity[p]
        for _ in range(needed):
            h = single_groups[single_idx]
            pools[p].extend(hash_groups[h])
            single_idx += 1
            remaining_capacity[p] -= 1

    # Sort each pool deterministically by filename
    for p in pools:
        pools[p].sort(key=lambda x: x["filename"])
        if len(pools[p]) != pool_sizes[p]:
            raise ValueError(f"Pool {p} expected {pool_sizes[p]} items, got {len(pools[p])}")

    return pools


def assert_manifest_hash_disjoint(manifest: dict[str, Any]) -> None:
    """Assert that no image SHA-256 appears in more than one pool."""
    seen_hashes: dict[str, str] = {}
    for pool_name, records in manifest["pools"].items():
        for r in records:
            h = r["sha256"]
            if h in seen_hashes:
                prev_pool = seen_hashes[h]
                if prev_pool != pool_name:
                    raise ValueError(
                        f"Hash {h[:12]}... (file {r['filename']}) appears in both '{prev_pool}' and '{pool_name}'! "
                        f"Protocol v1.1 requires strictly hash-disjoint pools."
                    )
            seen_hashes[h] = pool_name


def generate_single_held_out_manifest(
    records_by_prefix: dict[str, list[dict[str, Any]]],
    held_out: str,
    split_seed: int,
) -> dict[str, Any]:
    """Generate manifest for single-held-out runs (E1-E3)."""
    manifest_pools: dict[str, list[dict[str, Any]]] = {}

    for prefix in KNOWN_PREFIXES:
        if prefix == held_out:
            prefix_pools = partition_prefix_records(
                records_by_prefix[prefix],
                HELD_POOL_SIZES,
                seed=split_seed * 1000 + 999,
            )
            for p, recs in prefix_pools.items():
                manifest_pools[f"held_{p}"] = recs
        else:
            p_seed = split_seed * 1000 + KNOWN_PREFIXES.index(prefix)
            prefix_pools = partition_prefix_records(
                records_by_prefix[prefix],
                KNOWN_POOL_SIZES,
                seed=p_seed,
            )
            for p, recs in prefix_pools.items():
                manifest_pools[f"{prefix}_{p}"] = recs

    manifest = {
        "manifest_version": "v2",
        "run_id": f"H={held_out}_s={split_seed}",
        "held_out": held_out,
        "split_seed": split_seed,
        "pool_sizes": {p: len(recs) for p, recs in manifest_pools.items()},
        "pools": manifest_pools,
    }
    assert_manifest_hash_disjoint(manifest)
    return manifest


def generate_e4_pair_manifest(
    records_by_prefix: dict[str, list[dict[str, Any]]],
    h1: str,
    h2: str,
    split_seed: int,
) -> dict[str, Any]:
    """Generate manifest for E4 two-unknown pair experiments (h1, h2)."""
    manifest_pools: dict[str, list[dict[str, Any]]] = {}

    for prefix in KNOWN_PREFIXES:
        if prefix in (h1, h2):
            h_idx = 1 if prefix == h1 else 2
            prefix_pools = partition_prefix_records(
                records_by_prefix[prefix],
                HELD_POOL_SIZES,
                seed=split_seed * 2000 + h_idx * 100,
            )
            tag = "h1" if prefix == h1 else "h2"
            for p, recs in prefix_pools.items():
                manifest_pools[f"{tag}_{p}"] = recs
        else:
            p_seed = split_seed * 1000 + KNOWN_PREFIXES.index(prefix)
            prefix_pools = partition_prefix_records(
                records_by_prefix[prefix],
                KNOWN_POOL_SIZES,
                seed=p_seed,
            )
            for p, recs in prefix_pools.items():
                manifest_pools[f"{prefix}_{p}"] = recs

    manifest = {
        "manifest_version": "v2",
        "run_id": f"E4_H1={h1}_H2={h2}_s={split_seed}",
        "experiment": "E4",
        "h1": h1,
        "h2": h2,
        "split_seed": split_seed,
        "pool_sizes": {p: len(recs) for p, recs in manifest_pools.items()},
        "pools": manifest_pools,
    }
    assert_manifest_hash_disjoint(manifest)
    return manifest


def load_manifest_v2(path_or_run_id: str | Path) -> dict[str, Any]:
    """Load a Splits v2 manifest from JSON file."""
    path = Path(path_or_run_id)
    if not path.is_file():
        # Try finding in data/manifests/v2/
        candidate = Path("data/manifests/v2") / f"{path_or_run_id}.json"
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Manifest not found: {path_or_run_id}")
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert_manifest_hash_disjoint(manifest)
    return manifest


def get_pool_v2(
    manifest: dict[str, Any],
    prefix: str,
    pool_name: str,
) -> list[dict[str, Any]]:
    """Return records for one class/pool combination with PoolGuard check."""
    from src.data.guard import check_pool_access
    check_pool_access(pool_name)

    if "held_out" in manifest and prefix == manifest["held_out"]:
        key = f"held_{pool_name}"
    elif "h1" in manifest and prefix == manifest["h1"]:
        key = f"h1_{pool_name}"
    elif "h2" in manifest and prefix == manifest["h2"]:
        key = f"h2_{pool_name}"
    else:
        key = f"{prefix}_{pool_name}"

    if key not in manifest["pools"]:
        raise KeyError(f"Pool key '{key}' not found in manifest pools.")
    return manifest["pools"][key]


def get_combined_pool_v2(
    manifest: dict[str, Any],
    pool_name: str,
    exclude_held_out: bool = True,
) -> list[dict[str, Any]]:
    """Return combined records for all known classes for a given pool name with PoolGuard check."""
    from src.data.guard import check_pool_access
    check_pool_access(pool_name)

    result: list[dict[str, Any]] = []
    held_out = manifest.get("held_out")
    h1 = manifest.get("h1")
    h2 = manifest.get("h2")

    for prefix in KNOWN_PREFIXES:
        if exclude_held_out:
            if held_out and prefix == held_out:
                continue
            if h1 and h2 and prefix in (h1, h2):
                continue
        key = f"{prefix}_{pool_name}"
        if key in manifest["pools"]:
            result.extend(manifest["pools"][key])
    return result

