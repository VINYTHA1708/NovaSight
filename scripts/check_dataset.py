"""Dataset integrity verification tool (Protocol v1.1).

Verifies that all 1,800 NEU-CLS images exist and match the SHA-256 hashes
recorded in the manifests. Exits with non-zero code on any mismatch or missing file.

Usage:
    python scripts/check_dataset.py --root data/raw/NEU-CLS
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.hashing import sha256_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_dataset")

TOTAL_EXPECTED_IMAGES = 1800
KNOWN_PREFIXES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]


def load_manifest_expected_hashes() -> dict[str, str]:
    """Load filename -> expected SHA-256 mapping from manifests."""
    manifest_paths = list(REPO_ROOT.glob("data/manifests/v2/*.json"))
    if not manifest_paths:
        manifest_paths = list(REPO_ROOT.glob("data/manifests/*.json"))
    if not manifest_paths:
        raise FileNotFoundError("No manifest files found in data/manifests/ or data/manifests/v2/.")

    # Load from the first available manifest (all manifests cover the same 1,800 images)
    expected: dict[str, str] = {}
    with open(manifest_paths[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    for pool_name, records in data["pools"].items():
        for rec in records:
            fname = rec["filename"]
            expected[fname] = rec["sha256"]

    return expected


def check_dataset(root_dir: Path) -> bool:
    """Verify all 1,800 images in root_dir against manifest hashes.

    Args:
        root_dir: Directory containing NEU-CLS images.

    Returns:
        True if all 1,800 images exist and match hashes, False otherwise.
    """
    logger.info("Verifying dataset at: %s", root_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        logger.error("Dataset root directory does not exist: %s", root_dir)
        return False

    expected_hashes = load_manifest_expected_hashes()
    if len(expected_hashes) != TOTAL_EXPECTED_IMAGES:
        logger.error("Expected %d image hashes in manifest, found %d", TOTAL_EXPECTED_IMAGES, len(expected_hashes))
        return False

    missing_files: list[str] = []
    hash_mismatches: list[tuple[str, str, str]] = []
    checked_count = 0

    for filename, expected_hash in sorted(expected_hashes.items()):
        # Try direct path inside root_dir or recursively
        file_path = root_dir / filename
        if not file_path.exists():
            # Check subdirectories
            matches = list(root_dir.rglob(filename))
            if matches:
                file_path = matches[0]
            else:
                missing_files.append(filename)
                continue

        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            hash_mismatches.append((filename, expected_hash, actual_hash))
        checked_count += 1

    if missing_files:
        logger.error("Missing %d / %d images. First missing: %s",
                     len(missing_files), TOTAL_EXPECTED_IMAGES, missing_files[:5])

    if hash_mismatches:
        logger.error("Hash mismatches in %d images. Example: %s (expected %s, got %s)",
                     len(hash_mismatches), hash_mismatches[0][0], hash_mismatches[0][1], hash_mismatches[0][2])

    if not missing_files and not hash_mismatches and checked_count == TOTAL_EXPECTED_IMAGES:
        logger.info("Dataset check SUCCESS: all %d images present and verified against SHA-256 hashes.", TOTAL_EXPECTED_IMAGES)
        return True

    logger.error("Dataset check FAILED: %d checked, %d missing, %d mismatches.",
                 checked_count, len(missing_files), len(hash_mismatches))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Check NEU-CLS dataset integrity.")
    parser.add_argument("--root", type=str, default="data/raw/NEU-CLS",
                        help="Root directory of the NEU-CLS dataset.")
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.is_absolute():
        root_path = REPO_ROOT / root_path

    success = check_dataset(root_path)
    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
