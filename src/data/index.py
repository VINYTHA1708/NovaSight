"""NovaSight dataset index builder.

Scans data/raw/NEU-CLS/, builds a filename->class mapping,
verifies counts and readability, and writes reports/sanity_scan.json.

The sanity checks in this module replicate the APPLICATION INPUT checks
(Part 4.1) and are used to SCAN the evaluation pool images, NOT to
filter or drop them. Any image that would trip is reported but stays
in its evaluation pool.

Sanity checks applied here (from approved_policy, Part 4.1):
  - Undecodable file
  - Min side < 32 px
  - Near-constant image (grayscale pixel std < 1.0 on 0-255 scale)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import get_config

logger = logging.getLogger(__name__)

# Canonical prefix -> class name mapping (case-sensitive per spec)
PREFIX_TO_CLASS: dict[str, str] = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches",
}
CLASS_TO_PREFIX: dict[str, str] = {v: k for k, v in PREFIX_TO_CLASS.items()}
KNOWN_PREFIXES = list(PREFIX_TO_CLASS.keys())


def file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file.

    Args:
        path: Absolute path to the file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_filename(name: str) -> str | None:
    """Return the class prefix from a BMP filename, or None if unrecognised.

    Args:
        name: Filename (e.g., "Cr_001.bmp").

    Returns:
        One of the 6 known prefixes, or None.
    """
    for prefix in KNOWN_PREFIXES:
        if name.startswith(prefix + "_"):
            return prefix
    return None


def _check_image_sanity(
    path: Path,
    min_side: int,
    near_constant_std: float,
) -> list[str]:
    """Run application-input sanity checks on a single image.

    These are HEURISTIC checks, not calibrated. They replicate
    what the app would do on an uploaded image.

    Args:
        path: Path to the image file.
        min_side: Minimum allowed side length in pixels.
        near_constant_std: Std threshold (on 0-255 scale) below which
            the image is considered near-constant.

    Returns:
        List of triggered check names (empty = all pass).
    """
    triggered: list[str] = []

    # Check decodability
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        triggered.append("undecodable")
        return triggered  # can't proceed

    # Check size
    w, h = img.size
    if min(w, h) < min_side:
        triggered.append(f"min_side_lt_{min_side}")

    # Check near-constant (grayscale std)
    gray = np.array(img.convert("L"), dtype=np.float32)
    if gray.std() < near_constant_std:
        triggered.append("near_constant")

    return triggered


def build_index(raw_dir: Path | None = None) -> list[dict[str, Any]]:
    """Scan NEU-CLS directory and return a list of image records.

    Each record:
        {
            "path": str (relative to repo root),
            "filename": str,
            "prefix": str,          # one of the 6 known prefixes
            "class_name": str,      # full class name
            "sha256": str,
        }

    Args:
        raw_dir: Path to the NEU-CLS directory. Defaults to config value.

    Returns:
        Sorted list of image records.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If any per-class count is not 300.
    """
    cfg = get_config()
    if raw_dir is None:
        from src.config import REPO_ROOT
        raw_dir = REPO_ROOT / cfg["paths"]["raw_data"]

    if not raw_dir.exists():
        raise FileNotFoundError(f"NEU-CLS directory not found: {raw_dir}")

    records: list[dict[str, Any]] = []
    from src.config import REPO_ROOT

    for fname in sorted(raw_dir.iterdir()):
        if fname.suffix.lower() != ".bmp":
            continue
        prefix = classify_filename(fname.name)
        if prefix is None:
            logger.warning("Unrecognised filename (skipped from index): %s", fname.name)
            continue
        rel_path = fname.relative_to(REPO_ROOT).as_posix()
        records.append(
            {
                "path": rel_path,
                "filename": fname.name,
                "prefix": prefix,
                "class_name": PREFIX_TO_CLASS[prefix],
                "sha256": file_sha256(fname),
            }
        )

    # Validate per-class counts
    for prefix in KNOWN_PREFIXES:
        count = sum(1 for r in records if r["prefix"] == prefix)
        if count != 300:
            raise ValueError(
                f"Expected 300 images for prefix {prefix}, found {count}"
            )
    logger.info("Index built: %d images across %d classes", len(records), len(KNOWN_PREFIXES))
    return records


def run_sanity_scan(
    records: list[dict[str, Any]],
    raw_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run app sanity checks on all NEU images and write reports/sanity_scan.json.

    IMPORTANT: Images that trip a check are REPORTED but NOT removed
    from the index or any evaluation pool. This scan is informational.

    Args:
        records: Image records from build_index().
        raw_dir: Root directory for resolving paths.
        output_path: Where to write the JSON report.

    Returns:
        Sanity scan results dict.
    """
    cfg = get_config()
    sc = cfg["sanity_checks"]
    min_side: int = sc["min_side"]
    near_constant_std: float = sc["near_constant_std_threshold"]

    from src.config import REPO_ROOT
    if raw_dir is None:
        raw_dir = REPO_ROOT

    tripped: list[dict[str, Any]] = []
    for rec in records:
        path = REPO_ROOT / rec["path"]
        checks = _check_image_sanity(path, min_side, near_constant_std)
        if checks:
            tripped.append({"filename": rec["filename"], "checks": checks})

    result: dict[str, Any] = {
        "total_images": len(records),
        "tripped_count": len(tripped),
        "tripped_images": tripped,
        "note": (
            "Images that trip a check are REPORTED here but remain in "
            "their evaluation pool. Sanity checks are NEVER applied to "
            "evaluation data during pipeline runs."
        ),
        "approved_policy": {
            "min_side_px": min_side,
            "near_constant_std_threshold": near_constant_std,
        },
    }

    if output_path is None:
        output_path = REPO_ROOT / "reports" / "sanity_scan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Sanity scan complete: %d/%d images trip a check. Report: %s",
        len(tripped),
        len(records),
        output_path,
    )
    return result
