"""Script: build the NEU-CLS index and run the sanity scan.

Usage (from repo root):
    .venv\Scripts\python.exe scripts\build_index.py

Outputs:
    reports/sanity_scan.json
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow importing src/ without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.data.index import build_index, run_sanity_scan
from src.config import REPO_ROOT


def main() -> None:
    """Build index and run sanity scan."""
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    logger.info("Building NEU-CLS index...")
    records = build_index()
    logger.info("Index: %d images", len(records))

    # Per-class counts
    from collections import Counter
    counts = Counter(r["prefix"] for r in records)
    for prefix, count in sorted(counts.items()):
        logger.info("  %s: %d", prefix, count)

    logger.info("Running sanity scan (application checks on all %d images)...", len(records))
    result = run_sanity_scan(records)

    if result["tripped_count"] > 0:
        logger.warning(
            "SANITY SCAN: %d image(s) would trip app checks. "
            "They remain in evaluation pools (see reports/sanity_scan.json).",
            result["tripped_count"],
        )
        for item in result["tripped_images"]:
            logger.warning("  %s: %s", item["filename"], item["checks"])
    else:
        logger.info("SANITY SCAN: All %d images pass app checks.", result["total_images"])

    logger.info("Sanity scan written to reports/sanity_scan.json")


if __name__ == "__main__":
    main()
