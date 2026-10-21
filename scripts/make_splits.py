"""Script: generate all split manifests for E1/E2/E3/E4.

Usage (from repo root):
    .venv\Scripts\python.exe scripts\make_splits.py

Generates manifests in data/manifests/ for:
  - E1/E2 runs: all 6 held-out classes x seeds 0-4 (30 manifests)
  - E4 runs use the same manifests with 2 held-out classes handled at eval time
"""
from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.data.index import build_index, KNOWN_PREFIXES
from src.data.splits import make_splits
from src.config import get_config


def main() -> None:
    """Generate all split manifests."""
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    cfg = get_config()
    seeds = cfg["experiments"]["E2"]["seeds"]

    logger.info("Building index...")
    index = build_index()

    total = 0
    for held_out in KNOWN_PREFIXES:
        for seed in seeds:
            logger.info("Making split: held_out=%s, seed=%d", held_out, seed)
            try:
                manifest = make_splits(held_out, seed, index=index, overwrite=False)
                dup = manifest["near_dup_report"]
                logger.info(
                    "  Pools created; cross-pool near-dups: %d",
                    dup.get("cross_pool_near_dup_count", "N/A"),
                )
                total += 1
            except FileExistsError:
                logger.info("  Already exists, skipping.")
                total += 1

    logger.info("Done. %d manifests created/verified.", total)


if __name__ == "__main__":
    main()
