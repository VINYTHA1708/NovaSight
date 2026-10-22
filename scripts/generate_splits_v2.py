"""Generate Protocol v1.1 Splits v2 manifests (duplicate-group-aware, hash-disjoint).

Generates:
- 30 single-held-out manifests (E1-E3): 6 classes x 5 seeds (0..4) -> data/manifests/v2/H={held_out}_s={seed}.json
- 150 ordered-pair manifests (E4): 30 ordered pairs x 5 seeds (0..4) -> data/manifests/v2/E4_H1={h1}_H2={h2}_s={seed}.json

Leaves historical v1 manifests in data/manifests/ completely untouched.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import permutations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splits_v2 import (
    KNOWN_PREFIXES,
    generate_single_held_out_manifest,
    generate_e4_pair_manifest,
    load_dataset_records,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_splits_v2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Splits v2 manifests.")
    parser.add_argument("--raw-dir", type=str, default="data/raw/NEU-CLS",
                        help="Path to NEU-CLS raw images.")
    parser.add_argument("--out-dir", type=str, default="data/manifests/v2",
                        help="Output directory for v2 manifests.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Split seeds to generate.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = REPO_ROOT / raw_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset records from %s...", raw_dir)
    records_by_prefix = load_dataset_records(raw_dir)

    # 1. Single held-out manifests (E1-E3)
    logger.info("Generating 30 single-held-out manifests (6 classes x 5 seeds)...")
    single_count = 0
    for held_out in KNOWN_PREFIXES:
        for seed in args.seeds:
            manifest = generate_single_held_out_manifest(records_by_prefix, held_out, seed)
            out_path = out_dir / f"H={held_out}_s={seed}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            single_count += 1

    # 2. Ordered pair manifests (E4)
    logger.info("Generating 150 ordered-pair manifests (30 pairs x 5 seeds)...")
    pair_count = 0
    all_pairs = list(permutations(KNOWN_PREFIXES, 2))
    for h1, h2 in all_pairs:
        for seed in args.seeds:
            manifest = generate_e4_pair_manifest(records_by_prefix, h1, h2, seed)
            out_path = out_dir / f"E4_H1={h1}_H2={h2}_s={seed}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            pair_count += 1

    logger.info("Splits v2 generation complete! Generated %d single and %d pair manifests in %s.",
                single_count, pair_count, out_dir)


if __name__ == "__main__":
    main()
