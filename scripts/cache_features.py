"""Script: extract and cache features for all backbones and pools.

Usage (from repo root):
    .venv\Scripts\python.exe scripts\cache_features.py [--backbone B0] [--held-out Sc] [--seed 0]

Features are cached under data/processed/feature_cache/
keyed by (backbone_id, feature_mode, preprocessing_hash, pool_name, run_id).

For B1, features are extracted from the FINE-TUNED model saved after
the build_artifact step. During rehearsal, this script is called
after fine-tuning; during evaluation it uses frozen artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.config import get_config, REPO_ROOT
from src.data.splits import load_manifest, get_combined_pool, get_pool, KNOWN_PREFIXES
from src.data.transforms import get_eval_transform
from src.models.backbones import make_extractor
from src.utils.hashing import sha256_str

logger = logging.getLogger(__name__)


def get_cache_path(
    backbone_id: str,
    feature: str,
    held_out: str,
    split_seed: int,
    pool_name: str,
    prefix: str | None = None,
) -> Path:
    """Build a cache file path.

    Args:
        backbone_id: "B0", "B1", or "B2".
        feature: Feature mode string.
        held_out: Held-out class prefix.
        split_seed: Split seed.
        pool_name: Pool name (e.g., "train").
        prefix: Class prefix, or None for combined pools.

    Returns:
        Path to the .npz cache file.
    """
    cfg = get_config()
    cache_root = REPO_ROOT / cfg["paths"]["feature_cache"]
    run_key = f"H={held_out}_s={split_seed}"
    pool_key = f"{prefix}_{pool_name}" if prefix else pool_name
    name = f"{backbone_id}_{feature}_{run_key}_{pool_key}.npz"
    return cache_root / name


def extract_features(
    records: list[dict],
    extractor,
    backbone_id: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract features from a list of image records.

    Args:
        records: Image records with "path" field.
        extractor: Feature extractor module.
        backbone_id: Used to select the right transform.
        device: Torch device.
        batch_size: Inference batch size.

    Returns:
        (N, D) feature array.
    """
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from src.data.transforms import get_eval_transform

    transform = get_eval_transform(backbone_id)

    class SimpleDS:
        def __init__(self, records):
            self.records = records
        def __len__(self):
            return len(self.records)
        def __getitem__(self, i):
            path = REPO_ROOT / self.records[i]["path"]
            img = Image.open(path).convert("L")
            return transform(img)

    loader = DataLoader(SimpleDS(records), batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats = []
    extractor.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = extractor(batch)
            all_feats.append(feats.cpu().numpy())
    return np.vstack(all_feats)


def cache_pool_features(
    backbone_id: str,
    feature: str,
    held_out: str,
    split_seed: int,
    pool_names: list[str],
    extractor,
    device: torch.device,
    overwrite: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract and cache features for given pools.

    Args:
        backbone_id: Backbone identifier.
        feature: Feature mode string.
        held_out: Held-out class prefix.
        split_seed: Split seed.
        pool_names: List of pool names to process.
        extractor: Feature extractor.
        device: Torch device.
        overwrite: If True, re-extract even if cache exists.

    Returns:
        Dict of pool_name -> {prefix -> features array}.
    """
    manifest = load_manifest(held_out, split_seed)
    cache_root = REPO_ROOT / get_config()["paths"]["feature_cache"]
    cache_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, np.ndarray]] = {}

    for pool_name in pool_names:
        result[pool_name] = {}
        for prefix in KNOWN_PREFIXES:
            if prefix == held_out and pool_name not in ("enroll_pool", "unk_test", "new_test"):
                continue
            cache_path = get_cache_path(backbone_id, feature, held_out, split_seed, pool_name, prefix)
            if cache_path.exists() and not overwrite:
                data = np.load(cache_path)
                result[pool_name][prefix] = data["features"]
                logger.debug("Loaded from cache: %s", cache_path.name)
                continue

            # Determine which pool key to use
            if prefix == held_out:
                records = get_pool(manifest, prefix, pool_name)
            else:
                if pool_name in ("enroll_pool", "unk_test", "new_test"):
                    continue  # these are only for held-out class
                try:
                    records = get_pool(manifest, prefix, pool_name)
                except KeyError:
                    continue

            if not records:
                continue

            logger.info("Extracting %s/%s features: %s pool, %d images",
                        backbone_id, feature, f"{prefix}_{pool_name}", len(records))
            feats = extract_features(records, extractor, backbone_id, device)
            np.savez_compressed(cache_path, features=feats)
            result[pool_name][prefix] = feats

    return result


def main() -> None:
    """Extract features for one run configuration."""
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Cache backbone features")
    parser.add_argument("--backbone", default="B0", choices=["B0", "B1", "B2"])
    parser.add_argument("--feature", default="L4")
    parser.add_argument("--held-out", default="Sc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    extractor = make_extractor(args.backbone, args.feature)
    extractor.to(device)
    extractor.eval()

    pools = ["train", "early_stop", "sel_cal", "sel_eval", "cal", "test"]
    heldout_pools = ["enroll_pool", "unk_test", "new_test"]

    cache_pool_features(
        args.backbone, args.feature, args.held_out, args.seed,
        pools + heldout_pools, extractor, device, overwrite=args.overwrite,
    )
    logger.info("Feature caching complete.")


if __name__ == "__main__":
    main()

