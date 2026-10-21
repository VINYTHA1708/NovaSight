"""Script: fine-tune B1 (ResNet-18, layer4 unfrozen) for a primary run.

Usage:
    .venv/Scripts/python.exe scripts/run_finetune.py [--held-out Sc] [--seed 0] [--init-seed 0]

Outputs:
    data/processed/feature_cache/  (B1 features for all pools after fine-tuning)
    artifacts/base/                 (updated if --build-artifact is set)
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
from src.data.index import build_index, KNOWN_PREFIXES, PREFIX_TO_CLASS
from src.data.splits import load_manifest, get_pool
from src.models.finetune import finetune_b1
from src.models.backbones import _hash_state_dict
from scripts.cache_features import extract_features, get_cache_path

logger = logging.getLogger(__name__)


def collect_pool_records(manifest: dict, held_out: str, pool_name: str) -> list[dict]:
    """Collect all known-class records for a pool.

    Args:
        manifest: Loaded split manifest.
        held_out: Held-out class prefix.
        pool_name: Pool name.

    Returns:
        List of image records with class label.
    """
    records = []
    for prefix in KNOWN_PREFIXES:
        if prefix == held_out:
            continue
        pool_records = manifest["pools"].get(f"{prefix}_{pool_name}", [])
        records.extend(pool_records)
    return records


def main() -> None:
    """Fine-tune B1 for one run."""
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Fine-tune B1 for a primary run")
    parser.add_argument("--held-out", default="Sc")
    parser.add_argument("--seed", type=int, default=0, help="Split seed")
    parser.add_argument("--init-seed", type=int, default=0, help="Model init seed")
    parser.add_argument("--feature", default="L4", choices=["L4", "L234"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    held_out = args.held_out
    split_seed = args.seed
    init_seed = args.init_seed

    # Check if B1 features already cached
    cache_check = get_cache_path("B1", args.feature, held_out, split_seed, "train",
                                  [p for p in KNOWN_PREFIXES if p != held_out][0])
    if cache_check.exists() and not args.overwrite:
        logger.info("B1 features already cached. Use --overwrite to re-run.")
        return

    manifest = load_manifest(held_out, split_seed)

    # Build class mapping
    known_prefixes = [p for p in KNOWN_PREFIXES if p != held_out]
    class_to_idx = {prefix: i for i, prefix in enumerate(known_prefixes)}
    num_classes = len(known_prefixes)

    train_records = collect_pool_records(manifest, held_out, "train")
    early_stop_records = collect_pool_records(manifest, held_out, "early_stop")

    logger.info("Starting B1 fine-tuning: n_train=%d, n_val=%d, classes=%d",
                len(train_records), len(early_stop_records), num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, result = finetune_b1(
        train_records=train_records,
        early_stop_records=early_stop_records,
        class_to_idx=class_to_idx,
        num_classes=num_classes,
        feature=args.feature,
        init_seed=init_seed,
        device=device,
    )

    logger.info(
        "Fine-tune complete: best_epoch=%d, stopped_early=%s, elapsed=%.1fs, sha=%s",
        result.best_epoch, result.stopped_early, result.elapsed_s,
        result.weight_sha256[:16] + "...",
    )

    # Save fine-tune result metadata
    ft_meta_path = REPO_ROOT / "artifacts" / "base" / "b1_finetune_meta.json"
    with open(ft_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "held_out": held_out,
            "split_seed": split_seed,
            "init_seed": init_seed,
            "feature": args.feature,
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
            "elapsed_s": result.elapsed_s,
            "weight_sha256": result.weight_sha256,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
        }, f, indent=2)

    # Extract features with fine-tuned backbone
    # ResNetWithHead is itself a callable that returns a dict; wrap it
    # to return just the feature tensor matching the extract_features API.
    class _B1FeatureExtractor(torch.nn.Module):
        """Thin wrapper: calls ResNetWithHead and returns the feature tensor."""
        def __init__(self, model):
            super().__init__()
            self._model = model

        def forward(self, x):
            out = self._model(x)
            return out["features"]

    extractor = _B1FeatureExtractor(model)
    extractor.eval()

    all_pool_names = ["train", "early_stop", "sel_cal", "sel_eval", "cal", "test"]
    # Held-out pool keys in manifest: held_enroll_pool, held_unk_test, held_new_test
    heldout_pool_map = {
        "held_enroll_pool": "enroll_pool",
        "held_unk_test": "unk_test",
        "held_new_test": "new_test",
    }

    cache_root = REPO_ROOT / cfg["paths"]["feature_cache"]
    cache_root.mkdir(parents=True, exist_ok=True)

    for pool_name in all_pool_names:
        for prefix in KNOWN_PREFIXES:
            if prefix == held_out:
                continue
            records = manifest["pools"].get(f"{prefix}_{pool_name}", [])
            if not records:
                continue
            cache_path = get_cache_path("B1", args.feature, held_out, split_seed, pool_name, prefix)
            if cache_path.exists() and not args.overwrite:
                continue
            feats = extract_features(records, extractor, "B1", device)
            np.savez_compressed(cache_path, features=feats)
            logger.info("B1 features cached: %s/%s %s (%d)", "B1", args.feature,
                        f"{prefix}_{pool_name}", len(records))

    # Held-out pools (Sc enroll, unk, new_test)
    for manifest_key, pool_alias in heldout_pool_map.items():
        records = manifest["pools"].get(manifest_key, [])
        if not records:
            logger.warning("No records for held-out pool key '%s'", manifest_key)
            continue
        cache_path = get_cache_path("B1", args.feature, held_out, split_seed, pool_alias, held_out)
        if cache_path.exists() and not args.overwrite:
            continue
        feats = extract_features(records, extractor, "B1", device)
        np.savez_compressed(cache_path, features=feats)
        logger.info("B1 held-out features cached: %s %d", pool_alias, len(records))

    logger.info("B1 feature caching complete.")


if __name__ == "__main__":
    main()
