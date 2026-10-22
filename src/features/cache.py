"""Feature extraction and caching with multi-element cache keys (Protocol v1.1 Task E4).

Cache key elements:
    (backbone_id, weight_sha256, preprocessing_hash, feature, manifest_version, run_id, pool, class)

Cache path:
    data/processed/feature_cache/{manifest_version}/{backbone_id}_{feature}_{weight_sha256[:8]}_{preprocessing_hash[:8]}_{run_id}_{pool}_{class}.npz
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.config import get_config, REPO_ROOT
from src.data.guard import check_pool_access
from src.data.transforms import get_eval_transform
from src.utils.hashing import sha256_file, sha256_str

logger = logging.getLogger(__name__)


def compute_preprocessing_hash(backbone_id: str) -> str:
    """Compute deterministic hash of the preprocessing pipeline for the backbone."""
    # Preprocessing parameters: resize=224, grayscale->3 channels, normalisation
    cfg = get_config()
    pre_spec = {
        "backbone_id": backbone_id,
        "resize": 224,
        "mode": "RGB_repeated",
        "norm": "imagenet" if backbone_id in ("B0", "B1") else "dinov2",
    }
    import json
    return sha256_str(json.dumps(pre_spec, sort_keys=True))


def get_feature_cache_path(
    backbone_id: str,
    weight_sha256: str,
    preprocessing_hash: str,
    feature: str,
    manifest_version: str,
    run_id: str,
    pool: str,
    class_name: str | None = None,
) -> Path:
    """Build deterministic feature cache path from the 8-tuple key (Task E4)."""
    cfg = get_config()
    cache_root = REPO_ROOT / cfg["paths"]["feature_cache"] / manifest_version
    cls_tag = class_name if class_name else "all"
    fname = (
        f"{backbone_id}_{feature}_{weight_sha256[:8]}_{preprocessing_hash[:8]}_"
        f"{manifest_version}_{run_id}_{pool}_{cls_tag}.npz"
    )
    return cache_root / fname


class RecordDataset(Dataset):
    """Dataset wrapping image records with transforms."""

    def __init__(self, records: list[dict[str, Any]], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        rec = self.records[idx]
        p = Path(rec["path"])
        if not p.is_absolute():
            p = REPO_ROOT / p
        img = Image.open(p).convert("RGB")
        tensor = self.transform(img)
        return tensor, rec["filename"], rec.get("sha256", "")


def extract_features_from_records(
    records: list[dict[str, Any]],
    extractor: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract features from image records."""
    transform = get_eval_transform()
    dataset = RecordDataset(records, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    extractor = extractor.to(device)
    extractor.eval()

    all_feats = []
    with torch.no_grad():
        for tensors, _, _ in loader:
            tensors = tensors.to(device)
            feats = extractor(tensors)
            if isinstance(feats, torch.Tensor):
                all_feats.append(feats.cpu().numpy())
            elif isinstance(feats, dict):
                # If dict of layers, extract default
                val = list(feats.values())[-1]
                all_feats.append(val.cpu().numpy())
    if all_feats:
        return np.concatenate(all_feats, axis=0)
    return np.empty((0, getattr(extractor, "feature_dim", 512)), dtype=np.float32)


def get_or_extract_features(
    records: list[dict[str, Any]],
    extractor: torch.nn.Module,
    backbone_id: str,
    feature: str,
    manifest_version: str,
    run_id: str,
    pool: str,
    class_name: str | None = None,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Load features from cache if valid, otherwise extract and save (with PoolGuard check)."""
    check_pool_access(pool)

    weight_sha256 = getattr(extractor, "weight_sha256", "b0_imagenet")
    prep_hash = compute_preprocessing_hash(backbone_id)
    cache_path = get_feature_cache_path(
        backbone_id=backbone_id,
        weight_sha256=weight_sha256,
        preprocessing_hash=prep_hash,
        feature=feature,
        manifest_version=manifest_version,
        run_id=run_id,
        pool=pool,
        class_name=class_name,
    )

    if cache_path.is_file():
        try:
            data = np.load(cache_path)
            feats = data["features"]
            if len(feats) == len(records):
                return feats
        except Exception as e:
            logger.warning("Cache file %s corrupted, re-extracting: %s", cache_path, e)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feats = extract_features_from_records(records, extractor, device, batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feats,
        filenames=np.array([r["filename"] for r in records]),
        sha256s=np.array([r.get("sha256", "") for r in records]),
    )
    return feats
