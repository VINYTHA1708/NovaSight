"""Quick script: extract B1 held-out features using fresh fine-tune (same seed/data -> deterministic)."""
import sys, json, numpy as np, torch
sys.path.insert(0, '.')

from src.config import get_config, REPO_ROOT
from src.data.splits import load_manifest, KNOWN_PREFIXES
from src.models.finetune import finetune_b1
from src.models.backbones import ResNetWithHead
from src.utils.logging import setup_logging
from scripts.cache_features import extract_features, get_cache_path

setup_logging("INFO")
cfg = get_config()
held_out = "Sc"
split_seed = 0
init_seed = 0
feature = "L4"

manifest = load_manifest(held_out, split_seed)
known_prefixes = [p for p in KNOWN_PREFIXES if p != held_out]
class_to_idx = {prefix: i for i, prefix in enumerate(known_prefixes)}
num_classes = len(known_prefixes)

train_records = []
early_stop_records = []
for prefix in known_prefixes:
    train_records.extend(manifest["pools"].get(f"{prefix}_train", []))
    early_stop_records.extend(manifest["pools"].get(f"{prefix}_early_stop", []))

import logging
logger = logging.getLogger(__name__)
logger.info("Re-fine-tuning B1 (deterministic, seed=0) to extract held-out features...")
device = torch.device("cpu")
model, result = finetune_b1(
    train_records=train_records, early_stop_records=early_stop_records,
    class_to_idx=class_to_idx, num_classes=num_classes,
    feature=feature, init_seed=init_seed, device=device,
)
logger.info("sha=%s (must match 11c64ccb)", result.weight_sha256[:16])

class _B1Extractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
    def forward(self, x):
        return self._model(x)["features"]

extractor = _B1Extractor(model)
extractor.eval()

heldout_pool_map = {
    "held_enroll_pool": "enroll_pool",
    "held_unk_test": "unk_test",
    "held_new_test": "new_test",
}

for manifest_key, pool_alias in heldout_pool_map.items():
    records = manifest["pools"].get(manifest_key, [])
    if not records:
        logger.warning("Missing: %s", manifest_key)
        continue
    cache_path = get_cache_path("B1", feature, held_out, split_seed, pool_alias, held_out)
    feats = extract_features(records, extractor, "B1", device)
    np.savez_compressed(cache_path, features=feats)
    logger.info("B1 held-out cached: %s (%d imgs, shape=%s)", pool_alias, len(records), feats.shape)

logger.info("Done.")
