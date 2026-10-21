"""Script: build the frozen base artifact for the primary run.

Deployment procedure (Part 7):
  1. Load features (must exist from cache_features.py)
  2. Fit Gaussian model on TRAIN features
  3. Calibrate conformal scores on CAL features
  4. Save all to artifacts/base/
  5. Write provenance.json

Pre-declared primary run: held_out=Sc, split_seed=0, model_init_seed=0

Usage:
    .venv\Scripts\python.exe scripts\build_artifact.py [--backbone B0] [--feature L4]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.config import get_config, REPO_ROOT, config_sha256
from src.data.splits import load_manifest, KNOWN_PREFIXES
from src.models.gaussian import fit_gaussian
from src.models.conformal import build_conformal_scores
from src.utils.hashing import sha256_file, write_provenance
from scripts.cache_features import get_cache_path

logger = logging.getLogger(__name__)


def load_pool_features(
    backbone_id: str,
    feature: str,
    held_out: str,
    split_seed: int,
    pool_name: str,
) -> dict[str, np.ndarray]:
    """Load cached features for all known classes in a pool.

    Args:
        backbone_id: Backbone identifier.
        feature: Feature mode string.
        held_out: Held-out class prefix.
        split_seed: Split seed.
        pool_name: Pool name.

    Returns:
        Dict of class_name -> (n, D) feature array.
    """
    from src.data.index import PREFIX_TO_CLASS
    result: dict[str, np.ndarray] = {}
    for prefix in KNOWN_PREFIXES:
        if prefix == held_out:
            continue  # skip held-out class for known-class pools
        cache_path = get_cache_path(backbone_id, feature, held_out, split_seed, pool_name, prefix)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Feature cache missing: {cache_path}. "
                f"Run scripts/cache_features.py first."
            )
        data = np.load(cache_path)
        class_name = PREFIX_TO_CLASS[prefix]
        result[class_name] = data["features"]
    return result


def main() -> None:
    """Build and freeze the primary artifact."""
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Build frozen base artifact")
    parser.add_argument("--backbone", default="B0", choices=["B0", "B1", "B2"])
    parser.add_argument("--feature", default="L4")
    parser.add_argument("--held-out", default="Sc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=1, help="Mixture components (1=plain Gaussian)")
    args = parser.parse_args()

    cfg = get_config()
    held_out = args.held_out
    split_seed = args.seed
    artifact_dir = REPO_ROOT / "artifacts" / "base"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building artifact: backbone=%s, feature=%s, held_out=%s, seed=%d, k=%d",
                args.backbone, args.feature, held_out, split_seed, args.k)
    t0 = time.time()

    # ── Step 2: Load train features and fit Gaussian ──────────────────────────
    logger.info("Step 2: Fitting Gaussian on TRAIN features...")
    train_features = load_pool_features(args.backbone, args.feature, held_out, split_seed, "train")
    gaussian = fit_gaussian(train_features, k=args.k, kmeans_seed=cfg["gaussian"]["kmeans_seed"])
    logger.info("Gaussian fitted: D=%d, K=%d, shrinkage=%.4f, cond=%.2e",
                gaussian.feature_dim, len(gaussian.class_names),
                gaussian.shrinkage_intensity, gaussian.condition_number)

    # ── Step 5: Calibrate conformal on CAL features ───────────────────────────
    logger.info("Step 5: Building conformal calibration on CAL features...")
    cal_features = load_pool_features(args.backbone, args.feature, held_out, split_seed, "cal")
    conformal = build_conformal_scores(
        gaussian, cal_features,
        alpha_default=cfg["conformal"]["alpha_default"],
        alpha_grid=cfg["conformal"]["alpha_grid"],
    )
    logger.info("Conformal calibrated: |S_pool|=%d, tau_hat=%.4f",
                len(conformal.S_pool), conformal.tau_hat)

    # ── Step 6: Freeze artifact ────────────────────────────────────────────────
    logger.info("Step 6: Saving artifact to %s", artifact_dir)

    # Save Gaussian parameters
    gaussian_path = artifact_dir / "gaussian.npz"
    np.savez_compressed(
        gaussian_path,
        class_names=np.array(gaussian.class_names),
        A=gaussian.A,
        n_c=np.array(gaussian.n_c),
        k=gaussian.k,
        feature_dim=gaussian.feature_dim,
        condition_number=gaussian.condition_number,
        shrinkage_intensity=gaussian.shrinkage_intensity,
    )
    # Save class means separately (variable-length structure)
    means_path = artifact_dir / "class_means.npz"
    means_dict = {}
    n_sub_dict = {}
    for i, cname in enumerate(gaussian.class_names):
        means_dict[f"means_{i}"] = gaussian.class_means[i]
        n_sub_dict[f"n_sub_{i}"] = np.array(gaussian.n_sub[i])
    np.savez_compressed(means_path, **means_dict, **n_sub_dict)

    # Save conformal scores
    conformal_path = artifact_dir / "conformal.npz"
    conf_dict = {}
    for cname, scores in conformal.class_scores.items():
        conf_dict[f"scores_{cname}"] = scores
    conf_dict["S_pool"] = conformal.S_pool
    conf_dict["tau_hat"] = np.array([conformal.tau_hat])
    np.savez_compressed(conformal_path, **conf_dict)

    # Write metadata JSON
    meta = {
        "backbone_id": args.backbone,
        "feature": args.feature,
        "held_out": held_out,
        "split_seed": split_seed,
        "k": args.k,
        "class_names": gaussian.class_names,
        "feature_dim": gaussian.feature_dim,
        "n_c": gaussian.n_c,
        "condition_number": float(gaussian.condition_number),
        "shrinkage_intensity": float(gaussian.shrinkage_intensity),
        "tau_hat": float(conformal.tau_hat),
        "alpha_default": conformal.alpha_default,
        "alpha_grid": conformal.alpha_grid,
        "config_sha256": config_sha256(),
    }
    meta_path = artifact_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0

    # Compute artifact hashes for provenance
    artifact_hashes = {
        "gaussian.npz": sha256_file(gaussian_path),
        "class_means.npz": sha256_file(means_path),
        "conformal.npz": sha256_file(conformal_path),
        "metadata.json": sha256_file(meta_path),
    }

    # Write provenance
    prov_path = REPO_ROOT / "artifacts" / "provenance.json"
    write_provenance(
        prov_path,
        stage="S5_build_artifact",
        inputs={
            "train_cache": str(get_cache_path(args.backbone, args.feature, held_out, split_seed, "train", "Cr")),
            "cal_cache": str(get_cache_path(args.backbone, args.feature, held_out, split_seed, "cal", "Cr")),
        },
        outputs=artifact_hashes,
        notes=f"Primary run: held_out={held_out}, seed={split_seed}, backbone={args.backbone}, "
              f"feature={args.feature}, k={args.k}, elapsed={elapsed:.1f}s",
    )

    logger.info("Artifact built in %.1fs", elapsed)
    logger.info("Artifact hashes:")
    for fname, h in artifact_hashes.items():
        logger.info("  %s: %s", fname, h[:16] + "...")
    logger.info("Provenance written to %s", prov_path)


if __name__ == "__main__":
    main()

