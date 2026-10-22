"""Script: build and freeze artifact under Protocol v1.1 (Task E8).

Writes to artifacts/runs/{protocol}/{run_id}/ (NEVER artifacts/base/).
Saves:
  - gaussian.npz
  - class_means.npz
  - conformal.npz
  - metadata.json
  - provenance.json

Usage:
    .venv\\Scripts\\python.exe scripts\\build_artifact_v11.py --held-out Sc --seed 0 --device auto
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, REPO_ROOT
from src.data.guard import PoolGuard
from src.data.index import KNOWN_PREFIXES, PREFIX_TO_CLASS, CLASS_TO_PREFIX
from src.data.splits_v2 import load_manifest_v2, get_pool_v2
from src.features.cache import get_or_extract_features
from src.models.backbones import make_extractor
from src.models.conformal import build_conformal_scores
from src.models.gaussian import fit_gaussian
from src.utils.device import get_device_info, setup_device
from src.utils.hashing import canonical_file_sha256, canonical_json_sha256, sha256_file
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_artifact_for_run(
    held_out: str,
    split_seed: int,
    protocol: str = "v1.1",
    manifest_version: str = "v2",
    device_arg: str = "auto",
    custom_out_dir: Path | None = None,
) -> Path:
    """Build and freeze model artifact for a specified run."""
    device = setup_device(device_arg)
    dev_info = get_device_info(device)

    run_id = f"H={held_out}_s={split_seed}"
    manifest_path = REPO_ROOT / "data" / "manifests" / manifest_version / f"{run_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_sha = canonical_file_sha256(manifest_path)
    manifest = load_manifest_v2(manifest_path)

    # Check selection result
    selection_path = REPO_ROOT / "reports" / protocol / f"selection_{run_id}.json"
    if not selection_path.exists():
        # Fallback default configuration if selection file not yet generated
        chosen_config = {"backbone": "B0", "feature": "L4", "combination": "a", "k": 1}
        inheritance_variant = "H1"
        gate_params = {"alpha_cons": 0.01, "q_BH": 0.05, "eps_max": 0.10}
        selection_sha = "unselected_default"
    else:
        selection_sha = canonical_file_sha256(selection_path)
        with open(selection_path, encoding="utf-8") as f:
            sel_data = json.load(f)
        chosen_config = sel_data["chosen_config"]
        inheritance_variant = sel_data.get("inheritance_variant", "H1")
        gate_params = sel_data.get("gate_params", {"alpha_cons": 0.01, "q_BH": 0.05, "eps_max": 0.10})

    if custom_out_dir:
        artifact_dir = custom_out_dir / "artifacts" / "runs" / protocol / run_id
    else:
        artifact_dir = REPO_ROOT / "artifacts" / "runs" / protocol / run_id

    artifact_dir.mkdir(parents=True, exist_ok=True)

    held_out_prefix = CLASS_TO_PREFIX.get(held_out, held_out)
    known_prefixes = [p for p in KNOWN_PREFIXES if p != held_out_prefix]
    known_classes = [PREFIX_TO_CLASS[p] for p in known_prefixes]

    # Enforce pool guard: fitting only allowed on train, calibration only allowed on cal
    backbone_id = chosen_config["backbone"]
    feature_mode = chosen_config["feature"]
    k_val = chosen_config["k"]

    extractor = make_extractor(backbone_id)
    extractor.to(device)

    # 1. Fit Gaussian on TRAIN ONLY
    train_features = {}
    train_input_shas = []
    with PoolGuard(allowed_pools=["train"]):
        for prefix in known_prefixes:
            cls_name = PREFIX_TO_CLASS[prefix]
            records = get_pool_v2(manifest, prefix, "train")
            for r in records:
                train_input_shas.append(r.get("sha256", ""))
            feats = get_or_extract_features(
                records=records,
                extractor=extractor,
                backbone_id=backbone_id,
                feature=feature_mode,
                manifest_version=manifest_version,
                run_id=run_id,
                pool="train",
                class_name=cls_name,
                device=device,
            )
            train_features[cls_name] = feats

    gaussian = fit_gaussian(train_features, k=k_val)

    # 2. Conformal scores on CAL ONLY
    cal_features = {}
    cal_input_shas = []
    with PoolGuard(allowed_pools=["cal"]):
        for prefix in known_prefixes:
            cls_name = PREFIX_TO_CLASS[prefix]
            records = get_pool_v2(manifest, prefix, "cal")
            for r in records:
                cal_input_shas.append(r.get("sha256", ""))
            feats = get_or_extract_features(
                records=records,
                extractor=extractor,
                backbone_id=backbone_id,
                feature=feature_mode,
                manifest_version=manifest_version,
                run_id=run_id,
                pool="cal",
                class_name=cls_name,
                device=device,
            )
            cal_features[cls_name] = feats

    conformal = build_conformal_scores(gaussian, cal_features)

    # 3. Save artifact files
    # gaussian.npz
    np.savez_compressed(
        artifact_dir / "gaussian.npz",
        A=gaussian.A,
        class_names=np.array(gaussian.class_names),
        n_c=np.array(gaussian.n_c),
        k=np.array([gaussian.k]),
        feature_dim=np.array([gaussian.feature_dim]),
        shrinkage_intensity=np.array([gaussian.shrinkage_intensity]),
        condition_number=np.array([gaussian.condition_number]),
    )

    # class_means.npz
    means_dict = {}
    for i, (m, ns) in enumerate(zip(gaussian.class_means, gaussian.n_sub)):
        means_dict[f"means_{i}"] = m
        means_dict[f"n_sub_{i}"] = np.array(ns)
    np.savez_compressed(artifact_dir / "class_means.npz", **means_dict)

    # conformal.npz
    c_dict = {f"scores_{c}": conformal.class_scores[c] for c in conformal.class_scores}
    c_dict["S_pool"] = conformal.S_pool
    c_dict["tau_hat"] = np.array([conformal.tau_hat])
    np.savez_compressed(artifact_dir / "conformal.npz", **c_dict)

    # metadata.json
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        git_commit = "unknown"

    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    cfg_sha = canonical_file_sha256(cfg_path) if cfg_path.exists() else "none"

    metadata = {
        "protocol": protocol,
        "run_id": run_id,
        "held_out": held_out,
        "split_seed": split_seed,
        "manifest_version": manifest_version,
        "manifest_sha256": manifest_sha,
        "selection_file_sha256": selection_sha,
        "chosen_config": chosen_config,
        "inheritance_variant": inheritance_variant,
        "gate_params": gate_params,
        "git_commit": git_commit,
        "config_sha256": cfg_sha,
        "device_info": dev_info,
        "timestamp": time.time(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # provenance.json
    artifact_files = ["gaussian.npz", "class_means.npz", "conformal.npz", "metadata.json"]
    artifact_hashes = {fn: canonical_file_sha256(artifact_dir / fn) for fn in artifact_files}

    provenance = {
        "metadata": metadata,
        "artifact_hashes": artifact_hashes,
        "train_image_count": len(train_input_shas),
        "train_input_shas_head": train_input_shas[:5],
        "cal_image_count": len(cal_input_shas),
        "cal_input_shas_head": cal_input_shas[:5],
        "allowed_fitting_pools": ["train"],
        "allowed_cal_pools": ["cal"],
        "disallowed_test_pools": ["test", "unk_test", "new_test"],
    }
    with open(artifact_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    logger.info("Successfully built artifact for %s at %s", run_id, artifact_dir)
    return artifact_dir


def main() -> None:
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Build artifact under Protocol v1.1")
    parser.add_argument("--held-out", default="Sc", help="Held-out defect class")
    parser.add_argument("--seed", type=int, default=0, help="Split seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--protocol", default="v1.1")
    parser.add_argument("--manifest-version", default="v2")
    args = parser.parse_args()

    build_artifact_for_run(
        held_out=args.held_out,
        split_seed=args.seed,
        protocol=args.protocol,
        manifest_version=args.manifest_version,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
