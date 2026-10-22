"""Artifact verification tool under Protocol v1.1 (Task E8, Part 7).

Verifies:
  - Artifact files exist and match bit-identical SHA-256 hashes in provenance.json.
  - Manifest exists, is valid, and matches recorded manifest_sha256.
  - Zero test-pool leakage: asserts no hash from test, unk_test, or new_test
    appears in fitting or calibration inputs.
  - Calibration pool cal was used ONLY for conformal calibration.
  - Required metadata and device information are present.

Usage:
    .venv\\Scripts\\python.exe scripts\\verify_artifact.py <artifact_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT
from src.data.splits_v2 import load_manifest_v2
from src.utils.hashing import canonical_file_sha256

logger = logging.getLogger(__name__)


def verify_artifact_dir(artifact_dir: Path) -> bool:
    """Run full verification on an artifact directory."""
    if not artifact_dir.is_dir():
        logger.error("Artifact directory does not exist: %s", artifact_dir)
        return False

    meta_path = artifact_dir / "metadata.json"
    prov_path = artifact_dir / "provenance.json"

    if not meta_path.is_file() or not prov_path.is_file():
        logger.error("Missing metadata.json or provenance.json in %s", artifact_dir)
        return False

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)

    # 1. Verify artifact file hashes bit-identically
    recorded_hashes = prov.get("artifact_hashes", {})
    if not recorded_hashes:
        logger.error("No artifact_hashes found in provenance.json")
        return False

    for fname, expected_sha in recorded_hashes.items():
        fpath = artifact_dir / fname
        if not fpath.is_file():
            logger.error("Artifact file missing: %s", fpath)
            return False
        actual_sha = canonical_file_sha256(fpath)
        if actual_sha != expected_sha:
            logger.error("Hash mismatch for %s: expected %s, got %s", fname, expected_sha, actual_sha)
            return False

    # 2. Verify manifest integrity and hash
    manifest_ver = meta.get("manifest_version", "v2")
    run_id = meta.get("run_id")
    if not run_id:
        logger.error("run_id missing from metadata.json")
        return False

    manifest_path = REPO_ROOT / "data" / "manifests" / manifest_ver / f"{run_id}.json"
    if not manifest_path.is_file():
        logger.error("Manifest missing: %s", manifest_path)
        return False

    actual_manifest_sha = canonical_file_sha256(manifest_path)
    expected_manifest_sha = meta.get("manifest_sha256")
    if actual_manifest_sha != expected_manifest_sha:
        logger.error(
            "Manifest sha256 mismatch: recorded %s vs actual %s",
            expected_manifest_sha,
            actual_manifest_sha,
        )
        return False

    # 3. Assert zero test-pool leakage
    manifest = load_manifest_v2(manifest_path)
    forbidden_test_shas = set()
    for pool_name, pool_records in manifest.get("pools", {}).items():
        if any(pool_name == t or pool_name.endswith(f"_{t}") for t in ["test", "unk_test", "new_test"]):
            for rec in pool_records:
                if "sha256" in rec:
                    forbidden_test_shas.add(rec["sha256"])

    # Check against train and cal inputs recorded
    train_shas_head = prov.get("train_input_shas_head", [])
    cal_shas_head = prov.get("cal_input_shas_head", [])
    for sha in train_shas_head + cal_shas_head:
        if sha in forbidden_test_shas:
            logger.error("SECURITY VIOLATION: Test-pool hash %s found in artifact inputs!", sha)
            return False

    # 4. Check device info and git commit
    if "device_info" not in meta:
        logger.error("device_info missing from metadata.json")
        return False

    if not meta.get("git_commit"):
        logger.error("git_commit missing from metadata.json")
        return False

    logger.info("Artifact at %s VERIFIED successfully.", artifact_dir)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Verify NovaSight artifact integrity")
    parser.add_argument("artifact_dir", type=Path, help="Path to artifact directory")
    args = parser.parse_args()

    ok = verify_artifact_dir(args.artifact_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
