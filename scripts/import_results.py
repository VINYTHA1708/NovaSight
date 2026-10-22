"""Importer tool for RUN_BUNDLE.json results (Item B.4, Task E11).

Steps:
  1. Verifies commit sha in the bundle matches a valid committed commit in this repo.
  2. Verifies every artifact's sha256 against bundle record.
  3. Copies reports to reports/v1.1/ and artifacts to artifacts/runs/v1.1/.
  4. Runs scripts/verify_artifact.py on each imported artifact directory.
  5. Updates reports/run_registry.csv with COMPLETE status, paths, sha256, and timing.

Usage:
    .venv\\Scripts\\python.exe scripts\\import_results.py path/to/RUN_BUNDLE.json
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT
from src.utils.hashing import canonical_file_sha256
from scripts.verify_artifact import verify_artifact_dir

logger = logging.getLogger(__name__)


def verify_commit_in_repo(commit_sha: str) -> bool:
    """Verify that commit_sha exists in local git history."""
    try:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def import_run_bundle(bundle_path: Path) -> bool:
    """Import and verify run bundle."""
    if not bundle_path.is_file():
        logger.error("Bundle file does not exist: %s", bundle_path)
        return False

    with open(bundle_path, encoding="utf-8") as f:
        bundle = json.load(f)

    commit_sha = bundle.get("commit_sha")
    if not commit_sha or not verify_commit_in_repo(commit_sha):
        logger.error("Refusal: commit %s is not a valid committed object in this repo.", commit_sha)
        return False

    bundle_root = bundle_path.parent
    registry_path = REPO_ROOT / "reports" / "run_registry.csv"
    df = pd.read_csv(registry_path)

    # 1. Verify artifact hashes in the bundle
    runs = bundle.get("runs", [])
    for r in runs:
        if r.get("status") != "COMPLETE":
            continue
        art_hashes = r.get("artifact_hashes", {})
        for rel_path_str, expected_sha in art_hashes.items():
            file_path = bundle_root / rel_path_str
            if not file_path.is_file():
                logger.error("Artifact missing in bundle: %s", file_path)
                return False
            actual_sha = canonical_file_sha256(file_path)
            if actual_sha != expected_sha:
                logger.error("Corrupted bundle file %s: expected %s, got %s", file_path, expected_sha, actual_sha)
                return False

    # 2. Copy files to destination directories
    reports_dst = REPO_ROOT / "reports" / "v1.1"
    artifacts_dst = REPO_ROOT / "artifacts" / "runs" / "v1.1"
    reports_dst.mkdir(parents=True, exist_ok=True)
    artifacts_dst.mkdir(parents=True, exist_ok=True)

    imported_count = 0
    for r in runs:
        if r.get("status") != "COMPLETE":
            continue

        r_idx = r.get("row_idx")
        run_id = r.get("run_id")
        art_hashes = r.get("artifact_hashes", {})

        # Copy each file
        for rel_path_str in art_hashes:
            src_file = bundle_root / rel_path_str
            dst_file = REPO_ROOT / rel_path_str
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

        # If artifact was copied, run verify_artifact
        run_art_dir = artifacts_dst / run_id
        if run_art_dir.is_dir():
            if not verify_artifact_dir(run_art_dir):
                logger.error("Verification failed for imported artifact at %s", run_art_dir)
                return False

        # Update registry row
        if r_idx is not None and 0 <= r_idx < len(df):
            result_file = reports_dst / f"results_{run_id}.json"
            if result_file.is_file():
                df.at[r_idx, "status"] = "COMPLETE"
                df.at[r_idx, "result_path"] = str(result_file.relative_to(REPO_ROOT)).replace("\\", "/")
                df.at[r_idx, "result_sha256"] = canonical_file_sha256(result_file)
                df.at[r_idx, "runtime_s"] = r.get("runtime_s", "")
                df.at[r_idx, "finished"] = bundle.get("created_utc", "")
                df.at[r_idx, "notes"] = f"Protocol v1.1 commit {commit_sha[:8]}"

        imported_count += 1

    df.to_csv(registry_path, index=False)
    logger.info("Successfully imported and verified %d runs from %s", imported_count, bundle_path)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Import results from RUN_BUNDLE.json")
    parser.add_argument("bundle_path", type=Path, help="Path to RUN_BUNDLE.json")
    args = parser.parse_args()

    ok = import_run_bundle(args.bundle_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
