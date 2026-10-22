"""Colab & Local sequential job runner under Protocol v1.1 (Task E11, B.3).

Refuses to run if:
  - Working copy is dirty (unless explicitly flagged for synthetic testing).
  - Current git commit != --expect-commit.

Writes artifacts to <out>/artifacts/... and reports to <out>/reports/...
Generates <out>/RUN_BUNDLE.json with complete provenance and artifact hashes.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_jobs.py --gate 4 --rows all --out out_bundle --device auto --expect-commit <sha>
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT
from src.eval.rehearsal import CandidateResult
from src.utils.device import get_device_info, setup_device
from src.utils.hashing import canonical_file_sha256, canonical_json_sha256
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def check_git_clean_and_commit(expect_commit: str, allow_dirty: bool = False) -> str:
    """Verify git repository cleanliness and exact commit match."""
    try:
        # Check current commit
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        ).decode("ascii").strip()
    except Exception as e:
        raise RuntimeError(f"Failed to query git commit: {e}") from e

    if expect_commit.lower() != "any":
        if not current_commit.startswith(expect_commit) and not expect_commit.startswith(current_commit):
            raise ValueError(
                f"Commit mismatch refusal: expected '{expect_commit}', but repository HEAD is '{current_commit}'."
            )

    if not allow_dirty:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT)
        ).decode("utf-8").strip()
        if status_out:
            lines = status_out.splitlines()[:5]
            raise ValueError(
                f"Working copy dirty refusal: repo has uncommitted changes:\n" + "\n".join(lines)
            )

    return current_commit


def run_jobs(
    gate: int,
    rows: str,
    out_dir: Path,
    device_arg: str,
    expect_commit: str,
    allow_dirty: bool = False,
) -> Path:
    """Execute jobs sequentially and produce verified RUN_BUNDLE.json."""
    device = setup_device(device_arg)
    dev_info = get_device_info(device)
    commit = check_git_clean_and_commit(expect_commit, allow_dirty=allow_dirty)

    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = REPO_ROOT / "reports" / "run_registry.csv"
    if not registry_path.is_file():
        raise FileNotFoundError(f"Run registry not found: {registry_path}")

    df = pd.read_csv(registry_path)

    # Determine row indices
    if rows.lower() == "all":
        target_rows = list(df.index)
    else:
        target_rows = [int(r.strip()) for r in rows.split(",") if r.strip()]

    # Filter out rows if already complete in df
    bundle_runs: list[dict[str, Any]] = []

    from scripts.rehearse import run_rehearsal_for_run
    from scripts.build_artifact_v11 import build_artifact_for_run
    from scripts.evaluate_v11 import evaluate_run

    logger.info("Executing Gate %d jobs for %d rows on device %s", gate, len(target_rows), device)

    for r_idx in target_rows:
        if r_idx < 0 or r_idx >= len(df):
            continue
        row = df.iloc[r_idx]
        exp = str(row["experiment"])
        held_out = str(row["held_out"])
        seed = int(row["seed"])
        run_id = f"H={held_out}_s={seed}"

        t0 = time.time()
        run_record: dict[str, Any] = {
            "row_idx": r_idx,
            "experiment": exp,
            "held_out": held_out,
            "seed": seed,
            "run_id": run_id,
            "status": "RUNNING",
            "artifact_hashes": {},
        }

        try:
            if gate == 4:
                # Gate 4: Rehearsal & Model Selection
                sel_res = run_rehearsal_for_run(
                    held_out=held_out,
                    split_seed=seed,
                    device_arg=device_arg,
                    out_dir=out_dir / "reports" / "v1.1",
                )
                sel_file = out_dir / "reports" / "v1.1" / f"selection_{run_id}.json"
                run_record["artifact_hashes"][str(sel_file.relative_to(out_dir))] = canonical_file_sha256(sel_file)
                run_record["status"] = "COMPLETE"

            elif gate == 5:
                # Gate 5: Artifact Build
                art_dir = build_artifact_for_run(
                    held_out=held_out,
                    split_seed=seed,
                    device_arg=device_arg,
                    custom_out_dir=out_dir,
                )
                for fpath in art_dir.glob("*"):
                    if fpath.is_file():
                        run_record["artifact_hashes"][str(fpath.relative_to(out_dir))] = canonical_file_sha256(fpath)
                run_record["status"] = "COMPLETE"

            elif gate == 6:
                # Gate 6: Read-only Evaluation
                eval_res = evaluate_run(
                    held_out=held_out,
                    split_seed=seed,
                    device_arg=device_arg,
                    out_dir=out_dir / "reports" / "v1.1",
                )
                eval_file = out_dir / "reports" / "v1.1" / f"results_{run_id}.json"
                cm_file = out_dir / "reports" / "v1.1" / f"confusion_matrix_{run_id}.json"
                run_record["artifact_hashes"][str(eval_file.relative_to(out_dir))] = canonical_file_sha256(eval_file)
                run_record["artifact_hashes"][str(cm_file.relative_to(out_dir))] = canonical_file_sha256(cm_file)
                run_record["status"] = "COMPLETE"

            else:
                raise ValueError(f"Unsupported gate: {gate}")

        except Exception as e:
            logger.error("Job failed for %s: %s", run_id, e)
            run_record["status"] = "FAILED"
            run_record["error"] = str(e)

        run_record["runtime_s"] = round(time.time() - t0, 3)
        bundle_runs.append(run_record)

    bundle_data = {
        "gate": gate,
        "commit_sha": commit,
        "device_info": dev_info,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": bundle_runs,
    }

    bundle_path = out_dir / "RUN_BUNDLE.json"
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle_data, f, indent=2)

    logger.info("Run bundle written to %s", bundle_path)
    return bundle_path


def main() -> None:
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Sequential Protocol v1.1 job runner")
    parser.add_argument("--gate", type=int, required=True, choices=[4, 5, 6], help="Gate number (4, 5, or 6)")
    parser.add_argument("--rows", default="all", help="Comma-separated registry row numbers or 'all'")
    parser.add_argument("--out", type=Path, required=True, help="Output bundle directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device")
    parser.add_argument("--expect-commit", required=True, help="Expected git commit hash")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow running on dirty working copy (testing only)")
    args = parser.parse_args()

    run_jobs(
        gate=args.gate,
        rows=args.rows,
        out_dir=args.out,
        device_arg=args.device,
        expect_commit=args.expect_commit,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
