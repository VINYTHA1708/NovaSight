"""Artifact loader and SHA-256 provenance verifier for NovaSight runtime.

Loads base artifacts using pure numpy and json without importing torch or sklearn.
Strictly verifies file checksums against artifacts/provenance.json.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = Path("artifacts/base")
DEFAULT_PROVENANCE_PATH = Path("artifacts/provenance.json")


@dataclass(frozen=True)
class RuntimeArtifact:
    """Immutable in-memory container of verified model and calibration parameters."""

    metadata: dict[str, Any]
    class_names: list[str]
    class_means: list[np.ndarray]   # List of (k_c, D) float64 arrays
    n_c: list[int]
    n_sub: list[list[int]]
    A: np.ndarray                   # Precision matrix (D, D) float64
    condition_number: float
    shrinkage_intensity: float
    k: int
    feature_dim: int
    class_scores: dict[str, np.ndarray]  # Mondrian calibration scores per class
    S_pool: np.ndarray              # Pooled calibration scores
    tau_hat: float
    alpha_default: float
    alpha_grid: list[float]
    provenance_hashes: dict[str, str]


def _hash_file(file_path: Path) -> str:
    """Compute standard SHA-256 of raw file bytes."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_provenance_hashes(
    artifact_dir: Path,
    provenance_path: Path,
) -> dict[str, str]:
    """Verify raw byte SHA-256 hashes against artifacts/provenance.json.

    Args:
        artifact_dir: Directory containing base artifact files.
        provenance_path: Path to provenance.json.

    Returns:
        Dict of filename -> expected_sha256.

    Raises:
        FileNotFoundError: If artifact or provenance files do not exist.
        RuntimeError: If any artifact hash does not match provenance.json.
    """
    if not provenance_path.exists():
        raise FileNotFoundError(f"Provenance file not found: {provenance_path}")

    with open(provenance_path, "r", encoding="utf-8") as f:
        prov_entries = json.load(f)

    # Find the outputs recorded for S5_build_artifact
    expected_outputs: dict[str, str] = {}
    for entry in prov_entries:
        if entry.get("stage") == "S5_build_artifact" and "outputs" in entry:
            expected_outputs.update(entry["outputs"])

    if not expected_outputs:
        raise RuntimeError("No S5_build_artifact outputs found in provenance.json")

    required_files = ["gaussian.npz", "class_means.npz", "conformal.npz", "metadata.json"]
    for fn in required_files:
        if fn not in expected_outputs:
            raise RuntimeError(f"Missing expected hash for '{fn}' in provenance.json")
        fpath = artifact_dir / fn
        if not fpath.exists():
            raise FileNotFoundError(f"Required artifact file missing: {fpath}")
        actual_hash = _hash_file(fpath)
        expected_hash = expected_outputs[fn]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Artifact hash mismatch for {fn}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}\n"
                "Artifact integrity check failed. Refusing to load corrupted/modified artifact."
            )

    logger.info("All base artifact SHA-256 checksums verified against provenance.json.")
    return expected_outputs


def load_runtime_artifact(
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    provenance_path: Path | str = DEFAULT_PROVENANCE_PATH,
) -> RuntimeArtifact:
    """Load and verify frozen v1.0 base artifact with numpy only.

    Args:
        artifact_dir: Directory containing gaussian.npz, class_means.npz, conformal.npz, metadata.json.
        provenance_path: Path to provenance.json for checksum verification.

    Returns:
        RuntimeArtifact containing verified parameters.
    """
    art_dir = Path(artifact_dir)
    prov_path = Path(provenance_path)

    expected_hashes = verify_provenance_hashes(art_dir, prov_path)

    # 1. Metadata
    with open(art_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2. Gaussian precision and parameters
    g_data = np.load(art_dir / "gaussian.npz", allow_pickle=True)
    A = g_data["A"].astype(np.float64)
    class_names = [str(c) for c in g_data["class_names"]]
    n_c = [int(n) for n in g_data["n_c"]]
    k = int(g_data["k"])
    feature_dim = int(g_data["feature_dim"])
    shrinkage_intensity = float(g_data["shrinkage_intensity"])
    condition_number = float(g_data["condition_number"])

    # 3. Class sub-means
    m_data = np.load(art_dir / "class_means.npz")
    class_means = [m_data[f"means_{i}"].astype(np.float64) for i in range(len(class_names))]
    n_sub = [[int(ns) for ns in m_data[f"n_sub_{i}"]] for i in range(len(class_names))]

    # 4. Conformal calibration scores
    c_data = np.load(art_dir / "conformal.npz")
    class_scores = {cn: c_data[f"scores_{cn}"].astype(np.float64) for cn in class_names}
    S_pool = c_data["S_pool"].astype(np.float64)
    tau_hat = float(c_data["tau_hat"][0])

    alpha_default = float(metadata.get("alpha_default", 0.05))
    alpha_grid = [float(a) for a in metadata.get("alpha_grid", [0.01, 0.02, 0.05, 0.10, 0.20])]

    return RuntimeArtifact(
        metadata=metadata,
        class_names=class_names,
        class_means=class_means,
        n_c=n_c,
        n_sub=n_sub,
        A=A,
        condition_number=condition_number,
        shrinkage_intensity=shrinkage_intensity,
        k=k,
        feature_dim=feature_dim,
        class_scores=class_scores,
        S_pool=S_pool,
        tau_hat=tau_hat,
        alpha_default=alpha_default,
        alpha_grid=alpha_grid,
        provenance_hashes=expected_hashes,
    )
