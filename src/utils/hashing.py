"""NovaSight utility: file hashing and provenance tracking."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 of bytes.

    Args:
        data: Raw bytes.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_str(s: str) -> str:
    """Compute SHA-256 of a UTF-8 encoded string.

    Args:
        s: Input string.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    return sha256_bytes(s.encode("utf-8"))


def sha256_dir(directory: Path, extensions: list[str] | None = None) -> str:
    """Compute a deterministic SHA-256 over all files in a directory.

    Args:
        directory: Directory to hash.
        extensions: If given, only include files with these extensions.

    Returns:
        Hex-encoded SHA-256 combining all file hashes.
    """
    h = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if extensions and path.suffix not in extensions:
            continue
        h.update(path.as_posix().encode())
        h.update(sha256_file(path).encode())
    return h.hexdigest()


def write_provenance(
    output_path: Path,
    stage: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    notes: str = "",
) -> None:
    """Append a provenance entry to provenance.json.

    Args:
        output_path: Path to provenance.json.
        stage: Pipeline stage name (e.g., "S5_gaussian_fit").
        inputs: Dict of input pool names to hashes.
        outputs: Dict of output artifact names to hashes.
        notes: Optional notes.
    """
    entry: dict[str, Any] = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "notes": notes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.append(entry)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info("Provenance written: stage=%s -> %s", stage, output_path)
