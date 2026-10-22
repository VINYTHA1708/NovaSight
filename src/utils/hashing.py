"""NovaSight utility: file hashing, canonical serialization, and provenance tracking."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".json", ".csv", ".yaml", ".yml", ".md", ".txt", ".py"}


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


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file using raw bytes.

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


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize an object to deterministic canonical JSON bytes.

    Uses sorted keys, compact separators, and UTF-8 encoding.

    Args:
        obj: JSON-serializable object.

    Returns:
        UTF-8 encoded canonical JSON bytes.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_sha256(obj: Any) -> str:
    """Compute SHA-256 of deterministic canonical JSON bytes.

    Args:
        obj: JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return sha256_bytes(canonical_json_bytes(obj))


def canonical_file_sha256(path: Path) -> str:
    """Compute canonical line-ending-independent SHA-256 of a file.

    For text files (.json, .csv, .yaml, .yml, .md, .txt, .py), normalizes
    CRLF (\\r\\n) line endings to LF (\\n) and removes any UTF-8 BOM before hashing.
    For binary files, hashes raw bytes directly.

    Args:
        path: Path to file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    suffix = path.suffix.lower()
    raw = path.read_bytes()
    if suffix in TEXT_EXTENSIONS:
        # Strip UTF-8 BOM if present
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        # Normalize CRLF to LF
        normalized = raw.replace(b"\r\n", b"\n")
        return sha256_bytes(normalized)
    return sha256_bytes(raw)


def sha256_dir(directory: Path, extensions: list[str] | None = None, canonical: bool = False) -> str:
    """Compute a deterministic SHA-256 over all files in a directory.

    Args:
        directory: Directory to hash.
        extensions: If given, only include files with these extensions.
        canonical: If True, uses canonical line-ending-normalized hash for text files.

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
        file_hash = canonical_file_sha256(path) if canonical else sha256_file(path)
        h.update(file_hash.encode())
    return h.hexdigest()


def write_provenance(
    output_path: Path,
    stage: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    notes: str = "",
    device_info: dict[str, Any] | None = None,
    code_commit: str | None = None,
) -> None:
    """Write or append a provenance entry with full environment and hash metadata.

    Args:
        output_path: Path to provenance.json.
        stage: Pipeline stage name (e.g., "S5_build_artifact").
        inputs: Dict of input pool names/paths to hashes.
        outputs: Dict of output artifact names to hashes.
        notes: Optional notes.
        device_info: Optional dict with device, cuda, cudnn details.
        code_commit: Git commit SHA of the code running this stage.
    """
    entry: dict[str, Any] = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit or "unknown",
        "device_info": device_info or {"device": "cpu"},
        "inputs": inputs,
        "outputs": outputs,
        "notes": notes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(entry)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info("Provenance written: stage=%s -> %s", stage, output_path)
