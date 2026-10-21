"""NovaSight configuration loader.

Loads configs/default.yaml and exposes a typed namespace.
All paths are resolved relative to the repository root.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repository root = parent of src/
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the configuration as a dict.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return cfg


def config_sha256(path: Path = CONFIG_PATH) -> str:
    """Return the SHA-256 hash of the config file for provenance tracking.

    Args:
        path: Path to the YAML config file.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path from the config relative to REPO_ROOT.

    Args:
        cfg: Configuration dictionary.
        key: Dot-separated key path (e.g., "paths.raw_data").

    Returns:
        Absolute Path object.
    """
    keys = key.split(".")
    val = cfg
    for k in keys:
        val = val[k]
    return (REPO_ROOT / val).resolve()


# Singleton config loaded once
_cfg: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Return the global config singleton.

    Returns:
        Loaded configuration dictionary (cached after first call).
    """
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg
