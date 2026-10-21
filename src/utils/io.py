"""NovaSight I/O utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
    """Load and return JSON from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data as JSON.

    Args:
        data: JSON-serialisable object.
        path: Output file path.
        indent: JSON indentation level.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    logger.debug("Saved JSON to %s", path)
