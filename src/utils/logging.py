"""NovaSight logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure root logger with stream and optional file handler.

    Args:
        level: Logging level string (e.g., "INFO", "DEBUG").
        log_file: Optional path to a log file.
    """
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )
