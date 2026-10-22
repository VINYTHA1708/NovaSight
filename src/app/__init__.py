"""NovaSight Application Track package.

Provides storage abstraction, production stream simulation, and application lifecycle.
"""
from __future__ import annotations

from src.app.storage import StorageBackend, SQLiteStorage, PostgresStorage, get_storage

__all__ = [
    "StorageBackend",
    "SQLiteStorage",
    "PostgresStorage",
    "get_storage",
]
