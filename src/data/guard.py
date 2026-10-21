"""NovaSight data-access guard (Part 5.2).

Each pipeline stage declares which pools it may read.
Reading any undeclared pool raises a DataAccessError.
This guard is unit-tested.

Usage:
    with PoolGuard(allowed_pools=["train", "early_stop"]):
        records = get_combined_pool(manifest, "train")   # OK
        records = get_combined_pool(manifest, "test")    # raises DataAccessError
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


class DataAccessError(Exception):
    """Raised when code attempts to read a pool it is not permitted to read."""


# Thread-local (or module-level) guard state
_allowed_pools: set[str] | None = None
_guard_active: bool = False


def declare_pools(pool_names: list[str]) -> None:
    """Declare which pools the current stage may read.

    Args:
        pool_names: List of pool names this stage is allowed to access.
    """
    global _allowed_pools, _guard_active
    _allowed_pools = set(pool_names)
    _guard_active = True
    logger.debug("Pool guard active; allowed: %s", sorted(pool_names))


def clear_guard() -> None:
    """Disable the pool guard (no restrictions)."""
    global _allowed_pools, _guard_active
    _allowed_pools = None
    _guard_active = False


def check_pool_access(pool_name: str) -> None:
    """Assert that the current stage is allowed to read this pool.

    Args:
        pool_name: The pool name being accessed.

    Raises:
        DataAccessError: If the guard is active and pool_name is not allowed.
    """
    if not _guard_active:
        return
    # Strip class prefix for generic pool names (e.g., "Cr_train" -> "train")
    generic = pool_name.split("_", 1)[-1] if "_" in pool_name else pool_name
    if pool_name not in _allowed_pools and generic not in _allowed_pools:
        raise DataAccessError(
            f"Pool '{pool_name}' is not in the declared allowed pools "
            f"{sorted(_allowed_pools)}. This stage may only read: "
            f"{sorted(_allowed_pools)}"
        )


@contextmanager
def PoolGuard(allowed_pools: list[str]) -> Generator[None, None, None]:
    """Context manager that activates the pool guard for a block of code.

    Args:
        allowed_pools: Pool names allowed within this context.

    Yields:
        None.
    """
    global _allowed_pools, _guard_active
    prev_pools = _allowed_pools
    prev_active = _guard_active
    declare_pools(allowed_pools)
    try:
        yield
    finally:
        _allowed_pools = prev_pools
        _guard_active = prev_active
