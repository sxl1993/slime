"""Small scheduling helpers for SAO training."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def run_critic_updates(
    update_critic: Callable[[], object],
    refresh_values: Callable[[], T],
    *,
    update_count: int,
) -> T:
    """Run all critic updates, then refresh values from the updated critic."""
    if update_count <= 0:
        raise ValueError("update_count must be positive")

    for _ in range(update_count):
        update_critic()
    return refresh_values()
