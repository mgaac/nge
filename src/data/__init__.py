"""Data loading and generation utilities."""

from __future__ import annotations

from typing import Any

__all__ = ["load_dataset", "generated_dataset", "save_dataset"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .dataset import generated_dataset, load_dataset, save_dataset

        mapping = {
            "load_dataset": load_dataset,
            "generated_dataset": generated_dataset,
            "save_dataset": save_dataset,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

