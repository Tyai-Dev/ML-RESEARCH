"""Dataset registry.

A dataset is a builder function registered by name; it returns ``(X, y)``
arrays. Synthetic datasets build from parameters; file-backed datasets
should load from ``datasets/`` and be described by a committed manifest
(name, version, path, checksum) so the layout stays reproducible.

    @register_dataset("two-gaussians")
    def two_gaussians(n=1000, ...): ...
"""

from typing import Callable

_DATASETS: dict[str, Callable] = {}


def register_dataset(name: str):
    def decorator(fn: Callable) -> Callable:
        if name in _DATASETS:
            raise ValueError(f"dataset {name!r} is already registered")
        _DATASETS[name] = fn
        return fn

    return decorator


def get_dataset(name: str) -> Callable:
    try:
        return _DATASETS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; registered datasets: {sorted(_DATASETS)}"
        ) from None


def list_datasets() -> list[str]:
    return sorted(_DATASETS)
