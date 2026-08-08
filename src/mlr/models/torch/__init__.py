"""PyTorch models.

Torch is an optional dependency (``uv sync --extra torch``); this package
imports lazily so the pure-Python side works without it.
"""

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
