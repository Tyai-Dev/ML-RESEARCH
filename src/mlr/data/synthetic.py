"""Synthetic datasets for theory experiments."""

import numpy as np

from mlr.data.registry import register_dataset


@register_dataset("two-gaussians")
def two_gaussians(n: int = 1000, dim: int = 2, sep: float = 2.0, seed: int = 0):
    """Balanced binary classification: two isotropic Gaussians ``sep`` apart."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    center = np.zeros(dim)
    center[0] = sep / 2
    X = np.vstack(
        [
            rng.normal(loc=-center, scale=1.0, size=(n_neg, dim)),
            rng.normal(loc=center, scale=1.0, size=(n_pos, dim)),
        ]
    )
    y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]
