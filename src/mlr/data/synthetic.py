"""Synthetic datasets for theory experiments."""

import numpy as np

from mlr.data.registry import register_dataset


@register_dataset("bernoulli-samples")
def bernoulli_samples(n: int = 1000, p: float = 0.3, seed: int = 0):
    """Unsupervised i.i.d. Bernoulli(p) draws (0/1), for MLE estimation."""
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p, size=n).astype(float)


@register_dataset("multinoulli-samples")
def multinoulli_samples(n: int = 1000, probs=(0.5, 0.3, 0.2), seed: int = 0):
    """Unsupervised i.i.d. categorical draws (class indices 0..k-1)."""
    probs = np.asarray(probs, dtype=float)
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"probs must sum to 1, got {probs.sum()}")
    rng = np.random.default_rng(seed)
    return rng.choice(len(probs), size=n, p=probs).astype(float)


@register_dataset("gaussian-samples")
def gaussian_samples(n: int = 1000, mu: float = 2.0, sigma: float = 1.5, seed: int = 0):
    """Unsupervised i.i.d. N(mu, sigma^2) draws."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)


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
