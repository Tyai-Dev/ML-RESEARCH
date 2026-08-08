"""Closed-form maximum-likelihood estimators, pure NumPy.

These are the analytic MLE solutions derived in
research/mle/papers/01-maximum-likelihood-principle: no iteration, the
estimate is a formula of the sample. Each estimator implements the
unsupervised model contract: ``fit(X, y=None, on_epoch=None)``,
``nll(X)`` (average negative log-likelihood, the held-out score), and
``estimates()`` (named parameter estimates, logged by the runner).
"""

import numpy as np

from mlr.models.registry import register_model

_EPS = 1e-12


@register_model("bernoulli-mle")
class BernoulliMLE:
    """X in {0,1}; MLE is the sample mean: p_hat = x_bar."""

    def __init__(self):
        self.p: float | None = None

    def fit(self, X, y=None, on_epoch=None) -> "BernoulliMLE":
        X = np.asarray(X, dtype=float).ravel()
        self.p = float(X.mean())
        if on_epoch is not None:
            on_epoch(0, {"train_nll": self.nll(X)})
        return self

    def nll(self, X) -> float:
        X = np.asarray(X, dtype=float).ravel()
        p = np.clip(self.p, _EPS, 1 - _EPS)
        return float(-np.mean(X * np.log(p) + (1 - X) * np.log(1 - p)))

    def estimates(self) -> dict[str, float]:
        return {"p": self.p}


@register_model("multinoulli-mle")
class MultinoulliMLE:
    """X in {0..k-1}; MLE is the empirical frequency: p_hat_k = n_k / n."""

    def __init__(self, k: int | None = None):
        self.k = k
        self.probs: np.ndarray | None = None

    def fit(self, X, y=None, on_epoch=None) -> "MultinoulliMLE":
        X = np.asarray(X, dtype=int).ravel()
        k = self.k if self.k is not None else int(X.max()) + 1
        counts = np.bincount(X, minlength=k).astype(float)
        self.probs = counts / counts.sum()
        if on_epoch is not None:
            on_epoch(0, {"train_nll": self.nll(X)})
        return self

    def nll(self, X) -> float:
        X = np.asarray(X, dtype=int).ravel()
        return float(-np.mean(np.log(np.clip(self.probs[X], _EPS, None))))

    def estimates(self) -> dict[str, float]:
        return {f"p{k}": float(p) for k, p in enumerate(self.probs)}


@register_model("gaussian-mle")
class GaussianMLE:
    """X real; MLE is mu_hat = x_bar, sigma2_hat = (1/n) sum (x - x_bar)^2.

    Note the 1/n (biased) variance — that IS the maximum-likelihood
    estimate; Bessel's 1/(n-1) correction is a different estimator.
    """

    def __init__(self):
        self.mu: float | None = None
        self.sigma: float | None = None

    def fit(self, X, y=None, on_epoch=None) -> "GaussianMLE":
        X = np.asarray(X, dtype=float).ravel()
        self.mu = float(X.mean())
        self.sigma = float(np.sqrt(np.mean((X - self.mu) ** 2)))
        if on_epoch is not None:
            on_epoch(0, {"train_nll": self.nll(X)})
        return self

    def nll(self, X) -> float:
        X = np.asarray(X, dtype=float).ravel()
        var = max(self.sigma**2, _EPS)
        return float(
            0.5 * np.log(2 * np.pi * var) + np.mean((X - self.mu) ** 2) / (2 * var)
        )

    def estimates(self) -> dict[str, float]:
        return {"mu": self.mu, "sigma": self.sigma}
