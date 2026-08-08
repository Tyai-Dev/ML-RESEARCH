"""Gradient-based maximum-likelihood estimators, PyTorch.

Same distributions as mlr.models.pure.mle, but instead of the closed-form
solution each estimator *maximizes the log-likelihood numerically* — descend
on the average NLL with any optimizer from the suite. Run against the pure
versions on the same data, they converge to the same estimates; that
equivalence is the point (see the MLE paper).

Parameterizations keep the search unconstrained:
    Bernoulli   p = sigmoid(theta)
    Multinoulli p = softmax(logits)
    Gaussian    sigma = exp(log_sigma)
"""

import numpy as np
import torch

from mlr.models.registry import register_model
from mlr.models.torch.linear_discriminator import _TORCH_OPTIMIZERS


def _build_optimizer(name: str, params, optimizer_params: dict):
    if name not in _TORCH_OPTIMIZERS:
        raise KeyError(
            f"unknown optimizer {name!r}; registered: {sorted(_TORCH_OPTIMIZERS)}"
        )
    return _TORCH_OPTIMIZERS[name](params, dict(optimizer_params))


class _GradientMLE:
    """Shared fit loop: minimize self._nll_tensor(X) over self._params()."""

    def fit(
        self,
        X,
        y=None,
        optimizer: str = "adam",
        epochs: int = 200,
        on_epoch=None,
        **optimizer_params,
    ):
        X_t = self._as_tensor(X)
        self._init_params(X_t)
        opt = _build_optimizer(optimizer, self._params(), optimizer_params)
        for epoch in range(epochs):
            opt.zero_grad()
            loss = self._nll_tensor(X_t)
            loss.backward()
            opt.step()
            if on_epoch is not None:
                on_epoch(epoch, {"train_nll": float(loss)})
        return self

    def nll(self, X) -> float:
        with torch.no_grad():
            return float(self._nll_tensor(self._as_tensor(X)))

    def _as_tensor(self, X) -> torch.Tensor:
        return torch.as_tensor(np.asarray(X), dtype=torch.float32).ravel()


@register_model("bernoulli-mle-torch")
class BernoulliMLETorch(_GradientMLE):
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.theta: torch.Tensor | None = None

    def _init_params(self, X_t):
        torch.manual_seed(self.seed)
        self.theta = torch.zeros(1, requires_grad=True)

    def _params(self):
        return [self.theta]

    def _nll_tensor(self, X_t):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            self.theta.expand_as(X_t), X_t
        )

    def estimates(self) -> dict[str, float]:
        return {"p": float(torch.sigmoid(self.theta))}


@register_model("multinoulli-mle-torch")
class MultinoulliMLETorch(_GradientMLE):
    def __init__(self, k: int | None = None, seed: int = 0):
        self.k = k
        self.seed = seed
        self.logits: torch.Tensor | None = None

    def _init_params(self, X_t):
        torch.manual_seed(self.seed)
        k = self.k if self.k is not None else int(X_t.max()) + 1
        self.logits = torch.zeros(k, requires_grad=True)

    def _params(self):
        return [self.logits]

    def _nll_tensor(self, X_t):
        classes = X_t.long()
        return torch.nn.functional.cross_entropy(
            self.logits.expand(len(classes), -1), classes
        )

    def estimates(self) -> dict[str, float]:
        probs = torch.softmax(self.logits.detach(), dim=0)
        return {f"p{k}": float(p) for k, p in enumerate(probs)}


@register_model("gaussian-mle-torch")
class GaussianMLETorch(_GradientMLE):
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.mu: torch.Tensor | None = None
        self.log_sigma: torch.Tensor | None = None

    def _init_params(self, X_t):
        torch.manual_seed(self.seed)
        # start near the data so the search is well-conditioned
        self.mu = torch.tensor([float(X_t.mean())], requires_grad=True)
        self.log_sigma = torch.zeros(1, requires_grad=True)

    def _params(self):
        return [self.mu, self.log_sigma]

    def _nll_tensor(self, X_t):
        var = torch.exp(2.0 * self.log_sigma)
        return (
            0.5 * torch.log(2.0 * torch.pi * var)
            + ((X_t - self.mu) ** 2).mean() / (2.0 * var)
        ).squeeze()

    def estimates(self) -> dict[str, float]:
        return {
            "mu": float(self.mu),
            "sigma": float(torch.exp(self.log_sigma)),
        }
