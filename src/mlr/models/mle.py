"""One MLE model: pick a distribution, pick how to maximize its likelihood.

    model: mle
    model_params:
      distribution: gaussian     # any registered distribution
      method: mle                # 'mle' = closed form; 'gradient' = optimize the nll
      backend: pure              # gradient only: 'pure' (NumPy) or 'torch' (autograd)

The distribution supplies the math (mlr.distributions); this class only
decides whether to call ``.mle()`` or feed ``.unll`` to an optimizer.
"""

import numpy as np

from mlr.distributions import get_distribution
from mlr.models.registry import register_model
from mlr.training.optimizers import build_optimizer


@register_model("mle")
class MLEModel:
    def __init__(
        self,
        distribution: str,
        method: str = "mle",
        backend: str = "pure",
        seed: int = 0,
        **distribution_params,
    ):
        if method not in ("mle", "gradient"):
            raise ValueError(f"method must be 'mle' or 'gradient', got {method!r}")
        if backend not in ("pure", "torch"):
            raise ValueError(f"backend must be 'pure' or 'torch', got {backend!r}")
        self.dist = get_distribution(distribution, **distribution_params)
        self.method = method
        self.backend = backend
        self.seed = seed
        self.params: dict | None = None

    def _prepare(self, X) -> np.ndarray:
        dtype = int if self.dist.dtype == "int" else float
        return np.asarray(X, dtype=dtype).ravel()

    def fit(self, X, y=None, on_epoch=None, **training) -> "MLEModel":
        X = self._prepare(X)
        if self.method == "mle":
            if training:
                raise ValueError(
                    f"method 'mle' is closed-form; drop the training section ({training})"
                )
            self.params = self.dist.mle(X)
            if on_epoch is not None:
                on_epoch(0, {"train_nll": self.nll(X)})
        else:
            fit_gradient = (
                self._fit_gradient_pure if self.backend == "pure" else self._fit_gradient_torch
            )
            uparams = fit_gradient(X, on_epoch, **training)
            self.params = self.dist.to_natural(uparams)
        return self

    def _fit_gradient_pure(
        self, X, on_epoch, optimizer: str = "adam", epochs: int = 200, **optimizer_params
    ) -> dict:
        uparams = self.dist.init_uparams(X)
        opt = build_optimizer(optimizer, **optimizer_params)
        for epoch in range(epochs):
            opt.step(uparams, self.dist.unll_grads(uparams, X))
            if on_epoch is not None:
                on_epoch(epoch, {"train_nll": float(self.dist.unll(uparams, X, np))})
        return uparams

    def _fit_gradient_torch(
        self, X, on_epoch, optimizer: str = "adam", epochs: int = 200, **optimizer_params
    ) -> dict:
        import torch

        from mlr.models.torch.linear_discriminator import _TORCH_OPTIMIZERS

        if optimizer not in _TORCH_OPTIMIZERS:
            raise KeyError(
                f"unknown optimizer {optimizer!r}; registered: {sorted(_TORCH_OPTIMIZERS)}"
            )
        torch.manual_seed(self.seed)
        X_t = torch.as_tensor(
            X, dtype=torch.long if self.dist.dtype == "int" else torch.float32
        )
        uparams = {
            key: torch.as_tensor(value, dtype=torch.float32).requires_grad_(True)
            for key, value in self.dist.init_uparams(X).items()
        }
        opt = _TORCH_OPTIMIZERS[optimizer](list(uparams.values()), dict(optimizer_params))
        for epoch in range(epochs):
            opt.zero_grad()
            loss = self.dist.unll(uparams, X_t, torch)
            loss.backward()
            opt.step()
            if on_epoch is not None:
                on_epoch(epoch, {"train_nll": float(loss)})
        return {key: value.detach().numpy() for key, value in uparams.items()}

    def nll(self, X) -> float:
        if self.params is None:
            raise RuntimeError("model is not fitted")
        return self.dist.nll(self.params, self._prepare(X))

    def estimates(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in self.params.items():
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            if arr.size == 1:
                out[key] = float(arr[0])
            else:
                out.update({f"{key}{i}": float(v) for i, v in enumerate(arr)})
        return out
