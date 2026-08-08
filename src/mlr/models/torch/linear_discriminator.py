"""PyTorch twin of the pure linear discriminator.

Same hypothesis class (sigmoid of an affine map), same fit() contract, same
optimizer names as mlr.training.optimizers — but gradients come from
autograd and updates from torch.optim.
"""

import numpy as np
import torch
from torch import nn

from mlr.models.registry import register_model

_TORCH_OPTIMIZERS = {
    "sgd": lambda p, kw: torch.optim.SGD(p, lr=kw.pop("lr", 0.1), **kw),
    "momentum": lambda p, kw: torch.optim.SGD(
        p, lr=kw.pop("lr", 0.1), momentum=kw.pop("momentum", 0.9), **kw
    ),
    "adagrad": lambda p, kw: torch.optim.Adagrad(p, lr=kw.pop("lr", 0.1), **kw),
    "rmsprop": lambda p, kw: torch.optim.RMSprop(
        p, lr=kw.pop("lr", 0.01), alpha=kw.pop("rho", 0.9), **kw
    ),
    "adam": lambda p, kw: torch.optim.Adam(p, lr=kw.pop("lr", 0.01), **kw),
}


@register_model("linear-discriminator-torch")
class LinearDiscriminatorTorch:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.net: nn.Linear | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: str = "sgd",
        epochs: int = 200,
        batch_size: int | None = None,
        on_epoch=None,
        **optimizer_params,
    ) -> "LinearDiscriminatorTorch":
        if optimizer not in _TORCH_OPTIMIZERS:
            raise KeyError(
                f"unknown optimizer {optimizer!r}; registered: {sorted(_TORCH_OPTIMIZERS)}"
            )
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        n, d = X_t.shape

        torch.manual_seed(self.seed)
        self.net = nn.Linear(d, 1)
        opt = _TORCH_OPTIMIZERS[optimizer](self.net.parameters(), dict(optimizer_params))
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)

        for epoch in range(epochs):
            if batch_size is None:
                batches = [(X_t, y_t)]
            else:
                order = torch.randperm(n, generator=generator)
                batches = [
                    (X_t[order[s : s + batch_size]], y_t[order[s : s + batch_size]])
                    for s in range(0, n, batch_size)
                ]
            for xb, yb in batches:
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                opt.step()
            if on_epoch is not None:
                with torch.no_grad():
                    logits = self.net(X_t)
                    loss = float(loss_fn(logits, y_t))
                    acc = float(((logits >= 0) == (y_t == 1)).float().mean())
                on_epoch(epoch, {"train_loss": loss, "train_accuracy": acc})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        with torch.no_grad():
            logits = self.net(torch.as_tensor(np.asarray(X), dtype=torch.float32))
            return torch.sigmoid(logits).numpy().ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(int)))
