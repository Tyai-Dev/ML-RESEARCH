"""Linear discriminator trained on the logistic loss, pure NumPy.

The model is the hypothesis class h_w(x) = sigmoid(w·x + b), read as the
Bernoulli parameter P(y=1|x). Maximizing the Bernoulli likelihood is
equivalent to minimizing the logistic loss — see
research/linear-discriminator/papers/01-bernoulli-hypothesis-class.

Training is optimizer-driven: any optimizer from mlr.training.optimizers,
with ``batch_size`` selecting full-batch GD (None) or mini-batch SGD.
"""

import numpy as np

from mlr.models.registry import register_model
from mlr.training.optimizers import build_optimizer


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@register_model("linear-discriminator")
class LinearDiscriminator:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.params: dict[str, np.ndarray] | None = None

    def _loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        w, b = self.params["w"], self.params["b"]
        p = _sigmoid(X @ w + b[0])
        err = p - y
        n = len(X)
        grads = {"w": (X.T @ err) / n, "b": np.array([err.mean()])}
        eps = 1e-12
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        return loss, grads

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: str = "sgd",
        epochs: int = 200,
        batch_size: int | None = None,
        on_epoch=None,
        **optimizer_params,
    ) -> "LinearDiscriminator":
        """Minimize the logistic loss.

        ``batch_size=None`` is full-batch gradient descent; an integer gives
        stochastic mini-batches (reshuffled every epoch). Extra keyword
        arguments (lr, momentum, ...) go to the optimizer.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.params = {"w": rng.normal(scale=0.01, size=d), "b": np.zeros(1)}
        opt = build_optimizer(optimizer, **optimizer_params)

        for epoch in range(epochs):
            if batch_size is None:
                _, grads = self._loss_and_grads(X, y)
                opt.step(self.params, grads)
            else:
                order = rng.permutation(n)
                for start in range(0, n, batch_size):
                    idx = order[start : start + batch_size]
                    _, grads = self._loss_and_grads(X[idx], y[idx])
                    opt.step(self.params, grads)
            if on_epoch is not None:
                loss, _ = self._loss_and_grads(X, y)
                p = self.predict_proba(X)
                acc = float(np.mean((p >= 0.5) == (y == 1)))
                on_epoch(epoch, {"train_loss": loss, "train_accuracy": acc})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("model is not fitted")
        return _sigmoid(np.asarray(X, dtype=float) @ self.params["w"] + self.params["b"][0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(int)))
