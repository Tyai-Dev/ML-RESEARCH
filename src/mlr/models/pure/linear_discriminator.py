"""Linear discriminator trained on the logistic loss, pure NumPy.

The model is the hypothesis class h_w(x) = sigmoid(w·x + b), read as the
Bernoulli parameter P(y=1|x). Maximizing the Bernoulli likelihood is
equivalent to minimizing the logistic loss — see
research/linear-discriminator/papers/01-bernoulli-hypothesis-class.
"""

import numpy as np

from mlr.models.registry import register_model


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@register_model("linear-discriminator")
class LinearDiscriminator:
    def __init__(self, lr: float = 0.1, epochs: int = 200, seed: int = 0):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, on_epoch=None) -> "LinearDiscriminator":
        """Full-batch gradient descent on the logistic loss.

        ``on_epoch(epoch, metrics)`` is called once per epoch with the
        current training loss and accuracy, for metric tracking.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(scale=0.01, size=d)
        self.b = 0.0

        for epoch in range(self.epochs):
            p = _sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * float(err.mean())
            if on_epoch is not None:
                eps = 1e-12
                loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
                acc = float(np.mean((p >= 0.5) == (y == 1)))
                on_epoch(epoch, {"train_loss": loss, "train_accuracy": acc})
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("model is not fitted")
        return _sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(int)))
