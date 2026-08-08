"""Pure-NumPy optimizers, written from scratch.

An optimizer mutates a dict of parameter arrays in place given a dict of
gradients with the same keys:

    opt = build_optimizer("adam", lr=0.01)
    opt.step(params, grads)

Batching strategy is not the optimizer's business: full-batch gradient
descent (GD) versus stochastic/mini-batch (SGD) is chosen by the training
loop's ``batch_size`` (None = full batch), and composes with any optimizer
here.
"""

import numpy as np

_OPTIMIZERS: dict[str, type] = {}


def register_optimizer(name: str):
    def decorator(cls: type) -> type:
        if name in _OPTIMIZERS:
            raise ValueError(f"optimizer {name!r} is already registered")
        _OPTIMIZERS[name] = cls
        return cls

    return decorator


def list_optimizers() -> list[str]:
    return sorted(_OPTIMIZERS)


def build_optimizer(name: str, **params):
    try:
        cls = _OPTIMIZERS[name]
    except KeyError:
        raise KeyError(
            f"unknown optimizer {name!r}; registered: {sorted(_OPTIMIZERS)}"
        ) from None
    return cls(**params)


@register_optimizer("sgd")
class SGD:
    """Vanilla (stochastic) gradient descent: p <- p - lr * g."""

    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            params[key] -= self.lr * g


@register_optimizer("momentum")
class Momentum:
    """SGD with (heavy-ball) momentum: v <- mu*v + g;  p <- p - lr * v."""

    def __init__(self, lr: float = 0.1, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self._v: dict[str, np.ndarray] = {}

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            v = self._v.get(key)
            self._v[key] = v = self.momentum * (v if v is not None else 0.0) + g
            params[key] -= self.lr * v


@register_optimizer("adagrad")
class Adagrad:
    """Per-parameter lr shrinking with accumulated squared gradients."""

    def __init__(self, lr: float = 0.1, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self._s: dict[str, np.ndarray] = {}

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            self._s[key] = self._s.get(key, 0.0) + g * g
            params[key] -= self.lr * g / (np.sqrt(self._s[key]) + self.eps)


@register_optimizer("rmsprop")
class RMSprop:
    """Adagrad with an exponentially decaying squared-gradient average."""

    def __init__(self, lr: float = 0.01, rho: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self._s: dict[str, np.ndarray] = {}

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            s = self._s.get(key, 0.0)
            self._s[key] = s = self.rho * s + (1.0 - self.rho) * g * g
            params[key] -= self.lr * g / (np.sqrt(s) + self.eps)


@register_optimizer("adam")
class Adam:
    """Adam (Kingma & Ba, 2015) with bias correction."""

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t = 0

    def step(self, params: dict, grads: dict) -> None:
        self._t += 1
        b1, b2 = self.beta1, self.beta2
        for key, g in grads.items():
            m = self._m.get(key, 0.0)
            v = self._v.get(key, 0.0)
            self._m[key] = m = b1 * m + (1.0 - b1) * g
            self._v[key] = v = b2 * v + (1.0 - b2) * g * g
            m_hat = m / (1.0 - b1**self._t)
            v_hat = v / (1.0 - b2**self._t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
