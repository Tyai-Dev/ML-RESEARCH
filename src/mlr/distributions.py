"""Parametric distributions: one class per family, fitting-method-free.

A Distribution owns the math and nothing else:

- ``nll(params, X)``          average negative log-likelihood at *natural*
                              parameters (the held-out score)
- ``mle(X)``                  the closed-form maximum-likelihood estimate
- ``init_uparams()``          unconstrained parameterization for numerical
                              fitting (logit p, logits, log sigma) — licensed
                              by the invariance property of MLE
- ``unll(uparams, X, bk)``    the same nll over unconstrained parameters,
                              written backend-agnostically: pass ``numpy`` and
                              it computes, pass ``torch`` and it is
                              autograd-differentiable
- ``unll_grads(uparams, X)``  hand-derived NumPy gradients of ``unll`` — pure
                              gradient fitting has no autograd, so the chain
                              rule lives here (checked against finite
                              differences in tests)
- ``to_natural(uparams)``     map the unconstrained optimum back to natural
                              parameters

How the fit happens — closed form or any optimizer minimizing the nll — is
the model's business (mlr.models.mle), not the distribution's.
"""

import numpy as np

_EPS = 1e-12

_DISTRIBUTIONS: dict[str, type] = {}


def register_distribution(name: str):
    def decorator(cls: type) -> type:
        if name in _DISTRIBUTIONS:
            raise ValueError(f"distribution {name!r} is already registered")
        _DISTRIBUTIONS[name] = cls
        return cls

    return decorator


def get_distribution(name: str, **params):
    try:
        cls = _DISTRIBUTIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown distribution {name!r}; registered: {sorted(_DISTRIBUTIONS)}"
        ) from None
    return cls(**params)


def list_distributions() -> list[str]:
    return sorted(_DISTRIBUTIONS)


def _softplus(z, bk):
    # log(1 + e^z), stable via logaddexp(0, z)
    return bk.logaddexp(bk.zeros_like(z), z)


def _sigmoid(z, bk):
    return 1.0 / (1.0 + bk.exp(-z))


@register_distribution("bernoulli")
class Bernoulli:
    """x in {0,1}, P(x=1) = p."""

    dtype = "float"
    formula = "p_hat = x_bar (sample mean)"

    def estimator_variance(self, params: dict, n: int) -> dict[str, float]:
        """Theoretical Var of the MLE at true params: Var(p_hat) = p(1-p)/n."""
        p = params["p"]
        return {"p": p * (1 - p) / n}

    def nll(self, params: dict, X) -> float:
        X = np.asarray(X, dtype=float).ravel()
        p = np.clip(params["p"], _EPS, 1 - _EPS)
        return float(-np.mean(X * np.log(p) + (1 - X) * np.log(1 - p)))

    def mle(self, X) -> dict:
        return {"p": float(np.asarray(X, dtype=float).mean())}

    def init_uparams(self, X) -> dict[str, np.ndarray]:
        return {"theta": np.zeros(1)}  # p = sigmoid(theta) = 0.5

    def unll(self, uparams: dict, X, bk=np):
        # mean over samples of  softplus(theta) - x * theta
        theta = uparams["theta"][0]
        return _softplus(theta, bk) - bk.mean(X) * theta

    def unll_grads(self, uparams: dict, X) -> dict[str, np.ndarray]:
        # d/dtheta = sigmoid(theta) - x_bar
        theta = uparams["theta"]
        return {"theta": _sigmoid(theta, np) - np.mean(X)}

    def to_natural(self, uparams: dict) -> dict:
        return {"p": float(_sigmoid(np.asarray(uparams["theta"]), np)[0])}


@register_distribution("multinoulli")
class Multinoulli:
    """x in {0..k-1}, P(x=j) = p_j with sum p = 1."""

    dtype = "int"
    formula = "p_hat_k = n_k / n (empirical frequencies)"

    def __init__(self, k: int | None = None):
        self.k = k

    def estimator_variance(self, params: dict, n: int) -> dict[str, float]:
        """Marginally each count is binomial: Var(p_hat_k) = p_k(1-p_k)/n."""
        return {
            f"p{j}": p * (1 - p) / n for j, p in enumerate(np.asarray(params["p"]))
        }

    def _k(self, X) -> int:
        return self.k if self.k is not None else int(np.asarray(X).max()) + 1

    def nll(self, params: dict, X) -> float:
        X = np.asarray(X, dtype=int).ravel()
        probs = np.asarray(params["p"], dtype=float)
        return float(-np.mean(np.log(np.clip(probs[X], _EPS, None))))

    def mle(self, X) -> dict:
        X = np.asarray(X, dtype=int).ravel()
        counts = np.bincount(X, minlength=self._k(X)).astype(float)
        return {"p": counts / counts.sum()}

    def init_uparams(self, X) -> dict[str, np.ndarray]:
        return {"logits": np.zeros(self._k(np.asarray(X)))}  # uniform

    def unll(self, uparams: dict, X, bk=np):
        # mean over samples of  logsumexp(logits) - logits[x_i]
        logits = uparams["logits"]
        m = bk.max(logits)
        lse = m + bk.log(bk.sum(bk.exp(logits - m)))
        return lse - bk.mean(logits[X])

    def unll_grads(self, uparams: dict, X) -> dict[str, np.ndarray]:
        # d/dlogits = softmax(logits) - empirical frequencies
        logits = uparams["logits"]
        e = np.exp(logits - logits.max())
        softmax = e / e.sum()
        X = np.asarray(X, dtype=int).ravel()
        freq = np.bincount(X, minlength=len(logits)) / len(X)
        return {"logits": softmax - freq}

    def to_natural(self, uparams: dict) -> dict:
        logits = np.asarray(uparams["logits"], dtype=float)
        e = np.exp(logits - logits.max())
        return {"p": e / e.sum()}


@register_distribution("gaussian")
class Gaussian:
    """x real, N(mu, sigma^2). The MLE variance is the biased 1/n one."""

    dtype = "float"
    formula = "mu_hat = x_bar,  sigma2_hat = (1/n) sum (x - x_bar)^2"

    def estimator_variance(self, params: dict, n: int) -> dict[str, float]:
        """Var(mu_hat) = sigma^2/n; asymptotically Var(sigma_hat) = sigma^2/(2n)."""
        var = params["sigma"] ** 2
        return {"mu": var / n, "sigma": var / (2 * n)}

    def nll(self, params: dict, X) -> float:
        X = np.asarray(X, dtype=float).ravel()
        mu, var = params["mu"], max(params["sigma"] ** 2, _EPS)
        return float(0.5 * np.log(2 * np.pi * var) + np.mean((X - mu) ** 2) / (2 * var))

    def mle(self, X) -> dict:
        X = np.asarray(X, dtype=float).ravel()
        mu = float(X.mean())
        return {"mu": mu, "sigma": float(np.sqrt(np.mean((X - mu) ** 2)))}

    def init_uparams(self, X) -> dict[str, np.ndarray]:
        # start at the sample mean so the search is well-conditioned
        return {
            "mu": np.array([float(np.asarray(X, dtype=float).mean())]),
            "log_sigma": np.zeros(1),
        }

    def unll(self, uparams: dict, X, bk=np):
        # 0.5 log(2 pi) + log_sigma + mean((x - mu)^2) / (2 e^{2 log_sigma})
        mu, ls = uparams["mu"][0], uparams["log_sigma"][0]
        return (
            0.5 * np.log(2 * np.pi)
            + ls
            + bk.mean((X - mu) ** 2) / (2.0 * bk.exp(2.0 * ls))
        )

    def unll_grads(self, uparams: dict, X) -> dict[str, np.ndarray]:
        # d/dmu = (mu - x_bar)/sigma^2 ;  d/dlog_sigma = 1 - mean((x-mu)^2)/sigma^2
        X = np.asarray(X, dtype=float).ravel()
        mu, ls = uparams["mu"], uparams["log_sigma"]
        var = np.exp(2.0 * ls)
        return {
            "mu": (mu - X.mean()) / var,
            "log_sigma": 1.0 - np.mean((X - mu) ** 2) / var,
        }

    def to_natural(self, uparams: dict) -> dict:
        return {
            "mu": float(np.asarray(uparams["mu"])[0]),
            "sigma": float(np.exp(np.asarray(uparams["log_sigma"]))[0]),
        }
