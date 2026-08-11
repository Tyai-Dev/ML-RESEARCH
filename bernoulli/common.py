r"""Shared setup for the Bernoulli MLE experiments.

One place for the data, the configuration, and the two functions every
route needs — so mle_theoretical, mle_practical_pure, and
mle_practical_pytorch all estimate on the SAME sample and are directly
comparable (the pytorch file even asserts trajectory identity against
the pure file, which requires the identical sample schedule).
"""

import numpy as np

P_TRUE = 0.3        # the parameter we pretend not to know
N = 5_000           # sample size
SEED = 7            # reproducibility: data + SGD sample order

GD_LR, GD_STEPS = 1.0, 150
SGD_LR, SGD_EPOCHS = 0.1, 3


def make_data(seed: int = SEED) -> tuple[np.ndarray, np.random.Generator]:
    """The dataset x_1..x_N ~ Bernoulli(P_TRUE), plus the generator (so
    callers can draw the SGD schedule from the same random stream)."""
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, P_TRUE, size=N).astype(np.float64)
    return x, rng


def make_schedule(rng: np.random.Generator) -> np.ndarray:
    """The SGD sample order: SGD_EPOCHS reshuffled passes over the data.
    Both SGD implementations iterate this exact sequence."""
    return np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


def sigmoid(t):
    """sigma(t) = 1 / (1 + e^{-t}), mapping the real line onto (0, 1)."""
    return 1.0 / (1.0 + np.exp(-t))


def nll_of_p(p, xbar: float):
    """The average negative log-likelihood as a function of p in (0,1):
    NLL(p) = -x̄ log p - (1-x̄) log(1-p).  (Derivation: bernoulli.tex.)"""
    return -(xbar * np.log(p) + (1 - xbar) * np.log(1 - p))
