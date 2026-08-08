import numpy as np
import pytest

from mlr.data.synthetic import bernoulli_samples, gaussian_samples, multinoulli_samples
from mlr.distributions import get_distribution, list_distributions


def _sample(name):
    return {
        "bernoulli": bernoulli_samples(n=2000, p=0.3, seed=1),
        "multinoulli": multinoulli_samples(n=2000, probs=(0.5, 0.3, 0.2), seed=1).astype(int),
        "gaussian": gaussian_samples(n=2000, mu=2.0, sigma=1.5, seed=1),
    }[name]


def test_registry():
    assert list_distributions() == ["bernoulli", "gaussian", "multinoulli"]
    with pytest.raises(KeyError, match="unknown distribution"):
        get_distribution("nope")


def test_closed_forms():
    X = np.array([1, 1, 0, 1, 0], dtype=float)
    assert get_distribution("bernoulli").mle(X)["p"] == pytest.approx(0.6)

    X = np.array([0, 0, 1, 2, 2, 2])
    assert np.allclose(get_distribution("multinoulli").mle(X)["p"], [2 / 6, 1 / 6, 3 / 6])

    X = np.array([1.0, 2.0, 3.0])
    est = get_distribution("gaussian").mle(X)
    assert est["mu"] == pytest.approx(2.0)
    assert est["sigma"] ** 2 == pytest.approx(2.0 / 3.0)  # 1/n (biased) — that IS the MLE


@pytest.mark.parametrize("name", ["bernoulli", "multinoulli", "gaussian"])
def test_mle_minimizes_nll(name):
    """The defining property, checked against parameter perturbations."""
    dist = get_distribution(name)
    X = _sample(name)
    best = dist.mle(X)
    base = dist.nll(best, X)
    rng = np.random.default_rng(0)
    for _ in range(20):
        other = {}
        for key, value in best.items():
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            perturbed = np.clip(arr + rng.normal(scale=0.05, size=arr.shape), 1e-6, None)
            if key == "p" and arr.size > 1:
                perturbed = perturbed / perturbed.sum()
            other[key] = perturbed if arr.size > 1 else float(perturbed[0])
        assert dist.nll(other, X) >= base - 1e-12


@pytest.mark.parametrize("name", ["bernoulli", "multinoulli", "gaussian"])
def test_unll_grads_match_finite_differences(name):
    """Hand-derived gradients must agree with numerical differentiation."""
    dist = get_distribution(name)
    X = _sample(name)
    uparams = dist.init_uparams(X)
    for key in uparams:  # move off the (sometimes symmetric) init point
        uparams[key] = uparams[key] + 0.3
    grads = dist.unll_grads(uparams, X)
    h = 1e-6
    for key, grad in grads.items():
        for i in range(len(np.atleast_1d(uparams[key]))):
            up = {k: v.copy() for k, v in uparams.items()}
            up[key][i] += h
            down = {k: v.copy() for k, v in uparams.items()}
            down[key][i] -= h
            numeric = (dist.unll(up, X, np) - dist.unll(down, X, np)) / (2 * h)
            assert np.atleast_1d(grad)[i] == pytest.approx(numeric, abs=1e-5), (key, i)


@pytest.mark.parametrize("name", ["bernoulli", "multinoulli", "gaussian"])
def test_unll_agrees_with_nll_through_to_natural(name):
    """unll at uparams == nll at the mapped natural parameters."""
    dist = get_distribution(name)
    X = _sample(name)
    uparams = {k: v + 0.2 for k, v in dist.init_uparams(X).items()}
    assert float(dist.unll(uparams, X, np)) == pytest.approx(
        dist.nll(dist.to_natural(uparams), X), abs=1e-9
    )


@pytest.mark.parametrize("name", ["bernoulli", "multinoulli", "gaussian"])
def test_unll_torch_backend_matches_numpy(name):
    torch = pytest.importorskip("torch")
    dist = get_distribution(name)
    X = _sample(name)
    uparams = {k: v + 0.2 for k, v in dist.init_uparams(X).items()}
    X_t = torch.as_tensor(X, dtype=torch.long if dist.dtype == "int" else torch.float32)
    uparams_t = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in uparams.items()}
    assert float(dist.unll(uparams_t, X_t, torch)) == pytest.approx(
        float(dist.unll(uparams, X, np)), abs=1e-5
    )
