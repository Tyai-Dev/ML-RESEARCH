import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlr.data.synthetic import bernoulli_samples, gaussian_samples, multinoulli_samples
from mlr.models.pure.mle import BernoulliMLE, GaussianMLE, MultinoulliMLE
from mlr.models.torch.mle import BernoulliMLETorch, GaussianMLETorch, MultinoulliMLETorch


def test_bernoulli_gradient_matches_closed_form():
    X = bernoulli_samples(n=5000, p=0.3, seed=0)
    closed = BernoulliMLE().fit(X)
    grad = BernoulliMLETorch().fit(X, optimizer="adam", lr=0.1, epochs=300)
    assert grad.estimates()["p"] == pytest.approx(closed.p, abs=0.01)


def test_multinoulli_gradient_matches_closed_form():
    X = multinoulli_samples(n=5000, probs=(0.5, 0.3, 0.2), seed=0)
    closed = MultinoulliMLE().fit(X)
    grad = MultinoulliMLETorch().fit(X, optimizer="adam", lr=0.1, epochs=400)
    grad_probs = [grad.estimates()[f"p{k}"] for k in range(3)]
    assert np.allclose(grad_probs, closed.probs, atol=0.01)


def test_gaussian_gradient_matches_closed_form():
    X = gaussian_samples(n=5000, mu=2.0, sigma=1.5, seed=0)
    closed = GaussianMLE().fit(X)
    grad = GaussianMLETorch().fit(X, optimizer="adam", lr=0.05, epochs=500)
    assert grad.estimates()["mu"] == pytest.approx(closed.mu, abs=0.02)
    assert grad.estimates()["sigma"] == pytest.approx(closed.sigma, abs=0.02)


def test_nll_decreases_and_matches_pure_at_convergence():
    X = gaussian_samples(n=2000, mu=-1.0, sigma=0.5, seed=1)
    nlls = []
    grad = GaussianMLETorch().fit(
        X, optimizer="adam", lr=0.05, epochs=400,
        on_epoch=lambda e, m: nlls.append(m["train_nll"]),
    )
    assert nlls[-1] < nlls[0]
    closed = GaussianMLE().fit(X)
    assert grad.nll(X) == pytest.approx(closed.nll(X), abs=1e-3)
