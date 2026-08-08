import numpy as np
import pytest

from mlr.data.synthetic import bernoulli_samples, gaussian_samples, multinoulli_samples
from mlr.models.pure.mle import BernoulliMLE, GaussianMLE, MultinoulliMLE


def test_bernoulli_mle_is_sample_mean():
    X = np.array([1, 1, 0, 1, 0])
    model = BernoulliMLE().fit(X)
    assert model.p == pytest.approx(0.6)
    assert model.estimates() == {"p": pytest.approx(0.6)}


def test_bernoulli_recovers_true_p():
    X = bernoulli_samples(n=20000, p=0.3, seed=0)
    model = BernoulliMLE().fit(X)
    assert model.p == pytest.approx(0.3, abs=0.02)


def test_multinoulli_mle_is_empirical_frequency():
    X = np.array([0, 0, 1, 2, 2, 2])
    model = MultinoulliMLE().fit(X)
    assert np.allclose(model.probs, [2 / 6, 1 / 6, 3 / 6])
    assert model.probs.sum() == pytest.approx(1.0)


def test_multinoulli_recovers_true_probs():
    X = multinoulli_samples(n=20000, probs=(0.5, 0.3, 0.2), seed=0)
    model = MultinoulliMLE().fit(X)
    assert np.allclose(model.probs, [0.5, 0.3, 0.2], atol=0.02)


def test_gaussian_mle_uses_biased_variance():
    X = np.array([1.0, 2.0, 3.0])
    model = GaussianMLE().fit(X)
    assert model.mu == pytest.approx(2.0)
    assert model.sigma**2 == pytest.approx(2.0 / 3.0)  # 1/n, not 1/(n-1)


def test_gaussian_recovers_true_params():
    X = gaussian_samples(n=20000, mu=2.0, sigma=1.5, seed=0)
    model = GaussianMLE().fit(X)
    assert model.estimates()["mu"] == pytest.approx(2.0, abs=0.05)
    assert model.estimates()["sigma"] == pytest.approx(1.5, abs=0.05)


def test_nll_is_minimized_at_mle():
    """The defining property: no other parameter beats the MLE in-sample."""
    X = bernoulli_samples(n=5000, p=0.3, seed=1)
    fitted = BernoulliMLE().fit(X)
    other = BernoulliMLE()
    other.p = 0.5
    assert fitted.nll(X) < other.nll(X)


def test_run_experiment_unsupervised_logs_nll_and_estimates(tmp_path):
    import mlr  # noqa: F401
    from mlr.tracking import Tracker
    from mlr.training import run_experiment

    config = {
        "name": "bern",
        "topic": "mle",
        "model": "bernoulli-mle",
        "dataset": "bernoulli-samples",
        "dataset_params": {"n": 2000, "p": 0.3, "seed": 0},
        "test_size": 0.2,
        "seed": 42,
    }
    with Tracker(tmp_path / "t.db") as tr:
        run_id = run_experiment(config, tr)
        assert tr.last_metric(run_id, "test_nll") is not None
        assert tr.last_metric(run_id, "estimate.p") == pytest.approx(0.3, abs=0.05)
        assert tr.last_metric(run_id, "test_accuracy") is None
