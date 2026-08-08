import numpy as np
import pytest

from mlr.data.synthetic import bernoulli_samples, gaussian_samples, multinoulli_samples
from mlr.models.mle import MLEModel

DATA = {
    "bernoulli": lambda: bernoulli_samples(n=5000, p=0.3, seed=0),
    "multinoulli": lambda: multinoulli_samples(n=5000, probs=(0.5, 0.3, 0.2), seed=0),
    "gaussian": lambda: gaussian_samples(n=5000, mu=2.0, sigma=1.5, seed=0),
}


@pytest.mark.parametrize("name", DATA)
def test_closed_form_recovers_true_params(name):
    model = MLEModel(distribution=name).fit(DATA[name]())
    est = model.estimates()
    if name == "bernoulli":
        assert est["p"] == pytest.approx(0.3, abs=0.02)
    elif name == "multinoulli":
        assert [est["p0"], est["p1"], est["p2"]] == pytest.approx([0.5, 0.3, 0.2], abs=0.02)
    else:
        assert est["mu"] == pytest.approx(2.0, abs=0.05)
        assert est["sigma"] == pytest.approx(1.5, abs=0.05)


@pytest.mark.parametrize("name", DATA)
@pytest.mark.parametrize("backend", ["pure", "torch"])
def test_gradient_matches_closed_form(name, backend):
    """Concave log-likelihoods: every fitting route reaches the same optimum."""
    if backend == "torch":
        pytest.importorskip("torch")
    X = DATA[name]()
    closed = MLEModel(distribution=name).fit(X)
    grad = MLEModel(distribution=name, method="gradient", backend=backend).fit(
        X, optimizer="adam", lr=0.1, epochs=500
    )
    assert grad.nll(X) == pytest.approx(closed.nll(X), abs=1e-3)
    for key, value in closed.estimates().items():
        assert grad.estimates()[key] == pytest.approx(value, abs=0.01), key


def test_gradient_reports_decreasing_nll():
    X = DATA["gaussian"]()
    nlls = []
    MLEModel(distribution="gaussian", method="gradient", backend="pure").fit(
        X, optimizer="adam", lr=0.05, epochs=300,
        on_epoch=lambda e, m: nlls.append(m["train_nll"]),
    )
    assert len(nlls) == 300
    assert nlls[-1] < nlls[0]


def test_closed_form_rejects_training_section():
    with pytest.raises(ValueError, match="closed-form"):
        MLEModel(distribution="bernoulli").fit(DATA["bernoulli"](), optimizer="adam")


def test_invalid_method_or_backend():
    with pytest.raises(ValueError, match="method"):
        MLEModel(distribution="bernoulli", method="magic")
    with pytest.raises(ValueError, match="backend"):
        MLEModel(distribution="bernoulli", backend="jax")


def test_run_experiment_unsupervised_logs_nll_and_estimates(tmp_path):
    import mlr  # noqa: F401
    from mlr.tracking import Tracker
    from mlr.training import run_experiment

    config = {
        "name": "bern",
        "topic": "mle",
        "model": "mle",
        "model_params": {"distribution": "bernoulli"},
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
