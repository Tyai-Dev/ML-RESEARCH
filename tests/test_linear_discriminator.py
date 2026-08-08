import numpy as np
import pytest

from mlr.data.synthetic import two_gaussians
from mlr.models.pure.linear_discriminator import LinearDiscriminator

OPTIMIZERS = {
    "sgd": {"lr": 0.5},
    "momentum": {"lr": 0.1, "momentum": 0.9},
    "adam": {"lr": 0.05},
    "rmsprop": {"lr": 0.05},
    "adagrad": {"lr": 0.5},
}


@pytest.mark.parametrize("optimizer,params", OPTIMIZERS.items())
def test_learns_separable_data_with_every_optimizer(optimizer, params):
    X, y = two_gaussians(n=1000, sep=4.0, seed=1)
    model = LinearDiscriminator(seed=0)
    model.fit(X, y, optimizer=optimizer, epochs=200, **params)
    assert model.accuracy(X, y) > 0.95, optimizer


def test_minibatch_sgd_learns():
    X, y = two_gaussians(n=1000, sep=3.0, seed=2)
    model = LinearDiscriminator(seed=0)
    model.fit(X, y, optimizer="sgd", lr=0.2, epochs=30, batch_size=32)
    assert model.accuracy(X, y) > 0.9


def test_on_epoch_callback_reports_decreasing_loss():
    X, y = two_gaussians(n=500, sep=3.0, seed=2)
    losses = []
    model = LinearDiscriminator(seed=0)
    model.fit(
        X, y, optimizer="sgd", lr=0.5, epochs=100,
        on_epoch=lambda e, m: losses.append(m["train_loss"]),
    )
    assert len(losses) == 100
    assert losses[-1] < losses[0]


def test_predict_proba_in_unit_interval():
    X, y = two_gaussians(n=200, seed=3)
    model = LinearDiscriminator(seed=0).fit(X, y, epochs=50)
    p = model.predict_proba(X)
    assert np.all((p >= 0) & (p <= 1))


def test_deterministic_given_seed():
    X, y = two_gaussians(n=200, seed=4)
    m1 = LinearDiscriminator(seed=7).fit(X, y, epochs=50, batch_size=32)
    m2 = LinearDiscriminator(seed=7).fit(X, y, epochs=50, batch_size=32)
    assert np.allclose(m1.params["w"], m2.params["w"])
    assert np.allclose(m1.params["b"], m2.params["b"])
