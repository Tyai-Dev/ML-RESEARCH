import numpy as np

from mlr.data.synthetic import two_gaussians
from mlr.models.pure.linear_discriminator import LinearDiscriminator


def test_learns_separable_data():
    X, y = two_gaussians(n=1000, sep=4.0, seed=1)
    model = LinearDiscriminator(lr=0.5, epochs=300, seed=0)
    model.fit(X, y)
    assert model.accuracy(X, y) > 0.95


def test_on_epoch_callback_reports_decreasing_loss():
    X, y = two_gaussians(n=500, sep=3.0, seed=2)
    losses = []
    model = LinearDiscriminator(lr=0.5, epochs=100, seed=0)
    model.fit(X, y, on_epoch=lambda e, m: losses.append(m["train_loss"]))
    assert len(losses) == 100
    assert losses[-1] < losses[0]


def test_predict_proba_in_unit_interval():
    X, y = two_gaussians(n=200, seed=3)
    model = LinearDiscriminator(epochs=50).fit(X, y)
    p = model.predict_proba(X)
    assert np.all((p >= 0) & (p <= 1))


def test_deterministic_given_seed():
    X, y = two_gaussians(n=200, seed=4)
    m1 = LinearDiscriminator(epochs=50, seed=7).fit(X, y)
    m2 = LinearDiscriminator(epochs=50, seed=7).fit(X, y)
    assert np.allclose(m1.w, m2.w) and m1.b == m2.b
