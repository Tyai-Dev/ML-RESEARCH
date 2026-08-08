import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlr.data.synthetic import two_gaussians
from mlr.models.torch.linear_discriminator import LinearDiscriminatorTorch

OPTIMIZERS = {
    "sgd": {"lr": 0.5},
    "momentum": {"lr": 0.1, "momentum": 0.9},
    "adam": {"lr": 0.05},
    "rmsprop": {"lr": 0.05},
    "adagrad": {"lr": 0.5},
}


def test_registered():
    import mlr  # noqa: F401
    from mlr.models import list_models

    assert "linear-discriminator-torch" in list_models()


@pytest.mark.parametrize("optimizer,params", OPTIMIZERS.items())
def test_learns_separable_data_with_every_optimizer(optimizer, params):
    X, y = two_gaussians(n=1000, sep=4.0, seed=1)
    model = LinearDiscriminatorTorch(seed=0)
    model.fit(X, y, optimizer=optimizer, epochs=200, **params)
    assert model.accuracy(X, y) > 0.95, optimizer


def test_minibatch_and_callback():
    X, y = two_gaussians(n=1000, sep=3.0, seed=2)
    losses = []
    model = LinearDiscriminatorTorch(seed=0)
    model.fit(
        X, y, optimizer="adam", lr=0.05, epochs=30, batch_size=32,
        on_epoch=lambda e, m: losses.append(m["train_loss"]),
    )
    assert len(losses) == 30
    assert losses[-1] < losses[0]
    assert model.accuracy(X, y) > 0.9


def test_unknown_optimizer_raises():
    X, y = two_gaussians(n=100, seed=3)
    with pytest.raises(KeyError, match="unknown optimizer"):
        LinearDiscriminatorTorch().fit(X, y, optimizer="nope")


def test_agrees_with_pure_model():
    """Same hypothesis class: both should find near-identical decision quality."""
    from mlr.models.pure.linear_discriminator import LinearDiscriminator

    X, y = two_gaussians(n=1000, sep=2.0, seed=5)
    pure = LinearDiscriminator(seed=0).fit(X, y, optimizer="sgd", lr=0.5, epochs=300)
    tch = LinearDiscriminatorTorch(seed=0).fit(X, y, optimizer="sgd", lr=0.5, epochs=300)
    assert abs(pure.accuracy(X, y) - tch.accuracy(X, y)) < 0.02
