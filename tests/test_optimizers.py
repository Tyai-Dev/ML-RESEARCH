import numpy as np
import pytest

from mlr.training.optimizers import build_optimizer, list_optimizers

ALL = ["sgd", "momentum", "adagrad", "rmsprop", "adam"]


def test_registry_lists_all():
    assert list_optimizers() == sorted(ALL)
    with pytest.raises(KeyError, match="unknown optimizer"):
        build_optimizer("nope")


@pytest.mark.parametrize("name", ALL)
def test_minimizes_quadratic(name):
    """Every optimizer should drive f(p) = ||p||^2 toward zero.

    Adagrad gets a larger base lr — its effective step decays like
    1/sqrt(sum g^2), which is the point of the algorithm.
    """
    params = {"p": np.array([3.0, -2.0])}
    opt = build_optimizer(name, lr=1.0 if name == "adagrad" else 0.1)
    for _ in range(300):
        grads = {"p": 2.0 * params["p"]}
        opt.step(params, grads)
    assert np.linalg.norm(params["p"]) < 0.1


@pytest.mark.parametrize("name", ALL)
def test_updates_in_place_and_handles_multiple_params(name):
    params = {"w": np.ones(3), "b": np.zeros(1)}
    w_ref = params["w"]
    opt = build_optimizer(name, lr=0.1)
    opt.step(params, {"w": np.ones(3), "b": np.ones(1)})
    assert params["w"] is w_ref  # mutated in place, not replaced
    assert not np.allclose(params["w"], 1.0)
    assert not np.allclose(params["b"], 0.0)


def test_momentum_accumulates_velocity():
    plain = {"p": np.array([1.0])}
    heavy = {"p": np.array([1.0])}
    sgd = build_optimizer("sgd", lr=0.1)
    mom = build_optimizer("momentum", lr=0.1, momentum=0.9)
    for _ in range(5):
        sgd.step(plain, {"p": np.array([1.0])})
        mom.step(heavy, {"p": np.array([1.0])})
    # constant gradient => momentum has moved strictly further
    assert heavy["p"][0] < plain["p"][0]


def test_adam_bias_correction_first_step():
    params = {"p": np.array([0.0])}
    opt = build_optimizer("adam", lr=0.01)
    opt.step(params, {"p": np.array([1e-3])})
    # with bias correction the first step magnitude is ~lr regardless of tiny grads
    assert abs(params["p"][0] + 0.01) < 1e-3
