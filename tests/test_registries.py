import pytest

import mlr  # noqa: F401  (triggers built-in registrations)
from mlr.data import get_dataset, list_datasets, register_dataset
from mlr.models import get_model, list_models, register_model


def test_builtins_are_registered():
    assert "linear-discriminator" in list_models()
    assert "two-gaussians" in list_datasets()


def test_unknown_names_raise():
    with pytest.raises(KeyError, match="unknown model"):
        get_model("nope")
    with pytest.raises(KeyError, match="unknown dataset"):
        get_dataset("nope")


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_model("linear-discriminator")(object)
    with pytest.raises(ValueError, match="already registered"):
        register_dataset("two-gaussians")(lambda: None)
