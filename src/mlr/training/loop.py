"""Config-driven experiment runner.

A config (usually a YAML file under ``research/<topic>/experiments/``) names
a registered model and dataset plus their parameters, and a ``training``
section for the optimization recipe:

    name: baseline
    topic: linear-discriminator
    model: linear-discriminator          # or linear-discriminator-torch
    model_params: {seed: 0}
    dataset: two-gaussians
    dataset_params: {n: 1000, sep: 2.0, seed: 42}
    training:
      optimizer: adam                    # sgd | momentum | adam | rmsprop | adagrad
      lr: 0.05
      epochs: 300
      batch_size: 32                     # null / omitted = full-batch GD
    test_size: 0.2
    seed: 42
"""

import numpy as np

from mlr.data import get_dataset
from mlr.models import get_model
from mlr.tracking import Tracker


def train_test_split(X, y, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_test = int(round(n * test_size))
    test, train = perm[:n_test], perm[n_test:]
    return X[train], y[train], X[test], y[test]


def run_experiment(config: dict, tracker: Tracker, on_epoch=None) -> int:
    """Train per config, log everything to the tracker, return the run id.

    ``on_epoch(epoch, metrics)``, if given, is called alongside tracker
    logging — the CLI uses it to render live training progress.
    """
    name = config["name"]
    topic = config["topic"]
    model_name = config["model"]
    dataset_name = config["dataset"]
    model_params = config.get("model_params", {})
    dataset_params = config.get("dataset_params", {})
    training = dict(config.get("training", {}))
    training = {k: v for k, v in training.items() if v is not None}
    test_size = config.get("test_size", 0.2)
    seed = config.get("seed", 0)

    X, y = get_dataset(dataset_name)(**dataset_params)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size, seed)
    model = get_model(model_name)(**model_params)

    run_id = tracker.start_run(name, topic, model_name, dataset_name)
    tracker.log_params(
        run_id,
        {
            **{f"model.{k}": v for k, v in model_params.items()},
            **{f"dataset.{k}": v for k, v in dataset_params.items()},
            **{f"training.{k}": v for k, v in training.items()},
            "test_size": test_size,
            "seed": seed,
        },
    )
    try:
        def log_epoch(epoch: int, metrics: dict) -> None:
            for key, value in metrics.items():
                tracker.log_metric(run_id, key, value, step=epoch)
            if on_epoch is not None:
                on_epoch(epoch, metrics)

        model.fit(X_train, y_train, on_epoch=log_epoch, **training)
        tracker.log_metric(run_id, "test_accuracy", model.accuracy(X_test, y_test))
        tracker.finish_run(run_id)
    except Exception:
        tracker.finish_run(run_id, status="failed")
        raise
    return run_id
