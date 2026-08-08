import mlr  # noqa: F401  (registers built-ins)
from mlr.tracking import Tracker
from mlr.training import run_experiment

CONFIG = {
    "name": "e2e",
    "topic": "linear-discriminator",
    "model": "linear-discriminator",
    "model_params": {"lr": 0.5, "epochs": 100},
    "dataset": "two-gaussians",
    "dataset_params": {"n": 600, "sep": 3.0, "seed": 42},
    "test_size": 0.2,
    "seed": 42,
}


def test_run_experiment_logs_everything(tmp_path):
    with Tracker(tmp_path / "t.db") as tr:
        run_id = run_experiment(CONFIG, tr)

        (run,) = tr.list_runs("linear-discriminator")
        assert run["id"] == run_id and run["status"] == "finished"

        params = tr.get_params(run_id)
        assert params["model.lr"] == "0.5"
        assert params["dataset.n"] == "600"

        assert len(tr.metric_history(run_id, "train_loss")) == 100
        assert tr.last_metric(run_id, "test_accuracy") > 0.9


def test_cli_run_and_runs(tmp_path, capsys):
    import yaml

    from mlr.cli import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    db = str(tmp_path / "t.db")

    assert main(["run", str(config_path), "--db", db]) == 0
    assert "test_accuracy=" in capsys.readouterr().out

    assert main(["runs", "--db", db, "--topic", "linear-discriminator"]) == 0
    assert "linear-discriminator/e2e" in capsys.readouterr().out
