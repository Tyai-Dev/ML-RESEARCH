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


def test_cli_run_runs_best(tmp_path):
    import yaml
    from typer.testing import CliRunner

    from mlr.cli import app

    runner = CliRunner()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    db = str(tmp_path / "t.db")

    result = runner.invoke(app, ["run", str(config_path), "--db", db])
    assert result.exit_code == 0, result.output
    assert "test_accuracy=" in result.output

    result = runner.invoke(app, ["runs", "--db", db, "--topic", "linear-discriminator"])
    assert result.exit_code == 0, result.output
    assert "e2e" in result.output

    result = runner.invoke(app, ["best", "linear-discriminator", "--db", db])
    assert result.exit_code == 0, result.output
    assert "best test_accuracy" in result.output


def test_cli_lists_registered_names():
    from typer.testing import CliRunner

    from mlr.cli import app

    runner = CliRunner()
    assert "linear-discriminator" in runner.invoke(app, ["models"]).output
    assert "two-gaussians" in runner.invoke(app, ["datasets"]).output
