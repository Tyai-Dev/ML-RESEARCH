from mlr.tracking import Tracker


def test_run_lifecycle(tmp_path):
    with Tracker(tmp_path / "t.db") as tr:
        run_id = tr.start_run("baseline", "topic-a", "model-x", "data-y")
        tr.log_params(run_id, {"lr": 0.1, "epochs": 10})
        tr.log_metric(run_id, "loss", 0.9, step=0)
        tr.log_metric(run_id, "loss", 0.5, step=1)
        tr.log_metric(run_id, "test_accuracy", 0.93)
        tr.log_artifact(run_id, "figure", "results/figures/loss.pdf")
        tr.finish_run(run_id)

        (run,) = tr.list_runs("topic-a")
        assert run["status"] == "finished"
        assert tr.get_params(run_id) == {"lr": "0.1", "epochs": "10"}
        assert tr.metric_history(run_id, "loss") == [(0, 0.9), (1, 0.5)]
        assert tr.last_metric(run_id, "loss") == 0.5
        assert tr.last_metric(run_id, "test_accuracy") == 0.93


def test_best_run_ignores_unfinished_and_respects_mode(tmp_path):
    with Tracker(tmp_path / "t.db") as tr:
        a = tr.start_run("a", "t", "m", "d")
        tr.log_metric(a, "test_accuracy", 0.90)
        tr.finish_run(a)

        b = tr.start_run("b", "t", "m", "d")
        tr.log_metric(b, "test_accuracy", 0.95)
        tr.finish_run(b)

        c = tr.start_run("c", "t", "m", "d")  # never finished
        tr.log_metric(c, "test_accuracy", 0.99)

        best = tr.best_run("t", "test_accuracy")
        assert best["name"] == "b" and best["value"] == 0.95

        worst = tr.best_run("t", "test_accuracy", mode="min")
        assert worst["name"] == "a"


def test_persistence_across_connections(tmp_path):
    path = tmp_path / "t.db"
    with Tracker(path) as tr:
        run_id = tr.start_run("a", "t", "m", "d")
        tr.finish_run(run_id)
    with Tracker(path) as tr:
        assert len(tr.list_runs()) == 1
