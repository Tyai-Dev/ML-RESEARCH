"""Regenerate this paper's tables and figures from the experiment tracker.

Run via ``mlr paper research/linear-discriminator/papers/02-loss-based-training``
(or directly with the project venv's python, cwd anywhere). Reads the latest
finished run of each experiment name under the ``linear-discriminator`` topic
and writes:

    ../../results/tables/accuracy.tex     booktabs rows for Table 1
    ../../results/figures/loss-curve.pdf  training-loss curves
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlr.tracking import Tracker

PAPER_DIR = Path(__file__).resolve().parent
TOPIC_DIR = PAPER_DIR.parents[1]  # research/linear-discriminator
REPO_ROOT = TOPIC_DIR.parents[1]  # repo root, where tracking.db lives

# Validated categorical palette slots 1-3 (see dataviz reference palette);
# linestyle is the secondary encoding so identity survives grayscale print.
SERIES_STYLE = [
    {"color": "#2a78d6", "linestyle": "-"},
    {"color": "#eb6834", "linestyle": "--"},
    {"color": "#1baf7a", "linestyle": ":"},
]
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def latest_finished_runs(tracker: Tracker, topic: str) -> dict[str, dict]:
    """Latest finished run per experiment name, keyed and sorted by name."""
    latest: dict[str, dict] = {}
    for run in tracker.list_runs(topic):
        if run["status"] == "finished":
            latest[run["name"]] = run  # list_runs is id-ordered; last wins
    return dict(sorted(latest.items()))


def write_accuracy_table(tracker: Tracker, runs: dict[str, dict], out: Path) -> None:
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Experiment & Separation & Test accuracy \\",
        r"\midrule",
    ]
    for name, run in runs.items():
        sep = tracker.get_params(run["id"]).get("dataset.sep", "--")
        acc = tracker.last_metric(run["id"], "test_accuracy")
        lines.append(rf"{name} & {sep} & {acc:.4f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def write_loss_figure(tracker: Tracker, runs: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for style, (name, run) in zip(SERIES_STYLE, runs.items()):
        history = tracker.metric_history(run["id"], "train_loss")
        steps = [s for s, _ in history]
        values = [v for _, v in history]
        ax.plot(steps, values, label=name, linewidth=2, **style)

    ax.set_xlabel("epoch", color=INK_MUTED)
    ax.set_ylabel("training loss", color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.legend(frameon=False, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    tables = TOPIC_DIR / "results" / "tables"
    figures = TOPIC_DIR / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    with Tracker(REPO_ROOT / "tracking.db") as tracker:
        runs = latest_finished_runs(tracker, "linear-discriminator")
        if not runs:
            raise SystemExit(
                "no finished runs for topic 'linear-discriminator'; "
                "run the configs under research/linear-discriminator/experiments/ first"
            )
        if len(runs) > len(SERIES_STYLE):
            raise SystemExit(
                f"{len(runs)} experiments but only {len(SERIES_STYLE)} series styles; "
                "add validated palette slots before plotting more series"
            )
        write_accuracy_table(tracker, runs, tables / "accuracy.tex")
        write_loss_figure(tracker, runs, figures / "loss-curve.pdf")
    print(f"wrote {tables / 'accuracy.tex'} and {figures / 'loss-curve.pdf'}")


if __name__ == "__main__":
    main()
