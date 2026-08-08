"""Regenerate this paper's table and figure from the experiment tracker.

Reads the latest finished run of each experiment under the ``mle`` topic and
writes:

    ../../results/tables/mle-estimates.tex   estimates + held-out NLL table
    ../../results/figures/gaussian-fit.pdf   histogram + fitted densities
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mlr  # noqa: F401  (registers datasets)
from mlr.data import get_dataset
from mlr.tracking import Tracker

PAPER_DIR = Path(__file__).resolve().parent
TOPIC_DIR = PAPER_DIR.parents[1]  # research/mle
REPO_ROOT = TOPIC_DIR.parents[1]

# Validated categorical palette slots 1-2 + chrome inks (dataviz reference);
# linestyle separates the two fits when the curves coincide.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK_MUTED, GRIDLINE, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"


def latest_finished_runs(tracker: Tracker, topic: str) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for run in tracker.list_runs(topic):
        if run["status"] == "finished":
            latest[run["name"]] = run  # id-ordered; last wins
    return dict(sorted(latest.items()))


def run_estimates(tracker: Tracker, run_id: int) -> dict[str, float]:
    return {
        key.removeprefix("estimate."): tracker.last_metric(run_id, key)
        for key in tracker.metric_keys(run_id)
        if key.startswith("estimate.")
    }


def true_params(tracker: Tracker, run_id: int) -> str:
    params = tracker.get_params(run_id)
    keep = {
        k.removeprefix("dataset."): v
        for k, v in params.items()
        if k.startswith("dataset.") and k not in ("dataset.n", "dataset.seed")
    }
    return ", ".join(f"{k}={v}" for k, v in sorted(keep.items()))


def write_table(tracker: Tracker, runs: dict[str, dict], out: Path) -> None:
    lines = [
        r"\begin{tabular}{llrl}",
        r"\toprule",
        r"Experiment & True parameters & Test NLL & MLE estimates \\",
        r"\midrule",
    ]
    for name, run in runs.items():
        nll = tracker.last_metric(run["id"], "test_nll")
        est = run_estimates(tracker, run["id"])
        est_s = ", ".join(f"{k}={v:.3f}" for k, v in sorted(est.items()))
        lines.append(
            rf"{name} & {true_params(tracker, run['id'])} & {nll:.4f} & {est_s} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def write_gaussian_figure(tracker: Tracker, runs: dict[str, dict], out: Path) -> None:
    pure, tch = runs.get("gaussian-pure"), runs.get("gaussian-torch")
    if pure is None or tch is None:
        raise SystemExit("need finished gaussian-pure and gaussian-torch runs")
    params = tracker.get_params(pure["id"])
    n = int(params["dataset.n"])
    mu_true = float(params["dataset.mu"])
    sigma_true = float(params["dataset.sigma"])
    seed = int(params["dataset.seed"])
    X = get_dataset("gaussian-samples")(n=n, mu=mu_true, sigma=sigma_true, seed=seed)

    def normal_pdf(x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    grid = np.linspace(X.min(), X.max(), 400)
    est_p = run_estimates(tracker, pure["id"])
    est_t = run_estimates(tracker, tch["id"])

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.hist(
        X, bins=40, density=True, color=GRIDLINE, edgecolor="white", label="samples"
    )
    ax.plot(
        grid, normal_pdf(grid, mu_true, sigma_true),
        color=INK_MUTED, linestyle="--", linewidth=2, label="true density",
    )
    ax.plot(
        grid, normal_pdf(grid, est_p["mu"], est_p["sigma"]),
        color=BLUE, linewidth=2, label="closed-form MLE",
    )
    ax.plot(
        grid, normal_pdf(grid, est_t["mu"], est_t["sigma"]),
        color=ORANGE, linestyle=":", linewidth=2, label="gradient MLE (torch)",
    )
    ax.set_xlabel("x", color=INK_MUTED)
    ax.set_ylabel("density", color=INK_MUTED)
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
        runs = latest_finished_runs(tracker, "mle")
        if not runs:
            raise SystemExit(
                "no finished runs for topic 'mle'; run the configs under "
                "research/mle/experiments/ first"
            )
        write_table(tracker, runs, tables / "mle-estimates.tex")
        write_gaussian_figure(tracker, runs, figures / "gaussian-fit.pdf")
    print(f"wrote {tables / 'mle-estimates.tex'} and {figures / 'gaussian-fit.pdf'}")


if __name__ == "__main__":
    main()
