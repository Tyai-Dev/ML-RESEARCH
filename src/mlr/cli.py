"""mlr command line.

    mlr run <config.yaml> [--db tracking.db]   train + track an experiment
    mlr runs [--topic T] [--db tracking.db]    list tracked runs
    mlr paper <paper-dir>                      regenerate assets + compile PDF
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from mlr.tracking import Tracker
from mlr.training import run_experiment


def _cmd_run(args) -> int:
    config = yaml.safe_load(Path(args.config).read_text())
    with Tracker(args.db) as tracker:
        run_id = run_experiment(config, tracker)
        test_acc = tracker.last_metric(run_id, "test_accuracy")
    print(f"run {run_id} ({config['name']}) finished; test_accuracy={test_acc:.4f}")
    return 0


def _cmd_runs(args) -> int:
    with Tracker(args.db) as tracker:
        runs = tracker.list_runs(topic=args.topic)
        if not runs:
            print("no runs")
            return 0
        for run in runs:
            acc = tracker.last_metric(run["id"], "test_accuracy")
            acc_s = "-" if acc is None else f"{acc:.4f}"
            print(
                f"{run['id']:>4}  {run['topic']}/{run['name']}"
                f"  model={run['model']}  dataset={run['dataset']}"
                f"  status={run['status']}  test_accuracy={acc_s}"
            )
    return 0


def _cmd_paper(args) -> int:
    paper_dir = Path(args.paper_dir).resolve()
    if not (paper_dir / "main.tex").exists():
        print(f"error: no main.tex in {paper_dir}", file=sys.stderr)
        return 1
    assets = paper_dir / "generate_assets.py"
    if assets.exists():
        print(f"generating assets: {assets}")
        subprocess.run([sys.executable, str(assets)], check=True, cwd=paper_dir)
    # Two pdflatex passes resolve cross-references; add a bibtex pass here
    # if papers grow bibliographies (latexmk needs perl, so we avoid it).
    for i in (1, 2):
        print(f"pdflatex pass {i}...")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            check=True,
            cwd=paper_dir,
            stdout=subprocess.DEVNULL if i == 1 else None,
        )
    print(f"built {paper_dir / 'main.pdf'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mlr")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="train + track an experiment from a config")
    p_run.add_argument("config")
    p_run.add_argument("--db", default="tracking.db")
    p_run.set_defaults(fn=_cmd_run)

    p_runs = sub.add_parser("runs", help="list tracked runs")
    p_runs.add_argument("--topic", default=None)
    p_runs.add_argument("--db", default="tracking.db")
    p_runs.set_defaults(fn=_cmd_runs)

    p_paper = sub.add_parser("paper", help="regenerate assets and compile a paper")
    p_paper.add_argument("paper_dir")
    p_paper.set_defaults(fn=_cmd_paper)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
