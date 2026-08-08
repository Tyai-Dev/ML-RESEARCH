"""The CLI's real work, independent of how it was invoked.

Both the typer commands (mlr.cli.app) and the interactive menu
(mlr.cli.interactive) call these functions, so behavior stays identical
between the two front ends.
"""

import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from mlr.tracking import Tracker
from mlr.training import run_experiment

console = Console()


def repo_root(start: Path | None = None) -> Path:
    """Walk upward to the directory containing pyproject.toml."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "not inside the ML-RESEARCH repo (no pyproject.toml found upward of "
        f"{here})"
    )


def list_topics(root: Path) -> list[str]:
    research = root / "research"
    if not research.exists():
        return []
    return sorted(
        d.name
        for d in research.iterdir()
        if d.is_dir() and d.name != "papers-common"
    )


def list_experiments(root: Path, topic: str) -> list[Path]:
    exp_dir = root / "research" / topic / "experiments"
    return sorted(exp_dir.glob("*.yaml")) if exp_dir.exists() else []


def list_papers(root: Path, topic: str) -> list[Path]:
    papers_dir = root / "research" / topic / "papers"
    if not papers_dir.exists():
        return []
    return sorted(d for d in papers_dir.iterdir() if (d / "main.tex").exists())


def do_run(config_path: Path, db: str | Path) -> int:
    config = yaml.safe_load(Path(config_path).read_text())
    with Tracker(db) as tracker:
        run_id = run_experiment(config, tracker)
        test_acc = tracker.last_metric(run_id, "test_accuracy")
    acc_s = "-" if test_acc is None else f"{test_acc:.4f}"
    console.print(
        f"[green]run {run_id}[/green] ({config['name']}) finished; "
        f"test_accuracy={acc_s}"
    )
    return run_id


def do_runs(topic: str | None, db: str | Path) -> None:
    with Tracker(db) as tracker:
        runs = tracker.list_runs(topic=topic)
        if not runs:
            console.print("[yellow]no runs[/yellow]")
            return
        table = Table(title="Tracked runs" + (f" — {topic}" if topic else ""))
        for col in ("id", "topic", "name", "model", "dataset", "status", "test acc"):
            table.add_column(col)
        for run in runs:
            acc = tracker.last_metric(run["id"], "test_accuracy")
            table.add_row(
                str(run["id"]),
                run["topic"],
                run["name"],
                run["model"],
                run["dataset"],
                run["status"],
                "-" if acc is None else f"{acc:.4f}",
            )
        console.print(table)


def do_best(topic: str, metric: str, mode: str, db: str | Path) -> dict | None:
    with Tracker(db) as tracker:
        best = tracker.best_run(topic, metric, mode)
    if best is None:
        console.print(f"[yellow]no finished runs with metric {metric!r} in {topic!r}[/yellow]")
        return None
    console.print(
        f"[green]best {metric}[/green] in {topic}: run {best['id']} "
        f"({best['name']}) -> {best['value']:.4f}"
    )
    return best


def do_paper(paper_dir: Path) -> None:
    paper_dir = Path(paper_dir).resolve()
    if not (paper_dir / "main.tex").exists():
        raise FileNotFoundError(f"no main.tex in {paper_dir}")
    assets = paper_dir / "generate_assets.py"
    if assets.exists():
        console.print(f"generating assets: [cyan]{assets.name}[/cyan]")
        subprocess.run([sys.executable, str(assets)], check=True, cwd=paper_dir)
    # Two pdflatex passes resolve cross-references; add a bibtex pass here
    # if papers grow bibliographies (latexmk needs perl, so we avoid it).
    for i in (1, 2):
        console.print(f"pdflatex pass {i}...")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            check=True,
            cwd=paper_dir,
            stdout=subprocess.DEVNULL if i == 1 else None,
        )
    console.print(f"[green]built[/green] {paper_dir / 'main.pdf'}")


def do_models() -> list[str]:
    from mlr.models import list_models

    return list_models()


def do_datasets() -> list[str]:
    from mlr.data import list_datasets

    return list_datasets()
