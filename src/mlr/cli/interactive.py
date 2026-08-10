"""Interactive menu — what `mlr` with no arguments drops into.

A thin questionary layer over mlr.cli.actions; anything it can do, the
subcommands can do too.
"""

import sys
from pathlib import Path

import questionary

from mlr.cli import actions, scaffold
from mlr.cli.actions import console

DB = "tracking.db"
_BACK = "(back)"


def _select(message: str, choices: list[str]) -> str | None:
    """None means the user backed out (Esc/Ctrl+C or picked back)."""
    if not choices:
        console.print("[yellow]nothing to pick from[/yellow]")
        return None
    answer = questionary.select(message, choices=[*choices, _BACK]).ask()
    return None if answer in (None, _BACK) else answer


def _pick_topic(root: Path) -> str | None:
    return _select("Topic:", actions.list_topics(root))


def _run_experiment(root: Path) -> None:
    topic = _pick_topic(root)
    if topic is None:
        return
    configs = {p.stem: p for p in actions.list_experiments(root, topic)}
    name = _select("Experiment:", list(configs))
    if name is not None:
        actions.do_run(configs[name], DB)


def _build_paper(root: Path) -> None:
    topic = _pick_topic(root)
    if topic is None:
        return
    papers = {p.name: p for p in actions.list_papers(root, topic)}
    name = _select("Paper:", list(papers))
    if name is not None:
        actions.do_paper(papers[name])


def _show_runs(root: Path) -> None:
    topic = _select("Topic:", ["(all)", *actions.list_topics(root)])
    if topic is not None:
        actions.do_runs(None if topic == "(all)" else topic, DB)


def _best_run(root: Path) -> None:
    topic = _pick_topic(root)
    if topic is not None:
        actions.do_best(topic, "test_accuracy", "max", DB)


def _mle_study(root: Path) -> None:
    from mlr.study import STUDY_SPECS

    distribution = _select("Distribution:", sorted(STUDY_SPECS))
    if distribution is None:
        return
    choice = _select("Number of samples per experiment:", ["10", "100", "1000", "other"])
    if choice is None:
        return
    if choice == "other":
        choice = questionary.text("Samples:", default="1000").ask()
        if not choice:
            return
    experiments = questionary.text("Number of experiments to run:", default="100").ask()
    if not experiments:
        return
    actions.do_study(distribution, int(choice), int(experiments))


def _new_something(root: Path) -> None:
    kind = _select("Create a new:", ["topic", "experiment", "model", "paper"])
    if kind is None:
        return
    if kind == "topic":
        name = questionary.text("Topic name:").ask()
        if name:
            console.print(f"[green]created[/green] {scaffold.new_topic(root, name)}")
        return
    if kind == "model":
        name = questionary.text("Model name:").ask()
        if name:
            path = scaffold.new_model(root, name)
            console.print(f"[green]created[/green] {path}")
        return
    topic = _pick_topic(root)
    if topic is None:
        return
    if kind == "experiment":
        name = questionary.text("Experiment name:").ask()
        if not name:
            return
        model = _select("Model:", actions.do_models())
        dataset = _select("Dataset:", actions.do_datasets())
        path = scaffold.new_experiment(
            root,
            topic,
            name,
            **({"model": model} if model else {}),
            **({"dataset": dataset} if dataset else {}),
        )
        console.print(f"[green]created[/green] {path}")
    elif kind == "paper":
        slug = questionary.text("Paper slug (kebab-case):").ask()
        if not slug:
            return
        title = questionary.text("Title:", default=slug.replace("-", " ").title()).ask()
        assets = questionary.confirm("Include generate_assets.py?", default=False).ask()
        paper_dir = scaffold.new_paper(root, topic, slug, title or slug, assets=bool(assets))
        console.print(f"[green]created[/green] {paper_dir}")


def _studio(root: Path) -> None:
    from mlr.cli.app import _studio_open_dir

    action = _select("Studio:", ["open existing", "new", "graduate"])
    if action is None:
        return
    if action == "new":
        slug = questionary.text("Slug (kebab-case, e.g. bernoulli-mle):").ask()
        if slug:
            studio_dir = scaffold.new_studio(root, slug)
            console.print(f"[green]created[/green] {studio_dir}")
            _studio_open_dir(studio_dir)
        return
    name = _select("Which studio?", scaffold.list_studios(root))
    if name is None:
        return
    if action == "open existing":
        _studio_open_dir(root / "studio" / name)
    else:
        topic = questionary.text("Graduate into topic:").ask()
        if topic:
            created = scaffold.graduate_studio(root, name, topic)
            console.print(f"[green]paper:[/green] {created['paper']}")
            console.print(f"[green]notebook:[/green] {created['notebook']}")


_MENU = {
    "Run an experiment": _run_experiment,
    "Studio (notebook + LaTeX draft space)": _studio,
    "MLE study (sampling distribution of the estimator)": _mle_study,
    "Show tracked runs": _show_runs,
    "Best run of a topic": _best_run,
    "Build a paper (assets + PDF)": _build_paper,
    "New (topic / experiment / model / paper)": _new_something,
}


def menu() -> None:
    if not sys.stdin.isatty():
        console.print("no terminal attached; try `mlr --help` for subcommands")
        raise SystemExit(1)
    root = actions.repo_root()
    console.print("[bold]mlr[/bold] — ML research workbench (Esc backs out, Quit exits)")
    while True:
        try:
            choice = questionary.select("What do you want to do?", [*_MENU, "Quit"]).ask()
        except Exception:  # no usable console (piped output, CI, IDE task runner)
            console.print("interactive mode needs a real terminal; try `mlr --help`")
            raise SystemExit(1) from None
        if choice in (None, "Quit"):
            return
        try:
            _MENU[choice](root)
        except Exception as exc:  # keep the menu alive on action errors
            console.print(f"[red]error:[/red] {exc}")
