"""mlr command line (typer front end).

    mlr                        interactive menu
    mlr run CONFIG             train + track an experiment
    mlr runs [--topic T]       list tracked runs
    mlr best TOPIC             best run of a topic by metric
    mlr paper DIR              regenerate assets + compile a paper PDF
    mlr models / datasets      list what's registered
    mlr topics                 list research topics
    mlr new topic|experiment|model|paper ...   scaffold new pieces
"""

from pathlib import Path

import typer

from mlr.cli import actions, scaffold
from mlr.cli.actions import console

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    help="Personal ML research workbench: experiments, tracking, papers.",
)
new_app = typer.Typer(help="Scaffold a new topic, experiment, model, or paper.")
app.add_typer(new_app, name="new")

DB_OPTION = typer.Option("tracking.db", "--db", help="Path to the tracking database.")


@app.callback()
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from mlr.cli.interactive import menu

        menu()


@app.command()
def run(config: Path, db: str = DB_OPTION) -> None:
    """Train + track an experiment from a YAML config."""
    actions.do_run(config, db)


@app.command()
def runs(
    topic: str = typer.Option(None, help="Filter by topic."),
    db: str = DB_OPTION,
) -> None:
    """List tracked runs."""
    actions.do_runs(topic, db)


@app.command()
def best(
    topic: str,
    metric: str = typer.Option("test_accuracy", help="Metric to rank by."),
    mode: str = typer.Option("max", help="'max' or 'min'."),
    db: str = DB_OPTION,
) -> None:
    """Show the best finished run of a topic."""
    actions.do_best(topic, metric, mode, db)


@app.command()
def paper(paper_dir: Path) -> None:
    """Regenerate a paper's assets from the tracker and compile its PDF."""
    actions.do_paper(paper_dir)


@app.command()
def models() -> None:
    """List registered models."""
    for name in actions.do_models():
        console.print(f"- {name}")


@app.command()
def datasets() -> None:
    """List registered datasets."""
    for name in actions.do_datasets():
        console.print(f"- {name}")


@app.command()
def optimizers() -> None:
    """List available training optimizers."""
    for name in actions.do_optimizers():
        console.print(f"- {name}")


@app.command()
def topics() -> None:
    """List research topics."""
    root = actions.repo_root()
    for name in actions.list_topics(root):
        n_exp = len(actions.list_experiments(root, name))
        n_pap = len(actions.list_papers(root, name))
        console.print(f"- {name}  ({n_exp} experiments, {n_pap} papers)")


@new_app.command("topic")
def new_topic(name: str) -> None:
    """Create research/<name>/ with experiments/ and papers/."""
    topic_dir = scaffold.new_topic(actions.repo_root(), name)
    console.print(f"[green]created[/green] {topic_dir}")
    console.print(f"next: mlr new experiment {topic_dir.name} <name>")


@new_app.command("experiment")
def new_experiment(
    topic: str,
    name: str,
    model: str = typer.Option(None, help="Registered model name."),
    dataset: str = typer.Option(None, help="Registered dataset name."),
) -> None:
    """Create a config under research/<topic>/experiments/."""
    kwargs = {}
    if model:
        kwargs["model"] = model
    if dataset:
        kwargs["dataset"] = dataset
    path = scaffold.new_experiment(actions.repo_root(), topic, name, **kwargs)
    console.print(f"[green]created[/green] {path}")
    console.print(
        f"registered models: {', '.join(actions.do_models())} | "
        f"datasets: {', '.join(actions.do_datasets())}"
    )


@new_app.command("model")
def new_model(name: str) -> None:
    """Create a pure-python model skeleton, registered and importable."""
    path = scaffold.new_model(actions.repo_root(), name)
    console.print(f"[green]created[/green] {path} (import added to mlr/__init__.py)")


@new_app.command("paper")
def new_paper(
    topic: str,
    slug: str,
    title: str = typer.Option(None, help="Paper title (defaults to the slug)."),
    assets: bool = typer.Option(
        False, "--assets", help="Include a generate_assets.py template."
    ),
) -> None:
    """Create research/<topic>/papers/NN-<slug>/ with main.tex."""
    paper_dir = scaffold.new_paper(
        actions.repo_root(),
        topic,
        slug,
        title or slug.replace("-", " ").title(),
        assets=assets,
    )
    console.print(f"[green]created[/green] {paper_dir}")


def main(argv: list[str] | None = None) -> None:
    app(args=argv)


if __name__ == "__main__":
    main()
