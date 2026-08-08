"""Scaffolding for new topics, experiments, models, and papers.

Every function takes the repo root explicitly and returns the paths it
created, so the scaffolds are unit-testable against a tmp directory.
"""

import re
from pathlib import Path

_EXPERIMENT_TEMPLATE = """\
name: {name}
topic: {topic}
model: {model}
model_params: {{}}
dataset: {dataset}
dataset_params: {{}}
training:
  optimizer: sgd        # see `mlr optimizers`
  lr: 0.1
  epochs: 200
  batch_size: null      # null = full-batch GD; an int = mini-batch SGD
test_size: 0.2
seed: 42
"""

_MODEL_TEMPLATE = '''\
"""{title} model."""

import numpy as np

from mlr.models.registry import register_model


@register_model("{kebab}")
class {classname}:
    def __init__(self, epochs: int = 100, seed: int = 0):
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray, on_epoch=None) -> "{classname}":
        raise NotImplementedError("implement training for {kebab}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(int)))
'''

_PAPER_TEMPLATE = r"""\documentclass[11pt]{{article}}
\input{{../../../papers-common/preamble.tex}}

\title{{{title}}}
\author{{Yuval Lavie}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
TODO
\end{{abstract}}

\section{{Introduction}}

TODO

\end{{document}}
"""

_ASSETS_TEMPLATE = '''\
"""Regenerate this paper's tables/figures from the experiment tracker."""

from pathlib import Path

from mlr.tracking import Tracker

PAPER_DIR = Path(__file__).resolve().parent
TOPIC_DIR = PAPER_DIR.parents[1]
REPO_ROOT = TOPIC_DIR.parents[1]
TOPIC = TOPIC_DIR.name


def main() -> None:
    tables = TOPIC_DIR / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with Tracker(REPO_ROOT / "tracking.db") as tracker:
        runs = [r for r in tracker.list_runs(TOPIC) if r["status"] == "finished"]
        if not runs:
            raise SystemExit(f"no finished runs for topic {{TOPIC!r}}")
        # TODO: write .tex fragments / figures under TOPIC_DIR / "results"


if __name__ == "__main__":
    main()
'''


def _kebab(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _snake(name: str) -> str:
    return _kebab(name).replace("-", "_")


def _classname(name: str) -> str:
    return "".join(part.capitalize() for part in _kebab(name).split("-"))


def new_topic(root: Path, name: str) -> Path:
    topic = _kebab(name)
    topic_dir = root / "research" / topic
    if topic_dir.exists():
        raise FileExistsError(f"topic already exists: {topic_dir}")
    (topic_dir / "experiments").mkdir(parents=True)
    (topic_dir / "papers").mkdir()
    return topic_dir


def new_experiment(
    root: Path,
    topic: str,
    name: str,
    model: str = "TODO-registered-model",
    dataset: str = "TODO-registered-dataset",
) -> Path:
    exp_dir = root / "research" / _kebab(topic) / "experiments"
    if not exp_dir.exists():
        raise FileNotFoundError(f"no such topic: {topic} (run `mlr new topic {topic}` first)")
    path = exp_dir / f"{_kebab(name)}.yaml"
    if path.exists():
        raise FileExistsError(f"experiment already exists: {path}")
    path.write_text(
        _EXPERIMENT_TEMPLATE.format(
            name=_kebab(name), topic=_kebab(topic), model=model, dataset=dataset
        )
    )
    return path


def new_model(root: Path, name: str) -> Path:
    """Create a pure-python model file and register its import in mlr/__init__.py."""
    kebab, snake = _kebab(name), _snake(name)
    path = root / "src" / "mlr" / "models" / "pure" / f"{snake}.py"
    if path.exists():
        raise FileExistsError(f"model already exists: {path}")
    path.write_text(
        _MODEL_TEMPLATE.format(
            title=kebab.replace("-", " ").title(),
            kebab=kebab,
            classname=_classname(name),
        )
    )
    init = root / "src" / "mlr" / "__init__.py"
    marker = f"from mlr.models.pure import {snake} as _{snake}  # noqa: F401  (registers models)\n"
    content = init.read_text()
    if marker not in content:
        anchor = "__version__"
        content = content.replace(anchor, marker + "\n" + anchor, 1)
        init.write_text(content)
    return path


def new_paper(root: Path, topic: str, slug: str, title: str, assets: bool = False) -> Path:
    papers_dir = root / "research" / _kebab(topic) / "papers"
    if not papers_dir.exists():
        raise FileNotFoundError(f"no such topic: {topic} (run `mlr new topic {topic}` first)")
    numbers = [
        int(m.group(1))
        for d in papers_dir.iterdir()
        if (m := re.match(r"^(\d+)-", d.name))
    ]
    nn = max(numbers, default=0) + 1
    paper_dir = papers_dir / f"{nn:02d}-{_kebab(slug)}"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text(_PAPER_TEMPLATE.format(title=title))
    if assets:
        (paper_dir / "generate_assets.py").write_text(_ASSETS_TEMPLATE)
    return paper_dir
