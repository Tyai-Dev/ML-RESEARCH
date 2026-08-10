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


_STUDIO_TEX_TEMPLATE = r"""\documentclass[11pt]{{article}}
\input{{../../research/papers-common/preamble.tex}}

\title{{Studio: {title}}}
\author{{Yuval Lavie}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Question}}

What am I trying to show?

\section{{Assumptions}}

\section{{Theory}}

% Derivations grow here while the notebook explores the same ground.
% Compile anytime with:  mlr paper studio/{slug}

\section{{Findings}}

% What the notebook confirmed or refuted. When this section feels done,
% graduate:  mlr studio graduate {slug} <topic>

\end{{document}}
"""

_STUDIO_NOTEBOOK = """\
{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# Studio: {title}\\n",
    "\\n",
    "Exploration half — the theory lives in `main.tex` beside this notebook.\\n",
    "Edits to `src/mlr` apply live (autoreload); kernel state survives."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "%load_ext autoreload\\n",
    "%autoreload 2\\n",
    "\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "\\n",
    "import mlr\\n",
    "from mlr.data import get_dataset\\n",
    "from mlr.distributions import get_distribution\\n",
    "from mlr.models import get_model\\n",
    "from mlr.study import run_study\\n",
    "from mlr.tracking import Tracker"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# past experiments are already on disk — no reruns needed\\n",
    "with Tracker(\\"../../tracking.db\\") as tr:\\n",
    "    for run in tr.list_runs():\\n",
    "        print(run[\\"id\\"], run[\\"topic\\"], run[\\"name\\"], run[\\"status\\"])"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": []
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python (ml-research)",
   "language": "python",
   "name": "ml-research"
  }},
  "language_info": {{
   "name": "python"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 5
}}
"""


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


def new_studio(root: Path, slug: str) -> Path:
    """Create studio/<slug>/ with main.ipynb (exploration) + main.tex (theory)."""
    slug = _kebab(slug)
    studio_dir = root / "studio" / slug
    if studio_dir.exists():
        raise FileExistsError(f"studio already exists: {studio_dir}")
    studio_dir.mkdir(parents=True)
    title = slug.replace("-", " ").title()
    (studio_dir / "main.tex").write_text(
        _STUDIO_TEX_TEMPLATE.format(title=title, slug=slug)
    )
    (studio_dir / "main.ipynb").write_text(_STUDIO_NOTEBOOK.format(title=title))
    return studio_dir


def list_studios(root: Path) -> list[str]:
    studio_root = root / "studio"
    if not studio_root.exists():
        return []
    return sorted(d.name for d in studio_root.iterdir() if d.is_dir())


def graduate_studio(root: Path, slug: str, topic: str) -> dict[str, Path]:
    """Promote a studio draft into permanent MLR structure.

    The tex becomes the topic's next numbered paper (preamble path rewritten
    for the deeper directory), the notebook is archived under the topic for
    provenance, and the studio directory is removed. Code the draft converged
    on still needs a human: promote it into src/mlr and add a config.
    """
    slug = _kebab(slug)
    studio_dir = root / "studio" / slug
    if not studio_dir.exists():
        raise FileNotFoundError(f"no such studio: {studio_dir}")
    topic_dir = root / "research" / _kebab(topic)
    if not topic_dir.exists():
        new_topic(root, topic)

    paper_dir = new_paper(root, topic, slug, slug.replace("-", " ").title())
    tex = (studio_dir / "main.tex").read_text()
    (paper_dir / "main.tex").write_text(
        tex.replace("../../research/papers-common/", "../../../papers-common/")
    )

    notebooks_dir = topic_dir / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    notebook_dest = notebooks_dir / f"{slug}.ipynb"
    (studio_dir / "main.ipynb").replace(notebook_dest)
    (studio_dir / "main.tex").unlink()
    for leftover in studio_dir.iterdir():  # build artifacts (pdf, aux, ...)
        leftover.unlink()
    studio_dir.rmdir()
    return {"paper": paper_dir, "notebook": notebook_dest}


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
