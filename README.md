# ML-RESEARCH

A personal library to train and research ML models. Three kinds of output live
together and stay connected: **models** (pure Python + PyTorch), **papers**
(LaTeX theory notes and experiment writeups), and **data** (file-based
datasets + a SQLite experiment tracker).

Design rationale: [docs/plans/2026-08-08-ml-research-skeleton-design.md](docs/plans/2026-08-08-ml-research-skeleton-design.md)

## Layout

```
src/mlr/                 installable package
  models/pure/           NumPy-only models (theory-grade)
  models/torch/          PyTorch modules (optional extra)
  data/                  dataset registry + synthetic builders
  training/              config-driven experiment runner
  tracking/              SQLite tracker: runs, params, metrics, artifacts
research/
  papers-common/         shared LaTeX preamble
  <topic>/
    experiments/         YAML configs
    papers/NN-slug/      main.tex (+ generate_assets.py for writeups)
    results/             generated figures + .tex table fragments
datasets/                file-based data (blobs gitignored, manifests committed)
tracking.db              run metadata (gitignored; regenerate by re-running)
```

## Setup

Requires Anaconda (env `ml-research`) and, for papers, MiKTeX (`pdflatex`).

```sh
conda activate ml-research
pip install -e ".[dev]"        # installs mlr + pytest into the env
pip install -e ".[dev,torch]"  # additionally install PyTorch
pytest                         # test suite
```

## Workflow

(with `ml-research` activated)

```sh
# 1. Train + track an experiment (config names a registered model & dataset)
mlr run research/linear-discriminator/experiments/baseline.yaml

# 2. Inspect tracked runs
mlr runs --topic linear-discriminator

# 3. Rebuild a paper: regenerates its figures/tables from tracking.db,
#    then compiles with pdflatex (two passes)
mlr paper research/linear-discriminator/papers/02-loss-based-training
```

## Adding things

- **A model**: one file under `src/mlr/models/pure/` or `models/torch/` with
  `@register_model("name")`; expose `fit(X, y, on_epoch=None)` / `predict(X)`.
  Import it from `mlr/__init__.py` so configs can find it.
- **A dataset**: a builder in `src/mlr/data/` with `@register_dataset("name")`
  (synthetic), or files under `datasets/` + a manifest (real data).
- **A topic**: `research/<topic>/` with `experiments/` and `papers/`.
- **A paper**: `research/<topic>/papers/NN-slug/main.tex`; include
  `generate_assets.py` if it reports experimental results — write tables to
  `../../results/tables/` and figures to `../../results/figures/`, pulling
  numbers only from the tracker so the paper can never disagree with the runs.

## Example: linear discriminator

The `research/linear-discriminator/` topic is the reference vertical slice:
a pure-NumPy logistic model (`mlr.models.pure.linear_discriminator`), two
experiment configs, a theory note deriving the Bernoulli/logistic-loss
equivalence (paper 01), and a writeup whose accuracy table and loss curves
are generated from real tracked runs (paper 02).
