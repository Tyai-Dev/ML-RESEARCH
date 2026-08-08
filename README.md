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

(with `ml-research` activated, from the repo root)

```sh
mlr            # interactive menu: run experiments, build papers, scaffold — no args to remember
```

Or the direct subcommands:

```sh
# Train + track an experiment (config names a registered model & dataset)
mlr run research/linear-discriminator/experiments/baseline.yaml

# Inspect what you've run
mlr runs --topic linear-discriminator
mlr best linear-discriminator          # best run by test_accuracy
mlr topics                             # research topics + counts
mlr models                             # registered models
mlr datasets                           # registered datasets
mlr optimizers                         # available training optimizers

# Rebuild a paper: regenerates its figures/tables from tracking.db,
# then compiles with pdflatex (two passes)
mlr paper research/linear-discriminator/papers/02-loss-based-training

# Scaffold new pieces (layout + templates created for you)
mlr new topic my-idea
mlr new experiment my-idea baseline --model linear-discriminator --dataset two-gaussians
mlr new model "fancy tree"             # file + registration + import wired up
mlr new paper my-idea first-note --assets
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

## Training configuration

Every experiment config has a `training` section; `mlr run` shows a live
epoch/loss/accuracy progress bar while it executes:

```yaml
training:
  optimizer: adam     # sgd | momentum | adam | rmsprop | adagrad
  lr: 0.05            # extra keys (momentum, beta1, ...) go to the optimizer
  epochs: 300
  batch_size: 32      # null = full-batch GD; an int = mini-batch SGD
```

Pure models use the from-scratch NumPy optimizers in
`mlr/training/optimizers.py`; torch models map the same names onto
`torch.optim`, so a config works unchanged with either model family.

## Example: linear discriminator

The `research/linear-discriminator/` topic is the reference vertical slice:
a pure-NumPy logistic model and its PyTorch twin
(`linear-discriminator` / `linear-discriminator-torch`), three experiment
configs, a theory note deriving the Bernoulli/logistic-loss equivalence
(paper 01), and a writeup whose accuracy table and loss curves are
generated from real tracked runs (paper 02).
