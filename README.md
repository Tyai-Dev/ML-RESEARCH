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
mlr distributions                      # registered distributions
mlr study                              # repeated-sampling MLE study (prompts),
                                       #   or: mlr study bernoulli -n 100 -k 1000

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

## Studio: the draft space

Research starts messy; the studio is where it's allowed to. A studio is a
notebook and a LaTeX page that grow together — code on the left, theory on
the right:

```sh
mlr studio new bernoulli-mle    # creates studio/bernoulli-mle/{main.ipynb, main.tex}
mlr studio open                 # launches JupyterLab with both files (F5 does this too);
                                #   right-click main.tex -> Show LaTeX Preview for a live
                                #   PDF pane; arrange once, the layout persists
mlr paper studio/bernoulli-mle  # compile the draft tex from the terminal anytime
mlr studio graduate bernoulli-mle mle   # done exploring? promote it:
```

The notebook is pre-wired: `%autoreload` (edits to `src/mlr` apply live, no
kernel restarts), the `ml-research` Jupyter kernel, and imports for the
registries + tracker (past runs are on disk — nothing reloads or reruns).
**Graduation** turns the draft into permanent structure: the tex becomes the
topic's next numbered paper, the notebook is archived under
`research/<topic>/notebooks/`, and you promote the code that stabilized into
`src/mlr` (`mlr new model ...`) with experiment configs.

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

## Example: maximum likelihood

The `research/mle/` topic pairs a theory paper (the MLE principle with
closed-form derivations) with six tracked experiments over Bernoulli /
Multinoulli / Gaussian samples. Distributions live in `mlr.distributions` —
each owns its `nll`, its closed-form `.mle()`, and an unconstrained
parameterization with hand-derived gradients. One model composes them:

```yaml
model: mle
model_params:
  distribution: gaussian   # see `mlr distributions`
  method: gradient         # mle = closed form | gradient = optimize the nll
  backend: pure            # pure (NumPy grads) | torch (autograd)
```

The paper's results table shows every route recovers the same estimates.
Unsupervised runs are scored by held-out negative log-likelihood
(`test_nll`, so `mlr best mle --metric test_nll --mode min`).

## Example: linear discriminator

The `research/linear-discriminator/` topic is the reference vertical slice:
a pure-NumPy logistic model and its PyTorch twin
(`linear-discriminator` / `linear-discriminator-torch`), three experiment
configs, a theory note deriving the Bernoulli/logistic-loss equivalence
(paper 01), and a writeup whose accuracy table and loss curves are
generated from real tracked runs (paper 02).
