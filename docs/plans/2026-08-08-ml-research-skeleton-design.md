# ML-RESEARCH skeleton — design

Date: 2026-08-08
Status: approved

## Purpose

A personal library to train and research ML models. Three kinds of output live
together and stay connected:

- **Models** — pure Python (NumPy) implementations for theory work, and
  PyTorch modules for scale.
- **Papers** — LaTeX documents per research topic: theory notes and experiment
  writeups. A topic can accumulate several papers.
- **Data** — file-based datasets (not a database) plus a small SQLite database
  for run/metric tracking.

## Decisions

| Question | Decision |
|---|---|
| Training data storage | Files (Parquet/Arrow, memmap token bins later), registered via committed manifests; blobs gitignored under `datasets/`. A relational DB is for metadata only. |
| Experiment ↔ paper link | Auto-generated: a per-paper `generate_assets.py` queries the tracker and writes figures + `.tex` table fragments into `results/`; papers `\input` them. Theory-only papers skip this. |
| Repo layout | Library + topics: reusable `src/mlr` package, `research/<topic>/` bundles experiments + papers + results. |
| Experiment tracking | Own SQLite tracker (`mlr.tracking`): we own the schema, zero services, queryable from papers. |
| Tooling | `uv` + `pyproject.toml`, pytest, latexmk for papers. |

## Layout

```
ML-RESEARCH/
├── pyproject.toml
├── src/mlr/
│   ├── models/
│   │   ├── pure/            # NumPy-only models (theory-grade)
│   │   ├── torch/           # PyTorch modules
│   │   └── registry.py      # @register_model("name")
│   ├── data/
│   │   ├── registry.py      # dataset manifests: name, version, path, checksum
│   │   └── synthetic.py     # generated datasets for theory experiments
│   ├── training/            # loops, losses, eval, seeding
│   ├── tracking/            # SQLite: runs/params/metrics/artifacts + query API
│   └── cli.py               # `mlr run <config.yaml>`
├── research/
│   ├── papers-common/       # shared preamble.tex, refs style
│   └── <topic>/
│       ├── experiments/     # YAML configs (+ optional scripts)
│       ├── papers/NN-slug/  # main.tex, generate_assets.py, Makefile
│       └── results/         # generated figures/ and tables/ (.tex fragments)
├── datasets/                # raw/ processed/  (gitignored, manifests committed)
├── docs/plans/
└── tracking.db              # gitignored
```

## Components

### Tracking schema (SQLite)

- `runs(id, name, topic, model, dataset, status, started_at, finished_at)`
- `params(run_id, key, value)`
- `metrics(run_id, key, step, value, ts)`
- `artifacts(run_id, kind, path)`

Query API: `list_runs(topic=...)`, `best_run(topic, metric, mode)`,
`metric_history(run_id, key)` — this is the surface papers build tables from.

### Registries

`@register_model(name)` and `@register_dataset(name)` decorators populate
dicts; configs reference names, `mlr run` resolves them. Adding a model =
one file + one decorator.

### Vertical slice (proves every connection)

Pure-Python linear discriminator, trained on a synthetic two-Gaussian dataset
via `mlr run research/linear-discriminator/experiments/baseline.yaml`, logged
to SQLite; two papers under `research/linear-discriminator/papers/`:
`01-bernoulli-hypothesis-class` (theory, no assets) and
`02-loss-based-training` (writeup with auto-generated accuracy table + loss
curve figure).

## Out of scope (YAGNI)

Web UI, remote storage, MLflow/W&B integration, multi-machine training,
CI — add when a real need appears.
