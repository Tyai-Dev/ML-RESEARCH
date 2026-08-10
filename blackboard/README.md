# Blackboard

The MLR research studio — our own app, chalk on slate. One room, three
sections that stay visible together:

- **Files rail** (left): studio drafts, research topics, the `mlr` source.
- **Board** (center): notebook cells running in a persistent kernel with the
  whole `mlr` library preloaded (`get_distribution`, `run_study`, `Tracker`,
  …). Shift+Enter runs a cell; matplotlib figures render inline; state
  survives across runs. Notebooks load/save as real `.ipynb`.
- **Theory board** (right): the LaTeX page. Edit tab ↔ Preview tab; the
  compile button saves, runs two pdflatex passes, and shows the PDF — errors
  surface as a chalk-red log. Below it, the latest tracked runs straight
  from `tracking.db`.

## Run it

```sh
blackboard --root <ML-RESEARCH repo>     # or press F5 in VS Code ("blackboard")
```

Serves on http://127.0.0.1:8321 (local only) and opens the browser.

## Architecture

```
src/blackboard/
  kernel.py    persistent exec namespace, REPL semantics, figure capture
  latex.py     two-pass pdflatex wrapper
  server.py    FastAPI: workspace tree, file/notebook IO, execute, compile,
               pdf serving, tracker runs — every path sandboxed to --root
  static/      the frontend (vanilla HTML/CSS/JS, no build step)
```

Tests: `pytest blackboard/tests` (kernel semantics + API round-trips).

## Ideas queue

- Pin a cell's output into the tex as a margin note (Manuscript's trick)
- `mlr run` / `mlr study` launchers with live progress in the board
- Graduate button (studio → topic) in the header
- KaTeX-style rendered math in note cells
