# Blackboard

A research studio — our own app, chalk on slate, fully standalone. It works
on its own `workspace/` directory (gitignored — your boards are yours) and
has no coupling to the MLR project files. One room, three sections:

- **Files rail** (left): your boards. "+ new board" creates
  `<name>/main.ipynb` + `<name>/main.tex`.
- **Board** (center): notebook cells running in a persistent kernel
  (numpy + matplotlib preloaded; anything else — including `mlr` — is one
  `import` away since the env is shared). Shift+Enter runs a cell; figures
  render inline; state survives across runs. Notebooks save as real `.ipynb`.
- **Theory board** (right): the LaTeX page. Edit tab ↔ Preview tab; the
  compile button saves, runs two pdflatex passes, and shows the PDF — errors
  surface as a chalk-red log.

## Run it

```sh
blackboard                    # workspace/ next to where you run it
blackboard --root <dir>       # or point it anywhere
```

Or press F5 in VS Code ("blackboard" — uses blackboard/workspace/).
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
