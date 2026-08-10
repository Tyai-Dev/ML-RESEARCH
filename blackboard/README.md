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

## Editing & assistant

- **Autosave** — every edit (cells or tex) saves ~1s after you stop typing;
  switching files or closing the tab flushes first. The header shows
  "unsaved… / all saved".
- **Editor smarts** — Enter auto-indents (extra level after `:`, dedent after
  `return`/`pass`), Tab/Shift+Tab indent/dedent; each code cell has a
  **fix** button (ruff format + safe autofixes, remaining diagnostics shown
  under the cell; kernel-provided names like `np` aren't flagged).
- **Assistant** — the header's *assistant* tab is a streaming Claude chat
  (`claude-opus-5`; override with `BLACKBOARD_MODEL`). "Attach board" sends
  your current cells + tex as context; any fenced code block in a reply has
  an **→ insert as cell** button. Requires `ANTHROPIC_API_KEY` (or an
  `ant auth login` profile) in the environment that launches blackboard.

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
