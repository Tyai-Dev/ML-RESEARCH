"""Blackboard server: files, cells, theory board — one local app.

    blackboard [--root PATH] [--port 8321] [--no-browser]

Serves the UI at http://127.0.0.1:<port>/ and a small JSON API underneath.
The root is Blackboard's own workspace directory (created if missing) —
deliberately standalone; every file path in the API is validated to stay
inside it.
"""

import argparse
import threading
import webbrowser
from pathlib import Path

import nbformat
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from blackboard.kernel import Kernel
from blackboard.latex import compile_tex

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="blackboard")
kernel = Kernel()
kernel_lock = threading.Lock()
ROOT = Path.cwd()

_EDIT_SUFFIXES = {".tex", ".py", ".yaml", ".md", ".bib", ".txt"}

_TEX_TEMPLATE = """\\documentclass[11pt]{article}
\\usepackage{amsmath,amssymb,amsthm}
\\usepackage{graphicx}
\\usepackage[margin=1.1in]{geometry}

\\title{%s}
\\author{}
\\date{\\today}

\\begin{document}
\\maketitle

\\section{Question}

\\end{document}
"""


def _safe(rel: str) -> Path:
    path = (ROOT / rel).resolve()
    if not path.is_relative_to(ROOT):
        raise HTTPException(400, "path escapes the workspace")
    return path


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/workspace")
def workspace() -> dict:
    def tree(base: Path, rel_base: str, depth: int = 0) -> list[dict]:
        if not base.exists() or depth > 3:
            return []
        out = []
        for child in sorted(base.iterdir(), key=lambda c: (c.is_file(), c.name)):
            if child.name.startswith((".", "__")) or child.suffix in (".pdf", ".aux",
                    ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz", ".pyc"):
                continue
            rel = f"{rel_base}/{child.name}" if rel_base else child.name
            if child.is_dir():
                out.append({"name": child.name, "path": rel, "dir": True,
                            "children": tree(child, rel, depth + 1)})
            else:
                out.append({"name": child.name, "path": rel, "dir": False})
        return out

    return {"root": str(ROOT), "tree": tree(ROOT, "")}


class NewBoard(BaseModel):
    name: str


@app.post("/api/board")
def new_board(req: NewBoard) -> dict:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", req.name.lower()).strip("-")
    if not slug:
        raise HTTPException(400, "give the board a name")
    board_dir = ROOT / slug
    if board_dir.exists():
        raise HTTPException(409, f"board already exists: {slug}")
    board_dir.mkdir(parents=True)

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(f"# {req.name}"),
        nbformat.v4.new_code_cell(""),
    ]
    nbformat.write(nb, board_dir / "main.ipynb")
    (board_dir / "main.tex").write_text(
        _TEX_TEMPLATE % req.name.replace("\\", ""), encoding="utf-8"
    )
    return {"created": slug, "notebook": f"{slug}/main.ipynb", "tex": f"{slug}/main.tex"}


@app.get("/api/file")
def read_file(path: str) -> dict:
    target = _safe(path)
    if not target.is_file():
        raise HTTPException(404, f"no such file: {path}")
    if target.suffix not in _EDIT_SUFFIXES:
        raise HTTPException(400, f"not a text file blackboard edits: {target.suffix}")
    return {"path": path, "content": target.read_text(encoding="utf-8")}


class SaveFile(BaseModel):
    path: str
    content: str


@app.post("/api/file")
def save_file(req: SaveFile) -> dict:
    target = _safe(req.path)
    if target.suffix not in _EDIT_SUFFIXES:
        raise HTTPException(400, f"not a text file blackboard edits: {target.suffix}")
    target.write_text(req.content, encoding="utf-8")
    return {"saved": req.path}


@app.get("/api/notebook")
def read_notebook(path: str) -> dict:
    target = _safe(path)
    if not target.is_file() or target.suffix != ".ipynb":
        raise HTTPException(404, f"no such notebook: {path}")
    nb = nbformat.read(target, as_version=4)
    cells = [
        {"type": c.cell_type, "source": c.source}
        for c in nb.cells
        if c.cell_type in ("code", "markdown")
    ]
    return {"path": path, "cells": cells}


class SaveNotebook(BaseModel):
    path: str
    cells: list[dict]


@app.post("/api/notebook")
def save_notebook(req: SaveNotebook) -> dict:
    target = _safe(req.path)
    if target.suffix != ".ipynb":
        raise HTTPException(400, "not a notebook path")
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python (ml-research)", "language": "python", "name": "ml-research",
    }
    for cell in req.cells:
        maker = (nbformat.v4.new_code_cell if cell.get("type") == "code"
                 else nbformat.v4.new_markdown_cell)
        nb.cells.append(maker(cell.get("source", "")))
    nbformat.write(nb, target)
    return {"saved": req.path, "cells": len(nb.cells)}


class Execute(BaseModel):
    code: str


@app.post("/api/execute")
def execute(req: Execute) -> dict:
    with kernel_lock:
        result = kernel.run(req.code)
    return {
        "ok": result.ok, "stdout": result.stdout, "value": result.value,
        "error": result.error, "images": result.images,
    }


class Fix(BaseModel):
    code: str


@app.post("/api/fix")
def fix(req: Fix) -> dict:
    from blackboard.fixer import fix_code

    return fix_code(req.code)


class Chat(BaseModel):
    messages: list[dict]
    context: str | None = None
    provider: str = "claude"


@app.post("/api/chat")
def chat(req: Chat) -> StreamingResponse:
    from blackboard.chat import ChatError, stream_chat

    if not req.messages:
        raise HTTPException(400, "empty conversation")
    try:
        chunks = stream_chat(req.messages, context=req.context, provider=req.provider)
    except ChatError as exc:
        raise HTTPException(503, str(exc))
    return StreamingResponse(chunks, media_type="text/plain; charset=utf-8")


@app.post("/api/reset")
def reset() -> dict:
    with kernel_lock:
        kernel.reset()
    return {"reset": True}


class Compile(BaseModel):
    path: str  # a .tex file


@app.post("/api/compile")
def compile_endpoint(req: Compile) -> dict:
    target = _safe(req.path)
    if target.suffix != ".tex":
        raise HTTPException(400, "not a tex file")
    ok, log = compile_tex(target)
    pdf_rel = req.path.rsplit(".", 1)[0] + ".pdf"
    return {"ok": ok, "log": log, "pdf": f"/api/pdf?path={pdf_rel}" if ok else None}


@app.get("/api/pdf")
def serve_pdf(path: str) -> FileResponse:
    target = _safe(path)
    if not target.is_file() or target.suffix != ".pdf":
        raise HTTPException(404, "no pdf built yet")
    return FileResponse(target, media_type="application/pdf")


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(prog="blackboard")
    parser.add_argument(
        "--root", default="workspace", help="Blackboard workspace directory"
    )
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    ROOT = Path(args.root).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)

    # API keys for the assistant: load .env from the blackboard project dir
    # and from the current directory (later loads never override earlier ones)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parents[2] / ".env")  # blackboard/.env
    load_dotenv()  # ./.env (wherever blackboard was launched from)

    import uvicorn

    url = f"http://127.0.0.1:{args.port}"
    print(f"blackboard on {url}  (root: {ROOT})")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
