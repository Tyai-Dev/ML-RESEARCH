"""Assistant tools: act on workspace files directly, sandboxed to the root.

The assistant uses these to *apply* changes — writing tex, adding or editing
notebook cells — instead of printing content into the chat. Every path is
resolved and confined to the workspace root; notebook edits go through
nbformat so files stay valid.
"""

from pathlib import Path

import nbformat

_TEXT_SUFFIXES = {".tex", ".py", ".md", ".txt", ".bib", ".yaml", ".json"}
_READ_LIMIT = 50_000

DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read a text file from the workspace. Use it to see content that "
            "is not in the attached board (other boards, other files)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path, e.g. 'my-board/main.tex'"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write (create or fully overwrite) a text file in the workspace — "
            "use this to edit main.tex or create new files. Always write the "
            "COMPLETE file content, not a fragment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path, e.g. 'my-board/main.tex'"},
                "content": {"type": "string", "description": "The complete new file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "add_notebook_cell",
        "description": (
            "Insert a new cell into a notebook (.ipynb). Use this when the "
            "user asks to add code or notes to the notebook."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_path": {"type": "string", "description": "Workspace-relative .ipynb path"},
                "cell_type": {"type": "string", "enum": ["code", "markdown"]},
                "source": {"type": "string", "description": "The cell's content."},
                "index": {
                    "type": "integer",
                    "description": "0-based position to insert at; omit to append at the end.",
                },
            },
            "required": ["notebook_path", "cell_type", "source"],
        },
    },
    {
        "name": "edit_notebook_cell",
        "description": (
            "Replace the source of an existing notebook cell. The attached "
            "board labels cells 'cell N' — the 0-based index here is N-1."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_path": {"type": "string", "description": "Workspace-relative .ipynb path"},
                "index": {"type": "integer", "description": "0-based cell index (board's 'cell N' = index N-1)"},
                "source": {"type": "string", "description": "The cell's complete new content."},
            },
            "required": ["notebook_path", "index", "source"],
        },
    },
]


def _safe(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the workspace: {rel}")
    return path


def execute(root: Path, name: str, args: dict) -> tuple[str, bool]:
    """Run a tool; returns (result_text, is_error)."""
    try:
        return _DISPATCH[name](root, args), False
    except KeyError:
        return f"unknown tool: {name}", True
    except Exception as exc:
        return f"error: {exc}", True


def _read_file(root: Path, args: dict) -> str:
    path = _safe(root, args["path"])
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {args['path']}")
    if path.suffix == ".ipynb":
        nb = nbformat.read(path, as_version=4)
        return "\n\n".join(
            f"## cell {i + 1} ({c.cell_type})\n{c.source}" for i, c in enumerate(nb.cells)
        )[:_READ_LIMIT]
    if path.suffix not in _TEXT_SUFFIXES:
        raise ValueError(f"not a readable text file: {path.suffix}")
    return path.read_text(encoding="utf-8")[:_READ_LIMIT]


def _write_file(root: Path, args: dict) -> str:
    path = _safe(root, args["path"])
    if path.suffix not in _TEXT_SUFFIXES:
        raise ValueError(f"write_file only handles text files ({path.suffix}); notebooks go through the cell tools")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return f"wrote {len(args['content'])} chars to {args['path']}"


def _load_nb(root: Path, rel: str):
    path = _safe(root, rel)
    if path.suffix != ".ipynb" or not path.is_file():
        raise FileNotFoundError(f"no such notebook: {rel}")
    return path, nbformat.read(path, as_version=4)


def _add_cell(root: Path, args: dict) -> str:
    path, nb = _load_nb(root, args["notebook_path"])
    maker = (nbformat.v4.new_code_cell if args["cell_type"] == "code"
             else nbformat.v4.new_markdown_cell)
    index = args.get("index")
    if index is None or index > len(nb.cells):
        index = len(nb.cells)
    nb.cells.insert(index, maker(args["source"]))
    nbformat.write(nb, path)
    return f"added {args['cell_type']} cell at index {index} ({len(nb.cells)} cells total)"


def _edit_cell(root: Path, args: dict) -> str:
    path, nb = _load_nb(root, args["notebook_path"])
    index = args["index"]
    if not 0 <= index < len(nb.cells):
        raise IndexError(f"cell index {index} out of range (notebook has {len(nb.cells)} cells)")
    nb.cells[index].source = args["source"]
    nbformat.write(nb, path)
    return f"replaced cell {index} ({nb.cells[index].cell_type})"


_DISPATCH = {
    "read_file": _read_file,
    "write_file": _write_file,
    "add_notebook_cell": _add_cell,
    "edit_notebook_cell": _edit_cell,
}
