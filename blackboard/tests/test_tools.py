import nbformat
import pytest

from blackboard import tools


@pytest.fixture
def root(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell("x = 1"), nbformat.v4.new_markdown_cell("# t")]
    nbformat.write(nb, tmp_path / "b" / "main.ipynb")
    return tmp_path


def test_read_and_write_file(root):
    out, err = tools.execute(root, "read_file", {"path": "b/main.tex"})
    assert not err and out.startswith("\\documentclass")

    out, err = tools.execute(root, "write_file", {"path": "b/main.tex", "content": "\\documentclass{book}"})
    assert not err
    assert (root / "b" / "main.tex").read_text(encoding="utf-8") == "\\documentclass{book}"


def test_read_notebook_renders_cells(root):
    out, err = tools.execute(root, "read_file", {"path": "b/main.ipynb"})
    assert not err and "## cell 1 (code)" in out and "x = 1" in out


def test_add_cell_appends_and_inserts(root):
    out, err = tools.execute(root, "add_notebook_cell", {
        "notebook_path": "b/main.ipynb", "cell_type": "code", "source": "y = 2"})
    assert not err and "index 2" in out

    out, err = tools.execute(root, "add_notebook_cell", {
        "notebook_path": "b/main.ipynb", "cell_type": "markdown", "source": "top", "index": 0})
    assert not err and "index 0" in out

    nb = nbformat.read(root / "b" / "main.ipynb", as_version=4)
    assert [c.source for c in nb.cells] == ["top", "x = 1", "# t", "y = 2"]


def test_edit_cell_and_bounds(root):
    out, err = tools.execute(root, "edit_notebook_cell", {
        "notebook_path": "b/main.ipynb", "index": 0, "source": "x = 99"})
    assert not err
    nb = nbformat.read(root / "b" / "main.ipynb", as_version=4)
    assert nb.cells[0].source == "x = 99"

    out, err = tools.execute(root, "edit_notebook_cell", {
        "notebook_path": "b/main.ipynb", "index": 7, "source": "z"})
    assert err and "out of range" in out


def test_path_escape_and_unknown_tool(root):
    out, err = tools.execute(root, "write_file", {"path": "../evil.tex", "content": "x"})
    assert err and "escapes" in out
    out, err = tools.execute(root, "format_disk", {})
    assert err and "unknown tool" in out


def test_write_file_rejects_notebooks(root):
    out, err = tools.execute(root, "write_file", {"path": "b/main.ipynb", "content": "{}"})
    assert err and "cell tools" in out
