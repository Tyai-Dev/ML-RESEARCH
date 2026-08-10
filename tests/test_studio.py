import json

import pytest

from mlr.cli import scaffold


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "research").mkdir()
    pure = tmp_path / "src" / "mlr" / "models" / "pure"
    pure.mkdir(parents=True)
    (tmp_path / "src" / "mlr" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    return tmp_path


def test_new_studio_creates_notebook_and_tex(repo):
    studio_dir = scaffold.new_studio(repo, "Bernoulli MLE")
    assert studio_dir == repo / "studio" / "bernoulli-mle"
    nb = json.loads((studio_dir / "main.ipynb").read_text())  # valid JSON
    assert nb["metadata"]["kernelspec"]["name"] == "ml-research"
    assert "autoreload" in json.dumps(nb)
    tex = (studio_dir / "main.tex").read_text()
    assert r"\title{Studio: Bernoulli Mle}" in tex
    assert "../../research/papers-common/preamble.tex" in tex
    with pytest.raises(FileExistsError):
        scaffold.new_studio(repo, "bernoulli-mle")


def test_list_studios(repo):
    assert scaffold.list_studios(repo) == []
    scaffold.new_studio(repo, "b")
    scaffold.new_studio(repo, "a")
    assert scaffold.list_studios(repo) == ["a", "b"]


def test_graduate_moves_tex_and_notebook(repo):
    scaffold.new_studio(repo, "bernoulli-mle")
    created = scaffold.graduate_studio(repo, "bernoulli-mle", "mle")

    assert created["paper"] == repo / "research" / "mle" / "papers" / "01-bernoulli-mle"
    tex = (created["paper"] / "main.tex").read_text()
    assert "../../../papers-common/preamble.tex" in tex  # path rewritten for new depth
    assert "../../research/" not in tex

    assert created["notebook"] == repo / "research" / "mle" / "notebooks" / "bernoulli-mle.ipynb"
    assert created["notebook"].exists()
    assert not (repo / "studio" / "bernoulli-mle").exists()  # draft is gone


def test_graduate_into_existing_topic_numbers_after_existing_papers(repo):
    scaffold.new_topic(repo, "mle")
    scaffold.new_paper(repo, "mle", "principle", "Principle")
    scaffold.new_studio(repo, "bernoulli-mle")
    created = scaffold.graduate_studio(repo, "bernoulli-mle", "mle")
    assert created["paper"].name == "02-bernoulli-mle"


def test_graduate_missing_studio_raises(repo):
    with pytest.raises(FileNotFoundError, match="no such studio"):
        scaffold.graduate_studio(repo, "nope", "mle")
