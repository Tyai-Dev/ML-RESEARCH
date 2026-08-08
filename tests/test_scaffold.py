import pytest

from mlr.cli import scaffold


@pytest.fixture
def repo(tmp_path):
    """Minimal repo layout the scaffolds operate on."""
    (tmp_path / "research").mkdir()
    pure = tmp_path / "src" / "mlr" / "models" / "pure"
    pure.mkdir(parents=True)
    (tmp_path / "src" / "mlr" / "__init__.py").write_text(
        'from mlr.data import synthetic as _synthetic  # noqa: F401\n\n__version__ = "0.1.0"\n'
    )
    return tmp_path


def test_new_topic_creates_layout(repo):
    topic_dir = scaffold.new_topic(repo, "My Topic")
    assert topic_dir == repo / "research" / "my-topic"
    assert (topic_dir / "experiments").is_dir()
    assert (topic_dir / "papers").is_dir()
    with pytest.raises(FileExistsError):
        scaffold.new_topic(repo, "my-topic")


def test_new_experiment_writes_config(repo):
    scaffold.new_topic(repo, "t")
    path = scaffold.new_experiment(repo, "t", "baseline", model="m", dataset="d")
    text = path.read_text()
    assert "name: baseline" in text
    assert "model: m" in text and "dataset: d" in text
    with pytest.raises(FileNotFoundError):
        scaffold.new_experiment(repo, "nope", "x")


def test_new_model_registers_import(repo):
    path = scaffold.new_model(repo, "Fancy Tree")
    assert path.name == "fancy_tree.py"
    assert 'register_model("fancy-tree")' in path.read_text()
    assert "class FancyTree" in path.read_text()
    init = (repo / "src" / "mlr" / "__init__.py").read_text()
    assert "from mlr.models.pure import fancy_tree" in init
    assert init.index("fancy_tree") < init.index("__version__")


def test_new_paper_numbers_sequentially(repo):
    scaffold.new_topic(repo, "t")
    p1 = scaffold.new_paper(repo, "t", "first-idea", "First Idea")
    p2 = scaffold.new_paper(repo, "t", "second-idea", "Second Idea", assets=True)
    assert p1.name == "01-first-idea"
    assert p2.name == "02-second-idea"
    assert (p1 / "main.tex").exists() and not (p1 / "generate_assets.py").exists()
    assert (p2 / "generate_assets.py").exists()
    assert "\\title{First Idea}" in (p1 / "main.tex").read_text()
