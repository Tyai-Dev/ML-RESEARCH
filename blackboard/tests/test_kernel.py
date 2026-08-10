import pytest

from blackboard.kernel import Kernel


@pytest.fixture(scope="module")
def kernel():
    return Kernel()


def test_repl_semantics_trailing_expression(kernel):
    result = kernel.run("a = 20\na + 2")
    assert result.ok and result.value == "22"


def test_state_persists_across_runs(kernel):
    kernel.run("counter = 1")
    kernel.run("counter += 5")
    assert kernel.run("counter").value == "6"


def test_stdout_captured(kernel):
    result = kernel.run("print('chalk')")
    assert result.ok and result.stdout == "chalk\n"
    assert result.value is None


def test_error_reported_not_raised(kernel):
    result = kernel.run("1 / 0")
    assert not result.ok
    assert "ZeroDivisionError" in result.error


def test_mlr_is_preloaded(kernel):
    result = kernel.run("get_distribution('bernoulli').mle([1, 1, 0, 1])")
    assert result.ok and "0.75" in result.value


def test_matplotlib_figures_captured(kernel):
    result = kernel.run("plt.plot([1, 2, 3]); plt.gcf()")
    assert result.ok
    assert len(result.images) == 1
    assert len(result.images[0]) > 1000  # a real png, base64


def test_reset_clears_namespace(kernel):
    kernel.run("ghost = 1")
    kernel.reset()
    assert not kernel.run("ghost").ok
