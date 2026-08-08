import numpy as np
import pytest

from mlr.study import STUDY_SPECS, run_study


def test_unknown_distribution_raises():
    with pytest.raises(KeyError, match="no study spec"):
        run_study("nope", 100, 10)


@pytest.mark.parametrize("name", sorted(STUDY_SPECS))
def test_mean_estimate_near_truth(name):
    result = run_study(name, n_samples=1000, n_experiments=50, seed=0)
    for key, true in result.true_params.items():
        assert result.mean_estimate[key] == pytest.approx(true, abs=0.03), key


def test_observed_variance_tracks_theory():
    """Efficiency in action: observed Var(p_hat) ~ p(1-p)/n across experiments."""
    result = run_study("bernoulli", n_samples=500, n_experiments=200, seed=0)
    observed = result.observed_variance["p"]
    theoretical = result.theoretical_variance["p"]
    assert theoretical == pytest.approx(0.3 * 0.7 / 500)
    assert 0.5 * theoretical < observed < 2.0 * theoretical


def test_variance_shrinks_with_n():
    small = run_study("gaussian", n_samples=10, n_experiments=100, seed=0)
    large = run_study("gaussian", n_samples=1000, n_experiments=100, seed=0)
    assert large.observed_variance["mu"] < small.observed_variance["mu"]


def test_single_experiment_has_nan_variance():
    result = run_study("bernoulli", n_samples=100, n_experiments=1, seed=0)
    assert np.isnan(result.observed_variance["p"])
    assert len(result.estimates) == 1
