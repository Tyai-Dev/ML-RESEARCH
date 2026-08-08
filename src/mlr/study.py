"""Repeated-sampling MLE studies.

Draw ``n_experiments`` independent samples of size ``n_samples``, compute the
closed-form MLE on each, and aggregate — the observed spread of the estimates
across experiments is the *sampling distribution* of the estimator, compared
side by side with its theoretical variance (consistency and efficiency of
MLE, made visible).
"""

from dataclasses import dataclass, field

import numpy as np

from mlr.data import get_dataset
from mlr.distributions import get_distribution

# distribution name -> (sampler dataset, default true parameters)
STUDY_SPECS: dict[str, tuple[str, dict]] = {
    "bernoulli": ("bernoulli-samples", {"p": 0.3}),
    "multinoulli": ("multinoulli-samples", {"probs": (0.5, 0.3, 0.2)}),
    "gaussian": ("gaussian-samples", {"mu": 2.0, "sigma": 1.5}),
}


def _flatten(params: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in params.items():
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.size == 1:
            out[key] = float(arr[0])
        else:
            out.update({f"{key}{i}": float(v) for i, v in enumerate(arr)})
    return out


@dataclass
class StudyResult:
    distribution: str
    formula: str
    n_samples: int
    n_experiments: int
    true_params: dict[str, float]
    mean_estimate: dict[str, float]
    observed_variance: dict[str, float]  # across experiments (ddof=1)
    theoretical_variance: dict[str, float]  # at the true parameters
    mean_nll: float
    estimates: list[dict[str, float]] = field(repr=False, default_factory=list)


def run_study(
    distribution: str, n_samples: int, n_experiments: int, seed: int = 0
) -> StudyResult:
    if distribution not in STUDY_SPECS:
        raise KeyError(
            f"no study spec for {distribution!r}; available: {sorted(STUDY_SPECS)}"
        )
    dataset_name, true_params = STUDY_SPECS[distribution]
    dist = get_distribution(distribution)
    sampler = get_dataset(dataset_name)

    estimates: list[dict[str, float]] = []
    nlls: list[float] = []
    for i in range(n_experiments):
        X = sampler(n=n_samples, seed=seed + i, **true_params)
        fitted = dist.mle(X)
        estimates.append(_flatten(fitted))
        nlls.append(dist.nll(fitted, X))

    keys = estimates[0].keys()
    stacked = {key: np.array([e[key] for e in estimates]) for key in keys}
    natural_true = _true_params_dict(distribution, true_params)
    return StudyResult(
        distribution=distribution,
        formula=dist.formula,
        n_samples=n_samples,
        n_experiments=n_experiments,
        true_params=_flatten(natural_true),
        mean_estimate={k: float(v.mean()) for k, v in stacked.items()},
        observed_variance={
            k: float(v.var(ddof=1)) if n_experiments > 1 else float("nan")
            for k, v in stacked.items()
        },
        theoretical_variance=_flatten(
            dist.estimator_variance(natural_true, n_samples)
        ),
        mean_nll=float(np.mean(nlls)),
        estimates=estimates,
    )


def _true_params_dict(distribution: str, spec_params: dict) -> dict:
    """Sampler kwargs -> the distribution's natural parameter dict."""
    if distribution == "multinoulli":
        return {"p": np.asarray(spec_params["probs"], dtype=float)}
    return dict(spec_params)
