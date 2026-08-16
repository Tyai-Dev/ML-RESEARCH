r"""Shared definitions for the logistic-regression experiments.

One place for the model, the data, and the evaluation — so the three
solver files (logistic_sgd.py, logistic_gd.py, logistic_newton.py)
contain NOTHING but their own training loop, and all of them fit the
same problem and report on the same held-out test set.

The model (the statistical view, from logistic-regression.tex):
binary classification is estimating a Bernoulli whose parameter is a
function of x —

    Y | X=x  ~  Bernoulli( p(x) ),      p(x) = sigmoid(w . [1, x]),

and the per-sample loss every solver minimizes is the Bernoulli NLL
(cross-entropy)  -[ y log p + (1-y) log(1-p) ],  with gradient

    (sigmoid(w . x) - y) x        — residual times features.

Because the data is generated FROM this model (well-specified), the
Bayes error E[min(p, 1-p)] is computable and no classifier can beat
it — every solver's report is judged against that floor.
"""

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

SEED = 7
N = 25_000                                   # generated, then split
TEST_FRACTION = 0.2
W_TRUE = np.array([0.5, 2.0, -1.5])          # [intercept, w1, w2]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll(w, X, y):
    """Average Bernoulli NLL (= cross-entropy loss) of w on (X, y)."""
    p = sigmoid(X @ w)
    eps = 1e-12
    return float(-np.mean(y * np.log(p + eps)
                          + (1 - y) * np.log(1 - p + eps)))


def gradient(w, X, y):
    """The full gradient: (1/n) X^T (sigmoid(Xw) - y)."""
    return X.T @ (sigmoid(X @ w) - y) / len(y)


def make_data(seed: int = SEED):
    """The dataset and its split. x ~ N(0, I_2); y | x from the true
    model; a leading 1 on every row carries the intercept. Returns
    (X_train, X_test, y_train, y_test)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])
    p = sigmoid(X @ W_TRUE)
    y = (rng.uniform(size=N) < p).astype(np.float64)
    return train_test_split(X, y, test_size=TEST_FRACTION,
                            random_state=seed)


def bayes_error(X_test) -> float:
    """The floor no classifier can beat: E[min(p, 1-p)] under the TRUE
    model (computable only because the data is synthetic)."""
    p = sigmoid(X_test @ W_TRUE)
    return float(np.mean(np.minimum(p, 1 - p)))


def predict(w, X):
    """Labels at the 0.5 threshold (= the Bayes rule for symmetric
    costs; Theory/evaluation)."""
    return (sigmoid(X @ w) > 0.5).astype(int)


def evaluate(w, X_test, y_test, name: str, synthetic: bool = True,
             target_names=("class 0", "class 1")):
    """The final exam, identical for every solver: sklearn's
    classification report on the held-out split. On the synthetic
    problem (default) it also shows the fitted-vs-Bayes-floor gap and
    w vs w*; on real data (synthetic=False) there is no floor and no
    w* — the test set is the only truth."""
    y_pred = predict(w, X_test)
    err = float(np.mean(y_pred != y_test))
    print(f"\n=== {name}: classification report (test, "
          f"n={len(y_test):,}) ===")
    print(classification_report(y_test.astype(int), y_pred,
                                target_names=list(target_names),
                                digits=3))
    if synthetic:
        floor = bayes_error(X_test)
        print(f"test error {err:.4f}  vs Bayes floor {floor:.4f}  "
              f"(gap {err - floor:+.4f})")
        np.set_printoptions(precision=4, suppress=True)
        print(f"w fitted {w}   w* {W_TRUE}")
    else:
        print(f"test error {err:.4f}  (real data: no Bayes floor to "
              f"compare against)")
    return err
