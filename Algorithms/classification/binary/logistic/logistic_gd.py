r"""Logistic regression by gradient descent — the full gradient, logged.

The update. Every step uses ALL n samples:

    w  <-  w - lr * (1/n) X^T (sigmoid(Xw) - y)

Deterministic — same start, same steps, same answer every run — and
because the NLL is convex in w (Hessian X^T diag(p(1-p)) X / n is
PSD), there is ONE basin: GD cannot get trapped, only be slow.

Watch the log's ||grad|| column: it decays GEOMETRICALLY (a straight
line in log scale — each step multiplies the distance to the optimum
by roughly the same factor). That linear convergence rate is GD's
signature; compare it with logistic_newton.py's log, where the same
column SQUARES itself each step. The price of determinism: every one
of those steps costs a full pass over the dataset — the reason SGD
(logistic_sgd.py) exists.

Closing demo — when the MLE does not exist: on linearly SEPARABLE
data, scaling w up only ever helps, so ||w|| diverges and the
probabilities saturate; watched at the bottom of the log.

Run me with F5. Derivations: logistic-regression.tex.
"""

import numpy as np

from common import evaluate, gradient, make_data, nll, predict, sigmoid

GD_LR = 0.5
GD_STEPS = 3_000
PRINT_EVERY = 250


def train(X, y, X_test, y_test, verbose=True):
    """The training loop: full gradient, fixed step size."""
    w = np.zeros(X.shape[1])
    if verbose:
        print(f"GD: lr={GD_LR}, {GD_STEPS} full-gradient steps "
              f"(each = one pass over {len(y):,} samples)")
        print(f"{'step':>6} {'train NLL':>10} {'||grad||':>10} "
              f"{'test acc':>9}")
    for step in range(1, GD_STEPS + 1):
        g = gradient(w, X, y)
        w -= GD_LR * g
        if verbose and (step % PRINT_EVERY == 0 or step == 1):
            acc = float(np.mean(predict(w, X_test) == y_test))
            print(f"{step:>6} {nll(w, X, y):>10.4f} "
                  f"{np.linalg.norm(g):>10.2e} {acc:>9.4f}")
    return w


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_data()
    w = train(X_train, y_train, X_test, y_test)

    g_final = np.linalg.norm(gradient(w, X_train, y_train))
    print(f"\nfinal ||grad|| = {g_final:.2e}  (stationary => this is "
          f"THE optimum: convexity says there is only one)")
    assert g_final < 1e-6, "GD did not reach the optimum"

    evaluate(w, X_test, y_test, "GD")

    # ------------------------------------------------------------------
    # When the MLE does not exist: separable data
    # ------------------------------------------------------------------
    print("\n--- separable data: watch ||w|| diverge " + "-" * 20)
    X_sep = np.column_stack([np.ones(200), np.linspace(-2, 2, 200)])
    y_sep = (X_sep[:, 1] > 0).astype(float)          # perfectly split
    w_sep = np.zeros(2)
    for k in range(1, 30_001):
        w_sep -= 0.5 * gradient(w_sep, X_sep, y_sep)
        if k % 6_000 == 0:
            print(f"step {k:>6,}   ||w|| = {np.linalg.norm(w_sep):6.1f}"
                  f"   train NLL = {nll(w_sep, X_sep, y_sep):.5f}")
    print("||w|| grows forever: every scaling-up improves every margin,")
    print("so the NLL has no minimizer — regularize or stop early.")
    assert np.linalg.norm(w_sep) > 20
