r"""Logistic regression by Newton / IRLS — what it is, and why 8 steps.

WHAT NEWTON'S METHOD IS. GD uses only the slope: it steps downhill a
fixed learning rate, knowing nothing about how curved the valley is.
Newton's method fits a full quadratic (paraboloid) to the loss at the
current point — slope AND curvature — and jumps straight to that
paraboloid's minimum:

    w  <-  w - H^{-1} g,        g = gradient,  H = Hessian.

No learning rate: the curvature H decides the step size in every
direction at once (steep directions get small steps, flat ones large).
Near the optimum, where the quadratic fit is nearly exact, the error
doesn't shrink by a factor — it SQUARES: 1e-2 -> 1e-4 -> 1e-8.
Quadratic convergence. That is why the log below is 8 lines long
where logistic_gd.py needed 3000.

THE PIECES, for our loss (derived in logistic-regression.tex):

    g = (1/n) X^T (p - y),
    H = (1/n) X^T S X,   S = diag( p_i (1 - p_i) ).

Look at S: the weights are the per-sample BERNOULLI VARIANCES —
confident samples (p near 0 or 1) contribute little curvature,
uncertain ones (p near 1/2) a lot. H is PSD, so the NLL is convex:
Newton's jump target is the one global optimum.

WHY IT'S ALSO CALLED IRLS ("iteratively reweighted least squares"):
rearranging the Newton step gives  (X^T S X) d = X^T (y - p)  — which
is exactly the NORMAL EQUATION of a weighted least-squares problem
(linear-regression/, with weights S). So Newton-for-logistic =
"solve a weighted linear regression, update the weights, repeat":
the statistician's classical algorithm, and what R's glm() runs.

THE PRICE: each step builds and solves a d x d system — O(n d^2 + d^3)
— fine at d = 3, ruinous at d = 10^6. That cost is why deep learning
runs on first-order methods and why Theory/optimizers exists.

Run me with F5. Derivations: logistic-regression.tex.
"""

import numpy as np

from common import evaluate, gradient, make_data, nll, predict, sigmoid

NEWTON_STEPS = 8


def train(X, y, X_test, y_test, verbose=True, steps=NEWTON_STEPS,
          damping=0.0):
    """The training loop — watch ||grad|| SQUARE itself away.

    damping > 0 adds lambda*I to H before solving. Two reasons real
    data needs it (logistic_mnist.py): dead features (pixels that are
    zero in every image) put zero rows/columns in H — singular, no
    solve; and (near-)separable data sends the optimum to infinity —
    damping shrinks the jump (this is the Levenberg-Marquardt idea,
    and is exactly Newton on the L2-regularized loss)."""
    n = len(y)
    w = np.zeros(X.shape[1])
    if verbose:
        print(f"Newton/IRLS: {steps} steps, no learning rate "
              f"(H picks every step)"
              + (f", damping {damping:g}" if damping else ""))
        print(f"{'step':>5} {'train NLL':>10} {'||grad||':>11} "
              f"{'test acc':>9}")
    for step in range(1, steps + 1):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / n
        S = p * (1 - p)                      # Bernoulli variances!
        H = (X.T * S) @ X / n
        if damping:
            H = H + damping * np.eye(len(w))
        w = w - np.linalg.solve(H, g)        # the jump to the
        if verbose:                          # quadratic's minimum
            acc = float(np.mean(predict(w, X_test) == y_test))
            print(f"{step:>5} {nll(w, X, y):>10.6f} "
                  f"{np.linalg.norm(gradient(w, X, y)):>11.2e} "
                  f"{acc:>9.4f}")
    return w


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_data()
    w = train(X_train, y_train, X_test, y_test)

    g_final = np.linalg.norm(gradient(w, X_train, y_train))
    print(f"\n{NEWTON_STEPS} steps to ||grad|| = {g_final:.1e} — "
          f"the same optimum logistic_gd.py crawls to in 3000")
    assert g_final < 1e-12, "quadratic convergence should hit machine 0"

    evaluate(w, X_test, y_test, "Newton/IRLS")
