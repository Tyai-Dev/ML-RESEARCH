r"""Multivariate Gaussian — geometry, closure, conditioning, verified.

The distribution: x ~ N(mu, Sigma) on R^d, density proportional to
exp(-(x-mu)^T Sigma^{-1} (x-mu) / 2). Sigma's eigenvectors are the
ellipse axes; its eigenvalues, the variances along them.

Claims verified below (each an assert, 200k samples):

MOMENTS. Empirical mean = mu, empirical covariance = Sigma.

CLOSURE UNDER LINEAR MAPS. A x + b ~ N(A mu + b, A Sigma A^T) — the
property that makes Gaussians the linear algebra of probability.
Checked on a random A by comparing transformed-sample moments.

CONDITIONING IS LINEAR REGRESSION. For a 2D Gaussian,
    E[x0 | x1] = mu0 + (S01/S11)(x1 - mu1),
    Var[x0 | x1] = S00 - S01^2/S11    (independent of x1!).
Verified by binning the sample on x1: the empirical conditional means
lie on the predicted LINE, and the conditional variance is flat at
the predicted value. This is the generative justification of linear
regression (Algorithms/regression/linear): if (x, y) are jointly
Gaussian, the best predictor of y from x IS a line.

MLE. mu_hat = xbar and Sigma_hat = (1/n) sum (x-xbar)(x-xbar)^T; the
1/n makes Sigma_hat BIASED: E[Sigma_hat] = (n-1)/n Sigma — measured
at n = 5 over 20,000 replications (empirical factor ~0.8), the same
bias as the 1D gaussian folder, now as a matrix statement.

Run me with F5. Derivations: multivariate-gaussian.tex.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)
MU = np.array([1.0, -1.0])
SIGMA = np.array([[2.0, 1.2],
                  [1.2, 1.0]])
N = 200_000
X = rng.multivariate_normal(MU, SIGMA, size=N)

# ---- moments ---------------------------------------------------------
assert np.abs(X.mean(axis=0) - MU).max() < 0.01
assert np.abs(np.cov(X.T, bias=True) - SIGMA).max() < 0.02
print("moments: mean = mu, cov = Sigma: OK")

# ---- closure under linear maps ---------------------------------------
A = rng.normal(size=(2, 2))
b = np.array([0.5, -2.0])
Y = X @ A.T + b
assert np.abs(Y.mean(axis=0) - (A @ MU + b)).max() < 0.02
assert np.abs(np.cov(Y.T, bias=True) - A @ SIGMA @ A.T).max() < 0.05
print("closure: A x + b ~ N(A mu + b, A Sigma A^T): OK")

# ---- conditioning = linear regression --------------------------------
slope = SIGMA[0, 1] / SIGMA[1, 1]
cvar = SIGMA[0, 0] - SIGMA[0, 1] ** 2 / SIGMA[1, 1]
edges = np.linspace(-3.5, 1.5, 11)
mids, cond_mean, cond_var = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (X[:, 1] >= lo) & (X[:, 1] < hi)
    if m.sum() > 3_000:      # thin tail bins are too noisy to test
        # evaluate the line at the bin's EMPIRICAL mean of x1 (the mass
        # inside a bin is not centered at its midpoint - comparing at
        # the midpoint smuggles in a slope*offset bias)
        mids.append(X[m, 1].mean())
        cond_mean.append(X[m, 0].mean())
        cond_var.append(X[m, 0].var())
mids = np.array(mids)
pred = MU[0] + slope * (mids - MU[1])
err_line = np.abs(np.array(cond_mean) - pred).max()
err_var = np.abs(np.array(cond_var) - cvar).max()
print(f"conditioning: E[x0|x1] on the predicted line to {err_line:.3f};"
      f" Var[x0|x1] flat at {cvar:.3f} to {err_var:.3f}")
assert err_line < 0.05 and err_var < 0.08
print("the best predictor of one coordinate from the other IS a line: OK")

# ---- MLE bias of the 1/n covariance ----------------------------------
n_small, REPS = 5, 20_000
factor = np.mean([np.cov(rng.multivariate_normal(MU, SIGMA, n_small).T,
                         bias=True)[0, 0] for _ in range(REPS)]
                 ) / SIGMA[0, 0]
print(f"MLE bias: E[Sigma_hat]/Sigma at n=5 -> {factor:.3f} "
      f"(theory (n-1)/n = {(n_small - 1) / n_small:.3f})")
assert abs(factor - (n_small - 1) / n_small) < 0.02
print("all claims verified: OK")

# ---- picture ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax1.plot(X[:3000, 0], X[:3000, 1], ".", ms=2, color="#c3c2b7")
evals, evecs = np.linalg.eigh(SIGMA)
ang = np.linspace(0, 2 * np.pi, 100)
circ = np.stack([np.cos(ang), np.sin(ang)])
for k, c in [(1, "#eb6834"), (2, "#e34948")]:
    ell = MU[:, None] + evecs @ (k * np.sqrt(evals)[:, None] * circ)
    ax1.plot(ell[0], ell[1], color=c, lw=2, label=f"{k}σ ellipse")
ax1.set(title="the geometry: Sigma's eigen-ellipses", xlabel="x0",
        ylabel="x1")
ax1.legend(frameon=False, fontsize=8)
ax1.set_aspect("equal")

ax2.plot(X[:3000, 1], X[:3000, 0], ".", ms=2, color="#c3c2b7")
gx = np.linspace(-3.5, 1.5, 50)
ax2.plot(gx, MU[0] + slope * (gx - MU[1]), color="#eb6834", lw=2,
         label=r"E[$x_0|x_1$]: the regression line")
ax2.errorbar(mids, cond_mean, yerr=np.sqrt(cvar), fmt="o", ms=4,
             color="#2a78d6", label="binned conditional means ± sd")
ax2.set(title="conditioning IS linear regression", xlabel="x1",
        ylabel="x0")
ax2.legend(frameon=False, fontsize=8)
for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
