r"""Poisson regression: a Poisson whose rate is a function of X.

The same conditional-distribution recipe as logistic-regression/, applied
to COUNT data:

    Y | X=x ~ Poisson( lambda(x) ),    lambda(x) = exp(w . x)  (log link),

the exponential keeping the rate positive exactly as the sigmoid kept a
probability in (0,1). Per-sample NLL (dropping the log y! constant):

    NLL_i(w) = exp(w . x_i) - y_i (w . x_i)

and its gradient is — once again — residual times features:

    grad_i = (exp(w . x_i) - y_i) x_i = (lambda_i - y_i) x_i .

This is the third instance of the same pattern (Gaussian: (w.x - y) x,
Bernoulli: (sigmoid(w.x) - y) x, Poisson: (exp(w.x) - y) x) — the GLM
unification derived in poisson.tex. Stationarity is transcendental again
(no closed form), the NLL is convex (Hessian X^T diag(lambda) X), and
Newton/IRLS crushes it in a few steps.

Routes: Newton/IRLS, GD, SGD by hand, SGD by autograd (identical
schedule => identical trajectory). Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------------------------------------------------------------
# Data:  x ~ N(0, I_2),  y | x ~ Poisson( exp(w* . [1, x]) )
# ----------------------------------------------------------------------
SEED, N = 7, 5_000
W_TRUE = np.array([0.3, 0.8, -0.5])
rng = np.random.default_rng(SEED)

X = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])
lam_true = np.exp(X @ W_TRUE)
y = rng.poisson(lam_true).astype(np.float64)

GD_LR, GD_STEPS = 0.1, 4_000
SGD_LR, SGD_EPOCHS = 0.01, 3
NEWTON_STEPS = 8


def nll(w):
    z = X @ w
    return float(np.mean(np.exp(z) - y * z))


# ----------------------------------------------------------------------
# (1') Newton / IRLS:  g = (1/n) X^T (lambda - y),  H = (1/n) X^T diag(lambda) X
# ----------------------------------------------------------------------
def newton_irls():
    w = np.zeros(3)
    history = []
    for _ in range(NEWTON_STEPS):
        lam = np.exp(X @ w)
        g = X.T @ (lam - y) / N
        H = (X.T * lam) @ X / N          # weights = Poisson variances!
        w = w - np.linalg.solve(H, g)
        history.append(nll(w))
    return w, history


# ----------------------------------------------------------------------
# (2) Gradient descent
# ----------------------------------------------------------------------
def gradient_descent():
    w = np.zeros(3)
    history = []
    for _ in range(GD_STEPS):
        w = w - GD_LR * (X.T @ (np.exp(X @ w) - y)) / N
        history.append(nll(w))
    return w, history


schedule = np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


# ----------------------------------------------------------------------
# (3) SGD by hand:  grad_i = (exp(w.x_i) - y_i) x_i
# ----------------------------------------------------------------------
def sgd_manual():
    w = np.zeros(3)
    trajectory = []
    for i in schedule:
        w = w - SGD_LR * (np.exp(X[i] @ w) - y[i]) * X[i]
        trajectory.append(w.copy())
    return trajectory


# ----------------------------------------------------------------------
# (4) The same SGD by autograd: loss written verbatim as exp(z) - y z
# ----------------------------------------------------------------------
def sgd_torch():
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=SGD_LR)
    trajectory = []
    for i in schedule:
        optimizer.zero_grad()
        z = X_t[i] @ w
        loss = torch.exp(z) - y_t[i] * z
        loss.backward()
        optimizer.step()
        trajectory.append(w.detach().numpy().copy())
    return trajectory


w_newton, newton_hist = newton_irls()
w_gd, gd_hist = gradient_descent()
traj_sgd = sgd_manual()
traj_torch = sgd_torch()

assert np.allclose(traj_sgd, traj_torch, atol=1e-10), \
    "autograd disagrees with the hand-derived gradient"
assert np.allclose(w_newton, w_gd, atol=1e-3)
w_sgd = np.mean(traj_sgd[-N:], axis=0)

np.set_printoptions(precision=4, suppress=True)
print(f"w*                        : {W_TRUE}")
print(f"(1') Newton/IRLS (8 steps): {w_newton}   NLL {newton_hist[-1]:.6f}")
print(f"(2)  GD ({GD_STEPS} steps)     : {w_gd}   NLL {gd_hist[-1]:.6f}")
print(f"(3)  SGD, Polyak average  : {w_sgd}")
print(f"max |traj(3) - traj(4)|   : {np.max(np.abs(np.array(traj_sgd) - np.array(traj_torch))):.2e}")
print("autograd == hand gradient (allclose): OK")
assert np.allclose(w_newton, W_TRUE, atol=0.1)

# ----------------------------------------------------------------------
# Picture: counts vs rate, convergence, rate calibration
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

order = np.argsort(lam_true)[:2000]
ax1.plot(lam_true[order], y[order], ".", ms=2, color="#c3c2b7", label="counts y")
ax1.plot(lam_true[order], lam_true[order], color="#eb6834", lw=2,
         label="E[y|x] = λ(x)")
ax1.set(xlabel="true rate λ(x)", ylabel="observed count",
        title="count data: mean = rate, variance = rate")
ax1.legend(frameon=False, fontsize=8)

best = min(newton_hist[-1], gd_hist[-1])
ax2.semilogy(np.maximum(np.array(gd_hist) - best, 1e-16), color="#2a78d6",
             lw=1.5, label="GD")
ax2.semilogy(np.arange(len(newton_hist)) * (GD_STEPS // NEWTON_STEPS),
             np.maximum(np.array(newton_hist) - best, 1e-16), "o-",
             color="#eb6834", label="Newton/IRLS")
ax2.set(xlabel="step (Newton stretched)", ylabel="NLL - best (log)",
        title="transcendental stationarity, convex loss")
ax2.legend(frameon=False, fontsize=8)

M = 4000
X_test = np.column_stack([np.ones(M), rng.normal(size=(M, 2))])
ax3.loglog(np.exp(X_test @ W_TRUE), np.exp(X_test @ w_newton), ".", ms=2,
           color="#2a78d6", alpha=.4)
lims = [1e-2, 1e2]
ax3.loglog(lims, lims, color="#eb6834", lw=1.5)
ax3.set(xlabel="true λ(x)", ylabel="fitted exp(ŵ·x)",
        title="the estimated rate, as a function of x")

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
