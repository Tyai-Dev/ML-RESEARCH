r"""Linear regression: the normal equations vs gradient methods.

Model
-----
y = X theta* + noise, with X in R^{n x d} (first column = 1 for the
intercept) and noise ~ N(0, s^2). Least squares minimizes the mean squared
error — which is ALSO the Gaussian MLE for theta (see linear-regression.tex):

    J(theta) = (1/2n) ||X theta - y||^2 .

Routes
------
(1) Theoretical solution. grad J = (1/n) X^T (X theta - y) = 0 gives the
    NORMAL EQUATIONS  X^T X theta = X^T y, i.e. theta_hat =
    (X^T X)^{-1} X^T y  (unique when X has full column rank).
(2) Gradient descent:   theta <- theta - eta (1/n) X^T (X theta - y).
(3) SGD, one sample:    theta <- theta - eta (x_i^T theta - y_i) x_i .
(4) The same SGD via PyTorch autograd, identical schedule => identical
    trajectory (the by-now-standard proof by computation).

Why we sometimes AVOID the closed form even though it exists
------------------------------------------------------------
(a) Cost/memory: forming X^T X is O(n d^2) and solving is O(d^3) — for
    d ~ 10^5+ features that is infeasible, while one SGD step is O(d).
(b) Conditioning: cond(X^T X) = cond(X)^2. Squaring the condition number
    can destroy accuracy in floating point. Demonstrated below: on an
    ill-conditioned design in float32, the normal equations lose most
    digits while QR-based lstsq (and gradient methods) stay sane.
(c) Streaming/online: SGD updates on data that arrives one sample at a
    time and never stores the dataset.
(d) Generality: the moment the model stops being linear (add a sigmoid, a
    hidden layer), the closed form dies; the gradient recipe survives
    unchanged. The closed form is a property of THIS loss, not a method.

Run me with F5. Companion derivations: linear-regression.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------------------------------------------------------------
# Data:  y = 2 + 3 u - 5 v + noise   (theta* = [2, 3, -5])
# ----------------------------------------------------------------------
SEED, N = 7, 2_000
rng = np.random.default_rng(SEED)

u = rng.normal(size=N)
v = rng.normal(size=N)
X = np.column_stack([np.ones(N), u, v])          # design matrix with intercept
THETA_TRUE = np.array([2.0, 3.0, -5.0])
y = X @ THETA_TRUE + rng.normal(0.0, 1.0, size=N)

GD_LR, GD_STEPS = 0.1, 500
SGD_LR, SGD_EPOCHS = 0.01, 3


# ----------------------------------------------------------------------
# (1) Theoretical solution: the normal equations
# ----------------------------------------------------------------------
# grad J = (1/n) X^T (X theta - y) = 0  =>  X^T X theta = X^T y.
# np.linalg.solve does the d x d solve; never invert explicitly.
theta_closed = np.linalg.solve(X.T @ X, X.T @ y)


# ----------------------------------------------------------------------
# (2) Gradient descent — the full gradient, every step  (O(n d) per step)
# ----------------------------------------------------------------------
def gradient_descent() -> np.ndarray:
    theta = np.zeros(3)
    for _ in range(GD_STEPS):
        theta -= GD_LR * (X.T @ (X @ theta - y)) / N
    return theta


# ----------------------------------------------------------------------
# Shared sample schedule for (3) and (4)
# ----------------------------------------------------------------------
schedule = np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


# ----------------------------------------------------------------------
# (3) SGD — one sample per step, gradient by hand  (O(d) per step)
# ----------------------------------------------------------------------
# Per sample:  J_i(theta) = (1/2)(x_i^T theta - y_i)^2,
# grad J_i = (x_i^T theta - y_i) x_i — unbiased for the full gradient.
def sgd_manual() -> tuple[np.ndarray, list[np.ndarray]]:
    theta = np.zeros(3)
    trajectory = []
    for i in schedule:
        residual = X[i] @ theta - y[i]
        theta -= SGD_LR * residual * X[i]
        trajectory.append(theta.copy())
    return theta, trajectory


# ----------------------------------------------------------------------
# (4) The same SGD via PyTorch autograd
# ----------------------------------------------------------------------
def sgd_torch() -> list[np.ndarray]:
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    theta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([theta], lr=SGD_LR)
    trajectory = []
    for i in schedule:
        optimizer.zero_grad()
        loss = 0.5 * (X_t[i] @ theta - y_t[i]) ** 2
        loss.backward()          # autograd: (x_i^T theta - y_i) x_i
        optimizer.step()
        trajectory.append(theta.detach().numpy().copy())
    return trajectory


theta_gd = gradient_descent()
theta_sgd_last, traj_sgd = sgd_manual()
traj_torch = sgd_torch()

assert np.allclose(traj_sgd, traj_torch, atol=1e-10), \
    "autograd disagrees with the hand-derived gradient"

theta_sgd = np.mean(traj_sgd[-N:], axis=0)   # Polyak average, final epoch


# ----------------------------------------------------------------------
# The conditioning demonstration: why (X^T X)^{-1} X^T y can be a bad idea
# ----------------------------------------------------------------------
# Two nearly collinear features make X ill-conditioned; forming X^T X
# SQUARES that condition number. In float32 the normal equations then lose
# most of their accuracy, while QR-based lstsq works on X directly and
# keeps cond(X), not cond(X)^2.
w1 = rng.normal(size=N)
w2 = w1 + 1e-4 * rng.normal(size=N)          # nearly identical column
X_ill = np.column_stack([np.ones(N), w1, w2])
theta_ill_true = np.array([1.0, 2.0, 2.0])
y_ill = X_ill @ theta_ill_true + rng.normal(0, 0.1, size=N)

X32, y32 = X_ill.astype(np.float32), y_ill.astype(np.float32)
theta_normal32 = np.linalg.solve(X32.T @ X32, X32.T @ y32)     # cond(X)^2 route
theta_lstsq32 = np.linalg.lstsq(X32, y32, rcond=None)[0]       # cond(X) route
theta_ref = np.linalg.lstsq(X_ill, y_ill, rcond=None)[0]       # float64 reference

cond_X = np.linalg.cond(X_ill)
err_normal = np.linalg.norm(theta_normal32 - theta_ref)
err_lstsq = np.linalg.norm(theta_lstsq32 - theta_ref)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
np.set_printoptions(precision=5, suppress=True)
print(f"theta*                    : {THETA_TRUE}")
print(f"(1) normal equations      : {theta_closed}")
print(f"(2) GD, final iterate     : {theta_gd}")
print(f"(3) SGD, Polyak average   : {theta_sgd}")
print(f"max |traj(3) - traj(4)|   : {np.max(np.abs(np.array(traj_sgd) - np.array(traj_torch))):.2e}")
print("autograd == hand gradient (allclose): OK")
print()
print("--- conditioning demo (float32, nearly collinear features) ---")
print(f"cond(X) = {cond_X:.1e}   =>   cond(X^T X) = {cond_X**2:.1e}")
print(f"normal-equations error vs float64 reference : {err_normal:.3f}")
print(f"lstsq (QR) error vs float64 reference       : {err_lstsq:.6f}")


# ----------------------------------------------------------------------
# Picture: fit, SGD trajectories per coordinate, conditioning bars
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

order = np.argsort(u)
ax1.plot(u[order], (y - X[:, 2] * theta_closed[2])[order], ".", ms=2,
         color="#c3c2b7", label="y - v-part (data)")
ax1.plot(u[order], (theta_closed[0] + theta_closed[1] * u)[order],
         color="#2a78d6", lw=2, label="fit in u")
ax1.set(xlabel="u", ylabel="partial residual", title="the fitted plane, u-slice")
ax1.legend(frameon=False, fontsize=8)

colors = ["#2a78d6", "#1baf7a", "#e87ba4"]
for j, (c, name) in enumerate(zip(colors, ["intercept", "u", "v"])):
    ax2.plot([th[j] for th in traj_sgd], color=c, lw=.7, label=f"SGD {name}")
    ax2.axhline(theta_closed[j], color=c, ls="--", lw=1)
ax2.set(xlabel="SGD step", title="SGD coordinates -> normal-equation values")
ax2.legend(frameon=False, fontsize=8)

ax3.bar(["normal eq.\n(float32)", "lstsq QR\n(float32)"],
        [err_normal, err_lstsq], color=["#e34948", "#1baf7a"])
ax3.set_yscale("log")
ax3.set(ylabel="||error|| vs float64", title=f"cond(X)={cond_X:.0e}: squaring it hurts")

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
