r"""Logistic regression: estimating a Bernoulli whose parameter is a function of X.

The statistical view (the whole point)
--------------------------------------
Binary classification is not a new kind of problem. The joint distribution
of (X, Y) with Y in {0,1} always factors as

    D(x, y) = D_X(x) * Bernoulli(y; p(x)),     p(x) = P(Y=1 | X=x).

Y|X=x is Bernoulli AUTOMATICALLY — binary support forces it; that part
costs nothing. The one and only modeling assumption is the FORM of the
parameter function p(x). The linear model bets on linear LOG-ODDS
(p itself can't be linear — a line escapes [0,1]):

    log( p(x) / (1-p(x)) ) = w . x      =>      p(x) = sigmoid(w . x),

with x carrying a leading 1 for the intercept. So we are doing exactly
the bernoulli/ experiment again, with the constant p promoted to a
function of x — and the machinery composes accordingly:

    per-sample NLL       -[ y log p(x) + (1-y) log(1-p(x)) ]
    per-sample gradient  (sigmoid(w . x) - y) x
                          ^^^^^^^^^^^^^^^^^^  ^
                          bernoulli residual  regression feature vector

The routes
----------
(1) The theoretical route DIES here — the first time in this repo.
    Stationarity gives sum_i (sigmoid(w.x_i) - y_i) x_i = 0: transcendental
    in w, no algebraic solution (bernoulli gave m/n, least squares gave the
    normal equations; logistic gives nothing).
(1') The statistician's classical rescue: Newton / IRLS. The NLL is convex
    (Hessian (1/n) X^T diag(p(1-p)) X >= 0), so Newton's method converges
    in a handful of steps, and each step is a WEIGHTED least-squares solve
    — "iteratively reweighted least squares" — connecting straight back to
    linear-regression/.
(2) Gradient descent on the NLL.
(3) SGD, one sample per step.
(4) The same SGD by PyTorch autograd on binary_cross_entropy_with_logits,
    identical schedule => identical trajectory (the standard identity check).

Because the model is well-specified here (data really is logistic), the
fitted classifier approaches the BAYES floor: no classifier can beat
error E[min(p(X), 1-p(X))].

The animation (second window): the classifier taking shape live — the
probability field p̂(x) = sigmoid(ŵ·[1,x]) as a background that starts
flat (w = 0 => p̂ ≡ 0.5, no boundary at all) and sharpens each
iteration, GD's decision boundary rotating into place while Newton's
snaps there in 8 steps, beside the NLL gap plunging (quadratic vs
linear rate, log scale) and the test error falling onto the Bayes
floor — the line no classifier can cross.

Run me with F5. Companion derivations: logistic-regression.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------------------------------------------------------------
# Data:  x ~ N(0, I_2),  y | x ~ Bernoulli( sigmoid(w* . [1, x]) )
# ----------------------------------------------------------------------
SEED, N = 7, 5_000
W_TRUE = np.array([0.5, 2.0, -1.5])          # [intercept, w1, w2]
rng = np.random.default_rng(SEED)

X = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])
p_true = 1.0 / (1.0 + np.exp(-(X @ W_TRUE)))
y = (rng.uniform(size=N) < p_true).astype(np.float64)

GD_LR, GD_STEPS = 0.5, 3_000
SGD_LR, SGD_EPOCHS = 0.05, 3
NEWTON_STEPS = 8


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll(w, X, y):
    """Average conditional NLL (= cross-entropy loss)."""
    p = sigmoid(X @ w)
    eps = 1e-12
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


# ----------------------------------------------------------------------
# (1') Newton / IRLS — the statistician's solver
# ----------------------------------------------------------------------
# gradient  g = (1/n) X^T (p - y)
# Hessian   H = (1/n) X^T diag(p(1-p)) X   (PSD => NLL convex, one basin)
# Newton:   w <- w - H^{-1} g.  Each step solves the weighted least-squares
# system (X^T S X) d = X^T (y - p) with weights S = diag(p(1-p)) — IRLS.
# Quadratic convergence: gradient norm squares away each iteration.
def newton_irls():
    w = np.zeros(3)
    history, snaps = [], [w.copy()]
    for _ in range(NEWTON_STEPS):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / N
        S = p * (1 - p)                       # the Bernoulli variances!
        H = (X.T * S) @ X / N
        w = w - np.linalg.solve(H, g)
        history.append(nll(w, X, y))
        snaps.append(w.copy())
    return w, history, snaps


# ----------------------------------------------------------------------
# (2) Gradient descent — full gradient, every step
# ----------------------------------------------------------------------
def gradient_descent():
    w = np.zeros(3)
    history, snaps = [], [w.copy()]           # w after every step: the
    for k in range(GD_STEPS):                 # animation picks frames
        w = w - GD_LR * (X.T @ (sigmoid(X @ w) - y)) / N
        history.append(nll(w, X, y))
        snaps.append(w.copy())
    return w, history, snaps


# ----------------------------------------------------------------------
# Shared sample schedule for (3) and (4)
# ----------------------------------------------------------------------
schedule = np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


# ----------------------------------------------------------------------
# (3) SGD — one sample per step, gradient by hand
# ----------------------------------------------------------------------
# Per sample: grad = (sigmoid(w.x_i) - y_i) x_i — unbiased for the full
# gradient. Bernoulli residual times feature vector: the composition of
# this repo's two previous gradients.
def sgd_manual():
    w = np.zeros(3)
    trajectory = []
    for i in schedule:
        w = w - SGD_LR * (sigmoid(X[i] @ w) - y[i]) * X[i]
        trajectory.append(w.copy())
    return trajectory


# ----------------------------------------------------------------------
# (4) The same SGD by PyTorch autograd
# ----------------------------------------------------------------------
def sgd_torch():
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=SGD_LR)
    trajectory = []
    for i in schedule:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            X_t[i] @ w, y_t[i]
        )
        loss.backward()      # autograd: (sigmoid(w.x_i) - y_i) x_i
        optimizer.step()
        trajectory.append(w.detach().numpy().copy())
    return trajectory


w_newton, newton_hist, newton_snaps = newton_irls()
w_gd, gd_hist, gd_snaps = gradient_descent()
traj_sgd = sgd_manual()
traj_torch = sgd_torch()

assert np.allclose(traj_sgd, traj_torch, atol=1e-10), \
    "autograd disagrees with the hand-derived gradient"
w_sgd = np.mean(traj_sgd[-N:], axis=0)       # Polyak average, final epoch

# Newton and GD must agree on the (unique, convex) optimum
assert np.allclose(w_newton, w_gd, atol=1e-4)

# ----------------------------------------------------------------------
# The Bayes floor: no classifier beats E[min(p, 1-p)]
# ----------------------------------------------------------------------
M = 200_000
X_test = np.column_stack([np.ones(M), rng.normal(size=(M, 2))])
p_test = sigmoid(X_test @ W_TRUE)
y_test = (rng.uniform(size=M) < p_test).astype(float)

bayes_error = float(np.mean(np.minimum(p_test, 1 - p_test)))
fitted_error = float(np.mean((sigmoid(X_test @ w_newton) > 0.5) != y_test))

# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
np.set_printoptions(precision=4, suppress=True)
print(f"w*                        : {W_TRUE}")
print(f"(1') Newton/IRLS ({NEWTON_STEPS} steps): {w_newton}   NLL {newton_hist[-1]:.6f}")
print(f"(2)  GD ({GD_STEPS} steps)     : {w_gd}   NLL {gd_hist[-1]:.6f}")
print(f"(3)  SGD, Polyak average  : {w_sgd}")
print(f"max |traj(3) - traj(4)|   : {np.max(np.abs(np.array(traj_sgd) - np.array(traj_torch))):.2e}")
print("autograd == hand gradient (allclose): OK")
print()
print(f"Bayes error  E[min(p,1-p)]: {bayes_error:.4f}")
print(f"fitted classifier error   : {fitted_error:.4f}   (can approach, never beat)")
assert fitted_error < bayes_error + 0.01
assert np.allclose(w_newton, W_TRUE, atol=0.15)   # consistency, ~1/sqrt(n)

# ----------------------------------------------------------------------
# Separability warning: when the MLE does not exist
# ----------------------------------------------------------------------
# On linearly SEPARABLE data the NLL has no minimizer: scaling w up only
# ever helps (every margin grows), so ||w|| diverges and probabilities
# saturate at 0/1. Watch the norm grow under GD:
X_sep = np.column_stack([np.ones(200), np.linspace(-2, 2, 200)])
y_sep = (X_sep[:, 1] > 0).astype(float)              # perfectly separable
w_sep, norms = np.zeros(2), []
for k in range(30_000):
    w_sep -= 0.5 * (X_sep.T @ (sigmoid(X_sep @ w_sep) - y_sep)) / 200
    if k % 3000 == 0:
        norms.append(np.linalg.norm(w_sep))
print(f"\nseparable data: ||w|| along GD = {np.round(norms, 1)}  (diverging —")
print("the MLE does not exist; regularization or early stopping required)")
assert norms[-1] > norms[0] * 3

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

# (a) data + true and fitted decision boundaries (where w.x = 0)
sub = slice(0, 1500)
ax1.scatter(X[sub, 1], X[sub, 2], c=y[sub], cmap="coolwarm", s=6, alpha=.6)
g = np.linspace(-3, 3, 2)
for w_, c, lbl in [(W_TRUE, "#0b0b0b", "true boundary"),
                   (w_newton, "#1baf7a", "fitted boundary")]:
    ax1.plot(g, -(w_[0] + w_[1] * g) / w_[2], ls="--", color=c, lw=1.5, label=lbl)
ax1.set(xlabel="x1", ylabel="x2", title="data and decision boundaries",
        xlim=(-3, 3), ylim=(-3, 3))
ax1.legend(frameon=False, fontsize=8)

# (b) Newton's quadratic convergence vs GD's linear crawl
best = min(newton_hist[-1], gd_hist[-1])
ax2.semilogy(np.maximum(np.array(gd_hist) - best, 1e-16),
             color="#2a78d6", lw=1.5, label="GD")
ax2.semilogy(np.arange(len(newton_hist)) * (GD_STEPS // NEWTON_STEPS),
             np.maximum(np.array(newton_hist) - best, 1e-16),
             "o-", color="#eb6834", label="Newton/IRLS (8 steps)")
ax2.set(xlabel="step (Newton stretched for scale)", ylabel="NLL - best (log)",
        title="no closed form — but convex: both routes, one optimum")
ax2.legend(frameon=False, fontsize=8)

# (c) the point of it all: we estimated the FUNCTION p(x)
ax3.plot(p_test[:4000], sigmoid(X_test[:4000] @ w_newton), ".", ms=2,
         color="#2a78d6", alpha=.4)
ax3.plot([0, 1], [0, 1], color="#eb6834", lw=1.5)
ax3.set(xlabel="true p(x)", ylabel="fitted sigmoid(ŵ·x)",
        title="the estimated Bernoulli parameter, as a function of x")

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()

# ----------------------------------------------------------------------
# Animation: the classifier itself, taking shape iteration by iteration
# ----------------------------------------------------------------------
# The classifier is the probability FIELD p̂(x) = sigmoid(ŵ·[1,x]); the
# decision boundary is its 0.5 level set, the line ŵ·[1,x] = 0. Both
# start flat (w = 0 => p̂ ≡ 0.5, no boundary at all) and take shape as
# training runs. Three synchronized views:
#   left   — the field (background) with GD's boundary rotating into
#            place while Newton's snaps there in 8 steps;
#   middle — the loss progression, NLL - best on a log scale (GD's
#            linear-rate crawl vs Newton's quadratic plunge);
#   right  — what the customer cares about: test error falling to the
#            Bayes floor, the line no classifier can cross.
# Watch the middle and right panels disagree about when training is
# "done": the ERROR settles within ~6 steps but the NLL needs ~1000.
# The 0.5-threshold classifier uses only the DIRECTION of w, which
# locks almost immediately; the remaining thousand steps buy the
# MAGNITUDE of w — i.e. calibrated probabilities, sharpening the field
# from washed-out to confident without moving the boundary.
from matplotlib.animation import FuncAnimation  # noqa: E402

gs_ = np.linspace(-3, 3, 150)
GX, GY = np.meshgrid(gs_, gs_)


def field(w_):
    return sigmoid(w_[0] + w_[1] * GX + w_[2] * GY)


def boundary(w_):
    """The 0.5 level set ŵ·[1,x] = 0; undefined while w = 0."""
    if abs(w_[2]) < 1e-12:
        return gs_, np.full_like(gs_, np.nan)
    return gs_, -(w_[0] + w_[1] * gs_) / w_[2]


# GD converges geometrically, so uniform frames would spend 95% of the
# film on a finished boundary. LOG-SPACED frames instead: dense where
# the boundary is actually rotating (steps 1..100), sparse in the tail.
FRAME_STEPS = np.r_[0, np.unique(np.geomspace(1, GD_STEPS,
                                              110).astype(int))]

# per-frame test error on a held-out subset (the right panel's curve)
sub_t = slice(0, 20_000)
err_snaps = [float(np.mean((sigmoid(X_test[sub_t] @ gd_snaps[s]) > 0.5)
                           != y_test[sub_t])) for s in FRAME_STEPS]
best_nll = min(newton_hist[-1], gd_hist[-1])

figA = plt.figure(figsize=(13.5, 4.4))
gsA = figA.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
axA, axB, axC = [figA.add_subplot(gsA[i]) for i in range(3)]
figA.suptitle("logistic regression, live: the probability field and its "
              "boundary forming", fontsize=11)

im = axA.imshow(field(gd_snaps[0]), extent=(-3, 3, -3, 3),
                origin="lower", cmap="coolwarm", vmin=0, vmax=1,
                alpha=.45, zorder=0)
axA.scatter(X[sub, 1], X[sub, 2], c=y[sub], cmap="coolwarm", s=5,
            alpha=.5, zorder=1)
axA.plot(*boundary(W_TRUE), ls=":", color="#0b0b0b", lw=1.5,
         label="true boundary", zorder=2)
line_gd, = axA.plot([], [], color="#1baf7a", lw=2.2,
                    label="GD boundary", zorder=3)
line_nt, = axA.plot([], [], ls="--", color="#eb6834", lw=1.8,
                    label="Newton boundary", zorder=3)
titleA = axA.set_title("")
axA.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="x1", ylabel="x2")
axA.legend(frameon=False, fontsize=8, loc="upper left")

curve_gd, = axB.plot([], [], color="#1baf7a", lw=2, label="GD")
curve_nt, = axB.plot([], [], "o-", color="#eb6834", ms=4,
                     label="Newton (its 8 steps, to scale)")
axB.set_xscale("log")
axB.set_yscale("log")
axB.set(xlim=(1, GD_STEPS), ylim=(1e-16, 1),
        xlabel="step (log)", ylabel="NLL − best (log)",
        title="the loss progression")
axB.legend(frameon=False, fontsize=8)

axC.axhline(bayes_error, color="#0b0b0b", ls="--", lw=1.2,
            label=f"Bayes floor {bayes_error:.4f}")
curve_err, = axC.plot([], [], color="#2a78d6", lw=2, label="test error")
axC.set_xscale("log")
axC.set(xlim=(1, GD_STEPS), ylim=(bayes_error - .01, .52),
        xlabel="step (log)", ylabel="test error",
        title="falling to the uncrossable line")
axC.legend(frameon=False, fontsize=8)

for ax in (axA, axB, axC):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
figA.tight_layout()

FRAMES = len(FRAME_STEPS)
newton_gap = np.maximum(np.array(newton_hist) - best_nll, 1e-16)
gd_gap = np.maximum(np.array(gd_hist) - best_nll, 1e-16)


def update(k):
    s = FRAME_STEPS[k]
    w_ = gd_snaps[s]
    im.set_data(field(w_))
    line_gd.set_data(*boundary(w_))
    # Newton finishes while GD is still on its first frames
    j = min(k, NEWTON_STEPS)
    line_nt.set_data(*boundary(newton_snaps[j]))
    shown = FRAME_STEPS[1:k + 1]                 # step 0 not on log axis
    curve_gd.set_data(shown, gd_gap[shown - 1])
    curve_nt.set_data(np.arange(1, j + 1), newton_gap[:j])
    curve_err.set_data(shown, err_snaps[1:k + 1])
    titleA.set_text(f"step {s:>4d}   NLL {nll(w_, X, y):.4f}   "
                    f"test err {err_snaps[k]:.4f}")
    return im, line_gd, line_nt, curve_gd, curve_nt, curve_err, titleA


ani = FuncAnimation(figA, update, frames=FRAMES, interval=60,
                    blit=False, repeat=True)   # keep a ref!
plt.show()
