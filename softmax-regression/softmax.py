r"""Softmax regression: a multinoulli whose parameter is a function of X.

Bernoulli -> multinoulli, exactly as promised. For K classes,

    Y | X=x ~ Multinoulli( p_1(x), ..., p_K(x) ),
    p_k(x) = softmax(W x)_k = exp(w_k . x) / sum_j exp(w_j . x),

softmax being to the simplex what the sigmoid is to (0,1). W is a K x d
matrix, one weight vector per class. The conditional NLL of a sample is
the cross-entropy -log p_{y_i}(x_i), and its gradient w.r.t. row w_k is

    grad_{w_k} = ( p_k(x_i) - 1[y_i = k] ) x_i ,

i.e. (softmax - one_hot) outer x — residual times features, vectorized
over classes; for K = 2 it collapses to logistic regression. No closed
form (transcendental stationarity), but convex, so GD finds the optimum.

Routes: GD, SGD by hand, SGD via torch F.cross_entropy on the identical
schedule (trajectory identity assert). Because the model is
well-specified, the fitted classifier approaches the multiclass BAYES
floor  E[ 1 - max_k p_k(X) ].

The animation (second window): SGD training the classifier, ONE SAMPLE
PER ITERATION. The decision REGIONS are drawn as class-colored fields
whose opacity is the model's confidence max_k p_k(x) — at W = 0 the
plane is blank (every class ties at 1/3), and the regions materialize
and sharpen as samples arrive; the yellow ring marks the single point
whose gradient (softmax - onehot) ⊗ x produced the step. True region
borders are dotted black. A live report card shows the full 3x3
confusion matrix with per-class recall, accuracy, macro-F1 and NLL;
below, the NLL gap (descent, then the noise ball, Polyak line under
it) and the test error settling on the multiclass Bayes floor
E[1 - max p] while the region borders keep wobbling.

Note on identifiability: adding the same vector to every row of W leaves
softmax unchanged, so W itself is only identified up to that shift — we
therefore compare fitted PROBABILITIES to true probabilities, not W to W*.

Run me with F5. Companion derivations: softmax-regression.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------------------------------------------------------------
# Data: 3 classes in 2D, sampled from a true softmax model
# ----------------------------------------------------------------------
SEED, N, K, D = 7, 6_000, 3, 3            # D = 1 intercept + 2 features
rng = np.random.default_rng(SEED)

W_TRUE = np.array([[0.0, 2.0, 0.0],       # class 0 pulls +x1
                   [0.0, -1.0, 1.8],      # class 1 pulls -x1, +x2
                   [0.5, -1.0, -1.8]])    # class 2 pulls -x1, -x2

X = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])


def softmax(Z):
    """Row-wise softmax with the max-shift trick for stability."""
    E = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)


P_true = softmax(X @ W_TRUE.T)                     # (N, K)
y = (rng.uniform(size=N)[:, None] > np.cumsum(P_true, axis=1)).sum(axis=1)

GD_LR, GD_STEPS = 0.5, 3_000
SGD_LR, SGD_EPOCHS = 0.05, 3


def nll(W):
    P = softmax(X @ W.T)
    return float(-np.mean(np.log(P[np.arange(N), y] + 1e-12)))


# ----------------------------------------------------------------------
# (1) Gradient descent.  Full gradient: (1/n) (P - Y_onehot)^T X
# ----------------------------------------------------------------------
Y_onehot = np.eye(K)[y]                            # (N, K)


def gradient_descent():
    W = np.zeros((K, D))
    history = []
    for _ in range(GD_STEPS):
        P = softmax(X @ W.T)
        W = W - GD_LR * (P - Y_onehot).T @ X / N
        history.append(nll(W))
    return W, history


schedule = np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


# ----------------------------------------------------------------------
# (2) SGD by hand: grad rows = (softmax(Wx) - onehot(y)) outer x
# ----------------------------------------------------------------------
def sgd_manual():
    W = np.zeros((K, D))
    trajectory = []
    for i in schedule:
        p = softmax(X[i] @ W.T)
        W = W - SGD_LR * np.outer(p - Y_onehot[i], X[i])
        trajectory.append(W.copy())
    return trajectory


# ----------------------------------------------------------------------
# (3) The same SGD via torch F.cross_entropy (identical schedule)
# ----------------------------------------------------------------------
def sgd_torch():
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()
    W = torch.zeros((K, D), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([W], lr=SGD_LR)
    trajectory = []
    for i in schedule:
        optimizer.zero_grad()
        logits = (W @ X_t[i]).unsqueeze(0)          # (1, K)
        loss = torch.nn.functional.cross_entropy(logits, y_t[i].view(1))
        loss.backward()      # autograd: (softmax - onehot) outer x
        optimizer.step()
        trajectory.append(W.detach().numpy().copy())
    return trajectory


W_gd, gd_hist = gradient_descent()
traj_sgd = sgd_manual()
traj_torch = sgd_torch()

assert np.allclose(traj_sgd, traj_torch, atol=1e-10), \
    "autograd disagrees with the hand-derived gradient"
W_sgd = np.mean(traj_sgd[-N:], axis=0)

# ----------------------------------------------------------------------
# Evaluate: probabilities (identifiable), accuracy vs the Bayes floor
# ----------------------------------------------------------------------
M = 100_000
X_test = np.column_stack([np.ones(M), rng.normal(size=(M, 2))])
P_test_true = softmax(X_test @ W_TRUE.T)
y_test = (rng.uniform(size=M)[:, None] > np.cumsum(P_test_true, axis=1)).sum(axis=1)

P_test_fit = softmax(X_test @ W_gd.T)
prob_err = float(np.mean(np.abs(P_test_fit - P_test_true)))

bayes_error = float(np.mean(1 - P_test_true.max(axis=1)))
fitted_error = float(np.mean(P_test_fit.argmax(axis=1) != y_test))

np.set_printoptions(precision=4, suppress=True)
print(f"(1) GD    final NLL       : {gd_hist[-1]:.6f}")
print(f"(2) SGD   (Polyak) NLL    : {nll(W_sgd):.6f}")
print(f"max |traj(2) - traj(3)|   : {np.max(np.abs(np.array(traj_sgd) - np.array(traj_torch))):.2e}")
print("autograd == hand gradient (allclose): OK")
print(f"mean |p_fit - p_true|     : {prob_err:.4f}   (W is only identified up to")
print("                             a shared row-shift; probabilities are the object)")
print(f"Bayes error E[1 - max p]  : {bayes_error:.4f}")
print(f"fitted classifier error   : {fitted_error:.4f}")
assert prob_err < 0.02
assert fitted_error < bayes_error + 0.01

# ----------------------------------------------------------------------
# Picture: decision regions, convergence, probability calibration
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

g = np.linspace(-3, 3, 300)
GX, GY = np.meshgrid(g, g)
G = np.column_stack([np.ones(GX.size), GX.ravel(), GY.ravel()])
regions = softmax(G @ W_gd.T).argmax(axis=1).reshape(GX.shape)
ax1.contourf(GX, GY, regions, levels=[-.5, .5, 1.5, 2.5],
             colors=["#dbe7f6", "#ddf1e6", "#fbe9e7"])
sub = slice(0, 1500)
ax1.scatter(X[sub, 1], X[sub, 2], c=y[sub], cmap="viridis", s=5, alpha=.7)
ax1.set(xlabel="x1", ylabel="x2", title="fitted decision regions (K = 3)")

ax2.semilogy(np.maximum(np.array(gd_hist) - gd_hist[-1], 1e-16),
             color="#2a78d6", lw=1.5)
ax2.set(xlabel="GD step", ylabel="NLL - final (log)",
        title="convex cross-entropy: one basin")

ax3.plot(P_test_true[:4000, 0], P_test_fit[:4000, 0], ".", ms=2,
         color="#2a78d6", alpha=.4)
ax3.plot([0, 1], [0, 1], color="#eb6834", lw=1.5)
ax3.set(xlabel="true p_0(x)", ylabel="fitted p_0(x)",
        title="estimated class-0 probability function")

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()

# ----------------------------------------------------------------------
# Animation: SGD carving the decision regions, one sample per iteration
# ----------------------------------------------------------------------
from matplotlib.animation import FuncAnimation  # noqa: E402

XR, YR = 4.6, 3.0
agx = np.linspace(-XR, XR, 230)
agy = np.linspace(-YR, YR, 150)
AGX, AGY = np.meshgrid(agx, agy)
AG = np.column_stack([np.ones(AGX.size), AGX.ravel(), AGY.ravel()])
CLASS_RGB = np.array([[42, 120, 214],      # class 0 — blue
                      [61, 155, 53],       # class 1 — green
                      [235, 104, 52]]) / 255.0   # class 2 — orange


def region_img(W_):
    """Decision regions as an RGBA image: color = argmax class,
    opacity = confidence max_k p_k, rescaled so a three-way tie (1/3,
    the W = 0 state) is fully transparent. The classifier literally
    fades in as it learns."""
    P = softmax(AG @ W_.T)
    rgb = CLASS_RGB[P.argmax(axis=1)]
    alpha = np.clip((P.max(axis=1) - 1 / K) / (1 - 1 / K), 0, 1) * 0.5
    return np.concatenate([rgb, alpha[:, None]],
                          axis=1).reshape(AGX.shape + (4,))


sgd_snaps = [np.zeros((K, D))] + traj_sgd
N_SGD = len(traj_sgd)
FRAME_STEPS = np.r_[0, np.unique(np.geomspace(1, N_SGD,
                                              160).astype(int))]

# per-frame diagnostics on a held-out subset: the K x K confusion matrix
sub_t = slice(0, 20_000)
Xt_s, yt_s = X_test[sub_t], y_test[sub_t]


def confusion(W_):
    pred = softmax(Xt_s @ W_.T).argmax(axis=1)
    C = np.zeros((K, K), dtype=int)
    np.add.at(C, (yt_s, pred), 1)
    return C


conf_snaps = [confusion(sgd_snaps[s]) for s in FRAME_STEPS]
err_snaps = [1 - np.trace(C) / C.sum() for C in conf_snaps]
nll_snaps = [nll(sgd_snaps[s]) for s in FRAME_STEPS]
best_nll = gd_hist[-1]
polyak_gap = max(nll(W_sgd) - best_nll, 1e-16)


def report(k):
    """The multiclass report card: confusion matrix, per-class recall,
    accuracy, macro-F1 (Theory/evaluation for the definitions)."""
    C = conf_snaps[k]
    rec = np.diag(C) / np.maximum(C.sum(axis=1), 1)
    prec = np.diag(C) / np.maximum(C.sum(axis=0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    head = "        " + "".join(f"  pred{j}" for j in range(K))
    rows = "\n".join(
        f"true {i}  " + "".join(f"{C[i, j]:7d}" for j in range(K))
        + f"   rec {rec[i]:5.3f}" for i in range(K))
    return (f"{head}\n{rows}\n\n"
            f"accuracy {np.trace(C) / C.sum():5.3f}   "
            f"macro-F1 {f1.mean():5.3f}   NLL {nll_snaps[k]:.4f}")


figA = plt.figure(figsize=(11.5, 8.6))
gsA = figA.add_gridspec(2, 2, height_ratios=[1.9, 1],
                        hspace=.28, wspace=.24)
axA = figA.add_subplot(gsA[0, :])
axB = figA.add_subplot(gsA[1, 0])
axC = figA.add_subplot(gsA[1, 1])
figA.suptitle("SGD, live: one sample per iteration carves the "
              "decision regions (K = 3)", fontsize=12)

im = axA.imshow(region_img(sgd_snaps[0]), extent=(-XR, XR, -YR, YR),
                origin="lower", zorder=0)
axA.scatter(X[sub, 1], X[sub, 2], c=CLASS_RGB[y[sub]], s=8, alpha=.6,
            zorder=1)
regions_true = softmax(AG @ W_TRUE.T).argmax(axis=1).reshape(AGX.shape)
axA.contour(AGX, AGY, regions_true, levels=[0.5, 1.5], colors="k",
            linestyles=":", linewidths=1.3, zorder=2)
ring, = axA.plot([], [], "o", mfc="none", mec="#f0b400", ms=14, mew=2.8,
                 label="the sample that just pushed", zorder=4)
titleA = axA.set_title("")
axA.set(xlim=(-XR, XR), ylim=(-YR, YR), xlabel="x1", ylabel="x2")
axA.set_aspect("equal")
axA.legend(frameon=False, fontsize=8, loc="upper left")
report_txt = axA.text(.99, .02, "", transform=axA.transAxes,
                      family="monospace", fontsize=8, ha="right",
                      va="bottom", zorder=5,
                      bbox=dict(facecolor="white", alpha=.85,
                                edgecolor="#999", boxstyle="round"))

curve_nll, = axB.plot([], [], color="#1baf7a", lw=1.5,
                      label="SGD iterate")
axB.axhline(polyak_gap, color="#2a78d6", ls="--", lw=1.2,
            label=f"Polyak average {polyak_gap:.1e}")
axB.set_xscale("log")
axB.set_yscale("log")
axB.set(xlim=(1, N_SGD), ylim=(polyak_gap / 10, 2),
        xlabel="SGD step (log)", ylabel="NLL − best (log)",
        title="descent, then the noise ball; averaging beats it")
axB.legend(frameon=False, fontsize=8)

axC.axhline(bayes_error, color="#0b0b0b", ls="--", lw=1.2,
            label=f"Bayes floor E[1 − max p] = {bayes_error:.4f}")
curve_err, = axC.plot([], [], color="#2a78d6", lw=1.5,
                      label="test error")
axC.set_xscale("log")
axC.set(xlim=(1, N_SGD), ylim=(bayes_error - .02, .70),
        xlabel="SGD step (log)", ylabel="test error",
        title="the wobble never reaches the error")
axC.legend(frameon=False, fontsize=8)

for ax in (axB, axC):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
# no tight_layout: the equal-aspect big panel manages its own margins

FRAMES = len(FRAME_STEPS)
gap_snaps = np.maximum(np.array(nll_snaps) - best_nll, 1e-16)


def update(k):
    s = FRAME_STEPS[k]
    W_ = sgd_snaps[s]
    im.set_data(region_img(W_))
    if s >= 1:                                # the sample used at step s
        i = schedule[s - 1]
        ring.set_data([X[i, 1]], [X[i, 2]])
    shown = FRAME_STEPS[1:k + 1]              # step 0 not on log axis
    curve_nll.set_data(shown, gap_snaps[1:k + 1])
    curve_err.set_data(shown, err_snaps[1:k + 1])
    epoch = (s - 1) // N + 1 if s else 0
    titleA.set_text(f"step {s:>6,d} (epoch {epoch})")
    report_txt.set_text(report(k))
    return im, ring, curve_nll, curve_err, titleA, report_txt


ani = FuncAnimation(figA, update, frames=FRAMES, interval=80,
                    blit=False, repeat=True)   # keep a ref!
plt.show()
