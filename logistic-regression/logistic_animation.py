r"""The SGD film — the classifier trained one sample per frame.

This file adds nothing mathematical: it replays logistic_sgd.train's
recorded trajectory as the folder's animation (the layout the solver
files deliberately keep out of their way):

  big panel   — the probability field p̂(x) = sigmoid(ŵ·[1,x]) as a
                background that starts flat (w = 0) and sharpens; the
                fitted boundary moving EVERY iteration; a yellow ring
                on the single sample whose gradient produced the step;
                a live classification report card (confusion counts,
                recall/specificity, accuracy, precision, F1, NLL);
  below       — the NLL gap descending into the noise ball with the
                Polyak line under it, and the test error settling on
                the Bayes floor while the boundary keeps wobbling
                (the direction/magnitude asymmetry).

Frames are log-spaced: literally one iteration each at the start,
stretched over the noise-ball tail. Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from common import (W_TRUE, bayes_error, make_data, nll, predict,
                    sigmoid)
from logistic_newton import train as newton_train
from logistic_sgd import train as sgd_train

# ----------------------------------------------------------------------
# The ingredients: trajectory, optimum, floor
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = make_data()
print("recording the SGD trajectory (quiet run) ...")
w_polyak, traj, order = sgd_train(X_train, y_train, X_test, y_test,
                                  verbose=False, record=True)
w_opt = newton_train(X_train, y_train, X_test, y_test, verbose=False)
FLOOR = bayes_error(X_test)
BEST = nll(w_opt, X_train, y_train)

snaps = [np.zeros(3)] + traj
N_SGD = len(traj)
FRAME_STEPS = np.r_[0, np.unique(np.geomspace(1, N_SGD,
                                              160).astype(int))]

XR, YR = 4.6, 3.0
gx = np.linspace(-XR, XR, 230)
gy = np.linspace(-YR, YR, 150)
GX, GY = np.meshgrid(gx, gy)


def field(w_):
    return sigmoid(w_[0] + w_[1] * GX + w_[2] * GY)


def boundary(w_):
    if abs(w_[2]) < 1e-12:
        return gx, np.full_like(gx, np.nan)
    return gx, -(w_[0] + w_[1] * gx) / w_[2]


def confusion(w_):
    pred = predict(w_, X_test)
    t = y_test.astype(bool)
    p = pred.astype(bool)
    return (int(np.sum(p & t)), int(np.sum(p & ~t)),
            int(np.sum(~p & t)), int(np.sum(~p & ~t)))


conf_snaps = [confusion(snaps[s]) for s in FRAME_STEPS]
err_snaps = [(FP + FN) / len(y_test) for TP, FP, FN, TN in conf_snaps]
nll_snaps = [nll(snaps[s], X_train, y_train) for s in FRAME_STEPS]
polyak_gap = max(nll(w_polyak, X_train, y_train) - BEST, 1e-16)


def report(k):
    TP, FP, FN, TN = conf_snaps[k]
    n = TP + FP + FN + TN
    prec = TP / (TP + FP) if TP + FP else float("nan")
    rec = TP / (TP + FN) if TP + FN else float("nan")
    spec = TN / (TN + FP) if TN + FP else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and prec + rec else float("nan"))
    return (f"{'':8s}{'pred +':>8s}{'pred -':>8s}\n"
            f"{'true +':8s}{TP:8d}{FN:8d}   recall {rec:5.3f}\n"
            f"{'true -':8s}{FP:8d}{TN:8d}   spec.  {spec:5.3f}\n\n"
            f"accuracy {(TP + TN) / n:5.3f}  precision {prec:5.3f}\n"
            f"F1       {f1:5.3f}  NLL       {nll_snaps[k]:.4f}")


# ----------------------------------------------------------------------
# The window: classifier big, diagnostics below
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(11.5, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.9, 1],
                      hspace=.28, wspace=.24)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])
fig.suptitle("SGD, live: one sample per iteration pushes the boundary",
             fontsize=12)

sub = slice(0, 1500)
im = axA.imshow(field(snaps[0]), extent=(-XR, XR, -YR, YR),
                origin="lower", cmap="coolwarm", vmin=0, vmax=1,
                alpha=.45, zorder=0)
axA.scatter(X_train[sub, 1], X_train[sub, 2], c=y_train[sub],
            cmap="coolwarm", s=8, alpha=.55, zorder=1)
axA.plot(*boundary(W_TRUE), ls=":", color="#0b0b0b", lw=1.6,
         label="true boundary", zorder=2)
axA.plot(*boundary(w_opt), ls="--", color="#eb6834", lw=1.3,
         label="optimum (Newton)", zorder=2)
line_sgd, = axA.plot([], [], color="#1baf7a", lw=2.6,
                     label="SGD boundary", zorder=3)
ring, = axA.plot([], [], "o", mfc="none", mec="#f0b400", ms=14, mew=2.8,
                 label="the sample that just pushed", zorder=4)
titleA = axA.set_title("")
axA.set(xlim=(-XR, XR), ylim=(-YR, YR), xlabel="x1", ylabel="x2")
axA.set_aspect("equal")
axA.legend(frameon=False, fontsize=8, loc="upper left")
report_txt = axA.text(.99, .02, "", transform=axA.transAxes,
                      family="monospace", fontsize=8.5, ha="right",
                      va="bottom", zorder=5,
                      bbox=dict(facecolor="white", alpha=.82,
                                edgecolor="#999", boxstyle="round"))

curve_nll, = axB.plot([], [], color="#1baf7a", lw=1.5,
                      label="SGD iterate")
axB.axhline(polyak_gap, color="#2a78d6", ls="--", lw=1.2,
            label=f"Polyak average {polyak_gap:.1e}")
axB.set_xscale("log")
axB.set_yscale("log")
axB.set(xlim=(1, N_SGD), ylim=(polyak_gap / 10, 1),
        xlabel="SGD step (log)", ylabel="NLL − best (log)",
        title="descent, then the noise ball; averaging beats it")
axB.legend(frameon=False, fontsize=8)

axC.axhline(FLOOR, color="#0b0b0b", ls="--", lw=1.2,
            label=f"Bayes floor {FLOOR:.4f}")
curve_err, = axC.plot([], [], color="#2a78d6", lw=1.5,
                      label="test error")
axC.set_xscale("log")
axC.set(xlim=(1, N_SGD), ylim=(FLOOR - .01, .52),
        xlabel="SGD step (log)", ylabel="test error",
        title="the wobble never reaches the error")
axC.legend(frameon=False, fontsize=8)

for ax in (axA, axB, axC):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

gap_snaps = np.maximum(np.array(nll_snaps) - BEST, 1e-16)
N_TRAIN = len(y_train)


def update(k):
    s = FRAME_STEPS[k]
    w_ = snaps[s]
    im.set_data(field(w_))
    line_sgd.set_data(*boundary(w_))
    if s >= 1:
        i = order[s - 1]
        ring.set_data([X_train[i, 1]], [X_train[i, 2]])
    shown = FRAME_STEPS[1:k + 1]
    curve_nll.set_data(shown, gap_snaps[1:k + 1])
    curve_err.set_data(shown, err_snaps[1:k + 1])
    epoch = (s - 1) // N_TRAIN + 1 if s else 0
    titleA.set_text(f"step {s:>6,d} (epoch {epoch})")
    report_txt.set_text(report(k))
    return im, line_sgd, ring, curve_nll, curve_err, titleA, report_txt


ani = FuncAnimation(fig, update, frames=len(FRAME_STEPS), interval=80,
                    blit=False, repeat=True)   # keep a ref!
plt.show()
