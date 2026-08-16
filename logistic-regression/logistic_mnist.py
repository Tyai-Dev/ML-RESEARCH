r"""Logistic regression on real images — MNIST, 3 vs 5.

Same model, same three training loops (imported UNCHANGED from
logistic_sgd / logistic_gd / logistic_newton — they were written
generically and this file is the proof), new world:

    x  = 785 numbers  (a 28x28 image, flattened, + intercept)
    w  = 785 numbers  = ITSELF AN IMAGE — reshape ŵ to 28x28 and you
         see what the classifier looks for: red pixels vote "5",
         blue pixels vote "3". Ink where the digits differ, silence
         where they agree.

What real data breaks that synthetic data never could:

  NO BAYES FLOOR. We generated the synthetic p(x), so we knew the
  unbeatable error. Nobody knows p(image of a 3); the held-out test
  set is the only truth. The report card loses its answer key.

  DEAD FEATURES. Border pixels are zero in EVERY image, so the
  Hessian has zero rows/columns — plain Newton crashes on a singular
  solve. Cure: damping (H + lambda I), which is exactly Newton on the
  L2-regularized loss. Watched below.

  (NEAR-)SEPARABILITY IS REAL. 3-vs-5 with 785 free parameters is
  rich enough that the training NLL keeps sliding toward 0 while
  ||w|| grows — logistic_gd.py's "pathological" demo turns out to be
  the NORMAL situation for expressive models; finite steps act as
  early stopping. Watch train NLL vs test accuracy decouple in the
  logs: the generalization gap, live.

The animation: SGD's weight vector AS AN IMAGE, forming from noise
into a 3-vs-5 template one sample at a time, beside the test error.

Run me with F5. (Downloads MNIST once, ~12MB, to datasets/mnist/.)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from mnist import load_binary                     # noqa: E402

import logistic_sgd                               # noqa: E402
from common import evaluate, nll, predict, sigmoid  # noqa: E402
from logistic_gd import train as gd_train         # noqa: E402
from logistic_newton import train as newton_train  # noqa: E402

D0, D1 = 3, 5
NAMES = (f"digit {D0}", f"digit {D1}")

# ----------------------------------------------------------------------
# The data: real images, intercept column prepended
# ----------------------------------------------------------------------
X_train, y_train, X_test, y_test = load_binary(D0, D1)
X_train = np.column_stack([np.ones(len(X_train)), X_train])
X_test = np.column_stack([np.ones(len(X_test)), X_test])
dead = int(np.sum(X_train.max(axis=0) == 0))
print(f"MNIST {D0} vs {D1}: train {X_train.shape[0]:,} images, "
      f"test {X_test.shape[0]:,}, d = {X_train.shape[1]} "
      f"({dead} dead pixels — never inked in any training image)")

# ----------------------------------------------------------------------
# (1) SGD — the identical loop from logistic_sgd.py, now on images
# ----------------------------------------------------------------------
logistic_sgd.PRINT_EVERY = 3_000              # ~1 line per 1/4 epoch
print()
w_sgd, traj, _ = logistic_sgd.train(X_train, y_train, X_test, y_test,
                                    record=True)
evaluate(w_sgd, X_test, y_test, "SGD (Polyak)", synthetic=False,
         target_names=NAMES)

# ----------------------------------------------------------------------
# (2) GD — watch train NLL slide while test accuracy stalls:
#     the generalization gap on (near-)separable real data
# ----------------------------------------------------------------------
print()
w_gd = gd_train(X_train, y_train, X_test, y_test)
print(f"||w_gd|| = {np.linalg.norm(w_gd):.1f}  and still growing — "
      f"finite steps are early stopping here")
evaluate(w_gd, X_test, y_test, "GD (3000 steps)", synthetic=False,
         target_names=NAMES)

# ----------------------------------------------------------------------
# (3) Newton — singular without damping, fine with it
# ----------------------------------------------------------------------
print()
try:
    newton_train(X_train, y_train, X_test, y_test, verbose=False,
                 steps=1)
    print("undamped Newton survived?! (unexpected)")
except np.linalg.LinAlgError:
    print(f"undamped Newton: LinAlgError — singular Hessian, exactly "
          f"as the {dead} dead pixels predict")
w_nt = newton_train(X_train, y_train, X_test, y_test, damping=1e-4,
                    steps=8)
evaluate(w_nt, X_test, y_test, "Newton/IRLS (damped 1e-4)",
         synthetic=False, target_names=NAMES)

for name, w_ in [("SGD", w_sgd), ("GD", w_gd), ("Newton", w_nt)]:
    err = float(np.mean(predict(w_, X_test) != y_test))
    assert err < 0.05, f"{name} should be well under 5% error on 3v5"
print("\nall three solvers under 5% test error on real images: OK")

# ----------------------------------------------------------------------
# Picture: the classifier IS an image; and where it still fails
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12.5, 6.8))
gs = fig.add_gridspec(2, 8)
axW = fig.add_subplot(gs[:, 0:4])
# SGD's Polyak weights make the readable template (Newton's push
# toward separation is spiky — single pixels at +-9 wash the image
# out); robust color limits for the same reason
lim = np.percentile(np.abs(w_sgd[1:]), 99)
imW = axW.imshow(w_sgd[1:].reshape(28, 28), cmap="coolwarm",
                 vmin=-lim, vmax=lim)
axW.set_title(f"ŵ as an image: red votes '{D1}', blue votes '{D0}'")
axW.set_xticks([]), axW.set_yticks([])
fig.colorbar(imW, ax=axW, shrink=.8)

p_test = sigmoid(X_test @ w_sgd)
wrong = np.flatnonzero((p_test > 0.5) != y_test.astype(bool))
worst = wrong[np.argsort(np.abs(p_test[wrong] - 0.5))[::-1][:8]]
for j, idx in enumerate(worst):
    ax = fig.add_subplot(gs[j // 4, 4 + j % 4])
    ax.imshow(X_test[idx, 1:].reshape(28, 28), cmap="gray_r")
    truth = D1 if y_test[idx] else D0
    ax.set_title(f"is {truth}, P({D1})={p_test[idx]:.2f}", fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
fig.suptitle(f"the linear classifier for {D0} vs {D1} — and its most "
             f"confident mistakes", fontsize=11)
fig.tight_layout()

# ----------------------------------------------------------------------
# Animation: the template forming from noise, one sample at a time
# ----------------------------------------------------------------------
snaps = [np.zeros(X_train.shape[1])] + traj
N_SGD = len(traj)
FRAME_STEPS = np.r_[0, np.unique(np.geomspace(1, N_SGD,
                                              140).astype(int))]
err_snaps = [float(np.mean(predict(snaps[s], X_test) != y_test))
             for s in FRAME_STEPS]

figA = plt.figure(figsize=(10.5, 5.2))
gsA = figA.add_gridspec(1, 2, width_ratios=[1, 1.1])
axL, axR = figA.add_subplot(gsA[0]), figA.add_subplot(gsA[1])
figA.suptitle("SGD carving the 3-vs-5 template, one image at a time",
              fontsize=11)

vmax = np.abs(traj[-1][1:]).max()
imA = axL.imshow(snaps[0][1:].reshape(28, 28), cmap="coolwarm",
                 vmin=-vmax, vmax=vmax)
titleL = axL.set_title("")
axL.set_xticks([]), axL.set_yticks([])

curve, = axR.plot([], [], color="#2a78d6", lw=1.5, label="test error")
axR.set_xscale("log")
axR.set(xlim=(1, N_SGD), ylim=(0, 0.55),
        xlabel="SGD step (log)", ylabel="test error",
        title="error vs samples seen")
axR.legend(frameon=False, fontsize=8)
axR.grid(alpha=.3)
for side in ("top", "right"):
    axR.spines[side].set_visible(False)
figA.tight_layout()


def update(k):
    s = FRAME_STEPS[k]
    imA.set_data(snaps[s][1:].reshape(28, 28))
    shown = FRAME_STEPS[1:k + 1]
    curve.set_data(shown, err_snaps[1:k + 1])
    titleL.set_text(f"step {s:>6,d}   test err {err_snaps[k]:.4f}")
    return imA, curve, titleL


ani = FuncAnimation(figA, update, frames=len(FRAME_STEPS), interval=80,
                    blit=False, repeat=True)   # keep a ref!
plt.show()
