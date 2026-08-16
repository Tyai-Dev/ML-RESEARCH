r"""MNIST, all ten digits — softmax regression, the floor to beat.

The ML-style pipeline, each stage in plain sight:

  1. LOAD      60k train images -> 55k train / 5k val (model
               selection) / 10k test (touched ONCE, at the end).
  2. LOOK      a grid of examples + the class balance — never train
               on data you haven't looked at.
  3. TRAIN     the generic engine (training.py): fit(model, loss_fn,
               ...) — model and loss are ARGUMENTS, so upgrading the
               model (models.py) or racing losses is a one-line
               change.
  4. EXAMINE   the full post-mortem: sklearn classification report
               over 10 digits, the confusion matrix and its worst
               pairs, confidence statistics (does it know when it
               doesn't know?), the 10 learned templates as images,
               and the most confident mistakes.

The model: SoftmaxRegression — one linear layer, the conditional
multinoulli of softmax-regression/ in torch clothes, 7,850 parameters.
Expect ~92-93%: the linear floor. Every future model in models.py
must beat it on the SAME splits and report.

Run me with F5. Theory: softmax-regression.tex; metrics:
Theory/evaluation.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from mnist import load_mnist                      # noqa: E402

from models import SoftmaxRegression              # noqa: E402
from training import SEED, evaluate, fit          # noqa: E402

# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

X, y, X_test_np, y_test_np = load_mnist()
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
val_idx, train_idx = perm[:5_000], perm[5_000:]

to_t = lambda a, dt: torch.tensor(a, dtype=dt, device=DEVICE)
X_train = to_t(X[train_idx], torch.float32)
y_train = to_t(y[train_idx], torch.long)
X_val = to_t(X[val_idx], torch.float32)
y_val = to_t(y[val_idx], torch.long)
X_test = to_t(X_test_np, torch.float32)
y_test = to_t(y_test_np, torch.long)
print(f"1. LOAD   train {len(X_train):,} / val {len(X_val):,} / "
      f"test {len(X_test):,}   (28x28 grayscale, flattened to 784)")

# ----------------------------------------------------------------------
# 2. LOOK
# ----------------------------------------------------------------------
counts = np.bincount(y[train_idx])
print("2. LOOK   class balance (train):")
print("   digit  " + " ".join(f"{d:>5}" for d in range(10)))
print("   count  " + " ".join(f"{c:>5}" for c in counts))

fig_ex, axes = plt.subplots(2, 8, figsize=(11, 3.2))
for ax, i in zip(axes.ravel(), rng.choice(train_idx, 16, replace=False)):
    ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
    ax.set_title(int(y[i]), fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
fig_ex.suptitle("2. LOOK — sixteen training examples", fontsize=11)
fig_ex.tight_layout()

# ----------------------------------------------------------------------
# 3. TRAIN
# ----------------------------------------------------------------------
print("\n3. TRAIN")
model = SoftmaxRegression().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()                  # the multinoulli NLL
history = fit(model, loss_fn, X_train, y_train, X_val, y_val,
              epochs=12, batch=128, lr=1e-3)

# ----------------------------------------------------------------------
# 4. EXAMINE — the test set, touched once
# ----------------------------------------------------------------------
print("\n4. EXAMINE (test set, first and only touch)")
test_loss, test_acc = evaluate(model, loss_fn, X_test, y_test)
print(f"test loss {test_loss:.4f}   test accuracy {test_acc:.4f}\n")

with torch.no_grad():
    P = torch.softmax(model(X_test), dim=1).cpu().numpy()
y_pred = P.argmax(axis=1)
y_true = y_test_np

print(classification_report(y_true, y_pred, digits=3,
                            target_names=[str(d) for d in range(10)]))

# the confusion matrix, and where the model actually bleeds
C = confusion_matrix(y_true, y_pred)
print("confusion matrix (rows = truth, columns = prediction):")
print("      " + "".join(f"{d:>6}" for d in range(10)))
for d in range(10):
    print(f"  {d}  " + "".join(f"{C[d, j]:>6}" for j in range(10)))
off = C - np.diag(np.diag(C))
pairs = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1],
                                   C.shape))[0][:5]
print("\nworst confusions:")
for t, p in pairs:
    print(f"   true {t} read as {p}: {C[t, p]:>3} times "
          f"({C[t, p] / C[t].sum():.1%} of all {t}s)")

# does it know when it doesn't know?
conf = P.max(axis=1)
right = y_pred == y_true
print(f"\nconfidence: mean max-prob when correct {conf[right].mean():.3f}"
      f" vs when wrong {conf[~right].mean():.3f}")
bins = [0.5, 0.7, 0.9, 0.99, 1.001]
print("calibration (predicted confidence vs actual accuracy):")
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf >= lo) & (conf < hi)
    if m.sum():
        print(f"   conf [{lo:.2f},{hi:.2f}): predicted "
              f"~{conf[m].mean():.3f}, actual {right[m].mean():.3f} "
              f"({m.sum():,} samples)")

assert test_acc > 0.90, "the linear floor should clear 90%"

# ----------------------------------------------------------------------
# Figures: curves, confusion, templates, mistakes
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
ep = np.arange(1, len(history["train_loss"]) + 1)
ax1.plot(ep, history["train_loss"], "o-", color="#2a78d6", ms=3,
         label="train loss")
ax1.plot(ep, history["val_loss"], "o-", color="#eb6834", ms=3,
         label="val loss")
ax1.set(xlabel="epoch", ylabel="cross-entropy", title="loss curves")
ax1.legend(frameon=False, fontsize=8)
ax2.plot(ep, history["val_acc"], "o-", color="#3d9b35", ms=3)
ax2.set(xlabel="epoch", ylabel="val accuracy",
        title="validation accuracy")
for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()

fig_c, axc = plt.subplots(figsize=(5.6, 5))
imc = axc.imshow(np.log1p(off), cmap="inferno")
axc.set(xticks=range(10), yticks=range(10),
        xlabel="predicted", ylabel="true",
        title="confusions (diagonal removed, log scale)")
for t, p in pairs:
    axc.text(p, t, C[t, p], color="white", ha="center", va="center",
             fontsize=8)
fig_c.colorbar(imc, ax=axc, shrink=.8)
fig_c.tight_layout()

fig_t, axes_t = plt.subplots(2, 5, figsize=(10.5, 4.4))
T = model.templates()
lim = np.percentile(np.abs(T), 99)
for d, ax in enumerate(axes_t.ravel()):
    ax.imshow(T[d], cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_title(str(d), fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
fig_t.suptitle("the 10 learned templates — red pixels vote FOR the "
               "digit, blue AGAINST", fontsize=11)
fig_t.tight_layout()

wrong_idx = np.flatnonzero(~right)
worst = wrong_idx[np.argsort(conf[wrong_idx])[::-1][:16]]
fig_w, axes_w = plt.subplots(2, 8, figsize=(11, 3.4))
for ax, i in zip(axes_w.ravel(), worst):
    ax.imshow(X_test_np[i].reshape(28, 28), cmap="gray_r")
    ax.set_title(f"{y_true[i]} read {y_pred[i]} "
                 f"p={conf[i]:.2f}", fontsize=7.5)
    ax.set_xticks([]), ax.set_yticks([])
fig_w.suptitle("the most confident mistakes", fontsize=11)
fig_w.tight_layout()

plt.show()
