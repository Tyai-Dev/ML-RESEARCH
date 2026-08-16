r"""KMNIST, ten Kuzushiji hiragana — the linear floor on harder ground.

Same pipeline and same model as Experiments/mnist (models.py and
training.py are imported from there unchanged; upgrading the model
zoo upgrades both experiments at once). Only the dataset moved:
cursive Japanese characters (o, ki, su, tsu, na, ha, ma, ya, re, wo),
28x28 grayscale, 60k/10k - published as a deliberate MNIST drop-in
precisely because MNIST got too easy. Cursive strokes overlap heavily
in pixel space, so one rigid template per class - all a linear model
owns - should hurt: expect the floor well below MNIST's 92.7%.

Run me with F5. (Downloads KMNIST once, ~18MB, to datasets/kmnist/.)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "datasets"))
sys.path.insert(0, str(ROOT / "Experiments" / "mnist"))
from mnist import KMNIST_CLASSES, load_mnist      # noqa: E402
from models import SoftmaxRegression              # noqa: E402
from training import SEED, evaluate, fit          # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
NAMES = KMNIST_CLASSES

# 1. LOAD ---------------------------------------------------------------
X, y, X_test_np, y_test_np = load_mnist("kmnist")
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
val_idx, train_idx = perm[:5_000], perm[5_000:]
to_t = lambda a, dt: torch.tensor(a, dtype=dt, device=DEVICE)
X_train, y_train = to_t(X[train_idx], torch.float32), \
    to_t(y[train_idx], torch.long)
X_val, y_val = to_t(X[val_idx], torch.float32), \
    to_t(y[val_idx], torch.long)
X_test, y_test = to_t(X_test_np, torch.float32), \
    to_t(y_test_np, torch.long)
print(f"1. LOAD   KMNIST: train {len(X_train):,} / val {len(X_val):,}"
      f" / test {len(X_test):,}")

# 2. LOOK ---------------------------------------------------------------
print("2. LOOK   classes: " + " ".join(NAMES))
fig_ex, axes = plt.subplots(2, 8, figsize=(11, 3.2))
for ax, i in zip(axes.ravel(), rng.choice(train_idx, 16, replace=False)):
    ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
    ax.set_title(NAMES[y[i]], fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
fig_ex.suptitle("2. LOOK — sixteen Kuzushiji examples", fontsize=11)
fig_ex.tight_layout()

# 3. TRAIN --------------------------------------------------------------
print("\n3. TRAIN")
model = SoftmaxRegression().to(DEVICE)
history = fit(model, nn.CrossEntropyLoss(), X_train, y_train,
              X_val, y_val, epochs=12, batch=128, lr=1e-3)

# 4. EXAMINE ------------------------------------------------------------
print("\n4. EXAMINE (test set, first and only touch)")
test_loss, test_acc = evaluate(model, nn.CrossEntropyLoss(),
                               X_test, y_test)
print(f"test loss {test_loss:.4f}   test accuracy {test_acc:.4f}"
      f"   (MNIST linear floor was 0.9274)")
with torch.no_grad():
    P = torch.softmax(model(X_test), dim=1).cpu().numpy()
y_pred = P.argmax(axis=1)
print(classification_report(y_test_np, y_pred, digits=3,
                            target_names=NAMES))
C = confusion_matrix(y_test_np, y_pred)
off = C - np.diag(np.diag(C))
pairs = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1],
                                   C.shape))[0][:5]
print("worst confusions:")
for t, p in pairs:
    print(f"   true '{NAMES[t]}' read as '{NAMES[p]}': {C[t, p]:>3} "
          f"({C[t, p] / C[t].sum():.1%})")
conf = P.max(axis=1)
right = y_pred == y_test_np
print(f"confidence: correct {conf[right].mean():.3f} vs wrong "
      f"{conf[~right].mean():.3f}")
assert test_acc > 0.60, "even the linear floor should clear 60%"

# figures: templates + hardest mistakes ---------------------------------
fig_t, axes_t = plt.subplots(2, 5, figsize=(10.5, 4.4))
T = model.templates()
lim = np.percentile(np.abs(T), 99)
for d, ax in enumerate(axes_t.ravel()):
    ax.imshow(T[d], cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_title(NAMES[d], fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
fig_t.suptitle("the 10 learned templates — where one template per "
               "cursive class starts to fail", fontsize=11)
fig_t.tight_layout()

wrong_idx = np.flatnonzero(~right)
worst = wrong_idx[np.argsort(conf[wrong_idx])[::-1][:16]]
fig_w, axes_w = plt.subplots(2, 8, figsize=(11, 3.4))
for ax, i in zip(axes_w.ravel(), worst):
    ax.imshow(X_test_np[i].reshape(28, 28), cmap="gray_r")
    ax.set_title(f"{NAMES[y_test_np[i]]} read {NAMES[y_pred[i]]} "
                 f"p={conf[i]:.2f}", fontsize=7)
    ax.set_xticks([]), ax.set_yticks([])
fig_w.suptitle("the most confident mistakes", fontsize=11)
fig_w.tight_layout()
plt.show()
