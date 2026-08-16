r"""CIFAR-10 — real objects, real color, and the wall MLPs hit.

The pipeline and engine are the classification defaults
(Experiments/common): LOAD -> LOOK -> TRAIN -> EXAMINE with the delta
log and test_report. What changed is the world: 32x32 RGB photographs
of actual objects (3,072 input dims), where pixel-position rigidity
finally becomes fatal - a cat in the corner shares almost no pixels
with a cat in the center. Expect the linear floor near 40% and the
MLP near 50-55%: the gap to ~95% (modern CNNs) is the point. This
experiment sets the floors the convolutional zoo member must clear.

Run me with F5. (Downloads CIFAR-10 once, ~170MB, to
datasets/cifar10/.)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "datasets"))
sys.path.insert(0, str(ROOT / "Experiments" / "common"))
from cifar10 import CIFAR10_CLASSES, load_cifar10  # noqa: E402
from models import MLP, SoftmaxRegression          # noqa: E402
from training import SEED, fit, test_report        # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
NAMES = CIFAR10_CLASSES

# 1. LOAD ---------------------------------------------------------------
X, y, X_test_np, y_test_np = load_cifar10()
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
print(f"1. LOAD   CIFAR-10: train {len(X_train):,} / val "
      f"{len(X_val):,} / test {len(X_test):,}   (32x32 RGB = 3,072)")


def as_img(v):
    return v.reshape(3, 32, 32).transpose(1, 2, 0)


# 2. LOOK ---------------------------------------------------------------
print("2. LOOK   classes: " + " ".join(NAMES))
fig_ex, axes = plt.subplots(2, 8, figsize=(11, 3.2))
for ax, i in zip(axes.ravel(), rng.choice(train_idx, 16, replace=False)):
    ax.imshow(as_img(X[i]))
    ax.set_title(NAMES[y[i]], fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
fig_ex.suptitle("2. LOOK — sixteen CIFAR-10 examples", fontsize=11)
fig_ex.tight_layout()

# 3. TRAIN --------------------------------------------------------------
print("\n3. TRAIN — the linear floor first")
lin = SoftmaxRegression(d_in=3072).to(DEVICE)
fit(lin, nn.CrossEntropyLoss(), X_train, y_train, X_val, y_val,
    epochs=12, log_style="quiet")

print("\n3b. TRAIN — the MLP")
model = MLP(d_in=3072, hidden=(1024, 512, 256)).to(DEVICE)
history = fit(model, nn.CrossEntropyLoss(), X_train, y_train,
              X_val, y_val, epochs=30, batch=128, lr=1e-3)

# 4. EXAMINE ------------------------------------------------------------
print("\n4. EXAMINE (test set, first and only touch)")
print("linear floor:")
lin_loss, lin_acc = test_report(lin, nn.CrossEntropyLoss(),
                                X_test, y_test, NAMES,
                                top_confusions=3)
print("\nMLP:")
test_loss, test_acc = test_report(model, nn.CrossEntropyLoss(),
                                  X_test, y_test, NAMES)
print(f"\nladder: linear {lin_acc:.2%} -> MLP {test_acc:.2%} "
      f"(modern CNNs ~95%: that gap is convolution)")
assert lin_acc > 0.30 and test_acc > lin_acc

# figures: hardest mistakes ---------------------------------------------
with torch.no_grad():
    P = torch.softmax(model(X_test), dim=1).cpu().numpy()
y_pred, conf = P.argmax(axis=1), P.max(axis=1)
wrong_idx = np.flatnonzero(y_pred != y_test_np)
worst = wrong_idx[np.argsort(conf[wrong_idx])[::-1][:16]]
fig_w, axes_w = plt.subplots(2, 8, figsize=(11, 3.6))
for ax, i in zip(axes_w.ravel(), worst):
    ax.imshow(as_img(X_test_np[i]))
    ax.set_title(f"{NAMES[y_test_np[i]]}\nread {NAMES[y_pred[i]]} "
                 f"p={conf[i]:.2f}", fontsize=6.5)
    ax.set_xticks([]), ax.set_yticks([])
fig_w.suptitle("the most confident mistakes", fontsize=11)
fig_w.tight_layout()
plt.show()
