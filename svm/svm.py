r"""Support Vector Machine: maximize the margin, keep only the supporters.

The batch counterpart of the online margin story. Soft-margin SVM in its
unconstrained (hinge) form:

    min_w  J(w) = (lam/2) ||w||^2 + (1/n) sum_i max(0, 1 - y_i w.x_i),

which trades margin width (small ||w||) against margin violations (the
hinge). We optimize it with Pegasos (Shalev-Shwartz et al.): SGD on J
with step 1/(lam t) — each step shrinks w (the regularizer) and, if the
sampled example violates the margin, adds y x (the hinge subgradient;
the loss is not differentiable at the kink, so SUBgradients replace
gradients — see svm.tex).

What the script shows:
(1) Pegasos reaches the optimum of J: verified against a long full-batch
    subgradient run AND by random-perturbation local optimality.
(2) SUPPORT VECTORS: by the KKT conditions (Theory/optimization), the
    solution is a combination of only the points with y w.x <= 1 (on or
    inside the margin). Empirically: deleting all NON-support points —
    keeping the original 1/n weighting so the loss/regularizer balance
    is untouched — leaves the optimum essentially unchanged (their hinge
    terms are locally zero); deleting the support vectors moves it.
(3) SVM vs logistic regression on the same data: similar boundaries here,
    different philosophies (margin geometry vs conditional likelihood).

Run me with F5. Companion derivations: svm.tex.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Data: two overlapping Gaussian blobs (not separable), y in {-1, +1}
# ----------------------------------------------------------------------
N = 1_500
X = np.vstack([rng.normal([1.2, 1.2], 0.9, size=(N // 2, 2)),
               rng.normal([-1.2, -1.2], 0.9, size=(N // 2, 2))])
y = np.concatenate([np.ones(N // 2), -np.ones(N // 2)])
perm = rng.permutation(N)
X, y = X[perm], y[perm]
Xb = np.column_stack([np.ones(N), X])              # add intercept
LAM = 0.01


def objective(w):
    margins = y * (Xb @ w)
    return 0.5 * LAM * w @ w + np.mean(np.maximum(0, 1 - margins))


# ----------------------------------------------------------------------
# Pegasos: SGD on the regularized hinge with step 1/(lam t)
# ----------------------------------------------------------------------
def pegasos(Xb, y, lam, T=300_000):
    w = np.zeros(Xb.shape[1])
    for t in range(1, T + 1):
        i = rng.integers(len(y))
        eta = 1.0 / (lam * t)
        if y[i] * (Xb[i] @ w) < 1:                 # margin violated:
            w = (1 - eta * lam) * w + eta * y[i] * Xb[i]   # shrink + correct
        else:                                      # margin satisfied:
            w = (1 - eta * lam) * w               # shrink only
    return w


# full-batch subgradient reference (slow but sure). n_norm lets us refit
# on a subset while keeping the ORIGINAL 1/n weighting — needed for the
# support-vector invariance test below.
def batch_subgradient(Xb, y, lam, T=20_000, n_norm=None):
    n_norm = n_norm or len(y)
    w = np.zeros(Xb.shape[1])
    for t in range(1, T + 1):
        viol = y * (Xb @ w) < 1
        grad = lam * w - (y[viol] @ Xb[viol]) / n_norm
        w = w - (1.0 / (lam * t)) * grad
    return w


w_peg = pegasos(Xb, y, LAM)
w_ref = batch_subgradient(Xb, y, LAM)

print("--- Pegasos vs full-batch subgradient reference ---")
print(f"objective (Pegasos)  : {objective(w_peg):.5f}")
print(f"objective (reference): {objective(w_ref):.5f}")
assert abs(objective(w_peg) - objective(w_ref)) < 0.01

# local optimality: random perturbations never improve the objective
J0 = objective(w_ref)
improvements = sum(objective(w_ref + rng.normal(0, .02, size=3)) < J0 - 1e-9
                   for _ in range(500))
print(f"perturbations improving J: {improvements}/500")
assert improvements == 0

w = w_ref

# ----------------------------------------------------------------------
# Support vectors: only the points with y w.x <= 1 matter
# ----------------------------------------------------------------------
margins = y * (Xb @ w)
support = margins <= 1.0 + 1e-6
print(f"\nsupport vectors: {support.sum()} of {N} points "
      f"({support.mean():.0%}) carry the solution")

# The invariance the KKT structure promises: points with margin > 1 have
# ZERO hinge (and zero subgradient) in a neighborhood of w*, so removing
# them — while keeping the original 1/n weighting so the loss/regularizer
# balance is untouched — leaves the optimum EXACTLY unchanged. Removing
# the support vectors removes active terms and genuinely moves it.
w_keep_sv = batch_subgradient(Xb[support], y[support], LAM, T=40_000, n_norm=N)
w_drop_sv = batch_subgradient(Xb[~support], y[~support], LAM, T=40_000, n_norm=N)

dist_keep = np.linalg.norm(w_keep_sv - w)
dist_drop = np.linalg.norm(w_drop_sv - w)
print(f"||w after dropping NON-support pts - w*|| : {dist_keep:.4f}  (invariant)")
print(f"||w after dropping support vectors - w*|| : {dist_drop:.4f}  (moved)")
assert dist_keep < 0.05
assert dist_drop > 5 * dist_keep

# ----------------------------------------------------------------------
# Compare with logistic regression (Newton) on the same data
# ----------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

y01 = (y + 1) / 2
w_log = np.zeros(3)
for _ in range(20):
    p = sigmoid(Xb @ w_log)
    g = Xb.T @ (p - y01) / N + 1e-6 * w_log
    H = (Xb.T * (p * (1 - p))) @ Xb / N + 1e-6 * np.eye(3)
    w_log = w_log - np.linalg.solve(H, g)

err_svm = np.mean(np.sign(Xb @ w) != y)
err_log = np.mean(np.sign(Xb @ w_log) != y)
print(f"\ntrain error — SVM: {err_svm:.4f},  logistic: {err_log:.4f}")

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

ax1.scatter(X[~support][:, 0], X[~support][:, 1], c=y[~support],
            cmap="coolwarm", s=5, alpha=.25)
ax1.scatter(X[support][:, 0], X[support][:, 1], c=y[support],
            cmap="coolwarm", s=14, edgecolors="#0b0b0b", linewidths=.4,
            label="support vectors")
g = np.linspace(-4, 4, 2)
for offset, style in [(0, "-"), (1, "--"), (-1, "--")]:
    ax1.plot(g, -(w[0] + w[1] * g - offset) / w[2], style,
             color="#1baf7a", lw=1.5 if offset == 0 else 1)
ax1.set(title=f"soft margin: {support.sum()} support vectors carry it",
        xlim=(-4, 4), ylim=(-4, 4))
ax1.legend(frameon=False, fontsize=8)

ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=4, alpha=.3)
ax2.plot(g, -(w[0] + w[1] * g) / w[2], color="#1baf7a", lw=2, label="SVM")
ax2.plot(g, -(w_log[0] + w_log[1] * g) / w_log[2], color="#2a78d6", lw=2,
         ls="--", label="logistic")
ax2.set(title="margin geometry vs conditional likelihood",
        xlim=(-4, 4), ylim=(-4, 4))
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
