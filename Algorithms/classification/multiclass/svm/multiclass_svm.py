r"""Multiclass SVM — Crammer-Singer vs one-vs-rest, raced and verified.

Two ways to lift the SVM to K classes:

ONE-VS-REST (the reduction): train K independent binary soft-margin
SVMs, class k against everyone else; predict argmax of the K scores.
Simple, parallel, but the K problems never coordinate and the scores
are not trained to be comparable.

CRAMMER-SINGER (the joint objective): one problem over W in R^{K x d},

    min  lambda/2 ||W||_F^2
         + (1/n) sum_i max(0, 1 + max_{k != y_i} W_k.x_i - W_{y_i}.x_i)

- the margin is between the TRUE class score and the BEST WRONG one,
so the K rows are optimized against each other directly (the
multiclass hinge of the PA folder, now batch + regularized).

Verified below: (1) the hand subgradient of the CS loss (promote the
true row, demote the best-wrong row on violation, plus lambda W)
matches torch autograd at a random W to machine precision; (2) the
objective decreases under subgradient descent; (3) both methods
trained on identical data with logged epochs, sklearn reports, and
the joint objective at least matches the reduction.

Run me with F5. Derivations: multiclass-svm.pdf.
"""

import numpy as np
import torch
from sklearn.metrics import classification_report

SEED, K, D, LAM = 7, 3, 2, 1e-3
rng = np.random.default_rng(SEED)
CENTERS = np.array([[2.0, 0.0], [-1.2, 1.8], [-1.2, -1.8]])
N = 1_500
X = np.vstack([rng.normal(c, 0.85, size=(N // K, D)) for c in CENTERS])
y = np.repeat(np.arange(K), N // K)
perm = rng.permutation(N)
Xb = np.column_stack([np.ones(N), X])[perm]
y = y[perm]
Xtr, ytr, Xte, yte = Xb[:1100], y[:1100], Xb[1100:], y[1100:]

def cs_loss_grad(W, Xs, ys):
    """Crammer-Singer objective and its (sub)gradient, by hand."""
    S = Xs @ W.T
    Sy = S[np.arange(len(ys)), ys]
    S_masked = S.copy()
    S_masked[np.arange(len(ys)), ys] = -np.inf
    r = S_masked.argmax(axis=1)
    viol = 1 + S_masked[np.arange(len(ys)), r] - Sy
    active = viol > 0
    loss = LAM / 2 * np.sum(W**2) + np.mean(np.maximum(viol, 0))
    G = LAM * W.copy()
    for i in np.flatnonzero(active):          # promote y, demote r
        G[ys[i]] -= Xs[i] / len(ys)
        G[r[i]] += Xs[i] / len(ys)
    return loss, G

# ---- (1) hand subgradient == autograd at a random W ------------------
W0 = rng.normal(0, .3, size=(K, D + 1))
loss_h, G_h = cs_loss_grad(W0, Xtr[:200], ytr[:200])
Wt = torch.tensor(W0, requires_grad=True)
Xt, yt = torch.from_numpy(Xtr[:200]), torch.from_numpy(ytr[:200])
S = Xt @ Wt.T
Sy = S[torch.arange(200), yt]
S2 = S.clone()
S2[torch.arange(200), yt] = -1e18
viol = torch.clamp(1 + S2.max(dim=1).values - Sy, min=0)
L = LAM / 2 * (Wt**2).sum() + viol.mean()
L.backward()
gap = np.abs(G_h - Wt.grad.numpy()).max()
print(f"hand CS subgradient vs autograd: max gap = {gap:.2e}")
assert abs(loss_h - L.item()) < 1e-12 and gap < 1e-12

# ---- (2)+(3) train both, logged --------------------------------------
print("\nCrammer-Singer, subgradient descent:")
W = np.zeros((K, D + 1))
losses = []
for ep in range(1, 31):
    loss, G = cs_loss_grad(W, Xtr, ytr)
    W -= 0.5 * G
    losses.append(loss)
    if ep % 6 == 0:
        acc = np.mean((Xte @ W.T).argmax(1) == yte)
        print(f"   epoch {ep:>2}  loss {loss:.4f}  test acc {acc:.4f}")
assert losses[-1] < losses[0], "the objective must decrease"

print("one-vs-rest, K binary hinge SVMs:")
W_ovr = np.zeros((K, D + 1))
for k in range(K):
    w, yk = np.zeros(D + 1), np.where(ytr == k, 1.0, -1.0)
    for ep in range(30):
        marg = yk * (Xtr @ w)
        g = LAM * w - (yk[marg < 1, None] * Xtr[marg < 1]).sum(0) \
            / len(ytr)
        w -= 0.5 * g
    W_ovr[k] = w
acc_ovr = np.mean((Xte @ W_ovr.T).argmax(1) == yte)
acc_cs = np.mean((Xte @ W.T).argmax(1) == yte)
print(f"   one-vs-rest test acc {acc_ovr:.4f}")
print(f"\nCS {acc_cs:.4f} vs OvR {acc_ovr:.4f}")
assert acc_cs >= acc_ovr - 0.01, "the joint objective should match"

pred = (Xte @ W.T).argmax(1)
print(classification_report(yte, pred, digits=3))
assert acc_cs > 0.93
print("all claims verified: OK")
