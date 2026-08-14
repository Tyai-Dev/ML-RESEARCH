r"""Scaled dot-product attention — derived, computed by hand, verified.

Rung 2's model reads its context through a FIXED window: the weights
for "3 characters back" are separate parameters from "4 characters
back", and the window cannot grow without retraining. Attention removes
the rigidity: the model COMPUTES, per position, how much each previous
position matters — a data-dependent, differentiable dictionary lookup.

The forward pass (single head, sequence X in R^{T x d_model}):

    Q = X Wq        "what am I looking for?"      (T, d_h)
    K = X Wk        "what do I contain?"          (T, d_h)
    V = X Wv        "what do I offer if chosen?"  (T, d_h)

    S = Q K^T / sqrt(d_h)                          (T, T) scores
    S[i, j] = -inf  for j > i                      causal mask:
                                                   no peeking at the future
    A = softmax(S, per row)                        (T, T) attention weights
    O = A V                                        (T, d_h) outputs

Why the sqrt(d_h): if q, k have iid unit-variance entries, q . k has
variance d_h; dividing by sqrt(d_h) keeps scores O(1) so softmax
doesn't saturate at init (checked numerically below).

Row i of A is a probability distribution over positions <= i: position
i mixes the values of the positions it attends to. A is not a
parameter — it is RECOMPUTED from the data every forward pass. That is
the whole trick.

The backward pass, by the chain rule (derived in attention.tex):

    dV  = A^T dO
    dA  = dO V^T
    dS  = A * (dA - rowsum(dA * A))     softmax Jacobian, rows
                                        (masked entries: A=0 => dS=0)
    dQ  = dS K / sqrt(d_h)
    dK  = dS^T Q / sqrt(d_h)
    dWq = X^T dQ,  dWk = X^T dK,  dWv = X^T dV
    dX  = dQ Wq^T + dK Wk^T + dV Wv^T

This script computes BOTH passes by hand in numpy on a small example
(T=6, d_model=8, d_h=4) and asserts, against torch autograd in
float64: the outputs match, every gradient matches, and causality
holds (changing X at position t does not change any output before t).
Earning autograd, one mechanism at a time.

Run me with F5. Derivations: attention.tex.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED

T, D_MODEL, D_H = 6, 8, 4
rng = np.random.default_rng(SEED)


def attention_forward(X, Wq, Wk, Wv):
    """The forward pass, by hand. Returns output and intermediates."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(D_H)
    S = np.where(np.tril(np.ones((T, T))) == 1, S, -np.inf)   # causal
    S_shift = S - S.max(axis=1, keepdims=True)                # stable
    E = np.exp(S_shift)
    A = E / E.sum(axis=1, keepdims=True)
    O = A @ V
    return O, (Q, K, V, A)


def attention_backward(dO, X, Wq, Wk, Wv, Q, K, V, A):
    """The backward pass, by hand — the chain rule from the docstring,
    line for line."""
    dV = A.T @ dO
    dA = dO @ V.T
    dS = A * (dA - (dA * A).sum(axis=1, keepdims=True))       # softmax'
    dQ = dS @ K / np.sqrt(D_H)
    dK = dS.T @ Q / np.sqrt(D_H)
    dWq, dWk, dWv = X.T @ dQ, X.T @ dK, X.T @ dV
    dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
    return dX, dWq, dWk, dWv


# ----------------------------------------------------------------------
# The example, and the torch mirror
# ----------------------------------------------------------------------
X = rng.normal(size=(T, D_MODEL))
Wq, Wk, Wv = (rng.normal(size=(D_MODEL, D_H)) / np.sqrt(D_MODEL)
              for _ in range(3))
G = rng.normal(size=(T, D_H))          # loss = sum(O * G)  =>  dO = G

O_hand, (Q, K, V, A) = attention_forward(X, Wq, Wk, Wv)
dX_h, dWq_h, dWk_h, dWv_h = attention_backward(G, X, Wq, Wk, Wv,
                                               Q, K, V, A)

Xt = torch.tensor(X, requires_grad=True)
Wqt = torch.tensor(Wq, requires_grad=True)
Wkt = torch.tensor(Wk, requires_grad=True)
Wvt = torch.tensor(Wv, requires_grad=True)
Qt, Kt, Vt = Xt @ Wqt, Xt @ Wkt, Xt @ Wvt
St = Qt @ Kt.T / np.sqrt(D_H)
mask = torch.tril(torch.ones(T, T)) == 0
St = St.masked_fill(mask, float("-inf"))
At = torch.softmax(St, dim=1)
Ot = At @ Vt
(Ot * torch.tensor(G)).sum().backward()

print("hand vs autograd (float64):")
pairs = [("output O", O_hand, Ot.detach().numpy()),
         ("weights A", A, At.detach().numpy()),
         ("dX", dX_h, Xt.grad.numpy()),
         ("dWq", dWq_h, Wqt.grad.numpy()),
         ("dWk", dWk_h, Wkt.grad.numpy()),
         ("dWv", dWv_h, Wvt.grad.numpy())]
for name, hand, auto in pairs:
    gap = np.abs(hand - auto).max()
    print(f"  {name:<10} max gap {gap:.2e}")
    assert gap < 1e-12, f"{name} disagrees with autograd!"
print("attention forward AND backward, by hand == autograd: OK")

# ---- structural checks ------------------------------------------------
# rows of A are distributions supported on the past only
assert np.allclose(A.sum(axis=1), 1.0)
assert np.allclose(A[np.triu_indices(T, k=1)], 0.0), "future leaked!"

# causality through the whole computation: perturb the LAST position;
# outputs at earlier positions must not move at all
X2 = X.copy()
X2[-1] += 10.0
O2, _ = attention_forward(X2, Wq, Wk, Wv)
assert np.allclose(O2[:-1], O_hand[:-1]), "causality violated!"
assert not np.allclose(O2[-1], O_hand[-1])
print("causality: perturbing position t changes outputs at t only: OK")

# the sqrt(d_h) scale: with unit-variance q, k the raw scores have
# variance ~ d_h; scaled, ~1  (Monte Carlo, d_h = 64 for visibility)
dh = 64
q = rng.normal(size=(20000, dh))
k = rng.normal(size=(20000, dh))
raw = (q * k).sum(axis=1)
print(f"score variance, d_h={dh}: raw {raw.var():.1f} ≈ d_h; "
      f"scaled {(raw / np.sqrt(dh)).var():.2f} ≈ 1: OK")
assert abs(raw.var() / dh - 1) < 0.1
assert abs((raw / np.sqrt(dh)).var() - 1) < 0.1

# ----------------------------------------------------------------------
# Picture: the mechanism, visible
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))

im = ax1.imshow(A, cmap="viridis", vmin=0)
for i in range(T):
    for j in range(T):
        if j <= i:
            ax1.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                     fontsize=7,
                     color="white" if A[i, j] < .5 else "black")
ax1.set(xlabel="attended position j", ylabel="query position i",
        title="A: each row, a distribution over the past")
fig.colorbar(im, ax=ax1, shrink=.8)

gaps = [np.abs(h - a).max() for _, h, a in pairs]
ax2.barh(range(len(pairs)), np.maximum(gaps, 1e-18), color="#2a78d6")
ax2.set_yticks(range(len(pairs)),
               [name for name, _, _ in pairs])
ax2.set_xscale("log")
ax2.axvline(1e-12, color="#eb6834", ls="--", lw=1,
            label="assert tolerance")
ax2.set(xlabel="max |hand − autograd|",
        title="every tensor, machine precision")
ax2.legend(frameon=False, fontsize=8)
ax2.grid(alpha=.3, axis="x")
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

fig.tight_layout()
plt.show()
