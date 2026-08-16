r"""The neural n-gram (Bengio 2003) — backpropagation by hand, verified.

The parametric fix for counting's death (mlp_theoretical.py): stop
giving every context its own row. Instead:

  1. EMBED: each character gets a learned vector in R^d — a table
     C in R^{V x d}. A context of k characters becomes the
     concatenation of its k embeddings, a vector in R^{kd}.
  2. MAP: a one-hidden-layer network turns that vector into logits:

        E   = C[X]                     (B, k, d)   lookup
        F   = flatten(E)               (B, kd)
        H   = tanh(F W1 + b1)          (B, h)      hidden layer
        Z   = H W2 + b2                (B, V)      logits
        loss = CE(Z, Y) = mean_i [ -log softmax(Z_i)_{Y_i} ]

Parameters: Vd + kd*h + h + h*V + V — LINEAR in k (67k for k=8,
d=24, h=256), versus V^k(V-1) for the count table (10^16 at k=8).
Similar contexts now SHARE parameters through C, so a context never
seen in training still gets a sensible prediction. Distributed
representation — the idea that scales all the way to GPT.

BACKPROP, BY HAND. This file exists to earn the right to use autograd
on models we can no longer hand-check row by row. Every gradient below
is derived in ngram-mlp.tex and implemented here in numpy:

    dZ    = (softmax(Z) - onehot(Y)) / B      (the GLM residual, again)
    dW2   = H^T dZ          db2 = sum_i dZ_i
    dH    = dZ W2^T
    dPre  = dH * (1 - H^2)                    (tanh' = 1 - tanh^2)
    dW1   = F^T dPre        db1 = sum_i dPre_i
    dF    = dPre W1^T  -> unflatten -> scatter-add rows into dC

The proof by computation: identical initialization, identical batch
schedule, identical SGD steps in float64 — the numpy trajectory and the
torch autograd trajectory must coincide to machine precision, LOSS AND
EVERY PARAMETER TENSOR. Asserted below over 300 steps. After that, we
let autograd do the bookkeeping forever (mlp_practical_pytorch.py).

Run me with F5. Derivations: ngram-mlp.tex.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, load_everything

CONTEXT_K = 8            # characters of context
EMB_D = 24               # embedding dimension
HIDDEN = 256             # hidden layer width

ID_STEPS = 300           # identity-check steps (float64, CPU)
ID_BATCH = 32
ID_LR = 0.1


def init_params(rng: np.random.Generator, V: int):
    """Small random init (float64). Returned as a dict so the torch
    mirror can copy the SAME numbers."""
    return {
        "C":  rng.normal(0, 0.02, (V, EMB_D)),
        "W1": rng.normal(0, 0.02, (CONTEXT_K * EMB_D, HIDDEN)),
        "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0, 0.02, (HIDDEN, V)),
        "b2": np.zeros(V),
    }


def forward_backward(params, X, Y):
    """One full pass: loss, and the gradient of every parameter,
    entirely by hand. X: (B, k) int contexts; Y: (B,) int targets."""
    C, W1, b1, W2, b2 = (params[k] for k in ("C", "W1", "b1", "W2", "b2"))
    B = X.shape[0]

    # ---- forward ----
    E = C[X]                                   # (B, k, d) lookup
    F = E.reshape(B, -1)                       # (B, kd)
    pre = F @ W1 + b1
    H = np.tanh(pre)                           # (B, h)
    Z = H @ W2 + b2                            # (B, V) logits
    Zs = Z - Z.max(axis=1, keepdims=True)      # stable log-softmax
    logsumexp = np.log(np.exp(Zs).sum(axis=1, keepdims=True))
    logp = Zs - logsumexp
    loss = -logp[np.arange(B), Y].mean()

    # ---- backward (the derivation, executed) ----
    dZ = np.exp(logp)                          # softmax(Z)
    dZ[np.arange(B), Y] -= 1.0                 # minus onehot
    dZ /= B                                    # mean over the batch
    dW2 = H.T @ dZ
    db2 = dZ.sum(axis=0)
    dH = dZ @ W2.T
    dpre = dH * (1.0 - H ** 2)                 # tanh'
    dW1 = F.T @ dpre
    db1 = dpre.sum(axis=0)
    dF = dpre @ W1.T
    dE = dF.reshape(E.shape)                   # (B, k, d)
    dC = np.zeros_like(C)
    np.add.at(dC, X, dE)                       # scatter-add: rows reused
    return loss, {"C": dC, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def make_windows(ids: np.ndarray):
    """All (context, target) pairs: X[i] = k chars before position i."""
    T = len(ids) - CONTEXT_K
    X = np.stack([ids[i:i + CONTEXT_K] for i in range(T)])
    Y = ids[CONTEXT_K:]
    return X, Y


def make_batch_schedule(rng, n_positions, steps, batch):
    """steps x batch position indices — the shared schedule both
    implementations consume, so trajectories are comparable."""
    return rng.integers(0, n_positions, size=(steps, batch))


def sgd_by_hand(params, X, Y, schedule, lr):
    """Plain SGD with the hand gradients. Returns per-step losses."""
    losses = []
    for rows in schedule:
        loss, grads = forward_backward(params, X[rows], Y[rows])
        for k in params:
            params[k] -= lr * grads[k]
        losses.append(loss)
    return np.array(losses)


def sgd_by_torch(params0, X, Y, schedule, lr):
    """The same steps, autograd doing the calculus. float64 mirror of
    the same initial numbers."""
    P = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
         for k, v in params0.items()}
    opt = torch.optim.SGD(P.values(), lr=lr)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    losses = []
    for rows in schedule:
        xb, yb = Xt[rows], Yt[rows]
        E = P["C"][xb].reshape(len(rows), -1)
        H = torch.tanh(E @ P["W1"] + P["b1"])
        Z = H @ P["W2"] + P["b2"]
        loss = TF.cross_entropy(Z, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return np.array(losses), {k: v.detach().numpy() for k, v in P.items()}


if __name__ == "__main__":
    train_ids, val_ids, chars, stoi, itos = load_everything()
    V = len(chars)
    rng = np.random.default_rng(SEED)

    X, Y = make_windows(train_ids)
    schedule = make_batch_schedule(rng, len(X), ID_STEPS, ID_BATCH)

    params_hand = init_params(rng, V)
    params_init = {k: v.copy() for k, v in params_hand.items()}

    print(f"identity check: {ID_STEPS} SGD steps, batch {ID_BATCH}, "
          f"float64, CPU ...")
    loss_hand = sgd_by_hand(params_hand, X, Y, schedule, ID_LR)
    loss_auto, params_auto = sgd_by_torch(params_init, X, Y, schedule,
                                          ID_LR)

    loss_gap = np.abs(loss_hand - loss_auto).max()
    print(f"max |loss_hand - loss_autograd| : {loss_gap:.2e}")
    worst = 0.0
    for k in params_hand:
        gap = np.abs(params_hand[k] - params_auto[k]).max()
        worst = max(worst, gap)
        print(f"  {k:>3}: max param gap {gap:.2e}")
    assert loss_gap < 1e-10 and worst < 1e-10, \
        "hand backprop disagrees with autograd!"
    print("hand backprop == autograd, every parameter, every step: OK")

    # training is actually working, too (not just matching)
    assert loss_hand[-1] < loss_hand[:10].mean() - 0.3, \
        "loss should be falling"
    print(f"loss: {loss_hand[:10].mean():.3f} (start) -> "
          f"{loss_hand[-1]:.3f} (step {ID_STEPS})")

    # ------------------------------------------------------------------
    # Picture: trajectories coincide; per-step gap at machine precision
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("proof by computation: hand backprop = autograd, "
                 "one hidden layer deep", fontsize=11)

    steps = np.arange(ID_STEPS)
    ax1.plot(steps, loss_hand, color="#3d9b35", lw=2, label="hand backprop")
    ax1.plot(steps, loss_auto, color="#eb6834", lw=1, ls="--",
             label="autograd")
    ax1.set(xlabel="SGD step", ylabel="batch CE loss",
            title="two implementations, one trajectory")
    ax1.legend(frameon=False, fontsize=8)

    ax2.semilogy(steps, np.maximum(np.abs(loss_hand - loss_auto), 1e-18),
                 color="#2a78d6", lw=.8)
    ax2.axhline(1e-10, color="#eb6834", ls="--", lw=1,
                label="assert tolerance 1e-10")
    ax2.set(xlabel="SGD step", ylabel="|loss gap|",
            title=f"per-step gap (max = {loss_gap:.1e})")
    ax2.legend(frameon=False, fontsize=8)

    for ax in (ax1, ax2):
        ax.grid(alpha=.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.show()
