r"""Bigram language model — the practical solution, via PyTorch autograd.

Same objective, autograd does the calculus. The per-sample loss for a
transition pair (a, b) is F.cross_entropy(W[a], b) — exactly the
one-sample NLL through the softmax coordinate change — and reverse-mode
differentiation must produce the gradient we derived by hand:

    d loss / dW[a] = softmax(W[a]) - onehot(b),     zeros elsewhere.

Proof by computation (part 1, CPU, float64). Drive torch's SGD through
the IDENTICAL pair schedule, learning rate, and start W = 0 as the
hand-written SGD in bigram_practical_pure.py, and record the same probe
P('h' | 't') at every step. If autograd's gradient is our gradient, the
two probe trajectories must coincide to machine precision — asserted
over 20,000 steps. (The bernoulli move, on a 65x65 parameter.)

Sufficiency does the heavy lifting (part 2, GPU). The FULL-batch loss
over ~1M pairs collapses to counts (statistics never dies):

    L(W) = -(1/n) sum_ab n_ab log softmax(W_a)_b,

so one 65x65 count matrix carries the entire dataset's gradient. We
minimize this with Adam on the RTX 4070 and assert the learned
probabilities land on the count table to 1e-3 — two routes, one
destination, third implementation.

Run me with F5. Derivations: bigram.tex.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, get_device, load_everything, decode
from bigram_practical_pure import (SGD_LR, TRACK, count_model,
                                   make_schedule, sample_from,
                                   softmax_rows, sgd as sgd_by_hand)

N_IDENTITY = 20_000                  # steps compared step-for-step


def sgd_torch_tracked(pairs_a, pairs_b, schedule, V, track_ids):
    """Mirror of bigram_practical_pure.sgd: same start (W = 0), same
    lr, same pair order — but the gradient comes from autograd.
    float64 on CPU so the comparison is exact, not approximate."""
    W = torch.zeros((V, V), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([W], lr=SGD_LR)
    ta, tb = track_ids
    tracked = []
    for idx in schedule:
        with torch.no_grad():
            tracked.append(torch.softmax(W[ta], dim=0)[tb].item())
        a, b = int(pairs_a[idx]), int(pairs_b[idx])
        opt.zero_grad()
        loss = F.cross_entropy(W[a].unsqueeze(0),
                               torch.tensor([b]))
        loss.backward()              # autograd: softmax - onehot, row a
        opt.step()
    with torch.no_grad():
        tracked.append(torch.softmax(W[ta], dim=0)[tb].item())
    return np.array(tracked)


def train_full_batch(counts: np.ndarray, device: torch.device):
    """Full-batch training on the count-collapsed loss (sufficiency:
    the 65x65 count matrix IS the dataset, gradient-wise). Adam with a
    manual lr drop; returns the learned probability table."""
    C = torch.tensor(counts, dtype=torch.float32, device=device)
    W = torch.zeros_like(C, requires_grad=True)
    n = C.sum()
    for lr, steps in [(0.1, 3000), (0.01, 2000)]:
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = -(C * F.log_softmax(W, dim=1)).sum() / n
            loss.backward()
            opt.step()
    return torch.softmax(W, dim=1).detach().cpu().numpy(), float(loss)


if __name__ == "__main__":
    train_ids, val_ids, chars, stoi, itos = load_everything()
    V = len(chars)
    rng = np.random.default_rng(SEED)
    track_ids = (stoi[TRACK[0]], stoi[TRACK[1]])

    counts, P_hat, nll_floor = count_model(train_ids, V)
    pairs_a, pairs_b = train_ids[:-1], train_ids[1:]
    schedule = make_schedule(rng, len(pairs_a))[:N_IDENTITY]

    # ---- part 1: the identity, hand vs autograd, step for step ------
    print(f"identity check: {N_IDENTITY:,} SGD steps, float64, CPU ...")
    _, _, probe_hand = sgd_by_hand(pairs_a, pairs_b, schedule, V,
                                   track_ids=track_ids)
    probe_auto = sgd_torch_tracked(pairs_a, pairs_b, schedule, V,
                                   track_ids)
    diff = np.abs(probe_hand - probe_auto)
    print(f"probe = P({TRACK[1]!r} | {TRACK[0]!r})")
    print(f"max |hand - autograd|      : {diff.max():.2e}")
    assert np.allclose(probe_hand, probe_auto, atol=1e-10), \
        "autograd disagrees with the hand-derived softmax gradient!"
    print("autograd trajectory == hand-derived trajectory: OK")

    # ---- part 2: full-batch on the GPU, landing on the count table --
    device = get_device()
    t0 = time.perf_counter()
    P_learned, final_loss = train_full_batch(counts, device)
    dt = time.perf_counter() - t0
    err = np.abs(P_learned - P_hat).max()
    print(f"\nfull-batch Adam on {device} : {dt:.1f}s, "
          f"final NLL {final_loss:.4f} (floor {nll_floor:.4f})")
    print(f"max |learned - count table|: {err:.2e}")
    assert err < 1e-3, "training did not land on the count table!"
    print("two routes, one destination (GPU edition): OK")

    print("\n--- sample from the GPU-trained model " + "-" * 22)
    print(sample_from(P_learned.astype(np.float64) /
                      P_learned.astype(np.float64).sum(1, keepdims=True),
                      rng, itos))
    print("-" * 60)

    # ---- animation: the two probes draw together, gap pinned --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("proof by computation: autograd = hand softmax gradient",
                 fontsize=11)

    steps = np.arange(len(probe_hand))
    ax1.axhline(P_hat[track_ids], color="#111", ls=":", lw=1,
                label=f"count table P({TRACK[1]!r}|{TRACK[0]!r}) = "
                      f"{P_hat[track_ids]:.3f}")
    line_hand, = ax1.plot([], [], color="#3d9b35", lw=2,
                          label="hand gradient")
    line_auto, = ax1.plot([], [], color="#eb6834", lw=1, ls="--",
                          label="autograd")
    bead_hand, = ax1.plot([], [], "o", color="#3d9b35", ms=8)
    bead_auto, = ax1.plot([], [], "x", color="#eb6834", ms=8, mew=2)
    ax1.set(xlim=(0, len(steps)), ylim=(0, .5),
            xlabel="SGD step",
            ylabel=f"P({TRACK[1]!r} | {TRACK[0]!r})",
            title="two implementations, one trajectory")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")

    gap = np.maximum(diff, 1e-18)
    ax2.set_yscale("log")
    ax2.axhline(1e-10, color="#eb6834", ls="--", lw=1,
                label="assert tolerance 1e-10")
    line_gap, = ax2.plot([], [], color="#2a78d6", lw=.8,
                         label="|hand − autograd|")
    ax2.set(xlim=(0, len(steps)), ylim=(1e-19, 1e-8),
            xlabel="SGD step", ylabel="|hand − autograd|",
            title=f"per-step gap (max = {diff.max():.1e})")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    for ax in (ax1, ax2):
        ax.grid(alpha=.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()

    FRAMES = 200
    idx = np.linspace(0, len(steps) - 1, FRAMES).astype(int)

    def update(k):
        j = idx[k]
        line_hand.set_data(steps[:j + 1], probe_hand[:j + 1])
        line_auto.set_data(steps[:j + 1], probe_auto[:j + 1])
        bead_hand.set_data([steps[j]], [probe_hand[j]])
        bead_auto.set_data([steps[j]], [probe_auto[j]])
        line_gap.set_data(steps[:j + 1], gap[:j + 1])
        return line_hand, line_auto, bead_hand, bead_auto, line_gap

    ani = FuncAnimation(fig, update, frames=FRAMES, interval=35,
                        blit=True, repeat=True)   # keep a ref!
    plt.show()
