r"""Bernoulli MLE — the practical solution, via PyTorch autograd.

Same objective, third implementation. We again minimize
F(t) = NLL(sigma(t)) by one-sample SGD, but this time NOBODY writes the
gradient: we hand autograd the per-sample loss

    loss_i(t) = -[ x_i log sigma(t) + (1 - x_i) log(1 - sigma(t)) ]
              = F.binary_cross_entropy_with_logits(t, x_i)

(the "with_logits" form composes sigmoid + BCE in one numerically-stable
call — exactly our coordinate change), and reverse-mode differentiation
applies the same chain rule we did on paper, mechanically.

The proof by computation. If autograd truly computes
d loss_i / dt = sigma(t) - x_i, then driving torch's SGD through the
IDENTICAL sample schedule, learning rate, and start point as the
hand-written SGD in mle_practical_pure.py must produce the identical
trajectory — every one of the 15,000 iterates, to machine precision
(float64). That is asserted below, and ANIMATED: the two trajectories
draw simultaneously and never separate (left), while their per-step
|difference| stays pinned to zero on a log scale (right) — thousands of
steps deep, live.

Run me with F5. Derivations: mle_practical.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation

from common import SGD_LR, make_data, make_schedule
from mle_practical_pure import sgd as sgd_by_hand   # guarded: no side effects


def sgd_torch(x: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """One-sample SGD where the gradient comes from autograd. Mirrors
    mle_practical_pure.sgd step for step: same start t=0, same lr, same
    sample order. float64 so the comparison is exact, not approximate."""
    x_t = torch.from_numpy(x)                                # float64
    t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([t], lr=SGD_LR)

    traj = [torch.sigmoid(t).item()]
    for i in schedule:
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(t, x_t[i])
        loss.backward()          # autograd: d loss/dt = sigma(t) - x_i
        opt.step()               # t <- t - lr * grad  (same update rule)
        traj.append(torch.sigmoid(t).item())
    return np.array(traj)


if __name__ == "__main__":
    x, rng = make_data()
    schedule = make_schedule(rng)

    traj_hand = sgd_by_hand(x, schedule)      # the pure-numpy trajectory
    traj_auto = sgd_torch(x, schedule)        # the autograd trajectory

    diff = np.abs(traj_hand - traj_auto)
    print(f"steps compared             : {len(diff)}")
    print(f"max |hand - autograd|      : {diff.max():.2e}")
    print(f"final estimate (autograd)  : {traj_auto[-1]:.6f}")
    print(f"closed form x̄              : {x.mean():.6f}")

    # The identity: autograd == our chain rule, at every single step.
    assert np.allclose(traj_hand, traj_auto, atol=1e-10), \
        "autograd disagrees with the hand-derived gradient!"
    print("autograd trajectory == hand-derived trajectory: OK")

    # ------------------------------------------------------------------
    # Animation: the two trajectories draw together and never separate,
    # while the per-step gap stays pinned to (machine) zero — live.
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("proof by computation: autograd = hand chain rule",
                 fontsize=11)

    steps = np.arange(len(traj_hand))
    ax1.axhline(x.mean(), color="#111", ls=":", lw=1,
                label=f"closed form {x.mean():.4f}")
    line_hand, = ax1.plot([], [], color="#3d9b35", lw=2,
                          label="hand gradient")
    line_auto, = ax1.plot([], [], color="#eb6834", lw=1, ls="--",
                          label="autograd")
    # beads at the writing heads — watch them move as ONE point
    bead_hand, = ax1.plot([], [], "o", color="#3d9b35", ms=8)
    bead_auto, = ax1.plot([], [], "x", color="#eb6834", ms=8, mew=2)
    ax1.set(xlim=(0, len(steps)), ylim=(0.25, 0.55),
            xlabel="SGD step", ylabel="estimate of p",
            title="two implementations, one trajectory")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # zeros can't be drawn on a log axis; clip to float64's floor
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
        line_hand.set_data(steps[:j + 1], traj_hand[:j + 1])
        line_auto.set_data(steps[:j + 1], traj_auto[:j + 1])
        bead_hand.set_data([steps[j]], [traj_hand[j]])
        bead_auto.set_data([steps[j]], [traj_auto[j]])
        line_gap.set_data(steps[:j + 1], gap[:j + 1])
        return line_hand, line_auto, bead_hand, bead_auto, line_gap

    ani = FuncAnimation(fig, update, frames=FRAMES, interval=35,
                        blit=True, repeat=True)   # keep a ref!
    plt.show()
