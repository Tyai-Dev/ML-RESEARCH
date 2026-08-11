r"""Bernoulli MLE — the practical solution, by hand (numpy only).

Pretend the closed form p̂ = m/n is unavailable and *descend* the NLL,
the way we must for models with no closed form (logistic regression
onward). Two routes, both with hand-derived gradients:

The coordinate change (same objective, new coordinate).
The NLL lives on p in (0,1) — a constrained problem. Substitute
p = sigma(t) with t real: F(t) = NLL(sigma(t)). This is NOT a new
objective — it is the same NLL read through the sigmoid, and since
sigma is a bijection onto (0,1), minimizing F over R is minimizing NLL
over (0,1). The chain rule collapses beautifully (bernoulli.tex):

    F'(t) = sigma(t) - x̄            (full gradient)
    g_i(t) = sigma(t) - x_i          (per-sample; E[g_i] = F'(t))

Route GD:  t <- t - lr * (sigma(t) - x̄).   Uses all n samples per step;
deterministic, converges monotonically to sigma^{-1}(x̄).

Route SGD: t <- t - lr * (sigma(t) - x_i)  for one sample at a time.
Each step is cheap but noisy; with a constant step size the iterate
reaches a *noise ball* around the optimum and hovers, so we report the
Polyak–Ruppert average of the last epoch, which shrinks the noise.

The animation (the point of this file): watch the estimator descend.
Left panel — the NLL landscape with GD (orange) and SGD (green) beads
sliding down toward the closed-form minimum. Right panel — the same
progress as a loss/estimate curve growing over training time. GD's bead
glides; SGD's bead jitters into the noise ball. Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from common import (GD_LR, GD_STEPS, SGD_LR, make_data, make_schedule,
                    nll_of_p, sigmoid)


def gradient_descent(x: np.ndarray) -> np.ndarray:
    """Full-batch GD on F(t) = NLL(sigma(t)), from t=0 (p = 0.5).
    Returns the trajectory of p_k = sigma(t_k), k = 0..GD_STEPS."""
    xbar = x.mean()
    t, traj = 0.0, [sigmoid(0.0)]
    for _ in range(GD_STEPS):
        t -= GD_LR * (sigmoid(t) - xbar)      # F'(t) = sigma(t) - x̄
        traj.append(sigmoid(t))
    return np.array(traj)


def sgd(x: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """One-sample SGD on the same objective, visiting x[i] for i in
    `schedule`. Returns p after every step (len(schedule)+1 values)."""
    t, traj = 0.0, [sigmoid(0.0)]
    for i in schedule:
        t -= SGD_LR * (sigmoid(t) - x[i])     # g_i(t) = sigma(t) - x_i
        traj.append(sigmoid(t))
    return np.array(traj)


def animate(traj_gd: np.ndarray, traj_sgd: np.ndarray, xbar: float,
            p_closed: float, frames: int = 200) -> FuncAnimation:
    """Two synchronized views of the descent: beads on the NLL landscape
    (left) and the estimate-vs-training-progress curve growing (right)."""
    pg = np.linspace(0.02, 0.98, 400)
    nll = lambda p: nll_of_p(p, xbar)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.suptitle("Bernoulli MLE: the estimator descending the NLL",
                 fontsize=11)

    # -- left: the landscape and the two beads ------------------------
    ax1.plot(pg, nll(pg), color="#2a78d6", lw=2, zorder=1)
    ax1.plot(p_closed, nll(p_closed), "*", color="#111", ms=12,
             label=f"closed form p̂ = {p_closed:.4f}", zorder=3)
    bead_gd, = ax1.plot([], [], "o", color="#eb6834", ms=9,
                        label="GD (full gradient)", zorder=4)
    bead_sgd, = ax1.plot([], [], "o", color="#3d9b35", ms=7, alpha=.85,
                         label="SGD (one sample)", zorder=4)
    trail_sgd, = ax1.plot([], [], ".", color="#3d9b35", ms=2, alpha=.25,
                          zorder=2)
    ax1.set(xlabel="p", ylabel="NLL(p)", title="on the landscape")
    ax1.legend(frameon=False, fontsize=8)

    # -- right: estimate vs training progress -------------------------
    # GD takes GD_STEPS steps, SGD takes |schedule| steps; put both on a
    # common [0,1] "fraction of training" axis so they share a panel.
    xs_gd = np.linspace(0, 1, len(traj_gd))
    xs_sgd = np.linspace(0, 1, len(traj_sgd))
    ax2.axhline(p_closed, color="#111", ls="--", lw=1,
                label=f"closed form {p_closed:.4f}")
    line_gd, = ax2.plot([], [], color="#eb6834", lw=2, label="GD")
    line_sgd, = ax2.plot([], [], color="#3d9b35", lw=1, alpha=.8,
                         label="SGD")
    ax2.set(xlim=(0, 1), ylim=(0.25, 0.55),
            xlabel="fraction of training", ylabel="estimate of p",
            title="the same progress, unrolled in time")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    for ax in (ax1, ax2):
        ax.grid(alpha=.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()

    # frame k shows both routes at the same *fraction* of their training
    idx_gd = np.linspace(0, len(traj_gd) - 1, frames).astype(int)
    idx_sgd = np.linspace(0, len(traj_sgd) - 1, frames).astype(int)

    def update(k):
        i, j = idx_gd[k], idx_sgd[k]
        bead_gd.set_data([traj_gd[i]], [nll(traj_gd[i])])
        bead_sgd.set_data([traj_sgd[j]], [nll(traj_sgd[j])])
        trail_sgd.set_data(traj_sgd[:j + 1], nll(traj_sgd[:j + 1]))
        line_gd.set_data(xs_gd[:i + 1], traj_gd[:i + 1])
        line_sgd.set_data(xs_sgd[:j + 1], traj_sgd[:j + 1])
        return bead_gd, bead_sgd, trail_sgd, line_gd, line_sgd

    return FuncAnimation(fig, update, frames=frames, interval=35,
                         blit=True, repeat=True)


if __name__ == "__main__":
    x, rng = make_data()
    schedule = make_schedule(rng)
    p_closed = x.mean()                       # the target, for reference

    traj_gd = gradient_descent(x)
    traj_sgd = sgd(x, schedule)
    # Polyak–Ruppert: average the last epoch's iterates (len(x) steps)
    p_polyak = float(traj_sgd[-len(x):].mean())

    print(f"closed form  p̂ = x̄        : {p_closed:.6f}")
    print(f"GD final ({GD_STEPS} steps)      : {traj_gd[-1]:.6f}"
          f"   |diff| = {abs(traj_gd[-1] - p_closed):.2e}")
    print(f"SGD last iterate           : {traj_sgd[-1]:.6f}")
    print(f"SGD Polyak (last epoch avg): {p_polyak:.6f}"
          f"   |diff| = {abs(p_polyak - p_closed):.2e}")

    # GD must land on the closed form; SGD's average must be close.
    assert abs(traj_gd[-1] - p_closed) < 1e-8, "GD did not converge!"
    assert abs(p_polyak - p_closed) < 5e-3, "SGD average too far off!"
    print("descent reaches the theoretical optimum: OK")

    ani = animate(traj_gd, traj_sgd, x.mean(), p_closed)  # keep a ref!
    plt.show()
