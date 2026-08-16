r"""Bigram language model — the practical solution, by hand (numpy only).

Pretend the count table is unavailable and DESCEND, the way we must
from rung 2 of the ladder onward, where no closed form exists.

The ML translation (bigram.tex). The average training NLL over the
n transition pairs, written with counts n_ab and row weights
w_a = n_a / n, is a count-weighted cross-entropy:

    L = -(1/n) sum_ab n_ab log p(b|a) = sum_a w_a * CE(p̂_a, p_a),

where p̂_a is row a of the count model. Parameters: a V x V table of
LOGITS W (the same coordinate-change move as bernoulli's sigmoid, one
level up: softmax maps each unconstrained row of W onto the simplex).

The gradient, by the chain rule (derived in softmax-regression, reused
here): for row a,

    dL/dW_a = w_a * (softmax(W_a) - p̂_a),         (full gradient)
    g(W; a,b) = softmax(W_a) - onehot(b) on row a,  (per-sample, E = full)

"residual times nothing" — prediction minus target, the GLM signature.

Route GD. The loss separates across rows (no term couples them), so we
descend each row with its own full gradient softmax(W_a) - p̂_a — V
independent convex problems solved simultaneously. The unique fixed
point is softmax(W_a) = p̂_a: GRADIENT DESCENT CONVERGES TO THE COUNT
TABLE. Two routes, one destination — asserted below to 1e-3.

Route SGD. Visit transition pairs (a,b) one at a time, update only row
a with softmax(W_a) - onehot(b). Unbiased (averaging over pairs gives
the full gradient), cheap, noisy: one epoch over the ~1M pairs gets
close to the count table's NLL without ever building the count table.

The animation: GD's whole table converging — the error heatmap
|softmax(W_k) - p̂| fading to black, the train NLL falling onto the
count-MLE floor, and the model's text samples sharpening from uniform
noise toward Shakespeare-shaped babble, all in sync.

Run me with F5. Derivations: bigram.tex.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, load_everything, decode

GD_LR, GD_STEPS = 2.0, 3000
SGD_LR, SGD_STEPS = 0.5, 1_000_000          # ~one epoch over the pairs
CHECK_EVERY = 30                             # GD animation checkpoints
TRACK = ("t", "h")                           # scalar probe: P('h'|'t')


def softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, max-shifted for stability (same arithmetic
    torch uses, so the identity check can be exact)."""
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


def count_model(train_ids: np.ndarray, V: int):
    """The theoretical solution, needed here only as the target/floor:
    count table p̂ and its train NLL."""
    counts = np.zeros((V, V), dtype=np.float64)
    np.add.at(counts, (train_ids[:-1], train_ids[1:]), 1.0)
    row_tot = counts.sum(axis=1, keepdims=True)
    P_hat = np.where(row_tot > 0, counts / np.maximum(row_tot, 1), 1.0 / V)
    n = counts.sum()
    m = counts > 0
    nll_floor = -(counts[m] * np.log(P_hat[m])).sum() / n
    return counts, P_hat, nll_floor


def weighted_nll(W: np.ndarray, counts: np.ndarray) -> float:
    """Train NLL of the logit model, computed from the counts alone
    (sufficiency: the 65x65 count table is all the data we need)."""
    logP = np.log(softmax_rows(W))
    return float(-(counts * logP).sum() / counts.sum())


def gradient_descent(counts: np.ndarray, P_hat: np.ndarray):
    """Per-row full-gradient descent from W = 0 (uniform model).
    Returns the final W and checkpoint history for the animation."""
    V = counts.shape[0]
    W = np.zeros((V, V))
    snaps, losses, steps = [], [], []
    for k in range(GD_STEPS + 1):
        if k % CHECK_EVERY == 0:
            snaps.append(softmax_rows(W))
            losses.append(weighted_nll(W, counts))
            steps.append(k)
        W -= GD_LR * (softmax_rows(W) - P_hat)   # dL_a/dW_a per row
    return W, snaps, losses, steps


def sgd(pairs_a, pairs_b, schedule, V, track_ids=None):
    """One-pair-at-a-time SGD from W = 0. Updates ONLY the visited row:
    W[a] -= lr * (softmax(W[a]) - onehot(b)). If track_ids = (ta, tb)
    is given, records the probe P(tb|ta) at every step — the trajectory
    the pytorch file must reproduce exactly.

    Constant step size => the iterate hovers in a noise ball around the
    optimum (bernoulli's lesson, at 65x65 scale). The cure is the same:
    Polyak–Ruppert — average the iterates over the second half of
    training (thinned to every 100th snapshot; averaging W is averaging
    in parameter space, and the noise cancels). Returns (W_last,
    W_polyak, tracked)."""
    W = np.zeros((V, V))
    W_avg, n_avg = np.zeros((V, V)), 0
    polyak_from = len(schedule) // 2
    tracked = []
    for step, idx in enumerate(schedule):
        if track_ids is not None:
            ta, tb = track_ids
            tracked.append(softmax_rows(W[ta])[tb])
        a, b = pairs_a[idx], pairs_b[idx]
        p = softmax_rows(W[a])
        p[b] -= 1.0                              # softmax - onehot
        W[a] -= SGD_LR * p
        if step >= polyak_from and step % 100 == 0:
            W_avg += W
            n_avg += 1
    if track_ids is not None:
        ta, tb = track_ids
        tracked.append(softmax_rows(W[ta])[tb])
    W_polyak = W_avg / max(n_avg, 1)
    return W, W_polyak, np.array(tracked)


def make_schedule(rng: np.random.Generator, n_pairs: int) -> np.ndarray:
    """The visiting order for SGD_STEPS one-sample steps. Shared with
    the pytorch file so trajectories are comparable step for step."""
    order = rng.permutation(n_pairs)
    return order[:SGD_STEPS]


def sample_from(P_table, rng, itos, n_chars=240, start_id=0):
    """Ancestral sampling from a probability table."""
    V = P_table.shape[0]
    out, a = [], start_id
    for _ in range(n_chars):
        a = int(rng.choice(V, p=P_table[a]))
        out.append(a)
    return decode(out, itos)


def animate(snaps, losses, steps, P_hat, nll_floor, itos, rng):
    """Three synchronized views of GD learning the language model:
    the error heatmap fading, the loss falling to the MLE floor, and
    the samples sharpening."""
    n_frames = len(snaps)
    # pre-sample text at every checkpoint with a FIXED generator state
    # per frame, so successive frames differ by model, not by luck
    texts = []
    for S in snaps:
        r = np.random.default_rng(SEED)
        texts.append(sample_from(S, r, itos, n_chars=180))

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.3])
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
    fig.suptitle("gradient descent converges to the count table",
                 fontsize=11)

    im = ax1.imshow(np.abs(snaps[0] - P_hat), cmap="inferno",
                    vmin=0, vmax=0.05, aspect="auto")
    ax1.set(title=r"$|\mathrm{softmax}(W_k) - \hat{p}|$",
            xlabel="next char", ylabel="current char")
    fig.colorbar(im, ax=ax1, shrink=.8)

    ax2.axhline(nll_floor, color="#111", ls="--", lw=1,
                label=f"count-MLE floor {nll_floor:.3f}")
    ax2.axhline(np.log(P_hat.shape[0]), color="#898781", ls=":", lw=1,
                label=f"uniform log V = {np.log(P_hat.shape[0]):.3f}")
    lossline, = ax2.plot([], [], color="#eb6834", lw=2, label="GD train NLL")
    ax2.set(xlim=(0, steps[-1]), ylim=(nll_floor - .1, 4.4),
            xlabel="GD step", ylabel="NLL (nats/char)",
            title="the loss progression")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=.3)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    ax3.set_axis_off()
    ax3.set_title("samples from the model", fontsize=10)
    step_txt = ax3.text(0, .98, "", transform=ax3.transAxes, fontsize=8,
                        va="top", color="#eb6834", family="monospace")
    body_txt = ax3.text(0, .90, "", transform=ax3.transAxes, fontsize=7.5,
                        va="top", family="monospace", wrap=True)

    def update(k):
        im.set_data(np.abs(snaps[k] - P_hat))
        lossline.set_data(steps[:k + 1], losses[:k + 1])
        step_txt.set_text(f"step {steps[k]:>5d}   NLL {losses[k]:.3f}")
        body_txt.set_text(texts[k])
        return im, lossline, step_txt, body_txt

    return FuncAnimation(fig, update, frames=n_frames, interval=60,
                         blit=False, repeat=True)


if __name__ == "__main__":
    train_ids, val_ids, chars, stoi, itos = load_everything()
    V = len(chars)
    rng = np.random.default_rng(SEED)

    counts, P_hat, nll_floor = count_model(train_ids, V)
    print(f"count-MLE train NLL (floor): {nll_floor:.4f} nats/char")

    # ---- Route GD: all rows, full gradients, deterministic ----------
    W_gd, snaps, losses, steps = gradient_descent(counts, P_hat)
    gd_err = np.abs(softmax_rows(W_gd) - P_hat).max()
    print(f"GD ({GD_STEPS} steps)          : max|softmax(W) - p̂| = "
          f"{gd_err:.2e}")
    assert gd_err < 1e-3, "GD did not reach the count table!"
    print("two routes, one destination: OK")

    # ---- Route SGD: one pair at a time, never builds the table -----
    pairs_a, pairs_b = train_ids[:-1], train_ids[1:]
    schedule = make_schedule(rng, len(pairs_a))
    W_sgd, W_polyak, _ = sgd(pairs_a, pairs_b, schedule, V)
    nll_last = weighted_nll(W_sgd, counts)
    nll_poly = weighted_nll(W_polyak, counts)
    print(f"SGD ({SGD_STEPS:,} steps)  : last-iterate NLL {nll_last:.4f}"
          f"  (noise ball above floor {nll_floor:.4f})")
    print(f"SGD Polyak average         : NLL {nll_poly:.4f}"
          f"  (uniform start: {np.log(V):.4f})")
    assert nll_poly < nll_last, "averaging should beat the last iterate"
    assert nll_poly < nll_floor + 0.03, "SGD stalled far from the floor!"

    print("\n--- sample from the GD model " + "-" * 30)
    print(sample_from(softmax_rows(W_gd), rng, itos))
    print("-" * 60)

    ani = animate(snaps, losses, steps, P_hat, nll_floor, itos, rng)
    plt.show()
