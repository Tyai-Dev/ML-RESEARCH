r"""Markov chain MLE — learning from DEPENDENT data, report card included.

Every estimator in this repository so far assumed i.i.d. samples. A
Markov chain's samples are correlated by construction — X_{t+1} is
drawn from a distribution CHOSEN by X_t — and this file shows that
maximum likelihood survives the loss of independence, with the same
report-card discipline as bernoulli's mle_theoretical.py.

The estimator. For paths x^(r) = (x_1..x_T), r = 1..R, the likelihood

    L(pi0, P) = prod_r [ pi0(x_1^r) * prod_t P_{x_t^r, x_{t+1}^r} ]

separates into independent multinoulli problems (the log turns the
products into sums; no term couples two rows of P, nor P with pi0), so
by the count argument of Generative/text-to-text/bigram.tex, boxed in markov-chains.tex:

    P̂_ij   = n_ij / n_i        (transition counts, row-normalized)
    pî0(i) = #{r : x_1^r = i} / R   (first states — needs MANY paths;
                                     from ONE path it is degenerate, a
                                     point mass on the observed start)

The report card, under dependence — what survives and why:

CONSISTENT. The ergodic theorem (markov_chains_demo.py) does what the
LLN did for i.i.d. data: along one path, state i is visited
n_i ≈ T pi_i times, so n_i -> infinity and each row's multinoulli MLE
concentrates. Dependence slows nothing here — it only makes the row
sample sizes RANDOM.

RATE. Conditional on the visits, row i is an ordinary multinoulli
estimate from n_i samples:  Var(P̂_ij) ≈ P_ij (1 - P_ij) / (T pi_i).
Rare states learn slowly — the sqrt(T) law, weighted by occupancy.
Verified two ways below: the aggregate error falls with slope -1/2 in
total transitions (log-log), and the per-entry Monte Carlo standard
deviation matches the formula to a few percent.

CONFIDENCE INTERVALS still work. P̂_ij ± 1.96 sqrt(P̂(1-P̂)/n_i) covers
the truth ~95% of the time — measured over 1000 replicated paths. The
CLT behind it is a martingale CLT rather than the i.i.d. one, but the
practice is unchanged.

SUFFICIENT. The whole dataset enters only through the transition-count
matrix and the first-state counts.

The animation: one path streaming in, the error heatmap |P̂ - P|
fading, the error curve descending on the -1/2 line, and the occupancy
bars converging to pi (the ergodic theorem, live).

Run me with F5. Derivations: markov-chains.tex.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

SEED = 7
rng = np.random.default_rng(SEED)

STATES = ["Sunny", "Cloudy", "Rainy"]
P_TRUE = np.array([[0.6, 0.3, 0.1],
                   [0.2, 0.5, 0.3],
                   [0.1, 0.6, 0.3]])
PI0_TRUE = np.array([0.5, 0.2, 0.3])          # non-uniform start dist
V = len(STATES)


def stationary(P):
    w, vec = np.linalg.eig(P.T)
    pi = np.real(vec[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


PI = stationary(P_TRUE)


def simulate(P, T, rng, pi0=None):
    """One path; the start drawn from pi0 (default: stationary)."""
    path = np.empty(T, dtype=np.int64)
    path[0] = rng.choice(V, p=pi0 if pi0 is not None else PI)
    for t in range(1, T):
        path[t] = rng.choice(V, p=P[path[t - 1]])
    return path


def fit(paths, V):
    """The MLE: count transitions over all paths, normalize rows;
    estimate pi0 from the first states. Returns (P̂, pî0, counts)."""
    C = np.zeros((V, V))
    firsts = np.zeros(V)
    for p in paths:
        np.add.at(C, (p[:-1], p[1:]), 1.0)
        firsts[p[0]] += 1
    row = C.sum(axis=1, keepdims=True)
    P_hat = np.where(row > 0, C / np.maximum(row, 1), 1.0 / V)
    return P_hat, firsts / firsts.sum(), C


if __name__ == "__main__":
    # ---- consistency + rate: error vs data on the -1/2 line ---------
    print("rate check: RMS error vs total transitions (8 reps each)")
    Ts = [500, 2_000, 8_000, 32_000, 128_000]
    errs = []
    for T in Ts:
        e = [np.sqrt(np.mean((fit([simulate(P_TRUE, T, rng)], V)[0]
                              - P_TRUE) ** 2)) for _ in range(8)]
        errs.append(np.mean(e))
        print(f"  T = {T:>7,}   RMS|P̂ - P| = {errs[-1]:.5f}")
    slope = np.polyfit(np.log(Ts), np.log(errs), 1)[0]
    print(f"log-log slope: {slope:.3f}  (theory: -1/2)")
    assert abs(slope + 0.5) < 0.07, "dependent data still learns at √T!"

    # ---- the variance formula, entry by entry -----------------------
    print("\nvariance check (1000 replicated paths, T = 2000):")
    REPS, T = 1_000, 2_000
    hats = np.empty((REPS, V, V))
    covered = np.zeros((V, V))
    for r in range(REPS):
        p = simulate(P_TRUE, T, rng)
        P_hat, _, C = fit([p], V)
        hats[r] = P_hat
        n_i = C.sum(axis=1, keepdims=True)
        se = np.sqrt(P_hat * (1 - P_hat) / np.maximum(n_i, 1))
        covered += (np.abs(P_hat - P_TRUE) <= 1.96 * se)
    sd_emp = hats.std(axis=0)
    sd_theory = np.sqrt(P_TRUE * (1 - P_TRUE) / ((T - 1) * PI[:, None]))
    rel = np.abs(sd_emp - sd_theory) / sd_theory
    print(f"  max relative error of sd formula "
          f"p(1-p)/(T*pi_i): {rel.max():.3f}")
    assert rel.max() < 0.10, "the dependent-data variance formula holds"

    coverage = covered.mean() / REPS
    print(f"  95% CI coverage, averaged over entries: {coverage:.3f}")
    assert 0.93 < coverage < 0.97, "CIs should still cover ~95%"

    # ---- unbiasedness in practice -----------------------------------
    bias = np.abs(hats.mean(axis=0) - P_TRUE).max()
    print(f"  max |mean(P̂) - P| over entries: {bias:.4f} (unbiased-ish;"
          f"\n   exact unbiasedness needs fixed row counts — remark in tex)")

    # ---- pi0 becomes estimable with many paths ----------------------
    paths = [simulate(P_TRUE, 30, rng, pi0=PI0_TRUE) for _ in range(5_000)]
    _, pi0_hat, _ = fit(paths, V)
    err0 = np.abs(pi0_hat - PI0_TRUE).max()
    print(f"\npi0 from 5000 short paths: {np.round(pi0_hat, 3)} vs "
          f"{PI0_TRUE}  (max err {err0:.3f})")
    assert err0 < 0.02, "many paths make the initial distribution learnable"
    print("dependent data learned, report card verified: OK")

    # ---- a forecast from the learned chain --------------------------
    P_hat, _, _ = fit([simulate(P_TRUE, 100_000, rng)], V)
    demo = simulate(P_hat, 14, rng)
    print("\n14-day forecast from the LEARNED chain:")
    print("  " + " -> ".join(STATES[s] for s in demo))

    # ------------------------------------------------------------------
    # Animation: learning as the path streams in
    # ------------------------------------------------------------------
    long_path = simulate(P_TRUE, 200_000, rng)
    ts = np.unique(np.geomspace(20, len(long_path), 60).astype(int))
    snaps, occs, curve = [], [], []
    for t in ts:
        Ph, _, _ = fit([long_path[:t]], V)
        snaps.append(np.abs(Ph - P_TRUE))
        occs.append(np.bincount(long_path[:t], minlength=V) / t)
        curve.append(np.sqrt(np.mean((Ph - P_TRUE) ** 2)))

    fig = plt.figure(figsize=(12.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 0.9])
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
    fig.suptitle("MLE on one dependent stream: the chain is learned "
                 "as it runs", fontsize=11)

    im = ax1.imshow(snaps[0], cmap="inferno", vmin=0, vmax=0.15)
    ax1.set(xticks=range(V), yticks=range(V),
            xticklabels=STATES, yticklabels=STATES,
            title=r"$|\hat{P} - P|$")
    fig.colorbar(im, ax=ax1, shrink=.8)

    ax2.loglog(ts, curve, color="#cccccc", lw=1)
    dot2, = ax2.plot([], [], "o", color="#eb6834", ms=7)
    ref = curve[0] * (ts / ts[0]) ** -0.5
    ax2.loglog(ts, ref, "k--", lw=1, label=r"slope $-1/2$")
    ax2.set(xlabel="observed transitions T", ylabel=r"RMS $|\hat{P}-P|$",
            title="the √T law survives dependence")
    ax2.legend(frameon=False, fontsize=8)

    x = np.arange(V)
    bars = ax3.bar(x, occs[0], color="#2a78d6", alpha=.8)
    ax3.plot(x, PI, "k_", ms=24, mew=2, label="stationary pi")
    ax3.set(xticks=x, xticklabels=STATES, ylim=(0, 0.75),
            title="occupancy -> pi (ergodic thm)")
    ax3.legend(frameon=False, fontsize=8)

    for ax in (ax2, ax3):
        ax.grid(alpha=.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()

    def update(k):
        im.set_data(snaps[k])
        dot2.set_data([ts[k]], [curve[k]])
        for rect, h in zip(bars, occs[k]):
            rect.set_height(h)
        return [im, dot2, *bars]

    ani = FuncAnimation(fig, update, frames=len(ts), interval=120,
                        blit=False, repeat=True)
    plt.show()
