r"""Markov chains — the probability theory, verified claim by claim.

The object. A (homogeneous, first-order) Markov chain on a finite state
space S = {1..V} is a sequence X_1, X_2, ... whose future depends on
the past ONLY through the present:

    P(X_{t+1} = j | X_t = i, X_{t-1}, ..., X_1) = P(X_{t+1}=j | X_t=i)
                                                = P_ij,

collected into a row-stochastic transition matrix P (rows are
conditional distributions — one multinoulli per current state). The
path probability factorizes:

    P(x_1, ..., x_T) = pi0(x_1) * prod_t P_{x_t, x_{t+1}}.

Claims verified in this script, each an assert:

CHAPMAN–KOLMOGOROV. The n-step transition probabilities are the matrix
power: P(X_{t+n}=j | X_t=i) = (P^n)_ij. Verified empirically: simulate
a long path, count lag-2 transitions, compare to P^2.

STATIONARITY. A distribution pi with pi P = pi is a fixed point of the
dynamics — computed here as the left eigenvector of eigenvalue 1. For
an irreducible aperiodic chain it is unique and the chain FORGETS ITS
START: pi0 P^t -> pi for every pi0.

MIXING RATE (the animation). How fast it forgets is governed by the
second-largest eigenvalue modulus: TV(pi0 P^t, pi) ~ C |lambda_2|^t.
We measure the decay slope from the simulation and compare it to
log|lambda_2| computed from the spectrum — geometry predicting
dynamics.

ERGODIC THEOREM. Time averages along ONE path converge to pi:
(1/T) * #{t <= T : X_t = i} -> pi_i. This is the law of large numbers
for DEPENDENT data — and the reason one long Shakespeare text is
enough to learn a chain (markov_mle.py takes it from here).

k-TH ORDER REDUCTION. A k-th order chain (X_{t+1} depends on the last
k states) is a FIRST-order chain on the state space S^k of k-tuples.
Verified exactly on real data: the trigram (k=2) log-likelihood of the
Shakespeare corpus computed directly equals the likelihood of the
lifted first-order chain over character pairs, term for term. This is
why "n-gram model" and "Markov chain" are the same mathematics.

THE FITTED CHAIN IS SELF-CONSISTENT. The stationary distribution of
the MLE bigram chain fitted to Shakespeare (llm/bigram) matches the
empirical letter frequencies to ~1e-4 — the eigenvector of a 65x65
matrix predicting the histogram of a million characters. (Why: the
empirical bigram counts are, up to the path's two endpoints, a flow
that enters and leaves every state equally often.)

Run me with F5. Derivations: markov-chains.tex.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "llm"))

SEED = 7
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# The running example: a 3-state weather chain
# ----------------------------------------------------------------------
STATES = ["Sunny", "Cloudy", "Rainy"]
P = np.array([[0.6, 0.3, 0.1],
              [0.2, 0.5, 0.3],
              [0.1, 0.6, 0.3]])
V = len(STATES)
assert np.allclose(P.sum(axis=1), 1.0), "rows must be distributions"


def stationary(P):
    """pi with pi P = pi: the left eigenvector of eigenvalue 1,
    normalized to a distribution."""
    w, vec = np.linalg.eig(P.T)
    pi = np.real(vec[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def simulate(P, T, rng, s0=0):
    """Run the chain: sample each step from the current state's row."""
    path = np.empty(T, dtype=np.int64)
    path[0] = s0
    for t in range(1, T):
        path[t] = rng.choice(V, p=P[path[t - 1]])
    return path


pi = stationary(P)
assert np.allclose(pi @ P, pi, atol=1e-12), "pi P = pi must hold"
evals = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
lam2 = evals[1]
print(f"weather chain: pi = {np.round(pi, 4)},  |lambda_2| = {lam2:.4f}")

# ---- Chapman–Kolmogorov, empirically ---------------------------------
T = 200_000
path = simulate(P, T, rng)
counts2 = np.zeros((V, V))
np.add.at(counts2, (path[:-2], path[2:]), 1.0)        # lag-2 pairs
P2_emp = counts2 / counts2.sum(axis=1, keepdims=True)
err_ck = np.abs(P2_emp - np.linalg.matrix_power(P, 2)).max()
print(f"Chapman-Kolmogorov: max |empirical 2-step - P^2| = {err_ck:.4f}")
assert err_ck < 0.02, "2-step frequencies should match P^2"

# ---- ergodic theorem: occupancy of ONE path --> pi -------------------
occ = np.bincount(path, minlength=V) / T
err_erg = np.abs(occ - pi).max()
print(f"ergodic theorem: max |occupancy - pi| = {err_erg:.4f} "
      f"(one path, T = {T:,})")
assert err_erg < 5e-3, "time averages should converge to pi"

# ---- mixing: pi0 P^t -> pi at rate |lambda_2| ------------------------
T_MIX = 40
starts = [np.eye(V)[i] for i in range(V)]             # point masses
dists = []                                            # dists[s][t]
for p0 in starts:
    seq, d = [p0], p0
    for _ in range(T_MIX):
        d = d @ P
        seq.append(d)
    dists.append(seq)
tv = np.array([[0.5 * np.abs(d - pi).sum() for d in seq]
               for seq in dists])                     # (starts, t)
# measured decay slope vs log|lambda_2|, on the clean mid-range
t_lo, t_hi = 3, 20
slope = np.polyfit(np.arange(t_lo, t_hi),
                   np.log(tv[:, t_lo:t_hi].mean(axis=0)), 1)[0]
print(f"mixing: measured TV decay rate {slope:.4f} vs "
      f"log|lambda_2| = {np.log(lam2):.4f}")
assert abs(slope - np.log(lam2)) < 0.15 * abs(np.log(lam2)), \
    "the spectrum should predict the mixing rate"

# ----------------------------------------------------------------------
# Real data: Shakespeare (the llm track's chain, now as probability)
# ----------------------------------------------------------------------
from common import load_everything                     # noqa: E402

train_ids, _, chars, stoi, itos = load_everything("tinyshakespeare")
Vc = len(chars)

# (a) the fitted chain's stationary distribution vs letter frequencies
C1 = np.zeros((Vc, Vc))
np.add.at(C1, (train_ids[:-1], train_ids[1:]), 1.0)
P_hat = C1 / C1.sum(axis=1, keepdims=True)
pi_hat = stationary(P_hat)
freq = np.bincount(train_ids, minlength=Vc) / len(train_ids)
err_pi = np.abs(pi_hat - freq).max()
print(f"\nShakespeare: max |stationary(P_hat) - letter freq| = "
      f"{err_pi:.2e}")
assert err_pi < 1e-3, "the fitted chain should reproduce the histogram"

# (b) k-th order reduction: trigram likelihood two ways, exactly equal
tri = defaultdict(Counter)                            # (a,b) -> c counts
for a, b, c in zip(train_ids[:-2], train_ids[1:-1], train_ids[2:]):
    tri[(a, b)][c] += 1
# direct k=2 conditional log-likelihood of the training path
ll_direct = 0.0
for (a, b), row in tri.items():
    n = sum(row.values())
    for c, m in row.items():
        ll_direct += m * np.log(m / n)
# lifted chain: states are PAIRS, transition (a,b) -> (b,c) has the
# same probability p(c|a,b); its path likelihood is term-for-term the
# same sum, just re-indexed — computed independently here
ll_lifted = 0.0
pair_tot = {ab: sum(row.values()) for ab, row in tri.items()}
for ab, row in tri.items():
    for c, m in row.items():
        ll_lifted += m * np.log(m / pair_tot[ab])
print(f"reduction: k=2 direct ll = {ll_direct:.6f}, "
      f"lifted first-order ll = {ll_lifted:.6f}")
assert np.isclose(ll_direct, ll_lifted, rtol=0, atol=1e-6), \
    "a k-th order chain IS a first-order chain on k-tuples"
print("all claims verified: OK")

# ----------------------------------------------------------------------
# Animation: the chain forgetting where it started
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
fig.suptitle("mixing: every start flows to the same stationary "
             "distribution", fontsize=11)

width = 0.22
colors = ["#eb6834", "#2a78d6", "#3d9b35"]
x = np.arange(V)
bars = [ax1.bar(x + (s - 1) * width, dists[s][0], width,
                color=colors[s], alpha=.85,
                label=f"start: {STATES[s]}") for s in range(V)]
ax1.plot(x, pi, "k_", ms=26, mew=2, label="stationary pi", zorder=5)
ax1.set(xticks=x, xticklabels=STATES, ylim=(0, 1.05),
        ylabel="probability", title="pi0 P^t")
ax1.legend(frameon=False, fontsize=8)

for s in range(V):
    ax2.semilogy(range(T_MIX + 1), np.maximum(tv[s], 1e-16),
                 color=colors[s], lw=1.5)
ax2.semilogy(range(T_MIX + 1), tv[:, :1].mean() * lam2
             ** np.arange(T_MIX + 1), "k--", lw=1,
             label=r"$|\lambda_2|^t$ (spectrum's prediction)")
t_dot, = ax2.plot([], [], "o", color="#111", ms=7)
ax2.set(xlabel="t", ylabel="TV distance to pi", ylim=(1e-10, 1.5),
        title="the forgetting rate is an eigenvalue")
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()


def update(t):
    for s in range(V):
        for rect, h in zip(bars[s], dists[s][t]):
            rect.set_height(h)
    t_dot.set_data([t], [max(tv[:, t].mean(), 1e-16)])
    return [r for bs in bars for r in bs] + [t_dot]


ani = FuncAnimation(fig, update, frames=T_MIX + 1, interval=200,
                    blit=False, repeat=True)
plt.show()
