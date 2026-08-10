r"""Learning theory, empirically: LLN, Hoeffding, Glivenko–Cantelli.

Companion to learning-theory.tex. Three visual experiments:

(1) Law of large numbers — running means of Bernoulli(0.3) draws converge
    to p along individual sample paths.
(2) Hoeffding — the empirical probability that |x̄ - p| > eps, over many
    repetitions, sits UNDER the bound 2 exp(-2 n eps^2) at every n.
(3) Glivenko–Cantelli — the empirical CDF converges to the true CDF
    UNIFORMLY: sup_t |F_n(t) - F(t)| -> 0, at the DKW rate.

These are the three pillars under 'learning works': averages converge
(LLN), fast and quantifiably (Hoeffding), and uniformly over function
classes (GC) — uniformity being what lets us trust the empirical-risk
minimizer chosen from a whole class rather than one fixed hypothesis.

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

# ----------------------------------------------------------------------
# (1) LLN: running means of Bernoulli(p) paths
# ----------------------------------------------------------------------
p, n = 0.3, 10_000
paths = rng.binomial(1, p, size=(8, n))
running = np.cumsum(paths, axis=1) / np.arange(1, n + 1)

for path in running:
    ax1.plot(path, lw=.8, alpha=.7)
ax1.axhline(p, color="#0b0b0b", ls="--", lw=1)
ax1.set(xscale="log", xlabel="n (log)", ylabel="running mean",
        title="LLN: sample paths of x̄ₙ → p")

# every path should be near p at the end (strong LLN in action)
assert np.all(np.abs(running[:, -1] - p) < 0.02)

# ----------------------------------------------------------------------
# (2) Hoeffding: empirical tail vs the bound  2 exp(-2 n eps^2)
# ----------------------------------------------------------------------
eps, trials = 0.05, 20_000
ns = np.array([25, 50, 100, 200, 400, 800, 1600])
empirical = []
for n_i in ns:
    means = rng.binomial(n_i, p, size=trials) / n_i    # x̄ over `trials` runs
    empirical.append(np.mean(np.abs(means - p) > eps))
empirical = np.array(empirical)
bound = 2 * np.exp(-2 * ns * eps**2)

ax2.plot(ns, empirical, "o-", color="#2a78d6", label="empirical P(|x̄-p|>ε)")
ax2.plot(ns, np.minimum(bound, 1), "s--", color="#eb6834",
         label="Hoeffding bound")
ax2.set(xscale="log", yscale="log", xlabel="n (log)", ylabel="probability (log)",
        title=f"Hoeffding, ε={eps}: bound holds at every n")
ax2.legend(frameon=False, fontsize=8)

# the bound must actually bound (up to Monte-Carlo noise when both ~0)
assert np.all(empirical <= np.minimum(bound, 1) + 3 / np.sqrt(trials))

# ----------------------------------------------------------------------
# (3) Glivenko–Cantelli: sup |F_n - F| -> 0 (with the DKW envelope)
# ----------------------------------------------------------------------
# True distribution: standard normal. For each n, compute the empirical
# CDF's worst-case (sup over t) deviation from the true CDF.
from math import erf

def normal_cdf(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(erf)(t / np.sqrt(2)))

ns_gc = np.array([10, 30, 100, 300, 1000, 3000, 10000])
sups = []
for n_i in ns_gc:
    sample = np.sort(rng.normal(size=n_i))
    # sup |F_n - F| is attained at a jump of F_n: check both sides of each
    F = normal_cdf(sample)
    upper = np.arange(1, n_i + 1) / n_i - F     # just after each jump
    lower = F - np.arange(0, n_i) / n_i         # just before each jump
    sups.append(max(upper.max(), lower.max()))
sups = np.array(sups)

# DKW inequality: P(sup > eps) <= 2 exp(-2 n eps^2); the eps making the
# right side 0.05 is a 95% envelope for the sup:
envelope = np.sqrt(np.log(2 / 0.05) / (2 * ns_gc))

ax3.plot(ns_gc, sups, "o-", color="#1baf7a", label="sup |Fₙ - F| (one run)")
ax3.plot(ns_gc, envelope, "--", color="#eb6834", label="DKW 95% envelope")
ax3.set(xscale="log", yscale="log", xlabel="n (log)", ylabel="sup deviation (log)",
        title="Glivenko–Cantelli: uniform convergence of Fₙ")
ax3.legend(frameon=False, fontsize=8)

assert sups[-1] < 0.02  # uniform closeness at n = 10^4

print("LLN        : all 8 paths within 0.02 of p at n = 10^4")
print(f"Hoeffding  : empirical tail <= bound at all n (e.g. n=400: "
      f"{empirical[ns == 400][0]:.4f} <= {bound[ns == 400][0]:.4f})")
print(f"GC         : sup|F_n - F| at n=10^4 is {sups[-1]:.4f}")
print("all checks passed")

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
