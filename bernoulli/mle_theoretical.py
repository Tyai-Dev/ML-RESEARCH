r"""Bernoulli MLE — the theoretical solution.

From the likelihood to the estimator, three optimizer-preserving steps:

Step 1 — the likelihood. Each factor p^{x_i} (1-p)^{1-x_i} is a case
split (p for a one, 1-p for a zero); with m = sum_i x_i ones,

    L(p) = p^m (1-p)^{n-m}.

Only the count m matters (sufficiency): ordering is irrelevant.

Step 2 — the log. Strictly increasing, so the argmax stays put; powers
become a differentiable sum, immune to underflow:

    l(p) = m log p + (n-m) log(1-p).

Step 3 — normalize and negate (rescaling and sign don't move the
optimizer): NLL(p) = -(1/n) l(p) = -x̄ log p - (1-x̄) log(1-p).

The closed form. Differentiate l with the counts:
    l'(p) = m/p - (n-m)/(1-p) = 0
 => m(1-p) = (n-m) p  =>  m = np  =>  p̂ = m/n — the fraction of ones.
l''(p) = -m/p² - (n-m)/(1-p)² < 0: strictly concave, unique maximum.

This script verifies the theory numerically: the analytic p̂ = m/n must
coincide with a brute-force minimization of the NLL over a dense grid.
Figure: likelihood and NLL landscapes with the optimum marked on both.

Run me with F5. Full derivations: bernoulli.tex.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import P_TRUE, nll_of_p, make_data

x, _ = make_data()
n, m = len(x), int(x.sum())

# The theoretical solution — one line, backed by the derivation above.
p_closed = m / n                                     # = x.mean()

# Brute-force check: the analytic optimum must win on a dense grid.
grid = np.linspace(0.001, 0.999, 100_000)
p_brute = grid[np.argmin(nll_of_p(grid, x.mean()))]

print(f"true p                 : {P_TRUE}")
print(f"analytic  p̂ = m/n      : {p_closed:.6f}   (m = {m}, n = {n})")
print(f"brute-force argmin NLL : {p_brute:.6f}")
assert abs(p_closed - p_brute) < 1e-4, "theory disagrees with brute force!"
print("closed form == numerical optimum: OK")

# ----------------------------------------------------------------------
# Picture: both landscapes, one optimum
# ----------------------------------------------------------------------
pg = np.linspace(0.01, 0.99, 400)
# log-likelihood per sample (the likelihood itself underflows at n=5000 —
# exactly the Step-2 point)
loglik = -nll_of_p(pg, x.mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

ax1.plot(pg, loglik, color="#2a78d6", lw=2)
ax1.axvline(p_closed, color="#eb6834", ls="--", lw=1)
ax1.plot(p_closed, -nll_of_p(p_closed, x.mean()), "o", color="#eb6834",
         label=f"p̂ = m/n = {p_closed:.4f}")
ax1.set(xlabel="p", ylabel="log-likelihood / n",
        title="the log-likelihood: strictly concave")
ax1.legend(frameon=False, fontsize=8)

ax2.plot(pg, nll_of_p(pg, x.mean()), color="#2a78d6", lw=2)
ax2.axvline(P_TRUE, color="#898781", ls="--", lw=1, label=f"true p = {P_TRUE}")
ax2.plot(p_closed, nll_of_p(p_closed, x.mean()), "o", color="#eb6834",
         label=f"p̂ = {p_closed:.4f}")
ax2.set(xlabel="p", ylabel="NLL",
        title="the NLL: same optimum, ML convention")
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
