r"""Bernoulli MLE — the theoretical solution, in the language of statistics.

No machine learning here: no losses, no descent. A statistician sees data
x_1..x_n, proposes the model X ~ Bernoulli(p), and asks: which p makes
this data most probable? We MAXIMIZE the likelihood. (Turning it into a
loss to minimize is a translation the practical files perform)

Step 1 — the likelihood. Each factor p^{x_i} (1-p)^{1-x_i} is a case
split (p for a one, 1-p for a zero); with m = sum_i x_i ones,

    L(p) = p^m (1-p)^{n-m}.

Only the count m matters (SUFFICIENCY — see below): ordering is
irrelevant, m carries everything the sample knows about p.

Step 2 — the log. Strictly increasing, so the argmax stays put; products
become differentiable sums, immune to underflow:

    l(p) = m log p + (n-m) log(1-p).

The closed form. Differentiate l with the counts:
    l'(p) = m/p - (n-m)/(1-p) = 0
 => m(1-p) = (n-m) p  =>  m = np  =>  p̂ = m/n — the fraction of ones.
l''(p) = -m/p² - (n-m)/(1-p)² < 0: strictly concave, unique maximum.

----------------------------------------------------------------------
Properties of the estimator p̂ = m/n — the statistician's report card
----------------------------------------------------------------------
An estimator is a random variable (new sample => new p̂), so it has a
distribution, and we judge it by that distribution's shape:

UNBIASED.   E[p̂] = E[m]/n = np/n = p.       (m ~ Binomial(n, p))
On average, over repeated samples, p̂ hits the truth exactly.

VARIANCE.   Var(p̂) = Var(m)/n² = np(1-p)/n² = p(1-p)/n.
Shrinks like 1/n; its square root is the STANDARD ERROR
SE = sqrt(p(1-p)/n), estimated by plugging in p̂.

EFFICIENT.  The Fisher information of one Bernoulli sample is
I(p) = E[-d²/dp² log f(X;p)] = 1/(p(1-p)), so the Cramér–Rao bound says
NO unbiased estimator can beat Var >= 1/(n I(p)) = p(1-p)/n.
Our variance EQUALS the bound: p̂ is efficient — provably unimprovable
among unbiased estimators. (Full derivation: Theory/statistics.)

MSE.        MSE = Var + Bias² = p(1-p)/n + 0. For unbiased estimators
accuracy is purely a variance story.

CONSISTENT. Var -> 0 with E[p̂] = p forces p̂ -> p in probability
(this is exactly the law of large numbers, since p̂ = x̄).

ASYMPTOTICALLY NORMAL. By the CLT, sqrt(n)(p̂ - p) -> N(0, p(1-p)).
This is what buys CONFIDENCE INTERVALS: p̂ ± 1.96·SE covers the true p
about 95% of the time.

SUFFICIENT. L(p) depends on the data only through m (factorization
criterion), so m is a sufficient statistic — compressing 5000 bits into
one count loses nothing about p.

(Bonus, used later: MLE INVARIANCE — the MLE of any g(p) is g(p̂); this
is why reparameterizing through a sigmoid in the practical files is
harmless.)

This script verifies all the quantitative claims by Monte Carlo: many
replicated experiments give an empirical distribution of p̂ whose mean,
variance, and shape must match E[p̂] = p, Var = p(1-p)/n = CRLB, and the
normal curve. Figure: the fitted distribution, the log-likelihood with
its maximum, and the sampling distribution of p̂ against the CLT normal.

Run me with F5. Full derivations: bernoulli.tex.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import P_TRUE, make_data

x, rng = make_data()
n, m = len(x), int(x.sum())


def loglik(p):
    """l(p) = m log p + (n-m) log(1-p) — the whole sample enters only
    through the sufficient statistic m."""
    return m * np.log(p) + (n - m) * np.log(1 - p)


# ----------------------------------------------------------------------
# The estimate, and its report card (theory numbers vs plug-in numbers)
# ----------------------------------------------------------------------
p_hat = m / n  # the MLE: p̂ = m/n

# Brute-force check: the analytic optimum must win on a dense grid.
grid = np.linspace(0.001, 0.999, 100_000)
p_brute = grid[np.argmax(loglik(grid))]
assert abs(p_hat - p_brute) < 1e-4, "theory disagrees with brute force!"

var_theory = P_TRUE * (1 - P_TRUE) / n  # Var(p̂) = p(1-p)/n
se_plugin = np.sqrt(p_hat * (1 - p_hat) / n)  # SE with p̂ plugged in
crlb = 1 / (n * (1 / (P_TRUE * (1 - P_TRUE))))  # 1/(n I(p)) — same thing
ci_lo, ci_hi = p_hat - 1.96 * se_plugin, p_hat + 1.96 * se_plugin

print(f"n = {n},  m = sum x_i = {m}")
print(f"MLE            p̂ = m/n     : {p_hat:.6f}   (true p = {P_TRUE})")
print(f"brute-force argmax l(p)    : {p_brute:.6f}   -> agree: OK")
print()
print("estimator report card (theory):")
print(f"  bias                     : 0 (unbiased: E[p̂] = p)")
print(f"  Var(p̂) = p(1-p)/n        : {var_theory:.3e}")
print(f"  Cramér–Rao bound 1/(nI)  : {crlb:.3e}   -> equal: efficient")
print(f"  SE (plug-in)             : {se_plugin:.6f}")
print(
    f"  95% CI  p̂ ± 1.96 SE      : [{ci_lo:.4f}, {ci_hi:.4f}]"
    f"   (contains {P_TRUE}: {ci_lo <= P_TRUE <= ci_hi})"
)

# ----------------------------------------------------------------------
# Monte Carlo: repeat the whole experiment R times and check the claims.
# m ~ Binomial(n, p) exactly, so we can draw the replications directly.
# ----------------------------------------------------------------------
R = 5_000
p_hats = rng.binomial(n, P_TRUE, size=R) / n  # R independent p̂'s

mc_mean, mc_var = p_hats.mean(), p_hats.var()
print()
print(f"Monte Carlo over R = {R} replicated experiments:")
print(f"  mean of p̂  : {mc_mean:.6f}   vs p        = {P_TRUE}" f"   (unbiasedness)")
print(
    f"  var  of p̂  : {mc_var:.3e} vs p(1-p)/n = {var_theory:.3e}"
    f"   (variance = CRLB)"
)

# mean matches to ~4 sigma of the MC error; variance to a few percent
assert abs(mc_mean - P_TRUE) < 4 * np.sqrt(var_theory / R)
assert abs(mc_var - var_theory) / var_theory < 0.15
print("  empirical mean and variance match the theory: OK")

# ----------------------------------------------------------------------
# Picture: the fit, the log-likelihood, the sampling distribution
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.5, 3.8))

# (1) the distribution itself: true pmf vs fitted pmf
w = 0.35
ax1.bar(
    np.array([0, 1]) - w / 2,
    [1 - P_TRUE, P_TRUE],
    width=w,
    color="#898781",
    alpha=0.8,
    label=f"true: Bernoulli({P_TRUE})",
)
ax1.bar(
    np.array([0, 1]) + w / 2,
    [1 - p_hat, p_hat],
    width=w,
    color="#eb6834",
    alpha=0.9,
    label=f"fitted: Bernoulli({p_hat:.4f})",
)
ax1.set(xticks=[0, 1], xlabel="x", ylabel="P(X = x)", title="the model vs the fit")
ax1.legend(frameon=False, fontsize=8)

# (2) the log-likelihood, maximized (not negated — we are maximizing!)
pg = np.linspace(0.15, 0.5, 400)
ax2.plot(pg, loglik(pg), color="#2a78d6", lw=2)
ax2.axvline(p_hat, color="#eb6834", ls="--", lw=1)
ax2.plot(
    p_hat, loglik(p_hat), "o", color="#eb6834", label=f"argmax: p̂ = m/n = {p_hat:.4f}"
)
ax2.set(
    xlabel="p",
    ylabel=r"$\ell(p) = m\log p + (n-m)\log(1-p)$",
    title="the log-likelihood, maximized",
)
ax2.legend(frameon=False, fontsize=8)

# (3) the estimator IS a random variable: its sampling distribution,
#     with the CLT normal N(p, p(1-p)/n) over it
ax3.hist(
    p_hats,
    bins=40,
    density=True,
    color="#2a78d6",
    alpha=0.55,
    label=f"{R} replicated p̂'s",
)
zg = np.linspace(p_hats.min(), p_hats.max(), 300)
clt = np.exp(-((zg - P_TRUE) ** 2) / (2 * var_theory)) / np.sqrt(2 * np.pi * var_theory)
ax3.plot(zg, clt, color="#111", lw=2, label="CLT: N(p, p(1-p)/n)")
ax3.axvline(P_TRUE, color="#eb6834", ls="--", lw=1, label=f"true p = {P_TRUE}")
ax3.set(xlabel=r"$\hat{p}$", ylabel="density", title="the estimator's own distribution")
ax3.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
