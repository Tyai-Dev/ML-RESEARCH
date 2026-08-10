r"""Estimator theory, empirically: bias, MSE decomposition, Cramér–Rao.

Companion to statistics.tex. Three experiments:

(1) The MLE variance estimator (1/n) is biased by exactly (n-1)/n, and
    Bessel's 1/(n-1) fixes it — measured over many repetitions.
(2) MSE = bias^2 + variance, verified numerically for both estimators.
(3) The Cramér–Rao bound: Var(p_hat) for the Bernoulli MLE equals
    p(1-p)/n = 1/(n I(p)) — the MLE is efficient (achieves the bound).

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# (1) + (2): bias and MSE of the two variance estimators, N(0, sigma^2)
# ----------------------------------------------------------------------
SIGMA2_TRUE = 4.0          # true variance
n, trials = 10, 200_000    # small n makes the bias visible

samples = rng.normal(0.0, np.sqrt(SIGMA2_TRUE), size=(trials, n))
xbar = samples.mean(axis=1, keepdims=True)
ss = ((samples - xbar) ** 2).sum(axis=1)

var_mle = ss / n           # the 1/n estimator (the MLE)
var_bessel = ss / (n - 1)  # the 1/(n-1) estimator (unbiased)

# Theory:  E[var_mle] = (n-1)/n sigma^2 ;  E[var_bessel] = sigma^2
print("--- bias of the variance estimators (n = 10) ---")
print(f"E[1/n estimator]     : {var_mle.mean():.4f}"
      f"   theory (n-1)/n*sig2 = {(n-1)/n*SIGMA2_TRUE:.4f}")
print(f"E[1/(n-1) estimator] : {var_bessel.mean():.4f}"
      f"   theory  sigma^2     = {SIGMA2_TRUE:.4f}")
assert abs(var_mle.mean() - (n - 1) / n * SIGMA2_TRUE) < 0.02
assert abs(var_bessel.mean() - SIGMA2_TRUE) < 0.02

# MSE decomposition: MSE = bias^2 + variance, for each estimator.
# Fun fact this exposes: the biased MLE has LOWER MSE here — unbiasedness
# is a design choice, not a free lunch.
print("\n--- MSE = bias^2 + variance ---")
for name, est in [("1/n (MLE)", var_mle), ("1/(n-1)", var_bessel)]:
    bias = est.mean() - SIGMA2_TRUE
    variance = est.var()
    mse_direct = np.mean((est - SIGMA2_TRUE) ** 2)
    print(f"{name:10s}: bias²+var = {bias**2 + variance:.4f}"
          f" | direct MSE = {mse_direct:.4f}")
    assert abs((bias**2 + variance) - mse_direct) < 1e-6 * max(1, mse_direct)

# ----------------------------------------------------------------------
# (3) Cramér–Rao: the Bernoulli MLE achieves the bound
# ----------------------------------------------------------------------
# Fisher information of Bernoulli(p) per sample: I(p) = 1/(p(1-p)).
# CRLB for unbiased estimators: Var(p_hat) >= 1/(n I(p)) = p(1-p)/n.
# The MLE p_hat = x̄ is unbiased with Var = p(1-p)/n: efficient.
p_true, trials = 0.3, 200_000
ns = np.array([10, 30, 100, 300, 1000])
emp_var, crlb = [], []
for n_i in ns:
    p_hats = rng.binomial(n_i, p_true, size=trials) / n_i
    emp_var.append(p_hats.var())
    crlb.append(p_true * (1 - p_true) / n_i)
emp_var, crlb = np.array(emp_var), np.array(crlb)

print("\n--- Cramér–Rao for the Bernoulli MLE ---")
for n_i, ev, cr in zip(ns, emp_var, crlb):
    print(f"n={n_i:5d}:  Var(p̂) = {ev:.6f}   CRLB = {cr:.6f}")
assert np.allclose(emp_var, crlb, rtol=0.05)
print("MLE variance == CRLB at every n: efficient estimator")

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

ax1.hist(var_mle, bins=100, density=True, alpha=.6, color="#2a78d6",
         label="1/n (MLE)")
ax1.hist(var_bessel, bins=100, density=True, alpha=.6, color="#eb6834",
         label="1/(n-1) (Bessel)")
ax1.axvline(SIGMA2_TRUE, color="#0b0b0b", ls="--", lw=1, label="true σ²")
ax1.set(xlabel="estimate of σ²", ylabel="density",
        title=f"sampling distributions at n={n}: bias visible")
ax1.legend(frameon=False, fontsize=8)

ax2.loglog(ns, emp_var, "o-", color="#1baf7a", label="empirical Var(p̂)")
ax2.loglog(ns, crlb, "s--", color="#eb6834", label="CRLB = p(1-p)/n")
ax2.set(xlabel="n (log)", ylabel="variance (log)",
        title="Bernoulli MLE sits on the Cramér–Rao bound")
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
