r"""Multinoulli (categorical) — bernoulli with K faces, verified.

The distribution. One draw, K outcomes, P(X = j) = p_j with p on the
simplex. In one-hot coordinates e_X the moments are

    E[e_X] = p,     Cov[e_X] = diag(p) - p p^T

— variances p_j(1-p_j) on the diagonal (each indicator is a
bernoulli!) and NEGATIVE covariances -p_i p_j off it: one category
firing forbids the others. Verified below against 200k samples,
every entry of the covariance matrix at once.

The MLE (the Lagrange argument, derived in multinoulli.tex and used
by every softmax model in this repo): counts, normalized —
p̂_j = n_j / n. Verified two ways: it beats 500 random challenger
distributions (Gibbs), and its per-coordinate variance matches the
report-card formula p_j(1-p_j)/n over 1000 replications.

Run me with F5. Derivations: multinoulli.tex.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)
P = np.array([0.5, 0.3, 0.2])
K, N = len(P), 200_000

# ---- moments: the full covariance matrix in one check ----------------
X = rng.choice(K, size=N, p=P)
E = np.eye(K)[X]                              # one-hot samples (N, K)
mean_err = np.abs(E.mean(axis=0) - P).max()
cov_theory = np.diag(P) - np.outer(P, P)
cov_err = np.abs(np.cov(E.T, bias=True) - cov_theory).max()
print(f"mean:  max |empirical - p|                = {mean_err:.4f}")
print(f"cov:   max |empirical - (diag(p) - pp^T)| = {cov_err:.4f}")
assert mean_err < 5e-3 and cov_err < 5e-3
print("one category firing forbids the others: Cov(i,j) = -p_i p_j: OK")

# ---- MLE: counts win, and at the promised rate -----------------------
n = 2_000
p_hat = np.bincount(X[:n], minlength=K) / n

def nll(q, counts):
    return -np.sum(counts * np.log(q))

counts = np.bincount(X[:n], minlength=K)
worst = min(nll(rng.dirichlet(np.ones(K)), counts) - nll(p_hat, counts)
            for _ in range(500))
print(f"Gibbs: min excess NLL over 500 challengers = {worst:.3f} >= 0")
assert worst >= 0

REPS = 1_000
hats = np.array([np.bincount(rng.choice(K, size=n, p=P),
                             minlength=K) / n for _ in range(REPS)])
sd_err = np.abs(hats.std(axis=0)
                - np.sqrt(P * (1 - P) / n)) / np.sqrt(P * (1 - P) / n)
print(f"report card: sd of p̂_j matches sqrt(p(1-p)/n) to "
      f"{sd_err.max():.1%} over {REPS} replications")
assert sd_err.max() < 0.10
print("all claims verified: OK")

# ---- picture ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
w = 0.35
ax1.bar(np.arange(K) - w / 2, P, w, color="#898781", label="true p")
ax1.bar(np.arange(K) + w / 2, p_hat, w, color="#eb6834",
        label=f"MLE (n={n})")
ax1.set(xticks=range(K), xlabel="category", ylabel="probability",
        title="counts, normalized")
ax1.legend(frameon=False, fontsize=8)
im = ax2.imshow(cov_theory, cmap="coolwarm",
                vmin=-np.abs(cov_theory).max(),
                vmax=np.abs(cov_theory).max())
for i in range(K):
    for j in range(K):
        ax2.text(j, i, f"{cov_theory[i, j]:+.2f}", ha="center",
                 va="center", fontsize=9)
ax2.set(xticks=range(K), yticks=range(K),
        title=r"Cov = diag(p) $-$ pp$^T$")
fig.colorbar(im, ax=ax2, shrink=.8)
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)
ax1.grid(alpha=.3)
fig.tight_layout()
plt.show()
