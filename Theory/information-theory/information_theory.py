r"""Information theory, empirically: entropy, KL, and cross-entropy = MLE.

Companion to information-theory.tex. Four checks:

(1) The Bernoulli entropy curve H(p), maximal at p = 1/2.
(2) Gibbs' inequality KL(p || q) >= 0 with equality iff p = q, verified
    on thousands of random distribution pairs.
(3) The decomposition  H(p, q) = H(p) + KL(p || q)  (cross-entropy =
    entropy + divergence), verified numerically.
(4) The bridge to everything else in this repo: for Bernoulli data with
    empirical mean x̄, the NLL of a model q satisfies
        NLL(q) = H(p̂) + KL(p̂ || q),   p̂ = Bernoulli(x̄),
    so MINIMIZING NLL (maximum likelihood) IS minimizing the KL from the
    empirical distribution to the model. Verified on a grid of q.

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)


def entropy(p: np.ndarray) -> float:
    """H(p) = -sum p log p, in nats; 0 log 0 = 0 by continuity."""
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = sum p log(p/q); requires q > 0 wherever p > 0."""
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(p, q) = -sum p log q."""
    mask = p > 0
    return float(-(p[mask] * np.log(q[mask])).sum())


# ----------------------------------------------------------------------
# (2) Gibbs' inequality on random distribution pairs
# ----------------------------------------------------------------------
min_kl = np.inf
for _ in range(5000):
    k = rng.integers(2, 8)
    p = rng.dirichlet(np.ones(k))
    q = rng.dirichlet(np.ones(k))
    d = kl(p, q)
    min_kl = min(min_kl, d)
    assert d >= -1e-12, "Gibbs violated!"
p_same = rng.dirichlet(np.ones(5))
print("--- Gibbs' inequality ---")
print(f"5000 random pairs: min KL = {min_kl:.6f}  (never negative)")
print(f"KL(p || p) = {kl(p_same, p_same):.2e}  (equality iff p = q)")

# ----------------------------------------------------------------------
# (3) cross-entropy = entropy + KL
# ----------------------------------------------------------------------
p = rng.dirichlet(np.ones(6))
q = rng.dirichlet(np.ones(6))
lhs = cross_entropy(p, q)
rhs = entropy(p) + kl(p, q)
print("\n--- H(p,q) = H(p) + KL(p||q) ---")
print(f"H(p,q) = {lhs:.6f}   H(p)+KL = {rhs:.6f}")
assert abs(lhs - rhs) < 1e-12

# ----------------------------------------------------------------------
# (4) MLE == KL minimization (Bernoulli instance)
# ----------------------------------------------------------------------
# Sample data, form the empirical distribution p̂ = (1-x̄, x̄). For any
# model q = (1-q1, q1):  NLL(q) = -x̄ log q1 - (1-x̄) log(1-q1)
# equals H(p̂) + KL(p̂ || q). The q-dependence sits entirely in the KL
# term, so argmin NLL = argmin KL = p̂ — maximum likelihood fits the model
# to the empirical distribution in the KL sense.
x = rng.binomial(1, 0.3, size=5000).astype(float)
xbar = x.mean()
p_hat = np.array([1 - xbar, xbar])

q_grid = np.linspace(0.01, 0.99, 197)
nll = np.array([-(xbar * np.log(q) + (1 - xbar) * np.log(1 - q)) for q in q_grid])
h_plus_kl = np.array(
    [entropy(p_hat) + kl(p_hat, np.array([1 - q, q])) for q in q_grid]
)
print("\n--- MLE == KL minimization ---")
print(f"max |NLL(q) - (H + KL)(q)| over grid = {np.max(np.abs(nll - h_plus_kl)):.2e}")
print(f"argmin over the grid: q = {q_grid[np.argmin(nll)]:.4f}   (x̄ = {xbar:.4f})")
assert np.allclose(nll, h_plus_kl, atol=1e-12)
assert abs(q_grid[np.argmin(nll)] - xbar) < 0.01

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

pg = np.linspace(0.001, 0.999, 300)
H = -(pg * np.log(pg) + (1 - pg) * np.log(1 - pg))
ax1.plot(pg, H, color="#2a78d6", lw=2)
ax1.axvline(0.5, color="#898781", ls="--", lw=1)
ax1.set(xlabel="p", ylabel="H(p) [nats]",
        title="Bernoulli entropy: max log 2 at p = 1/2")

ax2.plot(q_grid, nll, color="#2a78d6", lw=3, label="NLL(q)")
ax2.plot(q_grid, h_plus_kl, color="#eb6834", lw=1, ls="--",
         label="H(p̂) + KL(p̂‖q)")
ax2.axhline(entropy(p_hat), color="#898781", ls=":", lw=1,
            label="H(p̂) — the floor")
ax2.axvline(xbar, color="#1baf7a", ls="--", lw=1, label="x̄ = argmin")
ax2.set(xlabel="model q", ylabel="nats",
        title="NLL = entropy + KL: MLE minimizes the KL")
ax2.legend(frameon=False, fontsize=8)

print("\nall checks passed")
for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
