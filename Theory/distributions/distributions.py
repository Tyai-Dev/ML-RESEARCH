r"""Distribution zoo: sample each, verify its moments, draw its shape.

Companion to distributions.tex. For every distribution in the reference we
(1) draw a large sample with numpy, (2) check the empirical mean/variance
against the theoretical formulas, and (3) plot the pmf/pdf. If a formula
in the tex were wrong, the asserts here would catch it — the code is the
test suite of the math.

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)
N = 200_000            # large: empirical moments ~ 1/sqrt(N) accurate
TOL = 0.02             # generous 2% tolerance for the checks


def check(name: str, empirical: float, theoretical: float) -> None:
    """Assert empirical ≈ theoretical and print the comparison."""
    ok = abs(empirical - theoretical) <= TOL * max(1.0, abs(theoretical))
    print(f"{name:38s} empirical {empirical:9.4f} | theory {theoretical:9.4f}")
    assert ok, f"{name}: {empirical} vs {theoretical}"


fig, axes = plt.subplots(2, 2, figsize=(11, 7))

# ----------------------------------------------------------------------
# Bernoulli(p): support {0,1}, E = p, Var = p(1-p)
# ----------------------------------------------------------------------
p = 0.3
xb = rng.binomial(1, p, size=N)
check("Bernoulli mean = p", xb.mean(), p)
check("Bernoulli var  = p(1-p)", xb.var(), p * (1 - p))

axes[0, 0].bar([0, 1], [1 - p, p], color="#2a78d6", width=.4)
axes[0, 0].set(title=f"Bernoulli(p={p}) pmf", xticks=[0, 1])

# ----------------------------------------------------------------------
# Multinoulli / categorical(p_1..p_k): one draw, k outcomes
# E[1{x=j}] = p_j, Var[1{x=j}] = p_j (1 - p_j), Cov = -p_i p_j (i != j)
# ----------------------------------------------------------------------
probs = np.array([0.5, 0.3, 0.2])
xc = rng.choice(3, size=N, p=probs)
for j, pj in enumerate(probs):
    check(f"Multinoulli P(x={j}) = p_{j}", np.mean(xc == j), pj)
# the negative covariance between category indicators:
cov01 = np.mean((xc == 0) & (xc == 1)) - np.mean(xc == 0) * np.mean(xc == 1)
check("Multinoulli Cov(1{x=0},1{x=1}) = -p0 p1", cov01, -probs[0] * probs[1])

axes[0, 1].bar(range(3), probs, color="#1baf7a", width=.5)
axes[0, 1].set(title="Multinoulli(0.5, 0.3, 0.2) pmf", xticks=range(3))

# ----------------------------------------------------------------------
# Normal(mu, sigma^2): E = mu, Var = sigma^2; 68/95/99.7 rule
# ----------------------------------------------------------------------
mu, sigma = 2.0, 1.5
xn = rng.normal(mu, sigma, size=N)
check("Normal mean = mu", xn.mean(), mu)
check("Normal var  = sigma^2", xn.var(), sigma**2)
check("Normal P(|x-mu|<sigma) = 0.6827", np.mean(np.abs(xn - mu) < sigma), 0.6827)

grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
pdf = np.exp(-((grid - mu) ** 2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
axes[1, 0].hist(xn, bins=80, density=True, color="#e1e0d9")
axes[1, 0].plot(grid, pdf, color="#eb6834", lw=2)
axes[1, 0].set(title=f"N({mu}, {sigma}²) pdf vs sample")

# ----------------------------------------------------------------------
# Multivariate Normal(mu, Sigma): E = mu, Cov = Sigma;
# marginals are normal; linear maps of gaussians are gaussian
# ----------------------------------------------------------------------
mu2 = np.array([1.0, -1.0])
Sigma = np.array([[2.0, 1.2],
                  [1.2, 1.0]])          # symmetric positive definite
xm = rng.multivariate_normal(mu2, Sigma, size=N)
check("MVN mean[0]", xm[:, 0].mean(), mu2[0])
check("MVN mean[1]", xm[:, 1].mean(), mu2[1])
check("MVN cov[0,0]", np.cov(xm.T)[0, 0], Sigma[0, 0])
check("MVN cov[0,1]", np.cov(xm.T)[0, 1], Sigma[0, 1])

# the 2-sigma covariance ellipse: eigendecomposition of Sigma gives axes
axes[1, 1].plot(xm[:2000, 0], xm[:2000, 1], ".", ms=2, color="#c3c2b7")
evals, evecs = np.linalg.eigh(Sigma)
angle = np.linspace(0, 2 * np.pi, 100)
circle = np.stack([np.cos(angle), np.sin(angle)])
ellipse = mu2[:, None] + evecs @ (2 * np.sqrt(evals)[:, None] * circle)
axes[1, 1].plot(ellipse[0], ellipse[1], color="#e34948", lw=2, label="2σ ellipse")
axes[1, 1].set(title="MVN sample + covariance ellipse")
axes[1, 1].legend(frameon=False, fontsize=8)
axes[1, 1].set_aspect("equal")

print("\nall moment checks passed")

for ax in axes.flat:
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
