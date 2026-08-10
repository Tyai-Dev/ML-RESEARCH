r"""Gaussian (1D normal) maximum-likelihood estimation: closed form, GD, SGD, autograd.

Problem
-------
Given i.i.d. samples x_1, ..., x_n ~ N(mu, sigma^2), estimate (mu, sigma).
The density and the average negative log-likelihood (NLL) are

    f(x; mu, sigma) = exp(-(x - mu)^2 / (2 sigma^2)) / sqrt(2 pi sigma^2),

    NLL(mu, sigma) = (1/2) log(2 pi) + log sigma
                     + (1/(2 sigma^2)) (1/n) sum_i (x_i - mu)^2.

The data enters only through the first two empirical moments
(sum x_i, sum x_i^2) — a two-dimensional sufficient statistic.

The four routes
---------------
(1) Theoretical solution (derived in gaussian.tex): setting both partial
    derivatives to zero gives
        mu_hat    = x̄                       (the sample mean)
        sigma_hat^2 = (1/n) sum (x_i - x̄)^2  (the 1/n variance — the MLE is
                                              the *biased* estimator!)
(2) Gradient descent on the NLL — full gradient, every step.
(3) SGD — one sample per step, unbiased gradient estimate.
(4) The same SGD with PyTorch autograd, identical sample schedule:
    trajectories must coincide if autograd matches the hand derivation.

Coordinates: mu is already unconstrained; sigma > 0 is not, so exactly as
with Bernoulli's p = sigmoid(t) we evaluate the SAME NLL at sigma = e^s,

    F(mu, s) = NLL(mu, e^s),      (mu, s) in R^2 .

Chain rule (derivations in gaussian.tex):
    dF/dmu = (mu - x̄) / e^{2s}
    dF/ds  = 1 - (1/n) sum (x_i - mu)^2 / e^{2s}
Setting both to zero reproduces route (1) — a good consistency check.

Run me with F5. Companion derivations: gaussian.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------
MU_TRUE, SIGMA_TRUE = 2.0, 1.5   # the parameters we pretend not to know
N = 5_000
SEED = 7

GD_LR, GD_STEPS = 0.1, 400       # gradient descent
SGD_LR, SGD_EPOCHS = 0.01, 3     # stochastic gradient descent

rng = np.random.default_rng(SEED)
x = rng.normal(MU_TRUE, SIGMA_TRUE, size=N)


# ----------------------------------------------------------------------
# (1) Theoretical solution
# ----------------------------------------------------------------------
# d NLL/d mu = (mu - x̄)/sigma^2 = 0            =>  mu = x̄  (regardless of sigma)
# d NLL/d sigma = 1/sigma - (1/n) sum (x_i-mu)^2 / sigma^3 = 0
#   => sigma^2 = (1/n) sum (x_i - x̄)^2.
# Note the 1/n: the MLE of the variance is the BIASED estimator
# (E[sigma_hat^2] = (n-1)/n sigma^2); Bessel's 1/(n-1) correction is a
# different estimator, chosen for unbiasedness, not by maximum likelihood.
mu_closed = x.mean()
sigma_closed = np.sqrt(np.mean((x - mu_closed) ** 2))   # 1/n, not 1/(n-1)


# ----------------------------------------------------------------------
# (2) Gradient descent — full gradient, every step
# ----------------------------------------------------------------------
# Minimize F(mu, s) = NLL(mu, e^s) over the unconstrained plane. The full
# gradient needs the data only through its empirical moments:
#     dF/dmu = (mu - x̄) / e^{2s}
#     dF/ds  = 1 - mean((x - mu)^2) / e^{2s}
def gradient_descent() -> list[tuple[float, float]]:
    mu, s = 0.0, 0.0     # start at N(0, 1) — an uninformed guess
    trajectory = []
    for _ in range(GD_STEPS):
        var = np.exp(2 * s)
        g_mu = (mu - x.mean()) / var
        g_s = 1.0 - np.mean((x - mu) ** 2) / var
        mu -= GD_LR * g_mu
        s -= GD_LR * g_s
        trajectory.append((mu, np.exp(s)))
    return trajectory


# ----------------------------------------------------------------------
# Shared sample schedule for (3) and (4)
# ----------------------------------------------------------------------
schedule = np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


# ----------------------------------------------------------------------
# (3) SGD — one sample per step, gradients by hand
# ----------------------------------------------------------------------
# Per sample i the NLL is  F_i(mu, s) = s + (x_i - mu)^2 / (2 e^{2s}) + c,
# whose gradient (unbiased estimate of the full one) is
#     dF_i/dmu = (mu - x_i) / e^{2s}
#     dF_i/ds  = 1 - (x_i - mu)^2 / e^{2s}
def sgd_manual() -> list[tuple[float, float]]:
    mu, s = 0.0, 0.0
    trajectory = []
    for i in schedule:
        var = np.exp(2 * s)
        g_mu = (mu - x[i]) / var
        g_s = 1.0 - (x[i] - mu) ** 2 / var
        mu -= SGD_LR * g_mu
        s -= SGD_LR * g_s
        trajectory.append((mu, np.exp(s)))
    return trajectory


# ----------------------------------------------------------------------
# (4) The same SGD, gradients by PyTorch autograd
# ----------------------------------------------------------------------
# The loss is written EXACTLY as F_i above (constants dropped — they do
# not affect gradients); autograd differentiates the composition for us.
def sgd_torch() -> list[tuple[float, float]]:
    x_t = torch.from_numpy(x)                       # float64
    mu = torch.zeros((), dtype=torch.float64, requires_grad=True)
    s = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([mu, s], lr=SGD_LR)
    trajectory = []
    for i in schedule:
        optimizer.zero_grad()
        loss = s + (x_t[i] - mu) ** 2 / (2 * torch.exp(2 * s))
        loss.backward()
        optimizer.step()
        trajectory.append((mu.item(), float(torch.exp(s))))
    return trajectory


traj_gd = gradient_descent()
traj_sgd = sgd_manual()
traj_torch = sgd_torch()

# Proof by computation: autograd's gradients ARE the hand-derived ones.
assert np.allclose(traj_sgd, traj_torch, atol=1e-10), \
    "autograd disagrees with the hand-derived gradients"

mu_gd, sigma_gd = traj_gd[-1]
# SGD hovers in a noise ball — average the final epoch (Polyak–Ruppert).
mu_sgd = float(np.mean([m for m, _ in traj_sgd[-N:]]))
sigma_sgd = float(np.mean([sg for _, sg in traj_sgd[-N:]]))


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
print(f"true (mu, sigma)          : ({MU_TRUE}, {SIGMA_TRUE})")
print(f"(1) closed form           : ({mu_closed:.6f}, {sigma_closed:.6f})")
print(f"(2) GD, final iterate     : ({mu_gd:.6f}, {sigma_gd:.6f})")
print(f"(3) SGD, Polyak average   : ({mu_sgd:.6f}, {sigma_sgd:.6f})")
print(f"max |traj(3) - traj(4)|   : {np.max(np.abs(np.array(traj_sgd) - np.array(traj_torch))):.2e}")
print("autograd == hand gradients (allclose): OK")
print(f"note: MLE variance uses 1/n — np.var default {np.var(x):.6f} matches, "
      f"Bessel 1/(n-1) gives {np.var(x, ddof=1):.6f}")


# ----------------------------------------------------------------------
# Picture: data + fitted density, GD path, SGD noise ball (mu coordinate)
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

grid = np.linspace(x.min(), x.max(), 300)
pdf = np.exp(-((grid - mu_closed) ** 2) / (2 * sigma_closed**2)) \
    / (np.sqrt(2 * np.pi) * sigma_closed)
ax1.hist(x, bins=50, density=True, color="#e1e0d9", edgecolor="white")
ax1.plot(grid, pdf, color="#2a78d6", lw=2, label="MLE fit")
ax1.set(xlabel="x", ylabel="density", title="sample + fitted N(mu, sigma)")
ax1.legend(frameon=False, fontsize=8)

ax2.plot([m for m, _ in traj_gd], color="#2a78d6", lw=2, label="mu")
ax2.plot([sg for _, sg in traj_gd], color="#1baf7a", lw=2, label="sigma")
ax2.axhline(mu_closed, color="#eb6834", ls="--", lw=1)
ax2.axhline(sigma_closed, color="#eb6834", ls="--", lw=1)
ax2.set(xlabel="GD step", title="GD: both coordinates converge")
ax2.legend(frameon=False, fontsize=8)

ax3.plot([m for m, _ in traj_sgd], color="#1baf7a", lw=.6, label="SGD mu (hand)")
ax3.plot([m for m, _ in traj_torch], color="#e87ba4", lw=.6, ls=":",
         label="SGD mu (autograd)")
ax3.axhline(mu_closed, color="#eb6834", ls="--", lw=1, label="closed form")
ax3.set(xlabel="SGD step", title="SGD: noise ball around the optimum")
ax3.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
