r"""Gaussian — sample it, estimate it, descend it. One file.

Parts mirror gaussian.pdf: SAMPLE (moments), ESTIMATE (MLE for mu and
the biased 1/n variance, checked against a grid; MAP for mu under a
Normal prior — precision-weighted average, flat-prior limit, small-n
MSE win), information functions (score moments vs Fisher matrix),
DESCEND (SGD on (mu, s) with sigma = e^s, logged loop, Polyak,
autograd identity).

Run me with F5. Derivations: gaussian.pdf (gaussian.tex).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

SEED, MU, SIG, N = 7, 2.0, 1.5, 5_000
rng = np.random.default_rng(SEED)
x = rng.normal(MU, SIG, size=N)

# ---------------------------------------------------------------- 1
print("1. SAMPLE")
print(f"   mean {x.mean():.4f} vs mu = {MU},  var {x.var():.4f} "
      f"vs sigma^2 = {SIG**2:.4f}")
assert abs(x.mean() - MU) < 0.06 and abs(x.var() - SIG**2) < 0.1

# ---------------------------------------------------------------- 2
print("\n2. ESTIMATE")
mu_mle, var_mle = x.mean(), x.var()
mus = np.linspace(1.8, 2.2, 401)
vs = np.linspace(1.9, 2.6, 401)
ll = lambda m, v: -0.5 * np.log(v) - np.mean((x - m) ** 2) / (2 * v)
grid_ll = np.array([[ll(m, v) for v in vs] for m in mus])
i, j = np.unravel_index(grid_ll.argmax(), grid_ll.shape)
print(f"   MLE mu {mu_mle:.4f} var {var_mle:.4f}   grid argmax "
      f"({mus[i]:.4f}, {vs[j]:.4f})")
assert abs(mus[i] - mu_mle) < 2e-3 and abs(vs[j] - var_mle) < 2e-3

# bias of the 1/n variance, measured at n = 5
reps = rng.normal(MU, SIG, size=(20_000, 5))
factor = reps.var(axis=1).mean() / SIG**2
print(f"   E[var_mle]/sigma^2 at n=5: {factor:.3f} (theory 0.8)")
assert abs(factor - 0.8) < 0.02

# MAP for mu (known sigma): precision-weighted average
mu0, tau = 0.0, 1.0
map_mu = lambda xb, n, t: ((n / SIG**2) * xb + mu0 / t**2) \
    / (n / SIG**2 + 1 / t**2)
assert abs(map_mu(mu_mle, N, 1e9) - mu_mle) < 1e-6   # flat prior
small = rng.normal(MU, SIG, size=(20_000, 3))
mse_mle = np.mean((small.mean(axis=1) - MU) ** 2)
mse_map = np.mean((map_mu(small.mean(axis=1), 3, 2.0) - MU) ** 2)
print(f"   MAP N(0,2^2) at n=3: MSE {mse_map:.4f} vs MLE {mse_mle:.4f}")
assert mse_map < mse_mle
print(f"   at n={N}: |MAP - MLE| = "
      f"{abs(map_mu(mu_mle, N, tau) - mu_mle):.2e}")

# information functions: score mean 0, covariance = Fisher matrix
s1 = (x - MU) / SIG**2
s2 = ((x - MU) ** 2 - SIG**2) / (2 * SIG**4)
I_th = np.diag([1 / SIG**2, 1 / (2 * SIG**4)])
C = np.cov(np.stack([s1, s2]))
print(f"   score means ({s1.mean():+.4f}, {s2.mean():+.4f}) ~ 0;  "
      f"max|Cov - I| = {np.abs(C - I_th).max():.4f}")
assert abs(s1.mean()) < 0.03 and abs(s2.mean()) < 0.01
assert np.abs(C - I_th).max() < 0.02

# ---------------------------------------------------------------- 3
LR, EPOCHS = 0.02, 3
print(f"\n3. DESCEND   SGD on F(mu, s), sigma = e^s, lr={LR}")
print(f"   {'step':>12} {'mu':>8} {'sigma':>8}")
mu, s = 0.0, 0.0
schedule = np.concatenate([rng.permutation(N) for _ in range(EPOCHS)])
traj = []
for k, i in enumerate(schedule, 1):
    d = x[i] - mu
    mu -= LR * (-d) / np.exp(2 * s)
    s -= LR * (1 - d**2 / np.exp(2 * s))
    traj.append((mu, s))
    if k % 3_000 == 0:
        print(f"   {k:>6,}/{len(schedule):,} {mu:>8.4f} "
              f"{np.exp(s):>8.4f}")
tail = np.array(traj[-N:])
# average in (mu, s) coordinates, then map: exp(mean s), not mean(e^s)
mu_p, sig_p = tail[:, 0].mean(), float(np.exp(tail[:, 1].mean()))
print(f"   Polyak: mu {mu_p:.4f} sigma {sig_p:.4f}   "
      f"closed form {mu_mle:.4f} {np.sqrt(var_mle):.4f}")
assert abs(mu_p - mu_mle) < 0.05 and abs(sig_p - np.sqrt(var_mle)) < 0.05

# autograd identity on the same schedule
p = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.SGD([p], lr=LR)
x_t = torch.from_numpy(x)
traj_a = []
for i in schedule:
    opt.zero_grad()
    F = p[1] + (x_t[i] - p[0]) ** 2 / (2 * torch.exp(2 * p[1]))
    F.backward()
    opt.step()
    traj_a.append(p.detach().numpy().copy())
gap = np.abs(np.array(traj) - np.array(traj_a)).max()
print(f"   autograd identity: max gap = {gap:.2e}")
assert gap < 1e-10
print("all claims verified: OK")

# ---------------------------------------------------------------- fig
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
g = np.linspace(MU - 4 * SIG, MU + 4 * SIG, 300)
ax1.hist(x, bins=60, density=True, color="#e1e0d9")
ax1.plot(g, np.exp(-(g - mu_p) ** 2 / (2 * sig_p**2))
         / (np.sqrt(2 * np.pi) * sig_p), color="#eb6834", lw=2,
         label="SGD fit")
ax1.set(title="sample vs fitted density")
ax1.legend(frameon=False, fontsize=8)
t = np.array(traj)
ax2.plot(t[:, 0], np.exp(t[:, 1]), lw=.5, color="#3d9b35")
ax2.plot(mu_mle, np.sqrt(var_mle), "k*", ms=12, label="MLE")
ax2.set(xlabel="mu", ylabel="sigma", title="SGD path in (mu, sigma)")
ax2.legend(frameon=False, fontsize=8)
for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
