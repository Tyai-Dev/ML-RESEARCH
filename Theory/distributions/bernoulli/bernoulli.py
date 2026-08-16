r"""Bernoulli — sample it, estimate it, descend it. One file.

Three parts, mirroring bernoulli.pdf:
  1. SAMPLE      draw from the distribution, verify both moments.
  2. ESTIMATE    the closed-form MLE p̂ = m/n (checked against brute
                 force, with its CI), and MAP with a Beta prior —
                 pseudo-counts, shrinkage, and the measured small-n
                 MSE win; prior drowned as n grows.
  3. DESCEND     the same MLE as an ML problem: NLL through the
                 sigmoid coordinate, F'(t) = sigmoid(t) - x̄, a logged
                 SGD loop into the noise ball, Polyak average out of
                 it, autograd identity as the proof.

Run me with F5. Derivations: bernoulli.pdf (bernoulli.tex).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

SEED, P_TRUE, N = 7, 0.3, 5_000
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# 1. SAMPLE — and verify the moments
# ----------------------------------------------------------------------
x = rng.binomial(1, P_TRUE, size=N).astype(np.float64)
print("1. SAMPLE")
print(f"   mean {x.mean():.4f} vs p        = {P_TRUE}")
print(f"   var  {x.var():.4f} vs p(1-p)   = {P_TRUE * (1 - P_TRUE):.4f}")
assert abs(x.mean() - P_TRUE) < 0.02
assert abs(x.var() - P_TRUE * (1 - P_TRUE)) < 0.02

# exponential family: A(theta) = log(1+e^theta) generates the moments
theta, h = np.log(P_TRUE / (1 - P_TRUE)), 1e-5
A = lambda t: np.log1p(np.exp(t))
A1 = (A(theta + h) - A(theta - h)) / (2 * h)
A2 = (A(theta + h) - 2 * A(theta) + A(theta - h)) / h ** 2
assert abs(A1 - P_TRUE) < 1e-6 and abs(A2 - P_TRUE * (1 - P_TRUE)) < 1e-4
print(f"   exp-family: A'(theta) = {A1:.4f} = p,  "
      f"A''(theta) = {A2:.4f} = p(1-p)")

# information functions: score has mean 0 and variance I(p) = 1/(p(1-p))
score = (x - P_TRUE) / (P_TRUE * (1 - P_TRUE))
I_p = 1 / (P_TRUE * (1 - P_TRUE))
print(f"   score: mean {score.mean():+.4f} (theory 0),  var "
      f"{score.var():.3f} vs I(p) = {I_p:.3f}")
assert abs(score.mean()) < 0.05 and abs(score.var() - I_p) / I_p < 0.02

# ----------------------------------------------------------------------
# 2. ESTIMATE — MLE closed form, then MAP
# ----------------------------------------------------------------------
m = int(x.sum())
p_mle = m / N
grid = np.linspace(0.001, 0.999, 100_000)
loglik = m * np.log(grid) + (N - m) * np.log(1 - grid)
se = np.sqrt(p_mle * (1 - p_mle) / N)
print("\n2. ESTIMATE")
print(f"   MLE  p_hat = m/n = {p_mle:.4f}   (brute-force argmax "
      f"{grid[loglik.argmax()]:.4f})")
print(f"   95% CI [{p_mle - 1.96 * se:.4f}, {p_mle + 1.96 * se:.4f}]"
      f"   contains p: {abs(p_mle - P_TRUE) < 1.96 * se}")
assert abs(p_mle - grid[loglik.argmax()]) < 1e-4

def p_map(m_, n_, a, b):
    """MAP under Beta(a, b): shifted counts — the prior as a-1
    imaginary ones and b-1 imaginary zeros."""
    return (m_ + a - 1) / (n_ + a + b - 2)

# flat prior recovers the MLE exactly
assert p_map(m, N, 1, 1) == p_mle
# small n: shrinkage toward 1/2 buys MSE (measured, 20k replications)
n_small, A, B = 20, 3, 3
ms = rng.binomial(n_small, P_TRUE, size=20_000)
mse_mle = np.mean((ms / n_small - P_TRUE) ** 2)
mse_map = np.mean((p_map(ms, n_small, A, B) - P_TRUE) ** 2)
print(f"   MAP Beta({A},{B}) at n={n_small}: MSE {mse_map:.5f} vs MLE "
      f"{mse_mle:.5f}  -> shrinkage wins small samples")
assert mse_map < mse_mle
# large n: data drowns the prior
gap = abs(p_map(m, N, A, B) - p_mle)
print(f"   at n={N}: |MAP - MLE| = {gap:.2e}  -> the prior fades")
assert gap < 1e-3

# ----------------------------------------------------------------------
# 3. DESCEND — the MLE as an ML problem (logged loop)
# ----------------------------------------------------------------------
sigmoid = lambda t: 1 / (1 + np.exp(-t))
LR, EPOCHS = 0.1, 3
print(f"\n3. DESCEND   SGD on F(t) = NLL(sigmoid(t)), lr={LR}, "
      f"{EPOCHS} epochs x {N:,} samples")
print(f"   {'step':>12} {'estimate':>9} {'NLL':>8}")
nll = lambda p: -(x.mean() * np.log(p) + (1 - x.mean()) * np.log(1 - p))
t, step = 0.0, 0
schedule = np.concatenate([rng.permutation(N) for _ in range(EPOCHS)])
traj = []
for i in schedule:
    t -= LR * (sigmoid(t) - x[i])            # g_i(t) = sigmoid(t) - x_i
    traj.append(t)
    step += 1
    if step % 3_000 == 0:
        print(f"   {step:>6,}/{len(schedule):,} "
              f"{sigmoid(t):>9.4f} {nll(sigmoid(t)):>8.4f}")
# Polyak: average the final epoch's PROBABILITY iterates (averaging in
# p-space, where the noise is symmetric around the optimum)
p_polyak = float(sigmoid(np.array(traj[-N:])).mean())
print(f"   last iterate {sigmoid(t):.4f} (noise ball)   "
      f"Polyak {p_polyak:.4f}   closed form {p_mle:.4f}")
assert abs(p_polyak - p_mle) < 1e-3

# proof by computation: autograd == the hand gradient, same schedule
t_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
opt = torch.optim.SGD([t_t], lr=LR)
x_t = torch.from_numpy(x)
traj_auto = []
for i in schedule:
    opt.zero_grad()
    torch.nn.functional.binary_cross_entropy_with_logits(
        t_t, x_t[i]).backward()
    opt.step()
    traj_auto.append(t_t.item())
gap = np.abs(np.array(traj) - np.array(traj_auto)).max()
print(f"   autograd identity: max |hand - autograd| = {gap:.2e}")
assert gap < 1e-10
print("all claims verified: OK")

# ----------------------------------------------------------------------
# Picture: the pmf, the landscape, the descent
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.6))
w = 0.35
ax1.bar([0 - w / 2, 1 - w / 2], [1 - P_TRUE, P_TRUE], w,
        color="#898781", label="true")
ax1.bar([0 + w / 2, 1 + w / 2], [1 - p_mle, p_mle], w,
        color="#eb6834", label=f"MLE {p_mle:.3f}")
ax1.set(xticks=[0, 1], title="the distribution vs the fit")
ax1.legend(frameon=False, fontsize=8)

pg = np.linspace(0.05, 0.95, 300)
ax2.plot(pg, [nll(p) for p in pg], color="#2a78d6", lw=2)
ax2.plot(p_mle, nll(p_mle), "o", color="#eb6834",
         label=f"argmin = m/n")
ax2.set(xlabel="p", ylabel="NLL", title="the landscape")
ax2.legend(frameon=False, fontsize=8)

est = sigmoid(np.array(traj))
ax3.plot(est, color="#3d9b35", lw=.7, label="SGD iterate")
ax3.axhline(p_mle, color="#111", ls="--", lw=1, label="closed form")
ax3.axhline(p_polyak, color="#2a78d6", ls=":", lw=1.4,
            label=f"Polyak {p_polyak:.4f}")
ax3.set(xlabel="SGD step", ylabel="estimate",
        title="descent, noise ball, average")
ax3.legend(frameon=False, fontsize=8)
for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
