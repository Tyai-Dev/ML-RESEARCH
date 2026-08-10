r"""Bernoulli maximum-likelihood estimation: closed form, GD, SGD, autograd.

Problem
-------
Given i.i.d. samples x_1, ..., x_n ~ Bernoulli(p), estimate p by maximizing
the likelihood. Getting from the likelihood to the loss we optimize takes
three steps, each preserving the optimizer:

Step 1 — the likelihood. Each factor p^{x_i} (1-p)^{1-x_i} is a case split:
it equals p when x_i = 1 and (1-p) when x_i = 0. With m = sum_i x_i ones
and n - m zeros, the product collapses to

    L(p) = p^m (1-p)^{n-m}.

The data enters only through the count m (equivalently x̄ = m/n): a
sufficient statistic — the ordering of the sample is irrelevant.

Step 2 — take the log. log is strictly increasing, so it does not move the
argmax; it turns the powers into a sum that is easy to differentiate (and
immune to the numerical underflow of multiplying 5000 numbers in (0,1)):

    l(p) = log L(p) = m log p + (n - m) log(1 - p).

Step 3 — normalize and negate. Dividing by the constant n > 0 rescales
without moving the argmax; negating turns maximization into minimization
(the ML convention: minimize losses). The average negative log-likelihood:

    NLL(p) = -(1/n) l(p) = -x̄ log p - (1 - x̄) log(1 - p).

So  argmax L  =  argmax l  =  argmin NLL  — one problem, three notations.

The four routes below
---------------------
(1) Theoretical solution:  l'(p) = 0  =>  p̂ = m/n.
(2) Gradient descent (GD) on the NLL — the full gradient, every step.
(3) Stochastic gradient descent (SGD) — one random sample per step,
    an unbiased but noisy estimate of the same gradient.
(4) SGD again, with the gradient computed by PyTorch autograd, driven
    through the identical sample schedule as (3): the trajectories must
    coincide to float precision if autograd matches the hand derivation.

One coordinate change, NOT a new objective: gradient methods want an
unconstrained variable, while p lives in (0,1). We simply evaluate the
SAME NLL at p = sigmoid(t),

    F(t) = NLL(sigmoid(t)),        t in R,

and differentiate the composition with the chain rule. With
sigmoid' = sigmoid (1 - sigmoid), everything cancels (derivation in
bernoulli.tex, eq. (grad)):

    F'(t) = sigmoid(t) - x̄ .

Run me with F5. Companion derivations: bernoulli.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------
P_TRUE = 0.3        # the parameter we pretend not to know
N = 5_000           # sample size
SEED = 7            # reproducibility: data + SGD sample order

GD_LR, GD_STEPS = 1.0, 100      # gradient descent
SGD_LR, SGD_EPOCHS = 0.1, 3     # stochastic gradient descent

rng = np.random.default_rng(SEED)

# The dataset: n draws of a Bernoulli(p_true) random variable, as floats
# so means and gradients are ordinary arithmetic.
x = rng.binomial(1, P_TRUE, size=N).astype(np.float64)


# ----------------------------------------------------------------------
# (1) Theoretical solution
# ----------------------------------------------------------------------
# Differentiate the log-likelihood l(p) = m log p + (n-m) log(1-p) — the
# counts keep the algebra honest:
#     l'(p) = m/p - (n-m)/(1-p) = 0
#  => m (1-p) = (n-m) p        (clear denominators)
#  => m - mp  = np - mp
#  => m = np  =>  p = m/n  — the fraction of ones in the sample.
# Uniqueness: l''(p) = -m/p² - (n-m)/(1-p)² < 0, so l is strictly concave
# and this stationary point is the global maximum.
p_closed = x.mean()  # x̄ = m/n


def sigmoid(t: float) -> float:
    """sigma(t) = 1 / (1 + e^{-t}), mapping the real line onto (0, 1)."""
    return 1.0 / (1.0 + np.exp(-t))


# ----------------------------------------------------------------------
# (2) Gradient descent — the full gradient, every step
# ----------------------------------------------------------------------
# Minimize F(t) = NLL(sigmoid(t)). By the chain rule (see bernoulli.tex),
#     F'(t) = sigmoid(t) - x̄ ,
# which uses the WHOLE dataset (through x̄) at every step: this is GD.
# F is convex in t (F''(t) = sigmoid'(t) > 0), so with a sane step size the
# iteration t <- t - eta F'(t) converges deterministically — no noise,
# no averaging needed — to the fixed point sigmoid(t*) = x̄.
def gradient_descent() -> list[float]:
    t = 0.0  # sigmoid(0) = 0.5 — an uninformed starting guess
    trajectory = []
    for _ in range(GD_STEPS):
        t -= GD_LR * (sigmoid(t) - x.mean())
        trajectory.append(sigmoid(t))
    return trajectory


# ----------------------------------------------------------------------
# Shared sample schedule for (3) and (4)
# ----------------------------------------------------------------------
# SGD visits one sample per step; the schedule is the sample order over all
# epochs. Both SGD implementations follow this exact sequence, so any
# difference between them can only come from the gradient computation.
schedule = np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


# ----------------------------------------------------------------------
# (3) Stochastic gradient descent — one sample per step, by hand
# ----------------------------------------------------------------------
# Replace the full gradient sigmoid(t) - x̄ with the ONE-SAMPLE estimate
#     g_i(t) = sigmoid(t) - x_i .
# It is unbiased:  E_i[g_i(t)] = sigmoid(t) - x̄  (i uniform over samples),
# so on average SGD walks in the same direction as GD, at 1/n the cost per
# step. The price is noise: with a constant step the iterate never settles,
# it hovers in a noise ball around the optimum. Remedy below the loop.
def sgd_manual() -> list[float]:
    t = 0.0
    trajectory = []
    for i in schedule:
        t -= SGD_LR * (sigmoid(t) - x[i])
        trajectory.append(sigmoid(t))
    return trajectory


# ----------------------------------------------------------------------
# (4) The same SGD, gradient by PyTorch autograd
# ----------------------------------------------------------------------
# binary_cross_entropy_with_logits(t, x_i) evaluates exactly the one-sample
# NLL F_i(t) = NLL_i(sigmoid(t)), so loss.backward() must produce
# t.grad = sigmoid(t) - x_i — the hand gradient of (3). Identical sample
# schedule + step size + start  =>  identical trajectory, or one of the two
# gradient derivations is wrong.
def sgd_torch() -> list[float]:
    x_t = torch.from_numpy(x)                     # float64, like the numpy path
    t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([t], lr=SGD_LR)
    trajectory = []
    for i in schedule:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(t, x_t[i])
        loss.backward()   # autograd fills t.grad with sigmoid(t) - x_i
        optimizer.step()  # t <- t - eta * t.grad
        trajectory.append(torch.sigmoid(t).item())
    return trajectory


traj_gd = gradient_descent()
traj_sgd = sgd_manual()
traj_torch = sgd_torch()

# Proof by computation: autograd's gradient IS the hand-derived gradient.
assert np.allclose(traj_sgd, traj_torch, atol=1e-10), \
    "autograd disagrees with the hand-derived gradient"

# GD converges outright; SGD does not — a constant-step iterate hovers in
# a noise ball around the optimum. Averaging the final epoch's iterates
# (Polyak–Ruppert) cancels that noise.
p_gd = traj_gd[-1]
p_sgd = float(np.mean(traj_sgd[-N:]))
p_torch = float(np.mean(traj_torch[-N:]))


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
print(f"true p                    : {P_TRUE}")
print(f"(1) closed form m/n       : {p_closed:.6f}")
print(f"(2) GD, final iterate     : {p_gd:.6f}   (|Δ| vs closed: {abs(p_gd - p_closed):.1e})")
print(f"(3) SGD, Polyak average   : {p_sgd:.6f}   (|Δ| vs closed: {abs(p_sgd - p_closed):.1e})")
print(f"(4) SGD via autograd      : {p_torch:.6f}")
print(f"max |traj(3) - traj(4)|   : {np.max(np.abs(np.array(traj_sgd) - np.array(traj_torch))):.2e}")
print("autograd == hand gradient (allclose): OK")


# ----------------------------------------------------------------------
# Picture: landscape, GD converging, SGD hovering (with autograd overlaid)
# ----------------------------------------------------------------------
grid = np.linspace(0.01, 0.99, 400)
nll = -(p_closed * np.log(grid) + (1 - p_closed) * np.log(1 - grid))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

ax1.plot(grid, nll, color="#2a78d6", lw=2)
ax1.axvline(P_TRUE, color="#898781", ls="--", lw=1, label=f"true p = {P_TRUE}")
ax1.plot(p_closed, np.interp(p_closed, grid, nll), "o", color="#eb6834",
         label=f"closed form = {p_closed:.4f}")
ax1.set(xlabel="p", ylabel="NLL", title="NLL landscape (strictly convex)")
ax1.legend(frameon=False, fontsize=8)

ax2.plot(traj_gd, color="#2a78d6", lw=2)
ax2.axhline(p_closed, color="#eb6834", ls="--", lw=1, label="closed form m/n")
ax2.set(xlabel="GD step", ylabel="p estimate",
        title="GD: deterministic convergence")
ax2.legend(frameon=False, fontsize=8)

ax3.plot(traj_sgd, color="#1baf7a", lw=.8, label="SGD by hand")
ax3.plot(traj_torch, color="#e87ba4", lw=.8, ls=":", label="SGD via autograd")
ax3.axhline(p_closed, color="#eb6834", ls="--", lw=1, label="closed form m/n")
ax3.set(xlabel="SGD step", ylabel="p estimate",
        title="SGD: noise ball around the optimum")
ax3.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
