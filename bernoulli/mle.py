r"""Bernoulli maximum-likelihood estimation, three ways.

Problem
-------
Given i.i.d. samples x_1, ..., x_n ~ Bernoulli(p), estimate p by maximizing
the likelihood — equivalently, minimizing the average negative
log-likelihood (NLL)

    L(p) = -(1/n) sum_i [ x_i log p + (1 - x_i) log(1 - p) ]
         = -x̄ log p - (1 - x̄) log(1 - p),          x̄ = (1/n) sum_i x_i.

Note the data enters only through x̄ (a sufficient statistic).

The three routes
----------------
(1) Theoretical solution:  solve L'(p) = 0  =>  p̂ = x̄  (unique: L is
    strictly convex on (0,1)).
(2) SGD, gradient derived by hand:  parameterize p = sigmoid(t) so the
    search over t is unconstrained (MLE invariance lets us map back), and
    descend the per-batch NLL whose gradient is  d/dt = sigmoid(t) - x̄_B.
(3) SGD via PyTorch autograd on the same objective
    (binary_cross_entropy_with_logits IS the Bernoulli NLL in logit form).

Routes (2) and (3) are driven through the *identical* batch sequence,
learning rate, and initialization — so if autograd computes exactly the
gradient we derived by hand, the two trajectories must agree to floating-
point precision. The script asserts this.

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
SEED = 7            # reproducibility: data + batch shuffles
LEARNING_RATE = 0.5
EPOCHS = 30
BATCH_SIZE = 64

rng = np.random.default_rng(SEED)

# The dataset: n draws of a Bernoulli(p_true) random variable, as floats
# so means and gradients are ordinary arithmetic.
x = rng.binomial(1, P_TRUE, size=N).astype(np.float64)


# ----------------------------------------------------------------------
# (1) Theoretical solution
# ----------------------------------------------------------------------
# Stationarity:  L'(p) = -x̄/p + (1-x̄)/(1-p) = 0  =>  p (1-x̄) = (1-p) x̄
# => p = x̄.  Second derivative  L''(p) = x̄/p² + (1-x̄)/(1-p)² > 0, so the
# NLL is strictly convex and x̄ is the unique global minimizer.
p_closed = x.mean()


# ----------------------------------------------------------------------
# Shared batch schedule
# ----------------------------------------------------------------------
# One list of index arrays per epoch, reshuffled every epoch. Both SGD
# implementations iterate this exact sequence, so any difference between
# them can only come from the gradient computation itself.
batches = [
    batch
    for _ in range(EPOCHS)
    for batch in np.array_split(rng.permutation(N), N // BATCH_SIZE)
]


# ----------------------------------------------------------------------
# (2) SGD with the gradient derived by hand
# ----------------------------------------------------------------------
def sigmoid(t: float) -> float:
    """sigma(t) = 1 / (1 + e^{-t}), mapping the real line onto (0, 1)."""
    return 1.0 / (1.0 + np.exp(-t))


def sgd_manual(batches: list[np.ndarray]) -> list[float]:
    """Minimize the NLL over t where p = sigma(t).

    Per batch B, the objective is
        L_B(t) = -(1/|B|) sum_{i in B} [ x_i t - log(1 + e^t) ]
               = log(1 + e^t) - x̄_B t,
    whose derivative (the whole point of the logit parameterization —
    sigma' = sigma(1-sigma) collapses everything) is
        dL_B/dt = sigma(t) - x̄_B.
    Update:  t <- t - eta (sigma(t) - x̄_B).

    Returns the trajectory of p = sigma(t) after every update.
    """
    t = 0.0  # sigma(0) = 0.5 — an uninformed starting guess
    trajectory = []
    for batch in batches:
        t -= LEARNING_RATE * (sigmoid(t) - x[batch].mean())
        trajectory.append(sigmoid(t))
    return trajectory


# ----------------------------------------------------------------------
# (3) SGD with the gradient computed by PyTorch autograd
# ----------------------------------------------------------------------
def sgd_torch(batches: list[np.ndarray]) -> list[float]:
    """The same optimization, but the gradient comes from autograd.

    binary_cross_entropy_with_logits(t, x_B) computes exactly
        (1/|B|) sum_{i in B} [ log(1 + e^t) - x_i t ],
    i.e. the batch NLL of Bernoulli(sigma(t)) — so loss.backward() must
    produce t.grad = sigma(t) - x̄_B, the derivative from sgd_manual.
    Identical batches + lr + init  =>  identical trajectory.
    """
    x_t = torch.from_numpy(x)                     # float64, like the numpy path
    t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([t], lr=LEARNING_RATE)
    trajectory = []
    for batch in batches:
        optimizer.zero_grad()
        xb = x_t[batch]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            t.expand(len(xb)), xb
        )
        loss.backward()   # autograd fills t.grad with sigma(t) - x̄_B
        optimizer.step()  # t <- t - eta * t.grad
        trajectory.append(torch.sigmoid(t).item())
    return trajectory


traj_manual = sgd_manual(batches)
traj_torch = sgd_torch(batches)

# Proof by computation: autograd's gradient IS the hand-derived gradient.
# Same data order, same eta, same t0 — the trajectories may differ only by
# float round-off. If this assertion ever fails, one of the two gradient
# derivations is wrong.
assert np.allclose(traj_manual, traj_torch, atol=1e-10), \
    "autograd disagrees with the hand-derived gradient"

# A single constant-step SGD iterate never converges — it wanders in a
# noise ball around the optimum (step size x batch noise). Averaging the
# iterates of the final epoch (Polyak–Ruppert) removes that noise.
last_epoch = N // BATCH_SIZE
p_sgd = float(np.mean(traj_manual[-last_epoch:]))
p_torch = float(np.mean(traj_torch[-last_epoch:]))


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
print(f"true p                 : {P_TRUE}")
print(f"(1) closed form  x̄     : {p_closed:.6f}")
print(f"(2) SGD by hand        : {p_sgd:.6f}")
print(f"(3) SGD via autograd   : {p_torch:.6f}")
print(f"max |traj(2)-traj(3)|  : {np.max(np.abs(np.array(traj_manual) - np.array(traj_torch))):.2e}")
print("autograd == hand gradient (allclose): OK")


# ----------------------------------------------------------------------
# Picture: the NLL landscape and the (coinciding) SGD trajectories
# ----------------------------------------------------------------------
grid = np.linspace(0.01, 0.99, 400)
nll = -(p_closed * np.log(grid) + (1 - p_closed) * np.log(1 - grid))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

ax1.plot(grid, nll, color="#2a78d6", lw=2)
ax1.axvline(P_TRUE, color="#898781", ls="--", lw=1, label=f"true p = {P_TRUE}")
ax1.plot(p_closed, np.interp(p_closed, grid, nll), "o", color="#eb6834",
         label=f"closed form = {p_closed:.4f}")
ax1.plot(p_sgd, np.interp(p_sgd, grid, nll), "x", color="#1baf7a", ms=9, mew=2,
         label=f"SGD (Polyak) = {p_sgd:.4f}")
ax1.set(xlabel="p", ylabel="NLL", title="NLL landscape (strictly convex)")
ax1.legend(frameon=False, fontsize=8)

ax2.plot(traj_manual, color="#1baf7a", lw=1.5, label="SGD by hand")
ax2.plot(traj_torch, color="#e87ba4", lw=1.5, ls=":", label="SGD via autograd")
ax2.axhline(p_closed, color="#eb6834", ls="--", lw=1, label="closed form x̄")
ax2.set(xlabel="SGD step", ylabel="p estimate",
        title="identical trajectories: autograd = hand gradient")
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
