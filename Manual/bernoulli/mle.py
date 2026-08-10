"""Bernoulli MLE two ways: the closed form and SGD on the NLL.

Draw x_1..x_n ~ Bernoulli(p_true), then estimate p by
  (1) the theoretical solution  p_hat = x_bar, and
  (2) SGD on the negative log-likelihood in logit space (p = sigmoid(t)).
Both should land on the same value — that is the point (the NLL is convex).

Self-contained: numpy + matplotlib. Press F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)
P_TRUE, N = 0.3, 5_000
x = rng.binomial(1, P_TRUE, size=N).astype(float)


# --- (1) theoretical solution: d/dp of the log-likelihood = 0  =>  p = x_bar
p_closed = x.mean()


# --- (2) SGD on the NLL. Optimize t with p = sigmoid(t) so p stays in (0,1);
#         dNLL/dt over a batch is simply mean(sigmoid(t) - x_batch).
def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


t, lr, epochs, batch_size = 0.0, 0.5, 30, 64
trajectory = [sigmoid(t)]
for _ in range(epochs):
    for idx in np.array_split(rng.permutation(N), N // batch_size):
        t -= lr * (sigmoid(t) - x[idx].mean())
        trajectory.append(sigmoid(t))

# a single SGD iterate wobbles around the optimum (constant step + batch
# noise) — average the last epoch (Polyak averaging) for the estimate
p_sgd = float(np.mean(trajectory[-(N // batch_size):]))


print(f"true p          : {P_TRUE}")
print(f"closed form x̄   : {p_closed:.4f}")
print(f"SGD on the NLL  : {p_sgd:.4f}")
print(f"|difference|    : {abs(p_closed - p_sgd):.2e}")


# --- picture: the NLL landscape with both estimates on it, and the SGD path
grid = np.linspace(0.01, 0.99, 400)
nll = -(x.mean() * np.log(grid) + (1 - x.mean()) * np.log(1 - grid))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(grid, nll, color="#2a78d6", lw=2)
ax1.axvline(P_TRUE, color="#898781", ls="--", lw=1, label=f"true p = {P_TRUE}")
ax1.plot(p_closed, np.interp(p_closed, grid, nll), "o", color="#eb6834",
         label=f"closed form = {p_closed:.4f}")
ax1.plot(p_sgd, np.interp(p_sgd, grid, nll), "x", color="#1baf7a", ms=9, mew=2,
         label=f"SGD = {p_sgd:.4f}")
ax1.set(xlabel="p", ylabel="NLL", title="NLL landscape (convex)")
ax1.legend(frameon=False, fontsize=8)

ax2.plot(trajectory, color="#1baf7a", lw=1.5)
ax2.axhline(p_closed, color="#eb6834", ls="--", lw=1, label="closed form")
ax2.set(xlabel="SGD step", ylabel="p estimate", title="SGD converges to x̄")
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
