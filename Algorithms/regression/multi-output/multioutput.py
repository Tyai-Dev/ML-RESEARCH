r"""Multi-output linear regression: linear -> multivariate Gaussian.

The regression sibling of softmax's Bernoulli -> multinoulli step. Now
the response is a VECTOR y in R^K:

    y | x  ~  N( W^T x , sigma^2 I_K ),      W in R^{d x K}.

Conditional MLE minimizes (1/2n) ||X W - Y||_F^2 and — unlike softmax —
THE CLOSED FORM SURVIVES: the Frobenius loss separates over output
columns, so each column of W solves an ordinary least-squares problem
with the SAME design matrix:

    W_hat = (X^T X)^{-1} X^T Y     (normal equations, K right-hand sides,
                                    ONE Gram factorization shared by all).

Two facts worth keeping:
- Even with correlated noise (full Sigma), as long as every output uses
  the same regressors, the MLE still equals per-column OLS (the seemingly-
  unrelated-regressions collapse).
- The "linearization trick" for classification — regress one-hot labels
  by least squares — IS this model applied to Y_onehot. It yields linear
  score functions but not probabilities (predictions escape [0,1] and
  don't sum to 1); softmax regression is the proper likelihood. We
  demonstrate the trick and its defect below.

Routes: closed form, GD, SGD by hand, SGD by autograd (identity check).
Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

SEED, N, D, K = 7, 3_000, 3, 2
rng = np.random.default_rng(SEED)

W_TRUE = np.array([[1.0, -2.0],
                   [2.0, 0.5],
                   [-1.0, 1.5]])                    # (D, K)
X = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])
Y = X @ W_TRUE + rng.normal(0, 0.8, size=(N, K))

GD_LR, GD_STEPS = 0.1, 800
SGD_LR, SGD_EPOCHS = 0.01, 3

# ----------------------------------------------------------------------
# (1) Closed form: normal equations with K right-hand sides
# ----------------------------------------------------------------------
W_closed = np.linalg.solve(X.T @ X, X.T @ Y)

# it IS per-column OLS: solving each output separately gives the same W
for k in range(K):
    wk = np.linalg.solve(X.T @ X, X.T @ Y[:, k])
    assert np.allclose(wk, W_closed[:, k])

# ----------------------------------------------------------------------
# (2) GD on (1/2n)||XW - Y||_F^2 :  grad = (1/n) X^T (XW - Y)
# ----------------------------------------------------------------------
def gradient_descent():
    W = np.zeros((D, K))
    for _ in range(GD_STEPS):
        W = W - GD_LR * X.T @ (X @ W - Y) / N
    return W


schedule = np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


# ----------------------------------------------------------------------
# (3) SGD by hand: grad_i = x_i outer (x_i^T W - y_i)  — residual x features
# ----------------------------------------------------------------------
def sgd_manual():
    W = np.zeros((D, K))
    trajectory = []
    for i in schedule:
        residual = X[i] @ W - Y[i]                 # (K,)
        W = W - SGD_LR * np.outer(X[i], residual)
        trajectory.append(W.copy())
    return trajectory


# ----------------------------------------------------------------------
# (4) The same SGD by autograd
# ----------------------------------------------------------------------
def sgd_torch():
    X_t, Y_t = torch.from_numpy(X), torch.from_numpy(Y)
    W = torch.zeros((D, K), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([W], lr=SGD_LR)
    trajectory = []
    for i in schedule:
        optimizer.zero_grad()
        loss = 0.5 * ((X_t[i] @ W - Y_t[i]) ** 2).sum()
        loss.backward()
        optimizer.step()
        trajectory.append(W.detach().numpy().copy())
    return trajectory


W_gd = gradient_descent()
traj_sgd = sgd_manual()
traj_torch = sgd_torch()

assert np.allclose(traj_sgd, traj_torch, atol=1e-10)
assert np.allclose(W_gd, W_closed, atol=1e-6)
W_sgd = np.mean(traj_sgd[-N:], axis=0)

np.set_printoptions(precision=4, suppress=True)
print("W* (columns = outputs):");           print(W_TRUE)
print("closed form:");                      print(W_closed)
print(f"GD == closed form to 1e-6: OK")
print(f"max |traj(3) - traj(4)|   : {np.max(np.abs(np.array(traj_sgd) - np.array(traj_torch))):.2e}")
print("autograd == hand gradient (allclose): OK")
assert np.allclose(W_closed, W_TRUE, atol=0.1)

# ----------------------------------------------------------------------
# The one-hot "linearization trick" for classification, and its defect
# ----------------------------------------------------------------------
# Take 3-class data (from the softmax experiment's model), regress the
# one-hot labels by least squares: scores are linear and argmax-usable,
# but the "probabilities" leave [0,1].
W3 = np.array([[0.0, 2.0, 0.0], [0.0, -1.0, 1.8], [0.5, -1.0, -1.8]])
P3 = np.exp(X @ W3.T); P3 /= P3.sum(1, keepdims=True)
y3 = (rng.uniform(size=N)[:, None] > np.cumsum(P3, axis=1)).sum(axis=1)
Y_onehot = np.eye(3)[y3]
W_trick = np.linalg.solve(X.T @ X, X.T @ Y_onehot)
scores = X @ W_trick
acc_trick = np.mean(scores.argmax(1) == y3)
acc_bayes = np.mean(P3.argmax(1) == y3)
print(f"\none-hot least-squares trick: accuracy {acc_trick:.3f} "
      f"(argmax of Bayes probs: {acc_bayes:.3f})")
print(f"but 'probabilities' range over [{scores.min():.2f}, {scores.max():.2f}]"
      f" — not a distribution; softmax regression is the proper likelihood")
assert scores.min() < 0 or scores.max() > 1

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

pred = X @ W_closed
for k, c in zip(range(K), ["#2a78d6", "#eb6834"]):
    ax1.plot(Y[:800, k], pred[:800, k], ".", ms=2, color=c, alpha=.5,
             label=f"output {k}")
lims = [Y.min(), Y.max()]
ax1.plot(lims, lims, color="#0b0b0b", lw=1)
ax1.set(xlabel="observed y_k", ylabel="fitted (W^T x)_k",
        title="both outputs, one shared Gram solve")
ax1.legend(frameon=False, fontsize=8)

ax2.hist(scores.ravel(), bins=60, color="#e34948")
ax2.axvspan(0, 1, color="#1baf7a", alpha=.15, label="[0, 1]")
ax2.set(xlabel="one-hot LS 'probabilities'", ylabel="count",
        title="the linearization trick escapes the simplex")
ax2.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
