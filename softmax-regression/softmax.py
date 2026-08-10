r"""Softmax regression: a multinoulli whose parameter is a function of X.

Bernoulli -> multinoulli, exactly as promised. For K classes,

    Y | X=x ~ Multinoulli( p_1(x), ..., p_K(x) ),
    p_k(x) = softmax(W x)_k = exp(w_k . x) / sum_j exp(w_j . x),

softmax being to the simplex what the sigmoid is to (0,1). W is a K x d
matrix, one weight vector per class. The conditional NLL of a sample is
the cross-entropy -log p_{y_i}(x_i), and its gradient w.r.t. row w_k is

    grad_{w_k} = ( p_k(x_i) - 1[y_i = k] ) x_i ,

i.e. (softmax - one_hot) outer x — residual times features, vectorized
over classes; for K = 2 it collapses to logistic regression. No closed
form (transcendental stationarity), but convex, so GD finds the optimum.

Routes: GD, SGD by hand, SGD via torch F.cross_entropy on the identical
schedule (trajectory identity assert). Because the model is
well-specified, the fitted classifier approaches the multiclass BAYES
floor  E[ 1 - max_k p_k(X) ].

Note on identifiability: adding the same vector to every row of W leaves
softmax unchanged, so W itself is only identified up to that shift — we
therefore compare fitted PROBABILITIES to true probabilities, not W to W*.

Run me with F5. Companion derivations: softmax-regression.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# ----------------------------------------------------------------------
# Data: 3 classes in 2D, sampled from a true softmax model
# ----------------------------------------------------------------------
SEED, N, K, D = 7, 6_000, 3, 3            # D = 1 intercept + 2 features
rng = np.random.default_rng(SEED)

W_TRUE = np.array([[0.0, 2.0, 0.0],       # class 0 pulls +x1
                   [0.0, -1.0, 1.8],      # class 1 pulls -x1, +x2
                   [0.5, -1.0, -1.8]])    # class 2 pulls -x1, -x2

X = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])


def softmax(Z):
    """Row-wise softmax with the max-shift trick for stability."""
    E = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)


P_true = softmax(X @ W_TRUE.T)                     # (N, K)
y = (rng.uniform(size=N)[:, None] > np.cumsum(P_true, axis=1)).sum(axis=1)

GD_LR, GD_STEPS = 0.5, 3_000
SGD_LR, SGD_EPOCHS = 0.05, 3


def nll(W):
    P = softmax(X @ W.T)
    return float(-np.mean(np.log(P[np.arange(N), y] + 1e-12)))


# ----------------------------------------------------------------------
# (1) Gradient descent.  Full gradient: (1/n) (P - Y_onehot)^T X
# ----------------------------------------------------------------------
Y_onehot = np.eye(K)[y]                            # (N, K)


def gradient_descent():
    W = np.zeros((K, D))
    history = []
    for _ in range(GD_STEPS):
        P = softmax(X @ W.T)
        W = W - GD_LR * (P - Y_onehot).T @ X / N
        history.append(nll(W))
    return W, history


schedule = np.concatenate([rng.permutation(N) for _ in range(SGD_EPOCHS)])


# ----------------------------------------------------------------------
# (2) SGD by hand: grad rows = (softmax(Wx) - onehot(y)) outer x
# ----------------------------------------------------------------------
def sgd_manual():
    W = np.zeros((K, D))
    trajectory = []
    for i in schedule:
        p = softmax(X[i] @ W.T)
        W = W - SGD_LR * np.outer(p - Y_onehot[i], X[i])
        trajectory.append(W.copy())
    return trajectory


# ----------------------------------------------------------------------
# (3) The same SGD via torch F.cross_entropy (identical schedule)
# ----------------------------------------------------------------------
def sgd_torch():
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()
    W = torch.zeros((K, D), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([W], lr=SGD_LR)
    trajectory = []
    for i in schedule:
        optimizer.zero_grad()
        logits = (W @ X_t[i]).unsqueeze(0)          # (1, K)
        loss = torch.nn.functional.cross_entropy(logits, y_t[i].view(1))
        loss.backward()      # autograd: (softmax - onehot) outer x
        optimizer.step()
        trajectory.append(W.detach().numpy().copy())
    return trajectory


W_gd, gd_hist = gradient_descent()
traj_sgd = sgd_manual()
traj_torch = sgd_torch()

assert np.allclose(traj_sgd, traj_torch, atol=1e-10), \
    "autograd disagrees with the hand-derived gradient"
W_sgd = np.mean(traj_sgd[-N:], axis=0)

# ----------------------------------------------------------------------
# Evaluate: probabilities (identifiable), accuracy vs the Bayes floor
# ----------------------------------------------------------------------
M = 100_000
X_test = np.column_stack([np.ones(M), rng.normal(size=(M, 2))])
P_test_true = softmax(X_test @ W_TRUE.T)
y_test = (rng.uniform(size=M)[:, None] > np.cumsum(P_test_true, axis=1)).sum(axis=1)

P_test_fit = softmax(X_test @ W_gd.T)
prob_err = float(np.mean(np.abs(P_test_fit - P_test_true)))

bayes_error = float(np.mean(1 - P_test_true.max(axis=1)))
fitted_error = float(np.mean(P_test_fit.argmax(axis=1) != y_test))

np.set_printoptions(precision=4, suppress=True)
print(f"(1) GD    final NLL       : {gd_hist[-1]:.6f}")
print(f"(2) SGD   (Polyak) NLL    : {nll(W_sgd):.6f}")
print(f"max |traj(2) - traj(3)|   : {np.max(np.abs(np.array(traj_sgd) - np.array(traj_torch))):.2e}")
print("autograd == hand gradient (allclose): OK")
print(f"mean |p_fit - p_true|     : {prob_err:.4f}   (W is only identified up to")
print("                             a shared row-shift; probabilities are the object)")
print(f"Bayes error E[1 - max p]  : {bayes_error:.4f}")
print(f"fitted classifier error   : {fitted_error:.4f}")
assert prob_err < 0.02
assert fitted_error < bayes_error + 0.01

# ----------------------------------------------------------------------
# Picture: decision regions, convergence, probability calibration
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

g = np.linspace(-3, 3, 300)
GX, GY = np.meshgrid(g, g)
G = np.column_stack([np.ones(GX.size), GX.ravel(), GY.ravel()])
regions = softmax(G @ W_gd.T).argmax(axis=1).reshape(GX.shape)
ax1.contourf(GX, GY, regions, levels=[-.5, .5, 1.5, 2.5],
             colors=["#dbe7f6", "#ddf1e6", "#fbe9e7"])
sub = slice(0, 1500)
ax1.scatter(X[sub, 1], X[sub, 2], c=y[sub], cmap="viridis", s=5, alpha=.7)
ax1.set(xlabel="x1", ylabel="x2", title="fitted decision regions (K = 3)")

ax2.semilogy(np.maximum(np.array(gd_hist) - gd_hist[-1], 1e-16),
             color="#2a78d6", lw=1.5)
ax2.set(xlabel="GD step", ylabel="NLL - final (log)",
        title="convex cross-entropy: one basin")

ax3.plot(P_test_true[:4000, 0], P_test_fit[:4000, 0], ".", ms=2,
         color="#2a78d6", alpha=.4)
ax3.plot([0, 1], [0, 1], color="#eb6834", lw=1.5)
ax3.set(xlabel="true p_0(x)", ylabel="fitted p_0(x)",
        title="estimated class-0 probability function")

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
