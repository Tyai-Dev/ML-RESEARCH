r"""Next-state prediction as classification — the derivation, executed.

Claims of markov-cross-entropy.pdf, each verified:
  1. Training a softmax classifier q(x_{t+1} | x_t) by cross-entropy
     over one simulated path recovers the COUNT transition table.
  2. The minimum loss equals the occupancy-weighted entropy of the
     empirical rows (the H + KL decomposition, KL term at zero).
  3. The hand gradient rho_i (softmax(Z_i) - P_hat_i) matches
     autograd to machine precision.
  4. The minimum loss approaches the chain's ENTROPY RATE
     sum_i pi_i H(P_i) — the sequence world's Bayes floor.

Run me with F5. Derivations: markov-cross-entropy.pdf.
"""

import numpy as np
import torch

SEED = 7
rng = np.random.default_rng(SEED)
STATES = ["Sunny", "Cloudy", "Rainy"]
P = np.array([[0.6, 0.3, 0.1],
              [0.2, 0.5, 0.3],
              [0.1, 0.6, 0.3]])
V, T = len(STATES), 200_000

# simulate one path; count
path = np.empty(T, dtype=np.int64)
path[0] = 0
for t in range(1, T):
    path[t] = rng.choice(V, p=P[path[t - 1]])
C = np.zeros((V, V))
np.add.at(C, (path[:-1], path[1:]), 1.0)
rho = C.sum(axis=1) / (T - 1)
P_hat = C / C.sum(axis=1, keepdims=True)

H_rows = -np.sum(P_hat * np.log(P_hat), axis=1)
floor_empirical = float(rho @ H_rows)               # sum_i rho_i H(P^_i)

def stationary(P):
    w, vec = np.linalg.eig(P.T)
    pi = np.real(vec[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()

pi = stationary(P)
entropy_rate = float(pi @ -np.sum(P * np.log(P), axis=1))
print(f"empirical floor sum rho_i H(P_hat_i) = {floor_empirical:.6f}")
print(f"true entropy rate  sum pi_i H(P_i)   = {entropy_rate:.6f}")

# ---- train the classifier: full-batch GD on the count-collapsed loss
def loss_grad(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    Q = E / E.sum(axis=1, keepdims=True)
    L = -np.sum(C * np.log(Q)) / (T - 1)
    G = rho[:, None] * (Q - P_hat)                  # eq. (grad)
    return L, G, Q

Z = np.zeros((V, V))
print("\ntraining the next-state classifier (full-batch GD):")
for step in range(1, 401):
    L, G, Q = loss_grad(Z)
    Z -= 5.0 * G
    if step % 100 == 0:
        print(f"   step {step:>3}  cross-entropy {L:.6f}  "
              f"max|q - count table| {np.abs(Q - P_hat).max():.2e}")

# 1. optimum = count table
assert np.abs(Q - P_hat).max() < 1e-3
print("1. trained softmax rows == count transition table: OK")
# 2. min loss = weighted row entropy (KL term at zero)
assert abs(L - floor_empirical) < 1e-6
print("2. min cross-entropy == sum rho_i H(P_hat_i): OK")

# 3. hand gradient == autograd at a random Z
Z0 = rng.normal(size=(V, V))
_, G_h, _ = loss_grad(Z0)
Zt = torch.tensor(Z0, requires_grad=True)
Ct = torch.from_numpy(C)
Lt = -(Ct * torch.log_softmax(Zt, dim=1)).sum() / (T - 1)
Lt.backward()
gap = np.abs(G_h - Zt.grad.numpy()).max()
print(f"3. hand gradient vs autograd: max gap {gap:.2e}: OK")
assert gap < 1e-12

# 4. floor -> entropy rate (ergodic theorem does the work)
assert abs(floor_empirical - entropy_rate) < 1e-3
print("4. achievable loss == the chain's entropy rate (to 1e-3): OK")
print("\na next-state classifier IS a Markov chain estimator; "
      "a language model is this, iterated.")
