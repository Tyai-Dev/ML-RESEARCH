r"""Multiclass perceptron — promote/demote, and the Kesler identity.

The algorithm. Weights W (K x d), score = W x, predict argmax. On a
mistake (predicted yh != true y):

    W[y]  += x        (promote the truth)
    W[yh] -= x        (demote the offender)

THE KESLER CONSTRUCTION (multiclass IS binary, proved by run):
stack W into one vector w in R^{Kd} and define joint features
phi(x, k) = x placed in block k. Then score(k) = w . phi(x, k), and
the promote/demote update is exactly the BINARY perceptron update
w += phi(x, y) - phi(x, yh). Verified below: both implementations,
driven through the same pass order, produce the IDENTICAL mistake
sequence and identical weights — so Novikoff's (R/gamma)^2 mistake
bound applies verbatim with joint-feature radius and margin, and the
bound is checked numerically.

Run me with F5. Derivations: multiclass-perceptron.pdf.
"""

import numpy as np
from sklearn.metrics import classification_report

SEED, K, D, N = 7, 3, 2, 900
rng = np.random.default_rng(SEED)
CENTERS = np.array([[2.5, 0.0], [-1.5, 2.2], [-1.5, -2.2]])
X = np.vstack([rng.normal(c, 0.6, size=(N // K, D)) for c in CENTERS])
y = np.repeat(np.arange(K), N // K)
perm = rng.permutation(N)
X, y = X[perm], y[perm]
Xb = np.column_stack([np.ones(N), X])          # intercept
Xtr, ytr, Xte, yte = Xb[:700], y[:700], Xb[700:], y[700:]

def train_matrix(passes=10):
    W = np.zeros((K, D + 1))
    mistakes = []
    for ep in range(passes):
        m = 0
        for i in range(len(ytr)):
            yh = int((W @ Xtr[i]).argmax())
            if yh != ytr[i]:
                W[ytr[i]] += Xtr[i]
                W[yh] -= Xtr[i]
                m += 1
                mistakes.append((ep, i))
        print(f"   pass {ep + 1:>2}: {m:>3} mistakes")
        if m == 0:
            break
    return W, mistakes

def train_kesler(passes=10):
    w = np.zeros(K * (D + 1))
    phi = lambda x, k: np.concatenate(
        [x if j == k else np.zeros(D + 1) for j in range(K)])
    mistakes = []
    for ep in range(passes):
        m = 0
        for i in range(len(ytr)):
            yh = int(np.argmax([w @ phi(Xtr[i], k) for k in range(K)]))
            if yh != ytr[i]:
                w += phi(Xtr[i], ytr[i]) - phi(Xtr[i], yh)
                m += 1
                mistakes.append((ep, i))
        if m == 0:
            break
    return w.reshape(K, D + 1), mistakes

print("promote/demote (K x d matrix view):")
W, mk1 = train_matrix()
w_k, mk2 = train_kesler()

assert mk1 == mk2, "mistake sequences must be identical"
assert np.allclose(W, w_k), "weights must be identical"
print(f"Kesler identity: {len(mk1)} mistakes, sequences and weights "
      f"identical: OK")

# Novikoff bound with joint features: R^2 = 2 max||x||^2 (two blocks
# change per update), gamma from the found separator (valid: the bound
# holds for ANY separating W*)
margins = np.array([(W[ytr[i]] - np.delete(W, ytr[i], 0)) @ Xtr[i]
                    for i in range(len(ytr))], dtype=object)
gam = min(float(np.min(m_)) for m_ in margins) / np.linalg.norm(W)
R2 = 2 * max(np.sum(Xtr ** 2, axis=1))
bound = R2 / gam**2
print(f"Novikoff (joint features): {len(mk1)} mistakes <= "
      f"(R/gamma)^2 = {bound:.0f}: {len(mk1) <= bound}")
assert gam > 0 and len(mk1) <= bound

pred = (Xte @ W.T).argmax(axis=1)
print(classification_report(yte, pred, digits=3))
assert np.mean(pred == yte) > 0.95
print("all claims verified: OK")
