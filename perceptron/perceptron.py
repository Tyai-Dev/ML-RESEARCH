r"""The perceptron: distribution-free learning, and the Novikoff bound.

The shift from everything before this folder: NO probabilistic model.
No p(x), no likelihood, no noise assumption — just a stream of labeled
points and a promise about MISTAKES. Labels are y in {-1, +1}.

Algorithm (Rosenblatt 1958). Keep w (init 0); for each example predict
sign(w . x); on a mistake (y (w . x) <= 0) update

    w  <-  w + y x .

That's it. The guarantee is the mistake bound (Novikoff 1962): if some
u with ||u|| = 1 separates the stream with margin gamma
(y (u . x) >= gamma for all points) and ||x|| <= R, then the perceptron
makes at most

    (R / gamma)^2

mistakes — EVER, regardless of the order or length of the stream, with
no distributional assumption at all. Proof in perceptron.tex (two
inequalities racing: w . u grows linearly per mistake, ||w||^2 at most
linearly, and Cauchy–Schwarz squeezes).

Multiclass version: one weight vector per class, predict
argmax_k (w_k . x); on a mistake promote the true class and demote the
predicted one:  w_y += x,  w_yhat -= x.

The script verifies the bound on separable data (binary), shows the
multiclass version converging, and shows the failure mode: on
non-separable data the perceptron never settles (mistakes keep coming
forever — the cue for Passive-Aggressive and SVM).

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Separable binary data with a known margin
# ----------------------------------------------------------------------
# Sample x in a disc, label by a true separator u*, discard points inside
# the margin band so that gamma is guaranteed.
N, GAMMA = 2_000, 0.15
U_STAR = np.array([0.6, 0.8])                     # ||u*|| = 1

points = []
while len(points) < N:
    x = rng.uniform(-1, 1, size=2)
    if np.linalg.norm(x) <= 1 and abs(U_STAR @ x) >= GAMMA:
        points.append(x)
X = np.array(points)
y = np.sign(X @ U_STAR)
R = np.linalg.norm(X, axis=1).max()
bound = (R / GAMMA) ** 2


# ----------------------------------------------------------------------
# Binary perceptron: run epochs until a full clean pass
# ----------------------------------------------------------------------
def perceptron(X, y, max_epochs=100):
    w = np.zeros(X.shape[1])
    mistakes, mistake_curve = 0, []
    for _ in range(max_epochs):
        clean = True
        for i in range(len(X)):
            if y[i] * (w @ X[i]) <= 0:            # mistake (ties count)
                w = w + y[i] * X[i]
                mistakes += 1
                clean = False
            mistake_curve.append(mistakes)
        if clean:
            return w, mistakes, mistake_curve, True
    return w, mistakes, mistake_curve, False


w_fin, mistakes, curve, converged = perceptron(X, y)
train_err = np.mean(np.sign(X @ w_fin) != y)

print("--- binary perceptron on separable data ---")
print(f"margin gamma = {GAMMA}, radius R = {R:.3f}")
print(f"mistakes made          : {mistakes}")
print(f"Novikoff bound (R/g)^2 : {bound:.0f}")
print(f"converged (clean pass) : {converged},  train error = {train_err:.3f}")
assert converged and mistakes <= bound and train_err == 0.0


# ----------------------------------------------------------------------
# Multiclass perceptron: promote the truth, demote the pretender
# ----------------------------------------------------------------------
K = 3
CENTERS = np.array([[1.5, 0.0], [-0.8, 1.3], [-0.8, -1.3]])
Xm = np.vstack([rng.normal(c, 0.35, size=(400, 2)) for c in CENTERS])
ym = np.repeat(np.arange(K), 400)
perm = rng.permutation(len(Xm))
Xm, ym = Xm[perm], ym[perm]


def multiclass_perceptron(X, y, K, max_epochs=100):
    W = np.zeros((K, X.shape[1]))
    mistakes = 0
    for _ in range(max_epochs):
        clean = True
        for i in range(len(X)):
            pred = int(np.argmax(W @ X[i]))
            if pred != y[i]:
                W[y[i]] += X[i]                   # promote the true class
                W[pred] -= X[i]                   # demote the imposter
                mistakes += 1
                clean = False
        if clean:
            return W, mistakes, True
    return W, mistakes, False


Wm, mistakes_m, converged_m = multiclass_perceptron(Xm, ym, K)
train_err_m = np.mean((Xm @ Wm.T).argmax(1) != ym)
print("\n--- multiclass perceptron (3 well-separated blobs) ---")
print(f"mistakes: {mistakes_m},  converged: {converged_m},  "
      f"train error: {train_err_m:.3f}")
assert converged_m and train_err_m == 0.0


# ----------------------------------------------------------------------
# The failure mode: non-separable data never converges
# ----------------------------------------------------------------------
y_noisy = y.copy()
flip = rng.choice(len(y), size=len(y) // 20, replace=False)   # 5% flipped
y_noisy[flip] *= -1
_, mistakes_noisy, curve_noisy, converged_noisy = perceptron(X, y_noisy, max_epochs=30)
print("\n--- 5% label noise: separability gone ---")
print(f"converged: {converged_noisy},  mistakes in 30 epochs: {mistakes_noisy}")
print("(mistakes accumulate forever — the cue for PA and SVM)")
assert not converged_noisy

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=5, alpha=.6)
g = np.linspace(-1, 1, 2)
ax1.plot(g, -U_STAR[0] / U_STAR[1] * g, "--", color="#0b0b0b", lw=1,
         label="true separator")
ax1.plot(g, -w_fin[0] / w_fin[1] * g, color="#1baf7a", lw=1.5,
         label="perceptron")
ax1.set(title=f"separable: {mistakes} mistakes ≤ bound {bound:.0f}",
        xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
ax1.legend(frameon=False, fontsize=8)

ax2.plot(curve, color="#2a78d6", lw=1.5, label="separable")
ax2.plot(curve_noisy, color="#e34948", lw=1.5, label="5% noise")
ax2.axhline(bound, color="#898781", ls="--", lw=1, label="Novikoff bound")
ax2.set(xlabel="examples processed", ylabel="cumulative mistakes",
        title="mistakes stop vs never stop")
ax2.legend(frameon=False, fontsize=8)

gx = np.linspace(-2.5, 3, 300)
gy = np.linspace(-2.5, 2.5, 300)
GX, GY = np.meshgrid(gx, gy)
G = np.column_stack([GX.ravel(), GY.ravel()])
regions = (G @ Wm.T).argmax(1).reshape(GX.shape)
ax3.contourf(GX, GY, regions, levels=[-.5, .5, 1.5, 2.5],
             colors=["#dbe7f6", "#ddf1e6", "#fbe9e7"])
ax3.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="viridis", s=4, alpha=.7)
ax3.set(title="multiclass perceptron regions")

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
