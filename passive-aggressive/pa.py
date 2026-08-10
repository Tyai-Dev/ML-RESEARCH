r"""Passive-Aggressive: online learning with a per-step optimization.

Where the perceptron applies one fixed-size correction to every mistake,
PA (Crammer et al., 2006) SOLVES a tiny constrained problem at each step:

    w_{t+1} = argmin_w  (1/2) ||w - w_t||^2
              s.t.      hinge(w; x_t, y_t) = max(0, 1 - y_t w.x_t) = 0.

Passive: if the example already has margin >= 1, do nothing. Aggressive:
otherwise, make the SMALLEST change that fully satisfies it. The KKT
machinery of Theory/optimization gives the closed-form step (derived in
pa.tex):

    w  <-  w + tau y x,      tau = hinge / ||x||^2            (PA)

and its noise-tolerant variants, which cap or dampen the step:

    PA-I :  tau = min( C , hinge / ||x||^2 )
    PA-II:  tau = hinge / ( ||x||^2 + 1/(2C) )

C is the aggressiveness budget: small C = don't trust any single example
too much. Multiclass PA constrains the margin between the true class and
the best wrong class, updating both (promote/demote, like the multiclass
perceptron, but with the optimized step size tau).

The script: (1) verifies each PA step against a brute-force numerical
solve of its little optimization problem; (2) races perceptron vs PA vs
PA-I on a noisy stream — plain PA slams into every mislabeled point,
PA-I's cap C restores robustness; (3) runs multiclass PA-I.

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)


def hinge(w, x, y):
    return max(0.0, 1.0 - y * (w @ x))


# ----------------------------------------------------------------------
# The three binary update rules
# ----------------------------------------------------------------------
def pa_step(w, x, y, variant="PA", C=0.1):
    loss = hinge(w, x, y)
    if loss == 0.0:
        return w                                   # passive
    nx2 = x @ x
    if variant == "PA":
        tau = loss / nx2
    elif variant == "PA-I":
        tau = min(C, loss / nx2)
    else:                                          # PA-II
        tau = loss / (nx2 + 1.0 / (2 * C))
    return w + tau * y * x                         # aggressive


# ----------------------------------------------------------------------
# (1) Verify the closed form against brute force, one step
# ----------------------------------------------------------------------
# The PA step claims to solve: min ||w - w_t||^2 / 2  s.t. hinge = 0.
# Check on a random instance by dense search over a grid around w_t.
w0 = np.array([0.3, -0.2])
x0, y0 = np.array([1.0, 2.0]), 1.0
w_pa = pa_step(w0, x0, y0, "PA")

g = np.linspace(-1.5, 1.5, 601)
GW1, GW2 = np.meshgrid(w0[0] + g, w0[1] + g)
feasible = 1 - y0 * (GW1 * x0[0] + GW2 * x0[1]) <= 1e-9
dist2 = (GW1 - w0[0]) ** 2 + (GW2 - w0[1]) ** 2
dist2[~feasible] = np.inf
brute = np.array([GW1.ravel()[np.argmin(dist2)], GW2.ravel()[np.argmin(dist2)]])

print("--- PA closed form vs brute-force constrained minimization ---")
print(f"closed form : {w_pa}")
print(f"brute force : {brute}")
assert np.allclose(w_pa, brute, atol=0.01)
assert abs(hinge(w_pa, x0, y0)) < 1e-12            # constraint active & met

# ----------------------------------------------------------------------
# (2) Noisy stream: perceptron vs PA vs PA-I
# ----------------------------------------------------------------------
N, ETA = 4_000, 0.07                               # 7% label noise
U_STAR = np.array([0.6, 0.8])
X = rng.uniform(-1, 1, size=(N, 2))
keep = np.abs(X @ U_STAR) >= 0.1
X = X[keep][:3000]
y_clean = np.sign(X @ U_STAR)
flip = rng.uniform(size=len(X)) < ETA
y = np.where(flip, -y_clean, y_clean)

# fresh clean test set to measure what each learner actually learned
X_test = rng.uniform(-1, 1, size=(20_000, 2))
y_test = np.sign(X_test @ U_STAR)


def run_online(rule):
    w = np.zeros(2)
    cum, cum_curve = 0, []
    for i in range(len(X)):
        if y[i] * (w @ X[i]) <= 0:
            cum += 1
        w = rule(w, X[i], y[i])
        cum_curve.append(cum)
    test_err = np.mean(np.sign(X_test @ w) != y_test)
    return w, cum_curve, test_err


perc_rule = lambda w, x, yy: w + yy * x if yy * (w @ x) <= 0 else w
results = {
    "perceptron": run_online(perc_rule),
    "PA": run_online(lambda w, x, yy: pa_step(w, x, yy, "PA")),
    "PA-I (C=0.1)": run_online(lambda w, x, yy: pa_step(w, x, yy, "PA-I", C=0.1)),
}

print(f"\n--- noisy stream ({ETA:.0%} flipped labels), clean test error ---")
for name, (_, curve, err) in results.items():
    print(f"{name:14s}: online mistakes {curve[-1]:4d},  test error {err:.4f}")
# the capped variant must beat plain PA on the clean test set
assert results["PA-I (C=0.1)"][2] < results["PA"][2]

# ----------------------------------------------------------------------
# (3) Multiclass PA-I: constrain the true-vs-best-wrong margin
# ----------------------------------------------------------------------
K = 3
CENTERS = np.array([[1.5, 0.0], [-0.8, 1.3], [-0.8, -1.3]])
Xm = np.vstack([rng.normal(c, 0.45, size=(500, 2)) for c in CENTERS])
ym = np.repeat(np.arange(K), 500)
perm = rng.permutation(len(Xm))
Xm, ym = Xm[perm], ym[perm]


def multiclass_pa(X, yv, K, C=0.1, epochs=3):
    W = np.zeros((K, X.shape[1]))
    for _ in range(epochs):
        for i in range(len(X)):
            scores = W @ X[i]
            wrong = np.argmax(np.where(np.arange(K) == yv[i], -np.inf, scores))
            loss = max(0.0, 1.0 - (scores[yv[i]] - scores[wrong]))
            if loss > 0:
                tau = min(C, loss / (2 * X[i] @ X[i]))
                W[yv[i]] += tau * X[i]             # promote the true class
                W[wrong] -= tau * X[i]             # demote the runner-up
    return W


Wm = multiclass_pa(Xm, ym, K)
acc_m = np.mean((Xm @ Wm.T).argmax(1) == ym)
print(f"\nmulticlass PA-I train accuracy: {acc_m:.3f}")
assert acc_m > 0.95

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

colors = {"perceptron": "#898781", "PA": "#e34948", "PA-I (C=0.1)": "#1baf7a"}
for name, (_, curve, _) in results.items():
    ax1.plot(curve, color=colors[name], lw=1.5, label=name)
ax1.set(xlabel="examples", ylabel="cumulative online mistakes",
        title="noisy stream: mistakes while learning")
ax1.legend(frameon=False, fontsize=8)

names = list(results)
ax2.bar(names, [results[n][2] for n in names],
        color=[colors[n] for n in names])
ax2.axhline(ETA, color="#0b0b0b", ls="--", lw=1, label="noise floor η")
ax2.set(ylabel="clean test error", title="what was actually learned")
ax2.legend(frameon=False, fontsize=8)

gx = np.linspace(-2.5, 3, 300); gy = np.linspace(-2.5, 2.5, 300)
GX, GY = np.meshgrid(gx, gy)
G = np.column_stack([GX.ravel(), GY.ravel()])
regions = (G @ Wm.T).argmax(1).reshape(GX.shape)
ax3.contourf(GX, GY, regions, levels=[-.5, .5, 1.5, 2.5],
             colors=["#dbe7f6", "#ddf1e6", "#fbe9e7"])
ax3.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="viridis", s=4, alpha=.7)
ax3.set(title="multiclass PA-I regions")

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
