r"""Loss functions, empirically: surrogates, calibration, robustness.

Companion to losses.tex. Four checks:

(1) The margin-loss family plotted on one axis (m = y f(x)): 0-1, hinge,
    squared hinge, logistic, exponential — and a pointwise assert that
    every convex surrogate upper-bounds the 0-1 loss.
(2) CALIBRATION: what does each loss recover at a point where
    P(Y=+1) = p? Minimize the population risk over a constant score f:
      - logistic: f* = logit(p)   -> sigma(f*) = p   (full probability!)
      - squared : f* = 2p - 1     -> p recoverable   (linear in p)
      - hinge   : f* = sign(2p-1) -> only the CLASS  (probability lost)
    All three still classify like Bayes (sign(f*) = sign(p - 1/2)):
    classification-calibrated, but carrying different information.
(3) ROBUSTNESS in regression: fitting a constant under squared loss gives
    the MEAN, under absolute loss the MEDIAN; with 10% gross outliers the
    mean is dragged, the median stands still. Huber interpolates.
(4) The exponential loss's fragility: its penalty on a mislabeled
    far-away point is e^{|f|} vs hinge's |f| — printed side by side.

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# (1) The margin family, and the surrogate property
# ----------------------------------------------------------------------
m = np.linspace(-3, 3, 601)
zero_one = (m <= 0).astype(float)
hinge = np.maximum(0, 1 - m)
sq_hinge = np.maximum(0, 1 - m) ** 2
logistic = np.log2(1 + np.exp(-m))          # base 2: value 1 at m = 0
exponential = np.exp(-m)

surrogates = {"hinge": hinge, "squared hinge": sq_hinge,
              "logistic (base 2)": logistic, "exponential": exponential}
for name, s in surrogates.items():
    assert np.all(s >= zero_one - 1e-12), f"{name} fails to dominate 0-1"
print("--- (1) every surrogate upper-bounds the 0-1 loss: OK ---")

# ----------------------------------------------------------------------
# (2) Calibration: population minimizers at P(Y=+1) = p
# ----------------------------------------------------------------------
# risk(f) = p * loss(f) + (1-p) * loss(-f); minimize over scalar f by grid.
fgrid = np.linspace(-6, 6, 4001)


def pop_minimizer(loss_fn, p):
    risk = p * loss_fn(fgrid) + (1 - p) * loss_fn(-fgrid)
    return fgrid[np.argmin(risk)]


losses_cal = {
    "logistic": lambda f: np.log(1 + np.exp(-f)),
    "squared": lambda f: (1 - f) ** 2,
    "hinge": lambda f: np.maximum(0, 1 - f),
}
theory = {
    "logistic": lambda p: np.log(p / (1 - p)),   # logit(p)
    "squared": lambda p: 2 * p - 1,
    "hinge": lambda p: np.sign(2 * p - 1),       # only the class survives
}

print("\n--- (2) population minimizers at P(Y=+1)=p ---")
print(f"{'p':>5} | {'logistic f*':>12} {'logit(p)':>9} | "
      f"{'squared f*':>10} {'2p-1':>6} | {'hinge f*':>8} {'sign':>5}")
for p in [0.2, 0.4, 0.6, 0.9]:
    row = []
    for name in ["logistic", "squared", "hinge"]:
        f_star = pop_minimizer(losses_cal[name], p)
        expected = theory[name](p)
        assert abs(f_star - expected) < 0.02, (name, p, f_star, expected)
        # every loss is classification-calibrated: sign matches Bayes
        assert np.sign(f_star) == np.sign(p - 0.5)
        row += [f_star, expected]
    print(f"{p:5.1f} | {row[0]:12.3f} {row[1]:9.3f} | "
          f"{row[2]:10.3f} {row[3]:6.2f} | {row[4]:8.2f} {row[5]:5.0f}")
print("logistic recovers p exactly (via sigmoid); hinge keeps only the sign")

# ----------------------------------------------------------------------
# (3) Robustness: mean vs median vs Huber under outliers
# ----------------------------------------------------------------------
clean = rng.normal(5.0, 1.0, size=900)
outliers = rng.normal(50.0, 5.0, size=100)          # 10% gross corruption
data = np.concatenate([clean, outliers])

cgrid = np.linspace(0, 60, 6001)


def fit_constant(loss):
    risks = [np.mean(loss(data - c)) for c in cgrid]
    return cgrid[np.argmin(risks)]


huber = lambda r, d=1.0: np.where(np.abs(r) <= d, 0.5 * r**2,
                                  d * (np.abs(r) - 0.5 * d))
c_sq = fit_constant(lambda r: r**2)
c_abs = fit_constant(np.abs)
c_hub = fit_constant(huber)

print("\n--- (3) fitting a constant to 90% N(5,1) + 10% N(50,5) ---")
print(f"squared  (-> mean)  : {c_sq:6.2f}   (dragged by outliers; "
      f"mean = {data.mean():.2f})")
print(f"absolute (-> median): {c_abs:6.2f}   (median = {np.median(data):.2f})")
print(f"Huber (delta=1)     : {c_hub:6.2f}")
assert abs(c_sq - data.mean()) < 0.05
assert abs(c_abs - np.median(data)) < 0.05
assert abs(c_abs - 5.0) < 0.5 and abs(c_hub - 5.0) < 0.5 and c_sq > 8

# ----------------------------------------------------------------------
# (4) Exponential loss vs hinge on a badly mislabeled point
# ----------------------------------------------------------------------
print("\n--- (4) penalty on a mislabeled point at score f ---")
for f in [2, 5, 10]:
    print(f"f = {f:2d}:  hinge = {1+f:4d}   exponential = {np.exp(f):12.1f}")
print("exponential's e^|f| makes AdaBoost exquisitely outlier-sensitive")

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

ax1.plot(m, zero_one, color="#0b0b0b", lw=2, label="0-1")
for (name, s), c in zip(surrogates.items(),
                        ["#2a78d6", "#1baf7a", "#eb6834", "#e87ba4"]):
    ax1.plot(m, s, color=c, lw=1.5, label=name)
ax1.set(xlabel="margin m = y f(x)", ylabel="loss", ylim=(0, 4),
        title="the margin family: convex roofs over 0-1")
ax1.legend(frameon=False, fontsize=8)

ps = np.linspace(0.02, 0.98, 100)
ax2.plot(ps, [1/(1+np.exp(-pop_minimizer(losses_cal["logistic"], p)))
              for p in ps], color="#2a78d6", lw=2, label="sigmoid(logistic f*)")
ax2.plot(ps, [(pop_minimizer(losses_cal["squared"], p) + 1) / 2
              for p in ps], color="#1baf7a", lw=2, label="(squared f*+1)/2")
ax2.plot(ps, [(pop_minimizer(losses_cal["hinge"], p) + 1) / 2
              for p in ps], color="#eb6834", lw=2, label="(hinge f*+1)/2")
ax2.plot([0, 1], [0, 1], ":", color="#898781", lw=1)
ax2.set(xlabel="true p", ylabel="recovered", title="calibration: what survives each loss")
ax2.legend(frameon=False, fontsize=8)

ax3.hist(data, bins=100, color="#c3c2b7")
for c, col, lbl in [(c_sq, "#e34948", "squared"), (c_abs, "#1baf7a", "absolute"),
                    (c_hub, "#2a78d6", "Huber")]:
    ax3.axvline(c, color=col, lw=2, label=f"{lbl}: {c:.1f}")
ax3.set(xlabel="value", ylabel="count", title="10% outliers: mean dragged, median firm")
ax3.legend(frameon=False, fontsize=8)

print("\nall checks passed")
for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
