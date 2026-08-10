r"""Learnability, empirically: overfitting, shattering, and the two rates.

Companion to learnability.tex (following Shalev-Shwartz & Ben-David,
"Understanding Machine Learning", Part I). Three experiments:

(1) The memorizer: ERM over ALL functions achieves train error 0 and test
    error 1/2 — perfect memory, zero learning. The reason inductive bias
    (a restricted hypothesis class H) is necessary.
(2) Shattering: the threshold class {x -> 1[x > a]} shatters any single
    point but NO pair — the labeling (1, 0) for x1 < x2 is exhaustively
    shown impossible. Hence VCdim(thresholds) = 1.
(3) The two rates, measured — with an honest subtlety. ERM over
    thresholds (d = 1), uniform X, target f(x) = 1[x > 1/2]:
      - realizable (clean labels): E[excess risk] ~ 1/m  (fast rate)
      - with 10% label noise, the UNIFORM-CONVERGENCE supremum
        E[ sup_h |L_S(h) - L_D(h)| ] ~ 1/sqrt(m) — representativeness
        over the WHOLE class, the quantity that costs eps^{-2} samples.
    Subtlety (worst case vs benign): the noisy EXCESS RISK of ERM decays
    ~ 1/m here, faster than the worst-case agnostic bound — uniform noise
    keeps the noise level away from 1/2 (a margin condition), and the
    risk penalty stops ERM from exploiting the empirical fluctuations
    that make the sup large. The eps^{-2} agnostic rate is a worst-case-
    over-distributions statement; benign distributions beat it. The demo
    measures all three slopes and asserts each.

Run me with F5.
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# (1) The memorizer: unrestricted ERM overfits totally
# ----------------------------------------------------------------------
# X ~ Uniform[0,1] (continuous), f(x) = 1[x > 1/2]. The memorizer stores
# the training set and answers 0 on anything unseen. Fresh points are
# almost surely unseen, so its test predictions are the constant 0.
m = 200
x_train = rng.uniform(size=m)
y_train = (x_train > 0.5).astype(int)
x_test = rng.uniform(size=100_000)
y_test = (x_test > 0.5).astype(int)

train_err_memorizer = 0.0                       # by construction
test_err_memorizer = np.mean(y_test != 0)       # predicts 0 everywhere new

print("--- (1) the memorizer ---")
print(f"train error = {train_err_memorizer:.3f}   "
      f"test error = {test_err_memorizer:.3f}  (chance level)")
assert test_err_memorizer > 0.49

# ----------------------------------------------------------------------
# (2) Thresholds cannot shatter two points  =>  VCdim = 1
# ----------------------------------------------------------------------
# h_a(x) = 1[x > a]. On x1 < x2, sweep a over all distinct regions and
# collect the achievable labelings: (1,1), (0,1), (0,0) — never (1,0),
# because x1 < x2 and x1 > a forces x2 > a.
x1, x2 = 0.3, 0.7
achievable = set()
for a in [-1.0, 0.5, 2.0]:                      # a below, between, above
    achievable.add(((x1 > a) * 1, (x2 > a) * 1))

print("\n--- (2) shattering check for thresholds on x1 < x2 ---")
print(f"achievable labelings: {sorted(achievable)}")
print("(1, 0) achievable?   :", (1, 0) in achievable)
assert (1, 0) not in achievable and len(achievable) == 3

# ----------------------------------------------------------------------
# (3) The two rates of the Fundamental Theorem
# ----------------------------------------------------------------------
# ERM for thresholds: pick the empirical-error-minimizing cut among the
# midpoints of consecutive sorted points (all distinct behaviors occur
# there — a Sauer-style fact: tau(m) = m + 1 for thresholds).
def erm_threshold(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    cuts = np.concatenate([[xs[0] - 1], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1]])
    # errors for each cut, computed incrementally: predicting 1[x > a]
    errs = [np.mean(ys != (xs > a)) for a in cuts]
    return cuts[int(np.argmin(errs))]


# True risk under Uniform[0,1] with target threshold 1/2 and noise eta:
#   L_D(h_a) = eta + (1 - 2 eta) |a - 1/2|      (excess = (1-2eta)|a-1/2|)
GRID = np.linspace(0.0, 1.0, 1001)   # a-values for the sup over the class


def sup_gap(x: np.ndarray, y: np.ndarray, eta: float) -> float:
    """sup over the class of |L_S(h_a) - L_D(h_a)| — representativeness.
    Vectorized: with sorted data, #(x <= a) and #(y=1 & x <= a) come from
    searchsorted + prefix sums, giving L_S on the whole grid at once."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    m_i = len(xs)
    ones_prefix = np.concatenate([[0], np.cumsum(ys)])
    k = np.searchsorted(xs, GRID, side="right")     # #points with x <= a
    ones_left = ones_prefix[k]                      # #(y=1, x <= a): predicted 0, wrong
    zeros_right = (m_i - k) - (ones_prefix[m_i] - ones_left)  # #(y=0, x > a)
    L_S = (ones_left + zeros_right) / m_i
    L_D = eta + (1 - 2 * eta) * np.abs(GRID - 0.5)
    return float(np.max(np.abs(L_S - L_D)))


def run_rate_experiment(eta: float, ms: np.ndarray, trials: int):
    """Returns (E[excess risk of ERM], E[sup-gap over the class]) per m."""
    excess, sup = [], []
    for m_i in ms:
        tot_ex, tot_sup = 0.0, 0.0
        for _ in range(trials):
            x = rng.uniform(size=m_i)
            y = (x > 0.5).astype(int)
            flip = rng.uniform(size=m_i) < eta
            y = np.where(flip, 1 - y, y)
            a = erm_threshold(x, y)
            tot_ex += (1 - 2 * eta) * abs(a - 0.5)   # excess over best in class
            tot_sup += sup_gap(x, y, eta)
        excess.append(tot_ex / trials)
        sup.append(tot_sup / trials)
    return np.array(excess), np.array(sup)


ms = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
realizable, _ = run_rate_experiment(eta=0.0, ms=ms, trials=600)
agn_excess, agn_sup = run_rate_experiment(eta=0.1, ms=ms, trials=600)

slope = lambda ys: np.polyfit(np.log(ms), np.log(ys), 1)[0]
slope_real, slope_sup, slope_agn_ex = slope(realizable), slope(agn_sup), slope(agn_excess)

print("\n--- (3) learning rates of ERM over thresholds (VC dim 1) ---")
print(f"realizable excess risk     : slope = {slope_real:.2f}   (theory: -1)")
print(f"noisy sup_h |L_S - L_D|    : slope = {slope_sup:.2f}   (theory: -1/2)")
print(f"noisy excess risk of ERM   : slope = {slope_agn_ex:.2f}   "
      f"(faster than worst case: margin!)")
assert -1.35 < slope_real < -0.75, "realizable rate should be ~ 1/m"
assert -0.75 < slope_sup < -0.30, "uniform-convergence sup should be ~ 1/sqrt(m)"
assert slope_agn_ex < -0.7, "benign noise: margin gives a fast excess rate"
print("all three slopes confirmed")

# ----------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

ax1.plot(x_test[:300], np.zeros(300), ".", ms=3, color="#e34948",
         label="memorizer on fresh points: always 0")
ax1.plot(x_train[:60], y_train[:60], "o", ms=4, color="#2a78d6",
         label="training points: memorized")
ax1.step([0, .5, .5, 1], [0, 0, 1, 1], color="#898781", lw=1.5,
         where="post", label="truth 1[x > ½]")
ax1.set(xlabel="x", ylabel="label", title="overfitting: train 0.0, test 0.5")
ax1.legend(frameon=False, fontsize=7, loc="center left")

ax2.loglog(ms, realizable, "o-", color="#2a78d6",
           label=f"realizable excess (slope {slope_real:.2f})")
ax2.loglog(ms, agn_sup, "s-", color="#eb6834",
           label=f"sup|L_S−L_D| noisy (slope {slope_sup:.2f})")
ax2.loglog(ms, agn_excess, "^-", color="#1baf7a",
           label=f"noisy excess (slope {slope_agn_ex:.2f}, margin)")
ax2.loglog(ms, 0.7 / ms, ":", color="#2a78d6", lw=1, label="~1/m")
ax2.loglog(ms, 0.35 / np.sqrt(ms), ":", color="#eb6834", lw=1, label="~1/√m")
ax2.set(xlabel="m (log)", ylabel="expectation (log)",
        title="learning rates, measured (VC dim 1)")
ax2.legend(frameon=False, fontsize=7)

for ax in (ax1, ax2):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
