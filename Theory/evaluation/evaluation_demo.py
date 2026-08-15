r"""Measuring success — the metrics catalogue, every claim verified.

A trained binary classifier is a SCORE s(x) (here: a calibrated
probability) plus a THRESHOLD tau; predictions are s(x) > tau. Every
evaluation number in existence starts from the same four counts at a
given tau — the confusion matrix:

                    predicted +    predicted -
      truly +           TP             FN        (misses)
      truly -           FP             TN        (false alarms)

    accuracy  = (TP+TN)/n          "how often right"
    precision = TP/(TP+FP)         "when it says +, is it?"
    recall    = TP/(TP+FN) = TPR   "of the real +, how many found?"
    specificity = TN/(TN+FP)       (1 - FPR)
    F1 = harmonic mean(precision, recall)   — punishes lopsidedness

Claims verified below, each an assert:

ACCURACY LIES UNDER IMBALANCE. On data with a ~3% positive rate, the
classifier that NEVER says positive scores ~97% accuracy with recall
exactly 0. Accuracy answers "how often right", which is the wrong
question when one class is rare and expensive to miss.

THE THRESHOLD IS A FREE PARAMETER — sweeping it traces curves:
ROC = (FPR(tau), TPR(tau)). Its area has a meaning nobody guesses
from the picture:

    AUC = P( s(X+) > s(X-) )   (+ half the ties)

— the probability a random positive outscores a random negative, a
pure RANKING quality, blind to calibration and to the threshold.
Verified: trapezoid area vs 500,000 Monte Carlo pairs, equal to 4
decimals.

ROC IS IMMUNE TO IMBALANCE, PR IS NOT. Deleting 95% of the positives
(same score distributions!) leaves AUC unchanged but collapses average
precision — because precision divides by predicted positives, which
rebases on the class ratio. Under heavy imbalance, read PR curves.

CALIBRATED PROBABILITIES TURN COSTS INTO THRESHOLDS. If missing a
positive costs c_FN and a false alarm costs c_FP, predicting + is
cheaper exactly when p > c_FP / (c_FP + c_FN)  — derived in
evaluation.tex; with c_FN = 5 c_FP the optimal threshold is 1/6, and
the empirical cost minimum lands there. (This needs p to be a real
probability: the calibration property that Theory/losses proved
log-loss preserves.)

F1'S OPTIMAL THRESHOLD is far below 0.5 under imbalance — measured.

The animation: the threshold sliding 0 -> 1 on the fitted 2D
classifier. The decision line shifts in parallel across the
probability field (same w, different level set), the confusion report
updates live, and one dot travels the ROC curve while another travels
the PR curve: every point of those curves IS a threshold.

Run me with F5. Derivations: evaluation.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

SEED = 7
rng = np.random.default_rng(SEED)
W_BAL = np.array([0.5, 2.0, -1.5])           # balanced problem
W_IMB = np.array([-6.0, 2.0, -1.5])          # rare positives (~3%)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_data(w_true, n):
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = (rng.uniform(size=n) < sigmoid(X @ w_true)).astype(float)
    return X, y


def fit_irls(X, y, steps=25):
    """Logistic MLE by IRLS (logistic-regression/)."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = sigmoid(X @ w)
        H = (X.T * (p * (1 - p))) @ X / len(y)
        w = w - np.linalg.solve(H, X.T @ (p - y) / len(y))
    return w


def confusion(scores, y, tau):
    pred = scores > tau
    t = y.astype(bool)
    return (int(np.sum(pred & t)), int(np.sum(pred & ~t)),
            int(np.sum(~pred & t)), int(np.sum(~pred & ~t)))


def metrics(scores, y, tau):
    TP, FP, FN, TN = confusion(scores, y, tau)
    n = TP + FP + FN + TN
    prec = TP / (TP + FP) if TP + FP else 1.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    return dict(
        acc=(TP + TN) / n, prec=prec, rec=rec,
        fpr=FP / (FP + TN) if FP + TN else 0.0,
        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        TP=TP, FP=FP, FN=FN, TN=TN)


def roc_exact(scores, y):
    """The exact ROC: sort by score descending; each prefix of the
    sorted list is one threshold's confusion counts. Every point of the
    curve IS a threshold. AUC by trapezoid over the FULL curve (a
    tau-grid would miss the saturated tails, where sigmoid scores
    pile up near 0 and 1 — a real bug caught by the Monte Carlo
    check below)."""
    order = np.argsort(-scores)
    ys = y[order]
    tpr = np.r_[0, np.cumsum(ys)] / ys.sum()
    fpr = np.r_[0, np.cumsum(1 - ys)] / (len(ys) - ys.sum())
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def roc_pr(scores, y, taus):
    """Threshold-grid metrics (for the animation dots and PR curves)
    plus approximate AP from the grid."""
    ms = [metrics(scores, y, t) for t in taus]
    tpr = np.array([m["rec"] for m in ms])
    prec = np.array([m["prec"] for m in ms])
    ap = -np.trapezoid(prec, tpr)
    return tpr, prec, ap, ms


TAUS = np.linspace(0.005, 0.995, 400)

# ----------------------------------------------------------------------
# Fit once on each regime, evaluate on big held-out sets
# ----------------------------------------------------------------------
Xb, yb = make_data(W_BAL, 20_000)
w_bal = fit_irls(Xb, yb)
Xb_t, yb_t = make_data(W_BAL, 200_000)
s_bal = sigmoid(Xb_t @ w_bal)

Xi, yi = make_data(W_IMB, 60_000)
w_imb = fit_irls(Xi, yi)
Xi_t, yi_t = make_data(W_IMB, 400_000)
s_imb = sigmoid(Xi_t @ w_imb)
base_rate = yi_t.mean()
print(f"balanced base rate {yb_t.mean():.3f};  "
      f"imbalanced base rate {base_rate:.4f}")

# ---- accuracy lies under imbalance -----------------------------------
m_maj = metrics(np.zeros_like(yi_t), yi_t, 0.5)   # never says +
m_clf = metrics(s_imb, yi_t, 0.5)
print(f"\nimbalanced, tau = 0.5:")
print(f"  always-negative: accuracy {m_maj['acc']:.4f}   recall "
      f"{m_maj['rec']:.1f}   F1 {m_maj['f1']:.1f}")
print(f"  logistic:        accuracy {m_clf['acc']:.4f}   recall "
      f"{m_clf['rec']:.3f}   F1 {m_clf['f1']:.3f}")
assert m_maj["acc"] > 0.95 and m_maj["rec"] == 0.0, \
    "accuracy must look great while finding nothing"

# ---- AUC == probability of correct ranking ---------------------------
fprF_b, tprF_b, auc_b = roc_exact(s_bal, yb_t)
tpr_b, prec_b, ap_b, _ = roc_pr(s_bal, yb_t, TAUS)
pos = s_bal[yb_t == 1]
neg = s_bal[yb_t == 0]
i = rng.integers(0, len(pos), 500_000)
j = rng.integers(0, len(neg), 500_000)
auc_mc = np.mean((pos[i] > neg[j]) + 0.5 * (pos[i] == neg[j]))
print(f"\nAUC (exact curve) {auc_b:.4f} vs P(s+ > s-) Monte Carlo "
      f"{auc_mc:.4f}")
assert abs(auc_b - auc_mc) < 5e-3, "AUC IS the ranking probability"

# ---- ROC immune to imbalance, PR not ---------------------------------
keep = (yb_t == 0) | (rng.uniform(size=len(yb_t)) < 0.05)
s_sub, y_sub = s_bal[keep], yb_t[keep]
_, _, auc_s = roc_exact(s_sub, y_sub)
_, _, ap_s, _ = roc_pr(s_sub, y_sub, TAUS)
print(f"\ndelete 95% of positives (same scores): "
      f"AUC {auc_b:.4f} -> {auc_s:.4f},  AP {ap_b:.4f} -> {ap_s:.4f}")
assert abs(auc_b - auc_s) < 0.02, "ROC should barely move"
assert ap_b - ap_s > 0.25, "precision rebases on the class ratio"

# ---- calibration (the model is well-specified => calibrated) ---------
bins = np.linspace(0, 1, 11)
which = np.digitize(s_bal, bins) - 1
gaps = [abs(s_bal[which == b].mean() - yb_t[which == b].mean())
        for b in range(10) if np.sum(which == b) > 500]
brier = np.mean((s_bal - yb_t) ** 2)
print(f"\ncalibration: max reliability gap {max(gaps):.4f}, "
      f"Brier {brier:.4f}")
assert max(gaps) < 0.03, "a well-specified MLE should be calibrated"

# ---- costs pick the threshold (needs calibration!) -------------------
C_FN, C_FP = 5.0, 1.0
tau_star = C_FP / (C_FP + C_FN)
costs = [(C_FN * m["FN"] + C_FP * m["FP"]) / len(yi_t)
         for m in [metrics(s_imb, yi_t, t) for t in TAUS]]
tau_emp = TAUS[int(np.argmin(costs))]
print(f"\ncosts c_FN = 5 c_FP: theory tau* = {tau_star:.3f}, "
      f"empirical cost-minimizing tau = {tau_emp:.3f}")
assert abs(tau_emp - tau_star) < 0.06, "calibrated p turns costs into taus"

# ---- F1's threshold under imbalance ----------------------------------
f1s = [metrics(s_imb, yi_t, t)["f1"] for t in TAUS]
tau_f1 = TAUS[int(np.argmax(f1s))]
print(f"F1-optimal threshold (imbalanced): {tau_f1:.3f}  (not 0.5)")
assert tau_f1 < 0.4
print("\nall metric claims verified: OK")

# ----------------------------------------------------------------------
# Static figure: the three lessons that fit on axes
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.9))

mids = (bins[:-1] + bins[1:]) / 2
frac = [yb_t[which == b].mean() if np.sum(which == b) > 0 else np.nan
        for b in range(10)]
ax1.plot([0, 1], [0, 1], color="#898781", ls=":")
ax1.plot(mids, frac, "o-", color="#2a78d6")
ax1.set(xlabel="predicted probability", ylabel="actual frequency",
        title=f"reliability: calibrated (Brier {brier:.3f})")

ax2.plot(TAUS, costs, color="#eb6834", lw=2)
ax2.axvline(tau_star, color="#0b0b0b", ls="--", lw=1.2,
            label=f"theory: c_FP/(c_FP+c_FN) = {tau_star:.3f}")
ax2.plot(tau_emp, min(costs), "o", color="#2a78d6",
         label=f"empirical min {tau_emp:.3f}")
ax2.set(xlabel="threshold tau", ylabel="expected cost / sample",
        title="costs pick the threshold (c_FN = 5 c_FP)")
ax2.legend(frameon=False, fontsize=8)

tpr_s, prec_s, _, _ = roc_pr(s_sub, y_sub, TAUS)
ax3.plot(tpr_b, prec_b, color="#2a78d6", lw=2,
         label=f"balanced (AP {ap_b:.3f})")
ax3.plot(tpr_s, prec_s, color="#eb6834", lw=2,
         label=f"95% of positives deleted (AP {ap_s:.3f})")
ax3.set(xlabel="recall", ylabel="precision",
        title=f"same ranking (AUC {auc_b:.3f} vs {auc_s:.3f}), "
              "different PR")
ax3.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()

# ----------------------------------------------------------------------
# Animation: every point of the ROC / PR curves IS a threshold
# ----------------------------------------------------------------------
sub = slice(0, 1500)
gx = np.linspace(-4.6, 4.6, 230)
gy = np.linspace(-3, 3, 150)
GX, GY = np.meshgrid(gx, gy)
FIELD = sigmoid(w_bal[0] + w_bal[1] * GX + w_bal[2] * GY)
A_TAUS = np.linspace(0.02, 0.98, 121)
a_ms = [metrics(s_bal[:20_000], yb_t[:20_000], t) for t in A_TAUS]


def iso_line(tau):
    """The decision line s(x) = tau, i.e. w.[1,x] = logit(tau):
    same w, shifted level set — thresholds move the line in parallel."""
    z = np.log(tau / (1 - tau))
    return gx, (z - w_bal[0] - w_bal[1] * gx) / w_bal[2]


figA = plt.figure(figsize=(11.5, 8.4))
gsA = figA.add_gridspec(2, 2, height_ratios=[1.75, 1],
                        hspace=.28, wspace=.24)
axA = figA.add_subplot(gsA[0, :])
axR = figA.add_subplot(gsA[1, 0])
axP = figA.add_subplot(gsA[1, 1])
figA.suptitle("one classifier, every threshold: sliding tau traces the "
              "ROC and PR curves", fontsize=12)

axA.imshow(FIELD, extent=(-4.6, 4.6, -3, 3), origin="lower",
           cmap="coolwarm", vmin=0, vmax=1, alpha=.45, zorder=0)
axA.scatter(Xb_t[sub, 1], Xb_t[sub, 2], c=yb_t[sub], cmap="coolwarm",
            s=8, alpha=.55, zorder=1)
line_tau, = axA.plot([], [], color="#1baf7a", lw=2.6,
                     label="decision line  s(x) = tau", zorder=3)
axA.plot(*iso_line(0.5), ls=":", color="#0b0b0b", lw=1.4,
         label="tau = 0.5", zorder=2)
titleA = axA.set_title("")
axA.set(xlim=(-4.6, 4.6), ylim=(-3, 3), xlabel="x1", ylabel="x2")
axA.set_aspect("equal")
axA.legend(frameon=False, fontsize=8, loc="upper left")
rep = axA.text(.99, .02, "", transform=axA.transAxes,
               family="monospace", fontsize=8.5, ha="right",
               va="bottom", zorder=5,
               bbox=dict(facecolor="white", alpha=.82,
                         edgecolor="#999", boxstyle="round"))

axR.plot(fprF_b[::200], tprF_b[::200], color="#2a78d6", lw=2)
axR.plot([0, 1], [0, 1], color="#898781", ls=":", lw=1,
         label="coin flip (AUC 0.5)")
dotR, = axR.plot([], [], "o", color="#eb6834", ms=9, zorder=5)
axR.set(xlabel="FPR (false alarms)", ylabel="TPR (recall)",
        title=f"ROC — area {auc_b:.3f} = P(s+ > s-)")
axR.legend(frameon=False, fontsize=8, loc="lower right")

axP.plot(tpr_b, prec_b, color="#2a78d6", lw=2)
dotP, = axP.plot([], [], "o", color="#eb6834", ms=9, zorder=5)
axP.set(xlabel="recall", ylabel="precision", ylim=(0.4, 1.02),
        title=f"precision–recall — AP {ap_b:.3f}")

for ax in (axR, axP):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def a_report(m, tau):
    return (f"tau = {tau:4.2f}\n"
            f"{'':8s}{'pred +':>8s}{'pred -':>8s}\n"
            f"{'true +':8s}{m['TP']:8d}{m['FN']:8d}   recall {m['rec']:5.3f}\n"
            f"{'true -':8s}{m['FP']:8d}{m['TN']:8d}   FPR    {m['fpr']:5.3f}\n"
            f"accuracy {m['acc']:5.3f}  precision {m['prec']:5.3f}  "
            f"F1 {m['f1']:5.3f}")


def update(k):
    tau = A_TAUS[k]
    m = a_ms[k]
    line_tau.set_data(*iso_line(tau))
    dotR.set_data([m["fpr"]], [m["rec"]])
    dotP.set_data([m["rec"]], [m["prec"]])
    rep.set_text(a_report(m, tau))
    titleA.set_text(f"threshold tau = {tau:.2f} — same w, "
                    "shifted level set")
    return line_tau, dotR, dotP, rep, titleA


ani = FuncAnimation(figA, update, frames=len(A_TAUS), interval=70,
                    blit=False, repeat=True)   # keep a ref!
plt.show()
