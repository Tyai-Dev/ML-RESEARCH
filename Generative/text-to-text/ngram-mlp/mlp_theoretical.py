r"""Why counting dies — the measurement that forces rung 2.

Rung 1 (Generative/text-to-text/bigram) proved that with ONE character of context, the MLE
is a count table and nothing can beat it on the training set. The
obvious next move is more context: model P(next | previous k chars) by
counting k-grams. This script does exactly that, honestly, for
k = 1..5 — and measures the catastrophe.

The mathematics of the failure. The k-gram count model is still a
perfectly good MLE (the rung-1 Lagrange argument applies verbatim, one
row per CONTEXT): p̂(b | ctx) = n_{ctx,b} / n_ctx. But the number of
possible contexts is V^k:

    k=1: 65 rows.   k=3: 274,625.   k=5: 1.16 * 10^9 rows.

The training set has only ~1M positions, so for k >= 3 most contexts
that CAN occur were never seen. Two symptoms, both measured below:

  1. UNSEEN CONTEXTS: the fraction of validation positions whose
     context never appears in training. For those the count model has
     no row at all — it knows literally nothing, and must fall back to
     a uniform guess over V characters (log 65 = 4.17 nats each).
  2. THIN ROWS: even seen contexts have tiny n_ctx, so their rows are
     noisy one-or-two-sample estimates (variance p(1-p)/n_ctx with
     n_ctx ~ 1: the estimator report card with n = 1).

Result: validation NLL improves from k=1 to k=2..3, then turns around
and gets WORSE with more context — more information available, worse
model — because the parameters-per-datapoint ratio explodes. That
U-turn is measured and plotted here; it is the entire reason
parametric models exist.

The fix (this rung's practical files): stop giving every context its
own row. EMBED characters into R^d, so similar contexts SHARE
parameters, and let a neural network map embedded contexts to logits
(Bengio et al. 2003). Parameters grow LINEARLY in k, not
exponentially, and unseen contexts get sensible predictions because
their characters were seen. mlp_practical_pytorch.py must beat every
count model in this file on the same validation slice.

Run me with F5. Derivations: ngram-mlp.tex.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_everything

train_ids, val_ids, chars, stoi, itos = load_everything()
V = len(chars)
ALPHA = 1.0                                     # Laplace smoothing

K_RANGE = [1, 2, 3, 4, 5]
rows = []
for k in K_RANGE:
    # sparse count table: dict context-tuple -> length-V count vector
    table = defaultdict(lambda: np.zeros(V))
    for i in range(k, len(train_ids)):
        table[tuple(train_ids[i - k:i])][train_ids[i]] += 1

    # validation NLL with Laplace smoothing; unseen context -> uniform
    nll, unseen = 0.0, 0
    n_val = len(val_ids) - k
    for i in range(k, len(val_ids)):
        ctx = tuple(val_ids[i - k:i])
        if ctx in table:
            c = table[ctx]
            nll += -np.log((c[val_ids[i]] + ALPHA) / (c.sum() + ALPHA * V))
        else:
            unseen += 1
            nll += np.log(V)                     # knows nothing: uniform
    nll /= n_val

    rows.append(dict(k=k, contexts_possible=V ** k,
                     contexts_seen=len(table),
                     unseen_frac=unseen / n_val, val_nll=nll))
    print(f"k={k}:  possible contexts {V ** k:>12,}   "
          f"seen {len(table):>9,}   unseen val {unseen / n_val:7.3%}   "
          f"val NLL {nll:.4f}   PPL {np.exp(nll):7.2f}")

# The U-turn: more context first helps, then hurts.
nlls = [r["val_nll"] for r in rows]
best_k = K_RANGE[int(np.argmin(nlls))]
assert nlls[-1] > min(nlls), "no U-turn?! counting should die by k=5"
print(f"\ncounting's sweet spot: k={best_k} "
      f"(val NLL {min(nlls):.4f}); beyond it, MORE context makes the "
      f"model WORSE. This is the curse of dimensionality, measured.")

# ----------------------------------------------------------------------
# Picture: the U-turn, the unseen-context explosion, the table blowup
# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.5, 3.8))

ax1.plot(K_RANGE, nlls, "o-", color="#eb6834", lw=2)
ax1.axhline(np.log(V), color="#898781", ls=":", lw=1,
            label=f"uniform log V = {np.log(V):.2f}")
ax1.annotate("more context,\nworse model", xy=(K_RANGE[-1], nlls[-1]),
             xytext=(-72, -30), textcoords="offset points", fontsize=8,
             arrowprops=dict(arrowstyle="->", color="#555"))
ax1.set(xlabel="context length k", ylabel="val NLL (nats/char)",
        title="the U-turn: counting's death", xticks=K_RANGE)
ax1.legend(frameon=False, fontsize=8)

ax2.plot(K_RANGE, [r["unseen_frac"] * 100 for r in rows], "o-",
         color="#2a78d6", lw=2)
ax2.set(xlabel="context length k", ylabel="% of val contexts unseen",
        title="what the model has never seen", xticks=K_RANGE)

ax3.semilogy(K_RANGE, [r["contexts_possible"] for r in rows], "o-",
             color="#898781", lw=2, label=r"possible: $V^k$")
ax3.semilogy(K_RANGE, [r["contexts_seen"] for r in rows], "o-",
             color="#3d9b35", lw=2, label="actually seen")
ax3.axhline(len(train_ids), color="#111", ls="--", lw=1,
            label=f"training chars ({len(train_ids):,})")
ax3.set(xlabel="context length k", ylabel="number of contexts",
        title="the table outgrows the data", xticks=K_RANGE)
ax3.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()
