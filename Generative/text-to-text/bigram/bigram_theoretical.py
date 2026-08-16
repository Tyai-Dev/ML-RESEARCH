r"""Bigram language model — the theoretical solution: counting IS MLE.

Language modeling in the language of statistics. Model the text as a
Markov chain of characters: each character depends only on the previous
one,

    P(c_1, ..., c_T) = P(c_1) * prod_{i} P(c_{i+1} | c_i).

The parameters are a V x V table p(b|a) = P(next = b | current = a),
one conditional multinoulli per row. (V = 65 for Tiny Shakespeare.)

Step 1 — the likelihood collapses to counts. Let n_ab = number of
times character a is followed by character b in the training text.
Grouping identical factors (the bernoulli move, one level up):

    L(p) = prod_a prod_b p(b|a)^{n_ab}.

Step 2 — the log separates the rows.

    l(p) = sum_a sum_b n_ab log p(b|a)
         = sum_a [ row a's own multinoulli log-likelihood ].

No term couples two rows: the problem splits into V independent
multinoulli MLE problems (Theory/distributions).

The closed form. Maximize row a subject to sum_b p(b|a) = 1 with a
Lagrange multiplier λ:

    d/dp(b|a) [ sum_b n_ab log p(b|a) - λ sum_b p(b|a) ] = 0
      => n_ab / p(b|a) = λ  =>  p(b|a) = n_ab / λ,

and the constraint forces λ = sum_b n_ab = n_a (row total), hence

    p̂(b|a) = n_ab / n_a  —  count and normalize. m/n, one level up.

Optimality, the information-theoretic way (Gibbs): for ANY other row
distribution q, the per-row excess NLL is exactly a KL divergence,

    NLL_a(q) - NLL_a(p̂) = KL(p̂_a || q_a) >= 0,

with equality iff q = p̂. This script verifies that by Monte Carlo:
random challenger distributions must never beat the count table.

The measuring stick. The average NLL per character (in nats) has a
name in language modeling once exponentiated:

    perplexity = exp(NLL)  —  "the effective number of choices per
    character". A uniform guesser scores exp(log 65) = 65; every model
    on the ladder must drive this down on the SAME validation slice.

The crack in the closed form (foreshadowing rung 2): validation
transitions never seen in training get p̂ = 0 and infinite NLL. We
measure how many there are, patch them with Laplace smoothing
(add-alpha), and note the lesson: pure counting cannot generalize to
unseen contexts — the reason the ladder continues.

Run me with F5. Full derivations: bigram.tex.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, load_everything, decode

train_ids, val_ids, chars, stoi, itos = load_everything()
V = len(chars)
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# The closed form: count transitions, normalize rows
# ----------------------------------------------------------------------
counts = np.zeros((V, V), dtype=np.float64)
np.add.at(counts, (train_ids[:-1], train_ids[1:]), 1.0)   # n_ab
row_tot = counts.sum(axis=1, keepdims=True)               # n_a

# p̂(b|a) = n_ab / n_a  (rows with n_a = 0 would be 0/0; every char
# appears mid-text in this corpus, but guard with uniform anyway)
P = np.where(row_tot > 0, counts / np.maximum(row_tot, 1), 1.0 / V)
assert np.allclose(P.sum(axis=1), 1.0), "rows must be distributions"

# Train NLL of the count model: -(1/n) sum_ab n_ab log p̂(b|a),
# summed only where n_ab > 0 (0 log 0 = 0 by convention).
n_pairs = counts.sum()
mask = counts > 0
nll_train = -(counts[mask] * np.log(P[mask])).sum() / n_pairs

print(f"vocabulary size V          : {V}")
print(f"training transitions       : {int(n_pairs):,}")
print(f"train NLL (count MLE)      : {nll_train:.4f} nats/char")
print(f"train perplexity           : {np.exp(nll_train):.2f}"
      f"   (uniform baseline: {V})")

# ----------------------------------------------------------------------
# Gibbs check: no challenger distribution beats the count table
# ----------------------------------------------------------------------
# For random rows a and random challengers q ~ Dirichlet(1), the excess
# row NLL sum_b n_ab (log p̂ - log q) = n_a * KL(p̂_a || q_a) must be >= 0.
worst = np.inf
for _ in range(500):
    a = int(rng.integers(V))
    if row_tot[a, 0] == 0:
        continue
    q = rng.dirichlet(np.ones(V))
    m_ = counts[a] > 0
    excess = (counts[a][m_] * (np.log(P[a][m_]) - np.log(q[m_]))).sum()
    worst = min(worst, excess)
assert worst >= 0, "a challenger beat the MLE — impossible if Gibbs holds!"
print(f"Gibbs check (500 trials)   : min excess NLL = {worst:.3f} >= 0: OK")

# ----------------------------------------------------------------------
# Validation: the zero-count catastrophe, and the Laplace patch
# ----------------------------------------------------------------------
val_a, val_b = val_ids[:-1], val_ids[1:]
unseen = int((counts[val_a, val_b] == 0).sum())
print(f"val transitions unseen in train: {unseen} of {len(val_a):,}"
      f"  -> pure MLE val NLL = infinity!")

ALPHA = 1.0                                   # Laplace: pretend each
P_smooth = (counts + ALPHA) / (row_tot + ALPHA * V)   # pair was seen once
nll_val = -np.log(P_smooth[val_a, val_b]).mean()
print(f"val NLL (alpha={ALPHA:g} smoothing): {nll_val:.4f} nats/char"
      f"   perplexity {np.exp(nll_val):.2f}")

# ----------------------------------------------------------------------
# Sample from the model — the first machine-generated "Shakespeare"
# ----------------------------------------------------------------------
def sample(P_table, n_chars=300, start="\n"):
    """Ancestral sampling: walk the Markov chain."""
    out, a = [], stoi[start]
    for _ in range(n_chars):
        a = int(rng.choice(V, p=P_table[a]))
        out.append(a)
    return decode(out, itos)

print("\n--- sample from the count model " + "-" * 30)
print(sample(P))
print("-" * 62)

# 'q' is followed by 'u' — the model must have learned English's
# hardest bigram rule
assert itos[int(P[stoi["q"]].argmax())] == "u", "q should predict u!"
print("sanity: argmax P(.|'q') = 'u': OK")

# ----------------------------------------------------------------------
# Picture: the whole model is one heatmap + a few rows of it
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

im = ax1.imshow(np.log1p(counts), cmap="viridis", aspect="auto")
ticks = [stoi[c] for c in ["\n", " ", ".", "A", "I", "T", "a", "e",
                           "o", "q", "t", "z"]]
ax1.set(xticks=ticks, yticks=ticks,
        xticklabels=[repr(itos[t])[1:-1] for t in ticks],
        yticklabels=[repr(itos[t])[1:-1] for t in ticks],
        xlabel="next char b", ylabel="current char a",
        title=r"the entire model: $\log(1 + n_{ab})$")
fig.colorbar(im, ax=ax1, shrink=.85)

# a few rows of the table = a few conditional distributions
for c, color in [("q", "#eb6834"), ("t", "#2a78d6"), (" ", "#3d9b35")]:
    ax2.plot(P[stoi[c]], color=color, lw=1.4,
             label=f"P( · | {repr(c)} )")
top = [("q", "u"), ("t", "h"), (" ", "t")]
for (c, b), color in zip(top, ["#eb6834", "#2a78d6", "#3d9b35"]):
    ax2.annotate(repr(b), (stoi[b], P[stoi[c], stoi[b]]),
                 color=color, fontsize=9, ha="center",
                 xytext=(0, 4), textcoords="offset points")
ax2.set(xlabel="next char index b", ylabel="probability",
        title="three rows of the table (conditional multinoullis)")
ax2.legend(frameon=False, fontsize=8)
ax2.grid(alpha=.3)
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

fig.tight_layout()
plt.show()
