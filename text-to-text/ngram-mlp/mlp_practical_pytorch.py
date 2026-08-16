r"""The neural n-gram, trained for real — and watching it learn English.

mlp_practical_pure.py earned autograd (hand backprop == torch to 1e-16);
this file uses it: the same architecture (k=8 context, d=24 embeddings,
256 tanh hidden units, ~68k parameters) trained properly on the GPU
with Adam, minibatches, and a learning-rate drop.

The claim to verify (the ladder's contract): this model must beat every
count model on the SAME validation slice — including counting's k=3
sweet spot (val NLL 1.9526) — because parameter sharing through the
embedding table lets it use k=8 contexts without the V^8 table. The
assert at the bottom is the rung's gate.

The animation (this rung's payoff): the EMBEDDING SPACE ORGANIZING.
Each character starts at a random point in R^24; we project every
checkpoint onto the final embedding's top-2 PCA plane (a fixed basis,
so motion is real, not axis wobble) and watch structure emerge: vowels
drift together, punctuation separates from letters, capitals cluster —
alongside the falling loss and the sharpening samples. Nobody told the
model what a vowel is. The loss carved it.

Run me with F5. Derivations: ngram-mlp.tex.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, get_device, load_everything, decode
from mlp_practical_pure import CONTEXT_K, EMB_D, HIDDEN

BATCH = 256
STEPS = 12_000
LR_SCHEDULE = [(1e-3, 8_000), (1e-4, 4_000)]    # simple step decay
CHECK_EVERY = 200                                # animation checkpoints
COUNT_BEST = 1.9526                              # k=3 count model (rung gate)
BIGRAM_VAL = 2.4819                              # rung-1 number


class NGramMLP(torch.nn.Module):
    """Bengio 2003, minimal: embed, concatenate, tanh, logits."""

    def __init__(self, V):
        super().__init__()
        self.C = torch.nn.Embedding(V, EMB_D)
        self.fc1 = torch.nn.Linear(CONTEXT_K * EMB_D, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, V)

    def forward(self, x):                        # x: (B, k) int
        e = self.C(x).flatten(1)                 # (B, kd)
        return self.fc2(torch.tanh(self.fc1(e)))  # logits (B, V)


def windows_tensor(ids: np.ndarray, device):
    """All (context, target) pairs as GPU tensors."""
    T = len(ids) - CONTEXT_K
    X = np.stack([ids[i:i + CONTEXT_K] for i in range(T)])
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(ids[CONTEXT_K:].copy()).to(device)
    return Xt, Yt


@torch.no_grad()
def full_nll(model, X, Y, batch=8192):
    """Exact average NLL over an entire split, batched."""
    total, n = 0.0, 0
    for i in range(0, len(X), batch):
        logits = model(X[i:i + batch])
        total += F.cross_entropy(logits, Y[i:i + batch],
                                 reduction="sum").item()
        n += len(Y[i:i + batch])
    return total / n


@torch.no_grad()
def sample_text(model, itos, device, n_chars=200, seed=SEED):
    """Ancestral sampling: feed the model its own output."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ctx = [0] * CONTEXT_K                        # start on newlines
    out = []
    for _ in range(n_chars):
        x = torch.tensor([ctx], device=device)
        p = torch.softmax(model(x)[0], dim=0).cpu()
        c = int(torch.multinomial(p, 1, generator=g))
        out.append(c)
        ctx = ctx[1:] + [c]
    return decode(out, itos)


def char_color(c: str) -> str:
    """Color code for the embedding scatter."""
    if c in "aeiou":
        return "#eb6834"          # vowels
    if c in "AEIOU":
        return "#f0a24f"          # capital vowels
    if c.isalpha():
        return "#2a78d6" if c.islower() else "#7fb3e8"  # consonants
    if c in " \n":
        return "#3d9b35"          # whitespace
    return "#898781"              # punctuation & digits


def animate(snaps, steps_hist, tr_hist, va_hist, texts, chars):
    """Embedding space organizing + loss progression + samples."""
    final = snaps[-1] - snaps[-1].mean(0)
    _, _, Vt = np.linalg.svd(final, full_matrices=False)
    basis = Vt[:2].T                             # fixed PCA plane
    proj = [(S - S.mean(0)) @ basis for S in snaps]
    allp = np.concatenate(proj)
    pad = 0.1 * (allp.max() - allp.min())

    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.15])
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
    fig.suptitle("the embedding space organizing itself "
                 "(nobody told it what a vowel is)", fontsize=11)

    labels = [ax1.text(0, 0, repr(c)[1:-1], fontsize=8, ha="center",
                       va="center", color=char_color(c),
                       fontweight="bold") for c in chars]
    ax1.set(xlim=(allp[:, 0].min() - pad, allp[:, 0].max() + pad),
            ylim=(allp[:, 1].min() - pad, allp[:, 1].max() + pad),
            title="characters in embedding space (PCA of $C$)")
    ax1.set_xticks([]), ax1.set_yticks([])

    ax2.axhline(BIGRAM_VAL, color="#898781", ls=":", lw=1,
                label=f"bigram val {BIGRAM_VAL}")
    ax2.axhline(COUNT_BEST, color="#111", ls="--", lw=1,
                label=f"count k=3 val {COUNT_BEST}")
    ltr, = ax2.plot([], [], color="#2a78d6", lw=1, label="train NLL")
    lva, = ax2.plot([], [], color="#eb6834", lw=2, label="val NLL")
    ax2.set(xlim=(0, steps_hist[-1]), ylim=(1.6, 3.2),
            xlabel="step", ylabel="NLL (nats/char)",
            title="the loss progression vs the count models")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=.3)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    ax3.set_axis_off()
    ax3.set_title("samples from the model", fontsize=10)
    head = ax3.text(0, .98, "", transform=ax3.transAxes, fontsize=8,
                    va="top", color="#eb6834", family="monospace")
    body = ax3.text(0, .90, "", transform=ax3.transAxes, fontsize=7.5,
                    va="top", family="monospace", wrap=True)

    def update(k):
        for lab, (px, py) in zip(labels, proj[k]):
            lab.set_position((px, py))
        ltr.set_data(steps_hist[:k + 1], tr_hist[:k + 1])
        lva.set_data(steps_hist[:k + 1], va_hist[:k + 1])
        head.set_text(f"step {steps_hist[k]:>6d}   val NLL "
                      f"{va_hist[k]:.4f}")
        body.set_text(texts[k])
        return [*labels, ltr, lva, head, body]

    return FuncAnimation(fig, update, frames=len(snaps), interval=90,
                         blit=False, repeat=True)


if __name__ == "__main__":
    train_ids, val_ids, chars, stoi, itos = load_everything()
    V = len(chars)
    device = get_device()
    torch.manual_seed(SEED)

    Xtr, Ytr = windows_tensor(train_ids, device)
    Xva, Yva = windows_tensor(val_ids, device)
    model = NGramMLP(V).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device}, parameters {n_params:,} "
          f"(the k=8 count table would need 65^8 ≈ 3.2e14 rows)")

    snaps, steps_hist, tr_hist, va_hist, texts = [], [], [], [], []
    g = torch.Generator(device="cpu").manual_seed(SEED)
    t0, step = time.perf_counter(), 0
    for lr, n_steps in LR_SCHEDULE:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_steps):
            if step % CHECK_EVERY == 0:
                snaps.append(model.C.weight.detach().cpu().numpy().copy())
                steps_hist.append(step)
                tr_hist.append(full_nll(model, Xtr[:65536], Ytr[:65536]))
                va_hist.append(full_nll(model, Xva, Yva))
                texts.append(sample_text(model, itos, device))
            rows = torch.randint(0, len(Xtr), (BATCH,), generator=g)
            loss = F.cross_entropy(model(Xtr[rows.to(device)]),
                                   Ytr[rows.to(device)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    dt = time.perf_counter() - t0

    nll_val = full_nll(model, Xva, Yva)
    snaps.append(model.C.weight.detach().cpu().numpy().copy())
    steps_hist.append(step)
    tr_hist.append(full_nll(model, Xtr[:65536], Ytr[:65536]))
    va_hist.append(nll_val)
    texts.append(sample_text(model, itos, device))

    print(f"trained {STEPS:,} steps in {dt:.1f}s")
    print(f"val NLL: {nll_val:.4f}  (PPL {np.exp(nll_val):.2f})")
    print(f"  vs bigram         {BIGRAM_VAL}  (PPL 11.96)")
    print(f"  vs count sweet spot k=3 {COUNT_BEST}  (PPL 7.05)")
    assert nll_val < COUNT_BEST, "the MLP must beat every count model!"
    print("rung 2 beats rung 1 and all count models: OK")

    print("\n--- sample " + "-" * 49)
    print(sample_text(model, itos, device, n_chars=300))
    print("-" * 60)

    ani = animate(snaps, steps_hist, tr_hist, va_hist, texts, chars)
    plt.show()
