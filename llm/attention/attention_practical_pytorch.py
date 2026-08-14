r"""One transformer block — attention put to work, and watched at work.

attention_theoretical.py verified the mechanism; this file builds the
smallest language model around it and climbs the rung. Architecture
(one block, the repeating unit GPT will stack):

    tok_emb[X] + pos_emb[0..T-1]          each position: what + where
      -> LayerNorm -> MULTI-HEAD causal attention -> + residual
      -> LayerNorm -> MLP (4x expand, GELU)        -> + residual
      -> LayerNorm -> linear head -> logits

Multi-head: run n_head attentions in parallel on d/n_head-dimensional
slices — different heads can look for different things (one tracks
recent letters, another watches for newlines/speaker turns) — then
concatenate. Residuals keep the input "visible" so the block learns a
CORRECTION, not a replacement (and gradients flow through the identity
path). LayerNorm keeps activations at unit scale, the same reason as
the sqrt(d_h) in the scores.

Why this beats rung 2's MLP: the MLP's window is position-rigid (the
weights for "3 back" know nothing of "4 back") and fixed at k=8.
Attention computes its own mixing weights per position, over T=64
characters — 8x the context, fewer assumptions, and the context
length is a dial, not an architecture change.

The rung's gate (asserted): validation NLL < 1.7583 (the MLP's score)
on the same slice.

The animation: an attention map ON REAL TEXT (a fixed validation
window) at every checkpoint — at init it is a diffuse causal smear; as
loss falls, structure appears: strong recency bands, columns lighting
up under spaces and newlines (word and line boundaries are what a
character model most needs), heads specializing. Alongside: the loss
progression through the MLP's line, and the samples sharpening.

Run me with F5. Derivations: attention.tex.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, get_device, load_everything, decode

CONTEXT_T = 64
N_EMBD = 128
N_HEAD = 4
DROPOUT = 0.0                          # regularization enters at rung 4
BATCH = 64
LR_SCHEDULE = [(1e-3, 12_000), (1e-4, 4_000)]
STEPS = sum(s for _, s in LR_SCHEDULE)
CHECK_EVERY = 400
MLP_VAL = 1.7583                       # rung-2 score — the gate


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention — the verified mechanism, batched.
    Keeps the last forward's attention weights for visualization."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(CONTEXT_T, CONTEXT_T))
        self.register_buffer("mask", mask.view(1, 1, CONTEXT_T, CONTEXT_T))
        self.last_attn = None

    def forward(self, x):                       # (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        # (B, n_head, T, d_h): each head works on its own slice
        q, k, v = (t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
                   for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / np.sqrt(C // N_HEAD)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        self.last_attn = att.detach()           # (B, nh, T, T)
        y = self.drop(att) @ v                  # weighted mix of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """The transformer's repeating unit: attend, then think.
    Pre-norm residuals: x + sublayer(norm(x))."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # communicate
        x = x + self.mlp(self.ln2(x))           # compute
        return x


class OneBlockLM(nn.Module):
    """token+position embeddings -> one Block -> head."""

    def __init__(self, V):
        super().__init__()
        self.tok = nn.Embedding(V, N_EMBD)
        self.pos = nn.Embedding(CONTEXT_T, N_EMBD)
        self.block = Block()
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, V)

    def forward(self, idx):                     # (B, T)
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.block(x)
        return self.head(self.ln_f(x))          # (B, T, V)


def get_batch(ids_t, batch, g):
    """Random windows: predict every next char in each window (T
    training signals per window — attention's efficiency bonus)."""
    ix = torch.randint(len(ids_t) - CONTEXT_T - 1, (batch,), generator=g)
    x = torch.stack([ids_t[i:i + CONTEXT_T] for i in ix])
    y = torch.stack([ids_t[i + 1:i + CONTEXT_T + 1] for i in ix])
    return x, y


@torch.no_grad()
def full_nll(model, ids_t, device, batch=128):
    """Average NLL over a whole split, non-overlapping windows."""
    model.eval()
    total, n = 0.0, 0
    starts = range(0, len(ids_t) - CONTEXT_T - 1, CONTEXT_T)
    starts = list(starts)
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        x = torch.stack([ids_t[s:s + CONTEXT_T] for s in chunk]).to(device)
        y = torch.stack([ids_t[s + 1:s + CONTEXT_T + 1]
                         for s in chunk]).to(device)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 y.reshape(-1), reduction="sum").item()
        n += y.numel()
    model.train()
    return total / n


@torch.no_grad()
def sample_text(model, itos, device, n_chars=200, seed=SEED):
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = []
    for _ in range(n_chars):
        logits = model(ctx[:, -CONTEXT_T:])[0, -1]
        p = torch.softmax(logits, dim=0).cpu()
        c = int(torch.multinomial(p, 1, generator=g))
        out.append(c)
        ctx = torch.cat([ctx, torch.tensor([[c]], device=device)], dim=1)
    model.train()
    return decode(out, itos)


@torch.no_grad()
def probe_attention(model, probe_x):
    """Head-averaged attention map over the fixed probe window."""
    model.eval()
    model(probe_x)
    A = model.block.attn.last_attn[0].mean(dim=0).cpu().numpy()
    model.train()
    return A


def animate(maps, steps_hist, tr_hist, va_hist, texts, probe_str):
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.15])
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
    fig.suptitle("attention learning where to look (head-averaged, "
                 "real validation text)", fontsize=11)

    im = ax1.imshow(maps[0], cmap="viridis", vmin=0,
                    vmax=np.percentile(np.array(maps), 99.5))
    ax1.set(xlabel="attended position", ylabel="query position",
            title=f"A on: “{probe_str[:34]}…”")
    fig.colorbar(im, ax=ax1, shrink=.8)

    ax2.axhline(MLP_VAL, color="#111", ls="--", lw=1,
                label=f"rung 2 (MLP) val {MLP_VAL}")
    ltr, = ax2.plot([], [], color="#2a78d6", lw=1, label="train NLL")
    lva, = ax2.plot([], [], color="#eb6834", lw=2, label="val NLL")
    ax2.set(xlim=(0, steps_hist[-1]), ylim=(1.4, 3.2),
            xlabel="step", ylabel="NLL (nats/char)",
            title="the loss progression through rung 2's line")
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
        im.set_data(maps[k])
        ltr.set_data(steps_hist[:k + 1], tr_hist[:k + 1])
        lva.set_data(steps_hist[:k + 1], va_hist[:k + 1])
        head.set_text(f"step {steps_hist[k]:>5d}   val NLL "
                      f"{va_hist[k]:.4f}")
        body.set_text(texts[k])
        return im, ltr, lva, head, body

    return FuncAnimation(fig, update, frames=len(maps), interval=90,
                         blit=False, repeat=True)


if __name__ == "__main__":
    train_ids, val_ids, chars, stoi, itos = load_everything()
    V = len(chars)
    device = get_device()
    torch.manual_seed(SEED)

    tr_t = torch.from_numpy(train_ids.copy())
    va_t = torch.from_numpy(val_ids.copy())
    model = OneBlockLM(V).to(device)
    print(f"device {device}, parameters "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # fixed probe window from validation text, for the animation
    probe_ids = va_t[1000:1000 + CONTEXT_T]
    probe_x = probe_ids.unsqueeze(0).to(device)
    probe_str = decode(probe_ids.numpy(), itos).replace("\n", "¶")

    g = torch.Generator().manual_seed(SEED)
    maps, steps_hist, tr_hist, va_hist, texts = [], [], [], [], []
    t0, step = time.perf_counter(), 0
    for lr, n_steps in LR_SCHEDULE:               # simple step decay
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for _ in range(n_steps):
            if step % CHECK_EVERY == 0:
                maps.append(probe_attention(model, probe_x))
                steps_hist.append(step)
                tr_hist.append(full_nll(model, tr_t[:100_000], device))
                va_hist.append(full_nll(model, va_t, device))
                texts.append(sample_text(model, itos, device))
            x, y = get_batch(tr_t, BATCH, g)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    dt = time.perf_counter() - t0

    nll_val = full_nll(model, va_t, device)
    maps.append(probe_attention(model, probe_x))
    steps_hist.append(STEPS)
    tr_hist.append(full_nll(model, tr_t[:100_000], device))
    va_hist.append(nll_val)
    texts.append(sample_text(model, itos, device))

    print(f"trained {STEPS:,} steps in {dt:.1f}s")
    print(f"val NLL: {nll_val:.4f}  (PPL {np.exp(nll_val):.2f})"
          f"   vs rung 2 MLP {MLP_VAL} (PPL 5.80)")
    assert nll_val < MLP_VAL, "one block must beat the fixed-window MLP!"
    print("rung 3 beats rung 2: OK")

    print("\n--- sample " + "-" * 49)
    print(sample_text(model, itos, device, n_chars=300))
    print("-" * 60)

    ani = animate(maps, steps_hist, tr_hist, va_hist, texts, probe_str)
    plt.show()
