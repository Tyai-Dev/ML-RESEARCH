r"""Training the GPT — the hand-written loop, with the grown-up knobs.

The loop is the same ~30 lines as every rung: sample windows, forward,
cross-entropy, backward, step. What's new is that at 3.2M parameters
the training knobs stop being optional, and each enters here with its
reason stated (gpt_ablations.py measures what happens without them):

  ADAMW — Adam with DECOUPLED weight decay: p <- p - lr*wd*p applied
    directly, not mixed into the gradient/moment machinery (that's the
    W in AdamW; Theory/optimizers). Decay only the matmul weights —
    biases, LayerNorms and embeddings stay free.
  WARMUP + COSINE — Adam's second-moment estimates are garbage for the
    first steps; a linear warmup keeps early steps small, then a
    cosine glide from peak to min learning rate replaces the crude
    step-drops of rungs 2-3:
        lr(t) = peak * t/warmup                          t < warmup
        lr(t) = min + (peak-min) * 0.5*(1 + cos(pi*u))   u = progress
  GRADIENT CLIPPING — rescale the global gradient norm to <= 1.0; one
    bad minibatch cannot throw the parameters across the landscape.

The gate (asserted): validation NLL < 1.6254 — rung 3's one-block
score — on the same slice. Expected: ~1.5, the biggest single drop on
the ladder, bought by DEPTH: one block is one hop of information
between positions; four blocks iterate the mixing, which is what
long-range structure (quotes that close, subjects that agree) needs.

The animation (the ladder's finale): the loss curve descending through
every previous rung's line; the perplexity ladder as a live bar chart,
GPT's bar shrinking; and the samples — the same fixed seed at every
checkpoint — evolving from noise toward scene headings, meter, and
dialogue.

Run me with F5. Derivations: gpt.tex.
"""

import math
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
from gpt_model import GPT, GPTConfig

BATCH = 64
MAX_STEPS = 6_000
PEAK_LR, MIN_LR = 6e-4, 6e-5
WARMUP = 300
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_EVERY = 250

RUNGS = [("bigram", 2.4819), ("ngram-mlp", 1.7583),
         ("attention", 1.6254)]                    # the lines to cross
GATE = RUNGS[-1][1]


def lr_at(step: int) -> float:
    """Warmup then cosine, as derived in the docstring."""
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    u = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return MIN_LR + (PEAK_LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * u))


def configure_adamw(model):
    """Decay matmul weights only: 2D+ parameters. Biases, LayerNorm
    gains, and (tied) embeddings stay decay-free."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=PEAK_LR, betas=(0.9, 0.95))


def get_batch(ids_t, block, batch, g, device):
    ix = torch.randint(len(ids_t) - block - 1, (batch,), generator=g)
    x = torch.stack([ids_t[i:i + block] for i in ix]).to(device)
    y = torch.stack([ids_t[i + 1:i + block + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def full_nll(model, ids_t, block, device, batch=64):
    """Exact NLL over a split, non-overlapping windows."""
    model.eval()
    total, n = 0.0, 0
    starts = list(range(0, len(ids_t) - block - 1, block))
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        x = torch.stack([ids_t[s:s + block] for s in chunk]).to(device)
        y = torch.stack([ids_t[s + 1:s + block + 1]
                         for s in chunk]).to(device)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1), reduction="sum").item()
        n += y.numel()
    model.train()
    return total / n


def sample_text(model, itos, device, n_chars=220, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, n_chars, generator=g)
    return decode(out[0, 1:].cpu().numpy(), itos)


def animate(steps_h, tr_h, va_h, texts):
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.8, 1.3])
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
    fig.suptitle("the ladder's finale: a GPT descends through "
                 "every previous rung", fontsize=11)

    colors = ["#898781", "#3d9b35", "#2a78d6"]
    for (name, nll), c in zip(RUNGS, colors):
        ax1.axhline(nll, color=c, ls="--", lw=1,
                    label=f"{name} {nll:.3f}")
    ltr, = ax1.plot([], [], color="#f0a24f", lw=1, label="train NLL")
    lva, = ax1.plot([], [], color="#eb6834", lw=2, label="val NLL")
    ax1.set(xlim=(0, steps_h[-1]), ylim=(1.2, 3.0),
            xlabel="step", ylabel="NLL (nats/char)",
            title="the loss progression")
    ax1.legend(frameon=False, fontsize=7)
    ax1.grid(alpha=.3)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    names = [n for n, _ in RUNGS] + ["gpt"]
    fixed = [math.exp(v) for _, v in RUNGS]
    bars = ax2.bar(names, fixed + [math.exp(va_h[0])],
                   color=colors + ["#eb6834"])
    ax2.set(ylabel="val perplexity", ylim=(0, 13),
            title="the ladder, live")
    ax2.tick_params(axis="x", labelsize=8, rotation=20)
    ax2.grid(alpha=.3, axis="y")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    ax3.set_axis_off()
    ax3.set_title("samples (same seed every checkpoint)", fontsize=10)
    head = ax3.text(0, .98, "", transform=ax3.transAxes, fontsize=8,
                    va="top", color="#eb6834", family="monospace")
    body = ax3.text(0, .90, "", transform=ax3.transAxes, fontsize=7.5,
                    va="top", family="monospace", wrap=True)

    def update(k):
        ltr.set_data(steps_h[:k + 1], tr_h[:k + 1])
        lva.set_data(steps_h[:k + 1], va_h[:k + 1])
        bars[-1].set_height(math.exp(va_h[k]))
        head.set_text(f"step {steps_h[k]:>5d}   val NLL {va_h[k]:.4f}"
                      f"   PPL {math.exp(va_h[k]):.2f}")
        body.set_text(texts[k])
        return [ltr, lva, bars[-1], head, body]

    return FuncAnimation(fig, update, frames=len(steps_h), interval=120,
                         blit=False, repeat=True)


if __name__ == "__main__":
    train_ids, val_ids, chars, stoi, itos = load_everything()
    device = get_device()
    torch.manual_seed(SEED)

    cfg = GPTConfig(vocab_size=len(chars))
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device}  |  GPT: {cfg.n_layer} layers, "
          f"{cfg.n_head} heads, d={cfg.n_embd}, T={cfg.block_size}  |  "
          f"{n_params / 1e6:.2f}M params")

    tr_t = torch.from_numpy(train_ids.copy())
    va_t = torch.from_numpy(val_ids.copy())
    opt = configure_adamw(model)
    g = torch.Generator().manual_seed(SEED)

    steps_h, tr_h, va_h, texts = [], [], [], []
    t0 = time.perf_counter()
    for step in range(MAX_STEPS):
        if step % EVAL_EVERY == 0:
            steps_h.append(step)
            tr_h.append(full_nll(model, tr_t[:100_000], cfg.block_size,
                                 device))
            va_h.append(full_nll(model, va_t, cfg.block_size, device))
            texts.append(sample_text(model, itos, device))
            print(f"  step {step:>5d}  lr {lr_at(step):.2e}  "
                  f"train {tr_h[-1]:.4f}  val {va_h[-1]:.4f}")
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y = get_batch(tr_t, cfg.block_size, BATCH, g, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    dt = time.perf_counter() - t0

    nll_val = full_nll(model, va_t, cfg.block_size, device)
    steps_h.append(MAX_STEPS)
    tr_h.append(full_nll(model, tr_t[:100_000], cfg.block_size, device))
    va_h.append(nll_val)
    texts.append(sample_text(model, itos, device))

    print(f"\ntrained {MAX_STEPS:,} steps in {dt / 60:.1f} min")
    print(f"val NLL: {nll_val:.4f}  (PPL {np.exp(nll_val):.2f})")
    for name, nll in RUNGS:
        print(f"  vs {name:<10} {nll:.4f}  (PPL {np.exp(nll):.2f})")
    assert nll_val < GATE, "the GPT must beat the one-block model!"
    print("rung 4 beats every rung below it: OK")

    ckpt = Path(__file__).with_name("gpt_checkpoint.pt")
    torch.save({"config": cfg.__dict__,
                "state_dict": model.state_dict(),
                "val_nll": nll_val}, ckpt)
    print(f"checkpoint -> {ckpt.name}")

    print("\n--- sample (400 chars) " + "-" * 37)
    print(sample_text(model, itos, device, n_chars=400))
    print("-" * 60)

    ani = animate(steps_h, tr_h, va_h, texts)
    plt.show()
