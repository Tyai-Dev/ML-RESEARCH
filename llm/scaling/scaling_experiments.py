r"""Scaling — which resource is the bottleneck, measured.

Rung 4 picked 4 layers / d=256 by taste. This script replaces taste
with two sweeps, holding everything else fixed (T=128, dropout 0.1,
3000 steps of the rung-4 recipe, same seed, same val slice):

  PARAM SWEEP: models from 0.2M to 8M parameters on the full corpus.
    Prediction (scaling-law folklore): val loss falls like a power law
    in parameters. What a 1M-char corpus actually delivers: gains that
    shrink and then stall — while the train-val gap widens with size.
    The big models aren't starved of capacity; they're starved of
    DATA.
  DATA SWEEP: the rung-4 model on 10% / 25% / 50% / 100% of the
    training characters. Every doubling of data keeps paying.

Together the two curves answer the resource question: at this corpus
size, the next nat lives in MORE TEXT, not more parameters — the
data-limited regime (the Chinchilla lesson, reproduced at desk
scale). The asserts encode exactly that shape: the param curve must
improve early and be allowed to stall late; the data curve must
improve at EVERY step; the overfit gap must grow with model size.

Runtime note: ~10-15 min on the RTX 4070 (9 training runs). Results
are also written to scaling_results.json so the numbers survive the
run. Run me with F5. Discussion: scaling.tex.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, get_device, load_everything

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt"))
from gpt_model import GPT, GPTConfig
from gpt_train import full_nll, get_batch, GRAD_CLIP, configure_adamw

STEPS = 3_000
BATCH = 64
PEAK_LR, MIN_LR, WARMUP = 6e-4, 6e-5, 150
EVAL_EVERY = 250

PARAM_SWEEP = [(1, 64), (2, 128), (3, 192), (4, 256), (6, 320)]
DATA_SWEEP = [0.10, 0.25, 0.50, 1.00]          # fraction of train chars
RESULTS = Path(__file__).with_name("scaling_results.json")


def lr_at(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    import math
    u = (step - WARMUP) / max(1, STEPS - WARMUP)
    return MIN_LR + (PEAK_LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * u))


def run(cfg, tr_t, va_t, device, tag):
    """One training run under the fixed recipe; returns history and
    the final train/val NLL."""
    torch.manual_seed(SEED)
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = configure_adamw(model)
    g = torch.Generator().manual_seed(SEED)
    hist = []
    t0 = time.perf_counter()
    for step in range(STEPS):
        if step % EVAL_EVERY == 0:
            hist.append((step, full_nll(model, va_t, cfg.block_size,
                                        device)))
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y = get_batch(tr_t, cfg.block_size, BATCH, g, device)
        loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab_size),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    va = full_nll(model, va_t, cfg.block_size, device)
    tr = full_nll(model, tr_t[:100_000], cfg.block_size, device)
    hist.append((STEPS, va))
    print(f"  [{tag}] {n_params / 1e6:5.2f}M params  "
          f"train {tr:.4f}  val {va:.4f}  "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return dict(params=n_params, train=tr, val=va, hist=hist)


if __name__ == "__main__":
    train_ids, val_ids, chars, stoi, itos = load_everything()
    device = get_device()
    tr_t = torch.from_numpy(train_ids.copy())
    va_t = torch.from_numpy(val_ids.copy())
    print(f"device {device}; {len(PARAM_SWEEP) + len(DATA_SWEEP)} "
          f"training runs of {STEPS} steps each", flush=True)

    print("\nPARAM SWEEP (full data):", flush=True)
    p_runs = []
    for n_layer, n_embd in PARAM_SWEEP:
        cfg = GPTConfig(vocab_size=len(chars), n_layer=n_layer,
                        n_head=max(1, n_embd // 64), n_embd=n_embd)
        p_runs.append(run(cfg, tr_t, va_t, device,
                          f"L{n_layer} d{n_embd}"))

    print("\nDATA SWEEP (rung-4 model):", flush=True)
    d_runs = []
    for frac in DATA_SWEEP:
        cfg = GPTConfig(vocab_size=len(chars))
        cut = int(len(tr_t) * frac)
        d_runs.append(dict(frac=frac,
                           **run(cfg, tr_t[:cut], va_t, device,
                                 f"{frac:4.0%} data")))

    RESULTS.write_text(json.dumps(
        dict(param_sweep=p_runs, data_sweep=d_runs), indent=1))
    print(f"\nresults -> {RESULTS.name}")

    # ---- the shape asserts ------------------------------------------
    vals = [r["val"] for r in p_runs]
    gaps = [r["val"] - r["train"] for r in p_runs]
    assert vals[1] < vals[0] and vals[2] < vals[1], \
        "small-model regime should improve with size"
    assert gaps[-1] > gaps[0] + 0.05, \
        "the overfit gap should grow with model size"
    late_gain = vals[-2] - vals[-1]
    early_gain = vals[0] - vals[1]
    assert late_gain < early_gain / 2, \
        "gains should shrink at the big end (data-limited)"
    dvals = [r["val"] for r in d_runs]
    assert all(b < a for a, b in zip(dvals, dvals[1:])), \
        "every data doubling should still pay"
    print("shape checks (diminishing param gains, growing overfit gap, "
          "monotone data gains): OK")
    print(f"verdict: at 1M chars the next nat lives in more DATA — "
          f"params {vals[0]:.3f}->{vals[-1]:.3f} (stalling), "
          f"data {dvals[0]:.3f}->{dvals[-1]:.3f} (still paying)")

    # ------------------------------------------------------------------
    # Figure: training curves (animated) + the two scaling curves
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13.5, 4.4))
    gs = fig.add_gridspec(1, 3)
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]
    fig.suptitle("which resource is the bottleneck? "
                 "(fixed budget, same recipe, same seed)", fontsize=11)

    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(p_runs)))
    lines = []
    for r, c in zip(p_runs, cmap):
        ln, = ax1.plot([], [], color=c, lw=1.8,
                       label=f"{r['params'] / 1e6:.1f}M")
        lines.append(ln)
    ax1.set(xlim=(0, STEPS), ylim=(1.4, 3.0), xlabel="step",
            ylabel="val NLL", title="param sweep: the loss curves")
    ax1.legend(frameon=False, fontsize=7, title="params", ncol=2)

    ax2.plot([r["params"] for r in p_runs], vals, "o-",
             color="#eb6834", lw=2)
    ax2.plot([r["params"] for r in p_runs],
             [r["train"] for r in p_runs], "o--", color="#898781",
             lw=1.5, label="train")
    ax2.set_xscale("log")
    ax2.fill_between([r["params"] for r in p_runs],
                     [r["train"] for r in p_runs], vals,
                     alpha=.15, color="#eb6834", label="overfit gap")
    ax2.set(xlabel="parameters", ylabel="NLL (nats/char)",
            title="more params: stalling (data-limited)")
    ax2.legend(frameon=False, fontsize=8)

    ax3.plot([r["frac"] for r in d_runs], dvals, "o-",
             color="#2a78d6", lw=2)
    ax3.set_xscale("log")
    ax3.set(xlabel="fraction of training data", ylabel="val NLL",
            title="more data: still paying",
            xticks=DATA_SWEEP,
            xticklabels=[f"{f:.0%}" for f in DATA_SWEEP])
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()

    frames = max(len(r["hist"]) for r in p_runs)

    def update(k):
        for ln, r in zip(lines, p_runs):
            h = r["hist"][:k + 1]
            ln.set_data([s for s, _ in h], [v for _, v in h])
        return lines

    ani = FuncAnimation(fig, update, frames=frames, interval=250,
                        blit=False, repeat=True)
    plt.show()
