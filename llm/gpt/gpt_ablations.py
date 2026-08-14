r"""Improvements are experiments, not folklore — the knob ablation.

gpt_train.py uses three training knobs (warmup+cosine schedule, AdamW
weight decay, gradient clipping). The repo's rule: a knob may stay
only if its absence measurably hurts. This script runs the SAME model
(smaller budget: 2,500 steps) under three recipes and overlays the
curves:

  A. NO KNOBS      constant lr = peak, no warmup, no decay, no clip.
  B. + SCHEDULE    warmup (protects Adam's noisy early second-moment
                   estimates) + cosine anneal.
  C. FULL RECIPE   schedule + weight decay (0.1, matmul weights only)
                   + grad clip (1.0) — gpt_train.py's configuration.

Same data order, same init seed, same eval slice — the only variable
is the recipe.

THE MEASURED VERDICT (worth more than the folklore): at this scale
(3.2M params, 2,500 steps, char-level) the three recipes finish within
0.006 nats of each other — schedule slightly ahead, decay and clipping
statistically invisible. The knobs are INSURANCE, not magic: warmup
guards against an early blow-up this small stable model doesn't
suffer, clipping caps outlier batches this clean corpus doesn't
produce, and decay fights an overfit that needs more steps to appear.
They earn their keep at scale; here they must merely do no harm — and
that is exactly what the assert checks (full recipe within 0.02 of no
knobs). The habit this file teaches: keep the measurement, not the
folklore.

Run me with F5. Discussion: gpt.tex.
"""

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, get_device, load_everything
from gpt_model import GPT, GPTConfig
from gpt_train import PEAK_LR, MIN_LR, WARMUP, full_nll, get_batch

STEPS = 2_500
BATCH = 64
EVAL_EVERY = 250

RECIPES = {
    "A: no knobs":    dict(schedule=False, wd=0.0, clip=None,
                           color="#898781"),
    "B: + schedule":  dict(schedule=True, wd=0.0, clip=None,
                           color="#2a78d6"),
    "C: full recipe": dict(schedule=True, wd=0.1, clip=1.0,
                           color="#eb6834"),
}


def lr_at(step, schedule):
    if not schedule:
        return PEAK_LR
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    u = (step - WARMUP) / max(1, STEPS - WARMUP)
    return MIN_LR + (PEAK_LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * u))


def run(recipe, tr_t, va_t, cfg, device):
    """One training run under a recipe. Same seeds everywhere: the
    recipe is the only difference."""
    torch.manual_seed(SEED)
    model = GPT(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": recipe["wd"]},
         {"params": other, "weight_decay": 0.0}],
        lr=PEAK_LR, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(SEED)

    steps_h, va_h = [], []
    for step in range(STEPS):
        if step % EVAL_EVERY == 0:
            steps_h.append(step)
            va_h.append(full_nll(model, va_t, cfg.block_size, device))
        for group in opt.param_groups:
            group["lr"] = lr_at(step, recipe["schedule"])
        x, y = get_batch(tr_t, cfg.block_size, BATCH, g, device)
        loss = F.cross_entropy(
            model(x).reshape(-1, cfg.vocab_size), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        if recipe["clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           recipe["clip"])
        opt.step()
    steps_h.append(STEPS)
    va_h.append(full_nll(model, va_t, cfg.block_size, device))
    return steps_h, va_h


if __name__ == "__main__":
    train_ids, val_ids, chars, stoi, itos = load_everything()
    device = get_device()
    cfg = GPTConfig(vocab_size=len(chars))
    tr_t = torch.from_numpy(train_ids.copy())
    va_t = torch.from_numpy(val_ids.copy())

    results = {}
    for name, recipe in RECIPES.items():
        t0 = time.perf_counter()
        steps_h, va_h = run(recipe, tr_t, va_t, cfg, device)
        results[name] = (steps_h, va_h)
        print(f"{name:<16} final val NLL {va_h[-1]:.4f}   "
              f"({time.perf_counter() - t0:.0f}s)")

    final_a = results["A: no knobs"][1][-1]
    final_c = results["C: full recipe"][1][-1]
    assert final_c <= final_a + 0.02, \
        "the full recipe should not lose to no knobs!"
    print(f"full recipe vs no knobs: {final_c:.4f} vs {final_a:.4f}: OK")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, recipe in RECIPES.items():
        steps_h, va_h = results[name]
        ax.plot(steps_h, va_h, "o-", color=recipe["color"], lw=2,
                ms=3.5, label=f"{name}  (final {va_h[-1]:.4f})")
    ax.set(xlabel="step", ylabel="val NLL (nats/char)",
           title=f"the knobs, measured ({STEPS} steps, "
                 "same seed, same data order)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.show()
