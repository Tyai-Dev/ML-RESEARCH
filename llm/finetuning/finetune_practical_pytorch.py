r"""Finetuning — what a trained model is worth on a new distribution.

The question. gpt_checkpoint.pt speaks Shakespeare (val NLL/char
1.4724 on the plays). Hand it a different author — Jane Austen, Pride
and Prejudice — and three numbers tell the whole transfer-learning
story on the SAME Austen validation slice:

  1. ZERO-SHOT: the Shakespeare model, untouched, scored on Austen.
     Bad — but far better than uniform (log 65 = 4.17): most of what
     a character LM knows (spelling, spacing, punctuation, common
     words) is ENGLISH, not Shakespeare. Knowledge transfers.
  2. FINETUNED: the checkpoint, trained a short budget on Austen at a
     LOW learning rate (a big lr would blast away the transferred
     knowledge before the new data can steer — catastrophic
     forgetting's first face).
  3. FROM SCRATCH: a fresh random GPT given the IDENTICAL budget on
     the same data. The gap between 2 and 3 is the cash value of the
     pretrained initialization: finetuning starts from "knows
     English, wrong author", scratch starts from noise.

Asserted: finetuned < zero-shot (adaptation works) and finetuned <
scratch (transfer beats tabula rasa at equal budget).

Tokenizer lock-in (the tokenization note's warning, now binding): the
checkpoint is welded to its 65-char Shakespeare alphabet, so Austen
must be SANITIZED into it — curly quotes to straight, em-dashes to
hyphens, characters outside the vocabulary dropped (the script reports
how much text survives; ~99%). Your alphabet is a lifetime commitment.

The animation: the finetune and scratch loss curves racing under the
zero-shot line, while samples from the finetuned model morph from
blank-verse dialogue toward drawing-room prose.

Run me with F5 (needs llm/gpt/gpt_checkpoint.pt — run gpt_train.py
first). Discussion: finetuning.tex.
"""

import sys
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, get_device, load_everything, decode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt"))
from gpt_model import GPT, GPTConfig
from gpt_train import full_nll, get_batch, sample_text

AUSTEN_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
AUSTEN_PATH = Path(__file__).resolve().parents[1] / "data" / "austen.txt"
CKPT = Path(__file__).resolve().parents[1] / "gpt" / "gpt_checkpoint.pt"

BUDGET = 1_000                       # steps, identical for both runs
BATCH = 64
FT_LR = 1e-4                         # low: steer, don't blast
EVAL_EVERY = 100

# characters outside the Shakespeare vocab, mapped into it
TRANSLATE = {"“": "", "”": "", "‘": "'", "’": "'", "—": "-", "–": "-",
             "_": "", "(": "", ")": "", "[": "", "]": "", "*": "",
             "æ": "ae", "é": "e", "è": "e", "\r": "", "﻿": ""}


def load_austen(vocab: set) -> str:
    """Download once, strip the Gutenberg wrapper, sanitize into the
    checkpoint's alphabet."""
    if not AUSTEN_PATH.exists():
        print(f"downloading Pride and Prejudice -> {AUSTEN_PATH}")
        urllib.request.urlretrieve(AUSTEN_URL, AUSTEN_PATH)
    raw = AUSTEN_PATH.read_text(encoding="utf-8")
    start = raw.index("*** START")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END")
    text = raw[start:end]
    for a, b in TRANSLATE.items():
        text = text.replace(a, b)
    kept = "".join(c for c in text if c in vocab)
    print(f"Austen: {len(text):,} chars, {len(kept) / len(text):.2%} "
          f"survive sanitization into the 65-char vocab")
    return kept


def train(model, tr_t, va_t, cfg, device, lr, tag):
    """The shared budget: BUDGET steps, constant low lr, eval curve.
    Early stopping via best-checkpoint (the tokenization lesson)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    steps_h, va_h, texts = [], [], []
    best_val, best_state = float("inf"), None
    for step in range(BUDGET + 1):
        if step % EVAL_EVERY == 0:
            v = full_nll(model, va_t, cfg.block_size, device)
            steps_h.append(step)
            va_h.append(v)
            texts.append(sample_text(model, ITOS, device, n_chars=200))
            if v < best_val:
                best_val = v
                best_state = {k: p.detach().clone()
                              for k, p in model.state_dict().items()}
            print(f"  [{tag}] step {step:>4d}  val NLL {v:.4f}")
        if step == BUDGET:
            break
        x, y = get_batch(tr_t, cfg.block_size, BATCH, g, device)
        loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab_size),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.load_state_dict(best_state)
    return steps_h, va_h, texts, best_val


if __name__ == "__main__":
    assert CKPT.exists(), "run llm/gpt/gpt_train.py first (checkpoint)"
    _, _, chars, stoi, itos = load_everything()   # Shakespeare vocab
    ITOS = itos
    device = get_device()
    torch.manual_seed(SEED)

    text = load_austen(set(chars))
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n_tr = int(len(ids) * 0.9)
    tr_t = torch.from_numpy(ids[:n_tr].copy())
    va_t = torch.from_numpy(ids[n_tr:].copy())

    saved = torch.load(CKPT, weights_only=False)
    cfg = GPTConfig(**saved["config"])

    # 1) zero-shot: Shakespeare weights, Austen text
    model = GPT(cfg).to(device)
    model.load_state_dict(saved["state_dict"])
    zero_shot = full_nll(model, va_t, cfg.block_size, device)
    print(f"zero-shot on Austen        : {zero_shot:.4f} nats/char "
          f"(Shakespeare val was {saved['val_nll']:.4f}; "
          f"uniform is {np.log(65):.2f})")

    # 2) finetune the checkpoint, low lr
    t0 = time.perf_counter()
    ft = train(model, tr_t, va_t, cfg, device, FT_LR, "finetune")
    # 3) identical budget from random init
    torch.manual_seed(SEED)
    scratch_model = GPT(cfg).to(device)
    sc = train(scratch_model, tr_t, va_t, cfg, device, 6e-4, "scratch")
    dt = time.perf_counter() - t0

    ft_best, sc_best = ft[3], sc[3]
    print(f"\nboth runs: {BUDGET} steps each, {dt / 60:.1f} min total")
    print(f"zero-shot          : {zero_shot:.4f}")
    print(f"finetuned  (best)  : {ft_best:.4f}")
    print(f"from scratch (best): {sc_best:.4f}")
    print(f"value of pretraining at this budget: "
          f"{sc_best - ft_best:.4f} nats/char")
    assert ft_best < zero_shot, "finetuning must beat zero-shot!"
    assert ft_best < sc_best, "transfer must beat tabula rasa!"
    print("finetune < zero-shot and finetune < scratch: OK")

    print("\n--- finetuned sample (Austen-flavored) " + "-" * 21)
    print(sample_text(model, itos, device, n_chars=300))
    print("-" * 60)

    # ------------------------------------------------------------------
    # Animation: the race, and the style morphing
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.suptitle("what pretraining is worth: finetune vs scratch on "
                 "Austen, same budget", fontsize=11)

    ax1.axhline(zero_shot, color="#898781", ls=":", lw=1.5,
                label=f"zero-shot {zero_shot:.3f}")
    lft, = ax1.plot([], [], "o-", color="#eb6834", lw=2, ms=3,
                    label="finetuned checkpoint")
    lsc, = ax1.plot([], [], "o-", color="#2a78d6", lw=2, ms=3,
                    label="from scratch")
    ax1.set(xlim=(0, BUDGET), ylim=(1.2, 3.4),
            xlabel="step", ylabel="Austen val NLL (nats/char)",
            title="the race")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=.3)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    ax2.set_axis_off()
    ax2.set_title("samples from the finetuned model", fontsize=10)
    head = ax2.text(0, .98, "", transform=ax2.transAxes, fontsize=8,
                    va="top", color="#eb6834", family="monospace")
    body = ax2.text(0, .90, "", transform=ax2.transAxes, fontsize=7.5,
                    va="top", family="monospace", wrap=True)

    def update(k):
        lft.set_data(ft[0][:k + 1], ft[1][:k + 1])
        lsc.set_data(sc[0][:k + 1], sc[1][:k + 1])
        head.set_text(f"step {ft[0][k]:>4d}   finetune val "
                      f"{ft[1][k]:.4f}")
        body.set_text(ft[2][k])
        return lft, lsc, head, body

    ani = FuncAnimation(fig, update, frames=len(ft[0]), interval=350,
                        blit=False, repeat=True)
    plt.show()
