r"""The same GPT on BPE tokens — compared honestly, per character.

The experiment: take rung 4's architecture UNCHANGED (4 layers, 4
heads, d=256, T=128) and change only the alphabet: 515 BPE tokens
(65 chars + 450 merges) instead of 65 characters. Two consequences
fight each other:

  FOR:  a 128-token window now spans ~297 characters of text (vs 128)
        — 2.3x the effective context; and the model no longer spends
        capacity spelling: ' the' is one prediction, not four.
  AGAINST: the corpus shrinks from ~1M char-positions to ~432k
        token-positions — fewer, harder prediction problems from the
        same data; and each miss costs more (a whole word, not a
        letter).

Which wins is an empirical question, so we measure it — in the only
comparable currency. Perplexity per TOKEN is meaningless across
tokenizers; the honest metric divides total validation NLL by the
number of CHARACTERS the evaluated tokens cover:

    NLL/char = sum_i NLL(token_i) / sum_i len(token_i).

WHAT ACTUALLY HAPPENS (measured): the AGAINST side dominates the late
game. With 2.3x fewer training positions, rung 4's 6000-step budget
is ~114 epochs over the token corpus — validation NLL/char bottoms
out near step 1500 (~1.52) and then climbs steadily while train loss
keeps falling: overfitting, arriving right on schedule. The tool that
answers it is EARLY STOPPING — keep the parameters from the best
validation point, not the last step — which enters the toolkit here,
where its absence first costs a quarter nat. Final verdict at this
scale: the best BPE model still loses to the char-GPT per character
(chars are a stronger alphabet when data, not context, is the
bottleneck). The metric made that an answerable question.

Hygiene that matters: the BPE merges are trained on the TRAINING
split only (a tokenizer fitted on validation text is leakage — it
would carry val statistics into the model's alphabet), and train/val
are encoded separately with those merges.

Run me with F5. Discussion: tokenization.tex.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEED, get_device, load_corpus, train_val_split
from bpe_theoretical import N_MERGES, train_bpe, encode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt"))
from gpt_model import GPT, GPTConfig
from gpt_train import lr_at, configure_adamw, MAX_STEPS, BATCH, GRAD_CLIP

CHAR_GPT = 1.4724                  # rung 4, NLL/char — the comparison
ATTN_GPT = 1.6254                  # rung 3 — the hard gate


@torch.no_grad()
def nll_per_char(model, ids_t, token_lens, block, device, batch=64):
    """Total NLL over non-overlapping windows, divided by the CHARS
    the evaluated target tokens cover — the cross-tokenizer metric."""
    model.eval()
    total, chars = 0.0, 0
    starts = list(range(0, len(ids_t) - block - 1, block))
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        x = torch.stack([ids_t[s:s + block] for s in chunk]).to(device)
        y = torch.stack([ids_t[s + 1:s + block + 1]
                         for s in chunk]).to(device)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1), reduction="sum").item()
        chars += int(token_lens[y.cpu().reshape(-1)].sum())
    model.train()
    return total / chars


if __name__ == "__main__":
    text = load_corpus()
    n_train_chars = int(len(text) * 0.9)
    train_text, val_text = text[:n_train_chars], text[n_train_chars:]

    print("training BPE on the TRAIN split only (no val leakage) ...")
    merges, _ = train_bpe(train_text, N_MERGES)

    # vocabulary: the 65 base chars + the merge products, in order
    base = sorted(set(text))
    vocab = base + [a + b for a, b in merges]
    stoi = {t: i for i, t in enumerate(vocab)}
    token_lens = np.array([len(t) for t in vocab])

    tr_tokens = encode(train_text, merges, {})
    va_tokens = encode(val_text, merges, {})
    assert "".join(tr_tokens) == train_text          # lossless, again
    assert "".join(va_tokens) == val_text
    tr_t = torch.tensor([stoi[t] for t in tr_tokens])
    va_t = torch.tensor([stoi[t] for t in va_tokens])
    ratio = len(train_text) / len(tr_t)
    print(f"vocab {len(vocab)}   train tokens {len(tr_t):,} "
          f"(chars/token {ratio:.3f})   val tokens {len(va_t):,}")

    device = get_device()
    torch.manual_seed(SEED)
    cfg = GPTConfig(vocab_size=len(vocab))           # all else rung 4
    model = GPT(cfg).to(device)
    print(f"device {device}, params "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, "
          f"context {cfg.block_size} tokens ≈ "
          f"{cfg.block_size * ratio:.0f} chars")

    opt = configure_adamw(model)
    g = torch.Generator().manual_seed(SEED)
    steps_h, va_h = [], []
    best_val, best_state, best_step = float("inf"), None, 0
    t0 = time.perf_counter()
    for step in range(MAX_STEPS):
        if step % 250 == 0:
            steps_h.append(step)
            va_h.append(nll_per_char(model, va_t, token_lens,
                                     cfg.block_size, device))
            if va_h[-1] < best_val:              # early stopping: keep
                best_val, best_step = va_h[-1], step   # the best point
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            if step % 500 == 0:
                print(f"  step {step:>5d}  val NLL/char {va_h[-1]:.4f}")
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        ix = torch.randint(len(tr_t) - cfg.block_size - 1, (BATCH,),
                           generator=g)
        x = torch.stack([tr_t[i:i + cfg.block_size] for i in ix]).to(device)
        y = torch.stack([tr_t[i + 1:i + cfg.block_size + 1]
                         for i in ix]).to(device)
        loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab_size),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    dt = time.perf_counter() - t0

    last_val = nll_per_char(model, va_t, token_lens, cfg.block_size,
                            device)
    steps_h.append(MAX_STEPS)
    va_h.append(last_val)
    # early stopping: restore the best-validation parameters
    model.load_state_dict(best_state)
    nll_char = nll_per_char(model, va_t, token_lens, cfg.block_size,
                            device)
    print(f"\ntrained {MAX_STEPS:,} steps in {dt / 60:.1f} min")
    print(f"last step val NLL/char : {last_val:.4f}  <- overfit "
          f"(~{MAX_STEPS * BATCH * cfg.block_size // len(tr_t)} epochs "
          f"over the token corpus)")
    print(f"early-stopped (step {best_step}): {nll_char:.4f}  "
          f"(bits/char {nll_char / np.log(2):.3f})")
    print(f"char-GPT val NLL/char  : {CHAR_GPT}  "
          f"(bits/char {CHAR_GPT / np.log(2):.3f})")
    verdict = ("BPE wins" if nll_char < CHAR_GPT
               else "chars win at this scale — data, not context, "
                    "is the bottleneck")
    print(f"verdict: {verdict}")
    assert nll_char < ATTN_GPT, \
        "a BPE GPT should at least beat the one-block char model"
    assert last_val > nll_char + 0.1, \
        "the overfit should be visible (else drop early stopping)"

    # sample — note whole words appearing per step
    g2 = torch.Generator().manual_seed(SEED)
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, 120, generator=g2)
    sample = "".join(vocab[int(i)] for i in out[0, 1:].cpu())
    print("\n--- sample (120 TOKENS — whole words per step) " + "-" * 12)
    print(sample)
    print("-" * 60)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(steps_h, va_h, "o-", color="#eb6834", lw=2, ms=3,
            label="BPE-GPT (515 tokens)")
    ax.plot(best_step, best_val, "*", color="#111", ms=14,
            label=f"early stop (step {best_step}): {best_val:.4f}")
    ax.axhline(CHAR_GPT, color="#111", ls="--", lw=1,
               label=f"char-GPT (rung 4) {CHAR_GPT}")
    ax.axhline(ATTN_GPT, color="#898781", ls=":", lw=1,
               label=f"one block (rung 3) {ATTN_GPT}")
    ax.set(xlabel="step", ylabel="val NLL per CHAR (nats)",
           title="same architecture, different alphabet — the U-curve "
                 "is overfitting")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.show()
