# LLM Track — Design

**Date:** 2026-08-14 · **Status:** approved · **Hardware:** RTX 4070 (torch 2.5.1, CUDA verified) · **Corpus:** Tiny Shakespeare (char-level)

## Goal

Build an LLM by hand, train it, and improve it — in raw PyTorch (`nn.Module` + autograd + `torch.optim`, hand-written training loop; no Lightning, no HF Trainer). Runs parallel to the classic-ML theory track; both live on `main`.

## Structure

```
llm/
  common.py            # corpus loading, char vocab, train/val split, device, seeds
  data/                # tinyshakespeare.txt (auto-downloaded, committed)
  bigram/              # rung 1
  ngram-mlp/           # rung 2
  attention/           # rung 3
  gpt/                 # rung 4
```

Bernoulli conventions carry over: three-file split where meaningful (`*_theoretical.py` / `*_practical_pure.py` / `*_practical_pytorch.py`), a `.tex` per folder deriving the math, every claim asserted, loss progressions and sampling animations everywhere.

## The ladder

Each rung exists because the previous one provably fails; each must numerically beat the last on the **shared measuring stick**: char-level validation NLL / perplexity on one fixed held-out slice, tracked in a README table. Beating the previous rung is an assert, not a hope.

1. **`bigram/`** — Language modeling is MLE. P(next|current) is a 65×65 conditional multinoulli; closed form = count rows and normalize (the `m/n` argument one level up). Same model as a logit table trained by SGD on cross-entropy must converge to the count table (two routes, one destination — on language). Pure-numpy SGD (softmax − onehot gradient) vs torch on identical schedules: exact trajectory identity, as in bernoulli. First perplexity number. Animation: sampled text + loss as the table learns.
2. **`ngram-mlp/`** — Why counting dies: 65^k rows; measure the fraction of val contexts never seen in training. Fix: char embeddings + concatenation + MLP (Bengio 2003). Animation: 2D embedding space organizing (vowels vs consonants) alongside loss.
3. **`attention/`** — Why fixed windows die. Derive scaled dot-product attention (queries/keys/values, softmax as a soft dictionary, causal masking). Theoretical file: hand-compute one attention forward+backward on a 4-token example, assert against torch. Single head → multi-head → full block (residuals + LayerNorm + MLP). Animation: attention maps forming during training.
4. **`gpt/`** — Assemble the decoder: token+position embeddings, N blocks, tied head; ~10M params, minutes on the 4070. Payoff animation: samples evolve noise → words → speaker names → iambic dialogue, synced to the loss curve.

## Training loop and the "improve it" arc

The loop is hand-written once in `llm/common.py` (~30 documented lines): batching of random context windows, train/val NLL estimation, checkpointing, sampling. From `gpt/` onward it gains real-world knobs — AdamW, LR warmup + cosine decay, gradient clipping — each introduced when its absence hurts, via an ablation script with before/after loss curves. Improvements are experiments, never folklore.

## Later (separate designs, YAGNI for now)

BPE tokenization (`llm/tokenization/`, merges by hand), scaling experiments (params vs data vs loss), finetuning.
