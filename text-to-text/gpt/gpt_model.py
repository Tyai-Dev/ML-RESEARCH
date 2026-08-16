r"""A GPT, assembled from verified parts.

There is nothing new in this file — that is its point. Every mechanism
was derived and machine-verified lower on the ladder:

  embeddings          rung 2 (and the scatter-add backward, by hand)
  cross-entropy/softmax gradient      rung 1 (softmax - onehot)
  causal attention    rung 3 (forward AND backward vs autograd, 1e-16)
  the block           rung 3 (pre-norm residuals: communicate, compute)

A GPT is: token + position embeddings, the rung-3 block stacked
N_LAYER deep, a final LayerNorm, and a linear head. Stacking is what
buys long-range structure: one block = one hop of information between
positions; N blocks = iterated mixing (an open quote can influence its
closing 100 characters later through repeated attention rounds).

Assembly details that matter (each documented where used):
  - WEIGHT TYING: the output head reuses the token embedding matrix.
    The map "char -> vector" and "vector -> char logits" share
    structure; tying saves V*d parameters and slightly regularizes.
  - INIT: normals with std 0.02; residual-path projections scaled by
    1/sqrt(2*N_LAYER) so the variance of the residual stream stays O(1)
    after 2*N_LAYER additions (the sqrt(d_h) argument, applied to
    depth).
  - DROPOUT: this model is large enough to overfit 1M chars, so
    regularization finally enters the ladder (rung 4 is where its
    absence measurably hurts).

Config below: 4 layers, 4 heads, d=256, T=128 context — ~3.2M
parameters, minutes on the RTX 4070. Training loop: gpt_train.py.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128        # context length T
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Rung 3's verified mechanism, unchanged: multi-head causal
    scaled dot-product attention."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask",
                             mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        nh, dh = self.cfg.n_head, C // self.cfg.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, nh, dh).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """Communicate (attention), then compute (MLP); pre-norm residuals."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full decoder-only transformer."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight        # weight tying

        self.apply(self._init)
        # residual projections: scale by 1/sqrt(2*n_layer) so the
        # residual stream's variance stays O(1) after all additions
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):                       # (B, T) -> (B, T, V)
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None,
                 generator=None):
        """Ancestral sampling with the usual knobs. temperature < 1
        sharpens the distribution; top_k keeps only the k most likely
        characters before sampling."""
        self.eval()
        for _ in range(n_new):
            logits = self(idx[:, -self.cfg.block_size:])[0, -1]
            logits = logits / temperature
            if top_k is not None:
                cut = torch.topk(logits, top_k).values[-1]
                logits[logits < cut] = float("-inf")
            p = torch.softmax(logits, dim=0).cpu()
            c = torch.multinomial(p, 1, generator=generator)
            idx = torch.cat([idx, c.view(1, 1).to(idx.device)], dim=1)
        self.train()
        return idx
